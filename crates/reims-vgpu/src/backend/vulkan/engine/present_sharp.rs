//! TEMP (lab, `REIMS_VGPU_LAB_PRESENT_SHARP=on`): the host window's scaled
//! present as one Catmull-Rom draw instead of a linear `vkCmdBlitImage`.
//!
//! Only the scaled case takes it — a guest frame the size of the window still
//! blits 1:1, which no filter improves. The draw writes exactly the pixels the
//! blit would have (the same aspect-fit viewport, slate around it) and leaves
//! both images in the layouts and dependency scopes the blit leaves them in,
//! so nothing after it — the present barrier, the registry's `transfer_read`
//! note, the next guest write's barrier — can tell the two apart:
//!
//! * the swapchain image enters and leaves the render pass in
//!   `TRANSFER_DST_OPTIMAL`, ordered after the acquire-wait barrier by the
//!   external dependency's `TRANSFER` source scope and before the present
//!   barrier by its `TRANSFER` destination scope;
//! * the guest frame goes to `SHADER_READ_ONLY_OPTIMAL` for the draw and back
//!   to the layout the blit would have read it in, the return barrier's
//!   `FRAGMENT_SHADER -> TRANSFER` dependency chaining the shader's read into
//!   the `TRANSFER` scope the registry records for the next writer to wait on.
//!
//! Per present it makes one swapchain-image view and one framebuffer, freed
//! when the same frame entry comes round again (its fence has retired by then).

use ash::vk;

use super::present_sharp_shader::{PRESENT_SHARP_FRAG_SPIRV, PRESENT_SHARP_VERT_SPIRV};

/// `REIMS_VGPU_LAB_PRESENT_SHARP=on`. Latched.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("REIMS_VGPU_LAB_PRESENT_SHARP").as_deref(), Ok("on" | "1")))
}

pub(crate) struct PresentSharp {
    render_pass: vk::RenderPass,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    /// One per presenter frame entry.
    sets: Vec<vk::DescriptorSet>,
    /// The swapchain-image view and framebuffer each entry's last draw used.
    transient: Vec<Option<(vk::ImageView, vk::Framebuffer)>>,
}

/// The guest frame one scaled present reads.
pub(crate) struct SharpSource {
    pub image: vk::Image,
    pub view: vk::ImageView,
    /// The layout the blit would have read it in, and the one it is left in.
    pub layout: vk::ImageLayout,
    pub width: u32,
    pub height: u32,
}

impl PresentSharp {
    pub(crate) unsafe fn create(
        device: &ash::Device,
        format: vk::Format,
        frames: usize,
    ) -> Result<Self, vk::Result> {
        let mut made = Self {
            render_pass: vk::RenderPass::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            sampler: vk::Sampler::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            transient: vec![None; frames],
        };
        if let Err(e) = made.build(device, format, frames) {
            made.destroy(device);
            return Err(e);
        }
        Ok(made)
    }

    unsafe fn build(&mut self, device: &ash::Device, format: vk::Format, frames: usize) -> Result<(), vk::Result> {
        let attachment = [vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .final_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)];
        let color_ref = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpass = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref)];
        let deps = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE),
        ];
        self.render_pass = device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachment)
                .subpasses(&subpass)
                .dependencies(&deps),
            None,
        )?;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        self.set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
            None,
        )?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(8)];
        let set_layouts = [self.set_layout];
        self.layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push),
            None,
        )?;
        // Linear, clamped: the nine taps are bilinear fetches by construction.
        self.sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(0.0),
            None,
        )?;
        let vert = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&PRESENT_SHARP_VERT_SPIRV),
            None,
        )?;
        let frag = match device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&PRESENT_SHARP_FRAG_SPIRV),
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                device.destroy_shader_module(vert, None);
                return Err(e);
            }
        };
        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(entry),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.layout)
            .render_pass(self.render_pass)
            .subpass(0)];
        let pipelines = device.create_graphics_pipelines(vk::PipelineCache::null(), &info, None);
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
        self.pipeline = pipelines.map_err(|(_, e)| e)?[0];
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(frames as u32)];
        self.pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(frames as u32)
                .pool_sizes(&sizes),
            None,
        )?;
        let layouts = vec![self.set_layout; frames];
        self.sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&layouts),
        )?;
        Ok(())
    }

    /// Free entry `ix`'s view and framebuffer. Only once its fence retired.
    pub(crate) unsafe fn release(&mut self, device: &ash::Device, ix: usize) {
        if let Some((view, fb)) = self.transient.get_mut(ix).and_then(Option::take) {
            device.destroy_framebuffer(fb, None);
            device.destroy_image_view(view, None);
        }
    }

    /// Free every entry's view and framebuffer — the queue is idle.
    pub(crate) unsafe fn release_all(&mut self, device: &ash::Device) {
        for ix in 0..self.transient.len() {
            self.release(device, ix);
        }
    }

    /// Record the scaled present of `src` into `dst` (in `TRANSFER_DST_OPTIMAL`)
    /// at viewport `vp` inside `extent`, slate around it. Leaves `dst` in
    /// `TRANSFER_DST_OPTIMAL` and `src` in `src.layout`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        ix: usize,
        src: &SharpSource,
        dst: vk::Image,
        format: vk::Format,
        extent: vk::Extent2D,
        vp: (u32, u32, u32, u32),
        clear: [f32; 4],
    ) -> Result<(), vk::Result> {
        self.release(device, ix);
        let view = device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(dst)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(super::color_subresource_range()),
            None,
        )?;
        let views = [view];
        let fb = match device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(self.render_pass)
                .attachments(&views)
                .width(extent.width)
                .height(extent.height)
                .layers(1),
            None,
        ) {
            Ok(fb) => fb,
            Err(e) => {
                device.destroy_image_view(view, None);
                return Err(e);
            }
        };
        self.transient[ix] = Some((view, fb));
        let set = self.sets[ix];
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(src.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info)],
            &[],
        );
        let to_read = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(src.layout)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(src.image)
            .subresource_range(super::color_subresource_range())];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_read,
        );
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue { float32: clear },
        }];
        device.cmd_begin_render_pass(
            cmd,
            &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(fb)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .clear_values(&clear_values),
            vk::SubpassContents::INLINE,
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        device.cmd_set_viewport(
            cmd,
            0,
            &[vk::Viewport {
                x: vp.0 as f32,
                y: vp.1 as f32,
                width: vp.2 as f32,
                height: vp.3 as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        device.cmd_set_scissor(
            cmd,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D {
                    x: vp.0 as i32,
                    y: vp.1 as i32,
                },
                extent: vk::Extent2D {
                    width: vp.2,
                    height: vp.3,
                },
            }],
        );
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[set], &[]);
        let size = [src.width as f32, src.height as f32];
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&size[0].to_le_bytes());
        bytes[4..].copy_from_slice(&size[1].to_le_bytes());
        device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, &bytes);
        device.cmd_draw(cmd, 3, 1, 0, 0);
        device.cmd_end_render_pass(cmd);
        let back = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(src.layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(src.image)
            .subresource_range(super::color_subresource_range())];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &back,
        );
        Ok(())
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        self.release_all(device);
        if self.pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.pool, None);
        }
        if self.pipeline != vk::Pipeline::null() {
            device.destroy_pipeline(self.pipeline, None);
        }
        if self.sampler != vk::Sampler::null() {
            device.destroy_sampler(self.sampler, None);
        }
        if self.layout != vk::PipelineLayout::null() {
            device.destroy_pipeline_layout(self.layout, None);
        }
        if self.set_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        if self.render_pass != vk::RenderPass::null() {
            device.destroy_render_pass(self.render_pass, None);
        }
        self.sets.clear();
    }
}
