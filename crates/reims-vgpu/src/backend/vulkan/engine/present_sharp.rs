//! TEMP (lab, `REIMS_VGPU_LAB_PRESENT_SHARP=on`): the host window's scaled
//! present as one edge-adaptive draw (FSR 1.0 EASU) instead of a linear
//! `vkCmdBlitImage`; with `REIMS_VGPU_LAB_PRESENT_RCAS` also FSR 1.0's RCAS
//! sharpening, as a second draw over the EASU result.
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
//!
//! With RCAS, EASU draws into one device-local image the size of the visible
//! part (`mid`) and RCAS draws from it into the swapchain image. One image
//! serves every frame entry: its render pass's incoming dependency waits for
//! the fragment reads of every earlier submission on the queue before the
//! next EASU overwrites it. A size change retires the old image to `retired`,
//! freed once every frame entry has come round since (each round retired that
//! entry's fence, so nothing in flight can still read it).

use ash::vk;

use super::present_sharp_shader::{PRESENT_RCAS_FRAG_SPIRV, PRESENT_SHARP_FRAG_SPIRV, PRESENT_SHARP_VERT_SPIRV};

/// `REIMS_VGPU_LAB_PRESENT_SHARP=on`. Latched.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("REIMS_VGPU_LAB_PRESENT_SHARP").as_deref(), Ok("on" | "1")))
}

/// `REIMS_VGPU_LAB_PRESENT_RCAS`: `on` = 0.2 stops (FSR's default), or a
/// number of stops in 0..=4 (0 the strongest, each stop halves it). The
/// linear factor `exp2(-stops)`, or `None` when unset/off. Latched; only
/// read with `PRESENT_SHARP` on.
pub(crate) fn rcas_sharp() -> Option<f32> {
    static SHARP: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *SHARP.get_or_init(|| {
        let v = std::env::var("REIMS_VGPU_LAB_PRESENT_RCAS").ok()?;
        let stops = match v.trim() {
            "on" => 0.2,
            "off" | "" => return None,
            s => s.parse::<f32>().ok().filter(|x| (0.0..=4.0).contains(x))?,
        };
        Some((-stops as f32).exp2())
    })
}

/// The EASU result RCAS reads: device-local, the size of the visible part.
struct MidImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    fb: vk::Framebuffer,
    width: u32,
    height: u32,
}

impl MidImage {
    unsafe fn destroy(self, device: &ash::Device) {
        if self.fb != vk::Framebuffer::null() {
            device.destroy_framebuffer(self.fb, None);
        }
        if self.view != vk::ImageView::null() {
            device.destroy_image_view(self.view, None);
        }
        if self.image != vk::Image::null() {
            device.destroy_image(self.image, None);
        }
        if self.memory != vk::DeviceMemory::null() {
            device.free_memory(self.memory, None);
        }
    }
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
    /// RCAS: its linear strength, or `None` (one EASU draw, as before).
    rcas: Option<f32>,
    /// RCAS: EASU into `mid` (`UNDEFINED -> SHADER_READ_ONLY_OPTIMAL`).
    mid_pass: vk::RenderPass,
    rcas_pipeline: vk::Pipeline,
    /// RCAS: one per frame entry, reading `mid`.
    rcas_sets: Vec<vk::DescriptorSet>,
    mid: Option<MidImage>,
    /// Images a size change replaced, with the `records` count at retirement.
    retired: Vec<(MidImage, u64)>,
    records: u64,
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
            rcas: rcas_sharp(),
            mid_pass: vk::RenderPass::null(),
            rcas_pipeline: vk::Pipeline::null(),
            rcas_sets: Vec::new(),
            mid: None,
            retired: Vec::new(),
            records: 0,
        };
        if let Err(e) = made.build(device, format, frames) {
            made.destroy(device);
            return Err(e);
        }
        if let Some(sharp) = made.rcas {
            crate::observe::off(format!("present_sharp rcas sharp={sharp:.3}"));
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
        // EASU pushes the source size (8 bytes); RCAS the visible part's
        // origin and size and its strength (20).
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(if self.rcas.is_some() { 20 } else { 8 })];
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
        // Both pipelines are built against the swapchain pass; `mid_pass` has
        // the same single attachment format and sample count, so it is
        // render-pass compatible with it and EASU runs in either.
        self.pipeline = self.make_pipeline(device, &PRESENT_SHARP_FRAG_SPIRV)?;
        let per_frame = if self.rcas.is_some() { 2 } else { 1 };
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count((frames * per_frame) as u32)];
        self.pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets((frames * per_frame) as u32)
                .pool_sizes(&sizes),
            None,
        )?;
        let layouts = vec![self.set_layout; frames];
        self.sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&layouts),
        )?;
        if self.rcas.is_some() {
            self.rcas_pipeline = self.make_pipeline(device, &PRESENT_RCAS_FRAG_SPIRV)?;
            self.rcas_sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&layouts),
            )?;
            let attachment = [vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            // In: the previous present's RCAS read of `mid` (any earlier
            // submission) before EASU overwrites it - an execution dependency
            // is enough for a write after a read. Out: EASU's writes visible
            // to RCAS's fragment reads.
            let deps = [
                vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                vk::SubpassDependency::default()
                    .src_subpass(0)
                    .dst_subpass(vk::SUBPASS_EXTERNAL)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ),
            ];
            self.mid_pass = device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachment)
                    .subpasses(&subpass)
                    .dependencies(&deps),
                None,
            )?;
        }
        Ok(())
    }

    /// One full-screen-triangle pipeline with `frag`, against `render_pass`.
    unsafe fn make_pipeline(&self, device: &ash::Device, frag: &[u32]) -> Result<vk::Pipeline, vk::Result> {
        let vert = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&PRESENT_SHARP_VERT_SPIRV),
            None,
        )?;
        let frag = match device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(frag), None) {
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
        Ok(pipelines.map_err(|(_, e)| e)?[0])
    }

    /// Free entry `ix`'s view and framebuffer. Only once its fence retired.
    pub(crate) unsafe fn release(&mut self, device: &ash::Device, ix: usize) {
        if let Some((view, fb)) = self.transient.get_mut(ix).and_then(Option::take) {
            device.destroy_framebuffer(fb, None);
            device.destroy_image_view(view, None);
        }
    }

    /// Free every entry's view and framebuffer, and every retired EASU image —
    /// the queue is idle.
    pub(crate) unsafe fn release_all(&mut self, device: &ash::Device) {
        for ix in 0..self.transient.len() {
            self.release(device, ix);
        }
        for (old, _) in self.retired.drain(..) {
            old.destroy(device);
        }
    }

    /// `mid` at `width` x `height`, made (and the old one retired) on a size
    /// change. `memory_type` picks a device-local type for the image's bits.
    unsafe fn ensure_mid(
        &mut self,
        device: &ash::Device,
        format: vk::Format,
        width: u32,
        height: u32,
        memory_type: &dyn Fn(u32, u64) -> Option<u32>,
    ) -> Result<(), vk::Result> {
        if self.mid.as_ref().is_some_and(|m| (m.width, m.height) == (width, height)) {
            return Ok(());
        }
        if let Some(old) = self.mid.take() {
            self.retired.push((old, self.records));
        }
        let mut made = MidImage {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            fb: vk::Framebuffer::null(),
            width,
            height,
        };
        let r = (|| -> Result<(), vk::Result> {
            made.image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let req = device.get_image_memory_requirements(made.image);
            let mt = memory_type(req.memory_type_bits, req.size).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
            made.memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            )?;
            device.bind_image_memory(made.image, made.memory, 0)?;
            made.view = device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(made.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(super::color_subresource_range()),
                None,
            )?;
            let views = [made.view];
            made.fb = device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.mid_pass)
                    .attachments(&views)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )?;
            Ok(())
        })();
        match r {
            Ok(()) => {
                self.mid = Some(made);
                Ok(())
            }
            Err(e) => {
                made.destroy(device);
                Err(e)
            }
        }
    }

    unsafe fn write_set(&self, device: &ash::Device, set: vk::DescriptorSet, view: vk::ImageView) {
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info)],
            &[],
        );
    }

    unsafe fn set_rect(device: &ash::Device, cmd: vk::CommandBuffer, x: u32, y: u32, w: u32, h: u32) {
        device.cmd_set_viewport(
            cmd,
            0,
            &[vk::Viewport {
                x: x as f32,
                y: y as f32,
                width: w as f32,
                height: h as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        device.cmd_set_scissor(
            cmd,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D {
                    x: x as i32,
                    y: y as i32,
                },
                extent: vk::Extent2D { width: w, height: h },
            }],
        );
    }

    /// Record the scaled present of `src` into `dst` (in `TRANSFER_DST_OPTIMAL`)
    /// at viewport `vp` inside `extent`, slate around it. Leaves `dst` in
    /// `TRANSFER_DST_OPTIMAL` and `src` in `src.layout`. `memory_type` picks a
    /// device-local memory type (RCAS's EASU image).
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
        memory_type: &dyn Fn(u32, u64) -> Option<u32>,
    ) -> Result<(), vk::Result> {
        self.release(device, ix);
        self.records += 1;
        // Every entry has come round since these were retired.
        let frames = self.transient.len() as u64;
        let records = self.records;
        let mut keep = Vec::with_capacity(self.retired.len());
        for (old, at) in self.retired.drain(..) {
            if records > at + frames + 1 {
                old.destroy(device);
            } else {
                keep.push((old, at));
            }
        }
        self.retired = keep;
        let rcas = match self.rcas {
            Some(sharp) => match self.ensure_mid(device, format, vp.2, vp.3, memory_type) {
                Ok(()) => Some(sharp),
                Err(e) => {
                    // EASU alone this present; the next one tries again.
                    if crate::observe::first_sight("present_rcas_mid_refused", 0) {
                        crate::observe::fail(format!("present_rcas_mid_refused {}x{} err={e:?}", vp.2, vp.3));
                    }
                    None
                }
            },
            None => None,
        };
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
        self.write_set(device, set, src.view);
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
        let size = [src.width as f32, src.height as f32];
        let mut easu_push = [0u8; 8];
        easu_push[..4].copy_from_slice(&size[0].to_le_bytes());
        easu_push[4..].copy_from_slice(&size[1].to_le_bytes());
        if let (Some(sharp), Some(mid)) = (rcas, self.mid.as_ref()) {
            // EASU over the whole of `mid`.
            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.mid_pass)
                    .framebuffer(mid.fb)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: mid.width,
                            height: mid.height,
                        },
                    }),
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            Self::set_rect(device, cmd, 0, 0, mid.width, mid.height);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[set], &[]);
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, &easu_push);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
            // RCAS from `mid` into the visible part of the window.
            let rcas_set = self.rcas_sets[ix];
            self.write_set(device, rcas_set, mid.view);
            let mut push = [0u8; 20];
            for (i, v) in [vp.0, vp.1, vp.2, vp.3].iter().enumerate() {
                push[i * 4..i * 4 + 4].copy_from_slice(&(*v as i32).to_le_bytes());
            }
            push[16..].copy_from_slice(&sharp.to_le_bytes());
            self.begin_window_pass(device, cmd, fb, extent, clear);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.rcas_pipeline);
            Self::set_rect(device, cmd, vp.0, vp.1, vp.2, vp.3);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[rcas_set], &[]);
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, &push);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        } else {
            self.begin_window_pass(device, cmd, fb, extent, clear);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            Self::set_rect(device, cmd, vp.0, vp.1, vp.2, vp.3);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[set], &[]);
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, &easu_push);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }
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

    /// The swapchain pass over the whole window, slate-cleared.
    unsafe fn begin_window_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        fb: vk::Framebuffer,
        extent: vk::Extent2D,
        clear: [f32; 4],
    ) {
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
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        self.release_all(device);
        if let Some(mid) = self.mid.take() {
            mid.destroy(device);
        }
        if self.pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.pool, None);
        }
        if self.pipeline != vk::Pipeline::null() {
            device.destroy_pipeline(self.pipeline, None);
        }
        if self.rcas_pipeline != vk::Pipeline::null() {
            device.destroy_pipeline(self.rcas_pipeline, None);
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
        if self.mid_pass != vk::RenderPass::null() {
            device.destroy_render_pass(self.mid_pass, None);
        }
        if self.render_pass != vk::RenderPass::null() {
            device.destroy_render_pass(self.render_pass, None);
        }
        self.sets.clear();
        self.rcas_sets.clear();
    }
}
