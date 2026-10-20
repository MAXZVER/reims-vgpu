//! TEMP (lab, `REIMS_VGPU_LAB_PRESENT_FG=on`): frame generation in the host
//! window's scaled present - between two guest frames the window shows a frame
//! built from the hardware optical flow between them (`VK_NV_optical_flow`, the
//! GPU's optical-flow engine, beside its shader cores), so a guest drawing 60
//! frames a second reaches the screen at 120.
//!
//! Per guest frame the window
//! 1. copies the guest frame into one of two optical-flow input images (a
//!    graphics submission; the frame stays in the layout the registry recorded),
//! 2. when a midpoint is wanted, runs the flow between the previous and this
//!    input in both directions on the optical-flow queue,
//! 3. builds the midpoint with `shaders/present_fg.comp` inside the frame's own
//!    command buffer, which waits for the flow, and scales that instead of the
//!    guest frame; the guest frame itself is presented half an interval later
//!    by the window loop.
//!
//! RTX 4060, 3456x1458: flow 1.7 ms (FAST, 4x4 grid) on the optical-flow
//! engine, the warp well under a millisecond on the shader cores.

use ash::vk;

use super::present_sharp_shader::PRESENT_FG_COMP_SPIRV;

/// `REIMS_VGPU_LAB_PRESENT_FG=on`. Latched.
pub(crate) fn on() -> bool {
    crate::host_window::present::lab_present_fg()
}

struct FgImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Flow images only: the same texels viewed as `R16G16_SINT` for the warp.
    int_view: vk::ImageView,
}

impl FgImage {
    unsafe fn destroy(&self, device: &ash::Device) {
        if self.int_view != vk::ImageView::null() {
            device.destroy_image_view(self.int_view, None);
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

pub(crate) struct PresentFg {
    width: u32,
    height: u32,
    input_format: vk::Format,
    session: vk::OpticalFlowSessionNV,
    inputs: [FgImage; 2],
    flow_fwd: FgImage,
    flow_bwd: FgImage,
    mid: FgImage,
    /// Which input the next guest frame is copied into.
    next: usize,
    /// Whether the other input holds the previous guest frame.
    have_prev: bool,
    /// The guest frame sequence last copied (a re-present of it copies nothing).
    last_seq: u64,
    /// The input pair (previous, current) the last flow ran on, for the warp,
    /// and the guest frame sequence of its current half.
    pair: Option<(usize, usize)>,
    pair_seq: u64,
    gfx_pool: vk::CommandPool,
    of_pool: vk::CommandPool,
    copy_cmds: Vec<vk::CommandBuffer>,
    copy_fences: Vec<vk::Fence>,
    of_cmds: Vec<vk::CommandBuffer>,
    copied: Vec<vk::Semaphore>,
    flowed: Vec<vk::Semaphore>,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

/// What the scaled present reads instead of the guest frame.
pub(crate) struct FgSource {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub layout: vk::ImageLayout,
}

const GRID: u32 = 4;

impl PresentFg {
    /// Everything for one guest frame size. `None` when the device has no
    /// optical flow or refuses a piece; the caller then presents without it.
    pub(crate) unsafe fn create(
        ctx: &super::context::DeviceContext,
        width: u32,
        height: u32,
        frames: usize,
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
    ) -> Result<Self, vk::Result> {
        let of = ctx.optical_flow.as_ref().ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        let device = &ctx.device;
        let formats = |usage: vk::OpticalFlowUsageFlagsNV| -> Vec<vk::Format> {
            let info = vk::OpticalFlowImageFormatInfoNV::default().usage(usage);
            let mut n = 0u32;
            let f = of.instance.fp().get_physical_device_optical_flow_image_formats_nv;
            let _ = f(ctx.pd, &info, &mut n, std::ptr::null_mut());
            let mut props = vec![vk::OpticalFlowImageFormatPropertiesNV::default(); n as usize];
            let _ = f(ctx.pd, &info, &mut n, props.as_mut_ptr());
            props.into_iter().map(|p| p.format).collect()
        };
        let inputs_ok = formats(vk::OpticalFlowUsageFlagsNV::INPUT);
        let outputs_ok = formats(vk::OpticalFlowUsageFlagsNV::OUTPUT);
        let input_format = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM]
            .into_iter()
            .find(|f| inputs_ok.contains(f))
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?;
        if !outputs_ok.contains(&vk::Format::R16G16_S10_5_NV) {
            return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
        }
        crate::observe::off(format!(
            "present_fg formats input={input_format:?} (offered {inputs_ok:?}) output={outputs_ok:?} {width}x{height}"
        ));
        let families = [ctx.gq, of.family];
        let null_image = || FgImage {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            int_view: vk::ImageView::null(),
        };
        let mut made = Self {
            width,
            height,
            input_format,
            session: vk::OpticalFlowSessionNV::null(),
            inputs: [null_image(), null_image()],
            flow_fwd: null_image(),
            flow_bwd: null_image(),
            mid: null_image(),
            next: 0,
            have_prev: false,
            last_seq: 0,
            pair: None,
            pair_seq: 0,
            gfx_pool: vk::CommandPool::null(),
            of_pool: vk::CommandPool::null(),
            copy_cmds: Vec::new(),
            copy_fences: Vec::new(),
            of_cmds: Vec::new(),
            copied: Vec::new(),
            flowed: Vec::new(),
            sampler: vk::Sampler::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
        };
        let r = (|| -> Result<(), vk::Result> {
            let make_image = |w: u32,
                              h: u32,
                              format: vk::Format,
                              usage: vk::ImageUsageFlags,
                              of_usage: Option<vk::OpticalFlowUsageFlagsNV>,
                              int_view: bool|
             -> Result<FgImage, vk::Result> {
                let mut img = null_image();
                let mut of_info = vk::OpticalFlowImageFormatInfoNV::default()
                    .usage(of_usage.unwrap_or(vk::OpticalFlowUsageFlagsNV::empty()));
                let mut ci = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                if of_usage.is_some() {
                    ci = ci
                        .sharing_mode(vk::SharingMode::CONCURRENT)
                        .queue_family_indices(&families)
                        .push_next(&mut of_info);
                }
                if int_view {
                    ci = ci.flags(vk::ImageCreateFlags::MUTABLE_FORMAT);
                }
                img.image = device.create_image(&ci, None)?;
                let req = device.get_image_memory_requirements(img.image);
                let mt = memory_type(req.memory_type_bits, req.size, false)
                    .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
                img.memory = device.allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
                    None,
                )?;
                device.bind_image_memory(img.image, img.memory, 0)?;
                let view = |f: vk::Format| {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(img.image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(f)
                            .subresource_range(super::color_subresource_range()),
                        None,
                    )
                };
                img.view = view(format)?;
                if int_view {
                    img.int_view = view(vk::Format::R16G16_SINT)?;
                }
                Ok(img)
            };
            let input_usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
            for i in 0..2 {
                self_inputs_set(&mut made, i, make_image(width, height, input_format, input_usage, Some(vk::OpticalFlowUsageFlagsNV::INPUT), false)?);
            }
            let (fw, fh) = (width.div_ceil(GRID), height.div_ceil(GRID));
            made.flow_fwd = make_image(
                fw,
                fh,
                vk::Format::R16G16_S10_5_NV,
                vk::ImageUsageFlags::SAMPLED,
                Some(vk::OpticalFlowUsageFlagsNV::OUTPUT),
                true,
            )?;
            made.flow_bwd = make_image(
                fw,
                fh,
                vk::Format::R16G16_S10_5_NV,
                vk::ImageUsageFlags::SAMPLED,
                Some(vk::OpticalFlowUsageFlagsNV::OUTPUT),
                true,
            )?;
            made.mid = make_image(
                width,
                height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
                None,
                false,
            )?;
            let sci = vk::OpticalFlowSessionCreateInfoNV::default()
                .width(width)
                .height(height)
                .image_format(input_format)
                .flow_vector_format(vk::Format::R16G16_S10_5_NV)
                .output_grid_size(vk::OpticalFlowGridSizeFlagsNV::TYPE_4X4)
                .performance_level(vk::OpticalFlowPerformanceLevelNV::FAST)
                .flags(vk::OpticalFlowSessionCreateFlagsNV::BOTH_DIRECTIONS);
            let r = (of.device.fp().create_optical_flow_session_nv)(device.handle(), &sci, std::ptr::null(), &mut made.session);
            if r != vk::Result::SUCCESS {
                return Err(r);
            }
            made.gfx_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.gq)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            made.of_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(of.family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let alloc = |pool| {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(frames as u32),
                )
            };
            made.copy_cmds = alloc(made.gfx_pool)?;
            made.of_cmds = alloc(made.of_pool)?;
            for _ in 0..frames {
                made.copy_fences.push(device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?);
                made.copied.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
                made.flowed.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
            }
            made.sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..5u32)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(if b == 4 {
                            vk::DescriptorType::STORAGE_IMAGE
                        } else {
                            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        })
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            made.set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(16)];
            let set_layouts = [made.set_layout];
            made.layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&ranges),
                None,
            )?;
            let module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&PRESENT_FG_COMP_SPIRV),
                None,
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let made_pipeline = device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(made.layout)],
                None,
            );
            device.destroy_shader_module(module, None);
            made.pipeline = made_pipeline.map_err(|(_, e)| e)?[0];
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(4 * frames as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(frames as u32),
            ];
            made.pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(frames as u32).pool_sizes(&sizes),
                None,
            )?;
            let layouts = vec![made.set_layout; frames];
            made.sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(made.pool)
                    .set_layouts(&layouts),
            )?;
            Ok(())
        })();
        match r {
            Ok(()) => Ok(made),
            Err(e) => {
                made.destroy(ctx);
                Err(e)
            }
        }
    }

    pub(crate) fn matches(&self, width: u32, height: u32) -> bool {
        (self.width, self.height) == (width, height)
    }

    /// Steps 1 and 2 for one present: copy guest frame `seq` (in `layout`, last
    /// written at `src_stage`/`src_access`) into the next input, and when
    /// `want_mid` run the flow from the previous input to it. Returns the
    /// semaphore the frame's command buffer must wait on before the warp, or
    /// `None` when there is no midpoint this time (first frame, a re-present,
    /// or not wanted).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn submit_flow(
        &mut self,
        ctx: &super::context::DeviceContext,
        ix: usize,
        src: vk::Image,
        layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        seq: u64,
        want_mid: bool,
    ) -> Result<Option<vk::Semaphore>, vk::Result> {
        if seq == self.last_seq {
            // A re-present of the frame already copied: no new pair.
            return Ok(None);
        }
        let Some(of) = ctx.optical_flow.as_ref() else {
            return Ok(None);
        };
        let device = &ctx.device;
        let cur = self.next;
        let prev = cur ^ 1;
        let flow = want_mid && self.have_prev;
        // Step 1: the copy, on the graphics queue, leaving `src` as it found it.
        device.wait_for_fences(&[self.copy_fences[ix]], true, u64::MAX)?;
        device.reset_fences(&[self.copy_fences[ix]])?;
        let cmd = self.copy_cmds[ix];
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let range = super::color_subresource_range();
        let barrier = |image, from, to, src_a, dst_a| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src_a)
                .dst_access_mask(dst_a)
                .old_layout(from)
                .new_layout(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        // First scope ALL_COMMANDS: the input this copy overwrites was read by
        // the previous pair's warps (compute, earlier frames on this queue),
        // which `src_stage` (the guest frame's last writer) does not cover.
        let _ = src_stage;
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[
                barrier(src, layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src_access, vk::AccessFlags::TRANSFER_READ),
                barrier(
                    self.inputs[cur].image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                ),
            ],
        );
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let corner = vk::Offset3D { x: self.width as i32, y: self.height as i32, z: 1 };
        device.cmd_blit_image(
            cmd,
            src,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.inputs[cur].image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit {
                src_subresource: layers,
                src_offsets: [vk::Offset3D::default(), corner],
                dst_subresource: layers,
                dst_offsets: [vk::Offset3D::default(), corner],
            }],
            vk::Filter::NEAREST,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[
                barrier(src, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, layout, vk::AccessFlags::empty(), vk::AccessFlags::empty()),
                barrier(
                    self.inputs[cur].image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::GENERAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::MEMORY_READ,
                ),
            ],
        );
        device.end_command_buffer(cmd)?;
        let signal: Vec<vk::Semaphore> = if flow { vec![self.copied[ix]] } else { Vec::new() };
        match ctx.queue_owner.as_ref() {
            Some(owner) => owner.submit_sync_ordered(&[cmd], &[], &[], &signal, self.copy_fences[ix])?,
            None => device.queue_submit(
                ctx.queue(),
                &[vk::SubmitInfo::default().command_buffers(&[cmd]).signal_semaphores(&signal)],
                self.copy_fences[ix],
            )?,
        }
        self.last_seq = seq;
        self.next = prev;
        let had_prev = self.have_prev;
        self.have_prev = true;
        if !flow || !had_prev {
            self.pair = None;
            return Ok(None);
        }
        // Step 2: the flow, on the optical-flow queue, after the copy.
        let ocmd = self.of_cmds[ix];
        device.reset_command_buffer(ocmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            ocmd,
            &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        device.cmd_pipeline_barrier(
            ocmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[
                barrier(self.flow_fwd.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::AccessFlags::empty(), vk::AccessFlags::MEMORY_WRITE),
                barrier(self.flow_bwd.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::AccessFlags::empty(), vk::AccessFlags::MEMORY_WRITE),
            ],
        );
        let bind = of.device.fp().bind_optical_flow_session_image_nv;
        for (point, view) in [
            (vk::OpticalFlowSessionBindingPointNV::INPUT, self.inputs[prev].view),
            (vk::OpticalFlowSessionBindingPointNV::REFERENCE, self.inputs[cur].view),
            (vk::OpticalFlowSessionBindingPointNV::FLOW_VECTOR, self.flow_fwd.view),
            (vk::OpticalFlowSessionBindingPointNV::BACKWARD_FLOW_VECTOR, self.flow_bwd.view),
        ] {
            let r = bind(device.handle(), self.session, point, view, vk::ImageLayout::GENERAL);
            if r != vk::Result::SUCCESS {
                device.end_command_buffer(ocmd)?;
                return Err(r);
            }
        }
        let exec = vk::OpticalFlowExecuteInfoNV::default();
        (of.device.fp().cmd_optical_flow_execute_nv)(ocmd, self.session, &exec);
        device.end_command_buffer(ocmd)?;
        let waits = [self.copied[ix]];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let signals = [self.flowed[ix]];
        let cmds = [ocmd];
        device.queue_submit(
            of.queue,
            &[vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signals)],
            vk::Fence::null(),
        )?;
        self.pair = Some((prev, cur));
        self.pair_seq = seq;
        Ok(Some(self.flowed[ix]))
    }

    /// Step 3, recorded into the frame's command buffer: the frame at `phase`
    /// (0 = the previous guest frame, 1 = `seq`) into `mid`, left in
    /// `TRANSFER_SRC_OPTIMAL` like a guest frame the present reads. The first
    /// present of a pair waits on the semaphore [`Self::submit_flow`] returned;
    /// later ones follow it on the queue (the barrier below waits for all
    /// earlier commands). `None` when the pair is not `seq`'s.
    pub(crate) unsafe fn record_warp(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        ix: usize,
        seq: u64,
        phase: f32,
    ) -> Option<FgSource> {
        let (prev, cur) = self.pair.filter(|_| self.pair_seq == seq)?;
        let range = super::color_subresource_range();
        let to_general = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.mid.image)
            .subresource_range(range)];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            &[],
            &to_general,
        );
        let set = self.sets[ix];
        let sampled = |view| {
            [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)]
        };
        let (a, b, f, g) = (
            sampled(self.inputs[prev].view),
            sampled(self.inputs[cur].view),
            sampled(self.flow_fwd.int_view),
            sampled(self.flow_bwd.int_view),
        );
        let out = [vk::DescriptorImageInfo::default()
            .image_view(self.mid.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let w = |binding: u32, ty: vk::DescriptorType| {
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(ty)
        };
        device.update_descriptor_sets(
            &[
                w(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&a),
                w(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&b),
                w(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&f),
                w(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&g),
                w(4, vk::DescriptorType::STORAGE_IMAGE).image_info(&out),
            ],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[set], &[]);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(self.width as i32).to_le_bytes());
        push[4..8].copy_from_slice(&(self.height as i32).to_le_bytes());
        push[8..12].copy_from_slice(&(GRID as i32).to_le_bytes());
        push[12..16].copy_from_slice(&phase.clamp(0.0, 1.0).to_le_bytes());
        device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
        device.cmd_dispatch(cmd, self.width.div_ceil(16), self.height.div_ceil(16), 1);
        let to_src = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.mid.image)
            .subresource_range(range)];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_src,
        );
        Some(FgSource {
            image: self.mid.image,
            view: self.mid.view,
            layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        })
    }

    /// Release everything (the device must be idle, or every frame entry retired).
    pub(crate) unsafe fn destroy(&mut self, ctx: &super::context::DeviceContext) {
        let device = &ctx.device;
        for f in self.copy_fences.drain(..) {
            let _ = device.wait_for_fences(&[f], true, u64::MAX);
            device.destroy_fence(f, None);
        }
        for s in self.copied.drain(..).chain(self.flowed.drain(..)) {
            device.destroy_semaphore(s, None);
        }
        if self.pipeline != vk::Pipeline::null() {
            device.destroy_pipeline(self.pipeline, None);
        }
        if self.layout != vk::PipelineLayout::null() {
            device.destroy_pipeline_layout(self.layout, None);
        }
        if self.pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.pool, None);
        }
        if self.set_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        if self.sampler != vk::Sampler::null() {
            device.destroy_sampler(self.sampler, None);
        }
        if self.gfx_pool != vk::CommandPool::null() {
            device.destroy_command_pool(self.gfx_pool, None);
        }
        if self.of_pool != vk::CommandPool::null() {
            device.destroy_command_pool(self.of_pool, None);
        }
        if let Some(of) = ctx.optical_flow.as_ref() {
            if self.session != vk::OpticalFlowSessionNV::null() {
                (of.device.fp().destroy_optical_flow_session_nv)(device.handle(), self.session, std::ptr::null());
            }
        }
        for img in self.inputs.iter().chain([&self.flow_fwd, &self.flow_bwd, &self.mid]) {
            img.destroy(device);
        }
        let _ = self.input_format;
    }
}

fn self_inputs_set(made: &mut PresentFg, i: usize, img: FgImage) {
    made.inputs[i] = img;
}
