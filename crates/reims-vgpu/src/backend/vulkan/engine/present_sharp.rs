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

/// `REIMS_VGPU_LAB_PRESENT_NIS`: NVIDIA Image Scaling (NVScaler) instead of
/// EASU (+RCAS), with its sharpness slider: `on` = 0.5 (the SDK samples'
/// default), or a number in 0..=1. `None` when unset/off. Latched; only read
/// with `PRESENT_SHARP` on. NVScaler upscales 1x..2x; outside that the present
/// keeps EASU.
pub(crate) fn nis_sharp() -> Option<f32> {
    static SHARP: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *SHARP.get_or_init(|| {
        let v = std::env::var("REIMS_VGPU_LAB_PRESENT_NIS").ok()?;
        match v.trim() {
            "on" => Some(0.5),
            "off" | "" => None,
            s => s.parse::<f32>().ok().filter(|x| (0.0..=1.0).contains(x)),
        }
    })
}

/// `REIMS_VGPU_LAB_PRESENT_DUMP=DIR` (diagnosis, 2026-10-19): when DIR/snap
/// exists, the next scaled present copies its source frame and its first
/// pass's result (EASU's or NVScaler's, what the window pass shows 1:1) to
/// host memory and writes them as DIR/NNN-src.ppm and DIR/NNN-out.ppm once
/// that present's frame entry comes round (its fence retired). Latched.
fn dump_dir() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var("REIMS_VGPU_LAB_PRESENT_DUMP").ok().filter(|d| !d.is_empty()))
        .as_deref()
}

/// One requested dump in flight: a host-visible buffer holding the source
/// (BGRA) and then the first pass's result.
struct PresentDump {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ix: usize,
    src: (u32, u32),
    out: (u32, u32),
    /// Whether the result is RGBA (NVScaler's storage image) or BGRA (EASU's).
    out_rgba: bool,
}

/// NVScaler's constants for one present (`NVScalerUpdateConfig`, SDR), as the
/// 28 words of `present_nis.comp`'s push block; `None` outside 1x..2x.
fn nis_config(sharpness: f32, src: (u32, u32), dst: (u32, u32)) -> Option<[u8; 112]> {
    let sharpen_slider = sharpness.clamp(0.0, 1.0) - 0.5;
    let (max_scale, min_scale, limit_scale) = if sharpen_slider >= 0.0 { (1.25, 1.25, 1.25) } else { (1.75, 1.0, 1.0) };
    let detect_ratio = 2.0 * 1127.0 / 1024.0;
    let detect_thres = 64.0 / 1024.0;
    let (min_contrast, max_contrast) = (2.0f32, 10.0f32);
    let (sharp_start_y, sharp_end_y) = (0.45f32, 0.9f32);
    let strength_min = (0.4 + sharpen_slider * min_scale * 1.2).max(0.0);
    let strength_max = 1.6 + sharpen_slider * max_scale * 1.8;
    let limit_min = (0.14 + sharpen_slider * limit_scale * 0.32).max(0.1);
    let limit_max = 0.5 + sharpen_slider * limit_scale * 0.6;
    let scale_x = src.0 as f32 / dst.0 as f32;
    let scale_y = src.1 as f32 / dst.1 as f32;
    if !(0.5..=1.0).contains(&scale_x) || !(0.5..=1.0).contains(&scale_y) {
        return None;
    }
    let floats = [
        detect_ratio,
        detect_thres,
        min_contrast,
        1.0 / (max_contrast - min_contrast),
        1.0,
        1.0 / 255.0,
        sharp_start_y,
        1.0 / (sharp_end_y - sharp_start_y),
        strength_min,
        strength_max - strength_min,
        limit_min,
        limit_max - limit_min,
        scale_x,
        scale_y,
        1.0 / dst.0 as f32,
        1.0 / dst.1 as f32,
        1.0 / src.0 as f32,
        1.0 / src.1 as f32,
    ];
    let uints = [0u32, 0, src.0, src.1, 0, 0, dst.0, dst.1];
    let mut out = [0u8; 112];
    for (i, f) in floats.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, u) in uints.iter().enumerate() {
        let o = 72 + i * 4;
        out[o..o + 4].copy_from_slice(&u.to_le_bytes());
    }
    Some(out)
}

/// NVScaler's two coefficient tables as 2 x 64 `R32G32B32A32_SFLOAT` images,
/// refilled from `buffer` by every present that uses them.
struct NisCoef {
    images: [(vk::Image, vk::DeviceMemory, vk::ImageView); 2],
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
}

impl NisCoef {
    unsafe fn destroy(self, device: &ash::Device) {
        for (image, memory, view) in self.images {
            if view != vk::ImageView::null() {
                device.destroy_image_view(view, None);
            }
            if image != vk::Image::null() {
                device.destroy_image(image, None);
            }
            if memory != vk::DeviceMemory::null() {
                device.free_memory(memory, None);
            }
        }
        if self.buffer != vk::Buffer::null() {
            device.destroy_buffer(self.buffer, None);
        }
        if self.buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(self.buffer_memory, None);
        }
    }
}

/// The first pass's result the window pass reads, the size of the visible
/// part: EASU's (a colour attachment, `fb`) or NVScaler's (a storage image,
/// no framebuffer).
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
    /// RCAS: its linear strength, or `None` (one EASU draw, as before). With
    /// NIS `Some(0.0)` (an exact copy of `mid`) unless PRESENT_RCAS is set too.
    rcas: Option<f32>,
    /// NIS: its sharpness, or `None` (EASU).
    nis: Option<f32>,
    nis_set_layout: vk::DescriptorSetLayout,
    nis_layout: vk::PipelineLayout,
    nis_pipeline: vk::Pipeline,
    nis_pool: vk::DescriptorPool,
    /// NIS: one per frame entry.
    nis_sets: Vec<vk::DescriptorSet>,
    nis_coef: Option<NisCoef>,
    /// RCAS: EASU into `mid` (`UNDEFINED -> SHADER_READ_ONLY_OPTIMAL`).
    mid_pass: vk::RenderPass,
    rcas_pipeline: vk::Pipeline,
    /// RCAS: one per frame entry, reading `mid`.
    rcas_sets: Vec<vk::DescriptorSet>,
    mid: Option<MidImage>,
    /// Images a size change replaced, with the `records` count at retirement.
    retired: Vec<(MidImage, u64)>,
    records: u64,
    /// `REIMS_VGPU_LAB_PRESENT_DUMP`: the dump in flight, and how many written.
    dump: Option<PresentDump>,
    dumps: u32,
    /// VSR (`present_vsr`): its quality, or `None`. Like NIS it writes `mid`
    /// (rgba8 storage) and the window pass copies it out at strength 0.
    vsr: Option<i32>,
    /// VSR: the guest frame converted to rgba8 at its own size (NGX's input).
    vsr_in: Option<MidImage>,
    /// VSR: the feature was created (recorded into an earlier present).
    vsr_created: bool,
    /// VSR: a create or evaluate refused; the present keeps EASU from here on.
    vsr_failed: bool,
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
            // With NIS the window pass is RCAS too: an exact copy unless
            // REIMS_VGPU_LAB_PRESENT_RCAS also asks for its sharpening.
            rcas: if nis_sharp().is_some() || super::present_vsr::quality().is_some() {
                Some(rcas_sharp().unwrap_or(0.0))
            } else {
                rcas_sharp()
            },
            nis: nis_sharp(),
            nis_set_layout: vk::DescriptorSetLayout::null(),
            nis_layout: vk::PipelineLayout::null(),
            nis_pipeline: vk::Pipeline::null(),
            nis_pool: vk::DescriptorPool::null(),
            nis_sets: Vec::new(),
            nis_coef: None,
            mid_pass: vk::RenderPass::null(),
            rcas_pipeline: vk::Pipeline::null(),
            rcas_sets: Vec::new(),
            mid: None,
            retired: Vec::new(),
            records: 0,
            dump: None,
            dumps: 0,
            vsr: super::present_vsr::quality(),
            vsr_in: None,
            vsr_created: false,
            vsr_failed: false,
        };
        if let Err(e) = made.build(device, format, frames) {
            made.destroy(device);
            return Err(e);
        }
        if let Some(sharp) = made.nis {
            crate::observe::off(format!("present_sharp nis sharpness={sharp:.3}"));
        } else if let Some(sharp) = made.rcas {
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
        if self.nis.is_some() {
            self.build_nis(device, frames)?;
        }
        Ok(())
    }

    /// NVScaler: set layout (sampler, source, output, two coefficient images),
    /// push range, compute pipeline, one set per frame entry.
    unsafe fn build_nis(&mut self, device: &ash::Device, frames: usize) -> Result<(), vk::Result> {
        let kinds = [
            (1, vk::DescriptorType::SAMPLER),
            (2, vk::DescriptorType::SAMPLED_IMAGE),
            (3, vk::DescriptorType::STORAGE_IMAGE),
            (4, vk::DescriptorType::SAMPLED_IMAGE),
            (5, vk::DescriptorType::SAMPLED_IMAGE),
        ];
        let bindings: Vec<_> = kinds
            .iter()
            .map(|&(b, t)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(t)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        self.nis_set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(112)];
        let set_layouts = [self.nis_set_layout];
        self.nis_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push),
            None,
        )?;
        let module = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&super::present_sharp_shader::PRESENT_NIS_COMP_SPIRV),
            None,
        )?;
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main"),
            )
            .layout(self.nis_layout)];
        let made = device.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
        device.destroy_shader_module(module, None);
        self.nis_pipeline = made.map_err(|(_, e)| e)?[0];
        let n = frames as u32;
        let sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLER).descriptor_count(n),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(3 * n),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(n),
        ];
        self.nis_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes),
            None,
        )?;
        let layouts = vec![self.nis_set_layout; frames];
        self.nis_sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.nis_pool)
                .set_layouts(&layouts),
        )?;
        Ok(())
    }

    /// The coefficient images, made on first use.
    unsafe fn ensure_nis_coef(
        &mut self,
        device: &ash::Device,
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
    ) -> Result<(), vk::Result> {
        if self.nis_coef.is_some() {
            return Ok(());
        }
        let mut made = NisCoef {
            images: [(vk::Image::null(), vk::DeviceMemory::null(), vk::ImageView::null()); 2],
            buffer: vk::Buffer::null(),
            buffer_memory: vk::DeviceMemory::null(),
        };
        let r = (|| -> Result<(), vk::Result> {
            for slot in made.images.iter_mut() {
                slot.0 = device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .extent(vk::Extent3D {
                            width: (super::present_nis_coef::NIS_TAPS / 4) as u32,
                            height: super::present_nis_coef::NIS_PHASES as u32,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )?;
                let req = device.get_image_memory_requirements(slot.0);
                let mt = memory_type(req.memory_type_bits, req.size, false).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
                slot.1 = device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mt),
                    None,
                )?;
                device.bind_image_memory(slot.0, slot.1, 0)?;
                slot.2 = device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(slot.0)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .subresource_range(super::color_subresource_range()),
                    None,
                )?;
            }
            made.buffer = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(Self::NIS_COEF_BYTES * 2)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )?;
            let req = device.get_buffer_memory_requirements(made.buffer);
            let mt = memory_type(req.memory_type_bits, req.size, false).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
            made.buffer_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            )?;
            device.bind_buffer_memory(made.buffer, made.buffer_memory, 0)?;
            Ok(())
        })();
        match r {
            Ok(()) => {
                self.nis_coef = Some(made);
                Ok(())
            }
            Err(e) => {
                made.destroy(device);
                Err(e)
            }
        }
    }

    /// Bytes of one coefficient table.
    const NIS_COEF_BYTES: u64 =
        (super::present_nis_coef::NIS_PHASES * super::present_nis_coef::NIS_TAPS * 4) as u64;

    /// Record the coefficient upload (`vkCmdUpdateBuffer`, then buffer ->
    /// image) into this present; leaves both images in
    /// `SHADER_READ_ONLY_OPTIMAL` ordered before compute reads.
    ///
    /// Every present, not once: the first version marked the tables uploaded
    /// when it recorded them, and a present whose command buffer never reached
    /// the queue (a swapchain recreated at the guest's mode switch) left them
    /// undefined for the rest of the boot - NVScaler then filtered with garbage
    /// weights: flat areas striped every few columns, the whole interface
    /// smeared (Mish 2026-10-19; the harness, whose one submit always lands,
    /// stayed clean). 4 KiB of updates per present cost nothing.
    unsafe fn upload_nis_coef(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        let Some(coef) = self.nis_coef.as_mut() else {
            return;
        };
        // The previous present's copy out of the buffer and NVScaler's reads of
        // the images finish before this one rewrites them (write after read:
        // an execution dependency, chained by the barriers below).
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[],
        );
        let tables = [&super::present_nis_coef::NIS_COEF_SCALE, &super::present_nis_coef::NIS_COEF_USM];
        for (i, t) in tables.iter().enumerate() {
            let mut bytes = Vec::with_capacity(Self::NIS_COEF_BYTES as usize);
            for row in t.iter() {
                for v in row {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            device.cmd_update_buffer(cmd, coef.buffer, i as u64 * Self::NIS_COEF_BYTES, &bytes);
        }
        let to_dst: Vec<_> = coef
            .images
            .iter()
            .map(|&(image, _, _)| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(super::color_subresource_range())
            })
            .collect();
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
            &[],
            &to_dst,
        );
        for (i, &(image, _, _)) in coef.images.iter().enumerate() {
            let region = [vk::BufferImageCopy::default()
                .buffer_offset(i as u64 * Self::NIS_COEF_BYTES)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: (super::present_nis_coef::NIS_TAPS / 4) as u32,
                    height: super::present_nis_coef::NIS_PHASES as u32,
                    depth: 1,
                })];
            device.cmd_copy_buffer_to_image(cmd, coef.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &region);
        }
        let to_read: Vec<_> = coef
            .images
            .iter()
            .map(|&(image, _, _)| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(super::color_subresource_range())
            })
            .collect();
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_read,
        );
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
        self.finish_dump(device, None);
    }

    /// `mid` at `width` x `height`, made (and the old one retired) on a size
    /// change. `memory_type` picks a device-local type for the image's bits.
    unsafe fn ensure_mid(
        &mut self,
        device: &ash::Device,
        format: vk::Format,
        width: u32,
        height: u32,
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
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
        // NVScaler writes `mid` as an rgba8 storage image; EASU renders it in
        // the window's format.
        let nis = self.nis.is_some() || self.vsr.is_some();
        let format = if nis { vk::Format::R8G8B8A8_UNORM } else { format };
        let usage = if nis {
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC
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
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let req = device.get_image_memory_requirements(made.image);
            let mt = memory_type(req.memory_type_bits, req.size, false).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
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
            if nis {
                return Ok(());
            }
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

    /// VSR's input: an rgba8 copy of the guest frame at its own size.
    unsafe fn ensure_vsr_in(
        &mut self,
        device: &ash::Device,
        width: u32,
        height: u32,
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
    ) -> Result<(), vk::Result> {
        if self.vsr_in.as_ref().is_some_and(|m| (m.width, m.height) == (width, height)) {
            return Ok(());
        }
        if let Some(old) = self.vsr_in.take() {
            self.retired.push((old, self.records));
        }
        let format = vk::Format::R8G8B8A8_UNORM;
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
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let req = device.get_image_memory_requirements(made.image);
            let mt = memory_type(req.memory_type_bits, req.size, false).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
            made.memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
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
            Ok(())
        })();
        match r {
            Ok(()) => {
                self.vsr_in = Some(made);
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
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
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
        self.finish_dump(device, Some(ix));
        // NIS: its constants for this scale and its coefficient images, or
        // EASU alone this present (outside 1x..2x, or no memory).
        let mut nis_cfg = match self.nis {
            Some(sharpness) => nis_config(sharpness, (src.width, src.height), (vp.2, vp.3)),
            None => None,
        };
        if nis_cfg.is_some() {
            if let Err(e) = self.ensure_nis_coef(device, memory_type) {
                if crate::observe::first_sight("present_nis_coef_refused", 0) {
                    crate::observe::fail(format!("present_nis_coef_refused err={e:?}"));
                }
                nis_cfg = None;
            }
        }
        // TEMP (lab, PRESENT_VSR_SETTLE): with a settle time VSR runs only on the
        // present that refines a settled frame; the others take EASU alone.
        let vsr_this = self.vsr.is_some()
            && (crate::host_window::present::lab_present_settle().is_none()
                || crate::host_window::present::lab_take_refine_present());
        let second = if (self.nis.is_some() && nis_cfg.is_none()) || (self.vsr.is_some() && !vsr_this) {
            None
        } else {
            self.rcas
        };
        let rcas = match second {
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
        // REIMS_VGPU_LAB_PRESENT_DUMP: the source as this present reads it.
        let dumping = rcas.is_some()
            && self.begin_dump(
                device,
                ix,
                (src.width, src.height),
                (vp.2, vp.3),
                // `mid` is rgba8 under NVScaler and VSR alike.
                (self.nis.is_some() && nis_cfg.is_some()) || self.vsr.is_some(),
                memory_type,
            );
        if let (true, Some(d)) = (dumping, self.dump.as_ref()) {
            Self::dump_copy(device, cmd, src.image, src.layout, d.buffer, 0, (src.width, src.height));
        }
        let to_read = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(src.layout)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(src.image)
            .subresource_range(super::color_subresource_range())];
        let nis_cfg = if rcas.is_some() { nis_cfg } else { None };
        // VSR this present: ready, a `mid`, and its input image.
        let vsr = match (self.vsr.filter(|_| vsr_this), rcas.is_some() && !self.vsr_failed && super::present_vsr::ready()) {
            (Some(q), true) => match self.ensure_vsr_in(device, src.width, src.height, memory_type) {
                Ok(()) => Some(q),
                Err(e) => {
                    if crate::observe::first_sight("present_vsr_in_refused", 0) {
                        crate::observe::fail(format!("present_vsr_in_refused err={e:?}"));
                    }
                    None
                }
            },
            _ => None,
        };
        if let (Some(q), Some(vin), Some(mid)) = (vsr, self.vsr_in.as_ref(), self.mid.as_ref()) {
            let (vin_image, vin_view, vin_w, vin_h) = (vin.image, vin.view, vin.width, vin.height);
            let (mid_image, mid_view, mid_w, mid_h) = (mid.image, mid.view, mid.width, mid.height);
            if !self.vsr_created {
                self.vsr_created = super::present_vsr::create_feature(device, cmd);
                if !self.vsr_created {
                    self.vsr_failed = true;
                }
            }
            if self.vsr_created {
                // The guest frame (in `src.layout`, a transfer-read layout) into
                // the rgba8 input at its own size; then NGX from it into `mid`.
                let barriers = [
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(vin_image)
                        .subresource_range(super::color_subresource_range()),
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(mid_image)
                        .subresource_range(super::color_subresource_range()),
                ];
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
                let whole = |w: u32, h: u32| [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: w as i32, y: h as i32, z: 1 }];
                let layers = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                device.cmd_blit_image(
                    cmd,
                    src.image,
                    src.layout,
                    vin_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageBlit {
                        src_subresource: layers,
                        src_offsets: whole(src.width, src.height),
                        dst_subresource: layers,
                        dst_offsets: whole(vin_w, vin_h),
                    }],
                    vk::Filter::NEAREST,
                );
                let vin_read = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(vin_image)
                    .subresource_range(super::color_subresource_range())];
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &vin_read,
                );
                let r = super::present_vsr::evaluate(
                    cmd,
                    &super::present_vsr::VsrImage {
                        image: vin_image,
                        view: vin_view,
                        format: vk::Format::R8G8B8A8_UNORM,
                        width: vin_w,
                        height: vin_h,
                    },
                    &super::present_vsr::VsrImage {
                        image: mid_image,
                        view: mid_view,
                        format: vk::Format::R8G8B8A8_UNORM,
                        width: mid_w,
                        height: mid_h,
                    },
                    q,
                );
                {
                    // TEMP (lab): how often VSR ran (with a settle time: once per settled frame).
                    static RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let n = RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if n <= 5 || n % 200 == 0 {
                        crate::observe::off(format!("present_vsr evaluate #{n} {vin_w}x{vin_h}->{mid_w}x{mid_h} ngx={r:#x}"));
                    }
                }
                if r != 0 {
                    self.vsr_failed = true;
                    crate::observe::fail(format!("present_vsr_refused reason=evaluate ngx={r:#x} {vin_w}x{vin_h}->{mid_w}x{mid_h}"));
                }
                // Whatever NGX did, `mid` goes to the window pass's layout
                // (a refused evaluate shows one undefined frame, then EASU).
                let mid_read = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(mid_image)
                    .subresource_range(super::color_subresource_range())];
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &mid_read,
                );
                let rcas_set = self.rcas_sets[ix];
                self.write_set(device, rcas_set, mid_view);
                let mut push = [0u8; 20];
                for (i, v) in [vp.0, vp.1, vp.2, vp.3].iter().enumerate() {
                    push[i * 4..i * 4 + 4].copy_from_slice(&(*v as i32).to_le_bytes());
                }
                push[16..].copy_from_slice(&rcas.unwrap_or(0.0).to_le_bytes());
                self.begin_window_pass(device, cmd, fb, extent, clear);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.rcas_pipeline);
                Self::set_rect(device, cmd, vp.0, vp.1, vp.2, vp.3);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[rcas_set], &[]);
                device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, &push);
                device.cmd_draw(cmd, 3, 1, 0, 0);
                device.cmd_end_render_pass(cmd);
                if dumping {
                    self.record_mid_dump(device, cmd, src.width, src.height);
                }
                return Ok(());
            }
        }
        let reader = if nis_cfg.is_some() {
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER
        } else {
            vk::PipelineStageFlags::FRAGMENT_SHADER
        };
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            reader,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_read,
        );
        let size = [src.width as f32, src.height as f32];
        let mut easu_push = [0u8; 8];
        easu_push[..4].copy_from_slice(&size[0].to_le_bytes());
        easu_push[4..].copy_from_slice(&size[1].to_le_bytes());
        if let (Some(cfg), Some(mid)) = (nis_cfg, self.mid.as_ref()) {
            // NVScaler into `mid`, then an exact copy into the window (the
            // RCAS pass at strength 0).
            let (mid_image, mid_view, mid_w, mid_h) = (mid.image, mid.view, mid.width, mid.height);
            self.upload_nis_coef(device, cmd);
            let coef = self
                .nis_coef
                .as_ref()
                .map(|c| (c.images[0].2, c.images[1].2))
                .unwrap_or_default();
            let set = self.nis_sets[ix];
            let sampler_info = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
            let src_info = [vk::DescriptorImageInfo::default()
                .image_view(src.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let out_info = [vk::DescriptorImageInfo::default()
                .image_view(mid_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let scale_info = [vk::DescriptorImageInfo::default()
                .image_view(coef.0)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let usm_info = [vk::DescriptorImageInfo::default()
                .image_view(coef.1)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let w = |b: u32, t: vk::DescriptorType| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b)
                    .descriptor_type(t)
            };
            device.update_descriptor_sets(
                &[
                    w(1, vk::DescriptorType::SAMPLER).image_info(&sampler_info),
                    w(2, vk::DescriptorType::SAMPLED_IMAGE).image_info(&src_info),
                    w(3, vk::DescriptorType::STORAGE_IMAGE).image_info(&out_info),
                    w(4, vk::DescriptorType::SAMPLED_IMAGE).image_info(&scale_info),
                    w(5, vk::DescriptorType::SAMPLED_IMAGE).image_info(&usm_info),
                ],
                &[],
            );
            // In: the previous present's window-pass read of `mid` (any earlier
            // submission) before NVScaler overwrites it.
            let to_general = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(mid_image)
                .subresource_range(super::color_subresource_range())];
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_general,
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.nis_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.nis_layout, 0, &[set], &[]);
            device.cmd_push_constants(cmd, self.nis_layout, vk::ShaderStageFlags::COMPUTE, 0, &cfg);
            // NIS_BLOCK_WIDTH x NIS_BLOCK_HEIGHT output pixels per group.
            device.cmd_dispatch(cmd, mid_w.div_ceil(32), mid_h.div_ceil(24), 1);
            let mid_read = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(mid_image)
                .subresource_range(super::color_subresource_range())];
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &mid_read,
            );
            let rcas_set = self.rcas_sets[ix];
            self.write_set(device, rcas_set, mid_view);
            let mut push = [0u8; 20];
            for (i, v) in [vp.0, vp.1, vp.2, vp.3].iter().enumerate() {
                push[i * 4..i * 4 + 4].copy_from_slice(&(*v as i32).to_le_bytes());
            }
            push[16..].copy_from_slice(&rcas.unwrap_or(0.0).to_le_bytes());
            self.begin_window_pass(device, cmd, fb, extent, clear);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.rcas_pipeline);
            Self::set_rect(device, cmd, vp.0, vp.1, vp.2, vp.3);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[rcas_set], &[]);
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, &push);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        } else if let (Some(sharp), Some(mid)) = (rcas.filter(|_| self.nis.is_none()), self.mid.as_ref()) {
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
        // REIMS_VGPU_LAB_PRESENT_DUMP: the first pass's result, as the window
        // pass just read it.
        if dumping {
            self.record_mid_dump(device, cmd, src.width, src.height);
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
            reader,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &back,
        );
        Ok(())
    }

    /// REIMS_VGPU_LAB_PRESENT_DUMP: `mid` (in SHADER_READ_ONLY) after the
    /// source copy the dump already holds.
    unsafe fn record_mid_dump(&self, device: &ash::Device, cmd: vk::CommandBuffer, src_w: u32, src_h: u32) {
        let (Some(d), Some(mid)) = (self.dump.as_ref(), self.mid.as_ref()) else {
            return;
        };
            let to_src = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(mid.image)
                .subresource_range(super::color_subresource_range())];
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_src,
            );
            let offset = src_w as u64 * src_h as u64 * 4;
            Self::dump_copy(device, cmd, mid.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, d.buffer, offset, (mid.width, mid.height));
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
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

    /// `REIMS_VGPU_LAB_PRESENT_DUMP`: a buffer for this present's dump when
    /// DIR/snap asks for one and none is in flight; the request file is
    /// consumed.
    unsafe fn begin_dump(
        &mut self,
        device: &ash::Device,
        ix: usize,
        src: (u32, u32),
        out: (u32, u32),
        out_rgba: bool,
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
    ) -> bool {
        let Some(dir) = dump_dir() else {
            return false;
        };
        if self.dump.is_some() {
            return false;
        }
        let snap = format!("{dir}/snap");
        if std::fs::remove_file(&snap).is_err() {
            return false;
        }
        let size = (src.0 as u64 * src.1 as u64 + out.0 as u64 * out.1 as u64) * 4;
        let Ok(buffer) = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST),
            None,
        ) else {
            return false;
        };
        let req = device.get_buffer_memory_requirements(buffer);
        let memory = memory_type(req.memory_type_bits, req.size, true).and_then(|mt| {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mt),
                    None,
                )
                .ok()
        });
        let Some(memory) = memory else {
            device.destroy_buffer(buffer, None);
            crate::observe::fail("present_dump_refused memory".to_string());
            return false;
        };
        if device.bind_buffer_memory(buffer, memory, 0).is_err() {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
            return false;
        }
        self.dump = Some(PresentDump { buffer, memory, ix, src, out, out_rgba });
        true
    }

    /// Write the dump in flight once its frame entry `ix` comes round again.
    unsafe fn finish_dump(&mut self, device: &ash::Device, ix: Option<usize>) {
        let Some(d) = self.dump.as_ref() else {
            return;
        };
        if ix.is_some_and(|i| i != d.ix) {
            return;
        }
        let d = self.dump.take().unwrap();
        let size = (d.src.0 as u64 * d.src.1 as u64 + d.out.0 as u64 * d.out.1 as u64) * 4;
        if let (Some(dir), Ok(ptr)) = (dump_dir(), device.map_memory(d.memory, 0, size, vk::MemoryMapFlags::empty())) {
            let _ = device.invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                .memory(d.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)]);
            let bytes = std::slice::from_raw_parts(ptr as *const u8, size as usize);
            let src_len = (d.src.0 * d.src.1 * 4) as usize;
            let n = self.dumps;
            self.dumps += 1;
            let write = |name: String, (w, h): (u32, u32), px: &[u8], rgba: bool| {
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                out.reserve((w * h * 3) as usize);
                for p in px.chunks_exact(4) {
                    if rgba {
                        out.extend_from_slice(&[p[0], p[1], p[2]]);
                    } else {
                        out.extend_from_slice(&[p[2], p[1], p[0]]);
                    }
                }
                let _ = std::fs::write(name, out);
            };
            write(format!("{dir}/{n:03}-src.ppm"), d.src, &bytes[..src_len], false);
            write(format!("{dir}/{n:03}-out.ppm"), d.out, &bytes[src_len..], d.out_rgba);
            device.unmap_memory(d.memory);
            crate::observe::fail(format!(
                "present_dump {n:03} src={}x{} out={}x{} rgba={}",
                d.src.0, d.src.1, d.out.0, d.out.1, d.out_rgba
            ));
        }
        device.destroy_buffer(d.buffer, None);
        device.free_memory(d.memory, None);
    }

    /// Copy `image` (in `layout`, a transfer-readable one) into the dump
    /// buffer at `offset`.
    unsafe fn dump_copy(
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        buffer: vk::Buffer,
        offset: u64,
        (w, h): (u32, u32),
    ) {
        let region = [vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 })];
        device.cmd_copy_image_to_buffer(cmd, image, layout, buffer, &region);
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
        if let Some(coef) = self.nis_coef.take() {
            coef.destroy(device);
        }
        if self.nis_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.nis_pool, None);
        }
        if self.nis_pipeline != vk::Pipeline::null() {
            device.destroy_pipeline(self.nis_pipeline, None);
        }
        if self.nis_layout != vk::PipelineLayout::null() {
            device.destroy_pipeline_layout(self.nis_layout, None);
        }
        if self.nis_set_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.nis_set_layout, None);
        }
        self.nis_sets.clear();
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
