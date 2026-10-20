//! TEMP (lab, `REIMS_VGPU_LAB_PRESENT_VSR_TILES=on`): NGX VSR on the parts of
//! the screen that stand still (Mish, 2026-10-20) - the window shows the fast
//! scaled present (EASU) everywhere, and over it the VSR-refined copy of every
//! cell that has not changed since it was refined.
//!
//! The guest frame is cut into cells of about 54x54 pixels, a multiple of the
//! scale's step so every cell edge falls on a whole window pixel (3456 -> 5120
//! is 27 -> 40). Per present, in the present's own command buffer:
//! 1. `shaders/present_refine_diff.comp` compares the guest frame with the last
//!    one it saw (`prev`, rgba8, also VSR's input), stamps every changed cell
//!    and drops its refined copy - before the window pass draws, so a stale
//!    cell never shows over new content;
//! 2. cells that have stood still for the settle time and hold no refined copy
//!    are merged into rectangles and run through VSR (with an apron of context
//!    around each, into `scratch`), within a per-present budget; the cells'
//!    part is copied into `cache` and the cells marked refined;
//! 3. inside the window pass, after the scaled draw, `shaders/present_refine.frag`
//!    draws `cache` over every refined cell.
//! The cells' stamps and flags live in one host-visible buffer the CPU reads
//! (without waiting) to pick the next cells; the GPU-side flags alone decide
//! what is drawn.
//!
//! RTX 4060, 3456x1458 -> 5120x2160, VSR Ultra: ~0.17 ms per call plus
//! ~2.8 ms per output megapixel (vsrbench `tiles`).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ash::vk;

use super::present_sharp_shader::{PRESENT_REFINE_DIFF_COMP_SPIRV, PRESENT_REFINE_FRAG_SPIRV, PRESENT_SHARP_VERT_SPIRV};

/// `REIMS_VGPU_LAB_PRESENT_VSR_TILES=on`: with `REIMS_VGPU_LAB_PRESENT_VSR`
/// (its quality), VSR per still cell instead of per settled frame. Latched.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("REIMS_VGPU_LAB_PRESENT_VSR_TILES").as_deref(), Ok("on" | "1")))
}

/// `REIMS_VGPU_LAB_PRESENT_VSR_TILES_MS`: the GPU time one present may spend
/// refining (estimated from the cost model below), default 3 ms. Latched.
fn budget_ms() -> f64 {
    static MS: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_PRESENT_VSR_TILES_MS")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|ms| *ms > 0.0)
            .unwrap_or(3.0)
    })
}

/// `REIMS_VGPU_LAB_PRESENT_VSR_TILES_SETTLE_MS`: how long a cell must stand
/// still before it is refined, default 120 ms. Latched.
fn settle() -> Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    Duration::from_millis(*MS.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_PRESENT_VSR_TILES_SETTLE_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(120)
    }))
}

/// VSR's GPU time per output megapixel at quality 0..=4 (RTX 4060, 3456 ->
/// 5120 full frames, vsrbench), and per call.
const MS_PER_OUT_MP: [f64; 5] = [0.3, 0.79, 0.95, 2.1, 2.83];
const MS_PER_CALL: f64 = 0.17;
/// The smallest input rectangle VSR is run on (below ~128 px NGX's Ultra costs
/// no more than Low, so the extra is spent on context instead).
const MIN_RECT: u32 = 128;
/// The largest merged rectangle, in cells per side.
const MAX_RECT_CELLS: u32 = 10;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Cells of a guest frame `iw` x `ih` shown at `ow` x `oh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub iw: u32,
    pub ih: u32,
    pub ow: u32,
    pub oh: u32,
    /// The scale as `step_out / step_in` in lowest terms.
    step_in: u32,
    step_out: u32,
    cell_in: u32,
    cell_out: u32,
    apron_in: u32,
    cols: u32,
    rows: u32,
}

impl Geometry {
    /// `None` when the scale is not an upscale by the same exact ratio on both
    /// axes with a step small enough for cells to land on whole pixels.
    pub(crate) fn new(iw: u32, ih: u32, ow: u32, oh: u32) -> Option<Self> {
        if iw == 0 || ih == 0 || ow <= iw || oh <= ih || u64::from(ih) * u64::from(ow) != u64::from(oh) * u64::from(iw) {
            return None;
        }
        let g = gcd(iw, ow);
        let (step_in, step_out) = (iw / g, ow / g);
        if step_in > 64 || ih % step_in != 0 {
            return None;
        }
        let cell_in = step_in * ((54 + step_in / 2) / step_in).max(1);
        let cell_out = cell_in / step_in * step_out;
        let apron_in = step_in * 24u32.div_ceil(step_in);
        let (cols, rows) = (iw.div_ceil(cell_in), ih.div_ceil(cell_in));
        if cols * rows > 8192 {
            return None;
        }
        Some(Self { iw, ih, ow, oh, step_in, step_out, cell_in, cell_out, apron_in, cols, rows })
    }

    fn count(&self) -> u32 {
        self.cols * self.rows
    }

    fn to_out(&self, v: u32) -> u32 {
        v / self.step_in * self.step_out
    }

    /// The input rectangle VSR runs on for cells `[cx0, cx1) x [cy0, cy1)`:
    /// the cells plus an apron, at least [`MIN_RECT`] where the frame allows,
    /// on step boundaries.
    fn vsr_in_rect(&self, cx0: u32, cy0: u32, cx1: u32, cy1: u32) -> [u32; 4] {
        let span = |c0: u32, c1: u32, limit: u32| -> (u32, u32) {
            let mut a = (c0 * self.cell_in).saturating_sub(self.apron_in);
            let mut b = (c1 * self.cell_in + self.apron_in).min(limit);
            while b - a < MIN_RECT && (a > 0 || b < limit) {
                a = a.saturating_sub(self.step_in);
                b = (b + self.step_in).min(limit);
            }
            (a, b)
        };
        let (x0, x1) = span(cx0, cx1, self.iw);
        let (y0, y1) = span(cy0, cy1, self.ih);
        [x0, y0, x1 - x0, y1 - y0]
    }
}

struct RefineImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl RefineImage {
    const NULL: Self = Self {
        image: vk::Image::null(),
        memory: vk::DeviceMemory::null(),
        view: vk::ImageView::null(),
    };

    unsafe fn destroy(&self, device: &ash::Device) {
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

pub(crate) struct TileRefine {
    geo: Geometry,
    /// The last guest frame the diff saw, rgba8 at its size; VSR's input.
    prev: RefineImage,
    prev_layout: vk::ImageLayout,
    /// The refined cells at the window size (rgba8), always `GENERAL`.
    cache: RefineImage,
    cache_ready: bool,
    /// VSR's output for the rectangle being refined, then copied from.
    scratch: RefineImage,
    /// `2 * count` words: stamps, then refined flags. Host-visible, mapped.
    cells: vk::Buffer,
    cells_memory: vk::DeviceMemory,
    cells_ptr: usize,
    sampler: vk::Sampler,
    diff_set_layout: vk::DescriptorSetLayout,
    diff_layout: vk::PipelineLayout,
    diff_pipeline: vk::Pipeline,
    draw_set_layout: vk::DescriptorSetLayout,
    draw_layout: vk::PipelineLayout,
    draw_pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    /// One per presenter frame entry (the guest frame's view changes).
    diff_sets: Vec<vk::DescriptorSet>,
    draw_set: vk::DescriptorSet,
    /// This present's stamp, and when recent stamps were made.
    stamp: u32,
    stamps: VecDeque<(u32, Instant)>,
    created_at: Instant,
    /// The stamp at which each cell's refine was recorded (0: none pending).
    scheduled: Vec<u32>,
    /// The guest frame sequence the cells were last compared against.
    diffed_seq: Option<u64>,
    /// Census: refines recorded, cells refined, presents that refined.
    rects: u64,
    cells_refined: u64,
}

/// The guest frame one present shows (for a generated midpoint, the real
/// frame behind it).
pub(crate) struct RefineFrame {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub layout: vk::ImageLayout,
}

impl TileRefine {
    pub(crate) fn geometry(&self) -> Geometry {
        self.geo
    }

    /// The refined-cell cache (`GENERAL`) and its size, for PRESENT_DUMP.
    pub(crate) fn cache_image(&self) -> (vk::Image, u32, u32) {
        (self.cache.image, self.geo.ow, self.geo.oh)
    }

    /// Everything for one geometry; `render_pass` is the window pass the
    /// overlay draws in.
    pub(crate) unsafe fn create(
        device: &ash::Device,
        geo: Geometry,
        render_pass: vk::RenderPass,
        frames: usize,
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
    ) -> Result<Self, vk::Result> {
        let count = geo.count() as usize;
        let mut made = Self {
            geo,
            prev: RefineImage::NULL,
            prev_layout: vk::ImageLayout::UNDEFINED,
            cache: RefineImage::NULL,
            cache_ready: false,
            scratch: RefineImage::NULL,
            cells: vk::Buffer::null(),
            cells_memory: vk::DeviceMemory::null(),
            cells_ptr: 0,
            sampler: vk::Sampler::null(),
            diff_set_layout: vk::DescriptorSetLayout::null(),
            diff_layout: vk::PipelineLayout::null(),
            diff_pipeline: vk::Pipeline::null(),
            draw_set_layout: vk::DescriptorSetLayout::null(),
            draw_layout: vk::PipelineLayout::null(),
            draw_pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            diff_sets: Vec::new(),
            draw_set: vk::DescriptorSet::null(),
            stamp: 0,
            stamps: VecDeque::new(),
            created_at: Instant::now(),
            scheduled: vec![0; count],
            diffed_seq: None,
            rects: 0,
            cells_refined: 0,
        };
        let r = (|| -> Result<(), vk::Result> {
            let make_image = |w: u32, h: u32, usage: vk::ImageUsageFlags| -> Result<RefineImage, vk::Result> {
                let mut img = RefineImage::NULL;
                let r = (|| -> Result<(), vk::Result> {
                    img.image = device.create_image(
                        &vk::ImageCreateInfo::default()
                            .image_type(vk::ImageType::TYPE_2D)
                            .format(vk::Format::R8G8B8A8_UNORM)
                            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .usage(usage)
                            .initial_layout(vk::ImageLayout::UNDEFINED),
                        None,
                    )?;
                    let req = device.get_image_memory_requirements(img.image);
                    let mt = memory_type(req.memory_type_bits, req.size, false).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
                    img.memory = device.allocate_memory(
                        &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
                        None,
                    )?;
                    device.bind_image_memory(img.image, img.memory, 0)?;
                    img.view = device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(img.image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::R8G8B8A8_UNORM)
                            .subresource_range(super::color_subresource_range()),
                        None,
                    )?;
                    Ok(())
                })();
                match r {
                    Ok(()) => Ok(img),
                    Err(e) => {
                        img.destroy(device);
                        Err(e)
                    }
                }
            };
            made.prev = make_image(geo.iw, geo.ih, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)?;
            made.cache = make_image(
                geo.ow,
                geo.oh,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            )?;
            made.scratch = make_image(
                geo.ow,
                geo.oh,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            )?;
            let bytes = (2 * count * 4) as u64;
            made.cells = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )?;
            let req = device.get_buffer_memory_requirements(made.cells);
            let mt = memory_type(req.memory_type_bits, req.size, true).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
            made.cells_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
                None,
            )?;
            device.bind_buffer_memory(made.cells, made.cells_memory, 0)?;
            let ptr = device.map_memory(made.cells_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;
            std::ptr::write_bytes(ptr as *mut u8, 0, bytes as usize);
            let whole = [vk::MappedMemoryRange::default()
                .memory(made.cells_memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            device.flush_mapped_memory_ranges(&whole)?;
            made.cells_ptr = ptr as usize;
            made.sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            // The diff: frame (sampled), prev (storage), cells.
            let diff_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            made.diff_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&diff_bindings),
                None,
            )?;
            let diff_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(24)];
            let diff_set_layouts = [made.diff_set_layout];
            made.diff_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&diff_set_layouts)
                    .push_constant_ranges(&diff_ranges),
                None,
            )?;
            let module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&PRESENT_REFINE_DIFF_COMP_SPIRV),
                None,
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipelines = device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(made.diff_layout)],
                None,
            );
            device.destroy_shader_module(module, None);
            made.diff_pipeline = pipelines.map_err(|(_, e)| e)?[0];
            // The overlay: cache (sampled), cells.
            let draw_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];
            made.draw_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&draw_bindings),
                None,
            )?;
            let draw_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(20)];
            let draw_set_layouts = [made.draw_set_layout];
            made.draw_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&draw_set_layouts)
                    .push_constant_ranges(&draw_ranges),
                None,
            )?;
            made.draw_pipeline = draw_pipeline(device, made.draw_layout, render_pass)?;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(frames as u32 + 1),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(frames as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(frames as u32 + 1),
            ];
            made.pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(frames as u32 + 1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = vec![made.diff_set_layout; frames];
            made.diff_sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(made.pool)
                    .set_layouts(&layouts),
            )?;
            let layouts = [made.draw_set_layout];
            made.draw_set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(made.pool)
                    .set_layouts(&layouts),
            )?[0];
            let cache_info = [vk::DescriptorImageInfo::default()
                .sampler(made.sampler)
                .image_view(made.cache.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let cells_info = [vk::DescriptorBufferInfo::default()
                .buffer(made.cells)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(made.draw_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&cache_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(made.draw_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&cells_info),
                ],
                &[],
            );
            Ok(())
        })();
        match r {
            Ok(()) => {
                crate::observe::off(format!(
                    "present_refine geometry {}x{}->{}x{} step={}/{} cell={}->{} apron={} cells={}x{} budget_ms={} settle_ms={}",
                    geo.iw,
                    geo.ih,
                    geo.ow,
                    geo.oh,
                    geo.step_in,
                    geo.step_out,
                    geo.cell_in,
                    geo.cell_out,
                    geo.apron_in,
                    geo.cols,
                    geo.rows,
                    budget_ms(),
                    settle().as_millis()
                ));
                Ok(made)
            }
            Err(e) => {
                made.destroy(device);
                Err(e)
            }
        }
    }

    fn cell_words(&self) -> &[u32] {
        // SAFETY: `cells_ptr` maps `2 * count` words for this object's life;
        // the GPU may be writing them - every value read is a hint only.
        unsafe { std::slice::from_raw_parts(self.cells_ptr as *const u32, 2 * self.geo.count() as usize) }
    }

    /// Step 1, every present: stamp the cells the guest frame changed since
    /// the last present and drop their refined copies. Records nothing on a
    /// layout it cannot read.
    pub(crate) unsafe fn record_diff(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        ix: usize,
        frame: &RefineFrame,
        seq: Option<u64>,
    ) {
        // The same guest frame again (a generated or refine present): nothing
        // can have changed since its own comparison.
        if seq.is_some() && seq == self.diffed_seq {
            return;
        }
        self.diffed_seq = seq;
        self.stamp = self.stamp.wrapping_add(1).max(1);
        let now = Instant::now();
        self.stamps.push_back((self.stamp, now));
        while self.stamps.len() > 2048 {
            self.stamps.pop_front();
        }
        let range = super::color_subresource_range();
        let image_barrier = |image, from, to, src, dst| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(from)
                .new_layout(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        // In: the frame to sampling (as the scaled present's own barrier
        // does), prev to storage, and the cells after any earlier reader.
        let memory = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        let images = [
            image_barrier(
                frame.image,
                frame.layout,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_READ,
            ),
            image_barrier(
                self.prev.image,
                self.prev_layout,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            ),
        ];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &memory,
            &[],
            &images,
        );
        self.prev_layout = vk::ImageLayout::GENERAL;
        let set = self.diff_sets[ix];
        let frame_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(frame.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let prev_info = [vk::DescriptorImageInfo::default()
            .image_view(self.prev.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let cells_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.cells)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&frame_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&prev_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&cells_info),
            ],
            &[],
        );
        let g = self.geo;
        let mut push = [0u8; 24];
        for (i, v) in [g.iw, g.ih, g.cell_in, g.cols, g.count(), self.stamp].iter().enumerate() {
            push[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.diff_pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.diff_layout, 0, &[set], &[]);
        device.cmd_push_constants(cmd, self.diff_layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
        device.cmd_dispatch(cmd, g.cols, g.rows, 1);
        // Out: the frame back to the layout it came in.
        let back = [image_barrier(
            frame.image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            frame.layout,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_READ,
        )];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &back,
        );
    }

    /// Step 2, every present: refine still cells within the budget, then make
    /// the cells' flags and `cache` visible to the overlay. Returns when the
    /// window should present again for more (`None`: nothing pending).
    pub(crate) unsafe fn record_refine(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        quality: i32,
    ) -> Option<Instant> {
        let g = self.geo;
        let count = g.count() as usize;
        let now = Instant::now();
        let settle = settle();
        let _ = device.invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
            .memory(self.cells_memory)
            .offset(0)
            .size(vk::WHOLE_SIZE)]);
        let words = self.cell_words().to_vec();
        // Stamps are pushed one by one, so a stamp's time is at a fixed offset
        // from the oldest in the ring (a scan per cell cost ~1M steps per
        // present in motion).
        let oldest = self.stamps.front().map(|&(s, _)| s);
        let when = |stamp: u32| -> Instant {
            if stamp == 0 {
                return self.created_at;
            }
            match oldest {
                // Older than the ring: long settled.
                Some(o) if stamp < o => self.created_at,
                Some(o) => self.stamps.get((stamp - o) as usize).map_or(now, |&(_, at)| at),
                None => now,
            }
        };
        // Cells that want a refine now, and the earliest one still settling.
        let mut ready = vec![false; count];
        let mut next_due: Option<Instant> = None;
        for i in 0..count {
            if words[count + i] != 0 {
                continue;
            }
            // A refine recorded in the last few presents has not landed in the
            // host's view yet.
            if self.scheduled[i] != 0 && self.stamp.wrapping_sub(self.scheduled[i]) < 6 && words[i] <= self.scheduled[i] {
                continue;
            }
            let due = when(words[i]) + settle;
            if due <= now {
                ready[i] = true;
            } else {
                next_due = Some(next_due.map_or(due, |d: Instant| d.min(due)));
            }
        }
        // Merge ready cells into rectangles: a run along a row, then down while
        // every cell of the run below is ready too.
        let mut taken = vec![false; count];
        let mut rects: Vec<(u32, u32, u32, u32)> = Vec::new();
        let mut spent = 0.0f64;
        let mut left_over = false;
        let ms_per_px = MS_PER_OUT_MP[quality.clamp(0, 4) as usize] / 1e6;
        let out_area = |r: [u32; 4]| f64::from(g.to_out(r[2])) * f64::from(g.to_out(r[3]));
        'scan: for cy in 0..g.rows {
            for cx in 0..g.cols {
                let i = (cy * g.cols + cx) as usize;
                if !ready[i] || taken[i] {
                    continue;
                }
                let mut cx1 = cx + 1;
                while cx1 < g.cols && cx1 - cx < MAX_RECT_CELLS && ready[(cy * g.cols + cx1) as usize] && !taken[(cy * g.cols + cx1) as usize] {
                    cx1 += 1;
                }
                let mut cy1 = cy + 1;
                while cy1 < g.rows
                    && cy1 - cy < MAX_RECT_CELLS
                    && (cx..cx1).all(|x| ready[(cy1 * g.cols + x) as usize] && !taken[(cy1 * g.cols + x) as usize])
                {
                    cy1 += 1;
                }
                let cost = MS_PER_CALL + out_area(g.vsr_in_rect(cx, cy, cx1, cy1)) * ms_per_px;
                if !rects.is_empty() && spent + cost > budget_ms() {
                    left_over = true;
                    break 'scan;
                }
                spent += cost;
                for y in cy..cy1 {
                    for x in cx..cx1 {
                        taken[(y * g.cols + x) as usize] = true;
                    }
                }
                rects.push((cx, cy, cx1, cy1));
            }
        }
        let range = super::color_subresource_range();
        let image_barrier = |image, from, to, src, dst| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(from)
                .new_layout(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        if !self.cache_ready {
            let to_general = [image_barrier(
                self.cache.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ,
            )];
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_general,
            );
            self.cache_ready = true;
        }
        if !rects.is_empty() {
            // prev (written by the diff) to VSR's input; scratch to its output.
            let images = [
                // (Already there when this present skipped the diff.)
                image_barrier(
                    self.prev.image,
                    self.prev_layout,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                ),
                image_barrier(
                    self.scratch.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::SHADER_WRITE,
                ),
            ];
            // FRAGMENT in the first scope: the previous present's overlay read
            // the cache this refine's copies write (with the diff skipped there
            // is no other barrier to carry that dependency).
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &images,
            );
            self.prev_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            let input = super::present_vsr::VsrImage {
                image: self.prev.image,
                view: self.prev.view,
                format: vk::Format::R8G8B8A8_UNORM,
                width: g.iw,
                height: g.ih,
            };
            let output = super::present_vsr::VsrImage {
                image: self.scratch.image,
                view: self.scratch.view,
                format: vk::Format::R8G8B8A8_UNORM,
                width: g.ow,
                height: g.oh,
            };
            let layers = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            for (n, &(cx0, cy0, cx1, cy1)) in rects.iter().enumerate() {
                if n > 0 {
                    // The previous rectangle's copy read scratch before this
                    // rectangle's apron may overwrite it.
                    let war = [vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)];
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &war,
                        &[],
                        &[],
                    );
                }
                let in_rect = g.vsr_in_rect(cx0, cy0, cx1, cy1);
                let out_rect = [g.to_out(in_rect[0]), g.to_out(in_rect[1]), g.to_out(in_rect[2]), g.to_out(in_rect[3])];
                let r = super::present_vsr::evaluate_rect(cmd, &input, &output, in_rect, out_rect, quality);
                if r != 0 {
                    crate::observe::fail(format!("present_refine_refused reason=evaluate ngx={r:#x} in={in_rect:?} out={out_rect:?}"));
                    return None;
                }
                let raw = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &raw,
                    &[],
                    &[],
                );
                // The cells' own part (exact window pixels) into the cache.
                let x0 = cx0 * g.cell_out;
                let y0 = cy0 * g.cell_out;
                let x1 = (cx1 * g.cell_out).min(g.ow);
                let y1 = (cy1 * g.cell_out).min(g.oh);
                let offset = vk::Offset3D { x: x0 as i32, y: y0 as i32, z: 0 };
                device.cmd_copy_image(
                    cmd,
                    self.scratch.image,
                    vk::ImageLayout::GENERAL,
                    self.cache.image,
                    vk::ImageLayout::GENERAL,
                    &[vk::ImageCopy {
                        src_subresource: layers,
                        src_offset: offset,
                        dst_subresource: layers,
                        dst_offset: offset,
                        extent: vk::Extent3D { width: x1 - x0, height: y1 - y0, depth: 1 },
                    }],
                );
                for cy in cy0..cy1 {
                    let first = (count as u32 + cy * g.cols + cx0) as u64 * 4;
                    device.cmd_fill_buffer(cmd, self.cells, first, u64::from(cx1 - cx0) * 4, 1);
                    for cx in cx0..cx1 {
                        self.scheduled[(cy * g.cols + cx) as usize] = self.stamp;
                    }
                }
                self.cells_refined += u64::from((cx1 - cx0) * (cy1 - cy0));
            }
            self.rects += rects.len() as u64;
            let n = self.rects;
            if n <= 8 || n % 500 < rects.len() as u64 {
                crate::observe::off(format!(
                    "present_refine refined rects={} this_present={} cells_total={} est_ms={spent:.2} left_over={left_over}",
                    n,
                    rects.len(),
                    self.cells_refined
                ));
            }
        }
        // Out: the diff's and the refine's writes to the overlay's reads, and
        // the cells' words to the host (Readback memory: without a HOST_READ
        // dependency the CPU's next look may see them arbitrarily late).
        let memory = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ)];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &memory,
            &[],
            &[],
        );
        if left_over {
            // More is ready: the next present, about a monitor period on (a
            // mailbox swapchain would otherwise take presents as fast as asked).
            Some(now + Duration::from_millis(6))
        } else {
            next_due
        }
    }

    /// Step 3, inside the window pass after its scaled draw: the refined
    /// cells over it. `vp` is the visible part (`x, y, width, height`), the
    /// viewport and scissor already set to it.
    pub(crate) unsafe fn record_overlay(&self, device: &ash::Device, cmd: vk::CommandBuffer, vp: (u32, u32, u32, u32)) {
        if !self.cache_ready {
            return;
        }
        let g = self.geo;
        let mut push = [0u8; 20];
        for (i, v) in [vp.0, vp.1, g.cell_out, g.cols, g.count()].iter().enumerate() {
            push[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.draw_pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.draw_layout, 0, &[self.draw_set], &[]);
        device.cmd_push_constants(cmd, self.draw_layout, vk::ShaderStageFlags::FRAGMENT, 0, &push);
        device.cmd_draw(cmd, 3, 1, 0, 0);
    }

    /// Only with the queue idle.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for p in [self.diff_pipeline, self.draw_pipeline] {
            if p != vk::Pipeline::null() {
                device.destroy_pipeline(p, None);
            }
        }
        for l in [self.diff_layout, self.draw_layout] {
            if l != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(l, None);
            }
        }
        for l in [self.diff_set_layout, self.draw_set_layout] {
            if l != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(l, None);
            }
        }
        if self.pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.pool, None);
        }
        if self.sampler != vk::Sampler::null() {
            device.destroy_sampler(self.sampler, None);
        }
        if self.cells_ptr != 0 {
            device.unmap_memory(self.cells_memory);
            self.cells_ptr = 0;
        }
        if self.cells != vk::Buffer::null() {
            device.destroy_buffer(self.cells, None);
        }
        if self.cells_memory != vk::DeviceMemory::null() {
            device.free_memory(self.cells_memory, None);
        }
        self.prev.destroy(device);
        self.cache.destroy(device);
        self.scratch.destroy(device);
        self.diff_pipeline = vk::Pipeline::null();
        self.draw_pipeline = vk::Pipeline::null();
        self.diff_layout = vk::PipelineLayout::null();
        self.draw_layout = vk::PipelineLayout::null();
        self.diff_set_layout = vk::DescriptorSetLayout::null();
        self.draw_set_layout = vk::DescriptorSetLayout::null();
        self.pool = vk::DescriptorPool::null();
        self.sampler = vk::Sampler::null();
        self.cells = vk::Buffer::null();
        self.cells_memory = vk::DeviceMemory::null();
        self.prev = RefineImage::NULL;
        self.cache = RefineImage::NULL;
        self.scratch = RefineImage::NULL;
    }
}

/// The overlay's pipeline: the window pass's full-viewport triangle and
/// `present_refine.frag`, no blending (a refined cell replaces the pass's).
unsafe fn draw_pipeline(device: &ash::Device, layout: vk::PipelineLayout, render_pass: vk::RenderPass) -> Result<vk::Pipeline, vk::Result> {
    let vert = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&PRESENT_SHARP_VERT_SPIRV), None)?;
    let frag = match device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&PRESENT_REFINE_FRAG_SPIRV), None) {
        Ok(m) => m,
        Err(e) => {
            device.destroy_shader_module(vert, None);
            return Err(e);
        }
    };
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachment = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
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
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0)];
    let pipelines = device.create_graphics_pipelines(vk::PipelineCache::null(), &info, None);
    device.destroy_shader_module(vert, None);
    device.destroy_shader_module(frag, None);
    Ok(pipelines.map_err(|(_, e)| e)?[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cells_of_3456_on_5120_land_on_whole_window_pixels() {
        let g = Geometry::new(3456, 1458, 5120, 2160).unwrap();
        assert_eq!((g.step_in, g.step_out), (27, 40));
        assert_eq!((g.cell_in, g.cell_out), (54, 80));
        assert_eq!((g.cols, g.rows), (64, 27));
        assert_eq!(g.apron_in, 27);
        // Every input rectangle VSR runs on maps to whole window pixels.
        for (cx0, cy0, cx1, cy1) in [(0, 0, 1, 1), (63, 26, 64, 27), (10, 5, 20, 15), (0, 0, 64, 27)] {
            let r = g.vsr_in_rect(cx0, cy0, cx1, cy1);
            for v in r {
                assert_eq!(v % g.step_in, 0, "{r:?}");
            }
            assert!(r[0] + r[2] <= g.iw && r[1] + r[3] <= g.ih, "{r:?}");
            assert!(r[2] >= MIN_RECT && r[3] >= MIN_RECT, "{r:?}");
            // The cells themselves lie inside the rectangle.
            assert!(r[0] <= cx0 * g.cell_in && r[1] <= cy0 * g.cell_in);
            assert!(r[0] + r[2] >= (cx1 * g.cell_in).min(g.iw) && r[1] + r[3] >= (cy1 * g.cell_in).min(g.ih));
        }
    }

    #[test]
    fn a_scale_without_whole_pixel_cells_has_no_geometry() {
        // Not the same ratio on both axes.
        assert!(Geometry::new(3456, 1458, 5120, 2000).is_none());
        // A downscale or 1:1.
        assert!(Geometry::new(5120, 2160, 3456, 1458).is_none());
        assert!(Geometry::new(3456, 1458, 3456, 1458).is_none());
        // 1536x648 on 1920x810: step 4 -> 5, cells of 56.
        let g = Geometry::new(1536, 648, 1920, 810).unwrap();
        assert_eq!((g.step_in, g.step_out, g.cell_in, g.cell_out), (4, 5, 56, 70));
    }
}
