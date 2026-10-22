//! TEMP (lab, `REIMS_VGPU_LAB_PRESENT_RAVU=<path to ravu-zoom-ar-r3-rgb.hook>`): the host window's scaled present
//! upscaled by RAVU-Zoom-AR r3 (`shaders/present_ravu.comp`) into `mid` in EASU's place, the window pass (RCAS) after
//! it as before. A compute pass over 16x16 output tiles ([`RavuPass`]); the per-pixel fragment form it replaces cost
//! ~8 ms a present on the lab's RTX 4060 (b986b) - see the shader's header and lab/tools/ravubench.
//!
//! The shader is ours; its two weight tables are not. They live in the mpv user shader the path names
//! (bjin/mpv-prescalers, LGPL-3.0-or-later) as `//!TEXTURE` blocks - little-endian f32 words in hex, though the
//! block declares `rgba16f` - and are read from there at run time, never embedded or committed. They are uploaded
//! as `R16G16B16A16_SFLOAT` images (linear filtering on that format is required by Vulkan; on the f32 one it is
//! not), from a host-visible staging buffer written once.
//!
//! The upload is recorded into presents until one of them is known to have run: [`RavuLut::confirm`] is called
//! when that present's frame entry comes round (its fence retired). NVScaler's tables, marked uploaded when
//! recorded, stayed undefined for a whole boot when that present never reached the queue (a swapchain recreated
//! at the guest's mode switch, Mish 2026-10-19); [`RavuLut::unconfirm`] covers the idle at a swapchain recreate.

use ash::vk;

/// `REIMS_VGPU_LAB_PRESENT_RAVU`: the hook file, or `None` when unset/off. Latched.
pub(crate) fn hook_path() -> Option<&'static str> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_PRESENT_RAVU")
            .ok()
            .filter(|v| !v.is_empty() && v != "off")
    })
    .as_deref()
}

/// One `//!TEXTURE` block: its name, width, height and RGBA texels.
struct HookTexture {
    name: String,
    width: u32,
    height: u32,
    texels: Vec<f32>,
}

/// The `//!TEXTURE` blocks of an mpv user shader.
fn hook_textures(text: &str) -> Result<Vec<HookTexture>, String> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let Some(name) = line.strip_prefix("//!TEXTURE ") else {
            continue;
        };
        let (mut size, mut data) = (None, None);
        for l in lines.by_ref() {
            if let Some(s) = l.strip_prefix("//!SIZE ") {
                let v: Vec<u32> = s
                    .split_whitespace()
                    .filter_map(|x| x.parse().ok())
                    .collect();
                size = (v.len() == 2).then(|| (v[0], v[1]));
            } else if !l.starts_with("//!") {
                data = Some(l.trim());
                break;
            }
        }
        let (width, height) = size.ok_or_else(|| format!("{name}: no //!SIZE"))?;
        let hex = data.ok_or_else(|| format!("{name}: no data"))?;
        let words = width as usize * height as usize * 4;
        if hex.len() != words * 8 {
            return Err(format!(
                "{name}: {} hex digits for {width}x{height} f32 rgba",
                hex.len()
            ));
        }
        let mut texels = Vec::with_capacity(words);
        for k in 0..words {
            let chunk = &hex[k * 8..k * 8 + 8];
            let mut bytes = [0u8; 4];
            for (b, pair) in bytes.iter_mut().zip(chunk.as_bytes().chunks_exact(2)) {
                let s = std::str::from_utf8(pair).map_err(|e| e.to_string())?;
                *b = u8::from_str_radix(s, 16).map_err(|e| format!("{name}: {e}"))?;
            }
            texels.push(f32::from_le_bytes(bytes));
        }
        out.push(HookTexture {
            name: name.trim().to_string(),
            width,
            height,
            texels,
        });
    }
    Ok(out)
}

/// `v` as an IEEE half, rounded to nearest even (overflow to infinity, NaN kept quiet).
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    // The 24-bit significand and how far it shifts down to the half's 10 bits (more for subnormals).
    let (sig, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        (man | 0x80_0000, (14 - e) as u32)
    } else {
        (man, 13u32)
    };
    let half = sig >> shift;
    let rest = sig & ((1 << shift) - 1);
    let mid = 1 << (shift - 1);
    let rounded = if rest > mid || (rest == mid && half & 1 == 1) {
        half + 1
    } else {
        half
    };
    if e <= 0 {
        // A carry out of the subnormal range lands on the smallest normal, which is right.
        sign | rounded as u16
    } else {
        // A carry out of the mantissa bumps the exponent, up to infinity, which is right too.
        sign | (((e as u32) << 10) + rounded) as u16
    }
}

/// The two weight tables as sampled images, and their staging copy.
pub(crate) struct RavuLut {
    /// `ravu_zoom_lut3` (45 x 2592) and `ravu_zoom_lut3_ar` (18 x 2592).
    images: [(vk::Image, vk::DeviceMemory, vk::ImageView, u32, u32); 2],
    staging: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    /// Byte offset of each table in `staging`.
    offsets: [u64; 2],
    /// The frame entry whose present last recorded the upload, until it is confirmed.
    pending: Option<usize>,
    ready: bool,
}

impl RavuLut {
    const NAMES: [&'static str; 2] = ["ravu_zoom_lut3", "ravu_zoom_lut3_ar"];

    /// Read the hook, make the images and fill the staging buffer. `memory_type(bits, bytes, host)`.
    pub(crate) unsafe fn create(
        device: &ash::Device,
        path: &str,
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        let mut textures = hook_textures(&text)?;
        let mut tables = Vec::with_capacity(2);
        for name in Self::NAMES {
            let i = textures
                .iter()
                .position(|t| t.name == name)
                .ok_or_else(|| format!("{path}: no //!TEXTURE {name}"))?;
            tables.push(textures.swap_remove(i));
        }
        let mut made = Self {
            images: [(
                vk::Image::null(),
                vk::DeviceMemory::null(),
                vk::ImageView::null(),
                0,
                0,
            ); 2],
            staging: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            offsets: [0; 2],
            pending: None,
            ready: false,
        };
        let r = (|| -> Result<(), vk::Result> {
            let format = vk::Format::R16G16B16A16_SFLOAT;
            for (slot, t) in made.images.iter_mut().zip(&tables) {
                slot.3 = t.width;
                slot.4 = t.height;
                slot.0 = device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width: t.width,
                            height: t.height,
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
                let mt = memory_type(req.memory_type_bits, req.size, false)
                    .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
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
                        .format(format)
                        .subresource_range(super::color_subresource_range()),
                    None,
                )?;
            }
            let mut bytes = Vec::new();
            for (k, t) in tables.iter().enumerate() {
                made.offsets[k] = bytes.len() as u64;
                for v in &t.texels {
                    bytes.extend_from_slice(&f16_bits(*v).to_le_bytes());
                }
                // Copy offsets must be multiples of the texel size (8).
                while bytes.len() % 16 != 0 {
                    bytes.push(0);
                }
            }
            made.staging = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes.len() as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )?;
            let req = device.get_buffer_memory_requirements(made.staging);
            let mt = memory_type(req.memory_type_bits, req.size, true)
                .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
            made.staging_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            )?;
            device.bind_buffer_memory(made.staging, made.staging_memory, 0)?;
            let p = device.map_memory(
                made.staging_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast::<u8>(), bytes.len());
            let flushed = device.flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                .memory(made.staging_memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)]);
            device.unmap_memory(made.staging_memory);
            flushed
        })();
        match r {
            Ok(()) => Ok(made),
            Err(e) => {
                made.destroy(device);
                Err(format!("vk={e:?}"))
            }
        }
    }

    /// The two views, for bindings 1 and 2.
    pub(crate) fn views(&self) -> [vk::ImageView; 2] {
        [self.images[0].2, self.images[1].2]
    }

    /// Record the upload into frame entry `ix`'s present unless one already ran; leaves both images in
    /// `SHADER_READ_ONLY_OPTIMAL` ordered before compute reads. Outside any render pass.
    pub(crate) unsafe fn record_upload(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        ix: usize,
    ) {
        if self.ready {
            return;
        }
        self.pending = Some(ix);
        let barrier = |image, old, new, src, dst| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(super::color_subresource_range())
        };
        // An earlier present's compute reads finish before the images are rewritten (write after read).
        let to_dst: Vec<_> = self
            .images
            .iter()
            .map(|i| {
                barrier(
                    i.0,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                )
            })
            .collect();
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_dst,
        );
        for (k, i) in self.images.iter().enumerate() {
            let region = [vk::BufferImageCopy::default()
                .buffer_offset(self.offsets[k])
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: i.3,
                    height: i.4,
                    depth: 1,
                })];
            device.cmd_copy_buffer_to_image(
                cmd,
                self.staging,
                i.0,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );
        }
        let to_read: Vec<_> = self
            .images
            .iter()
            .map(|i| {
                barrier(
                    i.0,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                )
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

    /// Frame entry `ix` came round (its fence retired): an upload it recorded has run.
    pub(crate) fn confirm(&mut self, ix: usize) {
        if self.pending == Some(ix) {
            self.pending = None;
            self.ready = true;
        }
    }

    /// The queue idled for a swapchain recreate: a pending upload may never have been submitted.
    pub(crate) fn unconfirm(&mut self) {
        self.pending = None;
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for (image, memory, view, _, _) in self.images {
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
        if self.staging != vk::Buffer::null() {
            device.destroy_buffer(self.staging, None);
        }
        if self.staging_memory != vk::DeviceMemory::null() {
            device.free_memory(self.staging_memory, None);
        }
    }
}

/// `shaders/present_ravu.comp`: bindings 0 the source, 1 and 2 the tables (combined image samplers), 3 `mid`
/// (rgba8 storage); push the source and output sizes (4 floats). One descriptor set per presenter frame entry.
pub(crate) struct RavuPass {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl RavuPass {
    /// Output pixels a side of one workgroup.
    const TILE: u32 = 16;

    pub(crate) unsafe fn create(device: &ash::Device, frames: usize) -> Result<Self, vk::Result> {
        let mut made = Self {
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
        };
        let r = (|| -> Result<(), vk::Result> {
            let bindings: Vec<_> = (0..4u32)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(if b == 3 {
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
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(16)];
            let set_layouts = [made.set_layout];
            made.layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )?;
            let module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(&super::present_sharp_shader::PRESENT_RAVU_COMP_SPIRV),
                None,
            )?;
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(c"main"),
                )
                .layout(made.layout)];
            let pipelines = device.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
            device.destroy_shader_module(module, None);
            made.pipeline = pipelines.map_err(|(_, e)| e)?[0];
            let n = frames as u32;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(3 * n),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(n),
            ];
            made.pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(n)
                    .pool_sizes(&sizes),
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
                made.destroy(device);
                Err(e)
            }
        }
    }

    /// Whether the pass can take a `src` frame shown at `out`: an upscale on both axes (the shader's shared tile
    /// holds the windows of at most 18 base texels a side, 16 output pixels at a scale <= 1.0625).
    pub(crate) fn takes(src: (u32, u32), out: (u32, u32)) -> bool {
        src.0 <= out.0 && src.1 <= out.1
    }

    /// Record the pass for frame entry `ix`: `src` (`SHADER_READ_ONLY_OPTIMAL`, `src_size`) and the tables into
    /// `mid` (`GENERAL`, `mid_size`). The caller orders the source and table reads before it and `mid`'s writes
    /// after it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        ix: usize,
        sampler: vk::Sampler,
        src: vk::ImageView,
        src_size: (u32, u32),
        tables: [vk::ImageView; 2],
        mid: vk::ImageView,
        mid_size: (u32, u32),
    ) {
        let set = self.sets[ix];
        let sampled = |v: vk::ImageView| {
            [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(v)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        };
        let infos = [sampled(src), sampled(tables[0]), sampled(tables[1])];
        let out_info = [vk::DescriptorImageInfo::default()
            .image_view(mid)
            .image_layout(vk::ImageLayout::GENERAL)];
        let mut writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(b, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            })
            .collect();
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_info),
        );
        device.update_descriptor_sets(&writes, &[]);
        let mut push = [0u8; 16];
        let sizes = [src_size.0 as f32, src_size.1 as f32, mid_size.0 as f32, mid_size.1 as f32];
        for (i, v) in sizes.iter().enumerate() {
            push[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[set], &[]);
        device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
        device.cmd_dispatch(cmd, mid_size.0.div_ceil(Self::TILE), mid_size.1.div_ceil(Self::TILE), 1);
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
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
        self.sets.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_bits_rounds_like_ieee() {
        assert_eq!(f16_bits(0.0), 0x0000);
        assert_eq!(f16_bits(-0.0), 0x8000);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(65520.0), 0x7c00);
        assert_eq!(f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f16_bits(f32::NAN) & 0x7e00, 0x7e00);
        // The smallest subnormal, and half of it (a tie, to even: zero).
        assert_eq!(f16_bits(5.960_464_5e-8), 0x0001);
        assert_eq!(f16_bits(2.980_232_2e-8), 0x0000);
        assert_eq!(f16_bits(6.097_555e-5), 0x03ff);
        assert_eq!(f16_bits(6.103_515_6e-5), 0x0400);
        // 1 + 2^-11 is a tie between 1 and 1 + 2^-10: to even.
        assert_eq!(f16_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f16_bits(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f16_bits(0.1), 0x2e66);
    }

    #[test]
    fn ravu_pass_takes_upscales_only() {
        assert!(RavuPass::takes((4096, 1728), (5120, 2160)));
        assert!(RavuPass::takes((5120, 2160), (5120, 2160)));
        assert!(!RavuPass::takes((5120, 2160), (4096, 1728)));
        assert!(!RavuPass::takes((4096, 2400), (5120, 2160)));
    }

    #[test]
    fn hook_textures_reads_f32_hex_blocks() {
        let text = "//!DESC x\n//!TEXTURE t1\n//!SIZE 1 1\n//!FORMAT rgba16f\n//!FILTER LINEAR\n\
                    0000803f000000400000000000000000\n//!TEXTURE t2\n//!SIZE 1 1\n//!FORMAT rgba16f\n\
                    000080bf000000000000000000000000\n";
        let t = hook_textures(text).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!((t[0].name.as_str(), t[0].width, t[0].height), ("t1", 1, 1));
        assert_eq!(t[0].texels, vec![1.0, 2.0, 0.0, 0.0]);
        assert_eq!(t[1].texels, vec![-1.0, 0.0, 0.0, 0.0]);
        assert!(hook_textures("//!TEXTURE t\n//!SIZE 2 1\n00\n").is_err());
    }
}
