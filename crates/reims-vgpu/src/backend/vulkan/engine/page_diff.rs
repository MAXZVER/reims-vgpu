//! Which 4 KiB pages of a framebuffer changed since the frame last written into
//! its guest pages — found on the GPU, so a writeback moves only those pages.
//!
//! # Why
//!
//! A presented framebuffer is written back into guest RAM at every DisplaySwap
//! (WindowServer copies out of it with the CPU). The whole frame crossed PCIe
//! and was landed by the CPU each time: 8 MB at 1920x1080 and 44 MB at
//! 5120x2160, ~7.5 ms a swap at 5K on the RTX 4060 lab host — the largest item
//! in the drain second. Most of a frame is usually the frame before it.
//!
//! # How
//!
//! Each framebuffer keeps a device-local copy of the bytes its guest pages were
//! last written from (`prev`). A payment copies the resident into a scratch
//! buffer (`cur`), and `shaders/page_diff.spvasm` compares the two a page per
//! workgroup, raising a flag per changed page and updating `prev` as it goes.
//! Only the flagged pages are then copied to host memory and landed. The first
//! payment of a framebuffer, or one after anything else wrote its pages (the
//! mapping's `guest_bytes_seq` moved), copies and lands the whole frame and
//! seeds `prev` from it.
//!
//! Lab A/B: `REIMS_VGPU_PAY_DIFF` (opt-in while measured).

use ash::vk;
use std::collections::HashMap;

use super::context::DeviceContext;
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::engine::DrawError;
use reims_vgpu_vulkan::memory::MemoryClass;

pub(crate) const PAGE_BYTES: u64 = 4096;
pub(crate) const WORDS_PER_PAGE: u32 = (PAGE_BYTES / 4) as u32;

/// `shaders/page_diff.spvasm`, assembled with `spirv-as --target-env vulkan1.0`
/// and checked with `spirv-val`.
pub(crate) const PAGE_DIFF_SPIRV: [u32; 369] = [
    0x07230203, 0x00010000, 0x00070000, 0x0000003d, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
    0x00000000, 0x00000001, 0x0007000f, 0x00000005, 0x00000001, 0x6e69616d, 0x00000000, 0x00000002,
    0x00000003, 0x00060010, 0x00000001, 0x00000011, 0x00000100, 0x00000001, 0x00000001, 0x00040047,
    0x00000002, 0x0000000b, 0x0000001a, 0x00040047, 0x00000003, 0x0000000b, 0x0000001b, 0x00040047,
    0x00000004, 0x00000006, 0x00000004, 0x00050048, 0x00000005, 0x00000000, 0x00000023, 0x00000000,
    0x00030047, 0x00000005, 0x00000003, 0x00040047, 0x00000006, 0x00000022, 0x00000000, 0x00040047,
    0x00000006, 0x00000021, 0x00000000, 0x00030047, 0x00000006, 0x00000018, 0x00040047, 0x00000007,
    0x00000022, 0x00000000, 0x00040047, 0x00000007, 0x00000021, 0x00000001, 0x00040047, 0x00000008,
    0x00000022, 0x00000000, 0x00040047, 0x00000008, 0x00000021, 0x00000002, 0x00030047, 0x00000008,
    0x00000019, 0x00050048, 0x00000009, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000009,
    0x00000001, 0x00000023, 0x00000004, 0x00030047, 0x00000009, 0x00000002, 0x00020013, 0x0000000a,
    0x00030021, 0x0000000b, 0x0000000a, 0x00040015, 0x0000000c, 0x00000020, 0x00000000, 0x00040015,
    0x0000000d, 0x00000020, 0x00000001, 0x00020014, 0x0000000e, 0x00040017, 0x0000000f, 0x0000000c,
    0x00000003, 0x00040020, 0x00000010, 0x00000001, 0x0000000f, 0x0004003b, 0x00000010, 0x00000002,
    0x00000001, 0x0004003b, 0x00000010, 0x00000003, 0x00000001, 0x0003001d, 0x00000004, 0x0000000c,
    0x0003001e, 0x00000005, 0x00000004, 0x00040020, 0x00000011, 0x00000002, 0x00000005, 0x0004003b,
    0x00000011, 0x00000006, 0x00000002, 0x0004003b, 0x00000011, 0x00000007, 0x00000002, 0x0004003b,
    0x00000011, 0x00000008, 0x00000002, 0x0004001e, 0x00000009, 0x0000000c, 0x0000000c, 0x00040020,
    0x00000012, 0x00000009, 0x00000009, 0x0004003b, 0x00000012, 0x00000013, 0x00000009, 0x00040020,
    0x00000014, 0x00000009, 0x0000000c, 0x00040020, 0x00000015, 0x00000002, 0x0000000c, 0x0004002b,
    0x0000000d, 0x00000016, 0x00000000, 0x0004002b, 0x0000000d, 0x00000017, 0x00000001, 0x0004002b,
    0x0000000c, 0x00000018, 0x00000001, 0x0004002b, 0x0000000c, 0x00000019, 0x00000100, 0x0003002a,
    0x0000000e, 0x0000001a, 0x00050036, 0x0000000a, 0x00000001, 0x00000000, 0x0000000b, 0x000200f8,
    0x0000001b, 0x0004003d, 0x0000000f, 0x0000001c, 0x00000002, 0x00050051, 0x0000000c, 0x0000001d,
    0x0000001c, 0x00000000, 0x0004003d, 0x0000000f, 0x0000001e, 0x00000003, 0x00050051, 0x0000000c,
    0x0000001f, 0x0000001e, 0x00000000, 0x00050041, 0x00000014, 0x00000020, 0x00000013, 0x00000016,
    0x0004003d, 0x0000000c, 0x00000021, 0x00000020, 0x00050041, 0x00000014, 0x00000022, 0x00000013,
    0x00000017, 0x0004003d, 0x0000000c, 0x00000023, 0x00000022, 0x000500ae, 0x0000000e, 0x00000024,
    0x0000001d, 0x00000021, 0x000300f7, 0x00000025, 0x00000000, 0x000400fa, 0x00000024, 0x00000025,
    0x00000026, 0x000200f8, 0x00000026, 0x00050084, 0x0000000c, 0x00000027, 0x0000001d, 0x00000023,
    0x000200f9, 0x00000028, 0x000200f8, 0x00000028, 0x000700f5, 0x0000000c, 0x00000029, 0x0000001f,
    0x00000026, 0x0000002a, 0x0000002b, 0x000700f5, 0x0000000e, 0x0000002c, 0x0000001a, 0x00000026,
    0x0000002d, 0x0000002b, 0x000400f6, 0x0000002e, 0x0000002b, 0x00000000, 0x000200f9, 0x0000002f,
    0x000200f8, 0x0000002f, 0x000500b0, 0x0000000e, 0x00000030, 0x00000029, 0x00000023, 0x000400fa,
    0x00000030, 0x00000031, 0x0000002e, 0x000200f8, 0x00000031, 0x00050080, 0x0000000c, 0x00000032,
    0x00000027, 0x00000029, 0x00060041, 0x00000015, 0x00000033, 0x00000006, 0x00000016, 0x00000032,
    0x0004003d, 0x0000000c, 0x00000034, 0x00000033, 0x00060041, 0x00000015, 0x00000035, 0x00000007,
    0x00000016, 0x00000032, 0x0004003d, 0x0000000c, 0x00000036, 0x00000035, 0x000500ab, 0x0000000e,
    0x00000037, 0x00000034, 0x00000036, 0x000300f7, 0x00000038, 0x00000000, 0x000400fa, 0x00000037,
    0x00000039, 0x00000038, 0x000200f8, 0x00000039, 0x0003003e, 0x00000035, 0x00000034, 0x000200f9,
    0x00000038, 0x000200f8, 0x00000038, 0x000500a6, 0x0000000e, 0x0000002d, 0x0000002c, 0x00000037,
    0x000200f9, 0x0000002b, 0x000200f8, 0x0000002b, 0x00050080, 0x0000000c, 0x0000002a, 0x00000029,
    0x00000019, 0x000200f9, 0x00000028, 0x000200f8, 0x0000002e, 0x000300f7, 0x0000003a, 0x00000000,
    0x000400fa, 0x0000002c, 0x0000003b, 0x0000003a, 0x000200f8, 0x0000003b, 0x00060041, 0x00000015,
    0x0000003c, 0x00000008, 0x00000016, 0x0000001d, 0x0003003e, 0x0000003c, 0x00000018, 0x000200f9,
    0x0000003a, 0x000200f8, 0x0000003a, 0x000200f9, 0x00000025, 0x000200f8, 0x00000025, 0x000100fd,
    0x00010038,
];

/// A buffer this module owns outright: its own `vkAllocateMemory`, alive across
/// payments. The pools recycle their slots at each fence, and a shadow has to
/// hold a frame from one swap to the next.
pub(crate) struct OwnedBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: u64,
    /// Host address of a mapped (host-visible) buffer, else null.
    pub mapped: *mut u8,
    coherent: bool,
}

// SAFETY: the handles and the mapping are only used under the engine lock.
unsafe impl Send for OwnedBuffer {}

impl OwnedBuffer {
    unsafe fn create(
        ctx: &DeviceContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        class: MemoryClass,
    ) -> Result<Self, DrawError> {
        let device = &ctx.device;
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateGuestGather, e)))?;
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let Some(index) = ctx.memory_type_for(req.memory_type_bits, req.size, class) else {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(DrawError::VkCall(VkCall::new(
                VkOp::PoolsBindGuestGather,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            )));
        };
        let memory = match unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index),
                None,
            )
        } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsBindGuestGather,
                    e,
                )));
            }
        };
        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(DrawError::VkCall(VkCall::new(
                VkOp::PoolsBindGuestGather,
                e,
            )));
        }
        let flags = ctx.memory_properties.memory_types[index as usize].property_flags;
        let (mapped, coherent) = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            match unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(p) => (
                    p.cast::<u8>(),
                    flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
                ),
                Err(e) => {
                    unsafe {
                        device.free_memory(memory, None);
                        device.destroy_buffer(buffer, None);
                    }
                    return Err(DrawError::VkCall(VkCall::new(VkOp::ReadbackMap, e)));
                }
            }
        } else {
            (std::ptr::null_mut(), true)
        };
        Ok(Self {
            buffer,
            memory,
            size,
            mapped,
            coherent,
        })
    }

    /// Make the device's writes visible to the host, for a non-coherent type.
    pub(crate) unsafe fn invalidate(&self, device: &ash::Device) -> Result<(), DrawError> {
        if self.coherent || self.mapped.is_null() {
            return Ok(());
        }
        let range = [vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        unsafe { device.invalidate_mapped_memory_ranges(&range) }
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackInvalidate, e)))
    }

    unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            if !self.mapped.is_null() {
                device.unmap_memory(self.memory);
            }
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// The compare kernel, its layout and the one descriptor set it is dispatched
/// with: payments run one at a time under the engine lock and wait for their
/// fence, so a single set is never in use twice.
pub(crate) struct DiffPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub set: vk::DescriptorSet,
}

impl DiffPipeline {
    unsafe fn create(ctx: &DeviceContext) -> Result<Self, DrawError> {
        let device = &ctx.device;
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&PAGE_DIFF_SPIRV),
                None,
            )
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ScatterCreateShaderModule, e)))?;
        let bindings = [0u32, 1, 2].map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let dsl = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ScatterCreateSetLayout, e)))?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        let set_layouts = [dsl];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ScatterCreatePipelineLayout, e)))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline = unsafe { device.create_compute_pipelines(ctx.pipeline_cache, &info, None) }
            .map_err(|(_, e)| DrawError::VkCall(VkCall::new(VkOp::ScatterCreatePipeline, e)))?[0];
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let dpool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ScatterCreateSetLayout, e)))?;
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(dpool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ScatterCreateSetLayout, e)))?[0];
        // The module, layout and pool stay alive for the process with the
        // pipeline; nothing here is torn down before the device is.
        Ok(Self {
            layout,
            pipeline,
            set,
        })
    }

    /// Point the set at this payment's three buffers.
    pub(crate) unsafe fn write_set(
        &self,
        device: &ash::Device,
        cur: &OwnedBuffer,
        prev: &OwnedBuffer,
        flags: &OwnedBuffer,
        frame: u64,
        pages: u64,
    ) {
        let infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(cur.buffer)
                .offset(0)
                .range(frame),
            vk::DescriptorBufferInfo::default()
                .buffer(prev.buffer)
                .offset(0)
                .range(frame),
            vk::DescriptorBufferInfo::default()
                .buffer(flags.buffer)
                .offset(0)
                .range(pages * 4),
        ];
        let writes = [0usize, 1, 2].map(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[i]))
        });
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}

/// One framebuffer's copy of the bytes its guest pages were last written from.
pub(crate) struct Shadow {
    /// `(map_generation, width, height, guest_bytes_seq after that write)`.
    pub key: (u32, u32, u32, u64),
    pub prev: OwnedBuffer,
}

#[derive(Default)]
pub(crate) struct PageDiffState {
    /// The `VkDevice` these objects belong to; a different one (device lost and
    /// recreated) drops everything rather than use dead handles.
    pub device: u64,
    pub pipeline: Option<DiffPipeline>,
    pub refused: bool,
    pub shadows: HashMap<u32, Shadow>,
    pub cur: Option<OwnedBuffer>,
    pub flags: Option<OwnedBuffer>,
    pub out: Option<OwnedBuffer>,
}

pub(crate) static STATE: std::sync::Mutex<Option<PageDiffState>> = std::sync::Mutex::new(None);

impl PageDiffState {
    /// The compare kernel, created on first use; `None` after a refusal.
    pub(crate) unsafe fn ensure_pipeline(&mut self, ctx: &DeviceContext) -> bool {
        if self.pipeline.is_none() && !self.refused {
            match unsafe { DiffPipeline::create(ctx) } {
                Ok(p) => self.pipeline = Some(p),
                Err(e) => {
                    self.refused = true;
                    crate::observe::Emit::decline("page_diff_pipeline", &e).fail();
                }
            }
        }
        self.pipeline.is_some()
    }

    /// A buffer in `slot` of at least `size` bytes, replacing a smaller one.
    pub(crate) unsafe fn ensure(
        ctx: &DeviceContext,
        slot: &mut Option<OwnedBuffer>,
        size: u64,
        usage: vk::BufferUsageFlags,
        class: MemoryClass,
    ) -> Result<(), DrawError> {
        if slot.as_ref().is_some_and(|b| b.size >= size) {
            return Ok(());
        }
        if let Some(old) = slot.take() {
            unsafe { old.destroy(&ctx.device) };
        }
        *slot = Some(unsafe { OwnedBuffer::create(ctx, size, usage, class)? });
        Ok(())
    }

    /// A shadow for `mapping_id` of at least `size` bytes; `true` when it is new
    /// or its key moved, i.e. it holds nothing the guest pages match.
    pub(crate) unsafe fn ensure_shadow(
        &mut self,
        ctx: &DeviceContext,
        mapping_id: u32,
        key: (u32, u32, u32, u64),
        size: u64,
    ) -> Result<bool, DrawError> {
        if let Some(s) = self.shadows.get(&mapping_id) {
            if s.key == key && s.prev.size >= size {
                return Ok(false);
            }
        }
        if let Some(old) = self.shadows.remove(&mapping_id) {
            unsafe { old.prev.destroy(&ctx.device) };
        }
        let prev = unsafe {
            OwnedBuffer::create(
                ctx,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryClass::DeviceLocal,
            )?
        };
        self.shadows.insert(mapping_id, Shadow { key, prev });
        Ok(true)
    }
}
