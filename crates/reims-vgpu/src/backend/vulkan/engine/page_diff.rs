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
//! Default on; `REIMS_VGPU_PAY_DIFF=off` pays whole frames instead.

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

/// `shaders/page_diff_one_trip.spvasm` (assembled the same way): the compare of
/// [`PAGE_DIFF_SPIRV`] and, for every page it flags or the host pre-marked in
/// `flags`, the copy of that page into `out` — one submission where the two-trip
/// form needed a second one to fetch the pages the first one named.
pub(crate) const PAGE_DIFF_ONE_TRIP_SPIRV: [u32; 556] = [
    0x07230203, 0x00010000, 0x00070000, 0x0000005c, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
    0x00000000, 0x00000001, 0x0007000f, 0x00000005, 0x00000001, 0x6e69616d, 0x00000000, 0x00000002,
    0x00000003, 0x00060010, 0x00000001, 0x00000011, 0x00000100, 0x00000001, 0x00000001, 0x00040047,
    0x00000002, 0x0000000b, 0x0000001a, 0x00040047, 0x00000003, 0x0000000b, 0x0000001b, 0x00040047,
    0x00000004, 0x00000006, 0x00000004, 0x00050048, 0x00000005, 0x00000000, 0x00000023, 0x00000000,
    0x00030047, 0x00000005, 0x00000003, 0x00040047, 0x00000006, 0x00000022, 0x00000000, 0x00040047,
    0x00000006, 0x00000021, 0x00000000, 0x00030047, 0x00000006, 0x00000018, 0x00040047, 0x00000007,
    0x00000022, 0x00000000, 0x00040047, 0x00000007, 0x00000021, 0x00000001, 0x00040047, 0x00000008,
    0x00000022, 0x00000000, 0x00040047, 0x00000008, 0x00000021, 0x00000002, 0x00040047, 0x00000009,
    0x00000022, 0x00000000, 0x00040047, 0x00000009, 0x00000021, 0x00000003, 0x00030047, 0x00000009,
    0x00000019, 0x00050048, 0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000a,
    0x00000001, 0x00000023, 0x00000004, 0x00030047, 0x0000000a, 0x00000002, 0x00020013, 0x0000000b,
    0x00030021, 0x0000000c, 0x0000000b, 0x00040015, 0x0000000d, 0x00000020, 0x00000000, 0x00040015,
    0x0000000e, 0x00000020, 0x00000001, 0x00020014, 0x0000000f, 0x00040017, 0x00000010, 0x0000000d,
    0x00000003, 0x00040020, 0x00000011, 0x00000001, 0x00000010, 0x0004003b, 0x00000011, 0x00000002,
    0x00000001, 0x0004003b, 0x00000011, 0x00000003, 0x00000001, 0x0003001d, 0x00000004, 0x0000000d,
    0x0003001e, 0x00000005, 0x00000004, 0x00040020, 0x00000012, 0x00000002, 0x00000005, 0x0004003b,
    0x00000012, 0x00000006, 0x00000002, 0x0004003b, 0x00000012, 0x00000007, 0x00000002, 0x0004003b,
    0x00000012, 0x00000008, 0x00000002, 0x0004003b, 0x00000012, 0x00000009, 0x00000002, 0x0004001e,
    0x0000000a, 0x0000000d, 0x0000000d, 0x00040020, 0x00000013, 0x00000009, 0x0000000a, 0x0004003b,
    0x00000013, 0x00000014, 0x00000009, 0x00040020, 0x00000015, 0x00000009, 0x0000000d, 0x00040020,
    0x00000016, 0x00000002, 0x0000000d, 0x00040020, 0x00000017, 0x00000004, 0x0000000d, 0x0004003b,
    0x00000017, 0x00000018, 0x00000004, 0x0004002b, 0x0000000e, 0x00000019, 0x00000000, 0x0004002b,
    0x0000000e, 0x0000001a, 0x00000001, 0x0004002b, 0x0000000d, 0x0000001b, 0x00000000, 0x0004002b,
    0x0000000d, 0x0000001c, 0x00000001, 0x0004002b, 0x0000000d, 0x0000001d, 0x00000002, 0x0004002b,
    0x0000000d, 0x0000001e, 0x00000100, 0x0004002b, 0x0000000d, 0x0000001f, 0x00000108, 0x0003002a,
    0x0000000f, 0x00000020, 0x00050036, 0x0000000b, 0x00000001, 0x00000000, 0x0000000c, 0x000200f8,
    0x00000021, 0x0004003d, 0x00000010, 0x00000022, 0x00000002, 0x00050051, 0x0000000d, 0x00000023,
    0x00000022, 0x00000000, 0x0004003d, 0x00000010, 0x00000024, 0x00000003, 0x00050051, 0x0000000d,
    0x00000025, 0x00000024, 0x00000000, 0x00050041, 0x00000015, 0x00000026, 0x00000014, 0x00000019,
    0x0004003d, 0x0000000d, 0x00000027, 0x00000026, 0x00050041, 0x00000015, 0x00000028, 0x00000014,
    0x0000001a, 0x0004003d, 0x0000000d, 0x00000029, 0x00000028, 0x000500ae, 0x0000000f, 0x0000002a,
    0x00000023, 0x00000027, 0x000300f7, 0x0000002b, 0x00000000, 0x000400fa, 0x0000002a, 0x0000002b,
    0x0000002c, 0x000200f8, 0x0000002c, 0x00050084, 0x0000000d, 0x0000002d, 0x00000023, 0x00000029,
    0x000500aa, 0x0000000f, 0x0000002e, 0x00000025, 0x0000001b, 0x000300f7, 0x0000002f, 0x00000000,
    0x000400fa, 0x0000002e, 0x00000030, 0x0000002f, 0x000200f8, 0x00000030, 0x00060041, 0x00000016,
    0x00000031, 0x00000008, 0x00000019, 0x00000023, 0x0004003d, 0x0000000d, 0x00000032, 0x00000031,
    0x000500ab, 0x0000000f, 0x00000033, 0x00000032, 0x0000001b, 0x000600a9, 0x0000000d, 0x00000034,
    0x00000033, 0x0000001c, 0x0000001b, 0x0003003e, 0x00000018, 0x00000034, 0x000200f9, 0x0000002f,
    0x000200f8, 0x0000002f, 0x000400e0, 0x0000001d, 0x0000001d, 0x0000001f, 0x000200f9, 0x00000035,
    0x000200f8, 0x00000035, 0x000700f5, 0x0000000d, 0x00000036, 0x00000025, 0x0000002f, 0x00000037,
    0x00000038, 0x000700f5, 0x0000000f, 0x00000039, 0x00000020, 0x0000002f, 0x0000003a, 0x00000038,
    0x000400f6, 0x0000003b, 0x00000038, 0x00000000, 0x000200f9, 0x0000003c, 0x000200f8, 0x0000003c,
    0x000500b0, 0x0000000f, 0x0000003d, 0x00000036, 0x00000029, 0x000400fa, 0x0000003d, 0x0000003e,
    0x0000003b, 0x000200f8, 0x0000003e, 0x00050080, 0x0000000d, 0x0000003f, 0x0000002d, 0x00000036,
    0x00060041, 0x00000016, 0x00000040, 0x00000006, 0x00000019, 0x0000003f, 0x0004003d, 0x0000000d,
    0x00000041, 0x00000040, 0x00060041, 0x00000016, 0x00000042, 0x00000007, 0x00000019, 0x0000003f,
    0x0004003d, 0x0000000d, 0x00000043, 0x00000042, 0x000500ab, 0x0000000f, 0x00000044, 0x00000041,
    0x00000043, 0x000300f7, 0x00000045, 0x00000000, 0x000400fa, 0x00000044, 0x00000046, 0x00000045,
    0x000200f8, 0x00000046, 0x0003003e, 0x00000042, 0x00000041, 0x000200f9, 0x00000045, 0x000200f8,
    0x00000045, 0x000500a6, 0x0000000f, 0x0000003a, 0x00000039, 0x00000044, 0x000200f9, 0x00000038,
    0x000200f8, 0x00000038, 0x00050080, 0x0000000d, 0x00000037, 0x00000036, 0x0000001e, 0x000200f9,
    0x00000035, 0x000200f8, 0x0000003b, 0x000300f7, 0x00000047, 0x00000000, 0x000400fa, 0x00000039,
    0x00000048, 0x00000047, 0x000200f8, 0x00000048, 0x0003003e, 0x00000018, 0x0000001c, 0x000200f9,
    0x00000047, 0x000200f8, 0x00000047, 0x000400e0, 0x0000001d, 0x0000001d, 0x0000001f, 0x0004003d,
    0x0000000d, 0x00000049, 0x00000018, 0x000500ab, 0x0000000f, 0x0000004a, 0x00000049, 0x0000001b,
    0x000300f7, 0x0000004b, 0x00000000, 0x000400fa, 0x0000004a, 0x0000004c, 0x0000004b, 0x000200f8,
    0x0000004c, 0x000200f9, 0x0000004d, 0x000200f8, 0x0000004d, 0x000700f5, 0x0000000d, 0x0000004e,
    0x00000025, 0x0000004c, 0x0000004f, 0x00000050, 0x000400f6, 0x00000051, 0x00000050, 0x00000000,
    0x000200f9, 0x00000052, 0x000200f8, 0x00000052, 0x000500b0, 0x0000000f, 0x00000053, 0x0000004e,
    0x00000029, 0x000400fa, 0x00000053, 0x00000054, 0x00000051, 0x000200f8, 0x00000054, 0x00050080,
    0x0000000d, 0x00000055, 0x0000002d, 0x0000004e, 0x00060041, 0x00000016, 0x00000056, 0x00000006,
    0x00000019, 0x00000055, 0x0004003d, 0x0000000d, 0x00000057, 0x00000056, 0x00060041, 0x00000016,
    0x00000058, 0x00000009, 0x00000019, 0x00000055, 0x0003003e, 0x00000058, 0x00000057, 0x000200f9,
    0x00000050, 0x000200f8, 0x00000050, 0x00050080, 0x0000000d, 0x0000004f, 0x0000004e, 0x0000001e,
    0x000200f9, 0x0000004d, 0x000200f8, 0x00000051, 0x000300f7, 0x00000059, 0x00000000, 0x000400fa,
    0x0000002e, 0x0000005a, 0x00000059, 0x000200f8, 0x0000005a, 0x00060041, 0x00000016, 0x0000005b,
    0x00000008, 0x00000019, 0x00000023, 0x0003003e, 0x0000005b, 0x0000001c, 0x000200f9, 0x00000059,
    0x000200f8, 0x00000059, 0x000200f9, 0x0000004b, 0x000200f8, 0x0000004b, 0x000200f9, 0x0000002b,
    0x000200f8, 0x0000002b, 0x000100fd, 0x00010038,
];

/// `shaders/page_diff_image.spvasm` (assembled the same way): the one-trip
/// compare reading the frame from the resident image through a sampled view
/// instead of from a buffer copy of it, for a packed BGRA8 UNORM resident. The
/// copy wrote the frame once and the compare read it back once more.
pub(crate) const PAGE_DIFF_IMAGE_SPIRV: [u32; 703] = [
    0x07230203, 0x00010000, 0x00070000, 0x00000076, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000005, 0x00000002, 0x6e69616d, 0x00000000, 0x00000003, 0x00000004, 0x00060010,
    0x00000002, 0x00000011, 0x00000100, 0x00000001, 0x00000001, 0x00040047, 0x00000003, 0x0000000b,
    0x0000001a, 0x00040047, 0x00000004, 0x0000000b, 0x0000001b, 0x00040047, 0x00000005, 0x00000006,
    0x00000004, 0x00050048, 0x00000006, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000006,
    0x00000003, 0x00040047, 0x00000007, 0x00000022, 0x00000000, 0x00040047, 0x00000007, 0x00000021,
    0x00000000, 0x00040047, 0x00000008, 0x00000022, 0x00000000, 0x00040047, 0x00000008, 0x00000021,
    0x00000001, 0x00040047, 0x00000009, 0x00000022, 0x00000000, 0x00040047, 0x00000009, 0x00000021,
    0x00000002, 0x00040047, 0x0000000a, 0x00000022, 0x00000000, 0x00040047, 0x0000000a, 0x00000021,
    0x00000003, 0x00030047, 0x0000000a, 0x00000019, 0x00050048, 0x0000000b, 0x00000000, 0x00000023,
    0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x0000000b,
    0x00000002, 0x00000023, 0x00000008, 0x00050048, 0x0000000b, 0x00000003, 0x00000023, 0x0000000c,
    0x00030047, 0x0000000b, 0x00000002, 0x00020013, 0x0000000c, 0x00030021, 0x0000000d, 0x0000000c,
    0x00040015, 0x0000000e, 0x00000020, 0x00000000, 0x00040015, 0x0000000f, 0x00000020, 0x00000001,
    0x00030016, 0x00000010, 0x00000020, 0x00020014, 0x00000011, 0x00040017, 0x00000012, 0x0000000e,
    0x00000003, 0x00040017, 0x00000013, 0x0000000f, 0x00000002, 0x00040017, 0x00000014, 0x00000010,
    0x00000004, 0x00040020, 0x00000015, 0x00000001, 0x00000012, 0x0004003b, 0x00000015, 0x00000003,
    0x00000001, 0x0004003b, 0x00000015, 0x00000004, 0x00000001, 0x00090019, 0x00000016, 0x00000010,
    0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00040020, 0x00000017,
    0x00000000, 0x00000016, 0x0004003b, 0x00000017, 0x00000007, 0x00000000, 0x0003001d, 0x00000005,
    0x0000000e, 0x0003001e, 0x00000006, 0x00000005, 0x00040020, 0x00000018, 0x00000002, 0x00000006,
    0x0004003b, 0x00000018, 0x00000008, 0x00000002, 0x0004003b, 0x00000018, 0x00000009, 0x00000002,
    0x0004003b, 0x00000018, 0x0000000a, 0x00000002, 0x0006001e, 0x0000000b, 0x0000000e, 0x0000000e,
    0x0000000e, 0x0000000e, 0x00040020, 0x00000019, 0x00000009, 0x0000000b, 0x0004003b, 0x00000019,
    0x0000001a, 0x00000009, 0x00040020, 0x0000001b, 0x00000009, 0x0000000e, 0x00040020, 0x0000001c,
    0x00000002, 0x0000000e, 0x00040020, 0x0000001d, 0x00000004, 0x0000000e, 0x0004003b, 0x0000001d,
    0x0000001e, 0x00000004, 0x0004002b, 0x0000000f, 0x0000001f, 0x00000000, 0x0004002b, 0x0000000f,
    0x00000020, 0x00000001, 0x0004002b, 0x0000000f, 0x00000021, 0x00000002, 0x0004002b, 0x0000000f,
    0x00000022, 0x00000003, 0x0004002b, 0x0000000e, 0x00000023, 0x00000000, 0x0004002b, 0x0000000e,
    0x00000024, 0x00000001, 0x0004002b, 0x0000000e, 0x00000025, 0x00000002, 0x0004002b, 0x0000000e,
    0x00000026, 0x00000100, 0x0004002b, 0x0000000e, 0x00000027, 0x00000108, 0x0003002a, 0x00000011,
    0x00000028, 0x00050036, 0x0000000c, 0x00000002, 0x00000000, 0x0000000d, 0x000200f8, 0x00000029,
    0x0004003d, 0x00000012, 0x0000002a, 0x00000003, 0x00050051, 0x0000000e, 0x0000002b, 0x0000002a,
    0x00000000, 0x0004003d, 0x00000012, 0x0000002c, 0x00000004, 0x00050051, 0x0000000e, 0x0000002d,
    0x0000002c, 0x00000000, 0x00050041, 0x0000001b, 0x0000002e, 0x0000001a, 0x0000001f, 0x0004003d,
    0x0000000e, 0x0000002f, 0x0000002e, 0x00050041, 0x0000001b, 0x00000030, 0x0000001a, 0x00000020,
    0x0004003d, 0x0000000e, 0x00000031, 0x00000030, 0x00050041, 0x0000001b, 0x00000032, 0x0000001a,
    0x00000021, 0x0004003d, 0x0000000e, 0x00000033, 0x00000032, 0x00050041, 0x0000001b, 0x00000034,
    0x0000001a, 0x00000022, 0x0004003d, 0x0000000e, 0x00000035, 0x00000034, 0x000500ae, 0x00000011,
    0x00000036, 0x0000002b, 0x0000002f, 0x000300f7, 0x00000037, 0x00000000, 0x000400fa, 0x00000036,
    0x00000037, 0x00000038, 0x000200f8, 0x00000038, 0x00050084, 0x0000000e, 0x00000039, 0x0000002b,
    0x00000031, 0x000500aa, 0x00000011, 0x0000003a, 0x0000002d, 0x00000023, 0x000300f7, 0x0000003b,
    0x00000000, 0x000400fa, 0x0000003a, 0x0000003c, 0x0000003b, 0x000200f8, 0x0000003c, 0x00060041,
    0x0000001c, 0x0000003d, 0x00000009, 0x0000001f, 0x0000002b, 0x0004003d, 0x0000000e, 0x0000003e,
    0x0000003d, 0x000500ab, 0x00000011, 0x0000003f, 0x0000003e, 0x00000023, 0x000600a9, 0x0000000e,
    0x00000040, 0x0000003f, 0x00000024, 0x00000023, 0x0003003e, 0x0000001e, 0x00000040, 0x000200f9,
    0x0000003b, 0x000200f8, 0x0000003b, 0x000400e0, 0x00000025, 0x00000025, 0x00000027, 0x000200f9,
    0x00000041, 0x000200f8, 0x00000041, 0x000700f5, 0x0000000e, 0x00000042, 0x0000002d, 0x0000003b,
    0x00000043, 0x00000044, 0x000700f5, 0x00000011, 0x00000045, 0x00000028, 0x0000003b, 0x00000046,
    0x00000044, 0x000400f6, 0x00000047, 0x00000044, 0x00000000, 0x000200f9, 0x00000048, 0x000200f8,
    0x00000048, 0x000500b0, 0x00000011, 0x00000049, 0x00000042, 0x00000031, 0x000400fa, 0x00000049,
    0x0000004a, 0x00000047, 0x000200f8, 0x0000004a, 0x00050080, 0x0000000e, 0x0000004b, 0x00000039,
    0x00000042, 0x000500b0, 0x00000011, 0x0000004c, 0x0000004b, 0x00000035, 0x000300f7, 0x0000004d,
    0x00000000, 0x000400fa, 0x0000004c, 0x0000004e, 0x0000004d, 0x000200f8, 0x0000004e, 0x00050089,
    0x0000000e, 0x0000004f, 0x0000004b, 0x00000033, 0x00050086, 0x0000000e, 0x00000050, 0x0000004b,
    0x00000033, 0x0004007c, 0x0000000f, 0x00000051, 0x0000004f, 0x0004007c, 0x0000000f, 0x00000052,
    0x00000050, 0x00050050, 0x00000013, 0x00000053, 0x00000051, 0x00000052, 0x0004003d, 0x00000016,
    0x00000054, 0x00000007, 0x0007005f, 0x00000014, 0x00000055, 0x00000054, 0x00000053, 0x00000002,
    0x0000001f, 0x0009004f, 0x00000014, 0x00000056, 0x00000055, 0x00000055, 0x00000002, 0x00000001,
    0x00000000, 0x00000003, 0x0006000c, 0x0000000e, 0x00000057, 0x00000001, 0x00000037, 0x00000056,
    0x00060041, 0x0000001c, 0x00000058, 0x00000008, 0x0000001f, 0x0000004b, 0x0004003d, 0x0000000e,
    0x00000059, 0x00000058, 0x000500ab, 0x00000011, 0x0000005a, 0x00000057, 0x00000059, 0x000300f7,
    0x0000005b, 0x00000000, 0x000400fa, 0x0000005a, 0x0000005c, 0x0000005b, 0x000200f8, 0x0000005c,
    0x0003003e, 0x00000058, 0x00000057, 0x000200f9, 0x0000005b, 0x000200f8, 0x0000005b, 0x000200f9,
    0x0000004d, 0x000200f8, 0x0000004d, 0x000700f5, 0x00000011, 0x0000005d, 0x00000028, 0x0000004a,
    0x0000005a, 0x0000005b, 0x000500a6, 0x00000011, 0x00000046, 0x00000045, 0x0000005d, 0x000200f9,
    0x00000044, 0x000200f8, 0x00000044, 0x00050080, 0x0000000e, 0x00000043, 0x00000042, 0x00000026,
    0x000200f9, 0x00000041, 0x000200f8, 0x00000047, 0x000300f7, 0x0000005e, 0x00000000, 0x000400fa,
    0x00000045, 0x0000005f, 0x0000005e, 0x000200f8, 0x0000005f, 0x0003003e, 0x0000001e, 0x00000024,
    0x000200f9, 0x0000005e, 0x000200f8, 0x0000005e, 0x000400e0, 0x00000025, 0x00000025, 0x00000027,
    0x0004003d, 0x0000000e, 0x00000060, 0x0000001e, 0x000500ab, 0x00000011, 0x00000061, 0x00000060,
    0x00000023, 0x000300f7, 0x00000062, 0x00000000, 0x000400fa, 0x00000061, 0x00000063, 0x00000062,
    0x000200f8, 0x00000063, 0x000200f9, 0x00000064, 0x000200f8, 0x00000064, 0x000700f5, 0x0000000e,
    0x00000065, 0x0000002d, 0x00000063, 0x00000066, 0x00000067, 0x000400f6, 0x00000068, 0x00000067,
    0x00000000, 0x000200f9, 0x00000069, 0x000200f8, 0x00000069, 0x000500b0, 0x00000011, 0x0000006a,
    0x00000065, 0x00000031, 0x000400fa, 0x0000006a, 0x0000006b, 0x00000068, 0x000200f8, 0x0000006b,
    0x00050080, 0x0000000e, 0x0000006c, 0x00000039, 0x00000065, 0x000500b0, 0x00000011, 0x0000006d,
    0x0000006c, 0x00000035, 0x000300f7, 0x0000006e, 0x00000000, 0x000400fa, 0x0000006d, 0x0000006f,
    0x0000006e, 0x000200f8, 0x0000006f, 0x00060041, 0x0000001c, 0x00000070, 0x00000008, 0x0000001f,
    0x0000006c, 0x0004003d, 0x0000000e, 0x00000071, 0x00000070, 0x00060041, 0x0000001c, 0x00000072,
    0x0000000a, 0x0000001f, 0x0000006c, 0x0003003e, 0x00000072, 0x00000071, 0x000200f9, 0x0000006e,
    0x000200f8, 0x0000006e, 0x000200f9, 0x00000067, 0x000200f8, 0x00000067, 0x00050080, 0x0000000e,
    0x00000066, 0x00000065, 0x00000026, 0x000200f9, 0x00000064, 0x000200f8, 0x00000068, 0x000300f7,
    0x00000073, 0x00000000, 0x000400fa, 0x0000003a, 0x00000074, 0x00000073, 0x000200f8, 0x00000074,
    0x00060041, 0x0000001c, 0x00000075, 0x00000009, 0x0000001f, 0x0000002b, 0x0003003e, 0x00000075,
    0x00000024, 0x000200f9, 0x00000073, 0x000200f8, 0x00000073, 0x000200f9, 0x00000062, 0x000200f8,
    0x00000062, 0x000200f9, 0x00000037, 0x000200f8, 0x00000037, 0x000100fd, 0x00010038,
];

/// TEMP (lab): `REIMS_VGPU_LAB_PAY_IMAGE=off` takes the frame copy again (the
/// two-trip or one-trip compare over a buffer). On by default. Latched.
pub(crate) fn image_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_PAY_IMAGE").map_or(true, |v| v != "off");
        crate::observe::off(format!("pay_diff_image on={on}"));
        on
    })
}

/// TEMP (lab): `REIMS_VGPU_LAB_PAY_ONE_TRIP=on` answers a page-diff payment with
/// one submission and one wait instead of two. Latched for the process.
pub(crate) fn one_trip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_PAY_ONE_TRIP").is_ok_and(|v| v == "on");
        crate::observe::off(format!("pay_diff_one_trip on={on}"));
        on
    })
}

/// TEMP (lab): `REIMS_VGPU_LAB_PAY_ASYNC=off` waits for the queue owner's reply
/// that `vkQueueSubmit` returned before waiting the payment's own fence, as the
/// sync readback submit does. On by default: the fence is the completion either
/// way, and the reply only puts a second thread wake-up on the drain's path.
/// Latched.
pub(crate) fn lab_async_submit_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_PAY_ASYNC").map_or(true, |v| v != "off");
        crate::observe::off(format!("pay_diff_async_submit on={on}"));
        on
    })
}

/// TEMP (lab): `REIMS_VGPU_LAB_PAY_SPIN=<us>` polls the payment's fence for up
/// to that long before blocking in `vkWaitForFences`; `=off` (or `0`) blocks at
/// once, as before. Default 1500 us: the drain thread has nothing else to do
/// until the answer lands, and a blocked wait adds the OS wake-up to every
/// payment. The answer is the same fence either way. Latched.
pub(crate) fn lab_spin_budget_us() -> u64 {
    static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *US.get_or_init(|| {
        let us = parse_spin_budget(std::env::var("REIMS_VGPU_LAB_PAY_SPIN").ok().as_deref());
        crate::observe::off(format!("pay_diff_spin budget_us={us}"));
        us
    })
}

/// The spin budget a `REIMS_VGPU_LAB_PAY_SPIN` value asks for: unset is the
/// default, `off` or anything unparsable-as-a-count is no spin, and a count is
/// clamped to 10 ms so a typo cannot pin the drain thread for seconds.
fn parse_spin_budget(raw: Option<&str>) -> u64 {
    const DEFAULT_US: u64 = 1500;
    const CEILING_US: u64 = 10_000;
    match raw.map(str::trim) {
        None => DEFAULT_US,
        Some(v) => v.parse::<u64>().map_or(0, |n| n.min(CEILING_US)),
    }
}

#[cfg(test)]
mod lab_spin_budget_tests {
    use super::parse_spin_budget;

    #[test]
    fn unset_spins_the_default_off_never_and_a_count_is_clamped() {
        assert_eq!(parse_spin_budget(None), 1500);
        assert_eq!(parse_spin_budget(Some("off")), 0);
        assert_eq!(parse_spin_budget(Some("0")), 0);
        assert_eq!(parse_spin_budget(Some(" 700 ")), 700);
        assert_eq!(parse_spin_budget(Some("999999")), 10_000);
        assert_eq!(parse_spin_budget(Some("garbage")), 0);
    }
}

/// TEMP (lab): `REIMS_VGPU_LAB_PAY_FLAGS_DEV=off` binds the image kernel to the
/// host-memory `flags` buffer directly, as before. On by default: every
/// workgroup's first act is to read its page's flag and wait for it at a
/// barrier before any texel is fetched, and in host memory that read crosses
/// PCIe once per page (14 400 at 5K). The kernel now reads and writes a
/// device-local copy instead, filled from `flags` by one copy before it and
/// copied back by one after; the host writes and reads `flags` exactly as
/// before, so the values each side sees are the same. Latched.
pub(crate) fn lab_flags_dev_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_PAY_FLAGS_DEV").map_or(true, |v| v != "off");
        crate::observe::off(format!("pay_diff_flags_dev on={on}"));
        on
    })
}

/// TEMP (lab): `REIMS_VGPU_LAB_PAY_REAP_AFTER=off` claims the payment's own slot
/// through `begin_entry`, which first retires every already-signalled slot (fence
/// reset, span reads, cleanup) in front of the submit. On by default: the same
/// retires run, in the same order, right after the submit instead. Latched.
pub(crate) fn lab_reap_after_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_PAY_REAP_AFTER").map_or(true, |v| v != "off");
        crate::observe::off(format!("pay_diff_reap_after on={on}"));
        on
    })
}

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

    /// Make the host's writes visible to the device, for a non-coherent type.
    pub(crate) unsafe fn flush(&self, device: &ash::Device) -> Result<(), DrawError> {
        if self.coherent || self.mapped.is_null() {
            return Ok(());
        }
        let range = [vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        unsafe { device.flush_mapped_memory_ranges(&range) }
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackInvalidate, e)))
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
    /// The one-trip kernel (four bindings, copies the pages it flags). The
    /// image kernel is one too: binding 0 is a sampled image and its push block
    /// carries the frame width and word count.
    pub one_trip: bool,
}

impl DiffPipeline {
    unsafe fn create(ctx: &DeviceContext, one_trip: bool, image: bool) -> Result<Self, DrawError> {
        let device = &ctx.device;
        let code: &[u32] = if image {
            &PAGE_DIFF_IMAGE_SPIRV
        } else if one_trip {
            &PAGE_DIFF_ONE_TRIP_SPIRV
        } else {
            &PAGE_DIFF_SPIRV
        };
        let one_trip = one_trip || image;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ScatterCreateShaderModule, e)))?;
        let binding_count: u32 = if one_trip { 4 } else { 3 };
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..binding_count)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(if image && b == 0 {
                        vk::DescriptorType::SAMPLED_IMAGE
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
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
            .size(if image { 16 } else { 8 })];
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
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(binding_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
        ];
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
            one_trip,
        })
    }

    /// Point the image kernel's set at this payment's resident view (sampled in
    /// `GENERAL`) and buffers.
    pub(crate) unsafe fn write_set_image(
        &self,
        device: &ash::Device,
        view: vk::ImageView,
        prev: &OwnedBuffer,
        flags: &OwnedBuffer,
        out: &OwnedBuffer,
        frame: u64,
        pages: u64,
    ) {
        let image = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let bufs = [
            vk::DescriptorBufferInfo::default()
                .buffer(prev.buffer)
                .offset(0)
                .range(frame),
            vk::DescriptorBufferInfo::default()
                .buffer(flags.buffer)
                .offset(0)
                .range(pages * 4),
            vk::DescriptorBufferInfo::default()
                .buffer(out.buffer)
                .offset(0)
                .range(frame),
        ];
        let mut writes = vec![vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&image)];
        for (i, b) in bufs.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(i as u32 + 1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(b)),
            );
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Point the set at this payment's buffers (`out` is the fourth binding of
    /// the one-trip kernel and is ignored by the two-trip one).
    pub(crate) unsafe fn write_set(
        &self,
        device: &ash::Device,
        cur: &OwnedBuffer,
        prev: &OwnedBuffer,
        flags: &OwnedBuffer,
        out: &OwnedBuffer,
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
            vk::DescriptorBufferInfo::default()
                .buffer(out.buffer)
                .offset(0)
                .range(frame),
        ];
        let n = if self.one_trip { 4 } else { 3 };
        let writes: Vec<vk::WriteDescriptorSet<'_>> = (0..n)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[i]))
            })
            .collect();
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
    /// The image kernel, created on first use beside the buffer one.
    pub image_pipeline: Option<DiffPipeline>,
    pub image_refused: bool,
    pub shadows: HashMap<u32, Shadow>,
    pub cur: Option<OwnedBuffer>,
    pub flags: Option<OwnedBuffer>,
    pub out: Option<OwnedBuffer>,
    /// TEMP (lab, `REIMS_VGPU_LAB_PAY_FLAGS_DEV`): the image kernel's flags in
    /// device memory, filled from and copied back to `flags` around it.
    pub flags_dev: Option<OwnedBuffer>,
}

pub(crate) static STATE: std::sync::Mutex<Option<PageDiffState>> = std::sync::Mutex::new(None);

impl PageDiffState {
    /// The compare kernel, created on first use; `None` after a refusal.
    pub(crate) unsafe fn ensure_pipeline(&mut self, ctx: &DeviceContext) -> bool {
        if self.pipeline.is_none() && !self.refused {
            match unsafe { DiffPipeline::create(ctx, one_trip_enabled(), false) } {
                Ok(p) => self.pipeline = Some(p),
                Err(e) => {
                    self.refused = true;
                    crate::observe::Emit::decline("page_diff_pipeline", &e).fail();
                }
            }
        }
        self.pipeline.is_some()
    }

    /// The image kernel, created on first use; `None` after a refusal (the
    /// payment then takes the frame copy).
    pub(crate) unsafe fn ensure_image_pipeline(&mut self, ctx: &DeviceContext) -> bool {
        if self.image_pipeline.is_none() && !self.image_refused {
            match unsafe { DiffPipeline::create(ctx, true, true) } {
                Ok(p) => self.image_pipeline = Some(p),
                Err(e) => {
                    self.image_refused = true;
                    crate::observe::Emit::decline("page_diff_image_pipeline", &e).fail();
                }
            }
        }
        self.image_pipeline.is_some()
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
