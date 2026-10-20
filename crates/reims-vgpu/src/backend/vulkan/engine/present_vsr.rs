//! TEMP (lab, `REIMS_VGPU_LAB_PRESENT_VSR=on|0..4`): the host window's scaled
//! present upscaled by NVIDIA RTX Video Super Resolution (NGX VSR) instead of
//! EASU or NVScaler — the guest renders below the panel and a Tensor-core
//! network brings it up to the window.
//!
//! The RTX Video SDK ships MSVC static libraries and this device builds with
//! MinGW, so VSR is reached through `rvsr.dll` (tools/rtx-vsr-shim, a C ABI
//! built with MSVC against the SDK), loaded at run time from beside the exe or
//! from `REIMS_VGPU_LAB_PRESENT_VSR_DLL`. Nothing links it; without the DLL, a
//! non-NVIDIA GPU or a driver without VSR the present keeps its other filters.
//!
//! NGX needs a few instance and device extensions (`VK_NVX_binary_import`,
//! `VK_NVX_image_view_handle`, ...): [`instance_extensions`] and
//! [`device_extensions`] are asked before the instance and the device are
//! created, and only advertised ones are enabled. [`init`] then binds NGX to
//! the device once; the present records [`create_feature`] (once) and
//! [`evaluate`] into its own command buffer.

use ash::vk;
use std::ffi::{c_char, c_void};

/// `REIMS_VGPU_LAB_PRESENT_VSR`: `on` = quality 4 (Ultra), or 0..=4
/// (0 = bicubic, 1 low .. 4 ultra). `None` when unset/off. Latched.
pub(crate) fn quality() -> Option<i32> {
    static Q: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    *Q.get_or_init(|| {
        let v = std::env::var("REIMS_VGPU_LAB_PRESENT_VSR").ok()?;
        match v.trim() {
            "on" => Some(4),
            "off" | "" => None,
            s => s.parse::<i32>().ok().filter(|q| (0..=4).contains(q)),
        }
    })
}

type FnExtensions = unsafe extern "C" fn(*mut u32, *mut *const *const c_char) -> i32;
type FnDeviceExtensions =
    unsafe extern "C" fn(vk::Instance, vk::PhysicalDevice, *mut u32, *mut *const *const c_char) -> i32;
type FnInit = unsafe extern "C" fn(
    vk::Instance,
    vk::PhysicalDevice,
    vk::Device,
    vk::PFN_vkGetInstanceProcAddr,
    vk::PFN_vkGetDeviceProcAddr,
    *const u16,
    *mut i32,
) -> i32;
type FnCreate = unsafe extern "C" fn(vk::Device, vk::CommandBuffer) -> i32;
type FnEvaluate = unsafe extern "C" fn(
    vk::CommandBuffer,
    vk::Image,
    vk::ImageView,
    vk::Format,
    u32,
    u32,
    vk::Image,
    vk::ImageView,
    vk::Format,
    u32,
    u32,
    *const u32,
    *const u32,
    i32,
) -> i32;

struct Api {
    instance_extensions: FnExtensions,
    device_extensions: FnDeviceExtensions,
    init: FnInit,
    create: FnCreate,
    evaluate: FnEvaluate,
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

fn api() -> Option<&'static Api> {
    static API: std::sync::OnceLock<Option<Api>> = std::sync::OnceLock::new();
    API.get_or_init(|| {
        quality()?;
        #[cfg(target_os = "windows")]
        unsafe {
            let path = std::env::var("REIMS_VGPU_LAB_PRESENT_VSR_DLL").unwrap_or_else(|_| "rvsr.dll".into());
            let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
            let module = LoadLibraryW(wide.as_ptr());
            if module.is_null() {
                crate::observe::fail(format!("present_vsr_refused reason=dll_not_loaded path={path}"));
                return None;
            }
            let sym = |name: &[u8]| {
                let p = GetProcAddress(module, name.as_ptr());
                (!p.is_null()).then_some(p)
            };
            let made = (|| {
                Some(Api {
                    instance_extensions: std::mem::transmute::<*mut c_void, FnExtensions>(sym(
                        b"rvsr_instance_extensions\0",
                    )?),
                    device_extensions: std::mem::transmute::<*mut c_void, FnDeviceExtensions>(sym(
                        b"rvsr_device_extensions\0",
                    )?),
                    init: std::mem::transmute::<*mut c_void, FnInit>(sym(b"rvsr_init\0")?),
                    create: std::mem::transmute::<*mut c_void, FnCreate>(sym(b"rvsr_create\0")?),
                    evaluate: std::mem::transmute::<*mut c_void, FnEvaluate>(sym(b"rvsr_evaluate\0")?),
                })
            })();
            if made.is_none() {
                crate::observe::fail(format!("present_vsr_refused reason=dll_missing_entry path={path}"));
            }
            made
        }
        #[cfg(not(target_os = "windows"))]
        {
            None
        }
    })
    .as_ref()
}

fn names(count: u32, list: *const *const c_char) -> Vec<*const c_char> {
    if list.is_null() {
        return Vec::new();
    }
    (0..count as usize).map(|i| unsafe { *list.add(i) }).collect()
}

/// The instance extensions VSR asks for (static strings owned by the DLL).
pub(crate) fn instance_extensions() -> Vec<*const c_char> {
    let Some(api) = api() else {
        return Vec::new();
    };
    let (mut n, mut list) = (0u32, std::ptr::null());
    let r = unsafe { (api.instance_extensions)(&mut n, &mut list) };
    if r != 0 {
        crate::observe::fail(format!("present_vsr_refused reason=instance_extensions ngx={r:#x}"));
        return Vec::new();
    }
    names(n, list)
}

/// The device extensions VSR asks for on `pd`.
pub(crate) fn device_extensions(instance: vk::Instance, pd: vk::PhysicalDevice) -> Vec<*const c_char> {
    let Some(api) = api() else {
        return Vec::new();
    };
    let (mut n, mut list) = (0u32, std::ptr::null());
    let r = unsafe { (api.device_extensions)(instance, pd, &mut n, &mut list) };
    if r != 0 {
        crate::observe::fail(format!("present_vsr_refused reason=device_extensions ngx={r:#x}"));
        return Vec::new();
    }
    names(n, list)
}

static READY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// The [`super::context::DeviceContext::generation`] NGX was bound to.
static BOUND_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Set once a later device asked: NGX (and the shim's feature and parameters)
/// still belong to the device that is gone, so VSR stays off rather than
/// evaluating against it, and the present falls back to its own filter.
static REBOUND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bind NGX to the device once; whether VSR is available on it.
pub(crate) fn init(
    entry: &ash::Entry,
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    device: &ash::Device,
    generation: u64,
) -> bool {
    use std::sync::atomic::Ordering;
    if let Some(ready) = READY.get() {
        if BOUND_GENERATION.load(Ordering::Acquire) == generation {
            return *ready;
        }
        if !REBOUND.swap(true, Ordering::AcqRel) {
            crate::observe::fail(format!(
                "present_vsr_refused reason=device_recreated bound_generation={} generation={generation}",
                BOUND_GENERATION.load(Ordering::Acquire)
            ));
        }
        return false;
    }
    BOUND_GENERATION.store(generation, Ordering::Release);
    *READY.get_or_init(|| {
        let Some(api) = api() else {
            return false;
        };
        let path: Vec<u16> = std::env::temp_dir()
            .to_string_lossy()
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let mut available = 0;
        let r = unsafe {
            (api.init)(
                instance.handle(),
                pd,
                device.handle(),
                entry.static_fn().get_instance_proc_addr,
                instance.fp_v1_0().get_device_proc_addr,
                path.as_ptr(),
                &mut available,
            )
        };
        crate::observe::off(format!("present_vsr init ngx={r:#x} available={available} quality={:?}", quality()));
        r == 0 && available == 1
    })
}

/// Whether [`init`] succeeded.
pub(crate) fn ready() -> bool {
    READY.get().copied().unwrap_or(false) && !REBOUND.load(std::sync::atomic::Ordering::Acquire)
}

/// Record the feature's creation into `cmd` (before its first evaluate).
pub(crate) unsafe fn create_feature(device: &ash::Device, cmd: vk::CommandBuffer) -> bool {
    let Some(api) = api() else {
        return false;
    };
    let r = (api.create)(device.handle(), cmd);
    crate::observe::off(format!("present_vsr create ngx={r:#x}"));
    r == 0
}

/// An image NGX reads or writes: the image, a 2D view of it, its format and size.
pub(crate) struct VsrImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

/// Record one super-resolution pass: the whole of `input` (SHADER_READ_ONLY)
/// into the whole of `output` (GENERAL, storage).
pub(crate) unsafe fn evaluate(cmd: vk::CommandBuffer, input: &VsrImage, output: &VsrImage, quality: i32) -> i32 {
    evaluate_rect(
        cmd,
        input,
        output,
        [0, 0, input.width, input.height],
        [0, 0, output.width, output.height],
        quality,
    )
}

/// [`evaluate`] from the `in_rect` part of `input` into the `out_rect` part
/// of `output` (`[x, y, width, height]` each); the rest of `output` is left
/// alone. TEMP (lab, PRESENT_VSR_TILES).
pub(crate) unsafe fn evaluate_rect(
    cmd: vk::CommandBuffer,
    input: &VsrImage,
    output: &VsrImage,
    in_rect: [u32; 4],
    out_rect: [u32; 4],
    quality: i32,
) -> i32 {
    let Some(api) = api() else {
        return -100;
    };
    (api.evaluate)(
        cmd,
        input.image,
        input.view,
        input.format,
        input.width,
        input.height,
        output.image,
        output.view,
        output.format,
        output.width,
        output.height,
        in_rect.as_ptr(),
        out_rect.as_ptr(),
        quality,
    )
}
