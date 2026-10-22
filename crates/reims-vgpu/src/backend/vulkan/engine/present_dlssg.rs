//! TEMP (lab, `REIMS_VGPU_LAB_PRESENT_DLSSG=on`): NVIDIA DLSS Frame Generation
//! (with Reflex) on the host window, through Streamline - an A/B against our
//! own frame generation (`present_fg`), which it excludes (both want the GPU's
//! optical-flow queue).
//!
//! `rslg.dll` (tools/rtx-vsr-shim/rslg.cpp, a C ABI over Streamline) is loaded
//! from `REIMS_VGPU_LAB_PRESENT_DLSSG_DIR` (default `E:/reims-lab/dlssg`, which
//! also holds the SDK's signed `sl.*.dll`, `nvngx_dlssg.dll` and
//! `NvLowLatencyVk.dll`). [`loader`] runs `slInit` before the Vulkan loader is
//! opened and names `sl.interposer.dll` as the loader: Streamline's
//! `vkCreateInstance`/`vkCreateDevice` proxies then add what DLSS-G needs and
//! its swapchain proxies present the generated frames; every other call goes to
//! `vulkan-1`. Per present the window records a frame token (Reflex sleep,
//! markers), tags the frame (the swapchain image as hudless colour, a constant
//! depth, zero motion vectors - a 2D screen has no geometry; DLSS-G's own
//! optical flow carries the motion) and sets 2D constants.

use std::ffi::c_void;

use ash::vk;

/// `REIMS_VGPU_LAB_PRESENT_DLSSG=on` (and PRESENT_FG off). Latched.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(std::env::var("REIMS_VGPU_LAB_PRESENT_DLSSG").as_deref(), Ok("on" | "1"))
            && !crate::host_window::present::lab_present_fg()
    })
}

/// TEMP (lab): `REIMS_VGPU_LAB_PRESENT_DLSS_SR=dlaa|quality|balanced|performance|ultra` - the present's upscale
/// (where VSR runs) by DLSS Super Resolution through Streamline instead; `REIMS_VGPU_LAB_PRESENT_DLSS_PRESET`
/// K/L/M (the DLSS 4.5 transformer presets L and M) or unset for the mode's default. `(sl::DLSSMode, sl::DLSSPreset)`.
pub(crate) fn dlss_sr() -> Option<(i32, i32)> {
    static SR: std::sync::OnceLock<Option<(i32, i32)>> = std::sync::OnceLock::new();
    *SR.get_or_init(|| {
        let mode = match std::env::var("REIMS_VGPU_LAB_PRESENT_DLSS_SR").ok()?.as_str() {
            "performance" => 1,
            "balanced" => 2,
            "quality" => 3,
            "ultra" => 4,
            "dlaa" => 6,
            other => {
                crate::observe::fail(format!("present_dlss_sr_refused reason=mode_unrecognized value={other}"));
                return None;
            }
        };
        let preset = match std::env::var("REIMS_VGPU_LAB_PRESENT_DLSS_PRESET").as_deref() {
            Ok("K" | "k") => 11,
            Ok("L" | "l") => 12,
            Ok("M" | "m") => 13,
            _ => 0,
        };
        Some((mode, preset))
    })
}

/// TEMP (lab, 10-21): `REIMS_VGPU_LAB_PRESENT_DLSS_MVEC=off` - the motion instance runs on zero vectors instead of the
/// optical flow's (the flow cost ~20 % of canvas fps in chain946; this is its A/B for the picture in motion).
pub(crate) fn dlss_motion_vectors() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_PRESENT_DLSS_MVEC").map_or(true, |v| v != "off"))
}

/// TEMP (lab, 10-21): `REIMS_VGPU_LAB_PRESENT_DLSS_PRESET_MOTION=K|L|M` - with PRESENT_VSR_SETTLE the moving frames
/// take DLSS with this preset (its own instance, viewport 2) instead of EASU, and the settled present keeps the main
/// preset (viewport 1): Mish's "K in motion, M on still". `sl::DLSSPreset`.
pub(crate) fn dlss_motion_preset() -> Option<i32> {
    static P: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    *P.get_or_init(|| match std::env::var("REIMS_VGPU_LAB_PRESENT_DLSS_PRESET_MOTION").as_deref() {
        Ok("K" | "k") => Some(11),
        Ok("L" | "l") => Some(12),
        Ok("M" | "m") => Some(13),
        _ => None,
    })
}

/// Streamline is wanted: DLSS-G or the DLSS upscale.
fn sl_wanted() -> bool {
    on() || dlss_sr().is_some()
}

static PRESENT_INDEX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A new present: the Streamline frame index both features of this present use.
pub(crate) fn begin_present() {
    PRESENT_INDEX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn present_index() -> u32 {
    PRESENT_INDEX.load(std::sync::atomic::Ordering::Relaxed)
}

fn dir() -> String {
    std::env::var("REIMS_VGPU_LAB_PRESENT_DLSSG_DIR").unwrap_or_else(|_| "E:/reims-lab/dlssg".into())
}

#[repr(C)]
struct RslgImage {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    layout: u32,
    format: u32,
    width: u32,
    height: u32,
    usage: u32,
}

type FnInit = unsafe extern "C" fn(*const u16, *const u16, i32) -> i32;
type FnSupported = unsafe extern "C" fn(vk::PhysicalDevice) -> i32;
type FnInt = unsafe extern "C" fn(i32) -> i32;
type FnFrame = unsafe extern "C" fn(u32) -> *mut c_void;
type FnMarker = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type FnTag = unsafe extern "C" fn(*mut c_void, vk::CommandBuffer, *const RslgImage, *const RslgImage, *const RslgImage, u32, u32) -> i32;
type FnConstants = unsafe extern "C" fn(*mut c_void, f32, f32, i32) -> i32;
type FnState = unsafe extern "C" fn(*mut u32, *mut u32, *mut u64) -> i32;
type FnDlss = unsafe extern "C" fn(i32, u32, u32, i32) -> i32;
type FnVoid = unsafe extern "C" fn();
type FnDlssVp = unsafe extern "C" fn(i32, i32, u32, u32, i32) -> i32;
type FnDlssEvalVp = unsafe extern "C" fn(
    i32,
    *mut c_void,
    vk::CommandBuffer,
    *const RslgImage,
    *const RslgImage,
    *const RslgImage,
    *const RslgImage,
    i32,
) -> i32;
type FnDlssEval = unsafe extern "C" fn(
    *mut c_void,
    vk::CommandBuffer,
    *const RslgImage,
    *const RslgImage,
    *const RslgImage,
    *const RslgImage,
    i32,
) -> i32;

struct Api {
    init: FnInit,
    supported: FnSupported,
    reflex: FnInt,
    dlssg: FnInt,
    frame: FnFrame,
    marker: FnMarker,
    tag: FnTag,
    constants: FnConstants,
    state: FnState,
    dlss: FnDlss,
    dlss_eval: FnDlssEval,
    /// The per-viewport pair (an older rslg.dll lacks them: no motion instance then).
    dlss_vp: Option<FnDlssVp>,
    dlss_eval_vp: Option<FnDlssEvalVp>,
    /// `slShutdown` (before the device it serves is destroyed; `rslg_init` then runs `slInit` again).
    shutdown: Option<FnVoid>,
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn api() -> Option<&'static Api> {
    static API: std::sync::OnceLock<Option<Api>> = std::sync::OnceLock::new();
    API.get_or_init(|| {
        if !sl_wanted() {
            return None;
        }
        #[cfg(target_os = "windows")]
        unsafe {
            let path = format!("{}/rslg.dll", dir()).replace('/', "\\");
            // LOAD_WITH_ALTERED_SEARCH_PATH: rslg's own dependency
            // (sl.interposer.dll) resolves from its directory.
            let module = LoadLibraryExW(wide(&path).as_ptr(), std::ptr::null_mut(), 0x8);
            if module.is_null() {
                crate::observe::fail(format!("present_dlssg_refused reason=dll_not_loaded path={path}"));
                return None;
            }
            let sym = |name: &[u8]| {
                let p = GetProcAddress(module, name.as_ptr());
                (!p.is_null()).then_some(p)
            };
            let made = (|| {
                Some(Api {
                    init: std::mem::transmute::<*mut c_void, FnInit>(sym(b"rslg_init\0")?),
                    supported: std::mem::transmute::<*mut c_void, FnSupported>(sym(b"rslg_supported\0")?),
                    reflex: std::mem::transmute::<*mut c_void, FnInt>(sym(b"rslg_reflex\0")?),
                    dlssg: std::mem::transmute::<*mut c_void, FnInt>(sym(b"rslg_dlssg\0")?),
                    frame: std::mem::transmute::<*mut c_void, FnFrame>(sym(b"rslg_frame\0")?),
                    marker: std::mem::transmute::<*mut c_void, FnMarker>(sym(b"rslg_marker\0")?),
                    tag: std::mem::transmute::<*mut c_void, FnTag>(sym(b"rslg_tag\0")?),
                    constants: std::mem::transmute::<*mut c_void, FnConstants>(sym(b"rslg_constants\0")?),
                    state: std::mem::transmute::<*mut c_void, FnState>(sym(b"rslg_state\0")?),
                    dlss: std::mem::transmute::<*mut c_void, FnDlss>(sym(b"rslg_dlss\0")?),
                    dlss_eval: std::mem::transmute::<*mut c_void, FnDlssEval>(sym(b"rslg_dlss_eval\0")?),
                    dlss_vp: sym(b"rslg_dlss_vp\0").map(|p| std::mem::transmute::<*mut c_void, FnDlssVp>(p)),
                    dlss_eval_vp: sym(b"rslg_dlss_eval_vp\0").map(|p| std::mem::transmute::<*mut c_void, FnDlssEvalVp>(p)),
                    shutdown: sym(b"rslg_shutdown\0").map(|p| std::mem::transmute::<*mut c_void, FnVoid>(p)),
                })
            })();
            if made.is_none() {
                crate::observe::fail(format!("present_dlssg_refused reason=dll_missing_entry path={path}"));
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

/// Before the Vulkan loader is opened: `slInit`, and the loader to open
/// instead of `vulkan-1` (`sl.interposer.dll`). `None` when DLSS-G is off or
/// Streamline refused - the device then comes up as always.
pub(crate) fn loader() -> Option<std::path::PathBuf> {
    static LOADER: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    LOADER
        .get_or_init(|| {
            let api = api()?;
            sl_init(api).then(|| std::path::PathBuf::from(format!("{}\\sl.interposer.dll", dir().replace('/', "\\"))))
        })
        .clone()
}

/// `slInit` through the shim (a no-op while Streamline is up). Whether it is up.
fn sl_init(api: &Api) -> bool {
    let logs = std::env::temp_dir().join("reims-sl");
    let _ = std::fs::create_dir_all(&logs);
    let verbose = i32::from(std::env::var("REIMS_VGPU_LAB_PRESENT_DLSSG_VERBOSE").is_ok_and(|v| v == "1"));
    let d = dir().replace('/', "\\");
    let r = unsafe { (api.init)(wide(&d).as_ptr(), wide(&logs.to_string_lossy()).as_ptr(), verbose) };
    crate::observe::off(format!("present_dlssg init sl={r} dir={d} logs={}", logs.display()));
    if r != 0 {
        crate::observe::fail(format!("present_dlssg_refused reason=sl_init sl={r}"));
    }
    r == 0
}

/// Which device Streamline serves: none created yet, the current one, none since `slShutdown`, or none for good.
const SL_DEVICE_NONE: u8 = 0;
const SL_DEVICE_CURRENT: u8 = 1;
const SL_DEVICE_SHUT: u8 = 2;
const SL_DEVICE_RETIRED: u8 = 3;
static SL_DEVICE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(SL_DEVICE_NONE);

/// The loader the engine's device is created through: [`loader`] for the first device, `vulkan-1` for any later one
/// (device-lost recovery), which then runs without DLSS and DLSS-G while the VM keeps running. Streamline serves one
/// device per process:
/// - a second device through the still-bound interposer crashed DLSS-G in its swapchain hook (10-22, Mish's work VM;
///   SL: "plugins already initialized but could be using the wrong device"), and `slSetVulkanInfo` cannot rebind
///   (`initializePlugins` refuses once initialized - pluginManager.cpp);
/// - `slShutdown` + `slInit` in the same process crashed both times on the new device's `vkCreateDevice` through the
///   interposer: b976 in nvngx_dlssg.dll calling into the unloaded sl.common.dll, b977 (modules pinned) in
///   sl.common.dll itself during `initializePlugins` (a null pointer left by the shutdown).
pub(crate) fn device_loader() -> Option<std::path::PathBuf> {
    use std::sync::atomic::Ordering;
    let path = loader()?;
    let retire = |why: &str| {
        if SL_DEVICE.swap(SL_DEVICE_RETIRED, Ordering::AcqRel) != SL_DEVICE_RETIRED {
            crate::observe::fail(format!("present_dlssg_refused reason={why} (DLSS and DLSS-G off from here)"));
        }
        None
    };
    match SL_DEVICE.load(Ordering::Acquire) {
        SL_DEVICE_NONE => {
            SL_DEVICE.store(SL_DEVICE_CURRENT, Ordering::Release);
            Some(path)
        }
        SL_DEVICE_SHUT => retire("restart_unsupported"),
        SL_DEVICE_CURRENT => retire("second_device"),
        _ => None,
    }
}

/// Before the device Streamline serves is destroyed (the window's swapchain already gone): `slShutdown`, as the SL
/// guide asks ("call slShutdown() before destroying ... devices"), so none of its threads outlives the device.
/// Nothing when it serves none.
pub(crate) fn release_device() {
    use std::sync::atomic::Ordering;
    if SL_DEVICE.load(Ordering::Acquire) != SL_DEVICE_CURRENT {
        return;
    }
    match api().and_then(|a| a.shutdown) {
        Some(shutdown) => {
            pin_sl_modules();
            unsafe { shutdown() };
            SL_DEVICE.store(SL_DEVICE_SHUT, Ordering::Release);
            crate::observe::off("present_dlssg shutdown (its device is going away)".to_string());
        }
        // An rslg.dll without the entry: the next device must not come through the interposer.
        None => SL_DEVICE.store(SL_DEVICE_RETIRED, Ordering::Release),
    }
}

/// Keep Streamline's and NGX's modules mapped for the life of the process. `slShutdown` frees the plugins, but NGX's
/// feature DLL (nvngx_dlssg.dll, loaded by the driver's _nvngx.dll) stays and keeps callbacks into sl.common.dll
/// (b976: nvngx_dlssg.dll executed 0x7fffa245e490, in the range sl.common had held, after the shutdown).
fn pin_sl_modules() {
    #[cfg(target_os = "windows")]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetModuleHandleExW(flags: u32, name: *const u16, module: *mut *mut c_void) -> i32;
        }
        const GET_MODULE_HANDLE_EX_FLAG_PIN: u32 = 0x1;
        let mut pinned = Vec::new();
        for name in [
            "sl.interposer.dll",
            "sl.common.dll",
            "sl.dlss.dll",
            "sl.dlss_g.dll",
            "sl.reflex.dll",
            "sl.pcl.dll",
            "nvngx_dlss.dll",
            "nvngx_dlssg.dll",
            "_nvngx.dll",
            "NvLowLatencyVk.dll",
        ] {
            let mut module = std::ptr::null_mut();
            // Only modules already loaded: the flag pins, it never loads.
            if unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, wide(name).as_ptr(), &mut module) } != 0 {
                pinned.push(name);
            }
        }
        crate::observe::off(format!("present_dlssg pinned {}", pinned.join(",")));
    }
}

/// Whether the current device came up through Streamline.
fn sl_device() -> bool {
    SL_DEVICE.load(std::sync::atomic::Ordering::Acquire) == SL_DEVICE_CURRENT
}

/// Whether the device came up through Streamline (DLSS-G active).
pub(crate) fn active() -> bool {
    on() && sl_device()
}

/// Whether the DLSS upscale can run: asked for and the device came up through Streamline.
pub(crate) fn dlss_sr_active() -> bool {
    dlss_sr().is_some() && sl_device()
}

/// One image handed to the DLSS upscale (the layout it is in when the evaluate runs).
pub(crate) struct DlssSrImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub layout: vk::ImageLayout,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub usage: vk::ImageUsageFlags,
}

fn rslg_image(i: &DlssSrImage) -> RslgImage {
    RslgImage {
        image: i.image,
        view: i.view,
        memory: i.memory,
        layout: i.layout.as_raw() as u32,
        format: i.format.as_raw() as u32,
        width: i.width,
        height: i.height,
        usage: i.usage.as_raw(),
    }
}

/// Whether the motion instance can run: a motion preset asked for and an rslg.dll with the per-viewport pair.
pub(crate) fn dlss_motion_active() -> bool {
    dlss_sr_active() && dlss_motion_preset().is_some() && api().is_some_and(|a| a.dlss_vp.is_some() && a.dlss_eval_vp.is_some())
}

/// Record a DLSS upscale of `input` into `output` (depth and motion vectors at the input size) into `cmd`, setting
/// the options first whenever the output size changed. `moving` picks the motion instance (viewport 2, the motion
/// preset - [`dlss_motion_active`]) over the main one (viewport 1). 0, or the shim's / Streamline's code.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dlss_sr_evaluate(
    cmd: vk::CommandBuffer,
    input: &DlssSrImage,
    output: &DlssSrImage,
    depth: &DlssSrImage,
    mvec: &DlssSrImage,
    reset: bool,
    moving: bool,
) -> i32 {
    static SET_FOR: [std::sync::Mutex<Option<(u32, u32)>>; 2] = [std::sync::Mutex::new(None), std::sync::Mutex::new(None)];
    let (Some(api), Some((mode, main_preset))) = (api(), dlss_sr()) else {
        return -100;
    };
    let motion = if moving {
        match (dlss_motion_preset(), api.dlss_vp, api.dlss_eval_vp) {
            (Some(p), Some(set), Some(eval)) => Some((p, set, eval)),
            _ => return -103,
        }
    } else {
        None
    };
    let (slot, vp, preset) = match motion {
        Some((p, _, _)) => (1usize, 2i32, p),
        None => (0usize, 1i32, main_preset),
    };
    {
        let mut set = SET_FOR[slot].lock().unwrap_or_else(|e| e.into_inner());
        if *set != Some((output.width, output.height)) {
            let r = match motion {
                Some((_, set_vp, _)) => set_vp(vp, mode, output.width, output.height, preset),
                None => (api.dlss)(mode, output.width, output.height, preset),
            };
            crate::observe::off(format!(
                "present_dlss_sr options sl={r} viewport={vp} mode={mode} preset={preset} {}x{}->{}x{}",
                input.width, input.height, output.width, output.height
            ));
            if r != 0 {
                return r;
            }
            *set = Some((output.width, output.height));
        }
    }
    let token = (api.frame)(present_index());
    if token.is_null() {
        return -101;
    }
    let (i, o, d, m) = (rslg_image(input), rslg_image(output), rslg_image(depth), rslg_image(mvec));
    match motion {
        Some((_, _, eval_vp)) => eval_vp(vp, token, cmd, &i, &o, &d, &m, i32::from(reset)),
        None => (api.dlss_eval)(token, cmd, &i, &o, &d, &m, i32::from(reset)),
    }
}

struct DlssgImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// The window's per-present DLSS-G state: the constant depth and zero motion
/// vectors at the window size, views of the swapchain images, the frame count.
pub(crate) struct Dlssg {
    width: u32,
    height: u32,
    depth: DlssgImage,
    mvec: DlssgImage,
    cleared: bool,
    views: Vec<(vk::Image, vk::ImageView)>,
    frame: u32,
    started: bool,
    failed: bool,
}

const DEPTH_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
const MVEC_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;
const AUX_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::STORAGE.as_raw() | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

impl Dlssg {
    pub(crate) fn matches(&self, width: u32, height: u32) -> bool {
        (self.width, self.height) == (width, height)
    }

    pub(crate) unsafe fn create(
        device: &ash::Device,
        width: u32,
        height: u32,
        memory_type: &dyn Fn(u32, u64, bool) -> Option<u32>,
    ) -> Result<Self, vk::Result> {
        let make = |format: vk::Format| -> Result<DlssgImage, vk::Result> {
            let mut img = DlssgImage { image: vk::Image::null(), memory: vk::DeviceMemory::null(), view: vk::ImageView::null() };
            let r = (|| -> Result<(), vk::Result> {
                img.image = device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D { width, height, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(AUX_USAGE)
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
                        .format(format)
                        .subresource_range(super::color_subresource_range()),
                    None,
                )?;
                Ok(())
            })();
            match r {
                Ok(()) => Ok(img),
                Err(e) => {
                    destroy_image(device, &img);
                    Err(e)
                }
            }
        };
        let depth = make(DEPTH_FORMAT)?;
        let mvec = match make(MVEC_FORMAT) {
            Ok(m) => m,
            Err(e) => {
                destroy_image(device, &depth);
                return Err(e);
            }
        };
        Ok(Self { width, height, depth, mvec, cleared: false, views: Vec::new(), frame: 0, started: false, failed: false })
    }

    /// Into the frame's command buffer, after the window pass drew `dst` (left
    /// in `PRESENT_SRC_KHR`): a frame token, the tags and the constants. The
    /// token, for the present markers; `None` when DLSS-G is not running.
    pub(crate) unsafe fn record(
        &mut self,
        device: &ash::Device,
        pd: vk::PhysicalDevice,
        cmd: vk::CommandBuffer,
        dst: vk::Image,
        format: vk::Format,
    ) -> Option<*mut c_void> {
        if self.failed {
            return None;
        }
        let api = api()?;
        if !self.started {
            self.started = true;
            let supported = (api.supported)(pd);
            let reflex = (api.reflex)(1);
            let on = (api.dlssg)(1);
            crate::observe::off(format!("present_dlssg start supported={supported} reflex={reflex} dlssg={on} {}x{}", self.width, self.height));
            if supported != 0 || on != 0 {
                self.failed = true;
                crate::observe::fail(format!("present_dlssg_refused reason=start supported={supported} dlssg={on}"));
                return None;
            }
        }
        let token = (api.frame)(present_index());
        self.frame = self.frame.wrapping_add(1);
        if token.is_null() {
            return None;
        }
        let range = super::color_subresource_range();
        if !self.cleared {
            let to_dst = [self.depth.image, self.mvec.image].map(|image| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range)
            });
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_dst);
            for image in [self.depth.image, self.mvec.image] {
                device.cmd_clear_color_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[range]);
            }
            let to_general = [self.depth.image, self.mvec.image].map(|image| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range)
            });
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &to_general);
            self.cleared = true;
        }
        let view = match self.views.iter().find(|(i, _)| *i == dst) {
            Some(&(_, v)) => v,
            None => {
                let v = device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(dst)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format)
                            .subresource_range(range),
                        None,
                    )
                    .ok()?;
                self.views.push((dst, v));
                v
            }
        };
        let hudless = RslgImage {
            image: dst,
            view,
            memory: vk::DeviceMemory::null(),
            layout: vk::ImageLayout::PRESENT_SRC_KHR.as_raw() as u32,
            format: format.as_raw() as u32,
            width: self.width,
            height: self.height,
            usage: (vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST).as_raw(),
        };
        let aux = |img: &DlssgImage, format: vk::Format| RslgImage {
            image: img.image,
            view: img.view,
            memory: img.memory,
            layout: vk::ImageLayout::GENERAL.as_raw() as u32,
            format: format.as_raw() as u32,
            width: self.width,
            height: self.height,
            usage: AUX_USAGE.as_raw(),
        };
        let (depth, mvec) = (aux(&self.depth, DEPTH_FORMAT), aux(&self.mvec, MVEC_FORMAT));
        let tag = (api.tag)(token, cmd, &hudless, &depth, &mvec, self.width, self.height);
        let constants = (api.constants)(token, 1.0 / self.width as f32, 1.0 / self.height as f32, 0);
        if (tag != 0 || constants != 0) && crate::observe::first_sight("present_dlssg_tag_refused", 0) {
            crate::observe::fail(format!("present_dlssg_refused reason=tag tag={tag} constants={constants}"));
        }
        (api.marker)(token, 2); // eRenderSubmitStart
        if self.frame % 600 == 1 {
            let (mut presented, mut status, mut vram) = (0u32, 0u32, 0u64);
            let r = (api.state)(&mut presented, &mut status, &mut vram);
            crate::observe::off(format!(
                "present_dlssg state sl={r} frame={} presented_per_present={presented} status={status:#x} vram_mb={}",
                self.frame,
                vram >> 20
            ));
        }
        Some(token)
    }

    /// The render-submit end and present markers around the frame's submit +
    /// present (`start`: before it).
    pub(crate) fn mark_present(token: *mut c_void, start: bool) {
        let Some(api) = api() else {
            return;
        };
        unsafe {
            if start {
                (api.marker)(token, 3); // eRenderSubmitEnd
                (api.marker)(token, 4); // ePresentStart
            } else {
                (api.marker)(token, 5); // ePresentEnd
            }
        }
    }

    /// Before the window's swapchain is destroyed for a recreation: DLSS-G off and the whole device idle, then the
    /// views of the old images dropped; the next [`Self::record`] turns DLSS-G on again. Streamline presents the
    /// generated frames into the swapchain from queues of its own, and the queue idle the recreation waits for is
    /// only ours (10-22, Mish's work VM: a same-size recreation at t=61.5 s, the GPU hung - 0.5 s tranches - and the
    /// device was lost at 63.7 s, twice in two boots).
    pub(crate) unsafe fn pause(&mut self, device: &ash::Device) {
        if self.started && !self.failed {
            if let Some(api) = api() {
                let r = (api.dlssg)(0);
                crate::observe::off(format!("present_dlssg pause dlssg={r} frame={}", self.frame));
            }
            self.started = false;
        }
        let _ = device.device_wait_idle();
        for (_, view) in self.views.drain(..) {
            device.destroy_image_view(view, None);
        }
    }

    /// Views of swapchain images that are gone (after a recreation), dropped.
    pub(crate) unsafe fn keep_views_of(&mut self, device: &ash::Device, images: &[vk::Image]) {
        self.views.retain(|(image, view)| {
            let keep = images.contains(image);
            if !keep {
                device.destroy_image_view(*view, None);
            }
            keep
        });
    }

    /// Only with the queue idle.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for (_, view) in self.views.drain(..) {
            device.destroy_image_view(view, None);
        }
        destroy_image(device, &self.depth);
        destroy_image(device, &self.mvec);
    }
}

unsafe fn destroy_image(device: &ash::Device, img: &DlssgImage) {
    if img.view != vk::ImageView::null() {
        device.destroy_image_view(img.view, None);
    }
    if img.image != vk::Image::null() {
        device.destroy_image(img.image, None);
    }
    if img.memory != vk::DeviceMemory::null() {
        device.free_memory(img.memory, None);
    }
}
