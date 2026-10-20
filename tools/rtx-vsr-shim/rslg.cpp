// rslg: a C ABI over NVIDIA Streamline (DLSS Frame Generation + Reflex + PC Latency) for reims-vgpu's host window.
//
// TEMP (lab, REIMS_VGPU_LAB_PRESENT_DLSSG). Streamline is C++ (sl.h); the Rust side loads this DLL and calls plain C.
// Integration mode: the interposer. After rslg_init (slInit) the host loads its Vulkan loader entry points from
// sl.interposer.dll instead of vulkan-1.dll; SL's vkCreateInstance/vkCreateDevice proxies add the extensions,
// features and queues DLSS-G needs, and its swapchain/present proxies run the generated frames. Every other Vulkan
// call goes straight to vulkan-1.
//
// Per present the host calls rslg_frame (a frame token, Reflex sleep, simulation markers), rslg_tag (hudless colour,
// depth, motion vectors, backbuffer extent), rslg_constants, and wraps vkQueuePresentKHR in rslg_marker(PresentStart/
// PresentEnd). Plugins load only from `plugin_dir` (the signed SDK DLLs); no over-the-air updates.
//
// Build: build-rslg.bat (MSVC; Streamline SDK headers + sl.interposer.lib, Vulkan headers).

#define NOMINMAX
#include <windows.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>

#include "sl.h"
#include "sl_consts.h"
#include "sl_dlss_g.h"
#include "sl_helpers_vk.h"
#include "sl_pcl.h"
#include "sl_reflex.h"

#define RSLG_API extern "C" __declspec(dllexport)

static bool g_inited = false;
static sl::ViewportHandle g_viewport{0};

static int rc(sl::Result r) {
    return r == sl::Result::eOk ? 0 : int(r) + 1000;
}

// slInit with DLSS-G, Reflex and PCL for Vulkan. `plugin_dir`: where sl.*.dll and nvngx_dlssg.dll are;
// `log_dir`: SL's log (may be null). Returns 0 or 1000 + sl::Result.
RSLG_API int rslg_init(const wchar_t* plugin_dir, const wchar_t* log_dir, int verbose) {
    if (g_inited) return 0;
    static const sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
    const wchar_t* paths[] = {plugin_dir};
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = verbose ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = log_dir;
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eDisableDebugText;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = 3;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "reims-vgpu";
    pref.projectId = "a8f1c3d2-6b4e-4f7a-9d2c-3e5b7a9c1f04";
    pref.renderAPI = sl::RenderAPI::eVulkan;
    sl::Result r = slInit(pref, sl::kSDKVersion);
    if (r != sl::Result::eOk) return rc(r);
    g_inited = true;
    return 0;
}

// Whether DLSS-G is supported on `pd` (0) or why not (1000 + sl::Result).
RSLG_API int rslg_supported(VkPhysicalDevice pd) {
    sl::AdapterInfo info{};
    info.vkPhysicalDevice = pd;
    return rc(slIsFeatureSupported(sl::kFeatureDLSS_G, info));
}

// Reflex on (low latency, plus boost when `boost`), PCL markers on.
RSLG_API int rslg_reflex(int boost) {
    sl::ReflexOptions o{};
    o.mode = boost ? sl::ReflexMode::eLowLatencyWithBoost : sl::ReflexMode::eLowLatency;
    return rc(slReflexSetOptions(o));
}

// DLSS-G on (1) or off (0) for the one viewport.
RSLG_API int rslg_dlssg(int on) {
    sl::DLSSGOptions o{};
    o.mode = on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    o.numFramesToGenerate = 1;
    return rc(slDLSSGSetOptions(g_viewport, o));
}

// A new frame: its token (opaque, valid until the next call), Reflex sleep and the simulation markers.
RSLG_API void* rslg_frame(uint32_t index) {
    sl::FrameToken* token = nullptr;
    if (slGetNewFrameToken(token, &index) != sl::Result::eOk) return nullptr;
    slReflexSleep(*token);
    slPCLSetMarker(sl::PCLMarker::eSimulationStart, *token);
    slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *token);
    return token;
}

// A PCL marker (sl::PCLMarker value) for `token`.
RSLG_API int rslg_marker(void* token, uint32_t marker) {
    if (!token) return -1;
    return rc(slPCLSetMarker(sl::PCLMarker(marker), *static_cast<sl::FrameToken*>(token)));
}

struct RslgImage {
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    uint32_t layout;  // VkImageLayout at present time
    uint32_t format;  // VkFormat
    uint32_t width;
    uint32_t height;
    uint32_t usage;   // VkImageUsageFlags
};

static sl::Resource resource(const RslgImage& i) {
    sl::Resource r(sl::ResourceType::eTex2d, (void*)i.image, (void*)i.memory, (void*)i.view, i.layout);
    r.width = i.width;
    r.height = i.height;
    r.nativeFormat = i.format;
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = i.usage;
    return r;
}

// Tag this frame's inputs, valid until present: the hudless colour (the final frame, backbuffer-sized), depth,
// motion vectors (pixel units, mvec_scale set in rslg_constants) and the backbuffer extent.
RSLG_API int rslg_tag(void* token, VkCommandBuffer cb, const RslgImage* hudless, const RslgImage* depth,
                      const RslgImage* mvec, uint32_t back_w, uint32_t back_h) {
    if (!token) return -1;
    sl::Resource h = resource(*hudless), d = resource(*depth), m = resource(*mvec);
    sl::Extent back{0, 0, back_w, back_h};
    sl::Extent mv{0, 0, mvec->width, mvec->height};
    sl::ResourceTag tags[] = {
        sl::ResourceTag(&h, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent, &back),
        sl::ResourceTag(&d, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &mv),
        sl::ResourceTag(&m, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &mv),
        sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle::eValidUntilPresent, &back),
    };
    sl::CommandBuffer* slcb = reinterpret_cast<sl::CommandBuffer*>(cb);
    return rc(slSetTagForFrame(*static_cast<sl::FrameToken*>(token), g_viewport, tags, 4, slcb));
}

// The common constants of a 2D screen: identity camera, motion vectors in pixels of the mvec image (scaled by
// mvec_scale to [-1, 1]), depth not inverted, camera motion included; `reset` when the frame has no connection to
// the previous one.
RSLG_API int rslg_constants(void* token, float mvec_scale_x, float mvec_scale_y, int reset) {
    if (!token) return -1;
    sl::Constants c{};
    auto identity = [](sl::float4x4& m) {
        m.setRow(0, {1, 0, 0, 0});
        m.setRow(1, {0, 1, 0, 0});
        m.setRow(2, {0, 0, 1, 0});
        m.setRow(3, {0, 0, 0, 1});
    };
    identity(c.cameraViewToClip);
    identity(c.clipToCameraView);
    identity(c.clipToLensClip);
    identity(c.clipToPrevClip);
    identity(c.prevClipToClip);
    c.jitterOffset = {0, 0};
    c.mvecScale = {mvec_scale_x, mvec_scale_y};
    c.cameraPinholeOffset = {0, 0};
    c.cameraPos = {0, 0, 0};
    c.cameraUp = {0, 1, 0};
    c.cameraRight = {1, 0, 0};
    c.cameraFwd = {0, 0, 1};
    c.cameraNear = 0.1f;
    c.cameraFar = 1000.0f;
    c.cameraFOV = 1.0f;
    c.cameraAspectRatio = 1.0f;
    c.depthInverted = sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eTrue;
    return rc(slSetConstants(c, *static_cast<sl::FrameToken*>(token), g_viewport));
}

// DLSS-G's state: frames presented per host present and its status bits.
RSLG_API int rslg_state(uint32_t* presented, uint32_t* status, uint64_t* vram) {
    sl::DLSSGState s{};
    sl::Result r = slDLSSGGetState(g_viewport, s, nullptr);
    if (presented) *presented = s.numFramesActuallyPresented;
    if (status) *status = uint32_t(s.status);
    if (vram) *vram = s.estimatedVRAMUsageInBytes;
    return rc(r);
}

RSLG_API void rslg_shutdown() {
    if (g_inited) slShutdown();
    g_inited = false;
}
