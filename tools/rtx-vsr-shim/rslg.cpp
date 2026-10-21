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
#include "sl_dlss.h"
#include "sl_dlss_g.h"
#include "sl_helpers_vk.h"
#include "sl_pcl.h"
#include "sl_reflex.h"

#define RSLG_API extern "C" __declspec(dllexport)

static bool g_inited = false;
static sl::ViewportHandle g_viewport{0};
// DLSS Super Resolution runs on its own viewport: with DLSS-G on viewport 0 in the same frame, the two set different
// tags (depth / motion vectors at the guest frame's size vs the window's) and constants (mvecScale), and a viewport
// holds one set of each per frame.
static sl::ViewportHandle g_viewport_sr{1};

static int rc(sl::Result r) {
    return r == sl::Result::eOk ? 0 : int(r) + 1000;
}

// slInit with DLSS-G, Reflex and PCL for Vulkan. `plugin_dir`: where sl.*.dll and nvngx_dlssg.dll are;
// `log_dir`: SL's log (may be null). Returns 0 or 1000 + sl::Result.
RSLG_API int rslg_init(const wchar_t* plugin_dir, const wchar_t* log_dir, int verbose) {
    if (g_inited) return 0;
    static const sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL, sl::kFeatureDLSS};
    const wchar_t* paths[] = {plugin_dir};
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = verbose ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = log_dir;
    // rslg_tag tags per frame (slSetTagForFrame), which SL refuses with eErrorInvalidIntegration unless
    // eUseFrameBasedResourceTagging is set here (b920c: every tag refused, DLSS-G never started).
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eDisableDebugText |
                 sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = 4;
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

// A new frame: its token (opaque, valid until the next call), Reflex sleep and the simulation markers. Idempotent per
// index: DLSS Super Resolution (in the present's filter) and DLSS-G (at present) share one frame's token, whichever
// asks first opens the frame.
static uint32_t g_frame_index = UINT32_MAX;
static sl::FrameToken* g_frame_token = nullptr;
RSLG_API void* rslg_frame(uint32_t index) {
    if (g_frame_token && index == g_frame_index) return g_frame_token;
    sl::FrameToken* token = nullptr;
    if (slGetNewFrameToken(token, &index) != sl::Result::eOk) return nullptr;
    slReflexSleep(*token);
    slPCLSetMarker(sl::PCLMarker::eSimulationStart, *token);
    slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *token);
    g_frame_index = index;
    g_frame_token = token;
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
static int set_constants(void* token, float mvec_scale_x, float mvec_scale_y, int reset, const sl::ViewportHandle& vp);

RSLG_API int rslg_constants(void* token, float mvec_scale_x, float mvec_scale_y, int reset) {
    return set_constants(token, mvec_scale_x, mvec_scale_y, reset, g_viewport);
}

static int set_constants(void* token, float mvec_scale_x, float mvec_scale_y, int reset, const sl::ViewportHandle& vp) {
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
    return rc(slSetConstants(c, *static_cast<sl::FrameToken*>(token), vp));
}

// Whether DLSS Super Resolution is supported on `pd` (0) or why not (1000 + sl::Result).
RSLG_API int rslg_dlss_supported(VkPhysicalDevice pd) {
    sl::AdapterInfo info{};
    info.vkPhysicalDevice = pd;
    return rc(slIsFeatureSupported(sl::kFeatureDLSS, info));
}

// DLSS Super Resolution options for the one viewport: `mode` an sl::DLSSMode value, the output size, `preset` an
// sl::DLSSPreset value for every mode (0 = default; 11 = K, 12 = L, 13 = M - the DLSS 4.5 transformer presets).
// The guest frame is LDR sRGB-encoded 8-bit.
RSLG_API int rslg_dlss(int mode, uint32_t out_w, uint32_t out_h, int preset) {
    sl::DLSSOptions o{};
    o.mode = sl::DLSSMode(mode);
    o.outputWidth = out_w;
    o.outputHeight = out_h;
    o.colorBuffersHDR = sl::Boolean::eFalse;
    sl::DLSSPreset p = sl::DLSSPreset(preset);
    o.dlaaPreset = o.qualityPreset = o.balancedPreset = o.performancePreset = o.ultraPerformancePreset =
        o.ultraQualityPreset = p;
    return rc(slDLSSSetOptions(g_viewport_sr, o));
}

// Upscale `in` into `out` with DLSS for this frame: tags (input colour, output colour, depth and motion vectors at
// the input size), the frame's constants (`reset` drops the history), then the evaluate recorded into `cb`.
RSLG_API int rslg_dlss_eval(void* token, VkCommandBuffer cb, const RslgImage* in, const RslgImage* out,
                            const RslgImage* depth, const RslgImage* mvec, int reset) {
    if (!token) return -1;
    sl::Resource i = resource(*in), o = resource(*out), d = resource(*depth), m = resource(*mvec);
    sl::Extent ein{0, 0, in->width, in->height};
    sl::Extent eout{0, 0, out->width, out->height};
    sl::ResourceTag tags[] = {
        sl::ResourceTag(&i, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilPresent, &ein),
        sl::ResourceTag(&o, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilPresent, &eout),
        sl::ResourceTag(&d, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &ein),
        sl::ResourceTag(&m, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &ein),
    };
    sl::FrameToken& t = *static_cast<sl::FrameToken*>(token);
    sl::CommandBuffer* slcb = reinterpret_cast<sl::CommandBuffer*>(cb);
    sl::Result r = slSetTagForFrame(t, g_viewport_sr, tags, 4, slcb);
    if (r != sl::Result::eOk) return rc(r);
    int c = set_constants(token, 1.0f / float(in->width), 1.0f / float(in->height), reset, g_viewport_sr);
    if (c) return c;
    const sl::BaseStructure* inputs[] = {&g_viewport_sr};
    return rc(slEvaluateFeature(sl::kFeatureDLSS, t, inputs, 1, slcb));
}

// DLSS-G's state: frames presented per host present and its status bits.
RSLG_API int rslg_state(uint32_t* presented, uint32_t* status, uint64_t* vram) {
    sl::DLSSGState s{};
    sl::DLSSGOptions o{};
    o.mode = sl::DLSSGMode::eOn;
    o.numFramesToGenerate = 1;
    // With the options SL also fills the VRAM estimate. numFramesActuallyPresented counts the frames presented
    // through DLSS-G's proxy since the previous call (not per present), so the caller's sampling interval is the
    // denominator; the rest says why generation is off when it is.
    sl::Result r = slDLSSGGetState(g_viewport, s, &o);
    std::fprintf(stderr,
                 "rslg_state r=%d presented_since_last=%u status=0x%x min_wh=%u max_generate=%u vsync_avail=%d "
                 "vram_mb=%llu\n",
                 int(r), s.numFramesActuallyPresented, uint32_t(s.status), s.minWidthOrHeight,
                 s.numFramesToGenerateMax, int(s.bIsVsyncSupportAvailable == sl::Boolean::eTrue),
                 (unsigned long long)(s.estimatedVRAMUsageInBytes >> 20));
    if (presented) *presented = s.numFramesActuallyPresented;
    if (status) *status = uint32_t(s.status);
    if (vram) *vram = s.estimatedVRAMUsageInBytes;
    return rc(r);
}

RSLG_API void rslg_shutdown() {
    if (g_inited) slShutdown();
    g_inited = false;
}
