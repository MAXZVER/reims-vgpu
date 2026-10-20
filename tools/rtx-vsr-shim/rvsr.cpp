// rvsr: a C ABI over NVIDIA RTX Video Super Resolution (NGX VSR, Vulkan) for reims-vgpu's host present.
//
// The device is built with the MinGW toolchain; the RTX Video SDK ships MSVC static libraries. This DLL is built
// with MSVC against the SDK (which is not part of this repository - see build.bat) and loaded at run time only when
// the present asks for VSR, so nothing else links it.
//
// Every entry returns 0 on success or the NVSDK_NGX_Result / a negative local code on failure. All Vulkan objects
// are the caller's; the caller records the returned work into its own command buffers and orders it with its own
// barriers (inputs SHADER_READ / GENERAL, output GENERAL with STORAGE usage, as NGX requires).

#include <vulkan/vulkan.h>

#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vsr_vk.h>

#include <cstdint>
#include <cstring>
#include <vector>

#define RVSR_API extern "C" __declspec(dllexport)

namespace {

NVSDK_NGX_Parameter* g_params = nullptr;
NVSDK_NGX_Handle* g_feature = nullptr;
bool g_inited = false;
std::vector<const char*> g_instance_names;
std::vector<const char*> g_device_names;

NVSDK_NGX_FeatureDiscoveryInfo discovery(const wchar_t* data_path) {
    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = NVSDK_NGX_Feature_VSR;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    info.Identifier.v.ApplicationId = 0;
    info.ApplicationDataPath = data_path ? data_path : L".";
    info.FeatureInfo = nullptr;
    return info;
}

NVSDK_NGX_Resource_VK resource(VkImage image, VkImageView view, VkFormat format, uint32_t w, uint32_t h, bool rw) {
    NVSDK_NGX_Resource_VK r{};
    r.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
    r.ReadWrite = rw;
    r.Resource.ImageViewInfo.ImageView = view;
    r.Resource.ImageViewInfo.Image = image;
    r.Resource.ImageViewInfo.SubresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    r.Resource.ImageViewInfo.Format = format;
    r.Resource.ImageViewInfo.Width = w;
    r.Resource.ImageViewInfo.Height = h;
    return r;
}

}  // namespace

// Instance extensions VSR needs, enabled by the caller before vkCreateInstance. The names stay valid until unload.
RVSR_API int rvsr_instance_extensions(uint32_t* count, const char* const** names) {
    NVSDK_NGX_FeatureDiscoveryInfo info = discovery(nullptr);
    uint32_t n = 0;
    VkExtensionProperties* props = nullptr;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &n, &props);
    if (NVSDK_NGX_FAILED(r)) return int(r);
    g_instance_names.clear();
    for (uint32_t i = 0; i < n; i++) g_instance_names.push_back(props[i].extensionName);
    *count = n;
    *names = g_instance_names.data();
    return 0;
}

// Device extensions VSR needs on this physical device, enabled by the caller before vkCreateDevice.
RVSR_API int rvsr_device_extensions(VkInstance instance, VkPhysicalDevice pd, uint32_t* count, const char* const** names) {
    NVSDK_NGX_FeatureDiscoveryInfo info = discovery(nullptr);
    uint32_t n = 0;
    VkExtensionProperties* props = nullptr;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, pd, &info, &n, &props);
    if (NVSDK_NGX_FAILED(r)) return int(r);
    g_device_names.clear();
    for (uint32_t i = 0; i < n; i++) g_device_names.push_back(props[i].extensionName);
    *count = n;
    *names = g_device_names.data();
    return 0;
}

// NGX on the caller's device, and whether this GPU and driver offer VSR. `available` is 1 when they do.
RVSR_API int rvsr_init(VkInstance instance, VkPhysicalDevice pd, VkDevice device, PFN_vkGetInstanceProcAddr gipa,
                       PFN_vkGetDeviceProcAddr gdpa, const wchar_t* data_path, int* available) {
    *available = 0;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init(0, data_path ? data_path : L".", instance, pd, device, gipa, gdpa,
                                               nullptr, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) return int(r);
    g_inited = true;
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&g_params);
    if (NVSDK_NGX_FAILED(r)) return int(r);
    int ok = 0;
    r = g_params->Get(NVSDK_NGX_Parameter_VSR_Available, &ok);
    if (NVSDK_NGX_FAILED(r)) return int(r);
    *available = ok;
    return 0;
}

// Record the feature's creation into `cb` (submitted and completed by the caller before the first evaluate).
RVSR_API int rvsr_create(VkDevice device, VkCommandBuffer cb) {
    if (!g_params) return -1;
    size_t scratch = 0;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetScratchBufferSize(NVSDK_NGX_Feature_VSR, g_params, &scratch);
    if (NVSDK_NGX_FAILED(r)) return int(r);
    if (scratch != 0) return -2;  // VSR is documented to need none; a request would need a buffer we do not pass
    NVSDK_NGX_Feature_Create_Params create{};
    r = NGX_VULKAN_CREATE_VSR_EXT1(device, cb, 1, 1, &g_feature, g_params, &create);
    return NVSDK_NGX_FAILED(r) ? int(r) : 0;
}

// Record one super-resolution pass into `cb`: `in` (sampled) rect -> `out` (storage) rect. Quality 0..4.
RVSR_API int rvsr_evaluate(VkCommandBuffer cb, VkImage in_image, VkImageView in_view, VkFormat in_format,
                           uint32_t in_w, uint32_t in_h, VkImage out_image, VkImageView out_view, VkFormat out_format,
                           uint32_t out_w, uint32_t out_h, const uint32_t in_rect[4], const uint32_t out_rect[4],
                           int quality) {
    if (!g_feature) return -1;
    NVSDK_NGX_Resource_VK in = resource(in_image, in_view, in_format, in_w, in_h, false);
    NVSDK_NGX_Resource_VK out = resource(out_image, out_view, out_format, out_w, out_h, true);
    NVSDK_NGX_VK_VSR_Eval_Params p{};
    p.pInput = &in;
    p.pOutput = &out;
    p.InputSubrectBase.X = in_rect[0];
    p.InputSubrectBase.Y = in_rect[1];
    p.InputSubrectSize.Width = in_rect[2];
    p.InputSubrectSize.Height = in_rect[3];
    p.OutputSubrectBase.X = out_rect[0];
    p.OutputSubrectBase.Y = out_rect[1];
    p.OutputSubrectSize.Width = out_rect[2];
    p.OutputSubrectSize.Height = out_rect[3];
    p.QualityLevel = NVSDK_NGX_VSR_QualityLevel(quality < 0 ? 0 : quality > 4 ? 4 : quality);
    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_VSR_EXT(cb, g_feature, g_params, &p);
    return NVSDK_NGX_FAILED(r) ? int(r) : 0;
}

// Release everything (the caller's device must be idle).
RVSR_API void rvsr_shutdown(VkDevice device) {
    if (g_feature) NVSDK_NGX_VULKAN_ReleaseFeature(g_feature);
    g_feature = nullptr;
    if (g_params) NVSDK_NGX_VULKAN_DestroyParameters(g_params);
    g_params = nullptr;
    if (g_inited) NVSDK_NGX_VULKAN_Shutdown1(device);
    g_inited = false;
}
