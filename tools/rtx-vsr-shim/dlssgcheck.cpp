// dlssgcheck: does Streamline come up on this machine with DLSS-G? rslg_init, then a Vulkan instance and device
// through sl.interposer.dll (SL's proxies add what DLSS-G needs), then rslg_supported on the NVIDIA GPU.
// Usage: dlssgcheck <dir with rslg.dll and the SL DLLs>

#define NOMINMAX
#include <windows.h>

#include <vulkan/vulkan.h>

#include <cstdio>
#include <string>
#include <vector>

typedef int (*FnInit)(const wchar_t*, const wchar_t*, int);
typedef int (*FnSupported)(VkPhysicalDevice);

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string dir = argc > 1 ? argv[1] : "E:\\reims-lab\\dlssg";
    std::wstring wdir(dir.begin(), dir.end());
    HMODULE rslg = LoadLibraryExA((dir + "\\rslg.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!rslg) {
        std::printf("rslg.dll not loaded (%lu)\n", GetLastError());
        return 1;
    }
    auto init = (FnInit)GetProcAddress(rslg, "rslg_init");
    auto supported = (FnSupported)GetProcAddress(rslg, "rslg_supported");
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    int r = init(wdir.c_str(), tmp, 1);
    std::printf("rslg_init=%d\n", r);
    if (r) return 1;
    HMODULE sl = GetModuleHandleA("sl.interposer.dll");
    auto gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(sl, "vkGetInstanceProcAddr");
    auto createInstance = (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VkInstance inst;
    r = createInstance(&ii, nullptr, &inst);
    std::printf("vkCreateInstance (SL proxy)=%d\n", r);
    if (r) return 1;
    auto enumPd = (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    auto props = (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");
    auto createDevice = (PFN_vkCreateDevice)gipa(inst, "vkCreateDevice");
    uint32_t n = 0;
    enumPd(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    enumPd(inst, &n, pds.data());
    VkPhysicalDevice pd = nullptr;
    for (auto p : pds) {
        VkPhysicalDeviceProperties pr;
        props(p, &pr);
        if (pr.vendorID == 0x10de) {
            pd = p;
            std::printf("GPU: %s driver %u.%u\n", pr.deviceName, pr.driverVersion >> 22, (pr.driverVersion >> 14) & 0xff);
        }
    }
    if (!pd) return 1;
    std::printf("rslg_supported (before device)=%d\n", supported(pd));
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = 0;
    q.queueCount = 1;
    q.pQueuePriorities = &prio;
    const char* ext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &q;
    di.enabledExtensionCount = 1;
    di.ppEnabledExtensionNames = ext;
    VkDevice dev;
    r = createDevice(pd, &di, nullptr, &dev);
    std::printf("vkCreateDevice (SL proxy)=%d\n", r);
    std::printf("rslg_supported (after device)=%d\n", supported(pd));
    return 0;
}
