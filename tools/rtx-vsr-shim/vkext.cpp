// vkext: lists the NVIDIA GPU's device extensions matching a substring (default "optical").
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <vector>
int main(int argc, char** argv) {
    const char* want = argc > 1 ? argv[1] : "optical";
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VkInstance inst;
    if (vkCreateInstance(&ii, nullptr, &inst)) return 1;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    for (auto p : pds) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(p, &pr);
        uint32_t m = 0;
        vkEnumerateDeviceExtensionProperties(p, nullptr, &m, nullptr);
        std::vector<VkExtensionProperties> e(m);
        vkEnumerateDeviceExtensionProperties(p, nullptr, &m, e.data());
        std::printf("%s (driver %u)\n", pr.deviceName, pr.driverVersion);
        for (auto& x : e)
            if (strstr(x.extensionName, want)) std::printf("  %s v%u\n", x.extensionName, x.specVersion);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> q(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, q.data());
        for (uint32_t i = 0; i < qn; i++) std::printf("  queue family %u flags=%#x count=%u\n", i, q[i].queueFlags, q[i].queueCount);
    }
    return 0;
}
