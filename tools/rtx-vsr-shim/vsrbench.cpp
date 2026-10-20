// vsrbench: GPU time of one NGX VSR evaluate per (input size, output size, quality), measured with Vulkan
// timestamps on the host GPU. Uses rvsr.dll (same directory) exactly as the device does.
// Usage: vsrbench [iterations]   (prints a table: in -> out, quality, median / p90 ms)

#include <vulkan/vulkan.h>
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                 \
    do {                                                                         \
        VkResult r_ = (x);                                                       \
        if (r_ != VK_SUCCESS) {                                                  \
            std::printf("vk error %d at %s:%d\n", int(r_), __FILE__, __LINE__);  \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

typedef int (*FnExt)(uint32_t*, const char* const**);
typedef int (*FnDevExt)(VkInstance, VkPhysicalDevice, uint32_t*, const char* const**);
typedef int (*FnInit)(VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr,
                      const wchar_t*, int*);
typedef int (*FnCreate)(VkDevice, VkCommandBuffer);
typedef int (*FnEval)(VkCommandBuffer, VkImage, VkImageView, VkFormat, uint32_t, uint32_t, VkImage, VkImageView,
                      VkFormat, uint32_t, uint32_t, const uint32_t*, const uint32_t*, int);

// Evaluates per submission: back to back, so the GPU stays clocked up as it is under a running guest.
static const int BATCH = 8;
static VkDevice dev;
static VkPhysicalDevice pd;
static VkPhysicalDeviceMemoryProperties memprops;

struct Img {
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
    uint32_t w, h;
};

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    std::printf("no memory type\n");
    std::exit(1);
}

static Img make(uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    Img m{};
    m.w = w;
    m.h = h;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    CHECK(vkCreateImage(dev, &ici, nullptr, &m.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, m.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(vkAllocateMemory(dev, &mai, nullptr, &m.mem));
    CHECK(vkBindImageMemory(dev, m.image, m.mem, 0));
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = m.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CHECK(vkCreateImageView(dev, &vci, nullptr, &m.view));
    return m;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
}

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 60;
    HMODULE lib = LoadLibraryA("rvsr.dll");
    if (!lib) {
        std::printf("rvsr.dll not found\n");
        return 1;
    }
    auto instExt = (FnExt)GetProcAddress(lib, "rvsr_instance_extensions");
    auto devExt = (FnDevExt)GetProcAddress(lib, "rvsr_device_extensions");
    auto init = (FnInit)GetProcAddress(lib, "rvsr_init");
    auto create = (FnCreate)GetProcAddress(lib, "rvsr_create");
    auto eval = (FnEval)GetProcAddress(lib, "rvsr_evaluate");

    uint32_t nie = 0;
    const char* const* ie = nullptr;
    instExt(&nie, &ie);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    ii.enabledExtensionCount = nie;
    ii.ppEnabledExtensionNames = ie;
    VkInstance inst;
    CHECK(vkCreateInstance(&ii, nullptr, &inst));
    uint32_t npd = 0;
    vkEnumeratePhysicalDevices(inst, &npd, nullptr);
    std::vector<VkPhysicalDevice> pds(npd);
    vkEnumeratePhysicalDevices(inst, &npd, pds.data());
    for (auto p : pds) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(p, &pr);
        if (pr.vendorID == 0x10de) {
            pd = p;
            std::printf("GPU: %s\n", pr.deviceName);
        }
    }
    if (!pd) return 1;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    vkGetPhysicalDeviceMemoryProperties(pd, &memprops);
    uint32_t nde = 0;
    const char* const* de = nullptr;
    devExt(inst, pd, &nde, &de);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0;  // NVIDIA family 0: graphics + compute
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nde;
    dci.ppEnabledExtensionNames = de;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q;
    vkGetDeviceQueue(dev, 0, 0, &q);
    int avail = 0;
    int r = init(inst, pd, dev, vkGetInstanceProcAddr, vkGetDeviceProcAddr, L".", &avail);
    std::printf("ngx init=%#x available=%d\n", r, avail);
    if (r || !avail) return 1;

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 2;
    VkQueryPool qp;
    CHECK(vkCreateQueryPool(dev, &qpi, nullptr, &qp));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
    auto submit = [&]() {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        CHECK(vkQueueSubmit(q, 1, &si, fence));
        CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        CHECK(vkResetFences(dev, 1, &fence));
    };
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cb, &bi));
    r = create(dev, cb);
    CHECK(vkEndCommandBuffer(cb));
    submit();
    std::printf("ngx create=%#x\n", r);
    if (r) return 1;

    struct Case {
        uint32_t iw, ih, ow, oh;
    } cases[] = {
        {3456, 1458, 5120, 2160}, {4096, 1728, 5120, 2160}, {2560, 1080, 5120, 2160}, {1536, 648, 1920, 810}};
    VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    std::printf("%-22s %-8s %10s %10s\n", "in -> out", "quality", "median ms", "p90 ms");
    for (auto c : cases) {
        Img in = make(c.iw, c.ih, u), out = make(c.ow, c.oh, u);
        for (int quality = 1; quality <= 4; quality++) {
            std::vector<double> ms;
            for (int i = 0; i < iters + 5; i++) {
                CHECK(vkResetCommandBuffer(cb, 0));
                CHECK(vkBeginCommandBuffer(cb, &bi));
                barrier(cb, in.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                barrier(cb, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
                vkCmdResetQueryPool(cb, qp, 0, 2);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                uint32_t ir[4] = {0, 0, c.iw, c.ih}, orr[4] = {0, 0, c.ow, c.oh};
                int e = 0;
                for (int k = 0; k < BATCH && !e; k++) {
                    e = eval(cb, in.image, in.view, VK_FORMAT_R8G8B8A8_UNORM, c.iw, c.ih, out.image, out.view,
                             VK_FORMAT_R8G8B8A8_UNORM, c.ow, c.oh, ir, orr, quality);
                    barrier(cb, out.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
                }
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
                CHECK(vkEndCommandBuffer(cb));
                if (e) {
                    std::printf("evaluate %#x\n", e);
                    return 1;
                }
                submit();
                uint64_t ts[2];
                vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                if (i >= 5) ms.push_back(double(ts[1] - ts[0]) * props.limits.timestampPeriod / 1e6 / BATCH);
            }
            std::sort(ms.begin(), ms.end());
            char label[64];
            std::snprintf(label, sizeof label, "%ux%u -> %ux%u", c.iw, c.ih, c.ow, c.oh);
            std::printf("%-22s %-8d %10.3f %10.3f\n", label, quality, ms[ms.size() / 2], ms[ms.size() * 9 / 10]);
        }
    }
    return 0;
}
