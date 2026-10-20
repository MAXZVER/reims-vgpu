// dlssgwin: does DLSS Frame Generation actually generate on this machine, outside the VM? A focused Win32 window,
// a Vulkan instance/device/swapchain through sl.interposer.dll, and per frame exactly the host's sequence
// (present_dlssg.rs): rslg_frame -> record (clear + a moving box) -> rslg_tag (swapchain image as hudless colour,
// zero depth and motion vectors) -> rslg_constants -> marker RenderSubmitStart; submit with RenderSubmitEnd/
// PresentStart, vkQueuePresentKHR, PresentEnd. Base rate capped at `fps`; every second it prints rslg_state
// (numFramesActuallyPresented counts frames presented since the previous call: ~2x the base frames when DLSS-G
// generates) and whether the window is the foreground one.
// Usage: dlssgwin [dir with rslg.dll and the SL DLLs] [seconds] [fps] [mailbox|fifo|immediate]

#define NOMINMAX
#include <windows.h>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct RslgImage {
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    uint32_t layout, format, width, height, usage;
};
typedef int (*FnInit)(const wchar_t*, const wchar_t*, int);
typedef int (*FnSupported)(VkPhysicalDevice);
typedef int (*FnInt)(int);
typedef void* (*FnFrame)(uint32_t);
typedef int (*FnMarker)(void*, uint32_t);
typedef int (*FnTag)(void*, VkCommandBuffer, const RslgImage*, const RslgImage*, const RslgImage*, uint32_t, uint32_t);
typedef int (*FnConstants)(void*, float, float, int);
typedef int (*FnState)(uint32_t*, uint32_t*, uint64_t*);

static PFN_vkGetInstanceProcAddr gipa;
static PFN_vkGetDeviceProcAddr gdpa;
#define IFN(inst, name) auto name = (PFN_##name)gipa(inst, #name)
#define DFN(dev, name) auto name = (PFN_##name)gdpa(dev, #name)
#define CHECK(x)                                                         \
    do {                                                                 \
        VkResult r_ = (x);                                               \
        if (r_ != VK_SUCCESS) {                                          \
            std::printf("%s failed: %d (line %d)\n", #x, int(r_), __LINE__); \
            return 1;                                                    \
        }                                                                \
    } while (0)

static bool g_quit = false;
static LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CLOSE || m == WM_DESTROY) {
        g_quit = true;
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}

static void take_foreground(HWND hwnd) {
    // A background process may not take the foreground; an Alt tap makes this the last input owner first.
    keybd_event(VK_MENU, 0, 0, 0);
    keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0);
    ShowWindow(hwnd, SW_SHOWNORMAL);
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string dir = argc > 1 ? argv[1] : "E:\\reims-lab\\dlssg";
    int seconds = argc > 2 ? std::atoi(argv[2]) : 20;
    int fps = argc > 3 ? std::atoi(argv[3]) : 60;
    std::string mode = argc > 4 ? argv[4] : "mailbox";
    std::wstring wdir(dir.begin(), dir.end());
    HMODULE rslg = LoadLibraryExA((dir + "\\rslg.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!rslg) {
        std::printf("rslg.dll not loaded (%lu)\n", GetLastError());
        return 1;
    }
    auto init = (FnInit)GetProcAddress(rslg, "rslg_init");
    auto supported = (FnSupported)GetProcAddress(rslg, "rslg_supported");
    auto reflex = (FnInt)GetProcAddress(rslg, "rslg_reflex");
    auto dlssg = (FnInt)GetProcAddress(rslg, "rslg_dlssg");
    auto frame = (FnFrame)GetProcAddress(rslg, "rslg_frame");
    auto marker = (FnMarker)GetProcAddress(rslg, "rslg_marker");
    auto tag = (FnTag)GetProcAddress(rslg, "rslg_tag");
    auto constants = (FnConstants)GetProcAddress(rslg, "rslg_constants");
    auto state = (FnState)GetProcAddress(rslg, "rslg_state");
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    std::wstring logs = std::wstring(tmp) + L"dlssgwin";
    CreateDirectoryW(logs.c_str(), nullptr);
    int r = init(wdir.c_str(), logs.c_str(), 1);
    std::printf("rslg_init=%d (SL log in %%TEMP%%\\dlssgwin)\n", r);
    if (r) return 1;
    HMODULE sl = GetModuleHandleA("sl.interposer.dll");
    gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(sl, "vkGetInstanceProcAddr");

    // Window.
    const uint32_t W = 1920, H = 1080;
    WNDCLASSW wc{};
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"dlssgwin";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT rc{0, 0, LONG(W), LONG(H)};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"dlssgwin", L"dlssgwin (DLSS-G test)", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 40, 40,
                              rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);
    take_foreground(hwnd);

    // Instance, surface, device.
    IFN(nullptr, vkCreateInstance);
    const char* iext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    ii.enabledExtensionCount = 2;
    ii.ppEnabledExtensionNames = iext;
    VkInstance inst;
    CHECK(vkCreateInstance(&ii, nullptr, &inst));
    IFN(inst, vkEnumeratePhysicalDevices);
    IFN(inst, vkGetPhysicalDeviceProperties);
    IFN(inst, vkGetPhysicalDeviceMemoryProperties);
    IFN(inst, vkCreateWin32SurfaceKHR);
    IFN(inst, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    IFN(inst, vkGetPhysicalDeviceSurfacePresentModesKHR);
    IFN(inst, vkCreateDevice);
    gdpa = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (auto p : pds) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(p, &pr);
        if (pr.vendorID == 0x10de) {
            pd = p;
            std::printf("gpu: %s\n", pr.deviceName);
            break;
        }
    }
    if (!pd) {
        std::printf("no NVIDIA GPU\n");
        return 1;
    }
    VkWin32SurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    si.hinstance = wc.hInstance;
    si.hwnd = hwnd;
    VkSurfaceKHR surface;
    CHECK(vkCreateWin32SurfaceKHR(inst, &si, nullptr, &surface));
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = 0;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    const char* dext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = 1;
    di.ppEnabledExtensionNames = dext;
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &di, nullptr, &dev));
    DFN(dev, vkGetDeviceQueue);
    DFN(dev, vkCreateSwapchainKHR);
    DFN(dev, vkGetSwapchainImagesKHR);
    DFN(dev, vkAcquireNextImageKHR);
    DFN(dev, vkQueuePresentKHR);
    DFN(dev, vkQueueSubmit);
    DFN(dev, vkCreateImage);
    DFN(dev, vkGetImageMemoryRequirements);
    DFN(dev, vkAllocateMemory);
    DFN(dev, vkBindImageMemory);
    DFN(dev, vkCreateImageView);
    DFN(dev, vkCreateBuffer);
    DFN(dev, vkGetBufferMemoryRequirements);
    DFN(dev, vkBindBufferMemory);
    DFN(dev, vkMapMemory);
    DFN(dev, vkCreateCommandPool);
    DFN(dev, vkAllocateCommandBuffers);
    DFN(dev, vkBeginCommandBuffer);
    DFN(dev, vkEndCommandBuffer);
    DFN(dev, vkResetCommandBuffer);
    DFN(dev, vkCmdPipelineBarrier);
    DFN(dev, vkCmdClearColorImage);
    DFN(dev, vkCmdCopyBufferToImage);
    DFN(dev, vkCreateFence);
    DFN(dev, vkWaitForFences);
    DFN(dev, vkResetFences);
    DFN(dev, vkCreateSemaphore);
    DFN(dev, vkDeviceWaitIdle);
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);

    // Swapchain.
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &n, modes.data());
    VkPresentModeKHR want = mode == "fifo" ? VK_PRESENT_MODE_FIFO_KHR
                            : mode == "immediate" ? VK_PRESENT_MODE_IMMEDIATE_KHR
                                                  : VK_PRESENT_MODE_MAILBOX_KHR;
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    for (auto m : modes)
        if (m == want) pm = m;
    std::printf("present mode %d (wanted %s)\n", int(pm), mode.c_str());
    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface;
    sci.minImageCount = std::max(3u, caps.minImageCount);
    sci.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sci.imageExtent = {W, H};
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = pm;
    sci.clipped = VK_TRUE;
    VkSwapchainKHR swap;
    CHECK(vkCreateSwapchainKHR(dev, &sci, nullptr, &swap));
    vkGetSwapchainImagesKHR(dev, swap, &n, nullptr);
    std::vector<VkImage> imgs(n);
    vkGetSwapchainImagesKHR(dev, swap, &n, imgs.data());
    std::vector<VkImageView> views(n);
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (uint32_t i = 0; i < n; i++) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = imgs[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_B8G8R8A8_UNORM;
        vi.subresourceRange = range;
        CHECK(vkCreateImageView(dev, &vi, nullptr, &views[i]));
    }

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0;
    };

    // Depth and motion vectors: zero, as the host tags them.
    const VkImageUsageFlags aux_usage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    auto make = [&](VkFormat f, RslgImage& out) -> bool {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = f;
        ci.extent = {W, H, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = aux_usage;
        if (vkCreateImage(dev, &ci, nullptr, &out.image)) return false;
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, out.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(dev, &ai, nullptr, &out.memory)) return false;
        vkBindImageMemory(dev, out.image, out.memory, 0);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = out.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = f;
        vi.subresourceRange = range;
        if (vkCreateImageView(dev, &vi, nullptr, &out.view)) return false;
        out.layout = VK_IMAGE_LAYOUT_GENERAL;
        out.format = f;
        out.width = W;
        out.height = H;
        out.usage = aux_usage;
        return true;
    };
    RslgImage depth{}, mvec{};
    if (!make(VK_FORMAT_R32_SFLOAT, depth) || !make(VK_FORMAT_R16G16_SFLOAT, mvec)) {
        std::printf("aux images failed\n");
        return 1;
    }

    // The moving box: a host buffer of white texels copied at a new x each frame.
    const uint32_t BW = 240, BH = 240;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = BW * BH * 4;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBuffer box;
    CHECK(vkCreateBuffer(dev, &bi, nullptr, &box));
    VkMemoryRequirements breq;
    vkGetBufferMemoryRequirements(dev, box, &breq);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bai.allocationSize = breq.size;
    bai.memoryTypeIndex =
        memtype(breq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem;
    CHECK(vkAllocateMemory(dev, &bai, nullptr, &bmem));
    vkBindBufferMemory(dev, box, bmem, 0);
    void* bp;
    vkMapMemory(dev, bmem, 0, bi.size, 0, &bp);
    std::memset(bp, 0xff, bi.size);

    // Frames in flight.
    const int F = 2;
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = 0;
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(dev, &pi, nullptr, &pool));
    VkCommandBuffer cbs[F];
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = F;
    CHECK(vkAllocateCommandBuffers(dev, &cai, cbs));
    VkFence fences[F];
    VkSemaphore acquired[F], rendered[F];
    for (int i = 0; i < F; i++) {
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        CHECK(vkCreateFence(dev, &fi, nullptr, &fences[i]));
        VkSemaphoreCreateInfo s{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        CHECK(vkCreateSemaphore(dev, &s, nullptr, &acquired[i]));
        CHECK(vkCreateSemaphore(dev, &s, nullptr, &rendered[i]));
    }

    auto barrier = [&](VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
                       VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = to;
        b.srcAccessMask = sa;
        b.dstAccessMask = da;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = range;
        vkCmdPipelineBarrier(cb, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    bool started = false, cleared = false;
    LARGE_INTEGER freq, t0, last, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    last = t0;
    double period = 1.0 / std::max(1, fps);
    uint32_t frames = 0, frames_at_last = 0;
    double next = 0;
    while (!g_quit) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        QueryPerformanceCounter(&now);
        double t = double(now.QuadPart - t0.QuadPart) / freq.QuadPart;
        if (t > seconds) break;
        if (t < next) {
            double wait = next - t;
            if (wait > 0.002) Sleep(DWORD((wait - 0.001) * 1000));
            continue;
        }
        next = std::max(next + period, t);
        if (!started) {
            started = true;
            int s = supported(pd), rf = reflex(1), on = dlssg(1);
            std::printf("start supported=%d reflex=%d dlssg=%d\n", s, rf, on);
        }
        int f = frames % F;
        vkWaitForFences(dev, 1, &fences[f], VK_TRUE, UINT64_MAX);
        vkResetFences(dev, 1, &fences[f]);
        uint32_t ix;
        VkResult ar = vkAcquireNextImageKHR(dev, swap, UINT64_MAX, acquired[f], VK_NULL_HANDLE, &ix);
        if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
            std::printf("acquire %d\n", int(ar));
            break;
        }
        void* token = frame(frames);
        VkCommandBuffer cb = cbs[f];
        vkResetCommandBuffer(cb, 0);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &cbi);
        if (!cleared) {
            for (VkImage a : {depth.image, mvec.image}) {
                barrier(cb, a, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                VkClearColorValue z{};
                vkCmdClearColorImage(cb, a, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &range);
                barrier(cb, a, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
            cleared = true;
        }
        VkImage img = imgs[ix];
        barrier(cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkClearColorValue bg{};
        bg.float32[0] = 0.10f;
        bg.float32[1] = 0.12f;
        bg.float32[2] = 0.20f;
        bg.float32[3] = 1.0f;
        vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &bg, 1, &range);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        uint32_t span = W - BW;
        uint32_t x = (frames * 8) % (2 * span);
        if (x > span) x = 2 * span - x;
        copy.imageOffset = {int32_t(x), int32_t((H - BH) / 2), 0};
        copy.imageExtent = {BW, BH, 1};
        vkCmdCopyBufferToImage(cb, box, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        if (token) {
            RslgImage hudless{img, views[ix], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                              VK_FORMAT_B8G8R8A8_UNORM, W, H,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
            int tg = tag(token, cb, &hudless, &depth, &mvec, W, H);
            int cs = constants(token, 1.0f / W, 1.0f / H, 0);
            if ((tg || cs) && frames < 3) std::printf("tag=%d constants=%d\n", tg, cs);
            marker(token, 2);  // eRenderSubmitStart
        }
        vkEndCommandBuffer(cb);
        if (token) {
            marker(token, 3);  // eRenderSubmitEnd
            marker(token, 4);  // ePresentStart
        }
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.waitSemaphoreCount = 1;
        sub.pWaitSemaphores = &acquired[f];
        sub.pWaitDstStageMask = &stage;
        sub.commandBufferCount = 1;
        sub.pCommandBuffers = &cb;
        sub.signalSemaphoreCount = 1;
        sub.pSignalSemaphores = &rendered[f];
        CHECK(vkQueueSubmit(queue, 1, &sub, fences[f]));
        VkPresentInfoKHR pinfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pinfo.waitSemaphoreCount = 1;
        pinfo.pWaitSemaphores = &rendered[f];
        pinfo.swapchainCount = 1;
        pinfo.pSwapchains = &swap;
        pinfo.pImageIndices = &ix;
        VkResult pr = vkQueuePresentKHR(queue, &pinfo);
        if (token) marker(token, 5);  // ePresentEnd
        if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) {
            std::printf("present %d\n", int(pr));
            break;
        }
        frames++;
        double since = double(now.QuadPart - last.QuadPart) / freq.QuadPart;
        if (since >= 1.0) {
            uint32_t presented = 0, status = 0;
            uint64_t vram = 0;
            int sr = state(&presented, &status, &vram);
            std::printf("t=%.0f base_fps=%.1f sl=%d presented_since_last=%u (x%.2f of base) status=0x%x vram_mb=%llu "
                        "foreground=%d\n",
                        t, (frames - frames_at_last) / since, sr, presented,
                        double(presented) / std::max(1u, frames - frames_at_last), status,
                        (unsigned long long)(vram >> 20), int(GetForegroundWindow() == hwnd));
            frames_at_last = frames;
            last = now;
            if (GetForegroundWindow() != hwnd) take_foreground(hwnd);
        }
    }
    vkDeviceWaitIdle(dev);
    std::printf("frames=%u\n", frames);
    DestroyWindow(hwnd);
    return 0;
}
