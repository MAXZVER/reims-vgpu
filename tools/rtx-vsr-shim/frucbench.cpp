// frucbench: wall time of one NVIDIA Optical Flow FRUC interpolation (NvOFFRUC.dll, CUDA cuDevicePtr ARGB frames)
// per frame size on the host GPU. Needs NvOFFRUC.dll and cudart64_110.dll (Optical Flow SDK) beside it and the
// SDK's NvOFFRUC.h on the include path. The CUDA driver API is taken from nvcuda.dll at run time.

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "NvOFFRUC.h"

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef CUresult (*FnInit)(unsigned);
typedef CUresult (*FnDeviceGet)(CUdevice*, int);
typedef CUresult (*FnCtxCreate)(CUcontext*, unsigned, CUdevice);
typedef CUresult (*FnMemAllocPitch)(CUdeviceptr*, size_t*, size_t, size_t, unsigned);
typedef CUresult (*FnMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*FnCtxSync)(void);

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = argc > 1 ? std::atoi(argv[1]) : 60;
    HMODULE cu = LoadLibraryA("nvcuda.dll");
    HMODULE fr = LoadLibraryA("NvOFFRUC.dll");
    if (!cu || !fr) {
        std::printf("nvcuda=%p NvOFFRUC=%p\n", cu, fr);
        return 1;
    }
    auto cuInit = (FnInit)GetProcAddress(cu, "cuInit");
    auto cuDeviceGet = (FnDeviceGet)GetProcAddress(cu, "cuDeviceGet");
    auto cuCtxCreate = (FnCtxCreate)GetProcAddress(cu, "cuCtxCreate_v2");
    auto cuMemAllocPitch = (FnMemAllocPitch)GetProcAddress(cu, "cuMemAllocPitch_v2");
    auto cuMemcpyHtoD = (FnMemcpyHtoD)GetProcAddress(cu, "cuMemcpyHtoD_v2");
    auto cuCtxSynchronize = (FnCtxSync)GetProcAddress(cu, "cuCtxSynchronize");
    auto create = (PtrToFuncNvOFFRUCCreate)GetProcAddress(fr, CreateProcName);
    auto reg = (PtrToFuncNvOFFRUCRegisterResource)GetProcAddress(fr, RegisterResourceProcName);
    auto process = (PtrToFuncNvOFFRUCProcess)GetProcAddress(fr, ProcessProcName);
    auto destroy = (PtrToFuncNvOFFRUCDestroy)GetProcAddress(fr, "NvOFFRUCDestroy");
    CUdevice dev;
    CUcontext ctx;
    if (cuInit(0) || cuDeviceGet(&dev, 0) || cuCtxCreate(&ctx, 0, dev)) {
        std::printf("cuda init failed\n");
        return 1;
    }
    struct Size {
        uint32_t w, h;
    } sizes[] = {{2560, 1080}, {3456, 1458}, {5120, 2160}};
    std::printf("%-12s %10s %10s\n", "size", "median ms", "p90 ms");
    for (auto s : sizes) {
        NvOFFRUC_CREATE_PARAM cp{};
        cp.uiWidth = s.w;
        cp.uiHeight = s.h;
        cp.pDevice = nullptr;
        cp.eResourceType = CudaResource;
        cp.eSurfaceFormat = ARGBSurface;
        cp.eCUDAResourceType = CudaResourceCuDevicePtr;
        NvOFFRUCHandle h = nullptr;
        std::printf("create %ux%u...\n", s.w, s.h);
        NvOFFRUC_STATUS st = create(&cp, &h);
        std::printf("create -> %d\n", int(st));
        if (st != NvOFFRUC_SUCCESS) {
            std::printf("create %ux%u -> %d\n", s.w, s.h, int(st));
            continue;
        }
        const int N = 4;  // 3 render frames in rotation + 1 interpolation target
        CUdeviceptr buf[N];
        size_t pitch = 0;
        for (auto& b : buf) cuMemAllocPitch(&b, &pitch, size_t(s.w) * 4, s.h, 16);
        NvOFFRUC_REGISTER_RESOURCE_PARAM rp{};
        for (int i = 0; i < N; i++) rp.pArrResource[i] = &buf[i];  // the address of the CUdeviceptr, as the SDK sample
        rp.uiCount = N;
        std::printf("register...\n");
        st = reg(h, &rp);
        std::printf("register -> %d pitch=%zu\n", int(st), pitch);
        if (st != NvOFFRUC_SUCCESS) {
            std::printf("register -> %d\n", int(st));
            return 1;
        }
        // A pattern that moves 8 px per frame: stripes and a block, enough texture for the flow to track.
        std::vector<uint32_t> host(size_t(pitch / 4) * s.h);
        std::vector<double> ms;
        for (int f = 0; f < iters + 5; f++) {
            for (uint32_t y = 0; y < s.h; y++)
                for (uint32_t x = 0; x < s.w; x++) {
                    uint32_t xx = x + f * 8;
                    bool block = (xx / 200) % 2 == (y / 200) % 2;
                    host[size_t(y) * (pitch / 4) + x] = block ? 0xff3060a0u : (0xff000000u | ((xx * 7) & 0xff) << 8);
                }
            CUdeviceptr* in = &buf[f % 3];
            cuMemcpyHtoD(*in, host.data(), pitch * s.h);
            cuCtxSynchronize();
            bool repeated = false;
            NvOFFRUC_PROCESS_IN_PARAMS ip{};
            ip.stFrameDataInput.pFrame = in;
            ip.stFrameDataInput.nTimeStamp = f * 2.0;
            ip.stFrameDataInput.nCuSurfacePitch = pitch;
            NvOFFRUC_PROCESS_OUT_PARAMS op{};
            op.stFrameDataOutput.pFrame = &buf[3];
            op.stFrameDataOutput.nTimeStamp = f * 2.0 - 1.0;
            op.stFrameDataOutput.nCuSurfacePitch = pitch;
            op.stFrameDataOutput.bHasFrameRepetitionOccurred = &repeated;
            auto t0 = std::chrono::steady_clock::now();
            if (f < 2) std::printf("process f=%d...\n", f);
            st = process(h, &ip, &op);
            cuCtxSynchronize();
            auto t1 = std::chrono::steady_clock::now();
            if (st != NvOFFRUC_SUCCESS) {
                std::printf("process -> %d\n", int(st));
                break;
            }
            if (f >= 5) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        destroy(h);
        if (ms.empty()) continue;
        std::sort(ms.begin(), ms.end());
        char label[32];
        std::snprintf(label, sizeof label, "%ux%u", s.w, s.h);
        std::printf("%-12s %10.3f %10.3f\n", label, ms[ms.size() / 2], ms[ms.size() * 9 / 10]);
    }
    return 0;
}
