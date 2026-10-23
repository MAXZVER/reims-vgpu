// ofbench: time of one NVIDIA Optical Flow (hardware OFA) execute per frame size, perf level and output grid, through
// the CUDA interface of nvofapi64.dll (shipped with the driver). The OFA engine is the same one a native Vulkan
// session drives, so this is the floor for computing motion between two guest frames before our own warp.
// Needs the Optical Flow SDK's NvOFInterface headers on the include path; the CUDA driver API comes from nvcuda.dll.

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda.h>  // the SDK's own copy (NvOFFRUCSample/cudart_header); only types are used
#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

typedef int (*FnInit)(unsigned);
typedef int (*FnDeviceGet)(int*, int);
typedef int (*FnCtxCreate)(CUcontext*, unsigned, int);
typedef int (*FnCtxSync)(void);
typedef NV_OF_STATUS(NVOFAPI* FnCreateInstance)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = argc > 1 ? std::atoi(argv[1]) : 100;
    HMODULE cu = LoadLibraryA("nvcuda.dll");
    HMODULE of = LoadLibraryA("nvofapi64.dll");
    if (!cu || !of) {
        std::printf("nvcuda=%p nvofapi64=%p\n", cu, of);
        return 1;
    }
    auto cuInit = (FnInit)GetProcAddress(cu, "cuInit");
    auto cuDeviceGet = (FnDeviceGet)GetProcAddress(cu, "cuDeviceGet");
    auto cuCtxCreate = (FnCtxCreate)GetProcAddress(cu, "cuCtxCreate_v2");
    auto cuCtxSynchronize = (FnCtxSync)GetProcAddress(cu, "cuCtxSynchronize");
    auto create = (FnCreateInstance)GetProcAddress(of, "NvOFAPICreateInstanceCuda");
    int dev;
    CUcontext ctx;
    if (cuInit(0) || cuDeviceGet(&dev, 0) || cuCtxCreate(&ctx, 0, dev) || !create) {
        std::printf("init failed\n");
        return 1;
    }
    NV_OF_CUDA_API_FUNCTION_LIST f{};
    if (create(NV_OF_API_VERSION, &f) != NV_OF_SUCCESS) {
        std::printf("NvOFAPICreateInstanceCuda failed\n");
        return 1;
    }
    struct Size {
        uint32_t w, h;
    } sizes[] = {{1728, 729}, {3456, 1458}, {5120, 2160}};
    struct Mode {
        NV_OF_PERF_LEVEL perf;
        NV_OF_OUTPUT_VECTOR_GRID_SIZE grid;
        const char* name;
    } modes[] = {{NV_OF_PERF_LEVEL_FAST, NV_OF_OUTPUT_VECTOR_GRID_SIZE_4, "fast/4"},
                 {NV_OF_PERF_LEVEL_MEDIUM, NV_OF_OUTPUT_VECTOR_GRID_SIZE_4, "medium/4"},
                 {NV_OF_PERF_LEVEL_SLOW, NV_OF_OUTPUT_VECTOR_GRID_SIZE_4, "slow/4"},
                 {NV_OF_PERF_LEVEL_FAST, NV_OF_OUTPUT_VECTOR_GRID_SIZE_1, "fast/1"}};
    std::printf("%-12s %-9s %10s %10s\n", "size", "mode", "median ms", "p90 ms");
    for (auto s : sizes) {
        for (auto m : modes) {
            NvOFHandle h = nullptr;
            if (f.nvCreateOpticalFlowCuda(ctx, &h) != NV_OF_SUCCESS) return 1;
            NV_OF_INIT_PARAMS ip{};
            ip.width = s.w;
            ip.height = s.h;
            ip.outGridSize = m.grid;
            ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
            ip.mode = NV_OF_MODE_OPTICALFLOW;
            ip.perfLevel = m.perf;
            ip.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
            ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
            NV_OF_STATUS st = f.nvOFInit(h, &ip);
            if (st != NV_OF_SUCCESS) {
                char err[512];
                uint32_t n = sizeof err;
                f.nvOFGetLastError(h, err, &n);
                std::printf("%ux%u %s: init %d %s\n", s.w, s.h, m.name, int(st), err);
                f.nvOFDestroy(h);
                continue;
            }
            NV_OF_BUFFER_DESCRIPTOR in{s.w, s.h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_ABGR8};
            NV_OF_BUFFER_DESCRIPTOR out{s.w / m.grid, s.h / m.grid, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2};
            NvOFGPUBufferHandle a, b, o;
            f.nvOFCreateGPUBufferCuda(h, &in, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &a);
            f.nvOFCreateGPUBufferCuda(h, &in, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &b);
            f.nvOFCreateGPUBufferCuda(h, &out, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &o);
            std::vector<double> ms;
            for (int i = 0; i < iters + 5; i++) {
                NV_OF_EXECUTE_INPUT_PARAMS ep{};
                ep.inputFrame = (i & 1) ? a : b;
                ep.referenceFrame = (i & 1) ? b : a;
                NV_OF_EXECUTE_OUTPUT_PARAMS op{};
                op.outputBuffer = o;
                auto t0 = std::chrono::steady_clock::now();
                st = f.nvOFExecute(h, &ep, &op);
                cuCtxSynchronize();
                auto t1 = std::chrono::steady_clock::now();
                if (st != NV_OF_SUCCESS) {
                    std::printf("execute %d\n", int(st));
                    break;
                }
                if (i >= 5) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            f.nvOFDestroyGPUBufferCuda(a);
            f.nvOFDestroyGPUBufferCuda(b);
            f.nvOFDestroyGPUBufferCuda(o);
            f.nvOFDestroy(h);
            if (ms.empty()) continue;
            std::sort(ms.begin(), ms.end());
            char label[32];
            std::snprintf(label, sizeof label, "%ux%u", s.w, s.h);
            std::printf("%-12s %-9s %10.3f %10.3f\n", label, m.name, ms[ms.size() / 2], ms[ms.size() * 9 / 10]);
        }
    }
    return 0;
}
