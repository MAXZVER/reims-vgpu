// ofpair: hardware optical flow (nvofapi64, CUDA interface) between two RGBA8 frames on disk, forward and backward,
// written as int16 pairs (S10.5 fixed point, one vector per grid cell). For checking a frame interpolation offline.
// Usage: ofpair a.rgba b.rgba width height grid(1|2|4) perf(5|10|20) out_fwd.flow out_bwd.flow

#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

typedef CUresult (*FnInit)(unsigned);
typedef CUresult (*FnDeviceGet)(CUdevice*, int);
typedef CUresult (*FnCtxCreate)(CUcontext*, unsigned, CUdevice);
typedef CUresult (*FnCtxSync)(void);
typedef CUresult (*FnHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*FnDtoH)(void*, CUdeviceptr, size_t);
typedef NV_OF_STATUS(NVOFAPI* FnCreateInstance)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);

static std::vector<uint8_t> load(const char* path, size_t n) {
    std::vector<uint8_t> v(n);
    FILE* f = std::fopen(path, "rb");
    if (!f || std::fread(v.data(), 1, n, f) != n) {
        std::printf("cannot read %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 9) {
        std::printf("usage: ofpair a.rgba b.rgba w h grid perf fwd.flow bwd.flow\n");
        return 1;
    }
    uint32_t w = std::atoi(argv[3]), h = std::atoi(argv[4]), grid = std::atoi(argv[5]);
    int perf = std::atoi(argv[6]);
    HMODULE cu = LoadLibraryA("nvcuda.dll"), of = LoadLibraryA("nvofapi64.dll");
    auto cuInit = (FnInit)GetProcAddress(cu, "cuInit");
    auto cuDeviceGet = (FnDeviceGet)GetProcAddress(cu, "cuDeviceGet");
    auto cuCtxCreate = (FnCtxCreate)GetProcAddress(cu, "cuCtxCreate_v2");
    auto cuCtxSynchronize = (FnCtxSync)GetProcAddress(cu, "cuCtxSynchronize");
    auto cuMemcpyHtoD = (FnHtoD)GetProcAddress(cu, "cuMemcpyHtoD_v2");
    auto cuMemcpyDtoH = (FnDtoH)GetProcAddress(cu, "cuMemcpyDtoH_v2");
    auto create = (FnCreateInstance)GetProcAddress(of, "NvOFAPICreateInstanceCuda");
    CUdevice dev;
    CUcontext ctx;
    if (cuInit(0) || cuDeviceGet(&dev, 0) || cuCtxCreate(&ctx, 0, dev)) return 1;
    NV_OF_CUDA_API_FUNCTION_LIST f{};
    if (create(NV_OF_API_VERSION, &f) != NV_OF_SUCCESS) return 1;
    std::vector<uint8_t> a = load(argv[1], size_t(w) * h * 4), b = load(argv[2], size_t(w) * h * 4);
    for (int dir = 0; dir < 2; dir++) {
        NvOFHandle hof = nullptr;
        f.nvCreateOpticalFlowCuda(ctx, &hof);
        NV_OF_INIT_PARAMS ip{};
        ip.width = w;
        ip.height = h;
        ip.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE(grid);
        ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        ip.mode = NV_OF_MODE_OPTICALFLOW;
        ip.perfLevel = NV_OF_PERF_LEVEL(perf);
        ip.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
        ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
        if (f.nvOFInit(hof, &ip) != NV_OF_SUCCESS) {
            std::printf("init failed\n");
            return 1;
        }
        NV_OF_BUFFER_DESCRIPTOR in{w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_ABGR8};
        uint32_t ow = (w + grid - 1) / grid, oh = (h + grid - 1) / grid;
        NV_OF_BUFFER_DESCRIPTOR out{ow, oh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2};
        NvOFGPUBufferHandle bi, br, bo;
        f.nvOFCreateGPUBufferCuda(hof, &in, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &bi);
        f.nvOFCreateGPUBufferCuda(hof, &in, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &br);
        f.nvOFCreateGPUBufferCuda(hof, &out, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &bo);
        auto upload = [&](NvOFGPUBufferHandle hb, const std::vector<uint8_t>& src) {
            CUdeviceptr p = f.nvOFGPUBufferGetCUdeviceptr(hb);
            NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
            f.nvOFGPUBufferGetStrideInfo(hb, &si);
            for (uint32_t y = 0; y < h; y++) cuMemcpyHtoD(p + size_t(y) * si.strideInfo[0].strideXInBytes, &src[size_t(y) * w * 4], size_t(w) * 4);
        };
        // forward: motion of each pixel of `a` into `b`; backward: of `b` into `a`
        upload(bi, dir == 0 ? a : b);
        upload(br, dir == 0 ? b : a);
        NV_OF_EXECUTE_INPUT_PARAMS ep{};
        ep.inputFrame = bi;
        ep.referenceFrame = br;
        ep.disableTemporalHints = NV_OF_TRUE;
        NV_OF_EXECUTE_OUTPUT_PARAMS op{};
        op.outputBuffer = bo;
        NV_OF_STATUS st = f.nvOFExecute(hof, &ep, &op);
        cuCtxSynchronize();
        if (st != NV_OF_SUCCESS) {
            std::printf("execute %d\n", int(st));
            return 1;
        }
        CUdeviceptr p = f.nvOFGPUBufferGetCUdeviceptr(bo);
        NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
        f.nvOFGPUBufferGetStrideInfo(bo, &si);
        std::vector<int16_t> flow(size_t(ow) * oh * 2);
        for (uint32_t y = 0; y < oh; y++) cuMemcpyDtoH(&flow[size_t(y) * ow * 2], p + size_t(y) * si.strideInfo[0].strideXInBytes, size_t(ow) * 4);
        FILE* o = std::fopen(argv[7 + dir], "wb");
        std::fwrite(flow.data(), 2, flow.size(), o);
        std::fclose(o);
        std::printf("%s: %ux%u vectors\n", argv[7 + dir], ow, oh);
        f.nvOFDestroyGPUBufferCuda(bi);
        f.nvOFDestroyGPUBufferCuda(br);
        f.nvOFDestroyGPUBufferCuda(bo);
        f.nvOFDestroy(hof);
    }
    return 0;
}
