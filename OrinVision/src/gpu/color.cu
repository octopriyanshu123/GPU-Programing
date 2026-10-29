// =====================================================================
// color.cu - RGB to grayscale
// =====================================================================
// One thread per pixel, grid-stride loop.
// Learning note: each thread reads 3 neighbouring bytes, so a warp reads
// one continuous block of 96 bytes -> memory access is coalesced.
// This kernel is MEMORY-BOUND: very little math per byte loaded.
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

namespace {

__global__ void rgbToGrayKernel(const uint8_t* __restrict__ rgb,
                                uint8_t* __restrict__ gray, int numPixels) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numPixels;
         i += blockDim.x * gridDim.x) {
        const uint8_t* p = rgb + 3 * i;
        gray[i] = grayFromRGB(p[0], p[1], p[2]);
    }
}

}  // namespace

namespace gpu {

void rgbToGray(const uint8_t* rgb, uint8_t* gray, int width, int height,
               cudaStream_t stream) {
    const int n = width * height;
    if (n == 0) return;
    const int threads = 256;
    const int blocks = divUp(n, threads);
    rgbToGrayKernel<<<blocks, threads, 0, stream>>>(rgb, gray, n);
    CUDA_CHECK_LAST();
}

}  // namespace gpu
