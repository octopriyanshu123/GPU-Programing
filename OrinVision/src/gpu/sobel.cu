// =====================================================================
// sobel.cu - Sobel edge detection with a shared-memory tile
// =====================================================================
// Each output pixel needs its 3x3 neighbourhood, so neighbouring threads
// read mostly the same pixels. A 16x16 block loads an 18x18 tile (1-pixel
// halo on every side) into shared memory ONCE, then every thread reads its
// 8 neighbours from shared memory: ~1.3 global reads per pixel instead of 8.
//
// Important rule shown here: NO early "return" before __syncthreads().
// Every thread in the block must reach the barrier, even threads that are
// outside the image; they still help load the tile.
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

namespace {

constexpr int kTile = 16;

__global__ void sobelKernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                            int w, int h) {
    __shared__ uint8_t tile[kTile + 2][kTile + 2];

    // Top-left corner of the tile (including halo) in image coordinates
    const int x0 = blockIdx.x * kTile - 1;
    const int y0 = blockIdx.y * kTile - 1;

    // Cooperative load of (16+2) x (16+2) pixels by 16 x 16 threads
    for (int j = threadIdx.y; j < kTile + 2; j += kTile)
        for (int i = threadIdx.x; i < kTile + 2; i += kTile)
            tile[j][i] = in[clampi(y0 + j, 0, h - 1) * w + clampi(x0 + i, 0, w - 1)];
    __syncthreads();

    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x < w && y < h) {
        const int tx = threadIdx.x + 1;   // +1 skips the halo
        const int ty = threadIdx.y + 1;
        out[y * w + x] = sobelMagnitude(
            tile[ty - 1][tx - 1], tile[ty - 1][tx], tile[ty - 1][tx + 1],
            tile[ty][tx - 1],                       tile[ty][tx + 1],
            tile[ty + 1][tx - 1], tile[ty + 1][tx], tile[ty + 1][tx + 1]);
    }
}

}  // namespace

namespace gpu {

void sobel(const uint8_t* in, uint8_t* out, int width, int height, cudaStream_t stream) {
    if (width == 0 || height == 0) return;
    const dim3 block(kTile, kTile);
    const dim3 grid(divUp(width, kTile), divUp(height, kTile));
    sobelKernel<<<grid, block, 0, stream>>>(in, out, width, height);
    CUDA_CHECK_LAST();
}

}  // namespace gpu
