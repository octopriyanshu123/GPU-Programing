// =====================================================================
// canny.cu - Canny edge detector
// =====================================================================
// Sobel (stage 4) gives thick, noisy edge strength. Canny turns it into
// thin, connected 1-pixel edge lines:
//   1. Non-maximum suppression: keep a pixel only if its gradient is the
//      largest ALONG the gradient direction (thins edges to 1 pixel).
//   2. Double threshold: > high = strong edge, > low = weak edge.
//   3. Hysteresis: a weak pixel is kept only if it is CONNECTED to a strong
//      one (through other kept pixels). This removes isolated noise but
//      keeps faint parts of real edges.
//
// Learning notes - hysteresis is a "flood fill", which is hard on a GPU:
//  - Each block repeats "weak pixel next to a strong one -> make it strong"
//    inside its shared-memory tile until nothing changes (__syncthreads_or
//    tells every thread whether ANY thread changed something).
//  - An edge can cross many blocks, so the whole kernel is launched again
//    until a global "changed" flag stays 0. The CPU has to read that flag,
//    so this is the one stage that waits for the GPU between kernels.
//  - Pixels only ever go weak -> strong, never back, so reading a
//    neighbour block's pixel "too early" is harmless: the next round
//    catches it. The final result is exactly the CPU flood fill.
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

namespace {

constexpr int kTile = 16;

__device__ inline int mag2At(const int16_t* gx, const int16_t* gy, int x, int y, int w, int h) {
    const int i = clampi(y, 0, h - 1) * w + clampi(x, 0, w - 1);
    const int a = gx[i], b = gy[i];
    return a * a + b * b;
}

// Step 1 + 2: thin and classify every pixel (0 / 1 weak / 2 strong)
__global__ void cannyNmsKernel(const int16_t* __restrict__ gx, const int16_t* __restrict__ gy,
                               uint8_t* __restrict__ classes, int w, int h, int low2, int high2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int i = y * w + x;
    int dx, dy;
    cannyDirection(gx[i], gy[i], dx, dy);
    classes[i] = cannyClass(mag2At(gx, gy, x, y, w, h), mag2At(gx, gy, x - dx, y - dy, w, h),
                            mag2At(gx, gy, x + dx, y + dy, w, h), low2, high2);
}

// Step 3: one round of hysteresis (see the notes at the top)
__global__ void cannyHysteresisKernel(uint8_t* classes, int w, int h, int* changed) {
    __shared__ uint8_t tile[kTile + 2][kTile + 2];

    const int x0 = blockIdx.x * kTile - 1;
    const int y0 = blockIdx.y * kTile - 1;
    for (int j = threadIdx.y; j < kTile + 2; j += kTile)
        for (int i = threadIdx.x; i < kTile + 2; i += kTile)
            tile[j][i] = classes[clampi(y0 + j, 0, h - 1) * w + clampi(x0 + i, 0, w - 1)];
    __syncthreads();

    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    const bool inside = x < w && y < h;   // NO early return: everyone must reach the barriers
    const int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

    bool blockChanged = false;
    while (true) {
        bool mine = false;
        if (inside && tile[ty][tx] == 1) {
            for (int dy = -1; dy <= 1 && !mine; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if (tile[ty + dy][tx + dx] == 2) { mine = true; break; }
            if (mine) tile[ty][tx] = 2;
        }
        // Returns true in EVERY thread if ANY thread of the block changed a pixel
        if (!__syncthreads_or(mine)) break;
        blockChanged = true;
    }

    if (inside) classes[y * w + x] = tile[ty][tx];
    if (blockChanged && threadIdx.x == 0 && threadIdx.y == 0) *changed = 1;
}

__global__ void cannyFinalizeKernel(const uint8_t* __restrict__ classes,
                                    uint8_t* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = classes[i] == 2 ? 255 : 0;
}

}  // namespace

namespace gpu {

int canny(const int16_t* gx, const int16_t* gy, uint8_t* classes, uint8_t* out,
          int width, int height, int low, int high, int* dChanged, int* hChanged,
          cudaStream_t stream) {
    if (width == 0 || height == 0) return 0;

    const dim3 nmsBlock(32, 8);
    const dim3 nmsGrid(divUp(width, 32), divUp(height, 8));
    cannyNmsKernel<<<nmsGrid, nmsBlock, 0, stream>>>(gx, gy, classes, width, height,
                                                     low * low, high * high);
    CUDA_CHECK_LAST();

    const dim3 block(kTile, kTile);
    const dim3 grid(divUp(width, kTile), divUp(height, kTile));
    int rounds = 0;
    do {
        CUDA_CHECK(cudaMemsetAsync(dChanged, 0, sizeof(int), stream));
        cannyHysteresisKernel<<<grid, block, 0, stream>>>(classes, width, height, dChanged);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpyAsync(hChanged, dChanged, sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));   // the CPU needs the flag now
        ++rounds;
    } while (*hChanged);

    const int n = width * height;
    const int threads = 256;
    cannyFinalizeKernel<<<divUp(n, threads), threads, 0, stream>>>(classes, out, n);
    CUDA_CHECK_LAST();
    return rounds;
}

}  // namespace gpu
