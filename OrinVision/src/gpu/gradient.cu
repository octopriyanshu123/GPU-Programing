// =====================================================================
// gradient.cu - Sobel gradients gx, gy (input for Harris and Canny)
// =====================================================================
// Same shared-memory tile as sobel.cu, but instead of the edge strength
// it stores the two signed gradients. Computed ONCE per frame and reused
// by both Harris and Canny (int16 is enough: |g| <= 4 * 255 = 1020).
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

namespace {

constexpr int kTile = 16;

__global__ void sobelGradientKernel(const uint8_t* __restrict__ in, int16_t* __restrict__ gx,
                                    int16_t* __restrict__ gy, int w, int h) {
    __shared__ uint8_t tile[kTile + 2][kTile + 2];

    const int x0 = blockIdx.x * kTile - 1;
    const int y0 = blockIdx.y * kTile - 1;
    for (int j = threadIdx.y; j < kTile + 2; j += kTile)
        for (int i = threadIdx.x; i < kTile + 2; i += kTile)
            tile[j][i] = in[clampi(y0 + j, 0, h - 1) * w + clampi(x0 + i, 0, w - 1)];
    __syncthreads();

    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x < w && y < h) {
        const int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
        int gxv, gyv;
        sobelGradient(tile[ty - 1][tx - 1], tile[ty - 1][tx], tile[ty - 1][tx + 1],
                      tile[ty][tx - 1],                       tile[ty][tx + 1],
                      tile[ty + 1][tx - 1], tile[ty + 1][tx], tile[ty + 1][tx + 1], gxv, gyv);
        gx[y * w + x] = static_cast<int16_t>(gxv);
        gy[y * w + x] = static_cast<int16_t>(gyv);
    }
}

}  // namespace

namespace gpu {

void sobelGradients(const uint8_t* in, int16_t* gx, int16_t* gy, int width, int height,
                    cudaStream_t stream) {
    if (width == 0 || height == 0) return;
    const dim3 block(kTile, kTile);
    const dim3 grid(divUp(width, kTile), divUp(height, kTile));
    sobelGradientKernel<<<grid, block, 0, stream>>>(in, gx, gy, width, height);
    CUDA_CHECK_LAST();
}

}  // namespace gpu
