// =====================================================================
// blur.cu - Gaussian blur, two versions to compare
// =====================================================================
// NAIVE (2D): each thread loops over a (2r+1) x (2r+1) window and reads
//   every neighbour from global memory. Radius 4 = 81 reads per pixel.
//
// SEPARABLE + SHARED MEMORY: a 2D Gaussian = horizontal 1D blur followed
//   by vertical 1D blur. Only 2*(2r+1) = 18 multiplies per pixel for r=4.
//   Each block first loads its tile (plus a border of r pixels, called the
//   "halo" or "apron") into shared memory, then all threads read from that
//   fast copy instead of global memory.
//
// The weights live in __constant__ memory: every thread of a warp reads
// the same weight at the same time, which constant memory serves in one
// broadcast. Compare both with: --blur naive / --blur separable
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

#include <stdexcept>

namespace {

__constant__ float c_weights[2 * kMaxBlurRadius + 1];

constexpr int kBlockX = 32;   // one warp wide -> coalesced row access
constexpr int kBlockY = 8;

// ---------------------------------------------------------------------
// Naive 2D blur (global memory only)
// ---------------------------------------------------------------------
__global__ void blurNaiveKernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                int w, int h, int r) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = clampi(y + dy, 0, h - 1);
        float rowSum = 0.0f;
        for (int dx = -r; dx <= r; ++dx)
            rowSum += c_weights[dx + r] * in[yy * w + clampi(x + dx, 0, w - 1)];
        sum += c_weights[dy + r] * rowSum;
    }
    out[y * w + x] = roundToByte(sum);
}

// ---------------------------------------------------------------------
// Separable pass 1: horizontal (uint8 -> float)
// Shared tile per block: blockDim.y rows x (blockDim.x + 2r) columns
// ---------------------------------------------------------------------
__global__ void blurRowsKernel(const uint8_t* __restrict__ in, float* __restrict__ out,
                               int w, int h, int r) {
    extern __shared__ float tile[];
    const int tileW = blockDim.x + 2 * r;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int yc = min(y, h - 1);               // keep loads in range
    float* row = tile + threadIdx.y * tileW;

    // Cooperative load: the row segment + r halo pixels on each side
    const int startX = blockIdx.x * blockDim.x - r;
    for (int i = threadIdx.x; i < tileW; i += blockDim.x)
        row[i] = in[yc * w + clampi(startX + i, 0, w - 1)];
    __syncthreads();

    if (x < w && y < h) {
        float sum = 0.0f;
        for (int k = -r; k <= r; ++k)
            sum += c_weights[k + r] * row[threadIdx.x + r + k];
        out[y * w + x] = sum;
    }
}

// ---------------------------------------------------------------------
// Separable pass 2: vertical (float -> uint8)
// Shared tile per block: (blockDim.y + 2r) rows x blockDim.x columns
// ---------------------------------------------------------------------
__global__ void blurColsKernel(const float* __restrict__ in, uint8_t* __restrict__ out,
                               int w, int h, int r) {
    extern __shared__ float tile[];
    const int tileH = blockDim.y + 2 * r;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int xc = min(x, w - 1);

    const int startY = blockIdx.y * blockDim.y - r;
    for (int j = threadIdx.y; j < tileH; j += blockDim.y)
        tile[j * blockDim.x + threadIdx.x] = in[clampi(startY + j, 0, h - 1) * w + xc];
    __syncthreads();

    if (x < w && y < h) {
        float sum = 0.0f;
        for (int k = -r; k <= r; ++k)
            sum += c_weights[k + r] * tile[(threadIdx.y + r + k) * blockDim.x + threadIdx.x];
        out[y * w + x] = roundToByte(sum);
    }
}

void checkRadius(int radius) {
    if (radius < 1 || radius > kMaxBlurRadius)
        throw std::invalid_argument("blur radius must be between 1 and 15");
}

}  // namespace

namespace gpu {

void setGaussianWeights(const std::vector<float>& weights) {
    if (weights.empty() || weights.size() > 2 * kMaxBlurRadius + 1)
        throw std::invalid_argument("invalid number of Gaussian weights");
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, weights.data(), weights.size() * sizeof(float)));
}

void gaussianBlurNaive(const uint8_t* in, uint8_t* out, int width, int height,
                       int radius, cudaStream_t stream) {
    checkRadius(radius);
    if (width == 0 || height == 0) return;
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(divUp(width, kBlockX), divUp(height, kBlockY));
    blurNaiveKernel<<<grid, block, 0, stream>>>(in, out, width, height, radius);
    CUDA_CHECK_LAST();
}

void gaussianBlurSeparable(const uint8_t* in, float* tmp, uint8_t* out,
                           int width, int height, int radius, cudaStream_t stream) {
    checkRadius(radius);
    if (width == 0 || height == 0) return;
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(divUp(width, kBlockX), divUp(height, kBlockY));

    const size_t rowsShared = size_t(kBlockY) * (kBlockX + 2 * radius) * sizeof(float);
    blurRowsKernel<<<grid, block, rowsShared, stream>>>(in, tmp, width, height, radius);
    CUDA_CHECK_LAST();

    const size_t colsShared = size_t(kBlockY + 2 * radius) * kBlockX * sizeof(float);
    blurColsKernel<<<grid, block, colsShared, stream>>>(tmp, out, width, height, radius);
    CUDA_CHECK_LAST();
}

}  // namespace gpu
