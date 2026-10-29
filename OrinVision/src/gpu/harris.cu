// =====================================================================
// harris.cu - Harris corner detector
// =====================================================================
// A corner is a place where the image changes strongly in TWO directions.
// For every pixel we sum gradient products over a 3x3 window:
//       sxx = sum gx*gx    syy = sum gy*gy    sxy = sum gx*gy
// and compute R = det - k * trace^2 (see harrisResponse in common.hpp):
//   R >> 0 : corner        R < 0 : edge        R ~ 0 : flat area
// Then keypoints = 3x3 peaks of R above a threshold (keypoints.cuh).
//
// Learning note: neighbouring threads use the same gradients, so each
// block loads an 18x18 gradient tile into shared memory once.
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"
#include "keypoints.cuh"

namespace {

constexpr int kTile = 16;

__global__ void harrisResponseKernel(const int16_t* __restrict__ gx,
                                     const int16_t* __restrict__ gy,
                                     long long* __restrict__ resp, int w, int h) {
    __shared__ int sgx[kTile + 2][kTile + 2];
    __shared__ int sgy[kTile + 2][kTile + 2];

    // Load the gradients of the tile + 1 pixel halo (clamped at the image edge)
    const int x0 = blockIdx.x * kTile - 1;
    const int y0 = blockIdx.y * kTile - 1;
    for (int j = threadIdx.y; j < kTile + 2; j += kTile)
        for (int i = threadIdx.x; i < kTile + 2; i += kTile) {
            const int g = clampi(y0 + j, 0, h - 1) * w + clampi(x0 + i, 0, w - 1);
            sgx[j][i] = gx[g];
            sgy[j][i] = gy[g];
        }
    __syncthreads();

    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x < w && y < h) {
        long long sxx = 0, syy = 0, sxy = 0;
        for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx) {
                const int a = sgx[threadIdx.y + dy][threadIdx.x + dx];
                const int b = sgy[threadIdx.y + dy][threadIdx.x + dx];
                sxx += a * a;
                syy += b * b;
                sxy += a * b;
            }
        resp[y * w + x] = harrisResponse(sxx, syy, sxy);
    }
}

}  // namespace

namespace gpu {

void harris(const int16_t* gx, const int16_t* gy, long long* response, int width, int height,
            long long threshold, Keypoint* out, unsigned* count, int capacity,
            cudaStream_t stream) {
    if (width == 0 || height == 0) return;
    const dim3 block(kTile, kTile);
    const dim3 grid(divUp(width, kTile), divUp(height, kTile));
    harrisResponseKernel<<<grid, block, 0, stream>>>(gx, gy, response, width, height);
    CUDA_CHECK_LAST();
    launchNmsCompact(static_cast<const long long*>(response), width, height, kFeatureBorder,
                     threshold, out, count, capacity, stream);
}

}  // namespace gpu
