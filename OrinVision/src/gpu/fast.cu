// =====================================================================
// fast.cu - FAST-9 corner detector ("Features from Accelerated Segment Test")
// =====================================================================
// Look at the 16 pixels on a circle of radius 3 around each pixel:
//
//             . . 0 1 2 . .
//             . F . . . 3 .        corner if 9 or more CONTIGUOUS circle
//             E . . . . . 4        pixels are all brighter than center+t
//             D . . c . . 5        or all darker than center-t
//             C . . . . . 6
//             . B . . . 7 .
//             . . A 9 8 . .
//
// Very little math per pixel (no gradients, no multiplications), which is
// why FAST is used in real-time SLAM and in ORB. The test itself is
// fastScore() in common.hpp; peaks -> keypoint list in keypoints.cuh.
//
// Learning note: each pixel reads 17 pixels, so the block loads a tile
// with a 3-pixel halo (22x22) into shared memory once.
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"
#include "keypoints.cuh"

namespace {

constexpr int kTile = 16;
constexpr int kR = 3;   // circle radius

__global__ void fastScoreKernel(const uint8_t* __restrict__ in, int* __restrict__ score,
                                int w, int h, int t) {
    __shared__ uint8_t tile[kTile + 2 * kR][kTile + 2 * kR];

    const int x0 = blockIdx.x * kTile - kR;
    const int y0 = blockIdx.y * kTile - kR;
    for (int j = threadIdx.y; j < kTile + 2 * kR; j += kTile)
        for (int i = threadIdx.x; i < kTile + 2 * kR; i += kTile)
            tile[j][i] = in[clampi(y0 + j, 0, h - 1) * w + clampi(x0 + i, 0, w - 1)];
    __syncthreads();

    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x >= w || y >= h) return;   // safe: no __syncthreads() below

    int s = 0;
    if (x >= kFeatureBorder && y >= kFeatureBorder && x < w - kFeatureBorder &&
        y < h - kFeatureBorder) {
        const int tx = threadIdx.x + kR, ty = threadIdx.y + kR;
        int ring[16];
        for (int i = 0; i < 16; ++i) {
            int dx, dy;
            fastCircleOffset(i, dx, dy);
            ring[i] = tile[ty + dy][tx + dx];
        }
        s = fastScore(ring, tile[ty][tx], t);
    }
    score[y * w + x] = s;
}

}  // namespace

namespace gpu {

void fast(const uint8_t* in, int* score, int width, int height, int threshold,
          Keypoint* out, unsigned* count, int capacity, cudaStream_t stream) {
    if (width == 0 || height == 0) return;
    const dim3 block(kTile, kTile);
    const dim3 grid(divUp(width, kTile), divUp(height, kTile));
    fastScoreKernel<<<grid, block, 0, stream>>>(in, score, width, height, threshold);
    CUDA_CHECK_LAST();
    launchNmsCompact(static_cast<const int*>(score), width, height, kFeatureBorder, 0, out,
                     count, capacity, stream);
}

}  // namespace gpu
