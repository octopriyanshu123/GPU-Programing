// =====================================================================
// keypoints.cuh - Turn a score image into a list of keypoints (GPU)
// =====================================================================
// Used by Harris and FAST. Every pixel checks "am I a 3x3 peak above the
// threshold?" and, if yes, reserves a slot in the output list with
// atomicAdd on a global counter (this is called STREAM COMPACTION).
//
// Learning notes:
//  - The order of the list is random (whichever thread wins the atomic
//    first), so verification sorts the lists before comparing.
//  - The counter keeps counting past `capacity`, so the CPU can see how
//    many keypoints were found even if the list was too small.
//  - No shared memory and no __syncthreads() here, so early returns are OK.
// =====================================================================
#pragma once

#include "common.hpp"
#include "cuda_utils.hpp"

template <typename T>
__global__ void nmsCompactKernel(const T* __restrict__ score, int w, int h, int border,
                                 T threshold, Keypoint* __restrict__ out,
                                 unsigned* __restrict__ count, int capacity) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < border || y < border || x >= w - border || y >= h - border) return;

    const int idx = y * w + x;
    if (score[idx] <= threshold || !isPeak3x3(score, idx, w)) return;

    const unsigned slot = atomicAdd(count, 1u);
    if (slot < static_cast<unsigned>(capacity)) {
        out[slot].x = x;
        out[slot].y = y;
        out[slot].score = static_cast<long long>(score[idx]);
    }
}

template <typename T>
inline void launchNmsCompact(const T* score, int w, int h, int border, T threshold,
                             Keypoint* out, unsigned* count, int capacity, cudaStream_t stream) {
    if (w <= 2 * border || h <= 2 * border) return;   // image too small: no keypoints
    const dim3 block(32, 8);
    const dim3 grid(divUp(w, 32), divUp(h, 8));
    nmsCompactKernel<T><<<grid, block, 0, stream>>>(score, w, h, border, threshold, out, count,
                                                    capacity);
    CUDA_CHECK_LAST();
}
