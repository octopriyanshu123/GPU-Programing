// =====================================================================
// orb.cu - ORB descriptors ("Oriented FAST and Rotated BRIEF")
// =====================================================================
// For every FAST corner:
//   1. ORIENTATION: the intensity centroid of a radius-15 disk around the
//      corner. The direction center -> centroid is the corner's "angle",
//      quantized into 30 bins of 12 degrees (orbAngleBin in common.hpp).
//   2. DESCRIPTOR: 256 brightness comparisons I(p1) < I(p2) between pixel
//      pairs of a fixed random pattern, ROTATED by the corner's angle ->
//      256 bits = 32 bytes. Two corners look alike if their descriptors
//      differ in few bits (Hamming distance), even if the image is rotated.
//
// Learning notes - ONE WARP (32 threads) PER KEYPOINT:
//  - Moments: lane i sums row i of the disk (31 rows), then the 32 partial
//    sums are added with __shfl_down_sync: threads exchange registers
//    directly, no shared memory and no __syncthreads() needed.
//  - Descriptor: 32 bytes = 32 lanes, each lane computes one byte (8 bits)
//    and the warp writes the 32 bytes in one coalesced store.
//  - The keypoint count lives on the GPU, so the kernel loops over keypoints
//    with a "grid-stride" loop: the CPU never has to read the count.
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

#include <algorithm>

namespace {

constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr int kWarpsPerBlock = 4;

__constant__ int c_orbCos[kOrbBins];
__constant__ int c_orbSin[kOrbBins];
// 30 x 256 x 4 bytes = 30 KB: lanes read different entries, so plain
// global memory (cached) is a better fit than constant memory here.
__device__ int8_t d_orbPattern[kOrbBins * kOrbPairs * 4];

__global__ void orbKernel(const uint8_t* __restrict__ img, int w, int h,
                          const Keypoint* __restrict__ kps, const unsigned* __restrict__ count,
                          int capacity, OrbFeature* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int numWarps = (gridDim.x * blockDim.x) >> 5;
    const int n = min(*count, static_cast<unsigned>(capacity));

    // All 32 lanes of a warp have the same k -> they stay together.
    for (int k = warp; k < n; k += numWarps) {
        const int x = kps[k].x, y = kps[k].y;
        OrbFeature& f = out[k];
        if (lane == 0) { f.x = x; f.y = y; }

        if (!orbInside(x, y, w, h)) {          // too close to the border
            if (lane == 0) f.angleBin = -1;
            f.desc[lane] = 0;
            continue;
        }

        // 1) Moments: lane i handles disk row dy = i - 15 (lane 31 idle)
        int m10 = 0, m01 = 0;
        if (lane < 2 * kOrbPatchRadius + 1)
            orbMomentsRow(img, w, x, y, lane - kOrbPatchRadius, m10, m01);
        for (int off = 16; off > 0; off >>= 1) {          // warp reduction
            m10 += __shfl_down_sync(kFullMask, m10, off);
            m01 += __shfl_down_sync(kFullMask, m01, off);
        }

        // 2) Lane 0 has the totals -> picks the angle bin -> tells everyone
        int bin = 0;
        if (lane == 0) bin = orbAngleBin(m10, m01, c_orbCos, c_orbSin);
        bin = __shfl_sync(kFullMask, bin, 0);

        // 3) Each lane computes one descriptor byte
        f.desc[lane] = orbDescriptorByte(img, w, x, y, d_orbPattern + bin * kOrbPairs * 4, lane);
        if (lane == 0) f.angleBin = bin;
    }
}

}  // namespace

namespace gpu {

void setOrbTables(const OrbTables& t) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_orbCos, t.cosQ, sizeof(t.cosQ)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_orbSin, t.sinQ, sizeof(t.sinQ)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_orbPattern, t.pattern, sizeof(t.pattern)));
}

void orb(const uint8_t* in, int width, int height, const Keypoint* keypoints,
         const unsigned* count, int capacity, OrbFeature* out, cudaStream_t stream) {
    if (width == 0 || height == 0 || capacity == 0) return;
    const int threads = 32 * kWarpsPerBlock;
    const int maxBlocks = divUp(capacity, kWarpsPerBlock);
    const int blocks = std::min(maxBlocks, smCount() * 8);
    orbKernel<<<blocks, threads, 0, stream>>>(in, width, height, keypoints, count, capacity, out);
    CUDA_CHECK_LAST();
}

}  // namespace gpu
