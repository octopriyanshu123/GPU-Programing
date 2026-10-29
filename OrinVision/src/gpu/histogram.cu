// =====================================================================
// histogram.cu - Histogram equalization in three kernels
// =====================================================================
//  1. histogramKernel   : count pixels per brightness (atomics)
//  2. equalizeLutKernel : prefix sum (scan) of the histogram -> lookup table
//  3. applyLutKernel    : remap every pixel through the lookup table
//
// Learning notes:
//  - Thousands of threads adding to the SAME 256 global counters would be
//    very slow (they all fight over the same addresses). Instead each block
//    builds a private histogram in fast SHARED memory, then adds it to the
//    global one once: 256 global atomics per block instead of 1 per pixel.
//  - The scan uses the Hillis-Steele algorithm: log2(256) = 8 steps.
// =====================================================================
#include "common.hpp"
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

#include <algorithm>

namespace {

constexpr int kBins = 256;

__global__ void histogramKernel(const uint8_t* __restrict__ gray,
                                unsigned* __restrict__ hist, int numPixels) {
    __shared__ unsigned local[kBins];

    // 1) Clear this block's private histogram
    for (int i = threadIdx.x; i < kBins; i += blockDim.x) local[i] = 0;
    __syncthreads();

    // 2) Count pixels (fast shared-memory atomics)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numPixels;
         i += blockDim.x * gridDim.x)
        atomicAdd(&local[gray[i]], 1u);
    __syncthreads();

    // 3) Merge into the global histogram (one atomic per non-empty bin)
    for (int i = threadIdx.x; i < kBins; i += blockDim.x)
        if (local[i]) atomicAdd(&hist[i], local[i]);
}

// Launched with exactly 1 block of 256 threads (one thread per bin).
__global__ void equalizeLutKernel(const unsigned* __restrict__ hist,
                                  uint8_t* __restrict__ lut, unsigned numPixels) {
    __shared__ unsigned cdf[kBins];
    __shared__ unsigned cdfMin;
    const int t = threadIdx.x;

    cdf[t] = hist[t];
    if (t == 0) cdfMin = 0xFFFFFFFFu;
    __syncthreads();

    // Inclusive prefix sum (Hillis-Steele): after the loop,
    // cdf[t] = hist[0] + hist[1] + ... + hist[t]
    for (int offset = 1; offset < kBins; offset <<= 1) {
        const unsigned add = (t >= offset) ? cdf[t - offset] : 0u;
        __syncthreads();          // everyone has read before anyone writes
        cdf[t] += add;
        __syncthreads();          // everyone has written before next read
    }

    // Smallest non-zero cdf value
    if (cdf[t] > 0) atomicMin(&cdfMin, cdf[t]);
    __syncthreads();

    lut[t] = equalizeValue(t, cdf[t], cdfMin, numPixels);
}

__global__ void applyLutKernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                               const uint8_t* __restrict__ lut, int numPixels) {
    // Copy the 256-byte table into shared memory once per block
    __shared__ uint8_t s_lut[kBins];
    for (int i = threadIdx.x; i < kBins; i += blockDim.x) s_lut[i] = lut[i];
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numPixels;
         i += blockDim.x * gridDim.x)
        out[i] = s_lut[in[i]];
}

// Enough blocks to keep every SM busy, but not one block per pixel.
int gridStrideBlocks(int n, int threads) {
    return std::max(1, std::min(divUp(n, threads), smCount() * 8));
}

}  // namespace

namespace gpu {

void histogram(const uint8_t* gray, unsigned* hist, int numPixels, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(hist, 0, kBins * sizeof(unsigned), stream));
    if (numPixels == 0) return;
    const int threads = 256;
    histogramKernel<<<gridStrideBlocks(numPixels, threads), threads, 0, stream>>>(
        gray, hist, numPixels);
    CUDA_CHECK_LAST();
}

void buildEqualizeLut(const unsigned* hist, uint8_t* lut, unsigned numPixels,
                      cudaStream_t stream) {
    equalizeLutKernel<<<1, kBins, 0, stream>>>(hist, lut, numPixels);
    CUDA_CHECK_LAST();
}

void applyLut(const uint8_t* in, uint8_t* out, const uint8_t* lut, int numPixels,
              cudaStream_t stream) {
    if (numPixels == 0) return;
    const int threads = 256;
    applyLutKernel<<<gridStrideBlocks(numPixels, threads), threads, 0, stream>>>(
        in, out, lut, numPixels);
    CUDA_CHECK_LAST();
}

}  // namespace gpu
