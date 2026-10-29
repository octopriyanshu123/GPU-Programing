// =====================================================================
// gpu_filters.hpp - Host functions that launch the CUDA kernels
// =====================================================================
// All pointers are DEVICE pointers (from GpuBuffer::device() or cudaMalloc).
// All functions are asynchronous: they queue work on `stream` and return
// immediately. Synchronize before reading results on the CPU.
//
// Implementations: src/gpu/color.cu, histogram.cu, blur.cu, sobel.cu,
//                  gradient.cu, harris.cu, fast.cu, orb.cu, canny.cu
// =====================================================================
#pragma once

#include "common.hpp"   // Keypoint, OrbFeature, OrbTables

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu {

// True if this binary contains kernel code the current GPU can run
// (e.g. built for sm_87 but running on an sm_86 laptop GPU -> false).
bool kernelsAvailable(std::string& error);

// --- src/gpu/color.cu ---
void rgbToGray(const uint8_t* rgb, uint8_t* gray, int width, int height,
               cudaStream_t stream = 0);

// --- src/gpu/histogram.cu ---
// hist must hold 256 unsigned ints (it is zeroed internally).
void histogram(const uint8_t* gray, unsigned* hist, int numPixels,
               cudaStream_t stream = 0);
// Builds a 256-entry equalization lookup table from the histogram.
void buildEqualizeLut(const unsigned* hist, uint8_t* lut, unsigned numPixels,
                      cudaStream_t stream = 0);
void applyLut(const uint8_t* in, uint8_t* out, const uint8_t* lut, int numPixels,
              cudaStream_t stream = 0);

// --- src/gpu/blur.cu ---
// Upload Gaussian weights (size 2*radius+1) to constant memory.
void setGaussianWeights(const std::vector<float>& weights);
void gaussianBlurNaive(const uint8_t* in, uint8_t* out, int width, int height,
                       int radius, cudaStream_t stream = 0);
// tmp must hold width*height floats (result of the horizontal pass).
void gaussianBlurSeparable(const uint8_t* in, float* tmp, uint8_t* out,
                           int width, int height, int radius,
                           cudaStream_t stream = 0);

// --- src/gpu/sobel.cu ---
void sobel(const uint8_t* in, uint8_t* out, int width, int height,
           cudaStream_t stream = 0);

// =====================================================================
// Feature extraction
// Keypoint lists: `count` (1 unsigned, set to 0 before the call) receives
// the number of keypoints FOUND; at most `capacity` are stored in `out`.
// =====================================================================

// --- src/gpu/gradient.cu --- (shared by Harris and Canny)
void sobelGradients(const uint8_t* in, int16_t* gx, int16_t* gy, int width, int height,
                    cudaStream_t stream = 0);

// --- src/gpu/harris.cu --- (response must hold width*height values)
void harris(const int16_t* gx, const int16_t* gy, long long* response, int width, int height,
            long long threshold, Keypoint* out, unsigned* count, int capacity,
            cudaStream_t stream = 0);

// --- src/gpu/fast.cu --- (score must hold width*height ints)
void fast(const uint8_t* in, int* score, int width, int height, int threshold,
          Keypoint* out, unsigned* count, int capacity, cudaStream_t stream = 0);

// --- src/gpu/orb.cu ---
// Upload the rotated test patterns (once, from cpu::orbTables()).
void setOrbTables(const OrbTables& tables);
// One OrbFeature per keypoint in `keypoints` (count read on the GPU).
void orb(const uint8_t* in, int width, int height, const Keypoint* keypoints,
         const unsigned* count, int capacity, OrbFeature* out, cudaStream_t stream = 0);

// --- src/gpu/canny.cu ---
// classes = width*height scratch bytes. dChanged / hChanged = one int in
// device / pinned host memory. Hysteresis repeats until nothing changes,
// so this function waits for the GPU (returns the number of rounds).
int canny(const int16_t* gx, const int16_t* gy, uint8_t* classes, uint8_t* out,
          int width, int height, int low, int high, int* dChanged, int* hChanged,
          cudaStream_t stream = 0);

}  // namespace gpu
