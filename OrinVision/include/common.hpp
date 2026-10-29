// =====================================================================
// common.hpp - Types and math shared by the CPU and GPU implementations
// =====================================================================
// The small math functions below are marked HD (__host__ __device__), so
// the EXACT same code runs on the CPU and inside GPU kernels. This is how
// we get bit-exact results that are easy to verify.
// =====================================================================
#pragma once

#include "image.hpp"

#include <math.h>
#include <cstdint>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Largest Gaussian blur radius supported (kernel size 2*15+1 = 31 taps).
constexpr int kMaxBlurRadius = 15;

// ---------------------------------------------------------------------
// Pipeline stages (order matters: also used as array indexes)
// ---------------------------------------------------------------------
enum Stage {
    kUpload = 0,   // CPU -> GPU copy (only does work in "device" memory mode)
    kGray,         // RGB -> grayscale
    kHistogram,    // count how many pixels have each brightness 0..255
    kLut,          // build the equalization lookup table from the histogram
    kApplyLut,     // remap every pixel through the lookup table
    kBlur,         // Gaussian blur (removes noise before edge detection)
    kSobel,        // Sobel edge detection
    kDownload,     // GPU -> CPU copy (only does work in "device" memory mode)
    kNumStages
};

inline const char* stageName(int s) {
    static const char* names[kNumStages] = {
        "upload", "grayscale", "histogram", "build LUT",
        "apply LUT", "gaussian blur", "sobel edges", "download"};
    return (s >= 0 && s < kNumStages) ? names[s] : "?";
}

enum class BlurMode {
    Naive,       // 2D loop reading global memory directly (simple, slower)
    Separable    // two 1D passes using shared memory tiles (optimized)
};

inline const char* toString(BlurMode m) {
    return m == BlurMode::Naive ? "naive" : "separable";
}

// The four images the pipeline produces.
struct PipelineOutputs {
    Image gray;        // stage 1
    Image equalized;   // stage 2 (histogram equalization = better contrast)
    Image blurred;     // stage 3
    Image edges;       // stage 4 (final result)
};

// Normalized 1D Gaussian weights, size 2*radius+1, sum = 1.
// sigma <= 0 selects a sensible default (radius / 2).
std::vector<float> makeGaussianWeights(int radius, float sigma);

// ---------------------------------------------------------------------
// Host + device math helpers
// ---------------------------------------------------------------------
HD inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Round a non-negative float to the nearest byte (0..255).
HD inline uint8_t roundToByte(float v) {
    int i = (int)(v + 0.5f);
    return (uint8_t)clampi(i, 0, 255);
}

// ITU-R BT.601 luma in fixed point: 0.299 R + 0.587 G + 0.114 B
// (77 + 150 + 29 = 256, so the weights sum to exactly 1.0)
HD inline uint8_t grayFromRGB(uint8_t r, uint8_t g, uint8_t b) {
    return (uint8_t)((77u * r + 150u * g + 29u * b + 128u) >> 8);
}

// Histogram-equalization mapping for one brightness value (bin).
//   cdf    = number of pixels with brightness <= bin
//   cdfMin = cdf of the darkest brightness that actually occurs
//   total  = number of pixels
// Integer math only -> identical on CPU and GPU.
HD inline uint8_t equalizeValue(int bin, unsigned cdf, unsigned cdfMin, unsigned total) {
    if (total <= cdfMin) return (uint8_t)bin;   // flat image: keep as-is
    if (cdf < cdfMin) return 0;                  // brightness not present
    unsigned long long range = total - cdfMin;
    unsigned long long num = (unsigned long long)(cdf - cdfMin) * 255ull + range / 2;
    return (uint8_t)(num / range);
}

// Sobel edge strength from a 3x3 neighborhood:
//   p00 p01 p02
//   p10  x  p12      (center pixel is not used)
//   p20 p21 p22
HD inline uint8_t sobelMagnitude(int p00, int p01, int p02,
                                 int p10,          int p12,
                                 int p20, int p21, int p22) {
    int gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);   // horizontal change
    int gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);   // vertical change
    float mag = sqrtf((float)(gx * gx + gy * gy));
    return roundToByte(mag);
}
