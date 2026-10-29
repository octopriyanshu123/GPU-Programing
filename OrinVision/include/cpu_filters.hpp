// =====================================================================
// cpu_filters.hpp - CPU reference implementations
// =====================================================================
// Two jobs:
//   1. Correctness: the GPU output is compared against these.
//   2. Speed baseline: how much faster is the GPU? (uses all CPU cores
//      through OpenMP when available, so the comparison is fair)
// =====================================================================
#pragma once

#include "common.hpp"

#include <cstdint>
#include <vector>

namespace cpu {

void rgbToGray(const uint8_t* rgb, uint8_t* gray, int width, int height);
void histogram(const uint8_t* gray, unsigned* hist, int numPixels);
void buildEqualizeLut(const unsigned* hist, uint8_t* lut, unsigned numPixels);
void applyLut(const uint8_t* in, uint8_t* out, const uint8_t* lut, int numPixels);
void gaussianBlur(const uint8_t* in, uint8_t* out, int width, int height,
                  const std::vector<float>& weights);
void sobel(const uint8_t* in, uint8_t* out, int width, int height);

// --- Feature extraction: src/cpu_features.cpp ---
void sobelGradients(const uint8_t* in, int16_t* gx, int16_t* gy, int width, int height);
// Keypoints are returned in raster order (top-left to bottom-right).
std::vector<Keypoint> harris(const uint8_t* in, int width, int height, long long threshold);
std::vector<Keypoint> fast(const uint8_t* in, int width, int height, int threshold);
// One OrbFeature per keypoint, same order (angleBin -1 = too close to the border).
std::vector<OrbFeature> orb(const uint8_t* in, int width, int height,
                            const std::vector<Keypoint>& keypoints);
void canny(const uint8_t* in, uint8_t* out, int width, int height, int low, int high);
// Rotated ORB test patterns (built once, shared with the GPU).
const OrbTables& orbTables();

// Run the full pipeline on the CPU. stageMs[] receives the time of each
// stage in milliseconds (upload/download are 0 on the CPU).
void runPipeline(const Image& rgb, const std::vector<float>& weights,
                 const FeatureConfig& features, PipelineOutputs& out,
                 double stageMs[kNumStages]);

// Number of CPU threads used (1 if built without OpenMP).
int numThreads();

}  // namespace cpu
