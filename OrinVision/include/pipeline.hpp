// =====================================================================
// pipeline.hpp - The complete GPU image-processing pipeline
// =====================================================================
//   RGB frame
//     -> [1] grayscale
//     -> [2] histogram equalization (histogram + LUT + apply LUT)
//     -> [3] Gaussian blur
//     -> [4] Sobel edge detection
//
// Buffers are allocated ONCE in the constructor and reused for every
// frame, like a real camera/video application. Each stage is timed with
// CUDA events recorded on one stream (no extra CPU syncs between stages).
// =====================================================================
#pragma once

#include "common.hpp"
#include "gpu_buffer.hpp"
#include "platform.hpp"

#include <vector>

struct PipelineConfig {
    int width = 0;
    int height = 0;
    MemoryMode memoryMode = MemoryMode::Managed;
    BlurMode blurMode = BlurMode::Separable;
    int blurRadius = 4;        // 1 .. kMaxBlurRadius
    float blurSigma = 0.0f;    // <= 0 means "radius / 2"
    FeatureConfig features;    // Harris / FAST / ORB / Canny on the blurred image
};

struct StageTimings {
    float ms[kNumStages] = {};   // time per stage
    float totalMs = 0.0f;        // first event to last event
};

class GpuPipeline {
public:
    explicit GpuPipeline(const PipelineConfig& config);
    ~GpuPipeline();

    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    // Process one RGB frame (must match the configured size).
    // If `out` is not null, the four result images are copied into it.
    StageTimings process(const Image& rgb, PipelineOutputs* out);

    const PipelineConfig& config() const { return cfg_; }
    const std::vector<float>& weights() const { return weights_; }

private:
    PipelineConfig cfg_;
    std::vector<float> weights_;

    // Buffers the CPU reads or writes: they follow the chosen memory mode.
    GpuBuffer<uint8_t> rgb_, gray_, equalized_, blurred_, edges_;

    // Feature results (CPU-visible, follow the memory mode too).
    GpuBuffer<unsigned> kpCounts_;            // [0] Harris found, [1] FAST found
    GpuBuffer<Keypoint> harrisKps_, fastKps_;
    GpuBuffer<OrbFeature> orb_;
    GpuBuffer<uint8_t> canny_;

    // GPU-only scratch buffers: always plain device memory.
    unsigned* hist_ = nullptr;     // 256 bins
    uint8_t* lut_ = nullptr;       // 256 entries
    float* blurTmp_ = nullptr;     // separable blur intermediate
    int16_t* gradX_ = nullptr;     // Sobel gradients (Harris + Canny)
    int16_t* gradY_ = nullptr;
    long long* harrisResp_ = nullptr;
    int* fastScore_ = nullptr;
    uint8_t* cannyClasses_ = nullptr;
    int* cannyChangedDev_ = nullptr;    // hysteresis "something changed" flag
    int* cannyChangedHost_ = nullptr;   // pinned copy the CPU reads

    cudaStream_t stream_ = nullptr;
    cudaEvent_t events_[kNumStages + 1] = {};

    // Decided at run time from platform(): true on Jetson.
    bool deviceSyncBeforeCpu_ = true;
};
