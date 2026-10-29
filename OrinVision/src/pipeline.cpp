// =====================================================================
// pipeline.cpp - Runs all GPU stages and times them with CUDA events
// =====================================================================
#include "pipeline.hpp"

#include "cpu_filters.hpp"   // cpu::orbTables() (same ORB pattern on CPU and GPU)
#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

// initCuda() lives in src/platform.cpp (Jetson / PC detection).

GpuPipeline::GpuPipeline(const PipelineConfig& config)
    : cfg_(config),
      // Jetson cannot let the CPU touch managed memory while ANY kernel runs;
      // a laptop GPU can, so the extra device-wide sync is skipped there.
      deviceSyncBeforeCpu_(!platform().concurrentManagedAccess) {
    if (cfg_.width <= 0 || cfg_.height <= 0)
        throw std::invalid_argument("pipeline size must be positive");
    if (cfg_.blurRadius < 1 || cfg_.blurRadius > kMaxBlurRadius)
        throw std::invalid_argument("blur radius must be between 1 and " +
                                    std::to_string(kMaxBlurRadius));

    const size_t n = size_t(cfg_.width) * cfg_.height;
    const MemoryMode m = cfg_.memoryMode;

    rgb_.allocate(n * 3, m);
    gray_.allocate(n, m);
    equalized_.allocate(n, m);
    blurred_.allocate(n, m);
    edges_.allocate(n, m);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hist_), 256 * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lut_), 256));
    if (cfg_.blurMode == BlurMode::Separable)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&blurTmp_), n * sizeof(float)));

    weights_ = makeGaussianWeights(cfg_.blurRadius, cfg_.blurSigma);
    gpu::setGaussianWeights(weights_);

    // ---- Feature extraction buffers (only for the enabled features) ----
    FeatureConfig& f = cfg_.features;
    if (f.orb) f.fast = true;                 // ORB describes the FAST corners
    if (f.maxKeypoints < 1) throw std::invalid_argument("maxKeypoints must be >= 1");
    const size_t cap = static_cast<size_t>(f.maxKeypoints);

    kpCounts_.allocate(2, m);
    if (f.harris || f.canny) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gradX_), n * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gradY_), n * sizeof(int16_t)));
    }
    if (f.harris) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&harrisResp_), n * sizeof(long long)));
        harrisKps_.allocate(cap, m);
    }
    if (f.fast) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fastScore_), n * sizeof(int)));
        fastKps_.allocate(cap, m);
    }
    if (f.orb) {
        orb_.allocate(cap, m);
        gpu::setOrbTables(cpu::orbTables());
    }
    if (f.canny) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cannyClasses_), n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cannyChangedDev_), sizeof(int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&cannyChangedHost_), sizeof(int)));
        canny_.allocate(n, m);
    }

    CUDA_CHECK(cudaStreamCreate(&stream_));
    for (auto& e : events_) CUDA_CHECK(cudaEventCreate(&e));
}

GpuPipeline::~GpuPipeline() {
    // No CUDA_CHECK in destructors: never exit() while cleaning up.
    for (auto& e : events_) if (e) cudaEventDestroy(e);
    if (stream_) cudaStreamDestroy(stream_);
    cudaFree(hist_);
    cudaFree(lut_);
    cudaFree(blurTmp_);
    cudaFree(gradX_);
    cudaFree(gradY_);
    cudaFree(harrisResp_);
    cudaFree(fastScore_);
    cudaFree(cannyClasses_);
    cudaFree(cannyChangedDev_);
    cudaFreeHost(cannyChangedHost_);
}

StageTimings GpuPipeline::process(const Image& rgb, PipelineOutputs* out) {
    if (rgb.width != cfg_.width || rgb.height != cfg_.height || rgb.channels != 3)
        throw std::invalid_argument("input image does not match pipeline configuration");

    const int w = cfg_.width, h = cfg_.height, n = w * h;

    // The CPU writes the new frame into the input buffer, like a camera
    // driver would. Not timed: it is the same in every memory mode.
    // Safe for managed memory: the GPU is idle (previous call synchronized).
    std::memcpy(rgb_.host(), rgb.data.data(), rgb.bytes());

    // Record an event BEFORE each stage; the time between two events is
    // the GPU time of that stage. No CPU waiting until the very end.
    CUDA_CHECK(cudaEventRecord(events_[kUpload], stream_));
    rgb_.toDevice(stream_);

    CUDA_CHECK(cudaEventRecord(events_[kGray], stream_));
    gpu::rgbToGray(rgb_.device(), gray_.device(), w, h, stream_);

    CUDA_CHECK(cudaEventRecord(events_[kHistogram], stream_));
    gpu::histogram(gray_.device(), hist_, n, stream_);

    CUDA_CHECK(cudaEventRecord(events_[kLut], stream_));
    gpu::buildEqualizeLut(hist_, lut_, static_cast<unsigned>(n), stream_);

    CUDA_CHECK(cudaEventRecord(events_[kApplyLut], stream_));
    gpu::applyLut(gray_.device(), equalized_.device(), lut_, n, stream_);

    CUDA_CHECK(cudaEventRecord(events_[kBlur], stream_));
    if (cfg_.blurMode == BlurMode::Naive)
        gpu::gaussianBlurNaive(equalized_.device(), blurred_.device(), w, h,
                               cfg_.blurRadius, stream_);
    else
        gpu::gaussianBlurSeparable(equalized_.device(), blurTmp_, blurred_.device(), w, h,
                                   cfg_.blurRadius, stream_);

    CUDA_CHECK(cudaEventRecord(events_[kSobel], stream_));
    gpu::sobel(blurred_.device(), edges_.device(), w, h, stream_);

    // ---------------- Feature extraction ----------------
    // Disabled features still record their event, so they show up as ~0 ms.
    const FeatureConfig& f = cfg_.features;
    const int cap = f.maxKeypoints;

    CUDA_CHECK(cudaEventRecord(events_[kHarris], stream_));
    CUDA_CHECK(cudaMemsetAsync(kpCounts_.device(), 0, kpCounts_.bytes(), stream_));
    if (f.harris || f.canny)   // gradients are shared: computed once, timed here
        gpu::sobelGradients(blurred_.device(), gradX_, gradY_, w, h, stream_);
    if (f.harris)
        gpu::harris(gradX_, gradY_, harrisResp_, w, h, f.harrisThreshold, harrisKps_.device(),
                    kpCounts_.device() + 0, cap, stream_);

    CUDA_CHECK(cudaEventRecord(events_[kFast], stream_));
    if (f.fast)
        gpu::fast(blurred_.device(), fastScore_, w, h, f.fastThreshold, fastKps_.device(),
                  kpCounts_.device() + 1, cap, stream_);

    CUDA_CHECK(cudaEventRecord(events_[kOrb], stream_));
    if (f.orb)
        gpu::orb(blurred_.device(), w, h, fastKps_.device(), kpCounts_.device() + 1, cap,
                 orb_.device(), stream_);

    CUDA_CHECK(cudaEventRecord(events_[kCanny], stream_));
    if (f.canny)
        gpu::canny(gradX_, gradY_, cannyClasses_, canny_.device(), w, h, f.cannyLow,
                   f.cannyHigh, cannyChangedDev_, cannyChangedHost_, stream_);

    CUDA_CHECK(cudaEventRecord(events_[kDownload], stream_));
    gray_.toHost(stream_);
    equalized_.toHost(stream_);
    blurred_.toHost(stream_);
    edges_.toHost(stream_);
    // Feature buffers (no-ops for disabled features: they are empty)
    kpCounts_.toHost(stream_);
    harrisKps_.toHost(stream_);
    fastKps_.toHost(stream_);
    orb_.toHost(stream_);
    canny_.toHost(stream_);

    CUDA_CHECK(cudaEventRecord(events_[kNumStages], stream_));
    CUDA_CHECK(cudaEventSynchronize(events_[kNumStages]));

    // Jetson has no concurrent managed access: make sure the whole GPU is
    // idle before the CPU touches managed buffers below. (On a laptop the
    // event sync above is enough, because our stream is the only work.)
    if (deviceSyncBeforeCpu_) CUDA_CHECK(cudaDeviceSynchronize());

    StageTimings t;
    for (int s = 0; s < kNumStages; ++s)
        CUDA_CHECK(cudaEventElapsedTime(&t.ms[s], events_[s], events_[s + 1]));
    CUDA_CHECK(cudaEventElapsedTime(&t.totalMs, events_[0], events_[kNumStages]));

    if (out) {
        auto copyOut = [w, h](const GpuBuffer<uint8_t>& src, Image& dst) {
            dst = Image(w, h, 1);
            std::memcpy(dst.data.data(), src.host(), dst.bytes());
        };
        copyOut(gray_, out->gray);
        copyOut(equalized_, out->equalized);
        copyOut(blurred_, out->blurred);
        copyOut(edges_, out->edges);

        // Keypoint lists: the GPU counted ALL keypoints, but stored at most `cap`
        const unsigned* counts = kpCounts_.host();
        auto copyList = [](const auto& buf, size_t count, auto& dst) {
            dst.assign(buf.host(), buf.host() + count);
        };
        out->harrisFound = f.harris ? counts[0] : 0;
        out->fastFound = f.fast ? counts[1] : 0;
        const size_t nHarris = std::min<size_t>(out->harrisFound, cap);
        const size_t nFast = std::min<size_t>(out->fastFound, cap);
        out->harris.clear();
        out->fast.clear();
        out->orb.clear();
        out->canny = Image();
        if (f.harris) copyList(harrisKps_, nHarris, out->harris);
        if (f.fast) copyList(fastKps_, nFast, out->fast);
        if (f.orb) copyList(orb_, nFast, out->orb);
        if (f.canny) copyOut(canny_, out->canny);
    }
    return t;
}
