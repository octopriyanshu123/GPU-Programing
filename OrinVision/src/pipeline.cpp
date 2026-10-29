// =====================================================================
// pipeline.cpp - Runs all GPU stages and times them with CUDA events
// =====================================================================
#include "pipeline.hpp"

#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

void initCuda(bool printInfo) {
    // Allow zero-copy (mapped) host memory. Must be set before the CUDA
    // context is created; on Jetson this is effectively always enabled.
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess) CUDA_CHECK(err);
    cudaGetLastError();   // clear a possible "already active" error

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    if (!printInfo) return;
    int runtimeVer = 0, driverVer = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVer));
    CUDA_CHECK(cudaDriverGetVersion(&driverVer));
    std::printf("GPU            : %s (compute %d.%d, %d SMs)\n", p.name, p.major, p.minor,
                p.multiProcessorCount);
    std::printf("Memory         : %.2f GB %s\n", p.totalGlobalMem / 1e9,
                p.integrated ? "(shared with CPU - integrated GPU)" : "(dedicated VRAM)");
    std::printf("CUDA runtime   : %d.%d   driver: %d.%d\n", runtimeVer / 1000,
                (runtimeVer % 1000) / 10, driverVer / 1000, (driverVer % 1000) / 10);
    std::printf("Managed memory : concurrent CPU access %s\n",
                p.concurrentManagedAccess ? "YES" : "NO (sync before CPU access)");
}

GpuPipeline::GpuPipeline(const PipelineConfig& config) : cfg_(config) {
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

    CUDA_CHECK(cudaEventRecord(events_[kDownload], stream_));
    gray_.toHost(stream_);
    equalized_.toHost(stream_);
    blurred_.toHost(stream_);
    edges_.toHost(stream_);

    CUDA_CHECK(cudaEventRecord(events_[kNumStages], stream_));
    CUDA_CHECK(cudaEventSynchronize(events_[kNumStages]));

    // Jetson has no concurrent managed access: make sure the whole GPU is
    // idle before the CPU touches managed buffers below.
    CUDA_CHECK(cudaDeviceSynchronize());

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
    }
    return t;
}
