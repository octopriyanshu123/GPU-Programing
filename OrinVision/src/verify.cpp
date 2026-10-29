// =====================================================================
// verify.cpp - Compare GPU results with the CPU reference
// =====================================================================
#include "verify.hpp"

#include "cpu_filters.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <tuple>

CompareResult compareImages(const Image& a, const Image& b, int tolerance) {
    CompareResult r;
    r.tolerance = tolerance;
    r.sizeOk = a.width == b.width && a.height == b.height &&
               a.channels == b.channels && a.bytes() == b.bytes();
    if (!r.sizeOk) return r;

    r.total = a.bytes();
    for (size_t i = 0; i < r.total; ++i) {
        const int d = std::abs(int(a.data[i]) - int(b.data[i]));
        if (d > 0) ++r.mismatches;
        if (d > r.maxDiff) r.maxDiff = d;
    }
    r.pass = r.maxDiff <= tolerance;
    return r;
}

namespace {

bool keypointLess(const Keypoint& a, const Keypoint& b) {
    return std::tie(a.y, a.x, a.score) < std::tie(b.y, b.x, b.score);
}
bool keypointEqual(const Keypoint& a, const Keypoint& b) {
    return a.x == b.x && a.y == b.y && a.score == b.score;
}

}  // namespace

CompareResult compareKeypoints(const std::vector<Keypoint>& gpu, unsigned gpuFound,
                               const std::vector<Keypoint>& cpu) {
    CompareResult r;
    r.sizeOk = true;
    r.total = cpu.size();

    std::vector<Keypoint> g = gpu, c = cpu;
    std::sort(g.begin(), g.end(), keypointLess);
    std::sort(c.begin(), c.end(), keypointLess);
    for (size_t i = 0; i < g.size(); ++i) {
        const bool duplicate = i > 0 && keypointEqual(g[i], g[i - 1]);
        const bool inCpu = std::binary_search(c.begin(), c.end(), g[i], keypointLess);
        if (duplicate || !inCpu) ++r.mismatches;
    }
    const long long countDiff = std::llabs((long long)gpuFound - (long long)cpu.size());
    r.maxDiff = static_cast<int>(std::min<long long>(countDiff, INT_MAX));
    r.mismatches += static_cast<size_t>(countDiff);
    r.pass = r.mismatches == 0;
    return r;
}

CompareResult compareOrb(const std::vector<OrbFeature>& gpu, const std::vector<OrbFeature>& cpu) {
    CompareResult r;
    r.sizeOk = gpu.size() == cpu.size();
    if (!r.sizeOk) return r;
    r.total = gpu.size();
    for (size_t i = 0; i < gpu.size(); ++i) {
        const OrbFeature& a = gpu[i];
        const OrbFeature& b = cpu[i];
        int hamming = 0;
        for (int k = 0; k < kOrbBytes; ++k)
            hamming += __builtin_popcount(static_cast<unsigned>(a.desc[k] ^ b.desc[k]));
        const bool same = a.x == b.x && a.y == b.y && a.angleBin == b.angleBin && hamming == 0;
        if (!same) ++r.mismatches;
        r.maxDiff = std::max(r.maxDiff, hamming);
    }
    r.pass = r.mismatches == 0;
    return r;
}

VerifyReport verifyStages(const Image& rgb, const PipelineOutputs& gpu,
                          const std::vector<float>& weights, const FeatureConfig& features) {
    const int w = rgb.width, h = rgb.height, n = w * h;
    VerifyReport rep;

    // Stage 1: grayscale from the original RGB input
    Image ref(w, h, 1);
    cpu::rgbToGray(rgb.data.data(), ref.data.data(), w, h);
    rep.items.push_back({"grayscale", "pixels", compareImages(gpu.gray, ref, 0)});

    // Stage 2: equalization, using the GPU's grayscale as input
    unsigned hist[256];
    uint8_t lut[256];
    cpu::histogram(gpu.gray.data.data(), hist, n);
    cpu::buildEqualizeLut(hist, lut, static_cast<unsigned>(n));
    cpu::applyLut(gpu.gray.data.data(), ref.data.data(), lut, n);
    rep.items.push_back({"equalized", "pixels", compareImages(gpu.equalized, ref, 0)});

    // Stage 3: blur, using the GPU's equalized image as input
    cpu::gaussianBlur(gpu.equalized.data.data(), ref.data.data(), w, h, weights);
    rep.items.push_back({"gaussian blur", "pixels", compareImages(gpu.blurred, ref, 1)});

    // Stage 4: Sobel, using the GPU's blurred image as input
    cpu::sobel(gpu.blurred.data.data(), ref.data.data(), w, h);
    rep.items.push_back({"sobel edges", "pixels", compareImages(gpu.edges, ref, 0)});

    // Feature extraction: all use the GPU's blurred image as input
    const uint8_t* blurred = gpu.blurred.data.data();
    if (features.harris) {
        const auto c = cpu::harris(blurred, w, h, features.harrisThreshold);
        rep.items.push_back({"harris corners", "keypoints",
                             compareKeypoints(gpu.harris, gpu.harrisFound, c)});
    }
    if (features.fast || features.orb) {
        const auto c = cpu::fast(blurred, w, h, features.fastThreshold);
        rep.items.push_back({"fast corners", "keypoints",
                             compareKeypoints(gpu.fast, gpu.fastFound, c)});
    }
    if (features.orb) {
        // Describe the GPU's own FAST corners on the CPU (stage isolation)
        const auto c = cpu::orb(blurred, w, h, gpu.fast);
        rep.items.push_back({"orb descriptors", "descriptors", compareOrb(gpu.orb, c)});
    }
    if (features.canny) {
        cpu::canny(blurred, ref.data.data(), w, h, features.cannyLow, features.cannyHigh);
        rep.items.push_back({"canny edges", "pixels", compareImages(gpu.canny, ref, 0)});
    }
    return rep;
}
