// =====================================================================
// verify.cpp - Compare GPU results with the CPU reference
// =====================================================================
#include "verify.hpp"

#include "cpu_filters.hpp"

#include <cstdlib>

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

VerifyReport verifyStages(const Image& rgb, const PipelineOutputs& gpu,
                          const std::vector<float>& weights) {
    const int w = rgb.width, h = rgb.height, n = w * h;
    VerifyReport rep;

    // Stage 1: grayscale from the original RGB input
    Image ref(w, h, 1);
    cpu::rgbToGray(rgb.data.data(), ref.data.data(), w, h);
    rep.results[0] = compareImages(gpu.gray, ref, 0);

    // Stage 2: equalization, using the GPU's grayscale as input
    unsigned hist[256];
    uint8_t lut[256];
    cpu::histogram(gpu.gray.data.data(), hist, n);
    cpu::buildEqualizeLut(hist, lut, static_cast<unsigned>(n));
    cpu::applyLut(gpu.gray.data.data(), ref.data.data(), lut, n);
    rep.results[1] = compareImages(gpu.equalized, ref, 0);

    // Stage 3: blur, using the GPU's equalized image as input
    cpu::gaussianBlur(gpu.equalized.data.data(), ref.data.data(), w, h, weights);
    rep.results[2] = compareImages(gpu.blurred, ref, 1);

    // Stage 4: Sobel, using the GPU's blurred image as input
    cpu::sobel(gpu.blurred.data.data(), ref.data.data(), w, h);
    rep.results[3] = compareImages(gpu.edges, ref, 0);

    return rep;
}
