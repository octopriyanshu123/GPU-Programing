// =====================================================================
// cpu_filters.cpp - CPU reference implementations
// =====================================================================
// Written to mirror the GPU math exactly (same helper functions from
// common.hpp, same border handling = clamp to edge).
// "#pragma omp parallel for" splits the outer loop across CPU cores when
// built with OpenMP; without OpenMP the pragma is simply ignored.
// =====================================================================
#include "cpu_filters.hpp"

#include <chrono>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<float> makeGaussianWeights(int radius, float sigma) {
    if (sigma <= 0.0f) sigma = radius > 0 ? radius / 2.0f : 1.0f;
    if (sigma < 0.5f) sigma = 0.5f;
    std::vector<float> w(2 * radius + 1);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double v = std::exp(-(double)(i * i) / (2.0 * sigma * sigma));
        w[i + radius] = static_cast<float>(v);
        sum += v;
    }
    for (auto& v : w) v = static_cast<float>(v / sum);
    return w;
}

namespace cpu {

int numThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void rgbToGray(const uint8_t* rgb, uint8_t* gray, int width, int height) {
    const int n = width * height;
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        gray[i] = grayFromRGB(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
}

void histogram(const uint8_t* gray, unsigned* hist, int numPixels) {
    for (int i = 0; i < 256; ++i) hist[i] = 0;
    for (int i = 0; i < numPixels; ++i) hist[gray[i]]++;
}

void buildEqualizeLut(const unsigned* hist, uint8_t* lut, unsigned numPixels) {
    // cdf[i] = number of pixels with brightness <= i  (inclusive prefix sum)
    unsigned cdf[256];
    unsigned running = 0;
    for (int i = 0; i < 256; ++i) { running += hist[i]; cdf[i] = running; }

    // Smallest non-zero cdf = cdf of the darkest brightness present
    unsigned cdfMin = 0xFFFFFFFFu;
    for (int i = 0; i < 256; ++i)
        if (cdf[i] > 0 && cdf[i] < cdfMin) cdfMin = cdf[i];

    for (int i = 0; i < 256; ++i) lut[i] = equalizeValue(i, cdf[i], cdfMin, numPixels);
}

void applyLut(const uint8_t* in, uint8_t* out, const uint8_t* lut, int numPixels) {
#pragma omp parallel for
    for (int i = 0; i < numPixels; ++i) out[i] = lut[in[i]];
}

void gaussianBlur(const uint8_t* in, uint8_t* out, int width, int height,
                  const std::vector<float>& weights) {
    const int r = static_cast<int>(weights.size() / 2);
    std::vector<float> tmp(size_t(width) * height);

    // Pass 1: horizontal blur (uint8 -> float)
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -r; k <= r; ++k)
                sum += weights[k + r] * in[y * width + clampi(x + k, 0, width - 1)];
            tmp[size_t(y) * width + x] = sum;
        }
    }
    // Pass 2: vertical blur (float -> uint8)
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -r; k <= r; ++k)
                sum += weights[k + r] * tmp[size_t(clampi(y + k, 0, height - 1)) * width + x];
            out[size_t(y) * width + x] = roundToByte(sum);
        }
    }
}

void sobel(const uint8_t* in, uint8_t* out, int width, int height) {
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        const int ym = clampi(y - 1, 0, height - 1) * width;
        const int y0 = y * width;
        const int yp = clampi(y + 1, 0, height - 1) * width;
        for (int x = 0; x < width; ++x) {
            const int xm = clampi(x - 1, 0, width - 1);
            const int xp = clampi(x + 1, 0, width - 1);
            out[y0 + x] = sobelMagnitude(in[ym + xm], in[ym + x], in[ym + xp],
                                         in[y0 + xm],             in[y0 + xp],
                                         in[yp + xm], in[yp + x], in[yp + xp]);
        }
    }
}

void runPipeline(const Image& rgb, const std::vector<float>& weights,
                 PipelineOutputs& out, double stageMs[kNumStages]) {
    using clock = std::chrono::steady_clock;
    auto msSince = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    const int w = rgb.width, h = rgb.height, n = w * h;
    out.gray = Image(w, h, 1);
    out.equalized = Image(w, h, 1);
    out.blurred = Image(w, h, 1);
    out.edges = Image(w, h, 1);
    unsigned hist[256];
    uint8_t lut[256];

    for (int s = 0; s < kNumStages; ++s) stageMs[s] = 0.0;

    auto t = clock::now();
    rgbToGray(rgb.data.data(), out.gray.data.data(), w, h);
    stageMs[kGray] = msSince(t);

    t = clock::now();
    histogram(out.gray.data.data(), hist, n);
    stageMs[kHistogram] = msSince(t);

    t = clock::now();
    buildEqualizeLut(hist, lut, static_cast<unsigned>(n));
    stageMs[kLut] = msSince(t);

    t = clock::now();
    applyLut(out.gray.data.data(), out.equalized.data.data(), lut, n);
    stageMs[kApplyLut] = msSince(t);

    t = clock::now();
    gaussianBlur(out.equalized.data.data(), out.blurred.data.data(), w, h, weights);
    stageMs[kBlur] = msSince(t);

    t = clock::now();
    sobel(out.blurred.data.data(), out.edges.data.data(), w, h);
    stageMs[kSobel] = msSince(t);
}

}  // namespace cpu
