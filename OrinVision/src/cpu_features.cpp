// =====================================================================
// cpu_features.cpp - CPU reference for feature extraction
// =====================================================================
//   Harris : corners from the structure tensor of the Sobel gradients
//   FAST   : corners from a brightness test on a 16-pixel ring
//   ORB    : orientation + 256-bit BRIEF descriptor for each FAST corner
//   Canny  : thin edges with double threshold + hysteresis
//
// All the per-pixel math comes from common.hpp, so the GPU must produce
// EXACTLY the same keypoints, descriptors and edge pixels.
// =====================================================================
#include "cpu_filters.hpp"

#include <cmath>
#include <vector>

namespace {

// Collect 3x3 peaks above `threshold` in raster order (pixels closer than
// `border` to the image edge are skipped).
template <typename T>
std::vector<Keypoint> collectPeaks(const std::vector<T>& score, int w, int h, int border,
                                   T threshold) {
    std::vector<Keypoint> kps;
    for (int y = border; y < h - border; ++y)
        for (int x = border; x < w - border; ++x) {
            const int idx = y * w + x;
            if (score[idx] > threshold && isPeak3x3(score.data(), idx, w)) {
                Keypoint k;
                k.x = x;
                k.y = y;
                k.score = score[idx];
                kps.push_back(k);
            }
        }
    return kps;
}

// Floor division by 4096 (also correct for negative numbers).
int floorDiv4096(int v) { return v >= 0 ? v / 4096 : -((-v + 4095) / 4096); }

OrbTables makeOrbTables() {
    OrbTables t{};
    for (int k = 0; k < kOrbBins; ++k) {
        const double a = 2.0 * 3.14159265358979323846 * k / kOrbBins;
        t.cosQ[k] = static_cast<int>(std::lround(4096.0 * std::cos(a)));
        t.sinQ[k] = static_cast<int>(std::lround(4096.0 * std::sin(a)));
    }

    // Random test pairs, concentrated near the center (BRIEF "G II" style).
    // Integer-only generator: the sum of 4 small uniform numbers is roughly
    // Gaussian -> the pattern is identical on every machine.
    uint32_t state = 0x0B0B5EEDu;
    auto rnd = [&state]() { state = state * 1664525u + 1013904223u; return state >> 8; };
    auto coord = [&]() {
        int s = 0;
        for (int i = 0; i < 4; ++i) s += static_cast<int>(rnd() % 13) - 6;
        return clampi(s, -kOrbPatternMax, kOrbPatternMax);
    };
    int base[kOrbPairs][4];
    for (auto& p : base) {
        do {
            for (int& c : p) c = coord();
        } while (p[0] == p[2] && p[1] == p[3]);   // the two points must differ
    }

    // Rotate the pattern for every bin (integer rotation with cosQ/sinQ).
    for (int k = 0; k < kOrbBins; ++k)
        for (int i = 0; i < kOrbPairs; ++i)
            for (int j = 0; j < 4; j += 2) {
                const int x = base[i][j], y = base[i][j + 1];
                const int c = t.cosQ[k], s = t.sinQ[k];
                t.pattern[k][i][j] = static_cast<int8_t>(floorDiv4096(x * c - y * s + 2048));
                t.pattern[k][i][j + 1] = static_cast<int8_t>(floorDiv4096(x * s + y * c + 2048));
            }
    return t;
}

}  // namespace

namespace cpu {

const OrbTables& orbTables() {
    static const OrbTables tables = makeOrbTables();
    return tables;
}

void sobelGradients(const uint8_t* in, int16_t* gx, int16_t* gy, int width, int height) {
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        const int ym = clampi(y - 1, 0, height - 1) * width;
        const int y0 = y * width;
        const int yp = clampi(y + 1, 0, height - 1) * width;
        for (int x = 0; x < width; ++x) {
            const int xm = clampi(x - 1, 0, width - 1);
            const int xp = clampi(x + 1, 0, width - 1);
            int gxv, gyv;
            sobelGradient(in[ym + xm], in[ym + x], in[ym + xp],
                          in[y0 + xm],             in[y0 + xp],
                          in[yp + xm], in[yp + x], in[yp + xp], gxv, gyv);
            gx[y0 + x] = static_cast<int16_t>(gxv);
            gy[y0 + x] = static_cast<int16_t>(gyv);
        }
    }
}

std::vector<Keypoint> harris(const uint8_t* in, int width, int height, long long threshold) {
    const size_t n = size_t(width) * height;
    std::vector<int16_t> gx(n), gy(n);
    sobelGradients(in, gx.data(), gy.data(), width, height);

    // Response per pixel: sum of gradient products over a 3x3 window
    std::vector<long long> resp(n);
#pragma omp parallel for
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            long long sxx = 0, syy = 0, sxy = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int i = clampi(y + dy, 0, height - 1) * width +
                                  clampi(x + dx, 0, width - 1);
                    sxx += gx[i] * gx[i];
                    syy += gy[i] * gy[i];
                    sxy += gx[i] * gy[i];
                }
            resp[size_t(y) * width + x] = harrisResponse(sxx, syy, sxy);
        }
    return collectPeaks(resp, width, height, kFeatureBorder, threshold);
}

std::vector<Keypoint> fast(const uint8_t* in, int width, int height, int threshold) {
    std::vector<int> score(size_t(width) * height, 0);
#pragma omp parallel for
    for (int y = kFeatureBorder; y < height - kFeatureBorder; ++y)
        for (int x = kFeatureBorder; x < width - kFeatureBorder; ++x) {
            int ring[16];
            for (int i = 0; i < 16; ++i) {
                int dx, dy;
                fastCircleOffset(i, dx, dy);
                ring[i] = in[(y + dy) * width + (x + dx)];
            }
            score[size_t(y) * width + x] = fastScore(ring, in[y * width + x], threshold);
        }
    return collectPeaks(score, width, height, kFeatureBorder, 0);
}

std::vector<OrbFeature> orb(const uint8_t* in, int width, int height,
                            const std::vector<Keypoint>& keypoints) {
    const OrbTables& t = orbTables();
    std::vector<OrbFeature> out(keypoints.size());
#pragma omp parallel for
    for (int k = 0; k < static_cast<int>(keypoints.size()); ++k) {
        const int x = keypoints[k].x, y = keypoints[k].y;
        OrbFeature& f = out[k];
        f.x = x;
        f.y = y;
        if (!orbInside(x, y, width, height)) continue;   // angleBin stays -1

        int m10 = 0, m01 = 0;
        for (int dy = -kOrbPatchRadius; dy <= kOrbPatchRadius; ++dy)
            orbMomentsRow(in, width, x, y, dy, m10, m01);
        f.angleBin = orbAngleBin(m10, m01, t.cosQ, t.sinQ);
        const int8_t* pattern = &t.pattern[f.angleBin][0][0];
        for (int b = 0; b < kOrbBytes; ++b)
            f.desc[b] = orbDescriptorByte(in, width, x, y, pattern, b);
    }
    return out;
}

void canny(const uint8_t* in, uint8_t* out, int width, int height, int low, int high) {
    const size_t n = size_t(width) * height;
    std::vector<int16_t> gx(n), gy(n);
    sobelGradients(in, gx.data(), gy.data(), width, height);

    auto mag2 = [&](int x, int y) {
        const size_t i = size_t(clampi(y, 0, height - 1)) * width + clampi(x, 0, width - 1);
        return gx[i] * gx[i] + gy[i] * gy[i];
    };

    // 1) Thin the edges (non-maximum suppression) + classify weak / strong
    std::vector<uint8_t> cls(n);
#pragma omp parallel for
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const size_t i = size_t(y) * width + x;
            int dx, dy;
            cannyDirection(gx[i], gy[i], dx, dy);
            cls[i] = cannyClass(mag2(x, y), mag2(x - dx, y - dy), mag2(x + dx, y + dy),
                                low * low, high * high);
        }

    // 2) Hysteresis: weak pixels survive only if connected to a strong one.
    //    Flood fill from every strong pixel (depth-first with a stack).
    std::vector<int> stack;
    for (size_t i = 0; i < n; ++i)
        if (cls[i] == 2) stack.push_back(static_cast<int>(i));
    while (!stack.empty()) {
        const int i = stack.back();
        stack.pop_back();
        const int x = i % width, y = i / width;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
                const int j = yy * width + xx;
                if (cls[j] == 1) { cls[j] = 2; stack.push_back(j); }
            }
    }
    for (size_t i = 0; i < n; ++i) out[i] = cls[i] == 2 ? 255 : 0;
}

}  // namespace cpu
