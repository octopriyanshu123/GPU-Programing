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
    // --- Feature extraction (all read the blurred image) ---
    kHarris,       // Harris corners (time includes the shared Sobel gradients)
    kFast,         // FAST-9 corners
    kOrb,          // ORB: orientation + binary descriptor for each FAST corner
    kCanny,        // Canny edges (thin, connected edge lines)
    kDownload,     // GPU -> CPU copy (only does work in "device" memory mode)
    kNumStages
};

inline const char* stageName(int s) {
    static const char* names[kNumStages] = {
        "upload", "grayscale", "histogram", "build LUT",
        "apply LUT", "gaussian blur", "sobel edges",
        "harris corners", "fast corners", "orb descriptors", "canny edges",
        "download"};
    return (s >= 0 && s < kNumStages) ? names[s] : "?";
}

enum class BlurMode {
    Naive,       // 2D loop reading global memory directly (simple, slower)
    Separable    // two 1D passes using shared memory tiles (optimized)
};

inline const char* toString(BlurMode m) {
    return m == BlurMode::Naive ? "naive" : "separable";
}

// ---------------------------------------------------------------------
// Feature extraction types
// ---------------------------------------------------------------------
// A detected point of interest (corner).
struct Keypoint {
    int x = 0, y = 0;
    long long score = 0;   // Harris response or FAST score (higher = stronger)
};

constexpr int kFeatureBorder = 3;    // Harris/FAST ignore this many pixels at the image edge
constexpr int kOrbBins = 30;         // orientation is quantized in 360/30 = 12 degree steps
constexpr int kOrbPairs = 256;       // descriptor = 256 pixel comparisons = 256 bits
constexpr int kOrbBytes = kOrbPairs / 8;
constexpr int kOrbPatchRadius = 15;  // orientation is measured on a disk of this radius
constexpr int kOrbPatternMax = 13;   // test points lie in [-13, 13] before rotation
constexpr int kOrbBorder = 19;       // > 13 * sqrt(2): rotated points stay inside the image

// ORB feature = keypoint + orientation + 256-bit binary descriptor.
struct OrbFeature {
    int x = 0, y = 0;
    int angleBin = -1;               // 0..kOrbBins-1; -1 = too close to the border
    uint8_t desc[kOrbBytes] = {};
};

// Precomputed ORB tables (identical on CPU and GPU, built in cpu_features.cpp):
// one rotated copy of the test pattern for every orientation bin.
struct OrbTables {
    int cosQ[kOrbBins];                      // cos(angle) * 4096, rounded
    int sinQ[kOrbBins];                      // sin(angle) * 4096, rounded
    int8_t pattern[kOrbBins][kOrbPairs][4];  // x1, y1, x2, y2 of each test pair
};

struct FeatureConfig {
    bool harris = true, fast = true, orb = true, canny = true;   // orb needs fast
    long long harrisThreshold = 5000000000LL;  // on 25*R (integer Harris, k = 1/25)
    int fastThreshold = 20;     // brightness difference to the center pixel
    int cannyLow = 40;          // Sobel gradient magnitude: weak edge above this
    int cannyHigh = 100;        //                           strong edge above this
    int maxKeypoints = 16384;   // keypoint list capacity per detector

    bool any() const { return harris || fast || orb || canny; }
};

// Everything the pipeline produces.
struct PipelineOutputs {
    Image gray;        // stage 1
    Image equalized;   // stage 2 (histogram equalization = better contrast)
    Image blurred;     // stage 3
    Image edges;       // stage 4 (Sobel)

    // Feature extraction (empty when the feature is disabled)
    std::vector<Keypoint> harris;     // GPU: in atomic (random) order
    std::vector<Keypoint> fast;
    unsigned harrisFound = 0;         // may be > harris.size() if the list was full
    unsigned fastFound = 0;
    std::vector<OrbFeature> orb;      // same order as `fast` (one per FAST corner)
    Image canny;                      // 255 = edge, 0 = background
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

// =====================================================================
// Feature extraction math (shared by CPU and GPU -> bit-exact results)
// =====================================================================

// Sobel gradients only (same formula as sobelMagnitude, without sqrt).
HD inline void sobelGradient(int p00, int p01, int p02,
                             int p10,          int p12,
                             int p20, int p21, int p22, int& gx, int& gy) {
    gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
    gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
}

// ---------------- Harris ----------------
// Structure tensor of a 3x3 window: sxx = sum(gx*gx), syy = sum(gy*gy),
// sxy = sum(gx*gy). Harris response R = det - k * trace^2 with k = 0.04.
// k = 1/25, so we return 25*R = 25*det - trace^2: pure integer math.
// (Max |g| = 1020 -> sxx <= 9.4e6 -> 25*sxx*syy <= 2.2e15, fits in 64 bits.)
HD inline long long harrisResponse(long long sxx, long long syy, long long sxy) {
    return 25 * (sxx * syy - sxy * sxy) - (sxx + syy) * (sxx + syy);
}

// ---------------- Non-maximum suppression ----------------
// Keep a pixel only if it is the largest in its 3x3 neighbourhood. Ties:
// neighbours that come BEFORE it in raster order must be strictly smaller,
// neighbours AFTER it may be equal -> exactly one pixel of a flat peak wins,
// the same one on CPU and GPU. idx must not be on the image border.
template <typename T>
HD inline bool isPeak3x3(const T* s, int idx, int w) {
    const T c = s[idx];
    return c > s[idx - w - 1] && c > s[idx - w] && c > s[idx - w + 1] && c > s[idx - 1] &&
           c >= s[idx + 1] && c >= s[idx + w - 1] && c >= s[idx + w] && c >= s[idx + w + 1];
}

// ---------------- FAST ----------------
// 16 pixels on a circle of radius 3 around the center (clockwise from top).
HD inline void fastCircleOffset(int i, int& dx, int& dy) {
    const int X[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    const int Y[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
    dx = X[i];
    dy = Y[i];
}

// True if the 16-bit circular mask has 9 or more set bits in a row.
HD inline bool hasArc9(unsigned mask16) {
    const unsigned d = mask16 | (mask16 << 16);   // repeat to handle the wrap-around
    unsigned run = d;
    for (int i = 1; i < 9; ++i) run &= d >> i;     // bit j survives if bits j..j+8 are set
    return (run & 0xFFFFu) != 0;
}

// FAST-9: corner if >= 9 CONTIGUOUS ring pixels are all brighter than
// center + t, or all darker than center - t. Returns 0 (not a corner) or a
// score = how much the arc pixels exceed the threshold (used for NMS).
HD inline int fastScore(const int ring[16], int center, int t) {
    unsigned bright = 0, dark = 0;
    int sumBright = 0, sumDark = 0;
    for (int i = 0; i < 16; ++i) {
        const int d = ring[i] - center;
        if (d > t)  { bright |= 1u << i; sumBright += d - t; }
        if (d < -t) { dark |= 1u << i;   sumDark += -d - t; }
    }
    int score = 0;
    if (hasArc9(bright)) score = sumBright;
    if (hasArc9(dark) && sumDark > score) score = sumDark;
    return score;
}

// ---------------- ORB ----------------
// Half width of the circular patch at row dy: largest u with u^2+dy^2 <= R^2.
HD inline int orbHalfWidth(int dy) {
    int u = 0;
    while ((u + 1) * (u + 1) + dy * dy <= kOrbPatchRadius * kOrbPatchRadius) ++u;
    return u;
}

// Intensity moments of ONE row of the circular patch (the GPU gives each
// row to a different thread; the CPU loops over all rows).
HD inline void orbMomentsRow(const uint8_t* img, int w, int x, int y, int dy,
                             int& m10, int& m01) {
    const int u = orbHalfWidth(dy);
    const uint8_t* row = img + (y + dy) * w + x;
    int sx = 0, s = 0;
    for (int dx = -u; dx <= u; ++dx) {
        sx += dx * row[dx];
        s += row[dx];
    }
    m10 += sx;        // sum of x * I
    m01 += dy * s;    // sum of y * I
}

// Orientation bin = the bin whose direction best matches the intensity
// centroid (m10, m01): the largest dot product. No atan2 -> no float
// rounding differences between CPU and GPU. Ties pick the lower bin.
HD inline int orbAngleBin(int m10, int m01, const int* cosQ, const int* sinQ) {
    int best = 0;
    long long bestDot = (long long)m10 * cosQ[0] + (long long)m01 * sinQ[0];
    for (int k = 1; k < kOrbBins; ++k) {
        const long long d = (long long)m10 * cosQ[k] + (long long)m01 * sinQ[k];
        if (d > bestDot) { bestDot = d; best = k; }
    }
    return best;
}

// One byte (8 bits) of the BRIEF descriptor: bit = I(p1) < I(p2) for 8
// rotated test pairs. pattern = the pairs of one orientation bin.
HD inline uint8_t orbDescriptorByte(const uint8_t* img, int w, int x, int y,
                                    const int8_t* pattern, int byteIdx) {
    uint8_t b = 0;
    for (int bit = 0; bit < 8; ++bit) {
        const int8_t* p = pattern + (byteIdx * 8 + bit) * 4;
        const int v1 = img[(y + p[1]) * w + (x + p[0])];
        const int v2 = img[(y + p[3]) * w + (x + p[2])];
        if (v1 < v2) b |= (uint8_t)(1u << bit);
    }
    return b;
}

HD inline bool orbInside(int x, int y, int w, int h) {
    return x >= kOrbBorder && y >= kOrbBorder && x < w - kOrbBorder && y < h - kOrbBorder;
}

// ---------------- Canny ----------------
// Pick the neighbour pair along the gradient direction with an integer
// tangent test (tan 22.5 deg ~ 13573 / 2^15, same trick as OpenCV).
// The pair is (x - dx, y - dy) and (x + dx, y + dy).
HD inline void cannyDirection(int gx, int gy, int& dx, int& dy) {
    const int ax = gx < 0 ? -gx : gx;
    const int ay15 = (gy < 0 ? -gy : gy) << 15;
    const int tg22 = ax * 13573;
    const int tg67 = tg22 + (ax << 16);            // tan 67.5 = tan 22.5 + 2
    if (ay15 < tg22)      { dx = 1; dy = 0; }        // gradient ~horizontal: left/right
    else if (ay15 > tg67) { dx = 0; dy = 1; }        // gradient ~vertical: up/down
    else { dx = ((gx ^ gy) < 0) ? -1 : 1; dy = 1; }  // diagonal
}

// Thin + threshold one pixel (squared magnitudes, so no sqrt):
// 0 = not an edge, 1 = weak edge, 2 = strong edge.
HD inline uint8_t cannyClass(int m2, int prev2, int next2, int low2, int high2) {
    if (m2 <= low2 || !(m2 > prev2 && m2 >= next2)) return 0;
    return m2 > high2 ? 2 : 1;
}
