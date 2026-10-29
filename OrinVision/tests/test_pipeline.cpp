// =====================================================================
// test_pipeline.cpp - Automated correctness tests (run with: ctest)
// =====================================================================
// Runs the GPU pipeline on many image sizes, in every memory mode and
// blur mode, and checks each stage against the CPU reference.
// Odd sizes (1x1, 7x5, 1023x769...) catch bugs at image borders and in
// partially-filled blocks, which is where most CUDA bugs hide.
// =====================================================================
#include "cpu_filters.hpp"
#include "pipeline.hpp"
#include "verify.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>

namespace {

int g_failures = 0;
int g_tests = 0;

void check(bool ok, const std::string& name) {
    ++g_tests;
    if (!ok) ++g_failures;
    std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", name.c_str());
}

void testPipeline(int w, int h, MemoryMode mem, BlurMode blur, int radius,
                  const FeatureConfig& features = FeatureConfig{}) {
    PipelineConfig cfg;
    cfg.width = w;
    cfg.height = h;
    cfg.memoryMode = mem;
    cfg.blurMode = blur;
    cfg.blurRadius = radius;
    cfg.features = features;

    const Image rgb = imageio::generateTestImage(w, h, 1234u + w * 7u + h);
    GpuPipeline pipeline(cfg);
    PipelineOutputs out;
    pipeline.process(rgb, &out);
    pipeline.process(rgb, &out);   // second frame must also be correct (buffer reuse)

    const VerifyReport rep = verifyStages(rgb, out, pipeline.weights(), features);
    for (const VerifyItem& item : rep.items) {
        char name[200];
        std::snprintf(name, sizeof(name), "%dx%d %-8s %-9s r=%-2d %-15s maxdiff=%d (%zu %s)",
                      w, h, toString(mem), toString(blur), radius, item.name.c_str(),
                      item.result.maxDiff, item.result.total, item.unit);
        check(item.result.pass, name);
    }
}

// ---------------------------------------------------------------------
// Feature extraction sanity tests on a simple synthetic picture: a bright
// square on a dark background has exactly 4 corners and a closed outline.
// ---------------------------------------------------------------------
Image squareImage(int w, int h, int x0, int y0, int size) {
    Image img(w, h, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const bool in = x >= x0 && x < x0 + size && y >= y0 && y < y0 + size;
            img.data[size_t(y) * w + x] = in ? 200 : 40;
        }
    return img;
}

bool near(const Keypoint& k, int x, int y, int dist) {
    return std::abs(k.x - x) <= dist && std::abs(k.y - y) <= dist;
}

void testFeaturesOnSquare() {
    const int w = 80, h = 60, x0 = 20, y0 = 15, s = 30;
    const Image img = squareImage(w, h, x0, y0, s);
    const int corners[4][2] = {{x0, y0}, {x0 + s - 1, y0}, {x0, y0 + s - 1}, {x0 + s - 1, y0 + s - 1}};

    auto cornersFound = [&](const std::vector<Keypoint>& kps) {
        for (const auto& c : corners) {
            bool any = false;
            for (const auto& k : kps) any = any || near(k, c[0], c[1], 3);
            if (!any) return false;
        }
        for (const auto& k : kps) {   // and nothing far away from a corner
            bool ok = false;
            for (const auto& c : corners) ok = ok || near(k, c[0], c[1], 3);
            if (!ok) return false;
        }
        return true;
    };

    const FeatureConfig def;
    const auto harris = cpu::harris(img.data.data(), w, h, def.harrisThreshold);
    check(cornersFound(harris), "harris finds the 4 square corners (" +
                                    std::to_string(harris.size()) + " keypoints)");
    const auto fast = cpu::fast(img.data.data(), w, h, def.fastThreshold);
    check(cornersFound(fast), "fast finds the 4 square corners (" +
                                  std::to_string(fast.size()) + " keypoints)");

    // Canny: the outline is closed and the inside of the square is empty
    Image edges(w, h, 1);
    cpu::canny(img.data.data(), edges.data.data(), w, h, def.cannyLow, def.cannyHigh);
    auto at = [&](int x, int y) { return edges.data[size_t(y) * w + x]; };
    bool outline = true;
    for (int x = x0 + 2; x < x0 + s - 2; ++x) {   // top and bottom side
        bool top = false, bottom = false;
        for (int d = -1; d <= 1; ++d) {
            top = top || at(x, y0 + d) == 255;
            bottom = bottom || at(x, y0 + s - 1 + d) == 255;
        }
        outline = outline && top && bottom;
    }
    check(outline, "canny draws the top/bottom outline of the square");
    check(at(x0 + s / 2, y0 + s / 2) == 0 && at(5, 5) == 0, "canny leaves flat areas empty");

    // ORB: shifting the picture must not change a corner's descriptor
    const Image big = squareImage(120, 100, 45, 40, 30);
    const Image moved = squareImage(120, 100, 52, 45, 30);   // +7, +5
    Keypoint a, b;
    a.x = 45; a.y = 40;
    b.x = 52; b.y = 45;
    const auto da = cpu::orb(big.data.data(), 120, 100, {a});
    const auto db = cpu::orb(moved.data.data(), 120, 100, {b});
    const bool sameDesc = da[0].angleBin >= 0 && da[0].angleBin == db[0].angleBin &&
                          std::equal(da[0].desc, da[0].desc + kOrbBytes, db[0].desc);
    check(sameDesc, "orb descriptor does not change when the image is shifted");

    // ORB tables: rotated points must stay inside the border
    const OrbTables& t = cpu::orbTables();
    int maxCoord = 0;
    for (const auto& bin : t.pattern)
        for (const auto& p : bin)
            for (int c = 0; c < 4; ++c) maxCoord = std::max(maxCoord, std::abs(int(p[c])));
    check(maxCoord < kOrbBorder, "orb rotated pattern fits in the border (max |coord| = " +
                                     std::to_string(maxCoord) + ")");
}

void testImageRoundTrip() {
    const Image a = imageio::generateTestImage(37, 21);
    const std::string path =
        (std::filesystem::temp_directory_path() / "orin_vision_test.ppm").string();
    std::string err;
    Image b;
    const bool ok = imageio::save(path, a, err) && imageio::load(path, b, err) &&
                    compareImages(a, b, 0).pass;
    std::filesystem::remove(path);
    check(ok, "PPM save/load round trip" + (err.empty() ? "" : " (" + err + ")"));
}

void testEqualizeFlatImage() {
    // A completely flat image must not be changed by equalization
    unsigned hist[256] = {};
    hist[100] = 50;
    uint8_t lut[256];
    cpu::buildEqualizeLut(hist, lut, 50);
    check(lut[100] == 100, "equalization keeps a flat image unchanged");
}

}  // namespace

int main() {
    try {
        initCuda(true);
        std::printf("\nUnit tests\n");
        testImageRoundTrip();
        testEqualizeFlatImage();
        testFeaturesOnSquare();

        std::printf("\nPipeline tests (GPU vs CPU)\n");
        const int sizes[][2] = {{1, 1}, {7, 5}, {33, 17}, {640, 480}, {1023, 769}};
        const MemoryMode mems[] = {MemoryMode::Device, MemoryMode::Managed, MemoryMode::ZeroCopy};
        const BlurMode blurs[] = {BlurMode::Naive, BlurMode::Separable};

        for (const auto& s : sizes)
            for (MemoryMode m : mems)
                for (BlurMode b : blurs)
                    testPipeline(s[0], s[1], m, b, 4);

        // Radius edge cases (smallest, largest, larger than the image)
        for (int r : {1, 15})
            for (BlurMode b : blurs)
                testPipeline(97, 61, MemoryMode::Managed, b, r);
        testPipeline(5, 3, MemoryMode::Managed, BlurMode::Separable, 15);

        std::printf("\nFeature extraction edge cases (GPU vs CPU)\n");
        // Tiny keypoint list: the GPU list overflows, the found-count must still match
        FeatureConfig small;
        small.maxKeypoints = 5;
        testPipeline(640, 480, MemoryMode::Device, BlurMode::Separable, 4, small);
        // Each feature alone (checks that ORB pulls in FAST, Canny works without Harris)
        for (const char* only : {"harris", "fast", "orb", "canny"}) {
            FeatureConfig f;
            f.harris = std::string(only) == "harris";
            f.fast = std::string(only) == "fast" || std::string(only) == "orb";
            f.orb = std::string(only) == "orb";
            f.canny = std::string(only) == "canny";
            testPipeline(333, 222, MemoryMode::Managed, BlurMode::Separable, 2, f);
        }
        // Low thresholds = many features, stress the atomics and hysteresis
        FeatureConfig dense;
        dense.harrisThreshold = 1000000;
        dense.fastThreshold = 5;
        dense.cannyLow = 10;
        dense.cannyHigh = 40;
        dense.maxKeypoints = 1 << 20;
        testPipeline(1023, 769, MemoryMode::ZeroCopy, BlurMode::Naive, 1, dense);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }

    std::printf("\n%d / %d checks passed\n", g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
