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

#include <cstdio>
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

void testPipeline(int w, int h, MemoryMode mem, BlurMode blur, int radius) {
    PipelineConfig cfg;
    cfg.width = w;
    cfg.height = h;
    cfg.memoryMode = mem;
    cfg.blurMode = blur;
    cfg.blurRadius = radius;

    const Image rgb = imageio::generateTestImage(w, h, 1234u + w * 7u + h);
    GpuPipeline pipeline(cfg);
    PipelineOutputs out;
    pipeline.process(rgb, &out);
    pipeline.process(rgb, &out);   // second frame must also be correct (buffer reuse)

    const VerifyReport rep = verifyStages(rgb, out, pipeline.weights());
    for (int i = 0; i < VerifyReport::kCount; ++i) {
        const CompareResult& r = rep.results[i];
        char name[160];
        std::snprintf(name, sizeof(name), "%dx%d %-8s %-9s r=%-2d %-13s maxdiff=%d",
                      w, h, toString(mem), toString(blur), radius, rep.names[i], r.maxDiff);
        check(r.pass, name);
    }
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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }

    std::printf("\n%d / %d checks passed\n", g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
