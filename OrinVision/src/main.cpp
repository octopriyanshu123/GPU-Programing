// =====================================================================
// main.cpp - OrinVision command-line application
// =====================================================================
// 1. Load (or generate) an RGB image
// 2. Run the GPU pipeline many times and collect per-stage timings
// 3. Run the CPU reference for a speed comparison
// 4. Verify every GPU stage against the CPU
// 5. Print a report, save result images and append to a CSV file
//
// Designed for headless use over SSH: no window, everything goes to the
// terminal and to files in the output folder.
// =====================================================================
#include "cpu_filters.hpp"
#include "pipeline.hpp"
#include "verify.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input;              // empty -> generate a test image
    int genWidth = 1920, genHeight = 1080;
    std::string outDir = "results";
    std::string csv;                // empty -> no CSV
    int iters = 20, warmup = 3, cpuIters = 3;
    MemoryMode mem = MemoryMode::Managed;
    bool memAuto = true;            // pick `mem` from the detected platform
    BlurMode blur = BlurMode::Separable;
    int radius = 4;
    float sigma = 0.0f;
    bool runCpu = true, verify = true, save = true, quiet = false;
};

void printUsage(const char* prog) {
    std::printf(
        "OrinVision - GPU image processing pipeline & benchmark for Jetson and PC GPUs\n\n"
        "Usage: %s [options]\n\n"
        "Input:\n"
        "  -i, --input FILE      image to process: JPG, PNG, BMP, TGA, GIF, PPM, PGM\n"
        "  -g, --generate WxH    generate a synthetic test image (default 1920x1080)\n\n"
        "Pipeline:\n"
        "  -m, --mem MODE        auto | device | managed | zerocopy   (default auto:\n"
        "                        managed on Jetson, device on a PC/laptop GPU)\n"
        "  -b, --blur MODE       naive | separable             (default separable)\n"
        "  -r, --radius R        blur radius 1..%d             (default 4)\n"
        "  -s, --sigma S         blur sigma, 0 = radius/2      (default 0)\n\n"
        "Benchmark:\n"
        "  -n, --iters N         timed GPU iterations          (default 20)\n"
        "  -w, --warmup N        untimed warm-up iterations    (default 3)\n"
        "      --cpu-iters N     CPU reference iterations      (default 3)\n"
        "      --no-cpu          skip CPU timing\n"
        "      --no-verify       skip correctness check\n\n"
        "Output:\n"
        "  -o, --output DIR      folder for result images      (default results)\n"
        "      --no-save         do not write images\n"
        "      --csv FILE        append timings to a CSV file\n"
        "  -q, --quiet           print only the summary line\n"
        "  -h, --help            show this help\n\n"
        "Examples:\n"
        "  %s                                   # 1080p test image, defaults\n"
        "  %s -g 3840x2160 -m device -n 50      # 4K, classic cudaMemcpy mode\n"
        "  %s -i photo.ppm -r 6 -o out          # your own image\n",
        prog, kMaxBlurRadius, prog, prog, prog);
}

bool parseSize(const std::string& s, int& w, int& h) {
    const auto x = s.find('x');
    if (x == std::string::npos) return false;
    try {
        w = std::stoi(s.substr(0, x));
        h = std::stoi(s.substr(x + 1));
    } catch (...) { return false; }
    return w > 0 && h > 0 && w <= 16384 && h <= 16384;
}

// Returns 0 = run, 1 = error, 2 = help shown
int parseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        auto intValue = [&](const char* name, int lo, int hi) {
            const std::string v = value(name);
            int n = 0;
            try { n = std::stoi(v); } catch (...) {
                throw std::runtime_error(std::string("invalid number for ") + name + ": " + v);
            }
            if (n < lo || n > hi)
                throw std::runtime_error(std::string(name) + " must be between " +
                                         std::to_string(lo) + " and " + std::to_string(hi));
            return n;
        };

        if (a == "-h" || a == "--help") { printUsage(argv[0]); return 2; }
        else if (a == "-i" || a == "--input") o.input = value("--input");
        else if (a == "-g" || a == "--generate") {
            if (!parseSize(value("--generate"), o.genWidth, o.genHeight))
                throw std::runtime_error("--generate expects WIDTHxHEIGHT, e.g. 1920x1080");
        }
        else if (a == "-m" || a == "--mem") {
            const std::string v = value("--mem");
            o.memAuto = (v == "auto");
            if (!o.memAuto && !parseMemoryMode(v, o.mem))
                throw std::runtime_error("unknown memory mode: " + v);
        }
        else if (a == "-b" || a == "--blur") {
            const std::string v = value("--blur");
            if (v == "naive") o.blur = BlurMode::Naive;
            else if (v == "separable") o.blur = BlurMode::Separable;
            else throw std::runtime_error("unknown blur mode: " + v);
        }
        else if (a == "-r" || a == "--radius") o.radius = intValue("--radius", 1, kMaxBlurRadius);
        else if (a == "-s" || a == "--sigma") {
            try { o.sigma = std::stof(value("--sigma")); }
            catch (...) { throw std::runtime_error("invalid --sigma"); }
        }
        else if (a == "-n" || a == "--iters") o.iters = intValue("--iters", 1, 100000);
        else if (a == "-w" || a == "--warmup") o.warmup = intValue("--warmup", 0, 1000);
        else if (a == "--cpu-iters") o.cpuIters = intValue("--cpu-iters", 1, 1000);
        else if (a == "--no-cpu") o.runCpu = false;
        else if (a == "--no-verify") o.verify = false;
        else if (a == "-o" || a == "--output") o.outDir = value("--output");
        else if (a == "--no-save") o.save = false;
        else if (a == "--csv") o.csv = value("--csv");
        else if (a == "-q" || a == "--quiet") o.quiet = true;
        else throw std::runtime_error("unknown option: " + a + "  (use --help)");
    }
    return 0;
}

struct Stats {
    double sum = 0.0;
    double min = std::numeric_limits<double>::max();
    int count = 0;
    void add(double v) { sum += v; min = std::min(min, v); ++count; }
    double avg() const { return count ? sum / count : 0.0; }
    double minOrZero() const { return count ? min : 0.0; }
};

// Minimum bytes each stage must read + write (for effective bandwidth).
double stageBytes(int stage, size_t n, MemoryMode mem) {
    switch (stage) {
        case kUpload:    return mem == MemoryMode::Device ? 3.0 * n : 0.0;
        case kGray:      return 4.0 * n;   // read RGB (3) + write gray (1)
        case kHistogram: return 1.0 * n;
        case kApplyLut:  return 2.0 * n;
        case kBlur:      return 2.0 * n;
        case kSobel:     return 2.0 * n;
        case kDownload:  return mem == MemoryMode::Device ? 4.0 * n : 0.0;
        default:         return 0.0;
    }
}

std::string timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

void appendCsv(const Options& o, const Image& img, const Stats gpu[], const Stats& gpuTotal,
               const Stats cpu[], double cpuTotal, bool verified) {
    const bool newFile = !fs::exists(o.csv) || fs::file_size(o.csv) == 0;
    std::ofstream f(o.csv, std::ios::app);
    if (!f) { std::fprintf(stderr, "warning: cannot write CSV %s\n", o.csv.c_str()); return; }
    if (newFile)
        f << "timestamp,width,height,mem_mode,blur_mode,radius,iters,stage,"
             "gpu_avg_ms,gpu_min_ms,cpu_ms,speedup,eff_bw_gbps,verified\n";

    const std::string ts = timestamp();
    const size_t n = img.pixels();
    auto row = [&](const std::string& stage, double gAvg, double gMin, double cMs, double bytes) {
        char line[512];
        const double speedup = (cMs > 0 && gAvg > 0) ? cMs / gAvg : 0.0;
        const double bw = (bytes > 0 && gAvg > 0) ? bytes / (gAvg * 1e6) : 0.0;
        std::snprintf(line, sizeof(line), "%s,%d,%d,%s,%s,%d,%d,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%d\n",
                      ts.c_str(), img.width, img.height, toString(o.mem), toString(o.blur),
                      o.radius, o.iters, stage.c_str(), gAvg, gMin, cMs, speedup, bw,
                      verified ? 1 : 0);
        f << line;
    };
    for (int s = 0; s < kNumStages; ++s)
        row(stageName(s), gpu[s].avg(), gpu[s].minOrZero(), o.runCpu ? cpu[s].avg() : 0.0,
            stageBytes(s, n, o.mem));
    row("total", gpuTotal.avg(), gpuTotal.minOrZero(), o.runCpu ? cpuTotal : 0.0, 0.0);
}

void saveImage(const fs::path& path, const Image& img) {
    std::string err;
    if (!imageio::save(path.string(), img, err))
        std::fprintf(stderr, "warning: %s\n", err.c_str());
}

int run(Options o) {
    initCuda(!o.quiet);

    // Run-time platform choice: same kernels, best memory strategy per device.
    if (o.memAuto) o.mem = defaultMemoryMode(platform());

    // ---------------- Input ----------------
    Image rgb;
    std::string inputName;
    if (!o.input.empty()) {
        std::string err;
        if (!imageio::load(o.input, rgb, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        if (rgb.channels == 1) rgb = imageio::grayToRgb(rgb);
        inputName = o.input;
    } else {
        rgb = imageio::generateTestImage(o.genWidth, o.genHeight);
        inputName = "generated test image";
    }
    const size_t n = rgb.pixels();

    if (!o.quiet) {
        std::printf("CPU threads    : %d\n", cpu::numThreads());
        std::printf("Input          : %s (%dx%d, %.1f MPixel)\n", inputName.c_str(), rgb.width,
                    rgb.height, n / 1e6);
        std::printf("Config         : memory=%s%s  blur=%s  radius=%d  iters=%d  warmup=%d\n\n",
                    toString(o.mem), o.memAuto ? " (auto)" : "", toString(o.blur), o.radius,
                    o.iters, o.warmup);
    }

    // ---------------- GPU benchmark ----------------
    PipelineConfig cfg;
    cfg.width = rgb.width;
    cfg.height = rgb.height;
    cfg.memoryMode = o.mem;
    cfg.blurMode = o.blur;
    cfg.blurRadius = o.radius;
    cfg.blurSigma = o.sigma;
    GpuPipeline pipeline(cfg);

    for (int i = 0; i < o.warmup; ++i) pipeline.process(rgb, nullptr);

    Stats gpu[kNumStages], gpuTotal;
    PipelineOutputs gpuOut;
    for (int i = 0; i < o.iters; ++i) {
        const bool last = (i == o.iters - 1);
        const StageTimings t = pipeline.process(rgb, last ? &gpuOut : nullptr);
        for (int s = 0; s < kNumStages; ++s) gpu[s].add(t.ms[s]);
        gpuTotal.add(t.totalMs);
    }

    // ---------------- CPU reference ----------------
    Stats cpu[kNumStages];
    double cpuTotal = 0.0;
    if (o.runCpu) {
        PipelineOutputs cpuOut;
        double ms[kNumStages];
        for (int i = 0; i < o.cpuIters; ++i) {
            cpu::runPipeline(rgb, pipeline.weights(), cpuOut, ms);
            for (int s = 0; s < kNumStages; ++s) cpu[s].add(ms[s]);
        }
        for (int s = 0; s < kNumStages; ++s) cpuTotal += cpu[s].avg();
    }

    // ---------------- Verification ----------------
    VerifyReport rep;
    bool verified = false;
    if (o.verify) {
        rep = verifyStages(rgb, gpuOut, pipeline.weights());
        verified = rep.allPass();
    }

    // ---------------- Report ----------------
    const double fps = gpuTotal.avg() > 0 ? 1000.0 / gpuTotal.avg() : 0.0;
    if (!o.quiet) {
        std::printf("%-15s %11s %11s %11s %9s %13s\n", "Stage", "GPU avg ms", "GPU min ms",
                    "CPU ms", "Speedup", "Eff. BW GB/s");
        std::printf("%s\n", std::string(75, '-').c_str());
        for (int s = 0; s < kNumStages; ++s) {
            const double bytes = stageBytes(s, n, o.mem);
            const bool cpuStage = o.runCpu && s != kUpload && s != kDownload;
            char cpuCol[32] = "-", spCol[32] = "-", bwCol[32] = "-";
            if (cpuStage) std::snprintf(cpuCol, sizeof(cpuCol), "%.3f", cpu[s].avg());
            if (cpuStage && gpu[s].avg() > 0)
                std::snprintf(spCol, sizeof(spCol), "%.1fx", cpu[s].avg() / gpu[s].avg());
            if (bytes > 0 && gpu[s].avg() > 0)
                std::snprintf(bwCol, sizeof(bwCol), "%.1f", bytes / (gpu[s].avg() * 1e6));
            std::printf("%-15s %11.3f %11.3f %11s %9s %13s\n", stageName(s), gpu[s].avg(),
                        gpu[s].minOrZero(), cpuCol, spCol, bwCol);
        }
        std::printf("%s\n", std::string(75, '-').c_str());
        char cpuTot[32] = "-", spTot[32] = "-";
        if (o.runCpu) {
            std::snprintf(cpuTot, sizeof(cpuTot), "%.3f", cpuTotal);
            std::snprintf(spTot, sizeof(spTot), "%.1fx", cpuTotal / gpuTotal.avg());
        }
        std::printf("%-15s %11.3f %11.3f %11s %9s\n\n", "TOTAL", gpuTotal.avg(),
                    gpuTotal.minOrZero(), cpuTot, spTot);

        std::printf("Throughput     : %.1f FPS on GPU", fps);
        if (o.runCpu && cpuTotal > 0) std::printf("  vs  %.1f FPS on CPU", 1000.0 / cpuTotal);
        std::printf("\n\n");

        if (o.verify) {
            std::printf("Verification (each GPU stage vs CPU on the same input):\n");
            for (int i = 0; i < VerifyReport::kCount; ++i) {
                const CompareResult& r = rep.results[i];
                std::printf("  %-14s %s  max diff %d (tolerance %d), differing pixels %zu / %zu\n",
                            rep.names[i], r.pass ? "PASS" : "FAIL", r.maxDiff, r.tolerance,
                            r.mismatches, r.total);
            }
            std::printf("\n");
        }
    }

    // ---------------- Save outputs ----------------
    if (o.save) {
        const fs::path dir(o.outDir);
        fs::create_directories(dir);
        saveImage(dir / "0_input.ppm", rgb);
        saveImage(dir / "1_gray.pgm", gpuOut.gray);
        saveImage(dir / "2_equalized.pgm", gpuOut.equalized);
        saveImage(dir / "3_blurred.pgm", gpuOut.blurred);
        saveImage(dir / "4_edges.pgm", gpuOut.edges);
        if (!o.quiet) std::printf("Images saved to: %s/\n", dir.string().c_str());
    }
    if (!o.csv.empty()) {
        const fs::path csvParent = fs::path(o.csv).parent_path();
        if (!csvParent.empty()) fs::create_directories(csvParent);
        appendCsv(o, rgb, gpu, gpuTotal, cpu, cpuTotal, verified);
        if (!o.quiet) std::printf("CSV appended  : %s\n", o.csv.c_str());
    }

    // One-line summary (always printed; handy in scripts and logs)
    char cpuPart[64] = "";
    if (o.runCpu) std::snprintf(cpuPart, sizeof(cpuPart), " | CPU %.3f ms", cpuTotal);
    std::printf("[%s] %dx%d mem=%s blur=%s r=%d | GPU %.3f ms (%.1f FPS)%s | verify: %s\n",
                timestamp().c_str(), rgb.width, rgb.height, toString(o.mem), toString(o.blur),
                o.radius, gpuTotal.avg(), fps, cpuPart,
                o.verify ? (verified ? "PASS" : "FAIL") : "skipped");

    return (o.verify && !verified) ? 3 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt;
        const int parsed = parseArgs(argc, argv, opt);
        if (parsed == 2) return 0;
        return run(opt);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
