// =====================================================================
// platform.cpp - Jetson / PC detection and CUDA start-up checks
// =====================================================================
#include "platform.hpp"

#include "cuda_utils.hpp"
#include "gpu_filters.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

namespace {

// Read a whole (small) text file; empty string if it does not exist.
std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // /proc/device-tree strings end with '\0'
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n')) s.pop_back();
    return s;
}

PlatformInfo detect() {
    PlatformInfo p;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    p.gpuName = prop.name;
    p.ccMajor = prop.major;
    p.ccMinor = prop.minor;
    p.smCount = prop.multiProcessorCount;
    p.totalMem = prop.totalGlobalMem;
    p.integrated = prop.integrated != 0;
    p.concurrentManagedAccess = prop.concurrentManagedAccess != 0;

    // Jetson boards always have these files; a laptop never does.
    const std::string model = readFile("/proc/device-tree/model");
    const bool tegraRelease = !readFile("/etc/nv_tegra_release").empty();
    const bool jetsonModel = model.find("Jetson") != std::string::npos ||
                             model.find("Orin") != std::string::npos;

    // The CUDA "integrated" flag is the main signal: it is what actually
    // changes how memory behaves. The files are a second opinion.
    if (p.integrated || tegraRelease || jetsonModel) {
        p.kind = PlatformKind::Jetson;
        p.boardModel = model;
    }
    return p;
}

}  // namespace

const PlatformInfo& platform() {
    static const PlatformInfo info = detect();
    return info;
}

MemoryMode defaultMemoryMode(const PlatformInfo& p) {
    // Jetson: CPU and GPU share RAM, so managed memory avoids useless copies.
    // PC: the GPU has its own VRAM; explicit copies over PCIe are the fastest
    // and most predictable (zero-copy would read over PCIe on every access).
    return p.isJetson() ? MemoryMode::Managed : MemoryMode::Device;
}

void initCuda(bool printInfo) {
    // Allow zero-copy (mapped) host memory. Must be set before the CUDA
    // context is created; on Jetson this is effectively always enabled.
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess) CUDA_CHECK(err);
    cudaGetLastError();   // clear a possible "already active" error

    CUDA_CHECK(cudaSetDevice(0));
    const PlatformInfo& p = platform();

    // Was this binary compiled for this GPU? (sm_87 code cannot run on an
    // sm_86 RTX 30xx and vice versa.) Check now with a clear message instead
    // of failing later with "no kernel image is available".
    std::string kernelErr;
    if (!gpu::kernelsAvailable(kernelErr)) {
        std::fprintf(stderr,
                     "[CUDA ERROR] the GPU kernels cannot run on %s (compute %d.%d): %s\n"
                     "  Rebuild for this GPU, e.g.:\n"
                     "    rm -rf build && cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=%d%d\n",
                     p.gpuName.c_str(), p.ccMajor, p.ccMinor, kernelErr.c_str(),
                     p.ccMajor, p.ccMinor);
        std::exit(EXIT_FAILURE);
    }

    if (!printInfo) return;
    int runtimeVer = 0, driverVer = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVer));
    CUDA_CHECK(cudaDriverGetVersion(&driverVer));
    if (p.isJetson())
        std::printf("Platform       : Jetson%s%s\n", p.boardModel.empty() ? "" : " - ",
                    p.boardModel.c_str());
    else
        std::printf("Platform       : PC / laptop (discrete GPU)\n");
    std::printf("GPU            : %s (compute %d.%d, %d SMs)\n", p.gpuName.c_str(), p.ccMajor,
                p.ccMinor, p.smCount);
    std::printf("Memory         : %.2f GB %s\n", p.totalMem / 1e9,
                p.integrated ? "(shared with CPU - integrated GPU)" : "(dedicated VRAM)");
    std::printf("CUDA runtime   : %d.%d   driver: %d.%d\n", runtimeVer / 1000,
                (runtimeVer % 1000) / 10, driverVer / 1000, (driverVer % 1000) / 10);
    std::printf("Managed memory : concurrent CPU access %s\n",
                p.concurrentManagedAccess ? "YES" : "NO (sync before CPU access)");
}
