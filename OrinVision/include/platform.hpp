// =====================================================================
// platform.hpp - Detect at RUN TIME whether we run on a Jetson or a PC
// =====================================================================
// The same binary logic works on both, but the best choices differ:
//
//                         Jetson Orin (integrated)   Laptop / PC (discrete)
//   GPU memory            shared with the CPU        own VRAM behind PCIe
//   cudaMemcpy            copies RAM -> same RAM     really crosses PCIe
//   Best default memory   managed (no copies)        device (explicit copies)
//   Managed memory        CPU must wait for GPU      CPU may access any time
//                         (no concurrent access)     (concurrent access)
//
// We ask the CUDA driver (cudaDeviceProp::integrated) and also look for
// the Jetson-only files /etc/nv_tegra_release and /proc/device-tree/model.
// =====================================================================
#pragma once

#include "gpu_buffer.hpp"

#include <cstddef>
#include <string>

enum class PlatformKind { Jetson, Discrete };

inline const char* toString(PlatformKind k) {
    return k == PlatformKind::Jetson ? "jetson" : "discrete";
}

struct PlatformInfo {
    PlatformKind kind = PlatformKind::Discrete;
    std::string gpuName;                  // e.g. "Orin" or "NVIDIA GeForce RTX 3080"
    std::string boardModel;               // Jetson board name, empty on a PC
    int ccMajor = 0, ccMinor = 0;         // compute capability (8.7 = Orin, 8.6 = RTX 30xx)
    int smCount = 0;
    size_t totalMem = 0;                  // bytes
    bool integrated = false;              // GPU shares RAM with the CPU
    bool concurrentManagedAccess = false; // CPU may touch managed memory while kernels run

    bool isJetson() const { return kind == PlatformKind::Jetson; }
};

// Detected once (on the first call), then cached. Needs a CUDA device.
const PlatformInfo& platform();

// Fastest memory mode for this platform (used when --mem is "auto").
MemoryMode defaultMemoryMode(const PlatformInfo& p);

// Print GPU information and prepare CUDA (call once at program start).
// Exits with a helpful message if the kernels were not compiled for this GPU.
void initCuda(bool printInfo);
