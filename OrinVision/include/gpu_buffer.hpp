// =====================================================================
// gpu_buffer.hpp - One buffer class, three Jetson memory strategies
// =====================================================================
// On Jetson, the CPU and GPU share the same physical RAM. This class lets
// you switch HOW a buffer is allocated with one setting, so you can
// measure which strategy is fastest on your board:
//
//   Device   : cudaMalloc (GPU) + cudaMallocHost (CPU) + explicit copies.
//              Same code you would write for a laptop/desktop GPU.
//   Managed  : cudaMallocManaged. One pointer for CPU and GPU, no copies.
//              Jetson rule: CPU must NOT touch it while a kernel runs.
//   ZeroCopy : cudaHostAlloc(..., Mapped). Pinned CPU memory the GPU reads
//              directly. No copies; good for data used once (camera frames).
//
// The class is RAII: memory is freed automatically in the destructor.
// =====================================================================
#pragma once

#include "cuda_utils.hpp"

#include <cstddef>
#include <string>

enum class MemoryMode { Device, Managed, ZeroCopy };

inline const char* toString(MemoryMode m) {
    switch (m) {
        case MemoryMode::Device:   return "device";
        case MemoryMode::Managed:  return "managed";
        case MemoryMode::ZeroCopy: return "zerocopy";
    }
    return "?";
}

inline bool parseMemoryMode(const std::string& s, MemoryMode& out) {
    if (s == "device")   { out = MemoryMode::Device;   return true; }
    if (s == "managed")  { out = MemoryMode::Managed;  return true; }
    if (s == "zerocopy") { out = MemoryMode::ZeroCopy; return true; }
    return false;
}

template <typename T>
class GpuBuffer {
public:
    GpuBuffer() = default;
    GpuBuffer(size_t count, MemoryMode mode) { allocate(count, mode); }
    ~GpuBuffer() { release(); }

    // Not copyable (would double-free), but movable.
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& other) noexcept { moveFrom(other); }
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) { release(); moveFrom(other); }
        return *this;
    }

    void allocate(size_t count, MemoryMode mode) {
        release();
        count_ = count;
        mode_ = mode;
        const size_t nbytes = bytes();
        if (nbytes == 0) return;

        switch (mode) {
            case MemoryMode::Device:
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_), nbytes));
                CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_), nbytes));
                break;
            case MemoryMode::Managed:
                CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&dev_), nbytes));
                host_ = dev_;                       // same pointer for CPU and GPU
                break;
            case MemoryMode::ZeroCopy:
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_), nbytes,
                                         cudaHostAllocMapped));
                CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dev_),
                                                    host_, 0));
                break;
        }
    }

    void release() {
        if (count_ == 0 && !host_ && !dev_) return;
        switch (mode_) {
            case MemoryMode::Device:
                if (dev_) cudaFree(dev_);
                if (host_) cudaFreeHost(host_);
                break;
            case MemoryMode::Managed:
                if (dev_) cudaFree(dev_);
                break;
            case MemoryMode::ZeroCopy:
                if (host_) cudaFreeHost(host_);
                break;
        }
        host_ = dev_ = nullptr;
        count_ = 0;
    }

    // Copy CPU -> GPU. Only Device mode needs it; otherwise it does nothing,
    // because CPU and GPU already see the same memory.
    void toDevice(cudaStream_t stream = 0) {
        if (mode_ == MemoryMode::Device && count_)
            CUDA_CHECK(cudaMemcpyAsync(dev_, host_, bytes(), cudaMemcpyHostToDevice, stream));
    }

    // Copy GPU -> CPU. Only Device mode needs it.
    void toHost(cudaStream_t stream = 0) {
        if (mode_ == MemoryMode::Device && count_)
            CUDA_CHECK(cudaMemcpyAsync(host_, dev_, bytes(), cudaMemcpyDeviceToHost, stream));
    }

    T* host() { return host_; }               // use this pointer in CPU code
    const T* host() const { return host_; }
    T* device() { return dev_; }              // pass this pointer to kernels
    const T* device() const { return dev_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    MemoryMode mode() const { return mode_; }

private:
    void moveFrom(GpuBuffer& o) {
        host_ = o.host_; dev_ = o.dev_; count_ = o.count_; mode_ = o.mode_;
        o.host_ = o.dev_ = nullptr; o.count_ = 0;
    }

    T* host_ = nullptr;
    T* dev_ = nullptr;
    size_t count_ = 0;
    MemoryMode mode_ = MemoryMode::Managed;
};
