// =====================================================================
// cuda_utils.hpp - Error checking and small helpers used everywhere
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Wrap EVERY CUDA API call with this. CUDA errors are silent otherwise.
// Example: CUDA_CHECK(cudaMalloc(&ptr, bytes));
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "[CUDA ERROR] %s\n  at %s:%d\n  call: %s\n", \
                         cudaGetErrorString(err_), __FILE__, __LINE__, #call); \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// Call right after a kernel launch: catches bad launch configurations.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Round-up integer division: how many blocks of size b cover a elements.
inline int divUp(int a, int b) { return (a + b - 1) / b; }

// Number of Streaming Multiprocessors (cached after the first call).
// Used to size grid-stride kernels so every SM gets work.
inline int smCount() {
    static int count = 0;
    if (count == 0) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev));
    }
    return count;
}
