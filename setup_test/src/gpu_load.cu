// gpu_load.cu - keeps the GPU busy for 10 seconds so you can watch it in jtop
// Compile: nvcc -O3 -arch=sm_87 gpu_load.cu -o gpu_load
// Run:     ./gpu_load

#include <cstdio>
#include <chrono>

// Heavy math kernel: each thread does many sin/multiply operations
__global__ void burn(float *data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        for (int k = 0; k < iters; k++) {
            x = x * 1.000001f + 0.000001f;
            x = sinf(x) * 0.5f + 0.5f;
        }
        data[i] = x;
    }
}

int main() {
    const int n = 1 << 22;          // ~4 million elements (16 MB)
    const int iters = 1000;         // work per thread per launch
    const double run_seconds = 10.0;

    float *data;
    cudaMallocManaged(&data, n * sizeof(float));   // shared CPU/GPU memory
    for (int i = 0; i < n; i++) data[i] = 0.5f;

    printf("Running GPU load for %.0f seconds... open jtop in another terminal\n", run_seconds);

    auto start = std::chrono::steady_clock::now();
    int launches = 0, last_sec = -1;

    while (true) {

        
        burn<<<(n + 255) / 256, 256>>>(data, n, iters);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        launches++;

        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        if ((int)elapsed != last_sec) {          // print once per second
            last_sec = (int)elapsed;
            printf("  %2d s  |  kernel launches: %d\n", last_sec, launches);
        }
        if (elapsed >= run_seconds) break;
    }

    printf("Done. Total launches: %d, data[0] = %f\n", launches, data[0]);
    cudaFree(data);
    return 0;
}