// test.cu
#include <cstdio>

__global__ void add(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + y[i];
}

int main() {
    int n = 1 << 20;
    float *x, *y;
    cudaMallocManaged(&x, n * sizeof(float));   // shared CPU/GPU memory
    cudaMallocManaged(&y, n * sizeof(float));

    for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }

    add<<<(n + 255) / 256, 256>>>(n, x, y);

    
    cudaDeviceSynchronize();

    printf("y[0] = %f (expected 3.0)\n", y[0]);
    cudaFree(x); cudaFree(y);
}