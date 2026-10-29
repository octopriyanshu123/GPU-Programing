// =====================================================================
// dot_product.cu  -  Dot product of two vectors (100 elements) on GPU
// =====================================================================
//
// WHAT IS A DOT PRODUCT?
//   a = [a0, a1, a2, ...]
//   b = [b0, b1, b2, ...]
//   a . b = a0*b0 + a1*b1 + a2*b2 + ...   (one single number)
//
// HOW WE DO IT ON THE GPU (2 steps):
//   Step 1: MULTIPLY  - each thread multiplies ONE pair: a[i] * b[i]
//   Step 2: ADD UP    - threads work together to add all products
//                       into one number (this is called a "reduction")
//
// Compile:  nvcc -arch=sm_87 dot_product.cu -o dot_product
// Run:      ./dot_product
// =====================================================================

#include <cstdio>

#define N 100                  // number of elements in each vector
#define THREADS_PER_BLOCK 128  // must be a power of 2 (64, 128, 256...)
                               // and >= N here, so 1 block is enough

// ---------------------------------------------------------------------
// GPU KERNEL: runs on the GPU, by many threads at the same time
// ---------------------------------------------------------------------
__global__ void dotProduct(const float *a, const float *b, float *result, int n)
{
    // "Shared memory" = fast memory that all threads in ONE block can see.
    // Each thread will put its product a[i]*b[i] in here.
    __shared__ float temp[THREADS_PER_BLOCK];

    // Global index: which element this thread is responsible for.
    //   blockIdx.x  = which block this thread is in
    //   blockDim.x  = how many threads per block
    //   threadIdx.x = this thread's number inside its block
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;   // local index inside the block (0..127)

    // ---------- STEP 1: MULTIPLY ----------
    // We have 128 threads but only 100 elements.
    // Threads 100..127 have no data, so they store 0 (adds nothing).
    if (i < n)
        temp[tid] = a[i] * b[i];
    else
        temp[tid] = 0.0f;

    // Wait until ALL threads in the block have written their product.
    // Without this, some threads might start adding before others finish.
    __syncthreads();

    // ---------- STEP 2: ADD UP (tree reduction) ----------
    // Instead of one thread adding 128 numbers one by one (slow),
    // threads add in pairs, halving the count each round:
    //
    //   Round 1: 128 numbers -> 64   (thread 0 adds temp[0]+temp[64], ...)
    //   Round 2:  64 numbers -> 32
    //   ...
    //   Round 7:   2 numbers -> 1    (final answer ends up in temp[0])
    //
    // Only 7 rounds instead of 127 additions in a row.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
            temp[tid] += temp[tid + stride];

        // Wait for all threads to finish this round before the next one.
        __syncthreads();
    }

    // ---------- SAVE THE RESULT ----------
    // Thread 0 now holds this block's total in temp[0].
    // atomicAdd safely adds it to the final result. (With only 1 block
    // a plain write would work, but atomicAdd also works if you later
    // make N bigger and use many blocks.)
    if (tid == 0)
        atomicAdd(result, temp[0]);
}

// ---------------------------------------------------------------------
// CPU (host) code: prepares data, launches the kernel, checks the answer
// ---------------------------------------------------------------------
int main()
{
    float *a, *b, *result;

    // Allocate memory that BOTH CPU and GPU can use.
    // On Jetson, CPU and GPU share the same RAM, so no cudaMemcpy needed.
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&result, sizeof(float));

    // Fill the vectors with easy-to-check values:
    //   a = [1, 2, 3, ..., 100]
    //   b = [2, 2, 2, ..., 2]
    // So a . b = 2 * (1 + 2 + ... + 100) = 2 * 5050 = 10100
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = 2.0f;
    }
    *result = 0.0f;   // start the sum at zero

    // Launch configuration: how many blocks do we need?
    // (N + THREADS - 1) / THREADS rounds up -> here (100+127)/128 = 1 block
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Launching %d block(s) x %d threads for %d elements\n",
           blocks, THREADS_PER_BLOCK, N);

    // Run the kernel on the GPU
    dotProduct<<<blocks, THREADS_PER_BLOCK>>>(a, b, result, N);

    // GPU runs in the background - wait for it to finish before reading.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Compute the same thing on the CPU to verify the GPU answer
    float cpu = 0.0f;
    for (int i = 0; i < N; i++)
        cpu += a[i] * b[i];

    printf("GPU result: %.1f\n", *result);
    printf("CPU result: %.1f\n", cpu);
    printf("%s\n", (*result == cpu) ? "MATCH - correct!" : "MISMATCH!");

    // Free the memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(result);
    return 0;
}