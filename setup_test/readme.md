# GPU Programming on NVIDIA Jetson (Orin)

A practical guide to the important things you need to know about CUDA programming on a Jetson, written for someone who has already run their first CUDA programs.

---

## Table of Contents

1. [Jetson vs Laptop GPU](#1-jetson-vs-laptop-gpu)
2. [Know Your Hardware](#2-know-your-hardware)
3. [The CUDA Programming Model](#3-the-cuda-programming-model)
4. [GPU Memory Hierarchy](#4-gpu-memory-hierarchy)
5. [Memory on Jetson (Most Important Section)](#5-memory-on-jetson-most-important-section)
6. [Always Check for Errors](#6-always-check-for-errors)
7. [Timing Your Code Correctly](#7-timing-your-code-correctly)
8. [Writing Fast Kernels](#8-writing-fast-kernels)
9. [Streams: Doing Things at the Same Time](#9-streams-doing-things-at-the-same-time)
10. [FP16, Tensor Cores and INT8](#10-fp16-tensor-cores-and-int8)
11. [Libraries: Don't Write Everything Yourself](#11-libraries-dont-write-everything-yourself)
12. [Python on Jetson](#12-python-on-jetson)
13. [Power, Clocks and Heat](#13-power-clocks-and-heat)
14. [Tools for Monitoring, Profiling and Debugging](#14-tools-for-monitoring-profiling-and-debugging)
15. [Common Mistakes](#15-common-mistakes)
16. [Learning Roadmap](#16-learning-roadmap)
17. [Command Cheat Sheet](#17-command-cheat-sheet)

---

## 1. Jetson vs Laptop GPU

| Topic | Laptop (discrete GPU) | Jetson (integrated GPU) |
|---|---|---|
| CPU | x86 (Intel / AMD) | ARM64 (aarch64) |
| GPU memory | Separate VRAM | **Shares system RAM with CPU** |
| CPU ↔ GPU link | PCIe (slow copies) | Same chip, same RAM |
| `cudaMemcpy` | Needed | Often unnecessary |
| Software install | CUDA Toolkit from NVIDIA | Comes with **JetPack** |
| `nvidia-smi` | Full info | Mostly `N/A`, use `jtop` |
| Power | High (50–150 W) | Low (7–60 W), selectable modes |
| Extra hardware | — | DLA, video encoders, VPI accelerators |

**Your CUDA code is the same.** What changes is memory handling, performance tuning and setup.

---

## 2. Know Your Hardware

Jetson Orin uses the **Ampere** GPU architecture with compute capability **8.7**, so always compile with:

```bash
nvcc -arch=sm_87 program.cu -o program
```

Approximate specs (check your own board with the program below):

| Board | CUDA cores | Tensor cores | RAM | Memory bandwidth |
|---|---|---|---|---|
| Orin Nano 8GB | 1024 | 32 | 8 GB | ~68 GB/s (higher in "Super" mode) |
| Orin NX 16GB | 1024 | 32 | 16 GB | ~102 GB/s |
| AGX Orin 64GB | 2048 | 64 | 64 GB | ~205 GB/s |

For comparison, a laptop RTX GPU often has 250–500 GB/s. **Memory bandwidth is usually the main limit on Jetson**, so programs that read/write a lot of data are slower than on a desktop GPU.

### Mini "deviceQuery" program

Save as `info.cu`, compile with `nvcc -arch=sm_87 info.cu -o info`:

```cpp
#include <cstdio>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("GPU name                 : %s\n", p.name);
    printf("Compute capability       : %d.%d\n", p.major, p.minor);
    printf("Streaming Multiprocessors: %d\n", p.multiProcessorCount);
    printf("Max threads per block    : %d\n", p.maxThreadsPerBlock);
    printf("Shared memory per block  : %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("Total memory (shared RAM): %.2f GB\n", p.totalGlobalMem / 1e9);
    printf("Integrated GPU           : %s\n", p.integrated ? "YES" : "NO");
    printf("Can map host memory      : %s\n", p.canMapHostMemory ? "YES" : "NO");
    printf("Concurrent managed access: %s\n", p.concurrentManagedAccess ? "YES" : "NO");
    return 0;
}
```

On Jetson you will see `Integrated GPU: YES`. Pay attention to `Concurrent managed access` — see [Section 5](#5-memory-on-jetson-most-important-section).

---

## 3. The CUDA Programming Model

```
Grid  (the whole kernel launch)
 ├── Block 0  ── Thread 0, Thread 1, ... Thread 255
 ├── Block 1  ── Thread 0, Thread 1, ... Thread 255
 └── ...
```

| Term | Meaning |
|---|---|
| **Thread** | One worker. Runs your kernel code once. |
| **Block** | A group of threads (max 1024). Threads in a block can share fast memory and synchronize with `__syncthreads()`. |
| **Grid** | All the blocks of one kernel launch. |
| **Warp** | 32 threads that execute the **same instruction together**. This is the real unit of execution in hardware. |
| **SM** (Streaming Multiprocessor) | A GPU "core cluster". Blocks are scheduled onto SMs. Orin Nano has 8 SMs. |

### Global index formula (memorize this)

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) { /* work on element i */ }
```

### Launch size formula

```cpp
int threads = 256;                          // good default (multiple of 32)
int blocks  = (n + threads - 1) / threads;  // round up
kernel<<<blocks, threads>>>(...);
```

### Grid-stride loop (handles any size of data)

```cpp
__global__ void scale(float *x, int n, float s) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
        x[i] *= s;
}
```

### Function keywords

| Keyword | Runs on | Called from |
|---|---|---|
| `__global__` | GPU | CPU (this is a kernel) |
| `__device__` | GPU | GPU |
| `__host__` | CPU | CPU (default) |

---

## 4. GPU Memory Hierarchy

From fastest to slowest:

| Memory | Where | Speed | Who can see it | Keyword |
|---|---|---|---|---|
| Registers | Inside each SM | Fastest | One thread | normal local variables |
| Shared memory | Inside each SM | Very fast | Threads in one block | `__shared__` |
| L1 / L2 cache | On chip | Fast | Automatic | — |
| Constant memory | DRAM, cached | Fast if all threads read same value | All threads (read only) | `__constant__` |
| Global memory | **System RAM** (on Jetson) | Slowest | Everyone | `cudaMalloc`, `cudaMallocManaged` |

**Golden rule:** read data from global memory **once**, keep working on it in registers/shared memory, write the result **once**.

---

## 5. Memory on Jetson (Most Important Section)

CPU and GPU share the **same physical RAM**. There are four ways to allocate memory:

| Method | Function | Copy needed? | Notes |
|---|---|---|---|
| Device memory | `cudaMalloc` | Yes (`cudaMemcpy`) | Same as laptop. Portable, but copies waste time on Jetson. |
| Pinned host memory | `cudaMallocHost` | Yes, but faster | Used for fast async copies. |
| **Zero-copy (mapped)** | `cudaHostAlloc(..., cudaHostAllocMapped)` | **No** | CPU and GPU access the same buffer. Good for data written by CPU, read once by GPU (e.g. camera frames). |
| **Unified / managed** | `cudaMallocManaged` | **No** | Easiest. Driver handles everything. |

### Zero-copy example

```cpp
float *h_data, *d_data;

cudaHostAlloc(&h_data, n * sizeof(float), cudaHostAllocMapped); // CPU pointer
cudaHostGetDevicePointer(&d_data, h_data, 0);                   // GPU pointer

// CPU fills h_data ...
kernel<<<blocks, threads>>>(d_data, n);   // GPU uses d_data (same memory!)
cudaDeviceSynchronize();
// CPU reads results from h_data

cudaFreeHost(h_data);
```

### ⚠️ Important gotcha: managed memory on Jetson

On Jetson, `concurrentManagedAccess` is **0**. This means:

> **The CPU must NOT touch managed memory while a GPU kernel is running.**

```cpp
kernel<<<b, t>>>(data);
data[0] = 5;               // ❌ WRONG - can crash (segmentation fault)
cudaDeviceSynchronize();
data[0] = 5;               // ✅ OK - kernel finished
```

Always call `cudaDeviceSynchronize()` (or sync the stream) **before** the CPU reads or writes managed memory.

### Which one should I use?

- **Learning / simple programs:** `cudaMallocManaged`
- **Camera / sensor data processed once:** zero-copy
- **Data used many times only by the GPU:** `cudaMalloc` (no copies needed if GPU creates the data itself)
- **Not sure?** Try 2 options and **measure** (Section 7).

### Remember the RAM limit

On an 8 GB board, the OS takes 1–2 GB, so roughly 5–6 GB is free for **CPU + GPU together**. If you run out:

- The program crashes or the system freezes (Linux OOM killer)
- Check usage with `jtop` or `free -h`
- Add swap for compiling big projects (not for GPU data — swap is very slow)

---

## 6. Always Check for Errors

CUDA errors are **silent** by default. A kernel can fail and your program continues with garbage results. Use this macro in every program:

```cpp
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                    cudaGetErrorString(err), __FILE__, __LINE__);     \
            exit(1);                                                  \
        }                                                             \
    } while (0)

// Usage:
CUDA_CHECK(cudaMallocManaged(&x, bytes));

kernel<<<blocks, threads>>>(x, n);
CUDA_CHECK(cudaGetLastError());        // catches launch errors (bad config)
CUDA_CHECK(cudaDeviceSynchronize());   // catches errors during execution
```

---

## 7. Timing Your Code Correctly

Kernel launches are **asynchronous** — the CPU continues immediately. Timing with a normal CPU clock without syncing gives wrong (too small) numbers. Use **CUDA events**:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

kernel<<<blocks, threads>>>(...);   // warm-up run (first launch is slower)

cudaEventRecord(start);
kernel<<<blocks, threads>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
```

Tips for fair benchmarks on Jetson:

- Do a **warm-up** run first
- Run `sudo jetson_clocks` so clocks don't change during the test
- Run several times and take the average
- Always write down the **power mode** you used

### Measuring bandwidth

```
Bandwidth (GB/s) = (bytes read + bytes written) / (time in seconds) / 1e9
```

Compare your result with the board's maximum (Section 2). If you reach 70–80%, your memory-bound kernel is already very good.

---

## 8. Writing Fast Kernels

### 8.1 Memory coalescing (biggest win)

Threads in a warp should access **neighboring** memory addresses.

```cpp
// ✅ GOOD: thread i reads x[i]  -> one combined memory transaction
y[i] = x[i];

// ❌ BAD: thread i reads x[i * 32] -> 32 separate transactions
y[i] = x[i * 32];
```

For 2D data stored row by row, let `threadIdx.x` move along the **row** (column index).

### 8.2 Avoid warp divergence

All 32 threads of a warp run the same instruction. If they take different `if` branches, both branches run one after another.

```cpp
// ❌ Half the warp waits each time
if (threadIdx.x % 2 == 0) doA(); else doB();

// ✅ Whole warps take the same path
if ((threadIdx.x / 32) % 2 == 0) doA(); else doB();
```

### 8.3 Use shared memory for data reused many times

Classic example: **tiled matrix multiplication**. Load a tile of A and B into `__shared__` arrays once, then every thread in the block reuses them many times.

### 8.4 Avoid shared memory bank conflicts

Shared memory has 32 banks. If many threads of a warp access **different addresses in the same bank**, accesses become serial. A common fix is padding:

```cpp
__shared__ float tile[32][32 + 1];   // +1 avoids conflicts when reading columns
```

### 8.5 Occupancy

**Occupancy** = active warps on an SM / maximum possible warps. Too few threads = GPU waits on memory with nothing else to do.

- Use block sizes of **128 or 256** as a start
- Too many registers or too much shared memory per thread lowers occupancy
- Higher occupancy is not always faster — **measure**

### 8.6 Compute-bound vs memory-bound

| Type | Meaning | How to speed up |
|---|---|---|
| **Memory-bound** | Waiting for data (most common on Jetson) | Read less data, coalesce, use shared memory, use FP16 (half the bytes) |
| **Compute-bound** | Math units are full | Use faster math (`__sinf`, `-use_fast_math`), FP16, Tensor Cores |

Nsight Compute tells you which one your kernel is (Section 14).

### 8.7 Reduce kernel launches

Each launch has overhead (a few microseconds). Many tiny kernels → slow. Combine small kernels into one ("kernel fusion") when possible, or use **CUDA Graphs** for repeated sequences.

### 8.8 Fast reductions with warp shuffle

Instead of shared memory for the last steps of a reduction, threads in a warp can exchange values directly:

```cpp
__inline__ __device__ float warpSum(float v) {
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // thread 0 of the warp has the total
}
```

---

## 9. Streams: Doing Things at the Same Time

A **stream** is a queue of GPU work. Work in the same stream runs in order; work in different streams can overlap.

```cpp
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

kernelA<<<blocks, threads, 0, s1>>>(...);   // these two can
kernelB<<<blocks, threads, 0, s2>>>(...);   // run at the same time

cudaStreamSynchronize(s1);
cudaStreamSynchronize(s2);

cudaStreamDestroy(s1);
cudaStreamDestroy(s2);
```

Useful for pipelines like: *camera frame N is captured* while *frame N-1 is processed on the GPU* while *frame N-2 is displayed*.

---

## 10. FP16, Tensor Cores and INT8

| Type | Size | Use |
|---|---|---|
| FP32 (`float`) | 4 bytes | Default, accurate |
| FP16 (`__half`) | 2 bytes | 2× less memory traffic, Tensor Cores, good for AI |
| INT8 | 1 byte | Fastest AI inference (needs calibration in TensorRT) |

```cpp
#include <cuda_fp16.h>

__global__ void addHalf(const __half *a, const __half *b, __half *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = __hadd(a[i], b[i]);
}
```

**Tensor Cores** do very fast small matrix multiplications. You usually use them through **cuBLAS**, **cuDNN** or **TensorRT** instead of writing them yourself (advanced: the `wmma` API).

Since Jetson is memory-bandwidth limited, switching from FP32 to FP16 often gives a big speedup just from moving half the bytes.

---

## 11. Libraries: Don't Write Everything Yourself

All of these come with JetPack or are available for Jetson:

| Library | What it does |
|---|---|
| **Thrust** | C++ STL-like GPU algorithms: `sort`, `reduce`, `transform` |
| **CUB** | Fast building blocks (block/warp reduce, scan) for your own kernels |
| **cuBLAS** | Matrix and vector math (GEMM, dot product) — uses Tensor Cores |
| **cuFFT** | Fast Fourier Transform |
| **cuRAND** | Random numbers on GPU |
| **NPP** | Image processing primitives (resize, filters, color convert) |
| **cuDNN** | Deep learning layers (convolution etc.) |
| **TensorRT** | Optimizes and runs trained AI models (FP16/INT8, GPU + DLA) |
| **VPI** | Computer vision using GPU, PVA and other Jetson accelerators |
| **OpenCV (CUDA build)** | Computer vision; JetPack's default OpenCV may be CPU-only |
| **DeepStream** | Complete video analytics pipelines |
| **Jetson Multimedia API / GStreamer** | Hardware video encode/decode, camera capture |

Example: dot product with Thrust in one line:

```cpp
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

thrust::device_vector<float> a(100, 1.0f), b(100, 2.0f);
float result = thrust::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
```

---

## 12. Python on Jetson

| Tool | Use | Install note |
|---|---|---|
| **PyTorch** | AI training/inference | Use NVIDIA's **Jetson builds** — normal `pip install torch` usually has **no GPU support** on Jetson |
| **CuPy** | NumPy on the GPU | Needs aarch64 + CUDA 12 build |
| **Numba** | Write CUDA kernels in Python (`@cuda.jit`) | Good for learning |
| **TensorRT Python** | Fast inference | Comes with JetPack |

Check PyTorch sees the GPU:

```python
import torch
print(torch.cuda.is_available())      # must be True
print(torch.cuda.get_device_name(0))  # should say Orin
```

**Tip:** NVIDIA's **Jetson containers** (`dusty-nv/jetson-containers` on GitHub) provide ready-made Docker images with PyTorch, TensorRT etc. already working. This avoids many install problems.

---

## 13. Power, Clocks and Heat

Jetson performance depends heavily on the power mode.

```bash
sudo nvpmodel -q          # show current mode
sudo nvpmodel -m 0        # switch to mode 0 (usually max performance)
sudo jetson_clocks        # lock clocks at max for current mode
sudo jetson_clocks --show # show current clock settings
```

Things to know:

- Mode numbers and wattages are **different on each board** — check them in `jtop` (CTRL tab) or `/etc/nvpmodel.conf`
- Changing some modes requires a **reboot**
- If the board gets too hot, it **throttles** (lowers clocks) → your program suddenly gets slower. Watch temperatures in `jtop`
- Use a fan / heatsink for long GPU workloads
- `jetson_clocks` uses more power — use it for benchmarking and demanding apps

---

## 14. Tools for Monitoring, Profiling and Debugging

| Tool | Command | Use |
|---|---|---|
| **jtop** | `jtop` | Live GPU/CPU/RAM/power/temperature dashboard |
| **tegrastats** | `sudo tegrastats` | Text version of jtop, good for logging |
| **Nsight Systems** | `nsys profile ./app` | Timeline of **whole program**: CPU, GPU kernels, copies, gaps |
| **Nsight Compute** | `sudo $(which ncu) -c 1 ./app` | Deep analysis of **one kernel** |
| **compute-sanitizer** | `compute-sanitizer ./app` | Finds memory bugs (out-of-bounds, race conditions) |
| **cuda-gdb** | `cuda-gdb ./app` | Step through GPU code like gdb |

### Profiling workflow

1. **`nsys`** first: where is time spent? Which kernel is slowest? Are there gaps?
2. **`ncu`** on the slowest kernel: is it memory-bound or compute-bound? Is occupancy low? Are accesses coalesced?
3. Fix → measure again → repeat.

### Debugging workflow

```bash
nvcc -G -g -arch=sm_87 app.cu -o app   # -G = device debug info (makes code slow!)
compute-sanitizer ./app                 # memory errors
compute-sanitizer --tool racecheck ./app  # shared memory race conditions
```

Remove `-G` for normal / fast builds.

---

## 15. Common Mistakes

| Mistake | Fix |
|---|---|
| Forgot `cudaDeviceSynchronize()` before reading results | Always sync before CPU reads GPU output |
| CPU touches managed memory while kernel runs | Sync first (Jetson has no concurrent managed access) |
| No error checking → silent wrong results | Use `CUDA_CHECK` macro |
| Missing `if (i < n)` bounds check | Threads past the end write garbage/crash |
| Compiled without `-arch=sm_87` | Always set the architecture |
| Forgot `__syncthreads()` in shared memory code | Add it after writing shared memory, before reading |
| `__syncthreads()` inside an `if` that not all threads reach | Program hangs — keep it outside divergent branches |
| Block size not multiple of 32 | Use 128, 256, 512 |
| Timing without sync or warm-up | Use CUDA events + warm-up |
| Using `cudaMemcpy` everywhere like on a laptop | Use managed or zero-copy memory |
| `pip install torch` gives CPU-only PyTorch | Use NVIDIA Jetson wheels/containers |
| Running out of the shared 8 GB RAM | Monitor with `jtop`, free memory, use smaller batches |
| Benchmark results change between runs | Set power mode, run `jetson_clocks`, check for throttling |

---

## 16. Learning Roadmap

**Stage 1 — Basics** ✅ (you are here)
- [x] Vector add
- [x] GPU load test + jtop
- [x] Dot product with shared memory reduction
- [ ] Add `CUDA_CHECK` and CUDA event timing to your programs

**Stage 2 — Core skills**
- [ ] Matrix multiplication: naive version → tiled shared-memory version, compare speed
- [ ] Matrix transpose (learn coalescing + bank conflicts)
- [ ] Parallel reduction with warp shuffle
- [ ] Prefix sum (scan)
- [ ] Histogram with atomics
- [ ] Compare `cudaMalloc` vs managed vs zero-copy speed on your Orin

**Stage 3 — Optimization**
- [ ] Profile every program with `nsys` and `ncu`
- [ ] Measure bandwidth and compare with board maximum
- [ ] FP16 versions of your kernels
- [ ] Streams and overlapping work
- [ ] Use Thrust and cuBLAS, compare with your own kernels

**Stage 4 — Real Jetson applications**
- [ ] Image processing: load an image, grayscale + blur on GPU
- [ ] Camera → GPU processing pipeline (GStreamer / VPI)
- [ ] Run an AI model with TensorRT (FP16 then INT8)
- [ ] Try the DLA for AI inference
- [ ] DeepStream video analytics

### Recommended resources

- **CUDA C++ Programming Guide** — NVIDIA's official reference
- **CUDA C++ Best Practices Guide** — optimization rules
- **CUDA for Tegra** (in CUDA docs) — Jetson memory details
- **NVIDIA Jetson Developer Forums** — answers to Jetson-specific problems
- **NVIDIA cuda-samples** (GitHub) — many example programs
- **jetson-containers** and **jetson-inference** by dusty-nv (GitHub) — AI on Jetson
- Book: *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj)

---

## 17. Command Cheat Sheet

```bash
# ---------- Compile ----------
nvcc -arch=sm_87 app.cu -o app              # normal build
nvcc -O3 -arch=sm_87 app.cu -o app          # optimized
nvcc -arch=sm_87 -use_fast_math app.cu -o app  # faster (less accurate) math
nvcc -G -g -arch=sm_87 app.cu -o app        # debug build

# ---------- System info ----------
cat /etc/nv_tegra_release                   # L4T version
nvcc --version                              # CUDA version
free -h                                     # RAM usage

# ---------- Monitor ----------
jtop                                        # dashboard (q to quit)
sudo tegrastats                             # text stats

# ---------- Power ----------
sudo nvpmodel -q                            # current power mode
sudo nvpmodel -m 0                          # set mode 0
sudo jetson_clocks                          # max clocks

# ---------- Profile ----------
nsys profile -o timeline ./app              # whole-program timeline
sudo $(which ncu) -c 1 ./app                # analyze one kernel
sudo $(which ncu) -c 1 --set full -o rep ./app  # full report file

# ---------- Debug ----------
compute-sanitizer ./app                     # memory errors
compute-sanitizer --tool racecheck ./app    # race conditions
cuda-gdb ./app                              # GPU debugger

# ---------- PATH setup (if nvcc not found) ----------
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

*Key idea to remember: on Jetson, the CPU and GPU share one pool of RAM with limited bandwidth. Avoid unnecessary copies, read memory efficiently, use FP16 where possible, and always measure.*


**The CUDA code you write is mostly the same**, but **the hardware, memory model, and software setup are quite different**. A CUDA kernel that runs on your laptop will usually run on a Jetson with little or no change. What changes is how you set things up, how you move data, and how you optimize.

## 1. Hardware architecture

On a **laptop**, you typically have an x86 CPU (Intel/AMD) and a separate (discrete) NVIDIA GPU with its own dedicated VRAM, connected over PCIe.

On a **Jetson** (Nano, Xavier, Orin), everything is on one chip (an SoC). It has an **ARM64 CPU** and an **integrated GPU**, and both share the **same physical RAM**. There is no separate VRAM and no PCIe link between them.

## 2. Memory handling (the biggest practical difference)

On a laptop, you normally copy data from CPU memory to GPU memory with `cudaMemcpy`, and these PCIe transfers are often a bottleneck.

On a Jetson, the CPU and GPU share memory, so those copies are just moving data around inside the same RAM. Calling `cudaMemcpy` is wasted work in many cases. Instead, you often use:

- **Zero-copy / mapped memory** (`cudaHostAlloc` with `cudaHostAllocMapped`)
- **Unified memory** (`cudaMallocManaged`)

These let the CPU and GPU work on the same buffer directly. Which one is fastest depends on the Jetson model and on how often the CPU touches the data, so it's worth benchmarking. NVIDIA has a "CUDA for Tegra" guide that covers this in detail.

Also remember that RAM is shared. If a Jetson has 8 GB, the OS, CPU programs, and GPU all split that 8 GB.

## 3. Software stack and installation

On a laptop, you install the standard NVIDIA driver and CUDA Toolkit for Windows or Linux x86.

On a Jetson, you **don't install CUDA the normal way**. You flash the board with **JetPack**, which includes:

- Linux for Tegra (an Ubuntu-based OS)
- CUDA, cuDNN, and TensorRT, built for that board

A few other differences come from this setup:

- **Python libraries** like PyTorch and TensorFlow need **ARM64 (aarch64) Jetson builds**. A plain `pip install torch` often gives you a version without GPU support, so use NVIDIA's Jetson wheels or containers.
- **`nvidia-smi` is limited or missing.** Use `tegrastats` or the popular `jtop` tool instead.

## 4. Compiling your code

Each GPU has its own compute capability, and you should compile for the right one:

| Device | Compute capability | nvcc flag |
|---|---|---|
| Jetson Nano | 5.3 | `-arch=sm_53` |
| Jetson Xavier | 7.2 | `-arch=sm_72` |
| Jetson Orin | 8.7 | `-arch=sm_87` |

You can compile directly on the Jetson (slower), or cross-compile on your x86 laptop for ARM64.

## 5. Performance and power

A Jetson GPU is **much smaller** than a laptop GPU. It has fewer CUDA cores and less memory bandwidth, and it is designed for **performance per watt** rather than raw speed.

Jetson also has **power modes**. You can switch between them with `sudo nvpmodel -m <mode>`, and `sudo jetson_clocks` locks the clocks at maximum for benchmarking. Your results can change a lot depending on the power mode, so always note which one you used.

## 6. Extra hardware on Jetson

Some Jetson models have additional accelerators that laptops don't:

- **DLA** (Deep Learning Accelerator) for running neural networks at low power, used through TensorRT
- **PVA / VIC** for vision and image processing, used through **VPI**
- **Hardware video encode/decode**, used through GStreamer or **DeepStream**

For AI work, the usual Jetson workflow is to train on a laptop or PC, then convert the model to **TensorRT** with FP16 or INT8 precision and deploy it on the Jetson.

## Typical workflow

1. **Develop and debug** your CUDA code or model on your laptop.
2. **Move it to the Jetson** and recompile for its architecture.
3. **Optimize for Jetson**: replace unnecessary memory copies with zero-copy or unified memory, use TensorRT for models, and tune the power mode.
4. **Profile** with Nsight Systems, `tegrastats`, or `jtop`.

If you tell me which Jetson you have (Nano, Orin Nano, AGX Orin, etc.) and what you're building (CUDA kernels, AI inference, computer vision), I can give you more specific setup steps or example code.jto



CUDA is installed but not in your PATH. Add it:

'''bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

sudo apt update
sudo apt install nvidia-jetpack


source ~/.bashrc
nvcc --version'''

sudo apt update
sudo apt install python3-pip
sudo pip3 install -U jetson-stats
sudo systemctl restart jtop.service
sudo reboot
jtop


nvcc -arch=sm_87 test.cu -o test && ./test 

sudo $(which ncu) -c 1 --set full -o report ./gpu_load

Nsight Compute (`ncu`) is a profiler that measures **how efficiently a kernel uses the GPU**. It comes with JetPack's CUDA toolkit.

## 1. Find it

```bash
which ncu
```

If it isn't found, look for it in one of these places:

```bash
ls /usr/local/cuda/bin/ncu
ls /opt/nvidia/nsight-compute/*/ncu
```

## 2. Profile one kernel launch

Your program launches the kernel hundreds of times, and profiling each launch is slow, so profile just **one** with `-c 1`. It needs `sudo` to read GPU counters:

```bash
sudo $(which ncu) -c 1 ./gpu_load
```

The `$(which ncu)` part is there because `sudo` doesn't use your PATH, so it needs the full path to `ncu`.

## 3. What to look at in the output

| Metric | Meaning |
|---|---|
| **Duration** | How long one kernel took |
| **Compute (SM) Throughput %** | How hard the math units are working |
| **Memory Throughput %** | How hard memory is working |
| **Achieved Occupancy** | How many threads were active compared with the maximum possible |

For your `burn` kernel, expect **high Compute** and **low Memory** throughput, because it does a lot of math on very little data. That makes it a **compute-bound** kernel. A kernel with high Memory throughput and low Compute would be **memory-bound**.

## 4. Get the full detailed report (optional)

```bash
sudo $(which ncu) -c 1 --set full -o report ./gpu_load
```

This creates `report.ncu-rep`. You can open it in the graphical **Nsight Compute** app (`ncu-ui`) on the Jetson or on your laptop. The app shows charts, a roofline plot, and suggestions for making the kernel faster.

**Tip:** `ncu` analyzes **one kernel** in depth. To see the **whole program timeline** (CPU, GPU, memory copies), use **Nsight Systems** (`nsys`) instead.