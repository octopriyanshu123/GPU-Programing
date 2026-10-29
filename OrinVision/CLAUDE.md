# CLAUDE.md — OrinVision project

Context for Claude / Claude Code when working on this project.

## What this is
CUDA C++ image-processing pipeline + benchmark for **Jetson Orin** (sm_87, CUDA 12.6, JetPack 6.x, 8 GB shared RAM).
Pipeline: RGB → grayscale → histogram equalization → Gaussian blur (naive | separable) → Sobel edges,
then feature extraction on the blurred image: Harris → FAST → ORB → Canny (`-f/--features`, default all).
The user is learning GPU programming; explain changes simply and keep heavy code comments.

## Environment
- Jetson is **headless**, used over SSH (user `octo`, host `tegra-ubuntu`). No display: output is files + terminal.
- Project path on Jetson: `~/jetson_gpu_programing/orin_vision`
- Long jobs run inside `tmux`. Monitor with `jtop` / `tegrastats`.

## Build / test / run
```bash
./scripts/build.sh                 # Release + ctest
./scripts/build.sh Debug           # build-debug/, -G device debug
./build/orin_vision --help
./scripts/run_benchmarks.sh        # sweep -> results/benchmark_<date>/
python3 scripts/convert_images.py results   # PGM -> PNG + overview.png
```

## Architecture rules
- Shared CPU/GPU math lives in `include/common.hpp` as `HD inline` functions → CPU and GPU results are bit-exact.
- CPU-visible buffers use `GpuBuffer<T>` (memory mode: device | managed | zerocopy). GPU-only scratch uses plain `cudaMalloc`.
- All kernels run on `GpuPipeline::stream_`; timing via CUDA events between stages; one sync at the end.
- Also runs on a PC/laptop GPU (dev machine: RTX 3080, sm_86, CUDA 12.1). Platform is detected at run time in `src/platform.cpp` (`platform()`, `cudaDeviceProp::integrated` + Tegra files); `--mem auto` = managed on Jetson, device on PC. CMake picks arch from host CPU: aarch64 → 87, x86_64 → 75;86;89.
- **Jetson: no concurrent managed access** → `cudaDeviceSynchronize()` before the CPU touches managed memory (skipped when `concurrentManagedAccess` is true).
- Every CUDA call wrapped in `CUDA_CHECK`; `CUDA_CHECK_LAST()` after each launch.
- No early `return` before `__syncthreads()` in kernels.
- Verification is stage-isolated (`src/verify.cpp`). Tolerance: 0 everywhere except blur = 1 (FMA rounding).
- Features must stay bit-exact: integer-only math in `common.hpp` (Harris uses 25·R in int64, ORB angle = integer dot-product over 30 bins, no atan2). GPU keypoint lists are in atomic order → verification sorts them; ORB is verified on the GPU's own FAST list.
- Canny hysteresis is the one exception to "one sync at the end": the host reads a changed-flag per round (`gpu::canny`).
- New stage checklist: README section "Adding a new stage".

## Conventions
- C++17, 4-space indent, `namespace gpu` / `namespace cpu` for filter functions.
- Every new kernel needs: CPU reference, verify check, test coverage incl. odd sizes (1x1, 7x5, 1023x769).
- Keep `-Wall -Wextra` clean.

## Status / next ideas
- v1.0: all stages, 3 memory modes, 2 blur modes, tests, benchmark scripts, charts.
- v1.1: runs on Jetson + PC (run-time detection), JPG/PNG input (stb_image), feature extraction (Harris, FAST, ORB, Canny).
- Ideas: FP16 blur, kernel fusion (gray+histogram), multi-stream frames, CUDA Graphs, camera input, NPP/VPI comparison, ORB matching, warp-aggregated atomics.
