# OrinVision

**A GPU image-processing pipeline and benchmark suite for NVIDIA Jetson Orin**, built with CUDA C++. It is designed to run **headless over SSH** (no monitor).

```
 RGB image ─► [1] Grayscale ─► [2] Histogram equalization ─► [3] Gaussian blur ─► [4] Sobel edges
                                   (histogram → LUT → apply)      (naive / separable)
```

The project exercises the core skills of CUDA programming:

| Skill | Where to look |
|---|---|
| Grid-stride loops, coalesced memory access | `src/gpu/color.cu` |
| Shared-memory atomics, parallel prefix sum (scan) | `src/gpu/histogram.cu` |
| Constant memory, shared-memory tiling with halo, naive vs optimized | `src/gpu/blur.cu` |
| 2D tiling, `__syncthreads()` rules | `src/gpu/sobel.cu` |
| **Jetson memory modes** (device / managed / zero-copy) | `include/gpu_buffer.hpp` |
| Streams + CUDA-event timing without extra syncs | `src/pipeline.cpp` |
| Bit-exact host/device shared code (`__host__ __device__`) | `include/common.hpp` |
| Automated GPU-vs-CPU verification, edge-case tests | `src/verify.cpp`, `tests/` |
| CMake, ctest, OpenMP CPU baseline, benchmark automation | `CMakeLists.txt`, `scripts/` |

---

## 1. Project structure

```
orin_vision/
├── CMakeLists.txt
├── README.md
├── CLAUDE.md                 Context file for Claude / Claude Code
├── include/
│   ├── common.hpp            Shared types + host/device math helpers
│   ├── cuda_utils.hpp        CUDA_CHECK, divUp, smCount
│   ├── gpu_buffer.hpp        RAII buffer: device / managed / zerocopy
│   ├── gpu_filters.hpp       Kernel launcher declarations
│   ├── cpu_filters.hpp       CPU reference declarations
│   ├── pipeline.hpp          GpuPipeline class
│   ├── image.hpp             Image struct + image loading (JPG/PNG/PPM...) and PPM/PGM saving
│   └── verify.hpp            GPU-vs-CPU comparison
├── src/
│   ├── main.cpp              Command-line app (benchmark, report, CSV)
│   ├── pipeline.cpp          Runs the stages and times them
│   ├── cpu_filters.cpp       CPU reference (OpenMP multi-core)
│   ├── image.cpp             File I/O + synthetic test image
│   ├── verify.cpp            Stage-isolated verification
│   └── gpu/
│       ├── color.cu          RGB → gray
│       ├── histogram.cu      Histogram, scan, LUT
│       ├── blur.cu           Gaussian blur (naive + separable)
│       └── sobel.cu          Sobel edge detection
├── tests/
│   └── test_pipeline.cpp     ctest: all modes, odd sizes, edge cases
└── scripts/
    ├── build.sh              Configure + build + test   (Jetson)
    ├── run_benchmarks.sh     Full sweep + tegrastats    (Jetson)
    ├── plot_results.py       CSV → PNG charts           (Jetson or PC)
    ├── convert_images.py     PGM → PNG, photo → PPM     (Jetson or PC)
    └── pc_sync.sh            push code / pull results   (PC)
```

---

## 2. Requirements (on the Jetson)

- JetPack 6.x (CUDA 12.x), Jetson Orin (compute capability 8.7)
- `cmake` ≥ 3.18 and Python helpers:

```bash
sudo apt update
sudo apt install cmake python3-matplotlib python3-pil tmux
```

---

## 3. Get the project onto the Jetson

From your **PC**, copy the zip and unpack it on the Jetson:

```bash
# On your PC (replace the IP with your Jetson's IP)
scp orin_vision.zip octo@192.168.1.50:~/jetson_gpu_programing/

# Then on the Jetson (via ssh)
ssh octo@192.168.1.50
cd ~/jetson_gpu_programing
python3 -m zipfile -e orin_vision.zip .      # or: unzip orin_vision.zip
cd orin_vision
chmod +x scripts/*.sh
```

> Find the Jetson's IP by running `hostname -I` on it.

---

## 4. Build and test

```bash
./scripts/build.sh            # Release build + runs all tests
./scripts/build.sh Debug      # Debug build in build-debug/ (for cuda-gdb)
```

Expected end of the test output:

```
... 
142 / 142 checks passed
100% tests passed, 0 tests failed out of 1
```

---

## 5. Run

```bash
./build/orin_vision                       # 1080p test image, defaults
./build/orin_vision --help                # all options
./build/orin_vision -g 3840x2160 -n 50    # 4K, 50 iterations
./build/orin_vision -m device             # classic cudaMalloc + cudaMemcpy
./build/orin_vision -b naive              # unoptimized blur, for comparison
./build/orin_vision -i photo.jpg -r 6     # your own image (jpg/png/ppm...), stronger blur
```

Example report layout (your numbers will differ):

```
Stage            GPU avg ms  GPU min ms      CPU ms   Speedup  Eff. BW GB/s
---------------------------------------------------------------------------
upload                0.004       0.003           -         -             -
grayscale               ...
histogram               ...
build LUT               ...
apply LUT               ...
gaussian blur           ...
sobel edges             ...
download              0.003       0.003           -         -             -
---------------------------------------------------------------------------
TOTAL                   ...

Verification (each GPU stage vs CPU on the same input):
  grayscale      PASS  max diff 0 (tolerance 0) ...
```

**Columns:**
- **GPU avg / min**: time per stage measured with CUDA events. *min* is the best case; *avg* includes noise.
- **CPU ms**: same stage on all CPU cores (OpenMP), so the speedup is a fair comparison.
- **Eff. BW**: minimum bytes the stage must move ÷ time. Compare it with your board's memory bandwidth (~68 GB/s on Orin Nano 8GB, higher in "Super" mode). If a memory-bound stage reaches 60–80% of that, it is well optimized.

**Exit codes:** `0` OK, `1` error, `3` verification failed. Useful in scripts.

### Using your own photos

JPG, PNG, BMP, TGA, GIF (first frame), PPM and PGM are read directly — no conversion needed.
The format is detected from the file contents (decoder: `third_party/stb/stb_image.h`).
Color images become RGB (a PNG alpha channel is ignored); gray images are used as gray.

```bash
./build/orin_vision -i photo.jpg
./build/orin_vision -i screenshot.png -o out
```

---

## 6. Headless workflow over SSH

### Recommended: VS Code Remote-SSH

Install the **Remote - SSH** extension on your PC, connect to `octo@<jetson-ip>`, and open the project folder. You edit files on the Jetson directly, use its terminal, and can open the PNG results in the editor. It is the most comfortable way to work without a monitor.

### Keep long jobs alive with tmux

If your SSH connection drops, normal programs are killed. `tmux` keeps them running:

```bash
tmux new -s work        # start a session
./scripts/run_benchmarks.sh
# Ctrl+b, then d        -> detach (job keeps running, you can log out)
tmux attach -t work     # come back later
```

### Monitor the GPU in a second SSH window

```bash
jtop                    # works fine over SSH
```

### View result images on your PC

The program writes `.pgm` / `.ppm` images. Convert them to PNG and build one overview picture, then copy it to your PC:

```bash
# On the Jetson
python3 scripts/convert_images.py results

# On your PC
scp octo@<jetson-ip>:~/jetson_gpu_programing/orin_vision/results/overview.png .
```

Or sync everything with the helper script on your PC (Linux, macOS, WSL, Git Bash):

```bash
export JETSON=octo@192.168.1.50
./scripts/pc_sync.sh pull       # results  Jetson -> PC
./scripts/pc_sync.sh push       # code     PC -> Jetson
./scripts/pc_sync.sh run -g 1280x720
```

> Tip: set up SSH keys once (`ssh-copy-id octo@<jetson-ip>`) so you don't type your password every time.

---

## 7. Full benchmark sweep

```bash
tmux new -s bench
./scripts/run_benchmarks.sh                       # 3 sizes × 3 memory modes × 2 blur modes
ITERS=50 SIZES="1920x1080" ./scripts/run_benchmarks.sh   # custom
```

Each run creates `results/benchmark_<date>/` with:

| File | Content |
|---|---|
| `system_info.txt` | Board model, L4T, power mode, CUDA version |
| `benchmark.csv` | Every stage of every run (open in Excel/LibreOffice) |
| `summary.log` | One line per run |
| `tegrastats.log` | GPU load, temperatures, power during the sweep |
| `1_total_time.png` | Total time per frame, all configurations + CPU |
| `2_stages_<size>.png` | Time spent in each stage |
| `3_speedup_<size>.png` | GPU speedup over CPU per stage |
| `images_<size>/` | Output images |

For stable, comparable numbers:

```bash
sudo nvpmodel -q          # note the power mode (it is saved in system_info.txt)
sudo jetson_clocks        # lock clocks at max before benchmarking
```

---

## 8. Profiling

```bash
# Whole-program timeline (CPU, kernels, copies) - open the file in Nsight Systems on your PC
nsys profile -o timeline ./build/orin_vision -n 10

# Deep analysis of the blur kernels (compiled with -lineinfo, so source lines are shown)
sudo $(which ncu) --kernel-name regex:blur -c 2 ./build/orin_vision -n 1 -w 0

# Full report file for the Nsight Compute GUI on your PC
sudo $(which ncu) --set full -c 8 -o report ./build/orin_vision -n 1 -w 0
```

Copy `.nsys-rep` / `.ncu-rep` files to your PC with `scp` and open them in the free Nsight Systems / Nsight Compute desktop apps.

```bash
# Memory bugs and race conditions
compute-sanitizer ./build/orin_vision -g 640x480 -n 1 -w 0
compute-sanitizer --tool racecheck ./build/orin_vision -g 640x480 -n 1 -w 0
```

---

## 9. Experiments to try (and what you should learn)

1. **Memory modes:** run `-m device`, `-m managed`, `-m zerocopy`. In device mode, upload/download take real time; in the others they are ~0. But do the *kernels* get slower in zero-copy mode? (Zero-copy memory may be accessed with different caching.)
2. **Naive vs separable blur:** compare `-b naive` and `-b separable` at `-r 2`, `-r 8`, `-r 15`. The naive cost grows with r², the separable cost with r.
3. **Image size:** at 320x240 the GPU may barely beat the CPU (launch overhead dominates). At 4K the GPU wins big. Where is the crossover?
4. **Power modes:** run the same benchmark in different `nvpmodel` modes and compare FPS per watt using `tegrastats.log`.
5. **Profile:** use `ncu` to check whether each kernel is memory-bound or compute-bound. Is `grayscale` close to peak bandwidth?

## 10. Ideas to extend the project

- **FP16 blur** using `__half` (half the memory traffic)
- **Fuse kernels**: grayscale + histogram in one pass (read the image once)
- **Multi-stream pipeline**: process frame N+1 while frame N is downloading
- **CUDA Graphs** to cut launch overhead for small images
- **Camera input** with GStreamer / `nvarguscamerasrc`, saving processed frames
- **Compare with libraries**: NPP or VPI implementations of the same filters
- **Canny edge detector** (adds non-maximum suppression + hysteresis)

### Adding a new stage

1. Add the math helper to `include/common.hpp` (as `HD inline`) so CPU and GPU share it.
2. Write the kernel in `src/gpu/yourstage.cu` and declare its launcher in `include/gpu_filters.hpp`.
3. Add the CPU version in `src/cpu_filters.cpp`.
4. Add a `Stage` enum entry, call it in `GpuPipeline::process()` and `cpu::runPipeline()`.
5. Add a check in `verify.cpp`, add the `.cu` file to `CMakeLists.txt`, and run `ctest`.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `nvcc not found` | `export PATH=/usr/local/cuda/bin:$PATH` (build.sh does this) |
| `cmake: command not found` | `sudo apt install cmake` |
| Build fails with "unsupported gpu architecture" | Your CUDA is too old; check `nvcc --version` |
| `CUDA ERROR ... out of memory` | 4K needs ~100–150 MB; close other GPU programs, check `jtop` |
| Segfault in managed mode after editing code | You touched managed memory while a kernel was running. Sync first |
| Numbers change a lot between runs | `sudo jetson_clocks`, check temperature/throttling in `jtop` |
| Charts not created | `sudo apt install python3-matplotlib` |
| SSH disconnect killed the benchmark | Run it inside `tmux` |
| `the GPU kernels cannot run on ...` / `no kernel image is available` | Binary was built for another GPU (e.g. Jetson build copied to a laptop). Rebuild on this machine: `rm -rf build && ./scripts/build.sh`, or force it with `-DCMAKE_CUDA_ARCHITECTURES=86` (RTX 30xx) / `87` (Orin) / `native` |

---

## 12. How to run the program (Jetson or laptop)

The same code runs on a **Jetson Orin** and on a **PC / laptop with an NVIDIA GPU**.
At start-up the program detects which one it is on (`src/platform.cpp`) and picks the best settings automatically.

| | Jetson Orin | PC / laptop (e.g. RTX 3080) |
|---|---|---|
| Needs | JetPack 6.x (CUDA 12.x) | NVIDIA driver + CUDA Toolkit 12.x, `cmake`, `g++` |
| Compiled for (automatic) | `sm_87` | `sm_75`, `sm_86`, `sm_89` (RTX 20/30/40) |
| Default memory mode (`--mem auto`) | `managed` (CPU and GPU share RAM) | `device` (explicit copies to VRAM) |
| First line of the output | `Platform : Jetson - ...` | `Platform : PC / laptop (discrete GPU)` |

**Step 1 – build** (always build on the machine you run on; a Jetson binary does not run on a laptop and vice versa):

```bash
cd OrinVision                 # Jetson: ~/jetson_gpu_programing/orin_vision
./scripts/build.sh            # configure + build + run all tests
```

**Step 2 – run:**

```bash
./build/orin_vision                          # 1080p generated test image, auto settings
./build/orin_vision --help                   # list every option
./build/orin_vision -g 3840x2160 -n 50       # 4K image, 50 timed iterations
./build/orin_vision -i photo.jpg -o out      # your own image (jpg, png, bmp, ppm...), results in out/
./build/orin_vision -m zerocopy              # force a memory mode (auto|device|managed|zerocopy)
./build/orin_vision -b naive                 # slow blur, to compare with separable
./build/orin_vision -q --csv results/my.csv  # one summary line + append timings to CSV
```

**Step 3 – look at the results:**

```bash
python3 scripts/convert_images.py results    # PGM -> PNG + overview.png
./scripts/run_benchmarks.sh                  # full sweep (sizes x memory x blur) + charts
```

The program prints the platform, GPU, per-stage timings, CPU-vs-GPU speedup and a PASS/FAIL check for every stage. It exits with `0` when everything is OK, `1` on an error and `3` if the verification failed.
