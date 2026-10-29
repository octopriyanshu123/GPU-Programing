#!/usr/bin/env bash
# =====================================================================
# run_benchmarks.sh - Full benchmark sweep with system monitoring
# =====================================================================
# Runs every combination of image size x memory mode x blur mode, logs
# tegrastats in the background, saves system info, and makes charts.
#
# Over SSH, run it inside tmux so it survives a disconnect:
#     tmux new -s bench
#     ./scripts/run_benchmarks.sh
#     (detach: Ctrl+b then d    re-attach later: tmux attach -t bench)
#
# Options (environment variables):
#     ITERS=50 SIZES="1920x1080 3840x2160" ./scripts/run_benchmarks.sh
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./build/orin_vision
ITERS="${ITERS:-30}"
RADIUS="${RADIUS:-4}"
read -r -a SIZES <<< "${SIZES:-1280x720 1920x1080 3840x2160}"
MEMS=(device managed zerocopy)
BLURS=(naive separable)

[[ -x "$BIN" ]] || { echo "Not built yet. Run: ./scripts/build.sh"; exit 1; }

OUT="results/benchmark_$(date +%Y%m%d_%H%M%S)"
CSV="$OUT/benchmark.csv"
mkdir -p "$OUT"

# ---------- System information ----------
{
    echo "date        : $(date)"
    echo "model       : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
    echo "L4T         : $(head -n1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
    echo "power mode  : $( (nvpmodel -q 2>/dev/null || echo 'unknown (run: sudo nvpmodel -q)') | tr '\n' ' ')"
    echo "CUDA        : $( (/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -n1) || echo unknown)"
    echo "kernel      : $(uname -r)"
    echo "iterations  : $ITERS"
    echo "blur radius : $RADIUS"
} | tee "$OUT/system_info.txt"
echo

# ---------- Background monitoring ----------
TEGRA_PID=""
if command -v tegrastats >/dev/null 2>&1; then
    tegrastats --interval 500 > "$OUT/tegrastats.log" 2>&1 &
    TEGRA_PID=$!
    echo "tegrastats logging -> $OUT/tegrastats.log"
fi
cleanup() { [[ -n "$TEGRA_PID" ]] && kill "$TEGRA_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ---------- Sweep ----------
TOTAL=$(( ${#SIZES[@]} * ${#MEMS[@]} * ${#BLURS[@]} ))
RUN=0
FAILED=0
for size in "${SIZES[@]}"; do
    for mem in "${MEMS[@]}"; do
        for blur in "${BLURS[@]}"; do
            RUN=$((RUN + 1))
            printf "[%2d/%d] " "$RUN" "$TOTAL"
            # Save the images only once per size (1st combination)
            SAVE_ARGS=(--no-save)
            [[ "$mem" == "${MEMS[0]}" && "$blur" == "${BLURS[0]}" ]] && SAVE_ARGS=(-o "$OUT/images_$size")
            if ! "$BIN" -g "$size" -m "$mem" -b "$blur" -r "$RADIUS" -n "$ITERS" \
                    --cpu-iters 1 --csv "$CSV" --quiet "${SAVE_ARGS[@]}" | tee -a "$OUT/summary.log"; then
                FAILED=$((FAILED + 1))
            fi
            sleep 1   # let clocks/temperature settle between runs
        done
    done
done

cleanup
echo
echo "Results folder: $OUT"
[[ $FAILED -gt 0 ]] && echo "WARNING: $FAILED run(s) failed verification - check summary.log"

# ---------- Charts ----------
if python3 -c "import matplotlib" 2>/dev/null; then
    python3 scripts/plot_results.py "$CSV"
else
    echo "For charts install matplotlib:  sudo apt install python3-matplotlib"
fi
