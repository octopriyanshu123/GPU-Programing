#!/usr/bin/env python3
"""
plot_results.py - Turn benchmark.csv into PNG charts.

Works headless (no monitor needed): charts are written as PNG files next
to the CSV, so you can copy them to your PC with scp.

Usage:
    python3 scripts/plot_results.py results/benchmark_XXXX/benchmark.csv

Requires: sudo apt install python3-matplotlib
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

KERNEL_STAGES = ["grayscale", "histogram", "build LUT", "apply LUT",
                 "gaussian blur", "sobel edges",
                 "harris corners", "fast corners", "orb descriptors", "canny edges"]
ALL_STAGES = ["upload"] + KERNEL_STAGES + ["download"]


def load(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            for key in ("gpu_avg_ms", "gpu_min_ms", "cpu_ms", "speedup", "eff_bw_gbps"):
                r[key] = float(r[key])
            r["size"] = f"{r['width']}x{r['height']}"
            r["pixels"] = int(r["width"]) * int(r["height"])
            rows.append(r)
    return rows


def sizes_sorted(rows):
    return sorted({r["size"] for r in rows}, key=lambda s: next(
        r["pixels"] for r in rows if r["size"] == s))


def chart_total_time(rows, out_dir):
    """Total GPU time per image size, one bar per memory/blur combination."""
    sizes = sizes_sorted(rows)
    combos = sorted({(r["mem_mode"], r["blur_mode"]) for r in rows})
    total = {(r["size"], r["mem_mode"], r["blur_mode"]): r["gpu_avg_ms"]
             for r in rows if r["stage"] == "total"}
    cpu = {r["size"]: r["cpu_ms"] for r in rows if r["stage"] == "total" and r["cpu_ms"] > 0}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.8 / (len(combos) + (1 if cpu else 0))
    for i, (mem, blur) in enumerate(combos):
        vals = [total.get((s, mem, blur), 0) for s in sizes]
        ax.bar([x + i * width for x in range(len(sizes))], vals, width, label=f"GPU {mem} / {blur}")
    if cpu:
        vals = [cpu.get(s, 0) for s in sizes]
        ax.bar([x + len(combos) * width for x in range(len(sizes))], vals, width,
               label="CPU (all cores)", color="gray")
    ax.set_xticks([x + width * len(combos) / 2 for x in range(len(sizes))])
    ax.set_xticklabels(sizes)
    ax.set_ylabel("time per frame (ms, lower is better)")
    ax.set_yscale("log")
    ax.set_title("Full pipeline time per frame")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")
    path = out_dir / "1_total_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def chart_stage_breakdown(rows, out_dir, size):
    """Stacked bars: where does the time go, per configuration."""
    data = defaultdict(dict)
    for r in rows:
        if r["size"] == size and r["stage"] in ALL_STAGES:
            data[(r["mem_mode"], r["blur_mode"])][r["stage"]] = r["gpu_avg_ms"]
    combos = sorted(data)
    if not combos:
        return None

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottoms = [0.0] * len(combos)
    labels = [f"{m}\n{b}" for m, b in combos]
    for stage in ALL_STAGES:
        vals = [data[c].get(stage, 0) for c in combos]
        ax.bar(labels, vals, bottom=bottoms, label=stage)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("GPU time (ms)")
    ax.set_title(f"Time per stage at {size}")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    path = out_dir / f"2_stages_{size}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def chart_speedup(rows, out_dir, size):
    """GPU speedup over the multi-core CPU for each kernel stage."""
    best = {}
    for r in rows:
        if r["size"] == size and r["stage"] in KERNEL_STAGES and r["speedup"] > 0:
            best[r["stage"]] = max(best.get(r["stage"], 0), r["speedup"])
    if not best:
        return None
    stages = [s for s in KERNEL_STAGES if s in best]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stages, [best[s] for s in stages], color="tab:green")
    ax.bar_label(bars, fmt="%.1fx")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="same speed as CPU")
    ax.set_ylabel("speedup (higher is better)")
    ax.set_title(f"Best GPU speedup vs CPU per stage at {size}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20)
    path = out_dir / f"3_speedup_{size}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    rows = load(csv_path)
    if not rows:
        sys.exit("CSV is empty")
    out_dir = csv_path.parent

    made = [chart_total_time(rows, out_dir)]
    for size in sizes_sorted(rows):
        made.append(chart_stage_breakdown(rows, out_dir, size))
        made.append(chart_speedup(rows, out_dir, size))

    print("Charts written:")
    for p in made:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
