#!/usr/bin/env python3
"""
exp_drawer.py — Experimental Roofline Drawer (CPU + MBLT Aries NPU)

Workflow:
  1. CPU roofline: compiles & runs hw_bench (measures peak GFLOPS and DRAM bandwidth).
  2. NPU roofline: uses spec values for the hardware roof (80 TOPS, 66.7 GB/s),
     then times actual model inference via npu_bench to get measured data points.
  3. Plots both rooflines and all data points on one chart.

Usage:
  python3 exp_drawer.py

Add NPU model entries in npu_models below.
Build npu_bench first:  make -f Makefile.npu
"""

import subprocess
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

_DIR      = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CPU benchmark paths
# ---------------------------------------------------------------------------
CPU_BENCH_SRC = os.path.join(_DIR, "hw_bench.cc")
CPU_BENCH_BIN = os.path.join(_DIR, "hw_bench")

# ---------------------------------------------------------------------------
# NPU benchmark path (build separately: make -f Makefile.npu)
# ---------------------------------------------------------------------------
NPU_BENCH_BIN = os.path.join(_DIR, "npu_bench")

# ---------------------------------------------------------------------------
# NPU hardware roof values.
# ---------------------------------------------------------------------------
NPU_PEAK_GOPS_SPEC     = 80 * 1000   # 80 TOPS spec
NPU_BANDWIDTH_GBS_SPEC = 66.7        # GB/s spec

# Paths to compiled benchmark models (copy from workstation after compile_bench.py)
NPU_COMPUTE_BENCH_MXQ   = os.path.join(_DIR, "compute_bench.mxq")
NPU_BANDWIDTH_BENCH_MXQ = os.path.join(_DIR, "bandwidth_bench.mxq")

# Known ops/bytes for the synthetic benchmark models (from create_bench_onnx.py)
# compute_bench: 8x 1x1 Conv [1, 1024, 64, 64]
COMPUTE_BENCH_GOPS   = 8 * 2 * 1024 * 1024 * 64 * 64 / 1e9        # ~68.7 GOPS
COMPUTE_BENCH_GBYTES = (8 * 1024 * 1024 + 1024 * 64 * 64 * 2) / 1e9  # 8MB weights + 8MB activations (int8)

# bandwidth_bench: Depthwise Conv [1, 256, 512, 512] kernel [3,3]
BANDWIDTH_BENCH_GOPS   = 2 * 256 * 512 * 512 * 9 / 1e9  # ~1.2 GOPS
BANDWIDTH_BENCH_GBYTES = 256 * 512 * 512 * 2 / 1e9      # ~128 MB input + output activations (int8)


# ===========================================================================
# CPU benchmark helpers
# ===========================================================================

def _cpu_needs_recompile():
    if not os.path.exists(CPU_BENCH_BIN):
        return True
    return os.path.getmtime(CPU_BENCH_SRC) > os.path.getmtime(CPU_BENCH_BIN)

def _compile_cpu_bench():
    cmd = ["g++", "-O3", "-march=native", "-o", CPU_BENCH_BIN, CPU_BENCH_SRC]
    print(f"Compiling CPU benchmark: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Compilation failed:\n", r.stderr)
        sys.exit(1)

def measure_cpu():
    if _cpu_needs_recompile():
        _compile_cpu_bench()
    print("Running CPU benchmark (~6s) ...")
    r = subprocess.run([CPU_BENCH_BIN], capture_output=True, text=True)
    if r.stderr:
        print(r.stderr, end="")
    if r.returncode != 0:
        print("CPU benchmark failed:", r.stdout, r.stderr)
        sys.exit(1)
    vals = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            vals[parts[0]] = float(parts[1])
    if "peak_gflops" not in vals or "bandwidth_gbs" not in vals:
        print("Unexpected CPU benchmark output:", r.stdout)
        sys.exit(1)
    return vals["peak_gflops"], vals["bandwidth_gbs"]


# ===========================================================================
# NPU benchmark helpers
# ===========================================================================

def measure_npu_model(mxq_path, num_runs, model_gops, model_gbytes):
    """
    Runs npu_bench on a single .mxq model.
    Returns (achieved_gops, arithmetic_intensity), or None if unavailable.
    """
    if not os.path.exists(NPU_BENCH_BIN):
        print(f"  [skip] {NPU_BENCH_BIN} not found — build with: make -f Makefile.npu")
        return None
    if not os.path.exists(mxq_path):
        print(f"  [skip] model file not found: {mxq_path}")
        return None
        
    cmd = [NPU_BENCH_BIN, mxq_path, str(num_runs),
           str(model_gops), str(model_gbytes)]
    print(f"Running NPU benchmark: {os.path.basename(mxq_path)} ...")
    
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [skip] npu_bench failed:\n{r.stdout}{r.stderr}")
        return None
        
    vals = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            vals[parts[0]] = float(parts[1])
            
    print(f"  avg latency : {vals.get('avg_latency_ms', 0):.2f} ms")
    print(f"  achieved    : {vals.get('achieved_gops', 0):.0f} GOPS")
    print(f"  AI (ops/B)  : {vals.get('arithmetic_intensity', 0):.1f}")
    
    return vals["achieved_gops"], vals["arithmetic_intensity"]


# ===========================================================================
# Plotting
# ===========================================================================

def plot_rooflines(cpu_peak, cpu_bw, npu_models, cpu_models=None,
                   npu_peak_gops=None, npu_bw_gbs=None,
                   filename="roofline_experimental.png"):
    fig, ax = plt.subplots(figsize=(14, 9))
    x = np.logspace(-2, 4, 1000)

    # --- 1. CPU Roofline ---
    y_cpu = np.minimum(cpu_peak, x * cpu_bw)
    ridge_cpu = cpu_peak / cpu_bw
    ax.plot(x, y_cpu, linewidth=4.0, linestyle="-", color="#1f77b4", alpha=1.0,
            label=f"CPU (measured)  {cpu_peak:.0f} GFLOPS | {cpu_bw:.1f} GB/s")
    
    # CPU Ridge Point (bold 제거)
    ax.scatter([ridge_cpu], [cpu_peak], color="white", edgecolors="#1f77b4", s=60, marker="o", linewidths=2, zorder=5)
    ax.annotate(f"Ridge: {ridge_cpu:.1f} ops/B",
                xy=(ridge_cpu, cpu_peak),
                xytext=(ridge_cpu * 2.5, cpu_peak * 0.6),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

    # --- 2. NPU Spec Roofline (촘촘한 빨간 점선) ---
    y_npu_spec = np.minimum(NPU_PEAK_GOPS_SPEC, x * NPU_BANDWIDTH_GBS_SPEC)
    ridge_npu_spec = NPU_PEAK_GOPS_SPEC / NPU_BANDWIDTH_GBS_SPEC
    ax.plot(x, y_npu_spec, linewidth=1.5, linestyle="--", color="red", dashes=(2, 2), alpha=0.8,
            label=f"MBLT Aries NPU (spec)  {NPU_PEAK_GOPS_SPEC:.0f} GOPS | {NPU_BANDWIDTH_GBS_SPEC:.1f} GB/s")
    
    # NPU Spec Ridge Point (bold 제거, 텍스트가 겹치지 않게 살짝 위로 올림)
    ax.scatter([ridge_npu_spec], [NPU_PEAK_GOPS_SPEC], color="white", edgecolors="red", s=60, marker="o", linewidths=2, zorder=5)
    ax.annotate(f"Ridge: {ridge_npu_spec:.1f} ops/B",
                xy=(ridge_npu_spec, NPU_PEAK_GOPS_SPEC),
                xytext=(ridge_npu_spec * 2.5, NPU_PEAK_GOPS_SPEC * 1.5),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

    # --- 3. NPU Measured Roofline & Ridge Point (진한 실선) ---
    if npu_peak_gops is not None and npu_bw_gbs is not None:
        y_npu_meas = np.minimum(npu_peak_gops, x * npu_bw_gbs)
        ridge_npu_meas = npu_peak_gops / npu_bw_gbs
        ax.plot(x, y_npu_meas, linewidth=4.0, linestyle="-", color="#d62728", alpha=1.0,
                label=f"MBLT Aries NPU (measured)  {npu_peak_gops:.0f} GOPS | {npu_bw_gbs:.1f} GB/s")
        
        # NPU Measured Ridge Point (bold 제거)
        ax.scatter([ridge_npu_meas], [npu_peak_gops], color="white", edgecolors="#d62728", s=60, marker="o", linewidths=2, zorder=5)
        ax.annotate(f"Ridge: {ridge_npu_meas:.1f} ops/B",
                    xy=(ridge_npu_meas, npu_peak_gops),
                    xytext=(ridge_npu_meas * 2.5, npu_peak_gops * 0.6),
                    fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

    # --- 4. ACTUAL MODEL POINTS (금색 동그라미) ---
    for m in npu_models:
        ax.scatter(m["ai"], m["perf"],
                   label=f"NPU: {m['name']}  ({m['perf']:.0f} GOPS)",
                   s=250, marker="o", color="#FFD700", edgecolors="black", linewidths=1.5, zorder=15)

    # --- CPU workload points (optional) ---
    if cpu_models:
        for m in cpu_models:
            ax.scatter(m["ai"], m["perf"],
                       label=f"CPU: {m['name']}  ({m['perf']:.0f} GFLOPS)",
                       s=200, marker="o", color="#2ca02c", edgecolors="black", zorder=15)

    # --- Formatting ---
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (ops/Byte)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance (GOPS / GFLOPS)", fontsize=12, fontweight="bold")
    ax.set_title("Experimental Roofline: CPU vs NPU Performance", fontsize=15, pad=20)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10, frameon=True, shadow=True)

    # --- 파일명 중복 방지 로직 ---
    base, ext = os.path.splitext(filename)
    out_path = os.path.join(_DIR, filename)
    counter = 1
    while os.path.exists(out_path):
        out_path = os.path.join(_DIR, f"{base}_{counter}{ext}")
        counter += 1

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n[Visual] Chart successfully saved → {out_path}")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" 1. Measuring CPU Baseline")
    print("="*50)
    cpu_peak, cpu_bw = measure_cpu()
    print(f"\n[RESULT] CPU: {cpu_peak:.1f} GFLOPS peak | {cpu_bw:.1f} GB/s | ridge {cpu_peak/cpu_bw:.1f} ops/B")

    # -----------------------------------------------------------------------
    # NPU models to benchmark.
    # -----------------------------------------------------------------------
    npu_model_configs = [
        {
            "name":   "ResNet-50",
            "mxq":    "/home/brkim/mblt-arch-bench/src/resnet50/resnet50.mxq",
            "gops":   7.7,     # ~7.7 GOPS for ResNet-50 (224x224, int8)
            "gbytes": 0.030,   # ~30 MB weight + activation traffic → 0.030 GB
            "runs":   50,
        },
    ]

    print("\n" + "="*50)
    print(" 2. Measuring NPU Hardware Limits (Synthetic Benchmarks)")
    print("="*50)
    npu_peak_gops = None
    npu_bw_gbs    = None

    result = measure_npu_model(NPU_COMPUTE_BENCH_MXQ,   200, COMPUTE_BENCH_GOPS,   COMPUTE_BENCH_GBYTES)
    if result is not None:
        npu_peak_gops = result[0]
        print(f"[RESULT] NPU peak (measured): {npu_peak_gops:.0f} GOPS")

    result = measure_npu_model(NPU_BANDWIDTH_BENCH_MXQ, 200, BANDWIDTH_BENCH_GOPS, BANDWIDTH_BENCH_GBYTES)
    if result is not None:
        achieved_gops_bw = result[0]
        latency_s  = BANDWIDTH_BENCH_GOPS / achieved_gops_bw
        npu_bw_gbs = BANDWIDTH_BENCH_GBYTES / latency_s
        print(f"[RESULT] NPU bandwidth (measured): {npu_bw_gbs:.1f} GB/s")

    if npu_peak_gops is None or npu_bw_gbs is None:
        print("[WARNING] NPU roof: using spec values (run compute_bench.mxq + bandwidth_bench.mxq to measure)")

    cpu_workloads = []

    print("\n" + "="*50)
    print(" 3. Measuring Actual NPU Models")
    print("="*50)
    all_npu_points = []
    
    for cfg in npu_model_configs:
        print(f"\n[Benchmarking] {cfg['name']} ...")
        result = measure_npu_model(cfg["mxq"], cfg["runs"], cfg["gops"], cfg["gbytes"])
        if result is not None:
            achieved_gops, ai = result
            all_npu_points.append({"name": cfg["name"], "perf": achieved_gops, "ai": ai})
            print(f"  -> Added '{cfg['name']}' to plot collection.")

    # -----------------------------------------------------------------------
    # Plot everything together
    # -----------------------------------------------------------------------
    print("\n" + "="*50)
    print(" 4. Generating Roofline Chart")
    print("="*50)
    if all_npu_points or cpu_workloads:
        plot_rooflines(cpu_peak, cpu_bw, all_npu_points, cpu_workloads,
                       npu_peak_gops=npu_peak_gops, npu_bw_gbs=npu_bw_gbs,
                       filename="roofline_experimental.png")
    else:
        print("[SKIP] No valid model data points to plot.")