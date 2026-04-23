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

# Python interpreter that has mblt_model_zoo installed.
# If exp_drawer.py is run from the mblt venv itself, sys.executable works fine.
# Override here if needed.
_ZOO_PYTHON = "/home/brkim/mblt/bin/python3"

# ---------------------------------------------------------------------------
# CPU benchmark paths
# ---------------------------------------------------------------------------
CPU_BENCH_SRC = os.path.join(_DIR, "hw_bench.cc")
CPU_BENCH_BIN = os.path.join(_DIR, "hw_bench")

# ---------------------------------------------------------------------------
# NPU benchmark paths (build separately: make -f Makefile.npu)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NPU hardware roof values.
# ---------------------------------------------------------------------------
NPU_PEAK_GOPS_SPEC     = 80 * 1000   # 80 TOPS spec
NPU_BANDWIDTH_GBS_SPEC = 66.7        # GB/s spec

# All running modes to benchmark. Each entry:
#   name        : label for the chart
#   bench_bin   : compiled runner binary
#   compute_mxq : compute-bound synthetic model
#   bw_mxq      : bandwidth-bound synthetic model
#   color       : line color on the roofline chart
NPU_MODES = [
    {
        "name":        "single",
        "bench_bin":   os.path.join(_DIR, "npu_bench"),
        "compute_mxq": os.path.join(_DIR, "compute_bench.mxq"),
        "bw_mxq":      os.path.join(_DIR, "bandwidth_bench.mxq"),
    },
    {
        "name":        "global4",
        "bench_bin":   os.path.join(_DIR, "npu_bench_global4"),
        "compute_mxq": os.path.join(_DIR, "compute_bench_global4.mxq"),
        "bw_mxq":      os.path.join(_DIR, "bandwidth_bench_global4.mxq"),
    },
    {
        "name":        "global8",
        "bench_bin":   os.path.join(_DIR, "npu_bench_global8"),
        "compute_mxq": os.path.join(_DIR, "compute_bench_global8.mxq"),
        "bw_mxq":      os.path.join(_DIR, "bandwidth_bench_global8.mxq"),
    },
    {
        "name":        "multi",
        "bench_bin":   os.path.join(_DIR, "npu_bench_multi"),
        "compute_mxq": os.path.join(_DIR, "compute_bench_multi.mxq"),
        "bw_mxq":      os.path.join(_DIR, "bandwidth_bench_multi.mxq"),
    },
]

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

def measure_zoo_model(model_name, infer_mode, num_runs):
    """
    Runs npu_bench_zoo.py for a single (model, mode) pair.
    Returns (achieved_gops, arithmetic_intensity), or None on failure.
    """
    zoo_script = os.path.join(_DIR, "npu_bench_zoo.py")
    if not os.path.exists(zoo_script):
        print(f"  [skip] npu_bench_zoo.py not found at {zoo_script}")
        return None

    zoo_mode = infer_mode
    cmd = [_ZOO_PYTHON, zoo_script,
           "--model", model_name, "--mode", zoo_mode, "--runs", str(num_runs)]
    print(f"Running zoo benchmark: {model_name} [{infer_mode}] ...")

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stderr:
        print(r.stderr.rstrip())
    if r.returncode != 0:
        print(f"  [skip] {model_name} [{infer_mode}] failed:\n{r.stdout}{r.stderr}")
        return None

    vals      = {}
    str_vals  = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            str_vals[key] = " ".join(parts[1:])
            try:
                vals[key] = float(parts[1])
            except ValueError:
                pass

    if "achieved_gops" not in vals or "arithmetic_intensity" not in vals:
        print(f"  [skip] unexpected output: {r.stdout!r}")
        return None

    precision = str_vals.get("precision", "?")
    print(f"  avg latency : {vals.get('avg_latency_ms', 0):.2f} ms")
    print(f"  achieved    : {vals.get('achieved_gops', 0):.0f} GOPS")
    print(f"  AI (ops/B)  : {vals.get('arithmetic_intensity', 0):.1f}")
    print(f"  precision   : {precision}")
    return vals["achieved_gops"], vals["arithmetic_intensity"], precision


def measure_npu_model(mxq_path, num_runs, model_gops, model_gbytes, bench_bin=None):
    """
    Runs a npu_bench binary on a single .mxq model.
    Returns (achieved_gops, arithmetic_intensity), or None if unavailable/error.
    """
    if bench_bin is None:
        bench_bin = os.path.join(_DIR, "npu_bench")
    if not os.path.exists(bench_bin):
        print(f"  [skip] {bench_bin} not found — build with: make -f Makefile.npu")
        return None
    if not os.path.exists(mxq_path):
        print(f"  [skip] model file not found: {mxq_path}")
        return None

    cmd = [bench_bin, mxq_path, str(num_runs),
           str(model_gops), str(model_gbytes)]
    print(f"Running NPU benchmark: {os.path.basename(mxq_path)} ({os.path.basename(bench_bin)}) ...")

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [skip] {os.path.basename(bench_bin)} failed:\n{r.stdout}{r.stderr}")
        return None

    vals = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                vals[parts[0]] = float(parts[1])
            except ValueError:
                pass

    if "achieved_gops" not in vals or "arithmetic_intensity" not in vals:
        print(f"  [skip] unexpected output from {os.path.basename(bench_bin)}: {r.stdout!r}")
        return None

    print(f"  avg latency : {vals.get('avg_latency_ms', 0):.2f} ms")
    print(f"  achieved    : {vals.get('achieved_gops', 0):.0f} GOPS")
    print(f"  AI (ops/B)  : {vals.get('arithmetic_intensity', 0):.1f}")

    return vals["achieved_gops"], vals["arithmetic_intensity"]


# ===========================================================================
# Plotting
# ===========================================================================

def plot_rooflines(cpu_peak, cpu_bw, npu_models, cpu_models=None,
                   npu_measured_modes=None,
                   filename="roofline_experimental.png"):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(14, 9))
    x = np.logspace(-2, 4, 1000)

    # --- 1. CPU Roofline (주석 처리 — 차트에 표시 안 함) ---
    # y_cpu = np.minimum(cpu_peak, x * cpu_bw)
    # ridge_cpu = cpu_peak / cpu_bw
    # ax.plot(x, y_cpu, linewidth=4.0, linestyle="-", color="#1f77b4", alpha=1.0,
    #         label=f"CPU (measured)  {cpu_peak:.0f} GFLOPS | {cpu_bw:.1f} GB/s")
    # ax.scatter([ridge_cpu], [cpu_peak], color="white", edgecolors="#1f77b4", s=60, marker="o", linewidths=2, zorder=5)
    # ax.annotate(f"Ridge: {ridge_cpu:.1f} ops/B", xy=(ridge_cpu, cpu_peak),
    #             xytext=(ridge_cpu * 2.5, cpu_peak * 0.6),
    #             fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

    # --- 2. NPU Spec Roofline (dashed) ---
    y_npu_spec = np.minimum(NPU_PEAK_GOPS_SPEC, x * NPU_BANDWIDTH_GBS_SPEC)
    ridge_npu_spec = NPU_PEAK_GOPS_SPEC / NPU_BANDWIDTH_GBS_SPEC
    ax.plot(x, y_npu_spec, linewidth=1.5, linestyle="--", color="red", dashes=(2, 2), alpha=0.8)
    ax.scatter([ridge_npu_spec], [NPU_PEAK_GOPS_SPEC], color="white", edgecolors="red",
               s=60, marker="o", linewidths=2, zorder=5)
    ax.annotate(f"Ridge: {ridge_npu_spec:.1f} ops/B",
                xy=(ridge_npu_spec, NPU_PEAK_GOPS_SPEC),
                xytext=(ridge_npu_spec * 2.5, NPU_PEAK_GOPS_SPEC * 1.5),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

    # --- 3. NPU Measured Rooflines (주석 처리 — 차트에 표시 안 함) ---
    # for mode in (npu_measured_modes or []):
    #     peak  = mode["peak_gops"]
    #     bw    = mode["bw_gbs"]
    #     color = mode["color"]
    #     name  = mode["name"]
    #     y_meas = np.minimum(peak, x * bw)
    #     ridge  = peak / bw
    #     ax.plot(x, y_meas, linewidth=4.0, linestyle="-", color=color, alpha=1.0,
    #             label=f"MBLT Aries NPU ({name}, measured)  {peak:.0f} GOPS | {bw:.1f} GB/s")
    #     ax.scatter([ridge], [peak], color="white", edgecolors=color, s=60, marker="o", linewidths=2, zorder=5)
    #     ax.annotate(f"Ridge: {ridge:.1f} ops/B", xy=(ridge, peak),
    #                 xytext=(ridge * 2.5, peak * 0.6),
    #                 fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

    # --- 4. ACTUAL MODEL POINTS (모델별 색상, 모드별 마커) ---
    _tab10 = plt.get_cmap("tab10").colors
    _mode_to_marker = {
        "single":  "o",
        "global4": "s",
        "global8": "D",
        "multi":   "^",
    }
    _model_to_color = {}   # base_name -> color (tab10 순서대로 할당)
    _seen_modes     = {}   # mode_name -> marker (등장 순서 보존용)

    for m in npu_models:
        base = m.get("base_name", m.get("name", "unknown"))
        if base not in _model_to_color:
            _model_to_color[base] = _tab10[len(_model_to_color) % len(_tab10)]
        mode_name = m.get("mode_name", "")
        marker = _mode_to_marker.get(mode_name, "o")
        if mode_name and mode_name not in _seen_modes:
            _seen_modes[mode_name] = marker

        ax.scatter(m["ai"], m["perf"],
                   s=80, marker=marker,
                   color=_model_to_color[base],
                   edgecolors="#888888", linewidths=0.5, zorder=15)

    # --- Legend (roofline + model colors + mode markers) ---
    legend_handles = []

    # Spec roofline
    legend_handles.append(Line2D(
        [0], [0], linestyle="--", color="red", linewidth=1.5,
        label=f"MBLT Aries NPU (spec)  {NPU_PEAK_GOPS_SPEC:.0f} TOPS | {NPU_BANDWIDTH_GBS_SPEC:.1f} GB/s",
    ))
    legend_handles.append(Patch(color="none", label=""))  # spacer

    # Model family → color
    legend_handles.append(Patch(color="none", label="── Models (color) ──"))
    for model_name, color in _model_to_color.items():
        legend_handles.append(Patch(facecolor=color, edgecolor="#888888",
                                    linewidth=0.5, label=model_name))
    legend_handles.append(Patch(color="none", label=""))  # spacer

    # Inference mode → marker shape
    legend_handles.append(Patch(color="none", label="── Modes (shape) ──"))
    for mode_name, marker in _seen_modes.items():
        legend_handles.append(Line2D(
            [0], [0], linestyle="none", marker=marker,
            color="#444444", markerfacecolor="#aaaaaa",
            markersize=8, markeredgewidth=0.8,
            label=mode_name,
        ))

    # --- Formatting ---
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (ops/Byte)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance (GOPS)", fontsize=12, fontweight="bold")
    ax.set_title("Experimental Roofline: MBLT Aries NPU", fontsize=15, pad=20)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1),
              fontsize=9.5, frameon=True, shadow=True)

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
    # NPU models to benchmark (via model zoo Python API).
    # Add/remove models here. FLOPs and AI are defined in npu_bench_zoo.py.
    # -----------------------------------------------------------------------
    NPU_ZOO_MODELS = [
        # Lightweight CNNs
        {"name": "MobileNet_V2",          "runs": 100},
        {"name": "EfficientNet_B0",        "runs": 100},
        # Classic CNNs
        {"name": "ResNet50",               "runs":  50},
        {"name": "DenseNet121",            "runs":  50},
        {"name": "VGG16",                  "runs":  50},
        # Modern CNNs
        {"name": "ConvNeXt_Tiny",          "runs":  50},
        # Transformers
        {"name": "Swin_T",                 "runs":  50},
        {"name": "ViT_Base_Patch16_224",   "runs":  50},
        # Object detection
        {"name": "YOLO11s",                "runs":  50},
        {"name": "YOLO11l",                "runs":  30},
        # Instance segmentation
        {"name": "YOLO11sSeg",             "runs":  30},
        # Pose estimation
        {"name": "YOLO11lPose",            "runs":  30},
    ]

    print("\n" + "="*50)
    print(" 2. Measuring NPU Hardware Limits (Synthetic Benchmarks, all modes)")
    print("="*50)
    npu_measured_modes = []

    for mode in NPU_MODES:
        print(f"\n--- Mode: {mode['name']} ---")
        peak_gops = None
        bw_gbs    = None

        result = measure_npu_model(mode["compute_mxq"], 200, COMPUTE_BENCH_GOPS, COMPUTE_BENCH_GBYTES,
                                   bench_bin=mode["bench_bin"])
        if result is not None:
            peak_gops = result[0]
            print(f"[RESULT] {mode['name']} peak (measured): {peak_gops:.0f} GOPS")

        result = measure_npu_model(mode["bw_mxq"], 200, BANDWIDTH_BENCH_GOPS, BANDWIDTH_BENCH_GBYTES,
                                   bench_bin=mode["bench_bin"])
        if result is not None:
            achieved_gops_bw = result[0]
            latency_s = BANDWIDTH_BENCH_GOPS / achieved_gops_bw
            bw_gbs    = BANDWIDTH_BENCH_GBYTES / latency_s
            print(f"[RESULT] {mode['name']} bandwidth (measured): {bw_gbs:.1f} GB/s")

        if peak_gops is not None and bw_gbs is not None:
            npu_measured_modes.append({
                "name":      mode["name"],
                "peak_gops": peak_gops,
                "bw_gbs":    bw_gbs,
                "color":     mode["color"],
            })
        else:
            print(f"[WARNING] {mode['name']}: skipping roofline (incomplete measurement)")

    cpu_workloads = []

    print("\n" + "="*50)
    print(" 3. Measuring Actual NPU Models (all modes via model zoo)")
    print("="*50)
    all_npu_points = []

    for mode in NPU_MODES:
        for cfg in NPU_ZOO_MODELS:
            print(f"\n[Benchmarking] {cfg['name']} [{mode['name']}] ...")
            result = measure_zoo_model(cfg["name"], mode["name"], cfg["runs"])
            if result is not None:
                achieved_gops, ai, precision = result
                label = f"{cfg['name']} [{mode['name']}] {precision}"
                all_npu_points.append({
                    "name":      label,
                    "base_name": cfg["name"],
                    "mode_name": mode["name"],
                    "perf":      achieved_gops,
                    "ai":        ai,
                })
                print(f"  -> Added '{label}' to plot.")

    # -----------------------------------------------------------------------
    # Plot everything together
    # -----------------------------------------------------------------------
    print("\n" + "="*50)
    print(" 4. Generating Roofline Chart")
    print("="*50)
    if all_npu_points or cpu_workloads:
        plot_rooflines(cpu_peak, cpu_bw, all_npu_points, cpu_workloads,
                       npu_measured_modes=npu_measured_modes,
                       filename="roofline_experimental.png")
    else:
        print("[SKIP] No valid model data points to plot.")