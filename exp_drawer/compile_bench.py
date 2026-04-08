#!/usr/bin/env python3
"""
compile_bench.py — Compiles compute_bench.onnx and bandwidth_bench.onnx to .mxq
                   using the Mobilint qbcompiler.

Run this INSIDE the qbcompiler Docker container on the workstation:

  # 1. Start container (adjust paths)
  docker run -it --ipc=host --gpus=all \\
    -v $(pwd):/workspace \\
    mobilint/qbcompiler:1.0-cuda12.8.1-ubuntu22.04

  # 2. Inside container
  pip install <qbcompiler_wheel>.whl
  cd /workspace
  python3 create_bench_onnx.py   # generates .onnx + calib_data/
  python3 compile_bench.py

Outputs:
  compute_bench.mxq
  bandwidth_bench.mxq

Then copy both .mxq files to the NPU server and run:
  python3 exp_drawer.py
"""

import numpy as np

from qbcompiler import mxq_compile, CalibrationConfig, Uint8InputConfig

# ---------------------------------------------------------------------------
# Shared calibration config (PTQ, per-channel, maxpercentile)
# ---------------------------------------------------------------------------
calib_config = CalibrationConfig(
    quantization_method=1,
    quantization_output=1,
    quantization_mode=1,
    percentile=0.9999,
    topk_ratio=0.01,
    max_each=2048,
    max_total=2048,
    hist_percentile=0.99,
    hist_use_gpu=False,
    hist_num_bins=2048,
    hist_num_samples=1000,
    hist_buffer_size=1024,
    hist_min_bin_width=0.0001,
    hist_search_percentile_min=0.95,
    hist_search_percentile_max=0.9999,
    hist_num_search=10,
    hist_search_type=0,
    act_scale_min=0.0005,
    weight_scale_min=0.0001,
    min_clip_ratio=0.5,
    layer_overrides={},
    optional=False
)

# No image preprocessing — inputs are already float32 tensors
no_uint8 = Uint8InputConfig(apply=False, inputs=[])


# ---------------------------------------------------------------------------
# Compile compute_bench
# ---------------------------------------------------------------------------
print("=" * 50)
print("Compiling compute_bench.onnx ...")
print("=" * 50)

mxq_compile(
    model="./compute_bench.onnx",
    calib_data_path="./calib_data/compute_bench",
    save_path="./compute_bench.mxq",
    backend="onnx",
    device="gpu",
    uint8_input_config=no_uint8,
    calibration_config=calib_config,
)
print("→ compute_bench.mxq saved\n")


# ---------------------------------------------------------------------------
# Compile bandwidth_bench
# ---------------------------------------------------------------------------
print("=" * 50)
print("Compiling bandwidth_bench.onnx ...")
print("=" * 50)

mxq_compile(
    model="./bandwidth_bench.onnx",
    calib_data_path="./calib_data/bandwidth_bench",
    save_path="./bandwidth_bench.mxq",
    backend="onnx",
    device="gpu",
    uint8_input_config=no_uint8,
    calibration_config=calib_config,
)
print("→ bandwidth_bench.mxq saved\n")

print("Done. Copy compute_bench.mxq and bandwidth_bench.mxq to the NPU server.")


# ---------------------------------------------------------------------------
# Compile Multi-core variants
# inference_scheme="multi": 4 Local Cores per Cluster process 4 inputs as a
# batch. Both clusters are used at runtime (setMultiCoreMode).
# ---------------------------------------------------------------------------
for bench in ("compute_bench", "bandwidth_bench"):
    print("=" * 50)
    print(f"Compiling {bench}_multi.onnx ...")
    print("=" * 50)
    mxq_compile(
        model=f"./{bench}.onnx",
        calib_data_path=f"./calib_data/{bench}",
        save_path=f"./{bench}_multi.mxq",
        backend="onnx",
        device="gpu",
        uint8_input_config=no_uint8,
        calibration_config=calib_config,
        inference_scheme="multi",
    )
    print(f"→ {bench}_multi.mxq saved\n")


# ---------------------------------------------------------------------------
# Compile Global4-core variants
# inference_scheme="global4": 4 Local Cores within one Cluster collaborate to
# process a single input (lower latency for large models).
# ---------------------------------------------------------------------------
for bench in ("compute_bench", "bandwidth_bench"):
    print("=" * 50)
    print(f"Compiling {bench}_global4.onnx ...")
    print("=" * 50)
    mxq_compile(
        model=f"./{bench}.onnx",
        calib_data_path=f"./calib_data/{bench}",
        save_path=f"./{bench}_global4.mxq",
        backend="onnx",
        device="gpu",
        uint8_input_config=no_uint8,
        calibration_config=calib_config,
        inference_scheme="global4",
    )
    print(f"→ {bench}_global4.mxq saved\n")


# ---------------------------------------------------------------------------
# Compile Global8-core variants
# inference_scheme="global8": all 8 Local Cores across both Clusters collaborate
# on a single input (maximum parallelism, lowest latency for very large models).
# ---------------------------------------------------------------------------
for bench in ("compute_bench", "bandwidth_bench"):
    print("=" * 50)
    print(f"Compiling {bench}_global8.onnx ...")
    print("=" * 50)
    mxq_compile(
        model=f"./{bench}.onnx",
        calib_data_path=f"./calib_data/{bench}",
        save_path=f"./{bench}_global8.mxq",
        backend="onnx",
        device="gpu",
        uint8_input_config=no_uint8,
        calibration_config=calib_config,
        inference_scheme="global8",
    )
    print(f"→ {bench}_global8.mxq saved\n")


print("Done. Copy all .mxq files to the NPU server.")
