#!/usr/bin/env python3
"""
npu_bench_zoo.py — Times NPU inference using the mblt_model_zoo Python API.

Loads a model from the model zoo, runs warmup + timed inference,
and prints results in the same format as npu_bench (C++), so
exp_drawer.py can parse both identically.

Usage:
    python3 npu_bench_zoo.py --model ResNet50 --mode global8 --runs 50

Output (stdout):
    avg_latency_ms       <ms>
    min_latency_ms       <ms>
    achieved_gops        <GOPS>
    arithmetic_intensity <ops/byte>

Diagnostic lines go to stderr so they don't interfere with parsing.
"""

import argparse
import importlib
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Model catalog
#
# gops : GFLOPs per inference (from model zoo README or well-known benchmarks).
#        For INT8 MACs: 1 MAC = 2 ops, so GFLOPs == GOPS.
# cls  : class name exported from mblt_model_zoo.vision
#
# gbytes is NOT stored here — it is derived at runtime from the MXQ file size
# (weights dominate DRAM traffic and the MXQ stores quantized weights).
# ---------------------------------------------------------------------------
MODEL_CATALOG = {
    # ---- Lightweight CNNs ----
    "MobileNet_V2": {"cls": "MobileNet_V2",          "gops":  0.30},
    "EfficientNet_B0": {"cls": "EfficientNet_B0",    "gops":  0.39},

    # ---- Classic CNNs ----
    "ResNet50":  {"cls": "ResNet50",                  "gops":  4.11},
    "DenseNet121": {"cls": "DenseNet121",             "gops":  2.87},
    "VGG16":     {"cls": "VGG16",                     "gops": 30.97},

    # ---- Modern CNNs ----
    "ConvNeXt_Tiny": {"cls": "ConvNeXt_Tiny",         "gops":  9.11},

    # ---- Transformers ----
    "Swin_T":    {"cls": "Swin_T",                    "gops":  4.49},
    "ViT_Base_Patch16_224": {"cls": "ViT_Base_Patch16_224", "gops": 17.58},

    # ---- Object detection ----
    "YOLO11s":   {"cls": "YOLO11s",                   "gops": 23.80},
    "YOLO11l":   {"cls": "YOLO11l",                   "gops": 93.21},

    # ---- Instance segmentation ----
    "YOLO11sSeg": {"cls": "YOLO11sSeg",               "gops": 38.18},

    # ---- Pose estimation ----
    "YOLO11lPose": {"cls": "YOLO11lPose",             "gops": 96.65},
}

WARMUP_RUNS = 5


def benchmark(model_name: str, infer_mode: str, num_runs: int) -> None:
    if model_name not in MODEL_CATALOG:
        print(
            f"Unknown model '{model_name}'. Available: {list(MODEL_CATALOG)}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg  = MODEL_CATALOG[model_name]
    gops = cfg["gops"]

    print(f"[zoo] Loading {model_name} [{infer_mode}] ...", file=sys.stderr)
    mod        = importlib.import_module("mblt_model_zoo.vision")
    ModelClass = getattr(mod, cfg["cls"])
    try:
        eng = ModelClass(infer_mode=infer_mode)
    except Exception as e:
        print(f"[zoo] Failed to load {model_name} [{infer_mode}]: {e}", file=sys.stderr)
        sys.exit(1)

    # Derive DRAM traffic from MXQ file size (weights dominate; 1 byte/param for INT8).
    mxq_filename = eng.model_cfg.get("filename", "")
    mxq_path = os.path.expanduser(
        f"~/.mblt_model_zoo/vision/aries/{infer_mode}/{mxq_filename}"
    )
    if mxq_filename and os.path.exists(mxq_path):
        gbytes = os.path.getsize(mxq_path) / 1e9
        print(f"[zoo] MXQ size: {gbytes*1e3:.1f} MB → used as DRAM traffic estimate", file=sys.stderr)
    else:
        gbytes = None
        print(f"[zoo] WARNING: MXQ not found at {mxq_path}; AI will be omitted", file=sys.stderr)

    # Build a dummy input with the dtype the model actually accepts.
    # Models differ: some expect uint8 (raw pixels), others float32/int8.
    buf   = eng.model.model.get_input_buffer_info()[0]
    shape = (buf.original_height, buf.original_width, buf.original_channel)
    dummy     = None
    precision = "unknown"
    _DTYPE_TO_PRECISION = {
        np.uint8:   "INT8",
        np.int8:    "INT8",
        np.float32: "INT8 (fp32 input)",
        np.float16: "INT8 (fp16 input)",
    }
    for dtype in (np.uint8, np.float32, np.int8, np.float16):
        candidate = np.zeros(shape, dtype=dtype)
        try:
            eng(candidate)
            dummy     = candidate
            precision = _DTYPE_TO_PRECISION.get(dtype, dtype.__name__)
            print(f"[zoo] Input dtype: {dtype.__name__} → precision: {precision}", file=sys.stderr)
            break
        except Exception:
            pass
    if dummy is None:
        print(f"[zoo] Could not find a valid input dtype for {model_name}", file=sys.stderr)
        eng.dispose()
        sys.exit(1)

    # Multi mode processes 4 inputs per call (batch=4 per cluster).
    # Pass a stacked (4, H, W, C) array to correctly exercise all 4 batch slots.
    # GOPS is counted as 4×gops per call. Other modes use batch=1.
    is_multi = (infer_mode == "multi")
    BATCH    = 4 if is_multi else 1
    infer_input = np.stack([dummy] * BATCH) if is_multi else dummy
    infer_fn    = eng.model.model.infer  # bypass MBLT_Engine wrapper for batch input

    print(f"[zoo] Batch size: {BATCH}", file=sys.stderr)
    print(f"[zoo] Warmup ({WARMUP_RUNS} runs) ...", file=sys.stderr)
    for _ in range(WARMUP_RUNS):
        infer_fn(infer_input)

    print(f"[zoo] Timing ({num_runs} runs) ...", file=sys.stderr)
    latencies_ms: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        infer_fn(infer_input)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    eng.dispose()

    avg_ms        = sum(latencies_ms) / len(latencies_ms)
    min_ms        = min(latencies_ms)
    # achieved_gops = total ops per call / call latency
    achieved_gops = (BATCH * gops) / (avg_ms / 1000.0)
    ai            = gops / gbytes if gbytes else None

    print(f"avg_latency_ms       {avg_ms:.3f}")
    print(f"min_latency_ms       {min_ms:.3f}")
    print(f"achieved_gops        {achieved_gops:.3f}")
    if ai is not None:
        print(f"arithmetic_intensity {ai:.3f}")
    print(f"precision            {precision}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NPU model zoo benchmark")
    parser.add_argument(
        "--model", required=True, choices=sorted(MODEL_CATALOG),
        help="Model name from the catalog",
    )
    parser.add_argument(
        "--mode", default="global8",
        choices=["single", "multi", "global4", "global8"],
        help="NPU inference mode",
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Number of timed inference runs",
    )
    args = parser.parse_args()
    benchmark(args.model, args.mode, args.runs)


if __name__ == "__main__":
    main()
