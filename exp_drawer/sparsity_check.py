#!/usr/bin/env python3
"""
sparsity_check.py — 입력값(zeros / ones / random)에 따라 latency가 달라지는지 확인.
activation sparsity가 런타임에 적용된다면 zeros 입력이 더 빠르게 나온다.
"""

import importlib
import sys
import time
import numpy as np

MODELS = ["ResNet50", "YOLO11s", "MobileNet_V2"]
MODE   = "global8"
RUNS   = 100
WARMUP = 10

MODEL_CATALOG = {
    "MobileNet_V2": {"cls": "MobileNet_V2"},
    "ResNet50":     {"cls": "ResNet50"},
    "YOLO11s":      {"cls": "YOLO11s"},
}

def bench(eng, dummy, runs, warmup):
    for _ in range(warmup):
        eng(dummy)
    t0 = time.perf_counter()
    for _ in range(runs):
        eng(dummy)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / runs  # avg ms

mod = importlib.import_module("mblt_model_zoo.vision")

print(f"{'Model':<24} {'Input':<10} {'avg_ms':>8}  {'diff vs zeros':>13}")
print("-" * 62)

for model_name in MODELS:
    cfg        = MODEL_CATALOG[model_name]
    ModelClass = getattr(mod, cfg["cls"])
    zoo_mode   = "single" if MODE == "base" else MODE

    try:
        eng = ModelClass(infer_mode=zoo_mode)
    except Exception as e:
        print(f"{model_name:<24} LOAD FAILED: {e}")
        continue

    buf   = eng.model.model.get_input_buffer_info()[0]
    shape = (buf.original_height, buf.original_width, buf.original_channel)

    # detect working dtype
    dummy_base = None
    working_dtype = None
    for dtype in (np.uint8, np.float32, np.int8, np.float16):
        candidate = np.zeros(shape, dtype=dtype)
        try:
            eng(candidate)
            dummy_base    = candidate
            working_dtype = dtype
            break
        except Exception:
            pass

    if dummy_base is None:
        print(f"{model_name:<24} dtype detection failed")
        eng.dispose()
        continue

    inputs = {
        "zeros":  np.zeros(shape, dtype=working_dtype),
        "ones":   np.ones(shape, dtype=working_dtype),
        "random": np.random.randint(0, 256, shape).astype(working_dtype)
                  if working_dtype in (np.uint8, np.int8)
                  else np.random.rand(*shape).astype(working_dtype),
    }

    results = {}
    for label, arr in inputs.items():
        ms = bench(eng, arr, RUNS, WARMUP)
        results[label] = ms

    eng.dispose()

    base_ms = results["zeros"]
    for label, ms in results.items():
        diff = f"{(ms - base_ms) / base_ms * 100:+.1f}%" if label != "zeros" else "baseline"
        print(f"{model_name:<24} {label:<10} {ms:>8.3f}  {diff:>13}")
    print()
