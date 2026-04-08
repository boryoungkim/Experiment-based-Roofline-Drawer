# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a hardware performance analysis toolkit for drawing Roofline models. It has two approaches:
- **Spec-based**: Plot rooflines from known hardware specs (peak TFLOPS, bandwidth)
- **Experiment-based**: Measure actual hardware characteristics via micro-benchmarks

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `spec_drawer/` | Python script to generate roofline charts from hardware specs and model data |
| `cache_estimation/` | C++ micro-benchmark to measure memory hierarchy latency (L1/L2/L3/DRAM) |
| `stride_access_test/` | C++ benchmark for stride access patterns targeting NPU (uses mobilint SDK) |

## Build & Run Commands

### Cache Estimation Benchmark (CPU)
```bash
# Compile
g++ -O3 cache_estimation/cache_bench.cc -o cache_estimation/cache_bench

# Run and capture data
./cache_estimation/cache_bench > cache_estimation/data.txt

# Visualize results
cd cache_estimation && python3 dataviz.py
```
> `-O3` is required — without it, loop overhead distorts latency measurements.

### Spec-based Roofline Chart
```bash
cd spec_drawer && python3 spec_drawer.py
# Outputs a PNG file in the current directory
```

## Key Design Concepts

### Cache Benchmark (`cache_estimation/cache_bench.cc`)
Uses **pointer chasing** to defeat CPU prefetching: builds a linked list via index array, shuffles it to break sequential access patterns, then measures the latency of random memory traversal. The shuffle maintains a single cycle so every element is visited. The working set size determines which cache level is exercised.

### Spec Roofline Chart (`spec_drawer/spec_drawer.py`)
`save_roofline_chart(hw_specs, model_specs, filename)` takes:
- `hw_specs`: list of `{name, peak_perf (GOPS), bandwidth (GB/s)}`
- `model_specs`: list of `{name, ai (OPs/Byte), perf (GOPS)}`

Plots `y = min(peak_perf, ai × bandwidth)` on a log-log scale. Ridge point = `peak_perf / bandwidth`.

### Stride Access Test (`stride_access_test/stride_test.cc`)
Targets NPU (Mobilint Aries) via `mobilint` SDK. Runs inference and measures PCIe transfer time (Host→NPU DRAM) and NPU execution time separately. Generates a Perfetto trace (`npu_stride_test_trace.json`) viewable at `https://ui.perfetto.dev/` for diagnosing bank conflicts and DMA stalls.
