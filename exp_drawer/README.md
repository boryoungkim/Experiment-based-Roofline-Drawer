## About MBLT multi-core usage
- Core mode is baked into the MXQ at compile time — need separate .mxq per mode
- Multi: mc.setMultiCoreMode({Cluster0, Cluster1}) — 4-batch processing per cluster, pass 4 input pointers
- Global4: mc.setGlobal4CoreMode({Cluster1}) — 4 local cores on 1 cluster process 1 input together
- Global8: mc.setGlobal8CoreMode() — all 8 local cores process 1 input together

| File | Purpose |
| :--- | :--- |
| npu_bench_multi.cc | Multi mode: passes 4 input pointers per call (batch=4 per cluster), reports 4 × gops / latency |
| npu_bench_global4.cc | Global4 mode: 1 input split across 4 local cores in Cluster1 |
| npu_bench_global8.cc | Global8 mode: 1 input split across all 8 local cores (both clusters) |

### Changes to existing files

- **compile_bench.py**: adds 6 new mxq_compile calls — compute_bench and bandwidth_bench each compiled with `inference_scheme="multi"`, `"global4"`, `"global8"`. Outputs: `*_multi.mxq`, `*_global4.mxq`, `*_global8.mxq`.
- **Makefile.npu**: added `npu_bench_multi`, `npu_bench_global4`, `npu_bench_global8` to `TARGETS`.

### Build & run

```bash
# Build all
cd exp_drawer && make -f Makefile.npu

# Run (example with compute_bench, 100 runs, 7.7 GOPS, 0.1 GB)
./npu_bench_multi    compute_bench_multi.mxq   100 7.7 0.1
./npu_bench_global4  compute_bench_global4.mxq 100 7.7 0.1
./npu_bench_global8  compute_bench_global8.mxq 100 7.7 0.1