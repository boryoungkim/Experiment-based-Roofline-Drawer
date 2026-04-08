// npu_bench_multi.cc — NPU benchmark using Multi-core mode.
//
// Multi mode: each Cluster's 4 Local Cores cooperate to process 4 inputs
// simultaneously (batch=4 per cluster). Requires a model compiled with
// inference_scheme="multi".
//
// Build:
//   make -f Makefile.npu npu_bench_multi
//
// Usage:
//   ./npu_bench_multi <model_multi.mxq> <num_runs> <model_gops> <model_gbytes>
//
//   model_gops   : ops for ONE inference item in GOPS (e.g. ResNet-50 ≈ 7.7)
//   model_gbytes : memory traffic for ONE inference item in GB
//
// Output (stdout):
//   avg_latency_ms  <ms>    (latency for one 4-item batch)
//   min_latency_ms  <ms>
//   achieved_gops   <GOPS>  (4 items × gops / latency)
//   arithmetic_intensity <ops/byte>

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <string>

#include "qbruntime/qbruntime.h"
#include "qbruntime/type.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s <model_multi.mxq> <num_runs> <model_gops> <model_gbytes>\n", prog);
    fprintf(stderr, "  model compiled with inference_scheme=\"multi\" is required.\n");
}

int main(int argc, char* argv[]) {
    if (argc != 5) { usage(argv[0]); return 1; }

    const char* mxq_path = argv[1];
    int         num_runs = std::atoi(argv[2]);
    double      gops     = std::atof(argv[3]);
    double      gbytes   = std::atof(argv[4]);

    if (num_runs <= 0 || gops <= 0.0 || gbytes <= 0.0) {
        fprintf(stderr, "Error: num_runs, gops, and gbytes must be positive.\n");
        usage(argv[0]);
        return 1;
    }

    mobilint::StatusCode sc;

    // --- Open NPU ---
    auto acc = mobilint::Accelerator::create(sc);
    if (!sc) {
        fprintf(stderr, "Failed to open NPU device (status %d).\n", int(sc));
        return 1;
    }

    // --- Configure Multi-core mode (both clusters) ---
    mobilint::ModelConfig mc;
    if (!mc.setMultiCoreMode({
            mobilint::Cluster::Cluster0,
            mobilint::Cluster::Cluster1,
        })) {
        fprintf(stderr, "Failed to set Multi-core mode.\n");
        return 1;
    }

    // --- Load model (must be compiled with inference_scheme="multi") ---
    auto model = mobilint::Model::create(mxq_path, mc, sc);
    if (!sc) {
        fprintf(stderr, "Failed to load model '%s' (status %d).\n", mxq_path, int(sc));
        return 1;
    }

    // --- Deploy on NPU ---
    sc = model->launch(*acc);
    if (!sc) {
        fprintf(stderr, "Failed to launch model (status %d).\n", int(sc));
        return 1;
    }

    // --- Prepare 4 dummy inputs (Multi mode processes 4 items per batch) ---
    auto buf_info = model->getInputBufferInfo()[0];
    size_t input_elems = buf_info.original_size();
    std::vector<float> dummy(input_elems, 0.0f);
    // Pass the same dummy buffer 4 times (one per local core in the cluster)
    std::vector<const float*> inputs = {
        dummy.data(), dummy.data(), dummy.data(), dummy.data()
    };

    // --- Warmup ---
    for (int i = 0; i < 3; i++) {
        auto out = model->infer({dummy.data(), dummy.data(), dummy.data(), dummy.data()}, sc);
        if (!sc) {
            fprintf(stderr, "Warmup inference failed (status %d).\n", int(sc));
            return 1;
        }
    }

    // --- Timed runs ---
    std::vector<double> latencies(num_runs);
    for (int i = 0; i < num_runs; i++) {
        auto t0  = Clock::now();
        auto out = model->infer({dummy.data(), dummy.data(), dummy.data(), dummy.data()}, sc);
        auto t1  = Clock::now();
        if (!sc) {
            fprintf(stderr, "Inference failed at run %d (status %d).\n", i, int(sc));
            return 1;
        }
        latencies[i] = Ms(t1 - t0).count();
    }

    model->dispose();

    // --- Stats ---
    // Multi mode processes 4 items per batch, so effective throughput is 4x
    constexpr int BATCH = 4;
    double avg_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / num_runs;
    double min_ms = *std::min_element(latencies.begin(), latencies.end());
    double achieved_gops = (BATCH * gops * 1000.0) / avg_ms;
    double ai = gops / gbytes;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "avg_latency_ms       " << avg_ms        << "\n";
    std::cout << "min_latency_ms       " << min_ms        << "\n";
    std::cout << "achieved_gops        " << achieved_gops << "\n";
    std::cout << "arithmetic_intensity " << ai            << "\n";

    return 0;
}
