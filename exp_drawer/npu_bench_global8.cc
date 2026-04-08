// npu_bench_global8.cc — NPU benchmark using Global8-core mode.
//
// Global8 mode: all 8 Local Cores across both Clusters work together to
// process a single input, maximizing parallelism for latency-critical large
// models. Requires a model compiled with inference_scheme="global8".
//
// Build:
//   make -f Makefile.npu npu_bench_global8
//
// Usage:
//   ./npu_bench_global8 <model_global8.mxq> <num_runs> <model_gops> <model_gbytes>
//
// Output (stdout):
//   avg_latency_ms  <ms>
//   min_latency_ms  <ms>
//   achieved_gops   <GOPS>
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
    fprintf(stderr, "Usage: %s <model_global8.mxq> <num_runs> <model_gops> <model_gbytes>\n", prog);
    fprintf(stderr, "  model compiled with inference_scheme=\"global8\" is required.\n");
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

    // --- Configure Global8-core mode (all 8 local cores, no argument needed) ---
    mobilint::ModelConfig mc;
    if (!mc.setGlobal8CoreMode()) {
        fprintf(stderr, "Failed to set Global8-core mode.\n");
        return 1;
    }

    // --- Load model (must be compiled with inference_scheme="global8") ---
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

    // --- Prepare dummy input (Global8 takes 1 input, split across 8 cores) ---
    auto buf_info = model->getInputBufferInfo()[0];
    size_t input_elems = buf_info.original_size();
    std::vector<float> dummy(input_elems, 0.0f);

    // --- Warmup ---
    for (int i = 0; i < 3; i++) {
        auto out = model->infer({dummy.data()}, sc);
        if (!sc) {
            fprintf(stderr, "Warmup inference failed (status %d).\n", int(sc));
            return 1;
        }
    }

    // --- Timed runs ---
    std::vector<double> latencies(num_runs);
    for (int i = 0; i < num_runs; i++) {
        auto t0  = Clock::now();
        auto out = model->infer({dummy.data()}, sc);
        auto t1  = Clock::now();
        if (!sc) {
            fprintf(stderr, "Inference failed at run %d (status %d).\n", i, int(sc));
            return 1;
        }
        latencies[i] = Ms(t1 - t0).count();
    }

    model->dispose();

    // --- Stats ---
    double avg_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / num_runs;
    double min_ms = *std::min_element(latencies.begin(), latencies.end());
    double achieved_gops = gops * 1000.0 / avg_ms;
    double ai = gops / gbytes;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "avg_latency_ms       " << avg_ms        << "\n";
    std::cout << "min_latency_ms       " << min_ms        << "\n";
    std::cout << "achieved_gops        " << achieved_gops << "\n";
    std::cout << "arithmetic_intensity " << ai            << "\n";

    return 0;
}
