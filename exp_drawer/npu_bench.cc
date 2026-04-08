// npu_bench.cc — Times NPU inference on a .mxq model and reports achieved GOPS.
//
// Uses Single-core mode with all 8 Local Cores and 8 threads to keep every
// core busy simultaneously. Without multithreading the SDK falls back to
// sequential (1-core) execution and throughput is severely underreported.
//
// Build (from this directory):
//   make -f Makefile.npu npu_bench
//
// Usage:
//   ./npu_bench <model.mxq> <num_runs> <model_gops> <model_gbytes>
//
//   num_runs     : inferences per thread (total = num_runs × 8 threads)
//   model_gops   : total ops for one inference in GOPS  (e.g. ResNet-50 ≈ 7.7)
//   model_gbytes : total memory traffic per inference in GB
//
// Output (stdout, parsed by exp_drawer.py):
//   avg_latency_ms       <ms>   (per-inference latency, averaged across all threads)
//   min_latency_ms       <ms>
//   achieved_gops        <GOPS> (aggregate across all 8 cores)
//   arithmetic_intensity <ops/byte>

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

#include "qbruntime/qbruntime.h"
#include "qbruntime/type.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static constexpr int N_CORES = 8;

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s <model.mxq> <num_runs> <model_gops> <model_gbytes>\n", prog);
    fprintf(stderr, "  num_runs     : inferences per thread (total = num_runs x %d threads)\n", N_CORES);
    fprintf(stderr, "  model_gops   : ops per inference in GOPS (e.g. 7.7 for ResNet-50)\n");
    fprintf(stderr, "  model_gbytes : memory traffic per inference in GB\n");
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
        fprintf(stderr, "  Hint: check driver and permissions.\n");
        return 1;
    }

    // --- Configure Single-core mode targeting all 8 Local Cores ---
    // The SDK dispatches each infer() call to the next available core.
    // Submitting from 8 threads simultaneously keeps all cores busy.
    mobilint::ModelConfig mc;
    if (!mc.setSingleCoreMode({
            {mobilint::Cluster::Cluster0, mobilint::Core::Core0},
            {mobilint::Cluster::Cluster0, mobilint::Core::Core1},
            {mobilint::Cluster::Cluster0, mobilint::Core::Core2},
            {mobilint::Cluster::Cluster0, mobilint::Core::Core3},
            {mobilint::Cluster::Cluster1, mobilint::Core::Core0},
            {mobilint::Cluster::Cluster1, mobilint::Core::Core1},
            {mobilint::Cluster::Cluster1, mobilint::Core::Core2},
            {mobilint::Cluster::Cluster1, mobilint::Core::Core3},
        })) {
        fprintf(stderr, "Failed to configure Single-core mode.\n");
        return 1;
    }

    // --- Load and launch model ---
    auto model = mobilint::Model::create(mxq_path, mc, sc);
    if (!sc) {
        fprintf(stderr, "Failed to load model '%s' (status %d).\n", mxq_path, int(sc));
        return 1;
    }
    sc = model->launch(*acc);
    if (!sc) {
        fprintf(stderr, "Failed to launch model (status %d).\n", int(sc));
        return 1;
    }

    // --- Prepare dummy inputs (one per thread; values don't affect timing) ---
    auto buf_info = model->getInputBufferInfo()[0];
    size_t input_elems = buf_info.original_size();
    // Each thread has its own buffer to avoid false sharing
    std::vector<std::vector<float>> dummy_bufs(N_CORES,
                                               std::vector<float>(input_elems, 0.0f));

    // --- Warmup: all threads run 3 inferences before timing starts ---
    {
        std::vector<std::thread> warmup_threads;
        for (int t = 0; t < N_CORES; t++) {
            warmup_threads.emplace_back([&, t]() {
                mobilint::StatusCode wsc;
                for (int i = 0; i < 3; i++)
                    model->infer({dummy_bufs[t].data()}, wsc);
            });
        }
        for (auto& th : warmup_threads) th.join();
    }

    // --- Timed runs ---
    // Each thread records per-inference latencies.
    // A shared atomic gate ensures all threads start simultaneously.
    std::vector<std::vector<double>> per_thread_latencies(N_CORES,
                                                          std::vector<double>(num_runs));
    std::atomic<bool> go{false};
    std::atomic<int>  ready_count{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < N_CORES; t++) {
        threads.emplace_back([&, t]() {
            mobilint::StatusCode tsc;
            float* buf = dummy_bufs[t].data();

            // Signal ready and spin-wait for all threads
            ready_count.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {}

            for (int i = 0; i < num_runs; i++) {
                auto t0  = Clock::now();
                model->infer({buf}, tsc);
                auto t1  = Clock::now();
                per_thread_latencies[t][i] = Ms(t1 - t0).count();
            }
        });
    }

    // Wait until all threads are ready, then release the gate
    while (ready_count.load(std::memory_order_acquire) < N_CORES) {}
    auto wall_start = Clock::now();
    go.store(true, std::memory_order_release);

    for (auto& th : threads) th.join();
    auto wall_end = Clock::now();

    model->dispose();

    // --- Aggregate stats ---
    // Flatten all per-thread latencies for avg/min
    std::vector<double> all_latencies;
    all_latencies.reserve(N_CORES * num_runs);
    for (auto& v : per_thread_latencies)
        all_latencies.insert(all_latencies.end(), v.begin(), v.end());

    double avg_ms = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0)
                    / all_latencies.size();
    double min_ms = *std::min_element(all_latencies.begin(), all_latencies.end());

    // Wall-clock throughput: total inferences / elapsed wall time
    double wall_ms        = Ms(wall_end - wall_start).count();
    int    total_infers   = N_CORES * num_runs;
    double achieved_gops  = (total_infers * gops * 1000.0) / wall_ms;
    double ai             = gops / gbytes;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "avg_latency_ms       " << avg_ms        << "\n";
    std::cout << "min_latency_ms       " << min_ms        << "\n";
    std::cout << "achieved_gops        " << achieved_gops << "\n";
    std::cout << "arithmetic_intensity " << ai            << "\n";

    return 0;
}
