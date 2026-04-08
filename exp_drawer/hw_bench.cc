// hw_bench.cc — Measures peak FLOP/s and memory bandwidth experimentally.
//
// Compile:  g++ -O3 -march=native -o hw_bench hw_bench.cc
// Run:      ./hw_bench
// Output:   two lines on stdout, parsed by exp_drawer.py
//             peak_gflops  <value>
//             bandwidth_gbs <value>

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>

using Clock    = std::chrono::high_resolution_clock;
using Seconds  = std::chrono::duration<double>;

// ---------------------------------------------------------------------------
// Peak FLOP/s
// ---------------------------------------------------------------------------
// Uses 32 independent accumulators so the compiler can issue many concurrent
// FMA instructions (hiding FMA latency).  With -O3 -march=native the inner
// loop auto-vectorizes into SIMD FMA: each a[j] = a[j]*mul + add is 2 FLOPs.
// ---------------------------------------------------------------------------
double measure_peak_gflops(double target_sec = 3.0) {
    const int N = 32;
    std::vector<double> a(N);
    for (int i = 0; i < N; i++) a[i] = 1.0 + i * 0.01;

    const double mul = 1.0 + 1e-8;
    const double add = 1e-9;

    long long iters = 0;
    auto t0 = Clock::now();

    while (Seconds(Clock::now() - t0).count() < target_sec) {
        for (int inner = 0; inner < 1000; inner++) {
            for (int j = 0; j < N; j++) {
                a[j] = a[j] * mul + add;   // 2 FLOPs: mul + add (FMA on modern ISAs)
            }
        }
        iters += 1000;
    }

    auto t1 = Clock::now();
    // Prevent dead-code elimination
    volatile double sink = 0.0;
    for (int j = 0; j < N; j++) sink += a[j];
    (void)sink;

    double elapsed     = Seconds(t1 - t0).count();
    double total_flops = static_cast<double>(iters) * N * 2;
    return total_flops / elapsed / 1e9;
}

// ---------------------------------------------------------------------------
// Memory bandwidth  (STREAM Triad: c[i] = a[i] + scalar*b[i])
// ---------------------------------------------------------------------------
// Array size is chosen to far exceed the LLC so every access goes to DRAM.
// Triad touches 3 arrays → 3 * sizeof(double) bytes per element per pass.
// ---------------------------------------------------------------------------
double measure_bandwidth_gbs(double target_sec = 3.0) {
    // 32 M doubles × 8 bytes = 256 MB per array, 768 MB total
    const size_t N = 32ULL * 1024 * 1024;
    std::vector<double> a(N, 1.0), b(N, 2.0), c(N, 0.0);
    const double scalar = 3.0;

    // Warmup: bring arrays to steady-state cache pressure
    for (size_t i = 0; i < N; i++) c[i] = a[i] + scalar * b[i];

    long long passes = 0;
    auto t0 = Clock::now();

    while (Seconds(Clock::now() - t0).count() < target_sec) {
        for (size_t i = 0; i < N; i++) c[i] = a[i] + scalar * b[i];
        passes++;
    }

    auto t1 = Clock::now();
    volatile double sink = c[0]; (void)sink;

    double elapsed = Seconds(t1 - t0).count();
    // 2 reads (a, b) + 1 write (c) = 3 × 8 bytes per element
    double bytes = static_cast<double>(passes) * N * 3 * sizeof(double);
    return bytes / elapsed / 1e9;
}

// ---------------------------------------------------------------------------
int main() {
    std::cerr << "[1/2] Measuring peak FLOP/s (3s) ..." << std::endl;
    double gflops = measure_peak_gflops(3.0);

    std::cerr << "[2/2] Measuring memory bandwidth (3s) ..." << std::endl;
    double gbs = measure_bandwidth_gbs(3.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "peak_gflops "   << gflops << "\n";
    std::cout << "bandwidth_gbs " << gbs    << "\n";

    return 0;
}
