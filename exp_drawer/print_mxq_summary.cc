// print_mxq_summary.cc — prints getModelSummary for a given .mxq file
// Build: make -f Makefile.npu print_mxq_summary
// Usage: ./print_mxq_summary <model.mxq>

#include <iostream>
#include <cstdlib>
#include "qbruntime/qbruntime.h"
#include "qbruntime/type.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.mxq>\n", argv[0]);
        return 1;
    }

    mobilint::StatusCode sc;
    auto acc = mobilint::Accelerator::create(sc);
    if (!sc) {
        fprintf(stderr, "Failed to open NPU (status %d)\n", int(sc));
        return 1;
    }

    auto model = mobilint::Model::create(argv[1], sc);
    if (!sc) {
        fprintf(stderr, "Failed to load model (status %d)\n", int(sc));
        return 1;
    }

    sc = model->launch(*acc);
    if (!sc) {
        fprintf(stderr, "Failed to launch model (status %d)\n", int(sc));
        return 1;
    }

    std::cout << "=== Model Summary: " << argv[1] << " ===\n";
    std::cout << mobilint::getModelSummary(argv[1]) << "\n";

    std::cout << "=== Runtime Info ===\n";
    std::cout << "Core mode: ";
    switch (model->getCoreMode()) {
        case mobilint::CoreMode::Single:  std::cout << "Single\n";  break;
        case mobilint::CoreMode::Multi:   std::cout << "Multi\n";   break;
        case mobilint::CoreMode::Global4: std::cout << "Global4\n"; break;
        case mobilint::CoreMode::Global8: std::cout << "Global8\n"; break;
        default:                          std::cout << "Other\n";   break;
    }

    auto cores = model->getTargetCores();
    std::cout << "Target cores (" << cores.size() << "):\n";
    for (const auto& c : cores) {
        std::cout << "  cluster " << (static_cast<int>(c.cluster) >> 16)
                  << "  core " << static_cast<int>(c.core) << "\n";
    }

    auto buf = model->getInputBufferInfo()[0];
    std::cout << "Input shape: " << buf.original_width << "W x "
              << buf.original_height << "H x " << buf.original_channel << "C\n";
    std::cout << "Input total elements: " << buf.original_size() << "\n";

    model->dispose();
    return 0;
}
