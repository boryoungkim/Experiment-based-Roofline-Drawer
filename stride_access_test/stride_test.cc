#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <mobilint.h> // SDK 헤더 (가정)

using namespace mobilint;

void run_memory_benchmark(const std::string& mxq_path) {
    // 1. 모델 요약 정보 확인 (디버깅용)
    std::cout << "Model Summary:\n" << getModelSummary(mxq_path) << std::endl;

    // 2. 내부 DMA 및 하드웨어 동작을 기록하기 위한 Tracing 시작 (매우 중요!)
    // 이 파일(.json)을 ui.perfetto.dev 에 올리면 NPU 내부의 뱅크 충돌이나 지연을 볼 수 있습니다.
    if (!startTracingEvents("npu_stride_test_trace.json")) {
        std::cerr << "Failed to start tracing!" << std::endl;
    }

    // 3. 모델 및 변형(Variant) 로드 (문서에 기반한 가상 호출)
    Model model(mxq_path);
    // (이 부분은 API 문서에 생략되어 있으나 통상적인 방법으로 Variant 0을 가져온다고 가정)
    // ModelVariantHandle variant = model.getVariant(0); 
    auto variant = /* variant 객체 획득 */;

    // 4. NPU 내부 DRAM(LPDDR)에 I/O 버퍼 할당 (acquireInputBuffer)
    std::vector<Buffer> input_bufs = variant.acquireInputBuffer();
    std::vector<Buffer> output_bufs = variant.acquireOutputBuffer();

    // CPU 쪽 Host 메모리 준비 (예: Dummy 데이터)
    std::vector<float> host_data(1024 * 1024, 1.0f); 
    std::vector<float*> host_input_ptrs = { host_data.data() };

    // ==========================================================
    // 실험 A: Host(CPU) -> NPU DRAM 복사 시간 측정 (PCIe 대역폭)
    // ==========================================================
    auto t1 = std::chrono::high_resolution_clock::now();
    
    StatusCode sc_in = variant.repositionInputs(host_input_ptrs, input_bufs);
    if (!sc_in) { // operator! 를 이용한 에러 체크
        std::cerr << "Error: " << statusCodeToString(sc_in) << std::endl;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pcie_time = t2 - t1;
    std::cout << "[PCIe] Host to NPU DRAM copy: " << pcie_time.count() << " ms\n";


    // ==========================================================
    // 실험 B: NPU 내부 실행 (NPU DRAM <-> SPM <-> 연산기)
    // ==========================================================
    auto t3 = std::chrono::high_resolution_clock::now();
    
    // 이 함수 안에서 컴파일러가 짜놓은 DMA 명령들이 수행됩니다.
    // (문서에 inferBuffer가 언급되어 있으므로 이를 사용)
    StatusCode sc_infer = model.inferBuffer(input_bufs, output_bufs);
    if (!sc_infer) {
        std::cerr << "Inference Error: " << statusCodeToString(sc_infer) << std::endl;
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> npu_time = t4 - t3;
    std::cout << "[NPU] Execution (DMA + PE): " << npu_time.count() << " ms\n";

    // 5. 할당 해제 및 Tracing 종료
    variant.releaseBuffer(input_bufs);
    variant.releaseBuffer(output_bufs);
    stopTracingEvents(); // trace.json 파일이 실제로 디스크에 써집니다.
    
    std::cout << "Done. Please check 'npu_stride_test_trace.json' at https://ui.perfetto.dev/\n";
}