
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>

void test_latency(size_t size_kb) {
    // 1. 데이터 준비 (4바이트 int 대신 8바이트 pointer-sized uint64_t 권장)
    //크기 정해놓고 그만큼 할당하는 방법: element들은 다 size_t로 하고, count = size/sizeof(size_t)
    size_t count = (size_kb * 1024) / sizeof(size_t); //64bit 자료형 써서 iteration count 계산
    if (count == 0) count = 1; // 너무 작을 경우를 대비해서...!
    std::vector<size_t> arr(count);

    // 2. 인덱스 연결 (0 -> 1 -> 2 ... -> n)
    std::iota(arr.begin(), arr.end(), 1); //std::iota(시작_지점, 끝_지점, 시작_값);해서 시작 값부터 ++하면서 하나씩 채워줌
    arr.back() = 0; // 마지막은 다시 처음으로 연결

    // 3. 셔플 (index(curr)이 순차가 되면 CPU prefetching 되니까, 이걸 방지 위해 순서를 무작위로 섞음)
    // 단순히 섞는 게 아니라 'Cycle'이 하나로 유지되도록 셔플하는 것이 정석
    std::random_device rd;
    std::mt19937 g(rd()); //random number generator
    for (size_t i = count - 1; i > 0; i--) {
        std::uniform_int_distribution<size_t> dist(0, i - 1); // 현재위치보다 작은 인덱스 범위 생성
        size_t j = dist(g); //dist중에서 하나 고름
        std::swap(arr[i], arr[j]); // 정해서 바꿈
    }

    // 4. 워밍업 (캐시에 데이터 로드)
    size_t curr = 0;
    for ( size_t i = 0; i < count * 2; i++) curr = arr[curr]; //한번씩 다 read함 ==> 캐시에 다 올려놓고 시작
    // 또는 curr_ptr = *curr_ptr;

    // 5. 실제 측정
    const size_t iterations = 100'000'000; // 1억 번 반복

    auto start = std::chrono::high_resolution_clock::now();
    
    for ( size_t i = 0; i < iterations; i++) {
        curr = arr[curr]; // 핵심: 포인터 체이싱
        //또는 curr_ptr = *curr_ptr;
    }
    
    // curr 값을 강제로 사용하게 만들어 컴파일러 최적화(루프 삭제) 방지
    if (curr == 0xdeadbeef) std::cout << "never happens";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 6. 결과 출력 (나노초 단위)
    double latency = (diff.count() * 1e9) / iterations;
    //std::cout << size_kb << " KB: " << latency << " ns" << std::endl;
    // 결과 출력: [KB단위] [지연시간]
    std::cout << std::setw(10) << size_kb << " " << std::fixed << std::setprecision(3) << latency << std::endl;
}

int main() {

    std::cout << "# Size(KB)  Latency(ns)" << std::endl;
    std::cout << "# -----------------------" << std::endl;
    // 1KB부터 256MB까지 테스트
    for (size_t size : {4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144}) {
        test_latency(size);
    }
    return 0;
}