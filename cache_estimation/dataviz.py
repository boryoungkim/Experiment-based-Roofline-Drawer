import matplotlib.pyplot as plt
import numpy as np
import os

# 데이터 파일 읽기
sizes = []
latencies = []

if not os.path.exists('data.txt'):
    print("Error: data.txt 파일이 없습니다. 먼저 C++ 프로그램을 실행하세요.")
    exit()

with open('data.txt', 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): 
            continue
        parts = line.split()
        if len(parts) >= 2:
            sizes.append(float(parts[0]))
            latencies.append(float(parts[1]))

# 그래프 설정
plt.figure(figsize=(12, 7))
plt.plot(sizes, latencies, marker='o', linestyle='-', linewidth=2, color='#2c3e50', markersize=8)

# X축: 메모리 크기 (Log2 스케일)
plt.xscale('log', base=2)

# X축 레이블 포맷팅 (KB, MB 단위 변환)
def format_size(size):
    if size < 1024:
        return f"{int(size)}K"
    else:
        return f"{int(size/1024)}M"

plt.xticks(sizes, [format_size(s) for s in sizes], rotation=45)

# 제목 및 레이블
plt.title('Memory Hierarchy Latency Analysis', fontsize=16, fontweight='bold')
plt.xlabel('Memory Pool Size (Log Scale)', fontsize=12)
plt.ylabel('Average Latency (ns)', fontsize=12)

# 배경 격자 및 보조선 (계단 모양을 잘 보기 위함)
plt.grid(True, which="both", ls="--", alpha=0.7)

# 하드웨어 영역 표시 (가이드라인 - 일반적인 CPU 기준)
# 실행 결과에 따라 이 선들의 위치는 달라질 수 있습니다.
plt.axhline(y=1.5, color='r', linestyle=':', alpha=0.5, label='L1 Range (~1ns)')
plt.axhline(y=15, color='g', linestyle=':', alpha=0.5, label='L2/L3 Range (~15ns)')
plt.axhline(y=80, color='b', linestyle=':', alpha=0.5, label='DRAM Range (~80ns)')
plt.legend()

# 이미지 파일로 저장 (dpi를 높이면 화질이 좋아집니다)
output_file = 'memory_latency_graph.png'
plt.tight_layout()
plt.savefig(output_file, dpi=300)

print(f"✅ 그래프가 '{output_file}'로 저장되었습니다.")