# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def save_roofline_chart(hw_specs, model_specs, filename="roofline_analysis.png"):
#     plt.figure(figsize=(12, 8))
    
#     # x축 범위 설정 (Arithmetic Intensity)
#     x = np.logspace(-2, 4, 1000)
    
#     # 1. 하드웨어 루프라인 (Lines)
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
#     for i, hw in enumerate(hw_specs):
#         name, peak, bw = hw['name'], hw['peak_perf'], hw['bandwidth']
#         color = colors[i % len(colors)]
        
#         y = np.minimum(peak, x * bw)
#         plt.plot(x, y, label=f'HW: {name}', linewidth=2.5, color=color, alpha=0.8)
        
#         # Ridge Point 계산 및 표시
#         ridge = peak / bw
#         plt.scatter([ridge], [peak], color=color, edgecolors='black', s=60, zorder=5)

#     # 2. 모델 성능 데이터 (Points)
#     markers = ['o', 's', 'v', 'D', '*']
#     for i, model in enumerate(model_specs):
#         m_name, ai, perf = model['name'], model['ai'], model['perf']
#         plt.scatter(ai, perf, label=f'Model: {m_name}', s=150, 
#                     marker=markers[i % len(markers)], edgecolors='black', zorder=10)

#     # 그래프 디테일 설정
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('Arithmetic Intensity (OPs/Byte)', fontsize=12, fontweight='bold')
#     plt.ylabel('Performance (GOPS)', fontsize=12, fontweight='bold')
#     plt.title('Roofline Model Analysis Result', fontsize=16, pad=20)
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10) # 범례를 그래프 밖으로 이동
    
#     # 이미지 저장
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300) # 고해상도(300dpi) 저장
#     plt.close() # 메모리 해제
    
#     print(f"✅ 그래프가 '{os.getcwd()}/{filename}'에 저장되었습니다.")

# # --- 데이터 입력 및 실행 ---
# # 단위: GFLOPS, GB/s, FLOPs/Byte
# # hw_data = [
# #     {'name': 'NVIDIA H100', 'peak_perf': 67000, 'bandwidth': 3350},
# #     {'name': 'NVIDIA A100', 'peak_perf': 19500, 'bandwidth': 1935},
# #     {'name': 'T4 GPU', 'peak_perf': 8100, 'bandwidth': 320}
# # ]

# # ARIES: 80TOPS, 66.7GB/s 
# hw_data = [
#     {'name': 'MBLT Aries', 'peak_perf': 80*1000, 'bandwidth': 66.7}
# ]

# # model_data = [
# #     {'name': 'LLM Inference', 'ai': 0.15, 'perf': 450},
# #     {'name': 'ConvNet Train', 'ai': 12.0, 'perf': 15000},
# #     {'name': 'Matrix Mult', 'ai': 50.0, 'perf': 55000}
# # ]
# model_data = [
#     {'name': 'MobileNet_V2', 'ai': 129, 'perf': 7046},
#     {'name': 'ResNet-50', 'ai': 257, 'perf': 25364},
#     {'name': 'YOLO-11S', 'ai': 2220, 'perf': 18659},
#     {'name': 'YOLO-11L', 'ai': 3495, 'perf': 24141}
    
    
#     # {'name': 'MobileNetV2', 'ai': 0.6, 'perf': 6930}, 
#     # {'name': 'ResNet-50', 'ai': 1.5, 'perf': 24656},   
#     # {'name': 'YOLO-11s', 'ai': 2.1, 'perf': 21952},   
#     # {'name': 'Llama-3.2-3B', 'ai': 0.05, 'perf': 73}  
# ]

# # 실행: 파일명 지정 가능
# save_roofline_chart(hw_data, model_data, "MBLT_Aries_Roofline_2.png")




import matplotlib.pyplot as plt
import numpy as np
import os

def save_roofline_chart(hw_specs, model_specs, filename="roofline_analysis.png"):
    plt.figure(figsize=(12, 8))
    
    # x축 범위 설정 (Arithmetic Intensity)
    x = np.logspace(-2, 4, 1000)
    
    # 1. 하드웨어 루프라인 (Lines)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, hw in enumerate(hw_specs):
        name, peak, bw = hw['name'], hw['peak_perf'], hw['bandwidth']
        color = colors[i % len(colors)]
        
        y = np.minimum(peak, x * bw)
        plt.plot(x, y, label=f'HW: {name}', linewidth=2.5, color=color, alpha=0.8)
        
        # --- Ridge Point 점 찍는 부분 삭제/주석 처리 ---
        # ridge = peak / bw
        # plt.scatter([ridge], [peak], color=color, edgecolors='black', s=60, zorder=5)

    # 2. 모델 성능 데이터 (Points)
    markers = ['o', 's', 'v', 'D', '*']
    for i, model in enumerate(model_specs):
        m_name, ai, perf = model['name'], model['ai'], model['perf']
        plt.scatter(ai, perf, label=f'Model: {m_name}', s=150, 
                    marker=markers[i % len(markers)], edgecolors='black', zorder=10)

    # 그래프 디테일 설정
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (OPs/Byte)', fontsize=12, fontweight='bold')
    plt.ylabel('Performance (GOPS)', fontsize=12, fontweight='bold')
    plt.title('MBLT Aries NPU Roofline Analysis', fontsize=16, pad=20)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"✅ 그래프가 '{os.getcwd()}/{filename}'에 저장되었습니다.")

# --- 데이터 입력 ---
hw_data = [
    {'name': 'MBLT Aries', 'peak_perf': 80*1000, 'bandwidth': 66.7}
]

model_data = [
    {'name': 'MobileNet_V2', 'ai': 129, 'perf': 7046},
    {'name': 'ResNet-50', 'ai': 257, 'perf': 25364},
    {'name': 'YOLO-11S', 'ai': 2220, 'perf': 18659},
    {'name': 'YOLO-11L', 'ai': 3495, 'perf': 24141}  # 24.141에서 24141로 수정
]

# 실행
save_roofline_chart(hw_data, model_data, "MBLT_Aries_Roofline_Final.png")