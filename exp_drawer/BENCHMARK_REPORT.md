# MBLT Aries NPU Experimental Roofline — Benchmark Report

이 문서는 MBLT Aries NPU의 실험적 루프라인 차트(`roofline_experimental_9.png`) 생성 과정,
측정 설정, 선택 옵션, 검증 실험을 정리한 것이다.

---

## 1. 측정 목적

하드웨어 스펙 시트(80 TOPS, 66.7 GB/s)는 이상적인 피크값이다.
실제 모델을 NPU에서 돌렸을 때 달성 가능한 성능이 어느 위치에 놓이는지를
루프라인 모델 위에 직접 찍어 확인하기 위한 측정이다.

---

## 2. 하드웨어 및 소프트웨어 환경

| 항목 | 값 |
|------|----|
| NPU | MBLT Aries (2 Cluster × 4 Local Core = 8 cores) |
| Peak performance (spec) | 80 TOPS (INT8) |
| DRAM bandwidth (spec) | 66.7 GB/s |
| SDK | Mobilint qbruntime (C++ API) / mblt_model_zoo (Python API) |
| Model format | MXQ (Mobilint 전용 컴파일 포맷, INT8 정적 양자화) |
| Python 환경 | `/home/brkim/mblt/bin/python3` (mblt_model_zoo 설치됨) |

---

## 3. 측정 방법

### 3-1. 실행 방식 — Python API (mblt_model_zoo)

`npu_bench_zoo.py`가 개별 모델 하나를 받아 벤치마크하고 결과를 stdout에 출력한다.
`exp_drawer.py`가 이를 subprocess로 호출해 파싱한다.

```
exp_drawer.py
  └─ measure_zoo_model(model_name, mode, runs)
       └─ subprocess: python3 npu_bench_zoo.py --model X --mode Y --runs N
            ├─ ModelClass(infer_mode=mode)   # HF에서 MXQ 자동 다운로드 (캐시됨)
            ├─ warmup: 5회
            └─ timing: N회 → avg latency → achieved GOPS
```

### 3-2. 타이밍 측정 단위

`time.perf_counter()` 로 감싼 단일 `eng(dummy)` 호출.
전처리(resize, normalize)·후처리(NMS 등)는 포함되지 않는다.
NPU raw inference latency만 측정한다.

### 3-3. GOPS 계산

```
achieved_gops = model_gops / (avg_latency_ms / 1000)
```

`model_gops`는 각 모델의 FLOPs 값(GFLOP = GOPS, INT8 기준 1 MAC = 2 ops).
출처: mblt_model_zoo README 또는 torchvision/ultralytics 공식 문서.

### 3-4. Arithmetic Intensity (AI) 계산

```
ai = model_gops / mxq_file_size_gb
```

MXQ 파일 크기를 DRAM 트래픽 추정치로 사용한다.
INT8 양자화 모델에서 가중치가 1 byte/parameter이고 가중치가 DRAM 트래픽을 지배하므로
합리적인 근사다.
이전 버전에서는 spec_drawer의 AI 값을 역산해 gbytes를 하드코딩했으나,
이 방식으로 대체했다.

---

## 4. 입력 데이터

모든 측정에 **더미 입력(zeros)**을 사용했다.

```python
dummy = np.zeros(shape, dtype=working_dtype)
```

`working_dtype`은 각 모델이 실제로 받아들이는 dtype으로 자동 탐지한다
(uint8 → float32 → int8 → float16 순서로 시도, 첫 성공 dtype 사용).

### 더미 입력 유효성 검증 — Activation Sparsity 실험

더미 입력(전부 0)이 activation sparsity acceleration을 인위적으로 유발해
latency를 과소평가할 가능성을 점검했다.

**실험 설정:** ResNet50, YOLO11s를 global8 모드에서 zeros / ones / random 입력으로 각각 100회 측정.

| 모델 | zeros (baseline) | ones | random |
|------|-----------------|------|--------|
| ResNet50 | 1.896 ms | 1.799 ms (−5.1%) | 1.856 ms (−2.1%) |
| YOLO11s | 4.608 ms | 4.525 ms (−1.8%) | 4.387 ms (−4.8%) |

**결론:**
- zeros가 가장 빠르지 않다 → activation sparsity 가속이 적용되지 않고 있음
- 입력 간 차이는 최대 5% 수준으로 run-to-run 측정 노이즈 범위 내
- 더미 입력 기반 벤치마크 결과는 신뢰할 수 있다

---

## 5. Inference Mode

Aries NPU는 동일한 모델을 4가지 core 운용 방식으로 실행할 수 있다.
모드는 MXQ 컴파일 타임에 결정된다 (런타임 전환 불가).

| 모드 | 설명 | 차트 색상 |
|------|------|-----------|
| `single` (= base) | 8 Local Core를 독립 단일 코어로 운용, 8개 infer 병렬 | 빨강 |
| `multi` | Cluster 단위 Multi-core, 4 input batch per cluster | 갈색 |
| `global4` | Cluster1의 4 Local Core가 협력해 1개 input 처리 | 주황 |
| `global8` | 8 Local Core 전체가 협력해 1개 input 처리 | 보라 |

> `single` 모드는 zoo API에서 `infer_mode="single"`, exp_drawer에서 내부적으로 "base"로 표기.

---

## 6. 측정 모델

총 12개 모델, 카테고리별 대표 1개 선정. 모든 모델은 INT8 정적 양자화.

| 카테고리 | 모델 | FLOPs (GOPS) | MXQ 크기 (global8) | 가용 모드 |
|----------|------|-------------|-------------------|-----------|
| Lightweight CNN | MobileNet_V2 | 0.30 | 4.4 MB | single, multi |
| Lightweight CNN | EfficientNet_B0 | 0.39 | — | 없음 (HF 404) |
| Classic CNN | ResNet50 | 4.11 | 26.1 MB | 4개 |
| Classic CNN | DenseNet121 | 2.87 | 14.1 MB | 4개 |
| Classic CNN | VGG16 | 30.97 | 138.7 MB | 4개 |
| Modern CNN | ConvNeXt_Tiny | 9.11 | 30.6 MB | 4개 |
| Transformer | Swin_T | 4.49 | — | 없음 (HF 404) |
| Transformer | ViT_Base_Patch16_224 | 17.58 | 92.8 MB | 4개 |
| Object detection | YOLO11s | 23.80 | 10.7 MB | 4개 |
| Object detection | YOLO11l | 93.21 | 27.4 MB | 4개 |
| Instance seg. | YOLO11sSeg | 38.18 | 12.4 MB | 4개 |
| Pose estimation | YOLO11lPose | 96.65 | 30.0 MB | 4개 |

- **EfficientNet_B0, Swin_T**: HuggingFace 저장소에 single/multi/global4/global8 MXQ가 없어 자동 스킵됨.
- **MobileNet_V2**: single, multi 모드만 존재.

---

## 7. 측정 결과 요약

| 모델 | 모드 | avg latency | achieved GOPS | AI (ops/B) |
|------|------|------------|--------------|-----------|
| MobileNet_V2 | base | 1.73 ms | 173 | 68 |
| MobileNet_V2 | multi | 1.33 ms | 225 | 68 |
| ResNet50 | base | 2.65 ms | 1551 | 158 |
| ResNet50 | global4 | 2.22 ms | 1855 | 158 |
| ResNet50 | global8 | 2.04 ms | 2016 | 158 |
| ResNet50 | multi | 4.02 ms | 1023 | 157 |
| DenseNet121 | base | 3.82 ms | 752 | 208 |
| DenseNet121 | global4 | 2.15 ms | 1336 | 206 |
| DenseNet121 | global8 | 2.18 ms | 1318 | 204 |
| DenseNet121 | multi | 4.48 ms | 641 | 202 |
| VGG16 | base | 7.90 ms | 3922 | 224 |
| VGG16 | global4 | 6.14 ms | 5043 | 223 |
| VGG16 | global8 | 5.65 ms | 5478 | 223 |
| VGG16 | multi | 12.98 ms | 2386 | 223 |
| ConvNeXt_Tiny | base | 8.88 ms | 1026 | 303 |
| ConvNeXt_Tiny | global4 | 5.03 ms | 1810 | 300 |
| ConvNeXt_Tiny | global8 | 4.49 ms | 2030 | 298 |
| ConvNeXt_Tiny | multi | 10.64 ms | 856 | 296 |
| ViT_Base_Patch16_224 | base | 25.44 ms | 691 | 191 |
| ViT_Base_Patch16_224 | global4 | 18.82 ms | 934 | 190 |
| ViT_Base_Patch16_224 | global8 | 12.46 ms | 1411 | 189 |
| ViT_Base_Patch16_224 | multi | 35.77 ms | 491 | 188 |
| YOLO11s | base | 11.16 ms | 2133 | 2320 |
| YOLO11s | global4 | 5.44 ms | 4377 | 2268 |
| YOLO11s | global8 | 5.75 ms | 4138 | 2230 |
| YOLO11s | multi | 12.23 ms | 1946 | 2234 |
| YOLO11l | base | 23.42 ms | 3980 | 3510 |
| YOLO11l | global4 | 11.34 ms | 8218 | 3448 |
| YOLO11l | global8 | 9.18 ms | 10158 | 3402 |
| YOLO11l | multi | 28.62 ms | 3256 | 3412 |
| YOLO11sSeg | base | 13.61 ms | 2805 | 3213 |
| YOLO11sSeg | global4 | 9.30 ms | 4106 | 3143 |
| YOLO11sSeg | global8 | 6.67 ms | 5729 | 3090 |
| YOLO11sSeg | multi | 15.64 ms | 2441 | 3099 |
| YOLO11lPose | base | 24.15 ms | 4001 | 3317 |
| YOLO11lPose | global4 | 11.65 ms | 8295 | 3263 |
| YOLO11lPose | global8 | 9.33 ms | 10359 | 3221 |
| YOLO11lPose | multi | 28.71 ms | 3366 | 3230 |

---

## 8. 루프라인 차트 설정

- **X축**: Arithmetic Intensity (ops/Byte), 로그 스케일
- **Y축**: Performance (GOPS), 로그 스케일
- **루프라인**: NPU spec만 표시 (80 TOPS, 66.7 GB/s, 점선)
- **데이터 포인트**: 모델별 마커 모양 × 모드별 색상
- **출력 파일**: `roofline_experimental_9.png` (300 DPI)

CPU 루프라인과 합성 벤치마크 기반 measured roofline은 코드에 존재하나 현재 주석 처리됨.

---

## 9. 재현 방법

```bash
# 전체 벤치마크 재실행 (약 5~10분)
cd exp_drawer
/home/brkim/mblt/bin/python3 exp_drawer.py

# 벤치마크 없이 차트만 재생성 (이전 결과 데이터 하드코딩)
/home/brkim/mblt/bin/python3 replay_plot.py

# 단일 모델 벤치마크
/home/brkim/mblt/bin/python3 npu_bench_zoo.py --model ResNet50 --mode global8 --runs 50

# Activation sparsity 검증 재실행
/home/brkim/mblt/bin/python3 sparsity_check.py
```

### MXQ 캐시 위치

```
~/.mblt_model_zoo/vision/aries/{mode}/{filename}.mxq
```

최초 실행 시 HuggingFace에서 자동 다운로드된다. 이후에는 캐시에서 로드.

---

## 10. 관련 파일

| 파일 | 역할 |
|------|------|
| `exp_drawer.py` | 메인 스크립트: CPU + NPU 벤치마크 실행 및 차트 생성 |
| `npu_bench_zoo.py` | 단일 모델 NPU 벤치마크 (Python API, subprocess로 호출) |
| `replay_plot.py` | 이전 측정 결과로 차트만 재생성 |
| `sparsity_check.py` | 입력값에 따른 latency 변화 검증 |
| `npu_bench.cc` | 단일 모델 NPU 벤치마크 (C++ API, single 모드) |
| `npu_bench_global{4,8}.cc` | C++ 벤치마크 global4/global8 모드 버전 |
| `compute_bench*.mxq` | 합성 compute-bound 모델 (peak TOPS 측정용) |
| `bandwidth_bench*.mxq` | 합성 bandwidth-bound 모델 (peak BW 측정용) |
