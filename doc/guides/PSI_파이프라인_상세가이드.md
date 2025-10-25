# 📘 PSI 파이프라인 상세 가이드

**위상천이간섭법(Phase Shifting Interferometry) 기반 3D 높이 재구성 완전 가이드**

이 문서는 `run_psi_pipeline.py`의 모든 기능을 상세히 설명합니다.

---

## 📑 목차

1. [개요](#1-개요)
2. [파이프라인 구조](#2-파이프라인-구조)
3. [단계별 상세 설명](#3-단계별-상세-설명)
   - [3.1 데이터 생성 (generate)](#31-데이터-생성-generate)
   - [3.2 UNet 사전학습 (pretrain)](#32-unet-사전학습-pretrain)
   - [3.3 PINN 미세조정 (finetune)](#33-pinn-미세조정-finetune)
   - [3.4 추론 (inference)](#34-추론-inference)
   - [3.5 테스트 (test)](#35-테스트-test)
   - [3.6 전체 실행 (all)](#36-전체-실행-all)
4. [옵션 상세 설명](#4-옵션-상세-설명)
5. [출력 파일 구조](#5-출력-파일-구조)
6. [고급 사용법](#6-고급-사용법)
7. [트러블슈팅](#7-트러블슈팅)
8. [FAQ](#8-faq)

---

## 1. 개요

### 1.1 PSI란?

**위상천이간섭법(Phase Shifting Interferometry)**은 레이저 간섭 패턴을 분석하여 나노미터 단위의 표면 높이를 측정하는 기술입니다.

**원리**:
```
레이저 간섭 → Bucket 이미지 생성 → 위상 복원 → 높이 계산
```

**이 파이프라인의 목표**:
- 입력: 16개의 Bucket 이미지 (4 파장 × 4 위상 시프트)
- 출력: 2D 표면 높이 맵

### 1.2 파이프라인 철학

**Physics-Informed Neural Network (PINN)** 접근:
- 물리 모델을 손실 함수에 통합
- 데이터 효율성 향상
- 물리적으로 타당한 결과 보장

**2단계 접근**:
1. **UNet 사전학습**: 빠른 수렴, 전역적 특징 학습
2. **PINN 미세조정** (선택적): 물리 제약 강화, 정밀도 향상

---

## 2. 파이프라인 구조

### 2.1 전체 흐름도

```
┌─────────────────┐
│  1. generate    │  합성 데이터 생성
│  (데이터 생성)   │  ├─ 훈련 데이터 (20개 기본)
│                 │  └─ 테스트 데이터 (5개 기본)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. pretrain    │  UNet 학습
│  (사전학습)      │  ├─ UNetPhysicsLoss 사용
│                 │  ├─ Adam 옵티마이저
│                 │  └─ 체크포인트 저장
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. finetune    │  PINN 미세조정 (선택적)
│  (미세조정)      │  └─ 고급 사용자용
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. inference   │  추론
│  (추론)         │  ├─ 테스트 데이터 예측
│                 │  └─ 시각화 저장
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. test        │  평가
│  (테스트)       │  ├─ RMSE, MAE, MAPE 계산
│                 │  └─ 메트릭 시각화
└─────────────────┘
```

### 2.2 디렉토리 구조

```
outputs/
├─ synthetic_data/          # 합성 데이터
│   ├─ train/
│   │   ├─ sample_0000/
│   │   │   ├─ bucket_00.bmp ~ bucket_15.bmp  # 16채널 입력
│   │   │   └─ ground_truth.npy                # 정답 높이
│   │   └─ sample_0001/
│   │       └─ ...
│   └─ test/
│       └─ ...
├─ models/                  # 학습된 모델
│   ├─ unet_best.pth       # 최고 성능 모델
│   └─ unet_final.pth      # 최종 에포크 모델
├─ results/                 # 결과
│   ├─ metrics.json        # 평가 메트릭
│   ├─ inference_results.json  # 추론 결과
│   ├─ unet_history.json   # 학습 이력
│   ├─ unet_training_curve.png  # 학습 곡선
│   ├─ metrics_distribution.png # 메트릭 분포
│   └─ visualizations/
│       ├─ inference_0000.png   # 예측 vs 실제
│       └─ ...
└─ logs/                    # 로그 (향후 추가)
```

---

## 3. 단계별 상세 설명

### 3.1 데이터 생성 (generate)

#### 목적
합성 bucket 이미지와 ground truth 높이 맵을 생성합니다.

#### 실행
```bash
python examples/run_psi_pipeline.py generate
```

#### 주요 옵션
```bash
--num-train-samples 20    # 훈련 샘플 수
--num-test-samples 5      # 테스트 샘플 수
--image-size 256          # 이미지 크기 (256x256)
```

#### 동작 과정

1. **높이 맵 생성**:
   ```python
   # 랜덤 가우시안 범프로 표면 생성
   height_map = generate_random_surface(size=(256, 256))
   ```

2. **물리 시뮬레이션**:
   ```python
   # 각 파장, 각 위상 시프트에 대해
   for wavelength in [5.0, 5.5, 6.05, 6.655]:  # μm
       for delta in [0, π/2, π, 3π/2]:
           phase = (4π / wavelength) * height
           bucket = A + B * cos(phase + delta)
   ```

3. **파일 저장**:
   - Ground truth: `.npy` 형식 (float32, 높은 정밀도)
   - Bucket 이미지: `.bmp` 형식 (uint8, 0-255)

#### 출력 예제
```
outputs/synthetic_data/train/sample_0000/
├─ bucket_00.bmp  # 파장 1, 위상 0
├─ bucket_01.bmp  # 파장 1, 위상 π/2
├─ bucket_02.bmp  # 파장 1, 위상 π
├─ bucket_03.bmp  # 파장 1, 위상 3π/2
├─ bucket_04.bmp  # 파장 2, 위상 0
...
├─ bucket_15.bmp  # 파장 4, 위상 3π/2
└─ ground_truth.npy
```

#### 시드 고정
- 훈련 데이터: `seed = 42 + sample_idx`
- 테스트 데이터: `seed = 10000 + sample_idx`
- 재현 가능성 보장

---

### 3.2 UNet 사전학습 (pretrain)

#### 목적
Bucket 이미지로부터 높이 맵을 재구성하는 UNet 모델을 학습합니다.

#### 실행
```bash
python examples/run_psi_pipeline.py pretrain
```

#### 주요 옵션
```bash
--epochs 50               # 학습 에포크 수
--batch-size 4            # 배치 크기
--learning-rate 1e-3      # 학습률
--patch-size 256          # 패치 크기
--smoothness-weight 1e-4  # 평활도 가중치
--device cuda             # GPU/CPU
```

#### 모델 구조

**UNet**:
```
입력: (N, 16, 256, 256)   # 16채널 bucket 이미지
  ↓
[Encoder]
  Conv1: 16 → 64
  Conv2: 64 → 128
  Conv3: 128 → 256
  Conv4: 256 → 512
  ↓
[Bottleneck]
  Conv5: 512 → 1024
  ↓
[Decoder]
  UpConv4: 1024 → 512 (+ skip connection)
  UpConv3: 512 → 256
  UpConv2: 256 → 128
  UpConv1: 128 → 64
  ↓
출력: (N, 1, 256, 256)    # 높이 맵
```

#### 손실 함수

**UNetPhysicsLoss**:
```python
total_loss = loss_data + λ * loss_smoothness

# 1. 데이터 일치 손실
phase_pred = (4π / wavelength) * height_pred
bucket_pred = A + B * cos(phase_pred + delta)
loss_data = MSE(bucket_pred, bucket_real)

# 2. 평활도 정규화
laplacian = ∇²height_pred
loss_smoothness = MSE(laplacian, 0)
```

**물리적 의미**:
- `loss_data`: 예측 높이가 물리적으로 일치하는가?
- `loss_smoothness`: 표면이 부드러운가? (노이즈 억제)

#### 학습 과정

1. **데이터 로딩**:
   ```python
   dataset = WaferPatchDataset(train_dir, patch_size=256, num_channels=16)
   loader = DataLoader(dataset, batch_size=4, shuffle=True)
   ```

2. **최적화**:
   ```python
   optimizer = Adam(model.parameters(), lr=1e-3)
   scheduler = StepLR(optimizer, step_size=epochs//3, gamma=0.5)
   ```

3. **체크포인트**:
   - 매 에포크마다 손실 감소 시 자동 저장
   - `unet_best.pth`: 최고 성능 모델
   - `unet_final.pth`: 마지막 에포크 모델

#### 학습 곡선 해석

**정상적인 학습**:
```
Total Loss: 1e-1 → 1e-3 → 1e-5 (로그 스케일 감소)
Data Loss:  1e-1 → 1e-3
Smooth Loss: 1e-3 → 1e-5
```

**문제 신호**:
- Loss가 증가: 학습률이 너무 큼
- Loss가 정체: 더 긴 학습 필요 또는 더 복잡한 모델 필요
- Loss가 진동: 배치 크기를 늘리거나 학습률 감소

---

### 3.3 PINN 미세조정 (finetune)

#### 목적
물리 제약을 더 강하게 적용하여 정밀도를 향상시킵니다.

#### 현재 상태
⚠️ **고급 기능 - 구현 예정**

대부분의 경우 UNet만으로 충분합니다. PINN 미세조정은:
- 극도로 높은 정밀도가 필요한 경우
- 물리 법칙을 엄격히 만족해야 하는 경우
- 노이즈가 많은 실제 데이터의 경우

에 유용합니다.

---

### 3.4 추론 (inference)

#### 목적
학습된 모델로 새로운 데이터에 대해 높이를 예측합니다.

#### 실행
```bash
python examples/run_psi_pipeline.py inference
```

#### 주요 옵션
```bash
--num-visualize 5   # 시각화할 결과 수
```

#### 동작 과정

1. **모델 로드**:
   ```python
   model = UNet(n_channels=16, n_classes=1)
   checkpoint = torch.load('unet_best.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   ```

2. **추론**:
   ```python
   with torch.no_grad():
       for buckets, _ in test_loader:
           height_pred = model(buckets)
   ```

3. **저장**:
   - 예측 결과: `inference_results.json`
   - 시각화: `visualizations/inference_XXXX.png`

#### 시각화 구성

각 `inference_XXXX.png`는 2×3 레이아웃:

```
┌──────────────┬──────────────┬──────────────┐
│  예측 높이    │ Ground Truth │  절대 오차    │
│              │              │              │
├──────────────┼──────────────┼──────────────┤
│  Bucket 0    │  Bucket 1    │  Bucket 2    │
│  (샘플)      │  (샘플)      │  (샘플)      │
└──────────────┴──────────────┴──────────────┘
```

**해석**:
- 예측과 GT가 유사 → 좋은 성능
- 오차 맵이 균일 → 체계적 오차 가능성
- 오차 맵이 랜덤 → 노이즈

---

### 3.5 테스트 (test)

#### 목적
추론 결과를 정량적으로 평가합니다.

#### 실행
```bash
python examples/run_psi_pipeline.py test
```

#### 계산되는 메트릭

**1. RMSE (Root Mean Square Error)**:
```python
RMSE = sqrt(mean((pred - gt)²))
```
- 단위: 높이 맵과 동일 (예: 미터)
- 의미: 평균적인 오차 크기
- 목표: 낮을수록 좋음

**2. MAE (Mean Absolute Error)**:
```python
MAE = mean(|pred - gt|)
```
- 단위: 높이 맵과 동일
- 의미: 절대 오차의 평균
- RMSE보다 이상치에 덜 민감

**3. MAPE (Mean Absolute Percentage Error)**:
```python
MAPE = mean(|pred - gt| / |gt|) × 100
```
- 단위: 퍼센트 (%)
- 의미: 상대적 오차
- 주의: gt가 0에 가까우면 불안정

#### 통계 정보

각 메트릭에 대해:
- `mean`: 평균
- `std`: 표준편차
- `min`: 최소값
- `max`: 최대값

#### 출력 예제

```json
{
  "rmse_mean": 1.234e-06,
  "rmse_std": 2.456e-07,
  "rmse_min": 5.678e-07,
  "rmse_max": 3.456e-06,
  "mae_mean": 0.987e-06,
  "mae_std": 1.234e-07,
  "mape_mean": 2.45,
  "mape_std": 0.67,
  "num_samples": 5
}
```

**해석**:
- RMSE ~1μm: 마이크로미터 단위 정확도
- MAPE ~2%: 평균 2% 상대 오차
- 낮은 std: 일관된 성능

#### 시각화

**`metrics_distribution.png`**:

```
┌──────────────┬──────────────┬──────────────┐
│ RMSE 히스토그램│ MAE 히스토그램 │ MAPE 히스토그램│
│              │              │              │
│   빈도        │   빈도        │   빈도        │
│    │         │    │         │    │         │
│  ──┼──       │  ──┼──       │  ──┼──       │
│     RMSE     │     MAE      │    MAPE(%)   │
└──────────────┴──────────────┴──────────────┘
```

**해석**:
- 정규 분포 → 정상적인 오차 분포
- 왼쪽으로 치우침 → 대부분 좋은 성능
- 이상치 있음 → 특정 샘플 문제 가능

---

### 3.6 전체 실행 (all)

#### 목적
모든 단계를 순차적으로 실행합니다.

#### 실행
```bash
python examples/run_psi_pipeline.py all
```

#### 실행 순서
```
generate → pretrain → inference → test
```

**주의**: `finetune`은 제외됩니다 (선택적 단계).

#### 권장 사용 시나리오

**첫 실행**:
```bash
python examples/run_psi_pipeline.py all
```

**빠른 테스트** (작은 데이터, 짧은 학습):
```bash
python examples/run_psi_pipeline.py all \
  --num-train-samples 10 \
  --num-test-samples 3 \
  --epochs 20
```

**프로덕션** (많은 데이터, 긴 학습):
```bash
python examples/run_psi_pipeline.py all \
  --num-train-samples 200 \
  --num-test-samples 50 \
  --epochs 200 \
  --device cuda
```

---

## 4. 옵션 상세 설명

### 4.1 경로 옵션

#### `--output-dir`
- **기본값**: `outputs`
- **설명**: 모든 출력 파일의 루트 디렉토리
- **예제**:
  ```bash
  --output-dir /path/to/results
  --output-dir ./experiment_1
  ```

### 4.2 데이터 생성 옵션

#### `--num-train-samples`
- **기본값**: `20`
- **범위**: `1 ~ 수천`
- **설명**: 훈련 샘플 수
- **권장**:
  - 빠른 테스트: `10`
  - 일반 사용: `50`
  - 고성능: `200+`
- **영향**:
  - 많을수록 일반화 성능 향상
  - 많을수록 학습 시간 증가

#### `--num-test-samples`
- **기본값**: `5`
- **범위**: `1 ~ 수백`
- **설명**: 테스트 샘플 수
- **권장**:
  - 최소: `5`
  - 통계적 유의성: `20+`
  - 철저한 평가: `50+`

#### `--image-size`
- **기본값**: `256`
- **선택지**: `64, 128, 256, 512`
- **설명**: 생성할 이미지 크기 (정사각형)
- **영향**:
  - 큰 크기: 더 상세, 더 느림
  - 작은 크기: 빠름, 덜 상세

### 4.3 학습 옵션

#### `--epochs`
- **기본값**: `50`
- **범위**: `10 ~ 1000`
- **설명**: 학습 에포크 수
- **권장**:
  - 빠른 테스트: `20`
  - 일반 사용: `50-100`
  - 최고 성능: `200+`
- **팁**: 학습 곡선을 보고 조기 종료 판단

#### `--batch-size`
- **기본값**: `4`
- **범위**: `1 ~ 32`
- **설명**: 한 번에 처리할 샘플 수
- **영향**:
  - 크면: 빠름, 메모리 많이 사용
  - 작으면: 느림, 메모리 적게 사용
- **GPU 메모리별 권장**:
  - 4GB: `batch_size=2`
  - 8GB: `batch_size=4`
  - 16GB+: `batch_size=8`

#### `--learning-rate`
- **기본값**: `1e-3` (0.001)
- **범위**: `1e-5 ~ 1e-2`
- **설명**: 학습률
- **권장**:
  - 안정적: `5e-4` (0.0005)
  - 기본: `1e-3` (0.001)
  - 빠른 수렴: `5e-3` (0.005)
- **팁**: 
  - Loss가 발산하면 줄이기
  - 수렴이 느리면 높이기

#### `--patch-size`
- **기본값**: `256`
- **설명**: 학습 시 사용할 패치 크기
- **주의**: `image_size`와 같거나 작아야 함

#### `--smoothness-weight`
- **기본값**: `1e-4`
- **범위**: `1e-6 ~ 1e-2`
- **설명**: 평활도 정규화 가중치 (λ)
- **영향**:
  - 크면: 더 부드러운 표면
  - 작으면: 더 상세한 표면
- **권장**:
  - 노이즈 많음: `1e-3`
  - 일반: `1e-4`
  - 세밀한 특징: `1e-5`

### 4.4 추론/평가 옵션

#### `--num-visualize`
- **기본값**: `5`
- **범위**: `1 ~ test_samples`
- **설명**: 시각화할 결과 수
- **주의**: 많을수록 디스크 공간 많이 사용

### 4.5 시스템 옵션

#### `--device`
- **기본값**: `cuda`
- **선택지**: `cuda`, `cpu`
- **설명**: 연산 디바이스
- **자동 처리**: CUDA 사용 불가 시 자동으로 CPU 사용

---

## 5. 출력 파일 구조

### 5.1 전체 구조

```
outputs/
├─ synthetic_data/
│   ├─ train/
│   │   ├─ sample_0000/
│   │   │   ├─ bucket_00.bmp
│   │   │   ├─ ...
│   │   │   ├─ bucket_15.bmp
│   │   │   └─ ground_truth.npy
│   │   └─ ...
│   └─ test/
│       └─ ...
├─ models/
│   ├─ unet_best.pth
│   └─ unet_final.pth
├─ results/
│   ├─ metrics.json
│   ├─ inference_results.json
│   ├─ unet_history.json
│   ├─ unet_training_curve.png
│   ├─ metrics_distribution.png
│   └─ visualizations/
│       ├─ inference_0000.png
│       └─ ...
└─ logs/  (향후 추가)
```

### 5.2 파일 형식

#### `ground_truth.npy`
- **형식**: NumPy binary
- **타입**: `float32`
- **형태**: `(H, W)`
- **단위**: 미터 (m)
- **로딩**:
  ```python
  height = np.load('ground_truth.npy')
  ```

#### `bucket_XX.bmp`
- **형식**: BMP 이미지
- **타입**: `uint8`
- **형태**: `(H, W)`
- **범위**: `0-255`
- **로딩**:
  ```python
  from PIL import Image
  bucket = np.array(Image.open('bucket_00.bmp'))
  ```

#### `unet_best.pth`
- **형식**: PyTorch checkpoint
- **내용**:
  ```python
  {
      'epoch': int,                    # 에포크 번호
      'model_state_dict': OrderedDict, # 모델 가중치
      'optimizer_state_dict': dict,    # 옵티마이저 상태
      'loss': float,                   # 손실값
      'config': dict                   # 설정
  }
  ```
- **로딩**:
  ```python
  checkpoint = torch.load('unet_best.pth')
  model.load_state_dict(checkpoint['model_state_dict'])
  ```

#### `metrics.json`
- **형식**: JSON
- **구조**:
  ```json
  {
      "rmse_mean": float,
      "rmse_std": float,
      "rmse_min": float,
      "rmse_max": float,
      "mae_mean": float,
      "mae_std": float,
      "mape_mean": float,
      "mape_std": float,
      "num_samples": int
  }
  ```

---

## 6. 고급 사용법

### 6.1 커스텀 손실 함수 가중치

코드에서 직접 수정:

```python
# examples/run_psi_pipeline.py의 train_unet() 함수 내부

criterion = UNetPhysicsLoss(
    wavelengths=self.wavelengths,
    num_buckets=self.num_buckets,
    smoothness_weight=1e-3  # 여기를 변경
)
```

### 6.2 모델 구조 변경

```python
# UNet 깊이 조절
model = UNet(
    n_channels=16,
    n_classes=1,
    bilinear=False  # True: 빠름, False: 더 정확
)
```

### 6.3 다른 파장 사용

```python
# src/data_generator.py
DEFAULT_WAVELENGTHS = [4.0, 5.0, 6.0, 7.0]  # 커스텀 파장 (μm)
```

### 6.4 조기 종료 구현

```python
# train_unet() 함수에 추가
patience = 10
best_epoch = 0

for epoch in range(self.config.epochs):
    # ... 학습 코드 ...
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = epoch
    elif epoch - best_epoch > patience:
        print(f"조기 종료: {patience} 에포크 동안 개선 없음")
        break
```

### 6.5 다중 GPU 학습

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

## 7. 트러블슈팅

### 7.1 메모리 관련

#### 문제: "CUDA out of memory"

**해결 방법**:

1. **배치 크기 줄이기**:
   ```bash
   --batch-size 2
   ```

2. **이미지 크기 줄이기**:
   ```bash
   --image-size 128
   ```

3. **CPU 사용**:
   ```bash
   --device cpu
   ```

4. **그래디언트 누적** (코드 수정 필요):
   ```python
   accumulation_steps = 4
   for i, (inputs, targets) in enumerate(loader):
       loss = criterion(model(inputs), inputs)
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### 7.2 성능 관련

#### 문제: "Loss가 감소하지 않음"

**원인 및 해결**:

1. **학습률이 너무 낮음**:
   ```bash
   --learning-rate 5e-3
   ```

2. **데이터가 너무 적음**:
   ```bash
   --num-train-samples 50
   ```

3. **에포크가 부족**:
   ```bash
   --epochs 200
   ```

4. **초기화 문제**:
   ```python
   # 코드에서 모델 초기화 확인
   def init_weights(m):
       if isinstance(m, nn.Conv2d):
           nn.init.kaiming_normal_(m.weight)
   model.apply(init_weights)
   ```

#### 문제: "Loss가 발산함 (NaN)"

**해결 방법**:

1. **학습률 줄이기**:
   ```bash
   --learning-rate 1e-4
   ```

2. **그래디언트 클리핑** (코드 추가):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **정규화 확인**:
   ```python
   # Bucket 이미지 정규화
   buckets = buckets / 255.0
   ```

#### 문제: "테스트 성능이 나쁨 (과적합)"

**해결 방법**:

1. **훈련 데이터 늘리기**:
   ```bash
   --num-train-samples 100
   ```

2. **평활도 가중치 늘리기**:
   ```bash
   --smoothness-weight 1e-3
   ```

3. **Dropout 추가** (코드 수정):
   ```python
   class UNet(nn.Module):
       def __init__(self, ...):
           ...
           self.dropout = nn.Dropout(0.5)
   ```

4. **조기 종료 사용**.

### 7.3 파일 관련

#### 문제: "Model file not found"

**해결 방법**:
```bash
# 학습부터 다시
python examples/run_psi_pipeline.py pretrain
```

#### 문제: "No inference results"

**해결 방법**:
```bash
# 추론부터 다시
python examples/run_psi_pipeline.py inference
python examples/run_psi_pipeline.py test
```

#### 문제: "Permission denied"

**해결 방법**:
```bash
# 출력 디렉토리 권한 확인
chmod -R 755 outputs/

# 또는 다른 디렉토리 사용
--output-dir ~/my_results
```

### 7.4 데이터 관련

#### 문제: "Dataset is empty"

**확인 사항**:

1. `outputs/synthetic_data/train/` 존재 확인
2. 샘플 디렉토리 내부에 16개 BMP 파일 확인
3. `ground_truth.npy` 파일 확인

**재생성**:
```bash
python examples/run_psi_pipeline.py generate
```

---

## 8. FAQ

### Q1: UNet만으로 충분한가요? PINN이 꼭 필요한가요?

**A**: 대부분의 경우 UNet만으로 충분합니다.

**UNet 사용 권장**:
- 빠른 추론 필요
- 대량의 데이터 있음
- 실시간 처리 필요

**PINN 추가 고려**:
- 극도의 정밀도 필요
- 데이터가 적음
- 물리 법칙 엄격히 준수

### Q2: 얼마나 정확한가요?

**A**: 조건에 따라 다릅니다.

**일반적인 성능**:
- RMSE: `1-10 μm` (마이크로미터)
- MAPE: `1-5%`

**영향 요인**:
- 훈련 데이터 양
- 학습 에포크 수
- 표면 복잡도
- 노이즈 수준

### Q3: 실제 데이터로도 작동하나요?

**A**: 현재 버전은 합성 데이터 전용입니다.

실제 데이터 지원 예정:
- 노이즈 처리
- 캘리브레이션
- 불완전한 데이터 처리

### Q4: 다른 파장을 사용하고 싶어요.

**A**: `src/data_generator.py`에서 변경:

```python
DEFAULT_WAVELENGTHS = [4.0, 5.0, 6.0, 7.0]  # μm 단위
```

주의: 파장 개수를 변경하면 채널 수도 변경됩니다.
- 3개 파장 × 4 buckets = 12 채널
- 4개 파장 × 4 buckets = 16 채널
- 5개 파장 × 4 buckets = 20 채널

### Q5: 학습 시간은 얼마나 걸리나요?

**A**: 시스템과 설정에 따라 다릅니다.

**예상 시간** (기본 설정):

| 하드웨어 | 데이터 생성 | 학습 (50 에포크) | 추론 | 총 시간 |
|---------|-----------|----------------|------|--------|
| CPU (i7) | 2분 | 25분 | 1분 | ~30분 |
| GPU (RTX 3080) | 2분 | 3분 | 10초 | ~5분 |
| GPU (A100) | 1분 | 1분 | 5초 | ~2분 |

### Q6: 결과를 어떻게 해석하나요?

**A**: 메트릭 기준:

**RMSE**:
- `< 1 μm`: 매우 우수
- `1-5 μm`: 우수
- `5-10 μm`: 양호
- `> 10 μm`: 개선 필요

**MAPE**:
- `< 1%`: 매우 우수
- `1-5%`: 우수
- `5-10%`: 양호
- `> 10%`: 개선 필요

### Q7: 모델을 저장하고 나중에 사용할 수 있나요?

**A**: 예, 가능합니다.

```python
# 저장 (자동으로 수행됨)
torch.save(checkpoint, 'unet_best.pth')

# 나중에 로드
model = UNet(n_channels=16, n_classes=1)
checkpoint = torch.load('unet_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 추론
with torch.no_grad():
    prediction = model(new_input)
```

### Q8: 여러 실험을 동시에 실행하려면?

**A**: 다른 출력 디렉토리 사용:

```bash
# 실험 1
python examples/run_psi_pipeline.py all --output-dir outputs/exp1

# 실험 2 (다른 터미널)
python examples/run_psi_pipeline.py all --output-dir outputs/exp2 --epochs 100
```

---

## 🎓 추가 리소스

- **[빠른 시작 가이드](PSI_파이프라인_빠른시작.md)**: 3분 안에 시작하기
- **[위상천이간섭법 가이드](concepts/위상천이간섭법_완전_가이드.md)**: PSI 이론 상세 설명
- **[손실 함수 가이드](04_손실함수.md)**: 물리 기반 손실 함수 설명

---

**문의사항이나 버그 발견 시 GitHub 이슈를 등록해주세요!**
