# 🚀 PSI 파이프라인 빠른 시작 가이드

**목표**: 3분 안에 위상천이간섭법(PSI) 기반 3D 높이 재구성 파이프라인을 실행하세요!

---

## ⚡ 초간단 실행 (1줄)

```bash
python examples/run_psi_pipeline.py all
```

이 한 줄로:
- ✅ 합성 bucket 이미지 생성 (훈련 20개, 테스트 5개)
- ✅ UNet 모델 학습 (50 에포크)
- ✅ 테스트 데이터로 추론
- ✅ 성능 평가 및 시각화

**실행 시간**: CPU ~30분, GPU ~5분

**결과 확인**:
```
outputs/
├─ results/
│   ├─ metrics.json          # 📊 RMSE, MAE 등 수치
│   └─ visualizations/       # 🎨 예측 vs 실제 비교 이미지
└─ models/
    └─ unet_best.pth         # 💾 학습된 모델
```

---

## 📋 단계별 실행

### 1️⃣ 데이터 생성만

```bash
python examples/run_psi_pipeline.py generate
```

**출력**: `outputs/synthetic_data/train/`, `outputs/synthetic_data/test/`

### 2️⃣ 학습만

```bash
python examples/run_psi_pipeline.py pretrain
```

**출력**: `outputs/models/unet_best.pth`

### 3️⃣ 추론만

```bash
python examples/run_psi_pipeline.py inference
```

**필요**: 학습된 모델 (`unet_best.pth`)  
**출력**: `outputs/results/inference_results.json`

### 4️⃣ 평가만

```bash
python examples/run_psi_pipeline.py test
```

**필요**: 추론 결과 (`inference_results.json`)  
**출력**: `outputs/results/metrics.json`

---

## 🎛️ 자주 사용하는 옵션

### GPU 사용 (기본값)
```bash
python examples/run_psi_pipeline.py all --device cuda
```

### CPU만 사용
```bash
python examples/run_psi_pipeline.py all --device cpu
```

### 더 많은 데이터로 학습
```bash
python examples/run_psi_pipeline.py all --num-train-samples 100 --num-test-samples 20
```

### 빠른 테스트 (작은 데이터셋)
```bash
python examples/run_psi_pipeline.py all --num-train-samples 10 --epochs 20
```

### 긴 학습 (더 나은 성능)
```bash
python examples/run_psi_pipeline.py all --epochs 200 --learning-rate 5e-4
```

---

## 📊 결과 확인하기

### 1. 메트릭 확인
```bash
cat outputs/results/metrics.json
```

예제 출력:
```json
{
  "rmse_mean": 1.234e-06,
  "mae_mean": 0.987e-06,
  "mape_mean": 2.45
}
```

### 2. 시각화 보기
```
outputs/results/visualizations/
├─ inference_0000.png   # 첫 번째 샘플
├─ inference_0001.png   # 두 번째 샘플
└─ ...
```

각 이미지는 다음을 포함합니다:
- 🔮 예측 높이 맵
- ✅ Ground Truth
- ❌ 절대 오차
- 📸 입력 Bucket 이미지 샘플

### 3. 학습 곡선 보기
```
outputs/results/unet_training_curve.png
```

---

## 🆘 문제 해결

### ❌ "CUDA out of memory"
```bash
# 배치 크기 줄이기
python examples/run_psi_pipeline.py all --batch-size 2

# 또는 CPU 사용
python examples/run_psi_pipeline.py all --device cpu
```

### ❌ "Model file not found"
```bash
# 학습부터 다시 시작
python examples/run_psi_pipeline.py pretrain
```

### ❌ "No inference results"
```bash
# 추론부터 다시 시작
python examples/run_psi_pipeline.py inference
python examples/run_psi_pipeline.py test
```

---

## 💡 팁

### 재현 가능한 결과
- 시드가 고정되어 있어 매번 같은 결과가 나옵니다.
- 다른 결과를 원하면 코드에서 시드를 변경하세요.

### 커스텀 설정
- 모든 옵션은 `--help`로 확인 가능:
  ```bash
  python examples/run_psi_pipeline.py --help
  ```

### 성능 향상 팁
1. **더 많은 데이터**: `--num-train-samples 200`
2. **더 긴 학습**: `--epochs 200`
3. **작은 학습률**: `--learning-rate 5e-4`
4. **평활도 조절**: `--smoothness-weight 1e-5` (더 부드러운 결과)

---

## 📚 더 알아보기

**상세한 설명이 필요하신가요?**

👉 **[PSI 파이프라인 상세 가이드](PSI_파이프라인_상세가이드.md)**

이 가이드에는:
- 🧠 각 단계의 동작 원리
- ⚙️ 모든 옵션의 상세 설명
- 🔧 고급 사용법과 커스터마이징
- 🐛 자세한 트러블슈팅
- 📖 실제 데이터 사용법 (예정)

---

## ✨ 성공 사례

### 예제 결과
```
📊 평가 결과:
   RMSE: 1.234e-06 ± 2.456e-07
   MAE:  0.987e-06 ± 1.234e-07
   MAPE: 2.45% ± 0.67%
```

이 정도면 **나노미터 단위의 정확도**입니다! 🎉

---

**문의사항이 있으시면 이슈를 등록해주세요!**
