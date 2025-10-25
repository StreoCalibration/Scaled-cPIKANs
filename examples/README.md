# Examples 폴더

이 폴더는 Scaled-cPIKAN 프로젝트의 **실행 가능한 예제 스크립트**를 포함합니다.

## 📌 목적

- **실제 사용 예제**: 전체 파이프라인 실행 및 시각화
- **데모 스크립트**: 프로젝트의 기능을 빠르게 확인
- **벤치마크**: 논문 결과 재현

## 🚨 중요: 단위 테스트와의 차이

### Examples vs Tests

| 구분 | Examples | Tests |
|------|----------|-------|
| **목적** | 실제 사용 시연 | 기능 검증 |
| **실행 시간** | 길다 (수 분 ~ 수십 분) | 짧다 (수 초 ~ 수 분) |
| **출력** | 플롯, 이미지, 모델 파일 | 성공/실패 메시지 |
| **데이터** | 큰 데이터셋, 고해상도 | 작은 데이터셋, 저해상도 |
| **실행 방법** | 직접 실행 | unittest 프레임워크 |

### 단위 테스트는 `tests/` 폴더에 있습니다

코드 검증 및 자동화된 테스트는 다음을 사용하세요:

```bash
# 전체 단위 테스트 실행
python -m unittest discover tests

# 특정 테스트만 실행
python -m unittest tests.test_helmholtz_solver
python -m unittest tests.test_reconstruction_pinn
python -m unittest tests.test_reconstruction_buckets
```

## 📂 파일 설명

### 1. `run_pipeline.py`
**전체 파이프라인 실행 스크립트**

- **기능**: 
  - 합성 데이터 생성 (사전학습용)
  - 모델 사전학습 (supervised)
  - 모델 미세조정 (physics-informed)
  - 최종 모델 저장

- **실행 방법**:
  ```bash
  # 기본 실행
  python examples/run_pipeline.py
  
  # 커스텀 설정
  python examples/run_pipeline.py \
      --pretrain-epochs 50 \
      --finetune-epochs 30 \
      --patch-size 128
  
  # 미세조정 데이터 생성
  python examples/run_pipeline.py --generate-finetune-data --num-finetune-samples 10
  ```

- **출력**:
  - `synthetic_data/train/`: 사전학습 데이터
  - `real_data/train/`: 미세조정 데이터 (생성 시)
  - `models/pinn_final.pth`: 최종 모델

- **소요 시간**: GPU 기준 10-15분, CPU 기준 30-60분

### 2. `solve_helmholtz_1d.py`
**1D Helmholtz 방정식 솔버 데모**

- **목적**: Scaled-cPIKAN PINN의 기본 기능 시연
- **문제**: u_xx + k²u = 0, u(0) = 0, u(1) = sin(k)
- **실행 방법**:
  ```bash
  python examples/solve_helmholtz_1d.py
  ```
- **출력**:
  - `helmholtz_loss_history.png`: 손실 곡선
  - `helmholtz_solution.png`: 해답 및 오차 플롯

### 3. `solve_reconstruction_pinn.py`
**3D 높이 재구성 (위상 맵 기반) 데모**

- **목적**: 다중 파장 위상 측정으로부터 3D 재구성
- **실행 방법**:
  ```bash
  python examples/solve_reconstruction_pinn.py
  ```
- **출력**:
  - `reconstruction_pinn_results/01_input_data.png`
  - `reconstruction_pinn_results/02_reconstruction_results.png`
  - `reconstruction_pinn_results/03_loss_history.png`

### 4. `solve_reconstruction_from_buckets.py`
**3D 높이 재구성 (버킷 이미지 기반) 데모**

- **목적**: 원시 버킷 강도 이미지로부터 직접 재구성
- **실행 방법**:
  ```bash
  # 먼저 데이터 생성 필요
  python -m reconstruction.data_generator
  
  # 재구성 실행
  python examples/solve_reconstruction_from_buckets.py
  ```
- **출력**:
  - `reconstruction_from_buckets_results/01_input_bucket_data.png`
  - `reconstruction_from_buckets_results/02_reconstruction_results.png`
  - `reconstruction_from_buckets_results/03_loss_history.png`

## 🔧 사용 시나리오

### 1. 빠른 기능 확인
```bash
# 1D 문제로 빠르게 확인 (2-5분)
python examples/solve_helmholtz_1d.py
```

### 2. 전체 파이프라인 실행
```bash
# 사전학습 + 미세조정 전체 과정 (10-15분)
python examples/run_pipeline.py
```

### 3. 특정 기능 테스트
```bash
# 위상 기반 재구성 테스트 (10-20분)
python examples/solve_reconstruction_pinn.py

# 버킷 기반 재구성 테스트 (15-30분)
python examples/solve_reconstruction_from_buckets.py
```

## ⚠️ 주의사항

1. **GPU 권장**: 예제 스크립트는 GPU에서 실행하는 것을 권장합니다.
   - CPU에서 실행 시 시간이 오래 걸릴 수 있습니다.

2. **디스크 공간**: 데이터 생성 시 충분한 디스크 공간이 필요합니다.
   - `run_pipeline.py`: ~100-500MB
   - `solve_reconstruction_from_buckets.py`: ~50-100MB

3. **메모리 요구사항**:
   - 최소 8GB RAM
   - GPU 메모리: 4GB 이상 권장

## 🧪 개발 및 테스트

### 코드 수정 후 검증

예제 스크립트를 수정한 경우:

1. **먼저 단위 테스트 실행**:
   ```bash
   python -m unittest discover tests
   ```

2. **단위 테스트 통과 후 예제 실행**:
   ```bash
   python examples/<modified_script>.py
   ```

### 새로운 기능 추가 시

1. **단위 테스트 작성**: `tests/` 폴더에 추가
2. **예제 스크립트 작성**: `examples/` 폴더에 추가 (선택사항)

## 📚 관련 문서

- **단위 테스트**: `tests/` 폴더 참조
- **API 문서**: `doc/` 폴더 참조
- **프로젝트 구조**: 루트 `README.md` 참조

## 💡 팁

- **빠른 디버깅**: 예제 스크립트의 에포크 수를 줄여 빠르게 테스트
  ```bash
  python examples/run_pipeline.py --pretrain-epochs 10 --finetune-epochs 5
  ```

- **결과 비교**: 여러 설정으로 실행 후 출력 이미지 비교
  
- **모델 재사용**: `run_pipeline.py`로 학습된 모델을 다른 스크립트에서 로드 가능

---

**질문이나 문제가 있으면 이슈를 등록해주세요!**
