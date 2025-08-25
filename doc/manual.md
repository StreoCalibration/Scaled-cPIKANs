# 사용자 매뉴얼: Scaled-cPIKAN

이 문서는 Scaled-cPIKAN 프로젝트의 개발 환경을 설정하고 제공된 예제를 실행하는 방법에 대한 지침을 제공합니다.

## 1. 설정

### 사전 요구사항

-   Python 3.8 이상
-   패키지 설치를 위한 `pip`

### 의존성

이 프로젝트는 다음 Python 라이브러리가 필요합니다:

-   `torch`: 핵심 딥러닝 프레임워크
-   `numpy`: 수치 연산
-   `matplotlib`: 결과 플롯 생성
-   `scipy`: 콜로케이션 포인트의 라틴 하이퍼큐브 샘플링

### 설치

1.  **리포지토리 복제 (아직 하지 않았다면):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **가상 환경 생성 및 활성화 (권장):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows에서는 `venv\Scripts\activate` 사용
    ```

3.  **필요한 패키지 설치:**
    프로젝트 루트 디렉토리에 `requirements.txt` 파일이 포함되어 있습니다. 다음 명령어를 실행하여 모든 의존성을 한 번에 설치할 수 있습니다:
    ```bash
    pip install -r requirements.txt
    ```
    *참고: CUDA가 지원되는 GPU가 있는 경우, 이를 지원하는 특정 버전의 PyTorch를 설치해야 할 수 있습니다. 자세한 내용은 [PyTorch 공식 웹사이트](https://pytorch.org/get-started/locally/)를 참조하십시오.*

## 2. 예제 실행 방법

프로젝트에는 두 가지 주요 예제가 포함되어 있습니다.

### 2.1. 1D 헬름홀츠 방정식 예제

이 예제는 모델의 능력을 잘 보여주는 1D 헬름홀츠 방정식을 풉니다.

1.  **프로젝트의 루트 디렉토리로 이동합니다.**

2.  **예제 스크립트 실행:**
    ```bash
    python examples/solve_helmholtz_1d.py
    ```

### 2.2. 3D 재구성 예제 (PINN 기반)

이 예제는 물리 정보 신경망(PINN)을 사용하여 다중 파장 레이저 데이터로부터 3D 표면 높이를 재구성하는 더 복잡한 역문제를 해결하는 방법을 보여줍니다.

1.  **프로젝트의 루트 디렉토리로 이동합니다.**

2.  **예제 스크립트 실행:**
    ```bash
    python examples/solve_reconstruction_pinn.py
    ```
    *참고: 이 스크립트는 계산량이 많으며, 특히 CPU에서 실행할 경우 완료하는 데 상당한 시간이 걸릴 수 있습니다 (10-20분 이상).*

### 2.3. 버킷 기반 3D 복원 워크플로우

이 워크플로우는 레이저 버킷 이미지를 직접 사용하여 Scaled-cPIKAN PINN으로 3D 표면을 복원하는 방법을 설명합니다. 전체 과정은 데이터 생성 → 학습 → 추론의 세 단계로 구성됩니다.

1.  **데이터 생성**
    ```bash
    python -m reconstruction.data_generator --num-lasers 4 --num-buckets 3 --wavelengths 5.0 5.5 6.05 6.655
    ```
    - `--num-lasers`: 시뮬레이션할 레이저 파장의 수
    - `--num-buckets`: 각 레이저에서 측정하는 버킷 이미지 개수
    - `--wavelengths`: 쉼표로 구분된 파장 목록(µm)
    실행 후 `reconstruction_data/` 폴더에 `ground_truth_height.npy`, `wrapped_phase_laser_1.npy` 등 여러 `.npy` 파일이 생성됩니다.

2.  **학습**
    ```bash
    python examples/solve_reconstruction_from_buckets.py --num-lasers 4 --num-buckets 3 --wavelengths 5.0 5.5 6.05 6.655
    ```
    버킷 이미지를 이용해 PINN을 학습하며, 결과 모델 가중치와 손실 이력은 `reconstruction_from_buckets_results/`에 저장됩니다.

    
3.  **추론**
    ```bash
    python -m reconstruction.main --num-lasers 4 --num-buckets 3 --wavelengths 5.0 5.5 6.05 6.655
    ```
    학습된 모델을 사용해 높이 맵을 복원하고, 최종 결과는 `reconstruction_data/reconstructed_height.npy`로 저장됩니다.

## 3. 예상 출력

### 3.1. 헬름홀츠 예제 출력

#### 콘솔 출력

스크립트를 실행하면 터미널에 훈련 진행 상황을 보여주는 출력이 나타납니다. 먼저 사용 중인 장치(CPU 또는 CUDA)가 표시됩니다. 그런 다음 Adam 및 L-BFGS 최적화 단계 모두에 대해 일정한 간격으로 손실 값을 기록합니다. 마지막으로 모델 예측의 최종 상대 L2 오차를 출력합니다.

출력은 다음과 유사합니다:
```
Using device: cuda
--- Starting Stage 1: Adam Optimization ---
[Adam] Epoch [500/2000] - loss_pde: 1.2345e-02 - loss_bc: 6.7890e-03 - total_loss: 2.5923e-02
...
[Adam] Epoch [2000/2000] - loss_pde: 1.0000e-04 - loss_bc: 2.0000e-05 - total_loss: 1.4000e-04

--- Starting Stage 2: L-BFGS Optimization ---
[L-BFGS] Epoch [1/1] - loss_pde: 9.8765e-06 - loss_bc: 4.3210e-07 - total_loss: 1.8379e-05

--- Training Finished ---

Saved loss history plot to helmholtz_loss_history.png
Final Relative L2 Error: 1.2345e-04
Saved solution plot to helmholtz_solution.png
```

### 생성된 파일

스크립트가 완료되면 프로젝트의 루트 디렉토리에 두 개의 새로운 이미지 파일이 생성됩니다:

1.  **`helmholtz_loss_history.png`**: 훈련 에포크에 따라 손실의 여러 구성 요소(PDE 손실, BC 손실, 총 손실)가 감소하는 것을 보여주는 플롯입니다.
2.  **`helmholtz_solution.png`**: Scaled-cPIKAN 모델이 예측한 해와 헬름홀츠 방정식의 정확한 분석적 해를 비교하는 플롯입니다. 이는 모델의 정확성을 시각적으로 확인하는 데 사용됩니다.

### 3.2. 3D 재구성 예제 출력

#### 콘솔 출력

스크립트는 먼저 합성 데이터 생성을 시작하고, 모델을 초기화한 다음, Adam 및 L-BFGS 최적화 단계의 진행 상황을 로깅합니다. 마지막으로, 재구성된 높이와 실제 높이 간의 최종 평균 제곱근 오차(RMSE)를 출력합니다.

```
--- Starting 3D Reconstruction with Scaled-cPIKAN PINN ---
Using device: cpu

Step 1: Generating synthetic data...
...
Step 2: Initializing the Scaled-cPIKAN model...
...
Step 3: Setting up and running the training...

Training with Adam for 10000 epochs (lr=0.001)...
[Adam] Epoch [500/10000] - loss_total: 1.2345e+00 - loss_data: 1.2300e+00 - loss_smoothness: 4.5000e-01
...
Fine-tuning with L-BFGS for 1 step(s)...
  Completed L-BFGS step. - loss_total: 5.4321e-02 - loss_data: 5.4000e-02 - loss_smoothness: 3.2100e-03

--- Training Complete ---

Step 4: Evaluating model and visualizing results...

Reconstruction complete. Final RMSE: 0.0987
Saved final result visualization to 'reconstruction_pinn_results/02_reconstruction_results.png'
Saved loss history plot to 'reconstruction_pinn_results/03_loss_history.png'

--- 3D Reconstruction with PINN Complete ---
```

#### 생성된 파일

스크립트는 실행 후 `reconstruction_pinn_results`라는 새 디렉토리를 생성합니다. 이 디렉토리에는 다음 파일이 포함됩니다:

1.  **`01_input_data.png`**: PINN 훈련에 사용된 실제 높이와 4개의 시뮬레이션된 래핑된 위상 맵을 보여주는 시각화입니다.
2.  **`02_reconstruction_results.png`**: 실제 높이, PINN에 의해 재구성된 높이, 그리고 둘 사이의 오차 맵을 나란히 비교합니다.
3.  **`03_loss_history.png`**: 훈련 과정 동안 총 손실, 데이터 충실도 손실, 평활도 손실의 감소를 보여주는 플롯입니다.

## 4. 전체 파이프라인 실행 (Full Pipeline Execution)

### 개요 (Overview)
이 프로젝트는 `examples/run_pipeline.py` 스크립트를 통해 전체 머신러닝 파이프라인을 실행할 수 있는 기능을 제공합니다. 이 파이프라인은 `Scaled-cPIKAN` 모델을 사용하여 웨이퍼 표면의 높이 맵을 재구성하는 과정을 자동화하며, 다음 세 가지 주요 단계로 구성됩니다:

1.  **데이터 생성 (Data Generation)**: 사전 훈련(pre-training)과 미세 조정(fine-tuning)에 사용될 합성 데이터셋을 생성합니다.
2.  **사전 훈련 (Pre-training)**: 생성된 합성 데이터의 정답 높이 맵(ground truth height)을 사용하여 `Scaled-cPIKAN` 모델을 지도 학습 방식으로 훈련합니다.
3.  **미세 조정 (Fine-tuning)**: 사전 훈련된 모델을 버킷 이미지만을 사용하여 물리 정보 기반으로 미세 조정합니다. 이 단계에서는 정답 높이 맵을 사용하지 않습니다.

### 사용법 (Usage)
`examples/run_pipeline.py` 스크립트는 다양한 명령줄 인자를 통해 파이프라인의 각 단계를 제어할 수 있습니다.

#### 기본 실행
기본 설정으로 전체 파이프라인을 실행하려면 다음 명령어를 사용합니다.
```bash
python examples/run_pipeline.py
```

#### 전체 인자 목록
`--help` 플래그를 사용하여 모든 사용 가능한 인자와 그에 대한 설명을 확인할 수 있습니다.
```bash
python examples/run_pipeline.py --help
```

다음은 주요 인자에 대한 설명입니다.

**데이터 관련 인자:**
-   `--pretrain-data-dir`: 사전 훈련 데이터셋을 저장할 디렉토리 (기본값: `synthetic_data/train`)
-   `--finetune-data-dir`: 미세 조정 데이터셋을 저장할 디렉토리 (기본값: `real_data/train`)
-   `--num-pretrain-samples`: 생성할 사전 훈련 샘플의 수 (기본값: `10`)
-   `--num-finetune-samples`: 생성할 미세 조정 샘플의 수 (기본값: `5`)
-   `--image-size`: 생성할 합성 이미지의 크기 (픽셀 단위, 기본값: `512`)
-   `--num-buckets`: 레이저 당 버킷 이미지의 수 (기본값: `3`)
-   `--wavelengths`: 사용할 레이저 파장 목록 (미터 단위, 기본값: `635e-9 525e-9 450e-9 405e-9`)
-   `--output-format`: 생성할 버킷 이미지의 포맷 (`npy`, `bmp`, `png` 중 선택, 기본값: `npy`)

**모델 저장 관련 인자:**
-   `--save-path`: 훈련된 최종 모델을 저장할 경로 (기본값: `models/pinn_final.pth`)

**훈련 관련 인자:**
-   `--patch-size`: 훈련에 사용할 이미지 패치의 크기 (기본값: `64`)
-   `--pretrain-epochs`: 사전 훈련 에포크 수 (기본값: `10`)
-   `--pretrain-lr`: 사전 훈련 학습률 (기본값: `1e-3`)
-   `--finetune-epochs`: 미세 조정 에포크 수 (기본값: `10`)
-   `--finetune-lr`: 미세 조정 학습률 (기본값: `1e-5`)
-   `--smoothness-weight`: 미세 조정 시 사용될 평활도 손실의 가중치 (기본값: `1e-7`)

### 사용 예시
-   **더 많은 에포크와 큰 패치 사이즈로 훈련 실행:**
    ```bash
    python examples/run_pipeline.py \
        --pretrain-epochs 50 \
        --finetune-epochs 30 \
        --patch-size 128 \
        --pretrain-lr 5e-4 \
        --save-path "models/pinn_large_patch.pth"
    ```
-   **데이터 생성만 수행 (기존 데이터가 없는 경우):**
    스크립트는 데이터 디렉토리가 이미 존재하면 데이터 생성을 건너뛰므로, 데이터만 생성하려면 먼저 해당 디렉토리를 삭제하거나 다른 경로를 지정해야 합니다.
    ```bash
    python examples/run_pipeline.py \
        --pretrain-data-dir "new_pretrain_data" \
        --finetune-data-dir "new_finetune_data" \
        --pretrain-epochs 0 \
        --finetune-epochs 0
    ```

### 예상 출력
스크립트를 실행하면 다음과 같은 순서로 출력이 표시됩니다.

1.  사용할 장치(CPU 또는 CUDA)가 표시됩니다.
2.  사전 훈련 및 미세 조정을 위한 데이터셋 생성이 진행됩니다 (해당 디렉토리가 비어있는 경우).
3.  사전 훈련 단계가 시작되고, 각 에포크마다 진행률과 손실이 표시됩니다.
4.  사전 훈련이 완료되면 미세 조정 단계가 시작되고, 마찬가지로 진행 상황이 출력됩니다.
5.  모든 과정이 끝나면 최종 모델이 지정된 경로에 저장됩니다.

```
Using device: cuda
Generating 10 data samples in synthetic_data/train...
Data generation complete for synthetic_data/train.
Generating 5 data samples in real_data/train...
Data generation complete for real_data/train.

==================================================
                    PRE-TRAINING
==================================================
Pre-train Epoch 1/10: 100%|██████████| 10/10 [00:01<00:00,  8.00it/s, loss=0.001]
Pre-train Epoch 1/10, Average Loss: 0.001234
...
Pre-training finished.

==================================================
                     FINE-TUNING
==================================================
Finetune Epoch 1/10: 100%|██████████| 5/5 [00:02<00:00,  2.50it/s, loss=0.05]
Finetune Epoch 1/10, Average Loss: 0.054321
...
Fine-tuning finished.

Final fine-tuned model saved to models/pinn_final.pth
```

## 5. 테스트 (Testing)

프로젝트에는 `tests/` 디렉토리에 단위 테스트가 포함되어 있습니다. 이 테스트들은 `unittest` 프레임워크를 기반으로 하며, 프로젝트의 핵심 구성 요소의 정확성을 보장하는 데 도움이 됩니다.

### 테스트 실행
모든 테스트 스위트(기존 `Scaled-cPIKAN` 및 새로운 U-Net 파이프라인 테스트 포함)를 실행하려면, 프로젝트의 루트 디렉토리에서 다음 명령을 실행하십시오:
```bash
python -m unittest discover tests
```
"OK" 메시지와 함께 모든 테스트가 통과해야 합니다.

### 테스트 구조
- **`tests/test_models.py`**: `Scaled_cPIKAN` 모델과 같은 원래 PINN 구성 요소를 테스트합니다.
- **`tests/test_data.py`**: 원래 데이터 처리 유틸리티를 테스트합니다.
- **`tests/test_integration.py`**: 원래 PINN 파이프라인의 통합 테스트를 수행합니다.
- **`tests/test_unet_pipeline.py`**: 새로운 U-Net 파이프라인의 핵심 구성 요소에 대한 테스트를 포함합니다. 여기에는 `UNet` 모델의 순방향 패스, `WaferPatchDataset`의 데이터 로딩 및 증강, `UNetPhysicsLoss`의 계산이 포함됩니다.
