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
    다음 명령어를 실행하여 모든 의존성을 설치할 수 있습니다:
    ```bash
    pip install torch numpy matplotlib scipy
    ```
    *참고: CUDA가 지원되는 GPU가 있는 경우, 이를 지원하는 특정 버전의 PyTorch를 설치해야 할 수 있습니다. 자세한 내용은 [PyTorch 공식 웹사이트](https://pytorch.org/get-started/locally/)를 참조하십시오.*

## 2. 예제 실행 방법

이 프로젝트에는 모델의 능력을 잘 보여주는 1D 헬름홀츠 방정식을 푸는 예제 스크립트가 포함되어 있습니다.

1.  **프로젝트의 루트 디렉토리로 이동합니다.**

2.  **예제 스크립트 실행:**
    ```bash
    python examples/solve_helmholtz_1d.py
    ```

## 3. 예상 출력

### 콘솔 출력

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
