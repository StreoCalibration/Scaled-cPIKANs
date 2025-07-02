#### **1\. 논리 뷰 (Logical View)**

기존 클래스 구조에 \*\*PhantomGenerator\*\*를 추가하고 \*\*DataManager\*\*의 역할을 확장합니다.

* **PhantomGenerator (신규 클래스)**  
  * **책임:** 검증 및 테스트를 위한 **디지털 팬텀(가상 데이터)을 생성**합니다.  
  * **핵심 로직:**  
    * **디지털 팬텀 정의:** configs/ 파일에 명시된 파라미터(예: 마이크로렌즈 어레이의 기하학적/광학적 특성)에 따라 3D 가상 객체(ground-truth)를 생성합니다.
    * **DHM 측정 시뮬레이션:**  
      1. 정의된 팬텀으로부터 **수치적 전파(예: 각 스펙트럼 방법)를 시뮬레이션**하여 센서 평면에서의 복소장 ψ\_o를 계산합니다.
      2. 계산된 복소장을 기반으로 **위상 천이(Phase-shifting)를 시뮬레이션**하여 여러 장의 간섭 무늬 이미지(I\_0, I\_π/2...)를 생성합니다.
      3. **현실적인 노이즈 추가:** CMOS 센서의 물리 기반 노이즈(샷 노이즈, 판독 노이즈) 및 PZT 오차 등을 시뮬레이션에 추가하여 "Sim-to-Real" 간극을 줄입니다.
  * **출력:** 시뮬레이션된 간섭 무늬 이미지 세트 및 검증에 사용할 Ground-Truth 3D 형상.  
* **DataManager (역할 확장)**  
  * **책임:** config 파일에 정의된 \*\*데이터 모드(mode)\*\*에 따라 가상 또는 실제 데이터를 로드하고 전처리합니다. 또한, 처리된 데이터를 저장하고 불러오는 기능을 포함합니다.
  * **핵심 로직:**  
    * **데이터 소스 분기:**  
      * mode: 'synthetic'일 경우: PhantomGenerator를 호출하여 생성된 데이터를 로드합니다.  
      * mode: 'real'일 경우: data/ 폴더에서 실제 측정된 이미지 파일을 로드합니다.  
    * **공통 전처리:** 데이터 소스와 무관하게 **도메인 스케일링**을 포함한 공통 전처리 로직을 수행합니다.
    * **데이터 저장/로드:** 처리된 데이터를 파일로 저장하고, 필요 시 불러와 다음 단계(훈련/평가)에서 사용합니다.

#### **2\. 개발 뷰 (Development View)**

PhantomGenerator 클래스를 위한 소스 파일을 추가하고, config.yaml 파일의 구조를 확장하여 데이터 모드 및 훈련/평가 파라미터를 제어합니다. 소스 코드는 역할별로 모듈화되어 있습니다.

* **디렉토리 구조 (수정)**  
  ```
  scaled-cpikan-dhm/
  ├── configs/
  │   └── microlens_config.yaml
  ├── data/
  │   ├── interferograms/     # 실제 데이터 (더미)
  │   └── processed/          # 처리된 데이터 (coords, interferograms_target, image_dims)
  ├── notebooks/
  │   └── 01_exploration.ipynb
  ├── src/
  │   ├── __init__.py
  │   ├── data_pipeline.py    # 데이터 로드, 전처리, 저장/로드
  │   ├── evaluate_pipeline.py# 모델 평가 및 시각화
  │   ├── losses.py           # 손실 함수 정의
  │   ├── models.py           # 모델 정의
  │   ├── phantom_generator.py# 가상 데이터 생성
  │   ├── train_pipeline.py   # 모델 훈련
  │   └── trainer.py          # 훈련 루프 관리
  ├── main.py                   # 메인 실행 스크립트
  └── requirements.txt          # Python 패키지 요구사항
  ```

* **설정 파일 (microlens_config.yaml) 예시**  
  ```yaml
  # 데이터 소스 설정
  data:
    mode: 'synthetic'  # 'synthetic' 또는 'real'로 설정하여 모드 전환

    # 실제 데이터 설정
    real:
      path: './data/interferograms/'

    # 가상 데이터(시뮬레이션) 설정
    synthetic:
      # 생성된 데이터 저장 경로 (선택 사항)
      save_path: './data/synthetic/'

      # 팬텀 및 광학계 파라미터 (Source 420 참고)
      phantom_type: 'microlens_array'
      optics:
        wavelength_nm: 632.8
        refractive_index_lens: 1.5
        refractive_index_medium: 1.0
      geometry:
        lens_pitch_um: 200
        focal_length_um: 1010
      # ... 기타 팬텀 파라미터

      # 노이즈 모델 설정 (Source 425 참고)
      noise:
        add_noise: true
        shot_noise_level: 0.01
        readout_noise_std: 0.005
        pzt_error_std: 0.02 # 라디안 단위

    # 처리된 데이터 저장 경로
    processed_data_path: './data/processed/'

  # 모델 하이퍼파라미터
  model:
    # ...

  # 훈련 설정
  training:
    learning_rate: 0.001
    epochs: 10
    # 모델 가중치를 저장할 경로
    checkpoint_path: './checkpoints/'
  ```

#### **3\. 프로세스 뷰 (Process View)**

`main.py` 스크립트는 `--mode` 명령줄 인자를 사용하여 데이터 준비, 훈련, 평가 단계를 독립적으로 또는 순차적으로 실행할 수 있습니다.

* **데이터 준비 프로세스 (`--mode data`)**  
  1. `main.py`가 `config.yaml`을 로드하고 `--mode data` 인자를 확인합니다.
  2. `DataManager`가 `config`의 `data.mode` 설정(예: 'synthetic')을 바탕으로 가상 간섭 무늬 이미지와 Ground-Truth를 생성합니다.
  3. 생성된 데이터는 `data/processed/` 경로에 `coords.npy`, `interferograms_target.npy`, `image_dims.npy` 파일로 저장됩니다.

* **훈련 프로세스 (`--mode train`)**  
  1. `main.py`가 `config.yaml`을 로드하고 `--mode train` 인자를 확인합니다.
  2. `src/train_pipeline.py`의 `run_training` 함수가 호출됩니다.
  3. `run_training` 함수는 `data/processed/` 경로에서 처리된 데이터를 불러와 `DataLoader`를 생성합니다.
  4. `Scaled_cPIKAN_Model`, `HelmholtzLoss`, `Trainer` 객체가 생성됩니다.
  5. `Trainer`가 훈련 루프를 시작하여 모델을 훈련합니다.
  6. 훈련 완료 후, `Trainer`가 학습된 모델 가중치를 `checkpoints/model_checkpoint.pth` 파일로 저장합니다.

* **평가 프로세스 (`--mode evaluate`)**  
  1. `main.py`가 `config.yaml`을 로드하고 `--mode evaluate` 인자를 확인합니다.
  2. `src/evaluate_pipeline.py`의 `run_evaluation` 함수가 호출됩니다.
  3. `run_evaluation` 함수는 `checkpoints/model_checkpoint.pth`에서 훈련된 모델 가중치를 로드합니다.
  4. `data/processed/` 경로에서 처리된 데이터를 불러와 평가에 필요한 `ground_truth` 및 `image_dims` 정보를 얻습니다.
  5. 로드된 모델을 사용하여 예측을 수행하고, `evaluate_and_visualize` 함수를 통해 예측된 위상과 실제 위상을 시각화합니다.

* **전체 프로세스 (`--mode all` 또는 인자 없음)**  
  위의 데이터 준비, 훈련, 평가 프로세스를 순차적으로 실행합니다.

### **4\. 물리 뷰 (Physical View)**

시스템이 배포되고 실행될 하드웨어 및 소프트웨어 환경을 설명합니다.

* **하드웨어 (Hardware)**  
  * **개발/훈련 머신:**  
    * **GPU:** **NVIDIA GPU (CUDA 지원 필수)**. KAN/PIKAN 계열은 MLP보다 훈련 속도가 느릴 수 있으므로 RTX 30/40 시리즈, A100 등 고성능 GPU가 중요합니다.  
    * **RAM:** 32GB 이상 권장 (고해상도 이미지 및 대규모 콜로케이션 포인트 처리).  
    * **Storage:** 빠른 데이터 I/O를 위한 SSD.  
* **소프트웨어 (Software)**  
  * **OS:** Linux (Ubuntu 22.04 권장) 또는 Windows 11 (WSL2 환경 권장).  
  * **IDE:** Visual Studio Code.  
  * **런타임:** Python 3.9+ (Conda 또는 Venv를 통한 가상 환경 구성).  
  * **핵심 드라이버/툴킷:** NVIDIA Driver, CUDA Toolkit, cuDNN (PyTorch의 GPU 연산에 필수).