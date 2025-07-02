# scaled-cpikan-dhm 프로젝트

이 프로젝트는 디지털 홀로그래피 현미경(DHM) 데이터를 처리하고, cPIKAN 모델을 훈련하며, 결과를 평가하는 파이프라인을 제공합니다.

## 프로젝트 구조

```
scaled-cpikan-dhm/
├── configs/                  # 설정 파일
│   └── microlens_config.yaml
├── data/                     # 데이터 저장소
│   ├── interferograms/     # 실제 데이터 (더미)
│   └── processed/          # 처리된 데이터 (coords, interferograms_target, image_dims)
├── notebooks/                # Jupyter 노트북
├── src/                      # 소스 코드
│   ├── __init__.py
│   ├── data_pipeline.py    # 데이터 로드 및 전처리
│   ├── evaluate_pipeline.py# 모델 평가 및 시각화
│   ├── losses.py           # 손실 함수 정의
│   ├── models.py           # 모델 정의
│   ├── phantom_generator.py# 가상 데이터 생성
│   ├── train_pipeline.py   # 모델 훈련
│   └── trainer.py          # 훈련 루프 관리
├── .vscode/                  # VS Code 설정
├── main.py                   # 메인 실행 스크립트
└── requirements.txt          # Python 패키지 요구사항
```

## 시작하기

### 1. 가상 환경 설정 및 의존성 설치

프로젝트 루트 디렉토리(`scaled-cpikan-dhm`)에서 다음 명령어를 실행하여 가상 환경을 설정하고 필요한 라이브러리를 설치합니다.

```bash
# 프로젝트 디렉토리로 이동
cd F:\Source\Test\scaled-cpikan-dhm

# 가상 환경 생성
python -m venv .venv

# 가상 환경 활성화 (Windows)
.venv\Scripts\activate

# 가상 환경 활성화 (macOS/Linux)
# source .venv/bin/activate

# 필요한 라이브러리 설치
pip install -r requirements.txt
```

### 2. 프로젝트 실행

`main.py` 스크립트는 `--mode` 인자를 사용하여 다양한 파이프라인 단계를 실행할 수 있습니다.

#### 데이터 준비 모드 (`--mode data`)

가상 데이터를 생성하고, 훈련에 적합한 형태로 전처리하여 `data/processed/` 디렉토리에 저장합니다. 이 단계는 학습이나 평가를 시작하기 전에 반드시 한 번 실행되어야 합니다.

```bash
python main.py --mode data
```

#### 모델 학습 모드 (`--mode train`)

`data/processed/`에 저장된 데이터를 불러와 모델을 훈련합니다. 훈련된 모델 가중치는 `checkpoints/` 디렉토리에 저장됩니다.

```bash
python main.py --mode train
```

#### 모델 평가 모드 (`--mode evaluate`)

`checkpoints/`에 저장된 모델 가중치와 `data/processed/`에 저장된 데이터를 불러와 모델의 성능을 평가하고 시각화합니다.

```bash
python main.py --mode evaluate
```

#### 모든 단계 실행 모드 (`--mode all` 또는 인자 없음)

데이터 준비, 모델 학습, 모델 평가 단계를 순차적으로 실행합니다. `--mode` 인자를 지정하지 않으면 기본적으로 이 모드로 실행됩니다.

```bash
python main.py --mode all
# 또는
python main.py
```

## 설정

`configs/microlens_config.yaml` 파일을 수정하여 데이터 생성 파라미터, 훈련 하이퍼파라미터 등을 조정할 수 있습니다.
