<!--
Scaled-cPIKANs 프로젝트에서 작업하는 AI 코딩 에이전트를 위한 지침.
짧고 구체적이며 저장소 전용 패턴에 집중할 것.
-->
# Scaled-cPIKAN — Copilot 지침 (간결판)

목표: 프로젝트 아키텍처, 핵심 워크플로우, 코드 패턴, 정확한 명령을 요약하여 
AI 코딩 에이전트가 즉시 생산적으로 작업할 수 있도록 지원.

- 전체 구조
  - 이 저장소는 PyTorch에서 Scaled-cPIKAN (Chebyshev 기반 PINNs)을 구현합니다.
  - 주요 구성 요소:
    - `src/models.py`: 핵심 모델 — `Scaled_cPIKAN`, `ChebyKANLayer`, 표준 `UNet`.
      cKAN 레이어는 Chebyshev 다항식을 계산하고 `torch.einsum`을 사용하여 계수를 결합합니다.
    - `src/loss.py`: 물리 기반 손실(PDE 잔차, BC/IC 항) 및 UNet/PINN 재구성 손실.
      손실 클래스는 스칼라 손실과 메트릭 딕셔너리를 반환합니다(예: UNet/PinntLoss 클래스의 `metrics`).
    - `src/data.py`: 샘플러/데이터셋: `LatinHypercubeSampler`, `PinnPatchDataset`,
      `WaferPatchDataset`. 패치는 실시간으로 추출되며 (C,H,W) 입력을 기대합니다.
    - `examples/*.py`: 실행 가능한 파이프라인 및 데모. `run_pipeline.py`는
      전체 사전학습→미세조정 파이프라인이며, `solve_*` 스크립트는 작은 데모입니다.

- 프로젝트 특화 패턴 & 제약사항
  - PINN을 위한 2단계 최적화: Adam 사전학습 후 L-BFGS 미세조정.
    `src/train.py` (Trainer) 및 README의 예시 출력 참조.
  - 도메인 스케일링: cKAN 입력은 [-1, 1]로 아핀 변환됩니다.
    `src/models.py`와 `src/data.py`의 `_affine_scale` / `affine_scale` 헬퍼 함수 참조.
  - Chebyshev 기저 구현: `ChebyKANLayer`는 T_k(x)를 반복적으로 구축하고
    (batch, in_features, K+1) 텐서로 스택합니다. einsum 사용: "bik,oik->bo".
  - PINN 데이터셋은 일반적으로 batch_size=1을 사용합니다(전체 패치/전체 그리드 PINN).
  - 실제 데이터 레이아웃: `real_data/train/sample_xxx/{bucket_*.bmp,ground_truth.npy}`.
    예시 및 `run_pipeline.py`는 `sample_*` 디렉토리를 기대합니다.

- 일반적인 개발자 워크플로우 (실행 가능한 명령)
  - 설치: pip install -r requirements.txt (Python 3.8+ 이상 사용,
    GPU 필요 시 CUDA PyTorch와 일치).
  - 전체 파이프라인 실행(합성 사전학습 데이터 생성, 선택적으로 미세조정 데이터 생성,
    사전학습, 미세조정):
    - `python examples/run_pipeline.py --help`로 플래그 확인.
    - 일반적인 실행: `python examples/run_pipeline.py` (기본값은
      `synthetic_data/train`과 `real_data/train` 사용).
  - 빠른 예시: 1D Helmholtz 데모 — `python examples/solve_helmholtz_1d.py`.
  - 합성 버킷 이미지 생성: `python examples/generate_bucket_data.py`.
  - 테스트: `python -m unittest discover tests`로 단위 테스트 실행.

- 코드 편집: 리뷰어와 테스트에 중요한 사항
  - `ChebyKANLayer`의 einsum 시그니처 유지. 차원/순서 변경 시
    forward와 테스트에서 주의 깊게 업데이트 필요.
  - 손실 가중치나 메트릭 업데이트 시, `src/loss.py`와
    가중치를 표시하는 모든 예시 플래그 업데이트(예: `--smoothness-weight`).
  - 데이터 형식 기대값: 함수는 float32 numpy 배열 또는 torch 텐서
    (docstring에 설명된 형태, 예: 버킷 입력 (C,H,W))를 기대합니다.

- 예시/테스트를 찾을 위치
  - `examples/run_pipeline.py` — 정규 종단 간 파이프라인 및 CLI 플래그.
  - `examples/solve_reconstruction_pinn.py`, `solve_reconstruction_from_buckets.py` —
    재구성 작업을 위한 현실적인 PINN 실행.
  - `tests/` — 소규모 단위/통합 테스트. `python -m unittest discover tests` 실행.

- AI 에이전트를 위한 빠른 편집 규칙
  - 작고 국소화된 변경 선호. 편집 후 단위 테스트 실행.
  - 모든 호출 사이트와 테스트를 업데이트하지 않는 한 `src/models.py`, `src/loss.py`, `src/data.py`의 
    공개 함수 시그니처 유지.
  - 훈련 동작 변경 시(최적화기, 스케줄러, 에포크 수), 
    `examples` 및 README 스니펫 업데이트.

- MCP (Model Context Protocol) Tool Usage

  - **Sequential Thinking**: 모든 복잡한 작업 전에 `mcp_sequentialthi_sequentialthinking` 사용
  
  - **Context7 Usage**: 코드 **분석, 수정, 새로 작성** 시 반드시 라이브러리 문서 확인
    
    **필수 사용 시점**:
    1. 기존 코드 분석 및 이해 시
    2. 코드 수정 또는 리팩토링 시
    3. 새로운 코드 작성 시
    4. API 사용법이 불확실할 때
    5. 에러 디버깅 시
    
    | 파일/작업 | 필수 확인 API | 라이브러리 |
    |---------|-------------|-----------|
    | `src/models.py` | `torch.autograd.grad`, `torch.einsum`, `nn.Module` | PyTorch |
    | `src/loss.py` | `torch.autograd.grad` (create_graph=True) | PyTorch |
    | `src/train.py` | `torch.optim.Adam`, `torch.optim.LBFGS`, `ExponentialLR` | PyTorch |
    | `src/data.py` | `scipy.stats.qmc.LatinHypercube`, `PIL.Image` | SciPy, Pillow |
    | `src/data_generator.py` | NumPy broadcasting, `PIL.Image` | NumPy, Pillow |
    | 새 모델 작성 | `nn.Module`, `nn.Parameter`, forward 패턴 | PyTorch |
    | 새 손실 함수 | `torch.autograd.grad`, 그래디언트 계산 | PyTorch |
    | 데이터 로더 작성 | `Dataset`, `DataLoader`, 샘플링 | PyTorch, SciPy |
    
    **워크플로 (코드 작업 전 필수)**:
    ```
    1. 작업 유형 파악 (분석/수정/새로작성)
    2. 위 표에서 관련 라이브러리 확인
    3. mcp_context7_resolve-library-id("라이브러리명")
    4. mcp_context7_get-library-docs(libraryID, query="API명 또는 패턴")
    5. 문서 확인 후 코드 분석/작성/수정
    6. python -m unittest discover tests
    ```
    
    **예시**:
    - 새 PINN 모델 작성 → PyTorch 문서에서 `nn.Module`, `nn.Parameter`, `forward` 패턴 확인
    - 손실 함수 수정 → PyTorch 문서에서 `torch.autograd.grad` 옵션 확인
    - 코드 분석 → 사용된 API의 정확한 동작 방식 확인
    
    **필수 쿼리 목록**: `.github/CONTEXT7_QUERIES.md` 참조

If anything above is unclear or you need a deeper section (e.g. model internals
or loss math), tell me which file or function and I'll expand the instruction.

---

## 한글 작성 규칙

**중요**: 이 프로젝트의 모든 대답과 문서는 **한글을 위주로 작성**합니다.

- **대답(응답) 작성**: AI 에이전트의 모든 응답은 한글로 작성합니다.
  - 기술 용어나 함수명, 변수명 등은 원문(영문)을 유지합니다 (예: `ChebyKANLayer`, `einsum`).
  - 설명, 지시사항, 피드백은 한글로 제공합니다.

- **문서(코드 주석, Docstring, README 등)**: 
  - 기존 코드의 docstring과 주석은 한글로 작성/유지합니다.
  - 새로운 코드 추가 시 주석과 docstring을 한글로 작성합니다.
  - 함수 서명(signature)과 변수명은 영문/원문을 유지합니다.

- **예시**:
  ```python
  def chebyshev_forward(x, coefficients):
      """
      Chebyshev 다항식을 이용한 정방향 계산.
      
      Args:
          x (torch.Tensor): 입력 텐서, 형태 (batch, in_features).
          coefficients (torch.Tensor): Chebyshev 계수, 형태 (out_features, in_features, K+1).
      
      Returns:
          torch.Tensor: 출력, 형태 (batch, out_features).
      """
      # Chebyshev 다항식 계산
      return torch.einsum("bik,oik->bo", x_cheby, coefficients)
  ```

- **우선순위**:
  1. 한글 설명과 지시사항
  2. 코드의 함수/변수명은 영문 유지
  3. 에러 메시지와 로깅도 한글 위주로 작성
  4. 기술 명칭(PDE, PINN, Chebyshev 등)은 원문 + 필요시 번역 괄호 추가
