<!--
Guidance for AI coding agents working on Scaled-cPIKANs.
Keep this short, concrete and focused on repository-specific patterns.
-->
# Scaled-cPIKAN — Copilot instructions (concise)

Goal: Help an AI coding agent be immediately productive by summarizing the
project architecture, key workflows, code patterns, and precise commands.

- Big picture
  - This repo implements Scaled-cPIKAN (Chebyshev-based PINNs) in PyTorch.
  - Major components:
    - `src/models.py`: core models — `Scaled_cPIKAN`, `ChebyKANLayer`, and a
      standard `UNet`. cKAN layers compute Chebyshev polynomials and use
      `torch.einsum` to combine coefficients.
    - `src/loss.py`: physics-informed loss (PDE residuals, BC/IC terms) and
      UNet/PINN reconstruction losses. Loss classes return both scalar loss and
      metric dicts (e.g., `metrics` on UNet/PinntLoss classes).
    - `src/data.py`: samplers/datasets: `LatinHypercubeSampler`, `PinnPatchDataset`,
      `WaferPatchDataset`. Patches are extracted on-the-fly; expect (C,H,W) inputs.
    - `examples/*.py`: runnable pipelines and demos. `run_pipeline.py` is the
      full pretrain→finetune pipeline; `solve_*` scripts are smaller demos.

- Project-specific patterns & constraints
  - Two-stage optimization for PINNs: Adam pre-training followed by L-BFGS
    fine-tuning. See `src/train.py` (Trainer) and example output in README.
  - Domain scaling: inputs to cKAN are affine-scaled into [-1, 1]. Look for
    `_affine_scale` / `affine_scale` helpers in `src/models.py` and `src/data.py`.
  - Chebyshev basis implementation: `ChebyKANLayer` builds T_k(x) iteratively
    and stacks them into a (batch, in_features, K+1) tensor; einsum is used:
    "bik,oik->bo".
  - PINN datasets commonly use batch_size=1 (full patch / full-grid PINN).
  - Real data layout: `real_data/train/sample_xxx/{bucket_*.bmp,ground_truth.npy}`.
    Examples and `run_pipeline.py` expect `sample_*` directories.

- Common developer workflows (commands you can run)
  - Install: pip install -r requirements.txt (use Python 3.8+ and match CUDA
    PyTorch if GPU is required).
  - Run full pipeline (generate synthetic pretrain data, optionally generate
    finetune data, pretrain, then finetune):
    - `python examples/run_pipeline.py --help` to inspect flags.
    - Typical run: `python examples/run_pipeline.py` (defaults use
      `synthetic_data/train` and `real_data/train`).
  - Quick example: 1D Helmholtz demo — `python examples/solve_helmholtz_1d.py`.
  - Generate synthetic bucket images: `python examples/generate_bucket_data.py`.
  - Tests: run the unit tests with `python -m unittest discover tests`.

- Code edits: what matters for reviewers/tests
  - Keep the einsum signature in `ChebyKANLayer` intact. Changing dims/ordering
    needs careful updates in the forward and tests.
  - When updating loss weighting or metrics, update both `src/loss.py` and
    any example flags that surface weights (e.g., `--smoothness-weight`).
  - Data format expectations: functions expect float32 numpy arrays or torch
    tensors with shapes described in docstrings (e.g., bucket inputs (C,H,W)).

- Where to look for examples / tests
  - `examples/run_pipeline.py` — canonical end-to-end pipeline and CLI flags.
  - `examples/solve_reconstruction_pinn.py`, `solve_reconstruction_from_buckets.py` —
    realistic PINN runs for reconstruction tasks.
  - `tests/` — small unit/integration tests. Run `python -m unittest discover tests`.

- Quick editing rules for AI agents
  - Prefer small, localized changes. Run unit tests after edits.
  - Preserve public function signatures in `src/models.py`, `src/loss.py`, and
    `src/data.py` unless updating all call sites and tests.
  - If changing training behavior (optimizers, schedulers, epoch counts),
    update `examples` and README snippets.

- MCP (Model Context Protocol) Tool Usage

  - **Sequential Thinking**: 모든 복잡한 작업 전에 `mcp_sequentialthi_sequentialthinking` 사용
  
  - **Context7 Usage**: 코드 수정 전 반드시 라이브러리 문서 확인
    
    | 파일 | 필수 확인 API | 라이브러리 |
    |------|-------------|-----------|
    | `src/models.py` | `torch.autograd.grad`, `torch.einsum` | PyTorch |
    | `src/loss.py` | `torch.autograd.grad` (create_graph=True) | PyTorch |
    | `src/train.py` | `torch.optim.Adam`, `torch.optim.LBFGS`, `ExponentialLR` | PyTorch |
    | `src/data.py` | `scipy.stats.qmc.LatinHypercube`, `PIL.Image` | SciPy, Pillow |
    | `src/data_generator.py` | NumPy broadcasting, `PIL.Image` | NumPy, Pillow |
    
    **워크플로**:
    ```
    1. 파일 수정 필요 → 위 표에서 라이브러리 확인
    2. mcp_context7_resolve-library-id("라이브러리명")
    3. mcp_context7_get-library-docs(libraryID, query="API명")
    4. 문서 확인 후 코드 작성
    5. python -m unittest discover tests
    ```
    
    **필수 쿼리 목록**: `.github/CONTEXT7_QUERIES.md` 참조

If anything above is unclear or you need a deeper section (e.g. model internals
or loss math), tell me which file or function and I'll expand the instruction.
