# Context7 사용 가이드 (Usage Guidelines)

**목표**: 코드 분석, 수정, 새로 작성 시 항상 최신 라이브러리 문서를 참조하여 정확한 구현 보장

---

## 📋 기본 원칙

### 🔴 필수 사용 규칙
코드 작업 전 **반드시** Context7로 관련 라이브러리 문서를 먼저 확인합니다.

**언제 사용하나요?**
1. ✅ 기존 코드 분석 및 이해
2. ✅ 코드 수정 또는 리팩토링
3. ✅ 새로운 코드 작성
4. ✅ API 사용법이 불확실할 때
5. ✅ 에러 디버깅 및 문제 해결
6. ✅ 성능 최적화
7. ✅ 새로운 기능 추가

### 🟢 워크플로우

```
코드 작업 요청
    ↓
1️⃣ 필요한 라이브러리 파악
    ↓
2️⃣ mcp_context7_resolve-library-id("라이브러리명")
    ↓
3️⃣ mcp_context7_get-library-docs(libraryID, query="구체적 질문")
    ↓
4️⃣ 문서 내용 확인 및 이해
    ↓
5️⃣ 코드 분석/작성/수정
    ↓
6️⃣ 테스트 실행 (python -m unittest discover tests)
```

---

## 🎯 라이브러리별 사용 시나리오

### PyTorch (src/models.py, src/loss.py, src/train.py)

**언제 검색하나요?**
- 새로운 신경망 레이어 추가
- 손실 함수 구현 또는 수정
- 그래디언트 계산 관련 작업
- 최적화 알고리즘 변경
- 학습률 스케줄러 설정

**검색 예시**:
```python
# 새 모델 작성 시
query="nn.Module subclassing best practices and __init__ super call"
query="nn.Parameter vs register_buffer differences"
query="forward method signature and return types"

# autograd 사용 시
query="torch.autograd.grad create_graph retain_graph differences"
query="computing hessian or second order derivatives"

# einsum 사용 시
query="torch.einsum notation examples for matrix multiplication"
query="einsum performance optimization tips"

# 옵티마이저 설정 시
query="Adam optimizer parameter groups and per-parameter options"
query="LBFGS closure function requirements and line search"
query="learning rate scheduler step timing and usage"
```

### SciPy (src/data.py)

**언제 검색하나요?**
- 새로운 샘플링 방법 추가
- Latin Hypercube 샘플링 수정
- 통계적 샘플링 알고리즘 이해

**검색 예시**:
```python
query="scipy.stats.qmc.LatinHypercube sampling parameters"
query="quasi-Monte Carlo sampling methods comparison"
query="Latin hypercube vs Sobol sequence differences"
```

### NumPy (src/data_generator.py, 전체)

**언제 검색하나요?**
- 배열 연산 최적화
- Broadcasting 규칙 확인
- 데이터 타입 변환
- 수학 연산 구현

**검색 예시**:
```python
query="numpy broadcasting rules multidimensional arrays"
query="numpy dtype conversion float32 float64"
query="vectorized operations vs loops performance"
query="numpy random sampling seed reproducibility"
```

### PIL/Pillow (src/data.py, src/data_generator.py)

**언제 검색하나요?**
- 이미지 로드/저장 수정
- 이미지 형식 변환
- 배열-이미지 변환

**검색 예시**:
```python
query="PIL Image.fromarray mode parameter for different dtypes"
query="converting numpy array to PIL Image data types"
query="PIL Image save format options and quality"
```

---

## 🆕 새로운 기능 추가 시

### 1. 새 라이브러리 사용
```python
# 예: matplotlib으로 시각화 추가
mcp_context7_resolve-library-id("matplotlib")
mcp_context7_get-library-docs(
    libraryID="/matplotlib/matplotlib",
    query="plotting subplots and saving figures programmatically"
)
```

### 2. 알려지지 않은 API 탐색
```python
# 예: PyTorch의 특정 기능 찾기
query="PyTorch gradient clipping methods"
query="PyTorch mixed precision training AMP usage"
query="PyTorch distributed training basics"
```

### 3. 디자인 패턴 확인
```python
query="PyTorch custom Dataset __getitem__ best practices"
query="PyTorch Lightning integration patterns"
query="PINN implementation patterns in PyTorch"
```

---

## � 자주 검색하는 쿼리 (Quick Reference)

이 섹션은 참고용입니다. **실제로는 작업에 맞춰 자유롭게 쿼리를 작성하세요.**

### 🔴 PyTorch

| 상황 | 쿼리 예시 |
|------|----------|
| 2차 미분 계산 | `torch.autograd.grad create_graph parameter for computing second-order derivatives` |
| einsum 이해 | `torch.einsum notation 'bik,oik->bo' explanation` |
| Adam 설정 | `torch.optim.Adam parameter groups and learning rate scheduling` |
| LBFGS 사용 | `torch.optim.LBFGS closure function requirements` |
| 스케줄러 | `ExponentialLR gamma parameter cumulative effect` |

### 🟡 SciPy

| 상황 | 쿼리 예시 |
|------|----------|
| LHS 샘플링 | `LatinHypercube sampling algorithm seed and bounds` |

### 🟡 NumPy

| 상황 | 쿼리 예시 |
|------|----------|
| 배열 연산 | `numpy broadcasting rules and dtype handling` |

### 🟡 PIL/Pillow

| 상황 | 쿼리 예시 |
|------|----------|
| 이미지 I/O | `PIL Image fromarray and save dtype conversion` |

---

## � 효과적인 쿼리 작성 팁

### ✅ 좋은 쿼리
- **구체적**: "torch.autograd.grad create_graph parameter"
- **맥락 포함**: "computing second-order derivatives for Laplacian"
- **문제 중심**: "how to fix LBFGS not converging"
- **비교 포함**: "Adam vs SGD for PINN training"

### ❌ 피해야 할 쿼리
- 너무 일반적: "pytorch"
- 너무 짧음: "grad"
- 모호함: "how to train"

---

## 🚀 실전 예시

### 예시 1: 새로운 손실 함수 추가

```python
# 작업: Huber 손실 추가
# 1. PyTorch 문서 확인
mcp_context7_resolve-library-id("PyTorch")
mcp_context7_get-library-docs(
    libraryID="/pytorch/pytorch",
    query="torch.nn.HuberLoss delta parameter and reduction options"
)

# 2. 구현 패턴 확인
mcp_context7_get-library-docs(
    libraryID="/pytorch/pytorch",
    query="custom loss function implementation nn.Module pattern"
)

# 3. 코드 작성
# 4. 테스트
```

### 예시 2: 코드 분석 (기존 ChebyKANLayer 이해)

```python
# 작업: ChebyKANLayer의 einsum 이해
# 1. einsum 문법 확인
mcp_context7_get-library-docs(
    libraryID="/pytorch/pytorch",
    query="torch.einsum notation examples 'bik,oik->bo' meaning"
)

# 2. 텐서 차원 이해
mcp_context7_get-library-docs(
    libraryID="/pytorch/pytorch",
    query="tensor dimension manipulation view reshape squeeze"
)

# 3. 분석 진행
```

### 예시 3: 버그 수정 (그래디언트 계산 오류)

```python
# 작업: "RuntimeError: grad can be implicitly created only for scalar outputs"
# 1. 에러 원인 조사
mcp_context7_get-library-docs(
    libraryID="/pytorch/pytorch",
    query="torch.autograd.grad scalar outputs requirement grad_outputs parameter"
)

# 2. 해결 방법 확인
mcp_context7_get-library-docs(
    libraryID="/pytorch/pytorch",
    query="computing gradients for non-scalar tensors sum mean reduction"
)

# 3. 수정 적용
```

---

## 📌 체크리스트

코드 작업 시 다음을 확인하세요:

- [ ] 관련 라이브러리 식별 완료
- [ ] Context7로 라이브러리 문서 검색 완료
- [ ] API 사용법 정확히 이해
- [ ] 예제 코드 참고
- [ ] 코드 작성/수정 완료
- [ ] 단위 테스트 작성/실행
- [ ] 통합 테스트 확인

---

## 🔗 추가 리소스

- PyTorch 공식 문서: https://pytorch.org/docs/
- SciPy 공식 문서: https://docs.scipy.org/
- NumPy 공식 문서: https://numpy.org/doc/
- Pillow 공식 문서: https://pillow.readthedocs.io/

**중요**: Context7을 통해 최신 문서를 확인하는 것이 항상 우선입니다!
