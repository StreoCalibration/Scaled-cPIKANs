# Context7 필수 쿼리 (Essential Queries)

프로젝트 개발 시 반드시 확인해야 할 Context7 쿼리 목록

---

## 🔴 PyTorch (src/models.py, src/loss.py, src/train.py)

### torch.autograd.grad
```
Query: "torch.autograd.grad create_graph parameter for computing second-order derivatives"
When: 2차 미분 계산 시 (Laplacian 등)
```

### torch.einsum
```
Query: "torch.einsum notation 'bik,oik->bo' explanation"
When: ChebyKANLayer 수정 시
```

### torch.optim.Adam
```
Query: "torch.optim.Adam parameter groups and learning rate scheduling"
When: 옵티마이저 설정 변경 시
```

### torch.optim.LBFGS
```
Query: "torch.optim.LBFGS closure function requirements"
When: L-BFGS 미세조정 추가/수정 시
```

### torch.optim.lr_scheduler.ExponentialLR
```
Query: "ExponentialLR gamma parameter cumulative effect"
When: 학습률 스케줄 변경 시
```

---

## 🟡 SciPy (src/data.py)

### scipy.stats.qmc.LatinHypercube
```
Query: "LatinHypercube sampling algorithm seed and bounds"
When: 샘플링 로직 수정 시
```

---

## 🟡 NumPy (src/data_generator.py)

### Array Operations
```
Query: "numpy broadcasting rules and dtype handling"
When: 배열 연산 수정 시
```

---

## 🟡 PIL/Pillow (src/data.py, src/data_generator.py)

### Image I/O
```
Query: "PIL Image fromarray and save dtype conversion"
When: 이미지 저장/로드 수정 시
```

---

## 사용 방법

```python
# 1. 라이브러리 ID 확인
mcp_context7_resolve-library-id("PyTorch")

# 2. 문서 조회
mcp_context7_get-library-docs(
    context7CompatibleLibraryID="/pytorch/pytorch",
    query="위 쿼리 복사"
)
```
