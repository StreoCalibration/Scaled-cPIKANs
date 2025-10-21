# Scaled-cPIKAN 알고리즘 구현 명세서

## 문서 목적

본 명세서는 Mostajeran과 Faroughi의 논문 "Scaled-cPIKANs: Domain Scaling in Chebyshev-based Physics-informed Kolmogorov-Arnold Networks"에 제시된 알고리즘을 **최대한 원본에 충실하게 구현**하기 위한 상세 구현 지침을 제공합니다.

이 문서는 기존 "기술 설계 문서"를 보완하며, 구현의 정확성을 검증하고 논문과의 일치성을 보장하는 데 중점을 둡니다.

---

## 1. 알고리즘 수학적 정의

### 1.1 체비쇼프 다항식 (Chebyshev Polynomials)

제1종 체비쇼프 다항식 $T_k(x)$는 다음과 같이 정의됩니다:

- **기저 케이스**:
  - $T_0(x) = 1$
  - $T_1(x) = x$

- **점화식** (k ≥ 1):
  - $T_{k+1}(x) = 2x \cdot T_k(x) - T_{k-1}(x)$

- **정의역**: $x \in [-1, 1]$ (필수 제약 조건)

- **주요 성질**:
  - 직교성: $\int_{-1}^{1} \frac{T_m(x) T_n(x)}{\sqrt{1-x^2}} dx = 0$ (m ≠ n)
  - 경계 값: $T_k(1) = 1$, $T_k(-1) = (-1)^k$
  - 진동 성질: 고차 다항식일수록 더 많은 진동 (고주파 표현 가능)

### 1.2 아핀 영역 스케일링 (Affine Domain Scaling)

물리적 도메인 $[x_{\min}, x_{\max}]$를 체비쇼프 다항식이 요구하는 표준 도메인 $[-1, 1]$로 매핑:

$$\hat{x} = 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1$$

**역변환**:
$$x = \frac{(\hat{x} + 1)}{2} \cdot (x_{\max} - x_{\min}) + x_{\min}$$

**검증 조건**:
- $x = x_{\min} \Rightarrow \hat{x} = -1$
- $x = x_{\max} \Rightarrow \hat{x} = 1$
- $x = \frac{x_{\min} + x_{\max}}{2} \Rightarrow \hat{x} = 0$

### 1.3 ChebyKAN 활성화 함수

각 엣지의 활성화 함수 $\phi_{j,i}(x_i)$는 체비쇼프 다항식의 선형 결합:

$$\phi_{j,i}(x_i) = \sum_{k=0}^{K} c_{j,i,k} \cdot T_k(x_i)$$

여기서:
- $c_{j,i,k}$: 학습 가능한 계수
- $K$: 체비쇼프 차수 (논문 권장: 3 또는 4)
- $x_i \in [-1, 1]$ (스케일링 후)

### 1.4 ChebyKAN 레이어 순방향 계산

입력 $\mathbf{x} \in \mathbb{R}^{\text{in\_features}}$에 대한 출력 $\mathbf{y} \in \mathbb{R}^{\text{out\_features}}$:

$$y_j = \sum_{i=1}^{\text{in\_features}} \phi_{j,i}(x_i) = \sum_{i=1}^{\text{in\_features}} \sum_{k=0}^{K} c_{j,i,k} \cdot T_k(x_i)$$

**효율적인 구현**: PyTorch einsum 사용
```python
# cheby_basis: (batch, in_features, K+1)
# cheby_coeffs: (out_features, in_features, K+1)
output = torch.einsum('bik,oik->bo', cheby_basis, cheby_coeffs)
```

### 1.5 Scaled-cPIKAN 전체 순방향 패스

1. **입력**: 물리적 좌표 $\mathbf{x}_{\text{phys}} \in [x_{\min}, x_{\max}]^d$
2. **아핀 스케일링**: $\mathbf{x}_{\text{scaled}} = \text{affine\_scale}(\mathbf{x}_{\text{phys}})$
3. **신경망 통과**:
   - 여러 ChebyKANLayer를 순차적으로 통과
   - 각 레이어 후 (마지막 제외): LayerNorm → Tanh
4. **출력**: 예측된 PDE 해 $u_{\text{pred}}(\mathbf{x}_{\text{phys}})$

### 1.6 물리 정보 손실 함수 (PINN Loss)

$$L_{\text{total}} = \lambda_{\text{pde}} \cdot L_{\text{pde}} + \lambda_{\text{bc}} \cdot L_{\text{bc}} + \lambda_{\text{ic}} \cdot L_{\text{ic}} + \lambda_{\text{data}} \cdot L_{\text{data}}$$

각 손실 항:

- **PDE 잔차 손실**:
  $$L_{\text{pde}} = \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left| \mathcal{F}[u](\mathbf{x}_i) \right|^2$$
  여기서 $\mathcal{F}$는 PDE 연산자 (자동 미분으로 계산)

- **경계 조건 손실**:
  $$L_{\text{bc}} = \frac{1}{N_{\text{bc}}} \sum_{i=1}^{N_{\text{bc}}} \left| u(\mathbf{x}_i^{\text{bc}}) - g(\mathbf{x}_i^{\text{bc}}) \right|^2$$

- **초기 조건 손실**:
  $$L_{\text{ic}} = \frac{1}{N_{\text{ic}}} \sum_{i=1}^{N_{\text{ic}}} \left| u(\mathbf{x}_i^{\text{ic}}, t_0) - u_0(\mathbf{x}_i^{\text{ic}}) \right|^2$$

- **데이터 손실** (역문제):
  $$L_{\text{data}} = \frac{1}{N_{\text{data}}} \sum_{i=1}^{N_{\text{data}}} \left| u(\mathbf{x}_i^{\text{data}}) - u_i^{\text{obs}} \right|^2$$

---

## 2. 구현 요구사항

### 2.1 ChebyKANLayer 구현 명세

**필수 구현 사항**:

1. **초기화 파라미터**:
   ```python
   def __init__(self, in_features: int, out_features: int, cheby_order: int)
   ```
   - `cheby_coeffs`: shape = `(out_features, in_features, cheby_order + 1)`
   - 초기화: Kaiming uniform (논문 기본 설정)

2. **체비쇼프 다항식 계산**:
   ```python
   # 반드시 리스트로 구축하여 in-place 연산 방지
   cheby_polys = []
   cheby_polys.append(torch.ones_like(x))  # T_0
   if cheby_order > 0:
       cheby_polys.append(x)  # T_1
   for k in range(1, cheby_order):
       T_next = 2 * x * cheby_polys[-1] - cheby_polys[-2]
       cheby_polys.append(T_next)
   cheby_basis = torch.stack(cheby_polys, dim=-1)
   ```

3. **Einsum 계산**:
   ```python
   output = torch.einsum('bik,oik->bo', cheby_basis, self.cheby_coeffs)
   ```
   - 인덱스 의미: b=batch, i=in_features, k=cheby_order, o=out_features
   - 이 시그니처는 논문 구현의 핵심이므로 변경 금지

4. **입력 검증**:
   ```python
   # Forward 시작 시 검증 (디버그 모드)
   assert torch.all((x >= -1.0) & (x <= 1.0)), "Input must be in [-1, 1]"
   ```

### 2.2 Scaled_cPIKAN 모델 명세

**필수 구현 사항**:

1. **초기화**:
   ```python
   def __init__(self, 
                layers_dims: list[int],
                cheby_order: int,
                domain_min: torch.Tensor,
                domain_max: torch.Tensor)
   ```

2. **아핀 스케일링 메서드**:
   ```python
   def _affine_scale(self, x: torch.Tensor) -> torch.Tensor:
       return 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
   ```
   - `domain_min`, `domain_max`는 버퍼로 등록 (학습되지 않음)
   - **순방향 패스의 첫 번째 연산**으로 반드시 호출

3. **네트워크 구조**:
   ```python
   for i in range(len(layers_dims) - 1):
       self.network.append(ChebyKANLayer(layers_dims[i], layers_dims[i+1], cheby_order))
       if i < len(layers_dims) - 2:  # 마지막 레이어 제외
           self.network.append(nn.LayerNorm(layers_dims[i+1]))
           self.network.append(nn.Tanh())
   ```

### 2.3 PhysicsInformedLoss 명세

**필수 구현 사항**:

1. **초기화**:
   ```python
   def __init__(self,
                pde_residual_fn: callable,
                bc_fns: list[callable],
                ic_fns: list[callable] = None,
                data_loss_fn: callable = None,
                loss_weights: dict = None)
   ```

2. **손실 계산**:
   - 각 손실 항을 MSE로 계산
   - 타겟은 항상 `torch.zeros_like()` (PDE 잔차, BC/IC 오차는 0이 목표)
   - 가중치 적용 후 합산

3. **자동 미분 사용**:
   ```python
   # PDE 잔차 예시 (Helmholtz 방정식)
   u = model(x_pde)
   u_xx = torch.autograd.grad(
       torch.autograd.grad(u.sum(), x_pde, create_graph=True)[0][:, 0].sum(),
       x_pde, create_graph=True
   )[0][:, 0]
   residual = u_xx + k**2 * u  # Should be 0
   ```

### 2.4 Trainer 명세 (2단계 최적화)

**1단계: Adam 최적화**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

for epoch in range(adam_epochs):  # 20,000 ~ 50,000
    optimizer.zero_grad()
    loss, loss_dict = loss_fn(model, ...)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

**2단계: L-BFGS 최적화**
```python
optimizer = torch.optim.LBFGS(
    model.parameters(),
    max_iter=20,
    history_size=100,
    line_search_fn="strong_wolfe"
)

def closure():
    optimizer.zero_grad()
    loss, _ = loss_fn(model, ...)
    loss.backward()
    return loss

for _ in range(lbfgs_epochs):  # 보통 1-10 스텝
    optimizer.step(closure)
```

---

## 3. 하이퍼파라미터 명세

### 3.1 논문 권장 vs 현재 구현

| 하이퍼파라미터 | 논문 권장값 | 현재 구현 | 상태 | 비고 |
|---|---|---|---|---|
| **네트워크 구조** | [2, 32, 32, 32, 1] | 예제마다 다름 | ⚠️ 통일 필요 | 2D 문제 기준 |
| **체비쇼프 차수 K** | 3 ~ 4 | 예제마다 다름 | ⚠️ 통일 필요 | K=3이 기본 권장 |
| **Adam 학습률** | 1e-3 | 1e-3 | ✅ 일치 | |
| **Adam 에포크** | 20,000 ~ 50,000 | 예제마다 다름 | ⚠️ 조정 필요 | 빠른 테스트용으로 현재 적음 |
| **학습률 스케줄러** | Exponential (γ≈0.9995) | 미사용 | ❌ 추가 필요 | 수렴 안정성 향상 |
| **L-BFGS max_iter** | 20 | 20 | ✅ 일치 | |
| **L-BFGS history_size** | 100 | 100 | ✅ 일치 | |
| **손실 가중치 λ_pde** | 1.0 | 1.0 | ✅ 일치 | |
| **손실 가중치 λ_bc** | 1.0 ~ 10.0 | 1.0 | ⚠️ 조정 가능 | 문제 의존적 |
| **콜로케이션 포인트** | Latin Hypercube | Latin Hypercube | ✅ 일치 | |

### 3.2 권장 설정 (논문 재현)

**1D/2D 문제**:
```python
model = Scaled_cPIKAN(
    layers_dims=[2, 32, 32, 32, 1],  # 2D 입력 → 1D 출력
    cheby_order=3,
    domain_min=torch.tensor([x_min, y_min]),
    domain_max=torch.tensor([x_max, y_max])
)

trainer.train(
    adam_epochs=20000,
    adam_lr=1e-3,
    lbfgs_epochs=5,
    use_scheduler=True,  # Exponential decay
    loss_weights={'pde': 1.0, 'bc': 1.0, 'ic': 1.0}
)
```

---

## 4. 훈련 프로토콜 (상세 단계)

### 4.1 데이터 준비

1. **콜로케이션 포인트 생성**:
   ```python
   from src.data import LatinHypercubeSampler
   
   # PDE 도메인 내부
   pde_sampler = LatinHypercubeSampler(
       n_points=10000,
       domain_min=[x_min, y_min],
       domain_max=[x_max, y_max]
   )
   pde_points = pde_sampler.sample()
   
   # 경계 조건 (4개 경계)
   bc_points_left = ...  # x = x_min
   bc_points_right = ... # x = x_max
   bc_points_bottom = ...
   bc_points_top = ...
   ```

2. **포인트 검증**:
   - PDE 포인트가 도메인 내부에 있는지 확인
   - 경계 포인트가 정확히 경계 위에 있는지 확인

### 4.2 모델 초기화

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Scaled_cPIKAN(
    layers_dims=[2, 32, 32, 32, 1],
    cheby_order=3,
    domain_min=torch.tensor([x_min, y_min], device=device),
    domain_max=torch.tensor([x_max, y_max], device=device)
).to(device)

# 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

### 4.3 손실 함수 정의

```python
def pde_residual_fn(model, points):
    # PDE 잔차 계산 (자동 미분 사용)
    points.requires_grad_(True)
    u = model(points)
    # ... 도함수 계산 ...
    return residual

def bc_fn(model, points):
    # 경계 조건 오차
    u_pred = model(points)
    u_bc = boundary_value(points)
    return u_pred - u_bc

loss_fn = PhysicsInformedLoss(
    pde_residual_fn=pde_residual_fn,
    bc_fns=[bc_fn_left, bc_fn_right, bc_fn_bottom, bc_fn_top],
    loss_weights={'pde': 1.0, 'bc': 1.0}
)
```

### 4.4 훈련 실행

```python
trainer = Trainer(model, loss_fn)

history = trainer.train(
    pde_points=pde_points,
    bc_points_dicts=[
        {'points': bc_points_left},
        {'points': bc_points_right},
        {'points': bc_points_bottom},
        {'points': bc_points_top}
    ],
    adam_epochs=20000,
    lbfgs_epochs=5,
    adam_lr=1e-3,
    log_interval=1000
)
```

### 4.5 결과 검증

```python
# 테스트 그리드에서 평가
test_grid = create_test_grid(resolution=100)
u_pred = model(test_grid).detach()

# 분석 해와 비교 (가능한 경우)
u_exact = analytical_solution(test_grid)
relative_l2_error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
print(f"Relative L2 Error: {relative_l2_error:.6e}")
```

---

## 5. 구현 검증 (Validation)

### 5.1 단위 테스트 (Unit Tests)

#### 5.1.1 체비쇼프 다항식 테스트

```python
def test_chebyshev_polynomials():
    """체비쇼프 다항식이 올바르게 계산되는지 검증"""
    x = torch.linspace(-1, 1, 100)
    
    layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=4)
    
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    
    # 체비쇼프 기저 추출 (layer의 forward에서)
    # ... 검증 로직 ...
    
    assert torch.allclose(T_0, torch.ones_like(x), atol=1e-6)
    assert torch.allclose(T_1, x, atol=1e-6)
    assert torch.allclose(T_2, 2*x**2 - 1, atol=1e-6)
```

#### 5.1.2 아핀 스케일링 테스트

```python
def test_affine_scaling():
    """아핀 스케일링이 올바르게 동작하는지 검증"""
    domain_min = torch.tensor([0.0, 0.0])
    domain_max = torch.tensor([10.0, 5.0])
    
    model = Scaled_cPIKAN([2, 4, 1], 3, domain_min, domain_max)
    
    # 경계 테스트
    x_min = torch.tensor([[0.0, 0.0]])
    x_max = torch.tensor([[10.0, 5.0]])
    x_mid = torch.tensor([[5.0, 2.5]])
    
    scaled_min = model._affine_scale(x_min)
    scaled_max = model._affine_scale(x_max)
    scaled_mid = model._affine_scale(x_mid)
    
    assert torch.allclose(scaled_min, torch.tensor([[-1.0, -1.0]]), atol=1e-6)
    assert torch.allclose(scaled_max, torch.tensor([[1.0, 1.0]]), atol=1e-6)
    assert torch.allclose(scaled_mid, torch.tensor([[0.0, 0.0]]), atol=1e-6)
```

### 5.2 통합 테스트 (Integration Tests)

#### 5.2.1 간단한 PDE 테스트 (Poisson 방정식)

```python
def test_poisson_equation_1d():
    """1D Poisson 방정식: u''(x) = -1, u(0)=0, u(1)=0
    분석 해: u(x) = x(1-x)/2"""
    
    # 모델, 손실, 훈련 설정
    # ...
    
    # 훈련
    trainer.train(adam_epochs=5000, lbfgs_epochs=2)
    
    # 검증
    x_test = torch.linspace(0, 1, 100).unsqueeze(1)
    u_pred = model(x_test).detach()
    u_exact = x_test * (1 - x_test) / 2
    
    error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    assert error < 1e-3, f"Error {error} exceeds tolerance"
```

### 5.3 벤치마크 재현 (Benchmark Reproduction)

#### 5.3.1 Helmholtz 방정식

논문의 Table 1 결과 재현:
- 1D Helmholtz: $u''(x) + k^2 u(x) = 0$, $k = 4\pi$
- 목표: Relative L2 Error < 1e-4

```python
def benchmark_helmholtz_1d():
    k = 4 * np.pi
    # ... 설정 ...
    
    error = train_and_evaluate()
    
    # 논문 결과와 비교
    paper_error = 5.2e-5  # 논문 Table 1
    assert error < 1e-4, f"Error {error} does not match paper quality"
    print(f"Our error: {error:.2e}, Paper error: {paper_error:.2e}")
```

---

## 6. 현재 구현 상태 분석

### 6.1 올바르게 구현된 항목 ✅

1. **ChebyKANLayer**:
   - 체비쇼프 다항식 점화식 정확
   - Einsum 연산 정확
   - 파라미터 초기화 적절

2. **Scaled_cPIKAN**:
   - 아핀 스케일링 메서드 정확
   - 네트워크 구조 (LayerNorm + Tanh) 적절
   - 버퍼 등록 정확

3. **PhysicsInformedLoss**:
   - 다중 손실 항 결합 로직 정확
   - MSE 계산 정확

4. **Trainer**:
   - 2단계 최적화 구조 정확
   - L-BFGS 설정 (max_iter, history_size) 정확

### 6.2 개선이 필요한 항목 ⚠️

1. **학습률 스케줄러 부재**:
   - 논문에서는 Exponential decay 사용
   - 현재 구현에는 스케줄러 없음
   - **해결**: Trainer에 스케줄러 옵션 추가

2. **하이퍼파라미터 불일치**:
   - 예제별로 layers_dims, cheby_order가 다름
   - 논문 권장 설정으로 통일 필요
   - **해결**: 기본값을 논문 권장값으로 설정

3. **입력 범위 검증 부재**:
   - ChebyKANLayer가 [-1, 1] 범위 외 입력 받을 때 경고 없음
   - **해결**: assert 또는 경고 추가

4. **동적 손실 가중치 미구현**:
   - 논문에서 언급한 GradNorm, NTK 기반 가중치 조정 없음
   - **해결**: 고급 기능으로 추후 추가 가능

### 6.3 누락된 기능 ❌

1. **학습률 스케줄러**:
   ```python
   # 추가 필요
   scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
   ```

2. **손실 가중치 자동 조정**:
   - GradNorm 또는 uncertainty weighting
   - 고급 기능, 필수는 아님

---

## 7. 구현 로드맵

### Phase 1: 핵심 개선 (1-2주)

- [ ] Trainer에 학습률 스케줄러 추가
- [ ] 모든 예제의 기본 하이퍼파라미터를 논문 권장값으로 통일
- [ ] ChebyKANLayer에 입력 범위 검증 추가
- [ ] 단위 테스트 작성 (체비쇼프, 아핀 스케일링)

### Phase 2: 검증 및 벤치마킹 (2-3주)

- [ ] Helmholtz 방정식 벤치마크 재현
- [ ] Allen-Cahn 방정식 벤치마크 재현
- [ ] 상대 L2 오차가 논문과 일치하는지 확인
- [ ] 통합 테스트 작성

### Phase 3: 고급 기능 (선택 사항)

- [ ] 동적 손실 가중치 (GradNorm)
- [ ] 적응형 콜로케이션 포인트 샘플링
- [ ] 3D 문제로 확장

---

## 8. 참고 자료

### 8.1 핵심 방정식 요약

```
아핀 스케일링:     x̂ = 2(x - x_min)/(x_max - x_min) - 1
체비쇼프 점화식:    T_{k+1}(x) = 2x·T_k(x) - T_{k-1}(x)
ChebyKAN 출력:     y_j = Σ_i Σ_k c_{j,i,k}·T_k(x_i)
PINN 손실:        L = λ_pde·L_pde + λ_bc·L_bc + λ_ic·L_ic
```

### 8.2 디버깅 체크리스트

훈련이 잘 안될 때:

1. [ ] 입력이 아핀 스케일링을 거쳤는가?
2. [ ] 스케일링 후 값이 [-1, 1] 범위인가?
3. [ ] PDE 잔차 계산에 `create_graph=True`를 사용했는가?
4. [ ] 경계 조건이 올바르게 정의되었는가?
5. [ ] 손실 가중치가 적절한가? (BC가 너무 약하면 증가)
6. [ ] 학습률이 적절한가? (너무 크면 발산)
7. [ ] 콜로케이션 포인트가 충분한가? (최소 1000개 이상)

---

## 부록 A: 코드 스니펫

### A.1 완전한 Helmholtz 예제

```python
import torch
import numpy as np
from src.models import Scaled_cPIKAN
from src.loss import PhysicsInformedLoss
from src.train import Trainer
from src.data import LatinHypercubeSampler

# 문제 정의: u''(x) + k^2 u(x) = 0, x ∈ [0, 1]
# BC: u(0) = 0, u(1) = 0
# 분석 해: u(x) = sin(k·x)

k = 4 * np.pi
domain_min = torch.tensor([0.0])
domain_max = torch.tensor([1.0])

# 모델
model = Scaled_cPIKAN(
    layers_dims=[1, 32, 32, 32, 1],
    cheby_order=3,
    domain_min=domain_min,
    domain_max=domain_max
)

# 데이터
pde_sampler = LatinHypercubeSampler(10000, [0.0], [1.0])
pde_points = pde_sampler.sample()
bc_left = torch.tensor([[0.0]])
bc_right = torch.tensor([[1.0]])

# 손실 함수
def pde_residual(model, points):
    points.requires_grad_(True)
    u = model(points)
    u_x = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0]
    return u_xx + k**2 * u

def bc_fn(model, points):
    return model(points)

loss_fn = PhysicsInformedLoss(
    pde_residual_fn=pde_residual,
    bc_fns=[bc_fn, bc_fn],
    loss_weights={'pde': 1.0, 'bc': 10.0}
)

# 훈련
trainer = Trainer(model, loss_fn)
history = trainer.train(
    pde_points=pde_points,
    bc_points_dicts=[{'points': bc_left}, {'points': bc_right}],
    adam_epochs=20000,
    lbfgs_epochs=5,
    adam_lr=1e-3
)

# 평가
x_test = torch.linspace(0, 1, 200).unsqueeze(1)
u_pred = model(x_test).detach()
u_exact = torch.sin(k * x_test)
error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
print(f"Relative L2 Error: {error:.6e}")
```

---

## 결론

본 명세서는 Scaled-cPIKAN 알고리즘을 논문에 최대한 충실하게 구현하기 위한 상세 지침을 제공합니다. 

**핵심 원칙**:
1. 체비쇼프 다항식 계산의 정확성
2. 아핀 스케일링의 필수적 적용
3. 2단계 최적화 전략 준수
4. 논문 권장 하이퍼파라미터 사용

현재 구현은 대부분 올바르나, 학습률 스케줄러 추가와 하이퍼파라미터 통일이 필요합니다. Phase 1 로드맵을 따라 개선하면 논문과 일치하는 결과를 얻을 수 있습니다.
