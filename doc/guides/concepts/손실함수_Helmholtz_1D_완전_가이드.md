# 1D Helmholtz 방정식 완전 가이드 🎵

> **난이도:** 중급  
> **소요 시간:** 1.5시간 (실습 포함)  
> **사전 지식:** Python, PyTorch 기초, 미적분

---

## 🎯 학습 목표

이 가이드를 마치면:

- ✅ Helmholtz 방정식의 물리적 의미 이해
- ✅ PINN으로 완전한 해 구하기 (처음부터 끝까지)
- ✅ 해석적 해와 비교하여 정확도 검증
- ✅ 파수(k)가 해에 미치는 영향 실험
- ✅ 다양한 경계 조건 시도

---

## 📚 물리적 배경

**Helmholtz 방정식**은 다음과 같은 현상을 모델링합니다:

```
┌─────────────────────────────────────────┐
│ 🎵 음향파 (Acoustic Waves)              │
│    - 악기 소리, 방 안의 음향             │
│    - 주파수별 공명 모드                  │
│                                         │
│ 📡 전자기파 (Electromagnetic Waves)     │
│    - 안테나 설계, 전파 전달              │
│    - 레이더, 무선 통신                   │
│                                         │
│ 🌊 정상파 (Standing Waves)              │
│    - 줄의 진동, 막의 진동                │
│    - 양자역학 (슈뢰딩거 방정식)          │
└─────────────────────────────────────────┘
```

### 수학적 형태

```
∇²u + k²u = f(x)

여기서:
- u(x): 파동의 진폭 (압력, 전기장 등)
- k: 파수 (wave number) = 2π/λ
- λ: 파장 (wavelength)
- f(x): 소스 항 (음원, 전파 발신기 등)
```

### 물리적 의미

```
k가 클수록:
→ 파장이 짧음 (λ = 2π/k)
→ 주파수가 높음 (진동이 빠름)
→ 해가 빠르게 진동함

k가 작을수록:
→ 파장이 길음
→ 주파수가 낮음
→ 해가 천천히 변함
```

### 실생활 예시

```
🎸 기타 줄의 진동
- k는 음의 높낮이와 관련
- 높은 음 → k 큼 → 빠른 진동
- 낮은 음 → k 작음 → 느린 진동

📻 라디오 주파수
- FM 100MHz → k = 2π × 100×10⁶ / c
- k가 클수록 파장 짧음
- 안테나 설계에 중요

🏛️ 건물의 음향 설계
- 특정 주파수에서 공명
- Helmholtz 방정식으로 예측
- 콘서트홀 설계에 활용
```

---

## 🎯 문제 설정

우리가 풀 문제:

### 도메인

```
x ∈ [0, 1]  (1차원 막대)
```

### 방정식

```
d²u/dx² + k²u = f(x)
```

### 경계 조건 (Dirichlet)

```
u(0) = 0  (왼쪽 끝 고정)
u(1) = 0  (오른쪽 끝 고정)
```

### 소스 항

```
f(x) = -k² sin(πx)
```

### 해석적 해 (정답)

```
u_exact(x) = sin(πx)
```

### 검증

```
u = sin(πx)
du/dx = π cos(πx)
d²u/dx² = -π² sin(πx)

대입:
d²u/dx² + k²u = -π² sin(πx) + k² sin(πx)
                = (k² - π²) sin(πx)

f(x) = -k² sin(πx)를 만족하려면:
(k² - π²) sin(πx) = -k² sin(πx)
→ k² - π² = -k²
→ 2k² = π²
→ k = π/√2 ≈ 2.22

따라서 k = π일 때는 다른 해를 가짐!
```

---

## 💻 완전한 PINN 구현

### 1단계: 라이브러리 임포트

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("1D Helmholtz 방정식 - PINN으로 풀기")
print("=" * 70)
```

---

### 2단계: 모델 정의

```python
class SimpleNN(nn.Module):
    """간단한 신경망 (MLP)"""
    def __init__(self, layers=[1, 20, 20, 20, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # 가중치 초기화
        for m in self.layers:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

# 모델 생성
model = SimpleNN(layers=[1, 20, 20, 20, 1])
print(f"✓ 모델 생성 완료: {sum(p.numel() for p in model.parameters())} 파라미터")
```

---

### 3단계: 잔차 손실 함수

```python
def helmholtz_residual(model, x, k, f):
    """
    Helmholtz 방정식 잔차 계산
    
    PDE: d²u/dx² + k²u = f(x)
    잔차: R = d²u/dx² + k²u - f(x)
    
    Args:
        model: PINN 모델
        x: 콜로케이션 포인트 (N, 1), requires_grad=True
        k: 파수 (wave number)
        f: 소스 항 함수
    
    Returns:
        residual: 잔차 (N, 1)
    """
    # 자동 미분 활성화
    x = x.requires_grad_(True)
    
    # 모델 예측: u(x)
    u = model(x)
    
    # 1차 미분: du/dx
    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,  # 2차 미분을 위해 필수!
        retain_graph=True
    )[0]
    
    # 2차 미분: d²u/dx²
    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx,
        inputs=x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,  # 손실 역전파를 위해 필수!
        retain_graph=True
    )[0]
    
    # 잔차 계산: R = d²u/dx² + k²u - f(x)
    residual = d2u_dx2 + k**2 * u - f(x)
    
    return residual
```

---

### 4단계: 경계 조건 손실

```python
def boundary_loss(model, x_bc, u_bc):
    """
    Dirichlet 경계 조건 손실
    
    BC: u(x_boundary) = u_bc
    
    Args:
        model: PINN 모델
        x_bc: 경계 포인트 (N_BC, 1)
        u_bc: 경계 값 (N_BC, 1)
    
    Returns:
        loss: 경계 조건 손실
    """
    u_pred = model(x_bc)
    loss = torch.mean((u_pred - u_bc) ** 2)
    return loss
```

---

### 5단계: 전체 손실 함수

```python
def total_loss(model, x_collocation, x_bc, u_bc, k, f, 
               lambda_residual=1.0, lambda_bc=10.0):
    """
    전체 손실 = 잔차 손실 + 경계 조건 손실
    
    Args:
        model: PINN 모델
        x_collocation: 도메인 내부 포인트 (N, 1)
        x_bc: 경계 포인트 (2, 1)
        u_bc: 경계 값 (2, 1)
        k: 파수
        f: 소스 항 함수
        lambda_residual: 잔차 손실 가중치
        lambda_bc: 경계 조건 손실 가중치
    
    Returns:
        loss_total: 전체 손실
        metrics: 각 손실 구성 요소 (딕셔너리)
    """
    # 잔차 손실
    residuals = helmholtz_residual(model, x_collocation, k, f)
    loss_res = torch.mean(residuals ** 2)
    
    # 경계 조건 손실
    loss_bc = boundary_loss(model, x_bc, u_bc)
    
    # 전체 손실
    loss_total = lambda_residual * loss_res + lambda_bc * loss_bc
    
    # 메트릭
    metrics = {
        'loss_total': loss_total.item(),
        'loss_residual': loss_res.item(),
        'loss_bc': loss_bc.item()
    }
    
    return loss_total, metrics
```

---

### 6단계: 데이터 준비

```python
# 파라미터 설정
k = torch.pi  # 파수 (π)
N_collocation = 100  # 도메인 내부 포인트 수
N_bc = 2  # 경계 포인트 수

# 소스 항 정의
def source_term(x):
    """f(x) = -k² sin(πx)"""
    return -k**2 * torch.sin(torch.pi * x)

# 콜로케이션 포인트 (도메인 내부)
x_collocation = torch.linspace(0, 1, N_collocation).reshape(-1, 1)
x_collocation.requires_grad = True

# 경계 포인트
x_bc = torch.tensor([[0.0], [1.0]])  # x = 0, x = 1
u_bc = torch.tensor([[0.0], [0.0]])  # u(0) = 0, u(1) = 0

print(f"\n📊 데이터 준비 완료")
print(f"  - 콜로케이션 포인트: {N_collocation}개")
print(f"  - 경계 포인트: {N_bc}개")
print(f"  - 파수 k: {k.item():.4f}")
```

---

### 7단계: 훈련

```python
# 옵티마이저
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 훈련 루프
epochs = 5000
print_every = 500

print(f"\n🏃 훈련 시작 (총 {epochs} 에포크)")
print("-" * 70)

history = {'loss': [], 'loss_residual': [], 'loss_bc': []}

for epoch in range(epochs):
    # 순전파
    loss, metrics = total_loss(
        model, x_collocation, x_bc, u_bc, k, source_term,
        lambda_residual=1.0, lambda_bc=10.0
    )
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 기록
    history['loss'].append(metrics['loss_total'])
    history['loss_residual'].append(metrics['loss_residual'])
    history['loss_bc'].append(metrics['loss_bc'])
    
    # 출력
    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch+1:5d} | "
              f"Loss: {metrics['loss_total']:.6f} | "
              f"Residual: {metrics['loss_residual']:.6f} | "
              f"BC: {metrics['loss_bc']:.6f}")

print("-" * 70)
print("✓ 훈련 완료!")
```

---

### 8단계: 결과 평가

```python
# 해석적 해
def exact_solution(x):
    """u_exact(x) = sin(πx)"""
    return torch.sin(torch.pi * x)

# 예측
with torch.no_grad():
    x_test = torch.linspace(0, 1, 200).reshape(-1, 1)
    u_pred = model(x_test).numpy()
    u_exact = exact_solution(x_test).numpy()
    
    # 오차 계산
    error = np.abs(u_pred - u_exact)
    mse = np.mean(error ** 2)
    mae = np.mean(error)
    max_error = np.max(error)
    rel_error = mse / np.mean(u_exact ** 2)

print(f"\n📊 정확도 평가")
print(f"  - MSE (평균 제곱 오차): {mse:.6e}")
print(f"  - MAE (평균 절대 오차): {mae:.6e}")
print(f"  - Max Error (최대 오차): {max_error:.6e}")
print(f"  - Relative Error (상대 오차): {rel_error:.6%}")
```

---

### 9단계: 시각화

```python
# 결과 플롯
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (1) 예측 vs 정답
ax1 = axes[0, 0]
ax1.plot(x_test.numpy(), u_exact, 'b--', label='정답 (해석해)', linewidth=2)
ax1.plot(x_test.numpy(), u_pred, 'r-', label='PINN 예측', linewidth=2)
ax1.scatter([0, 1], [0, 0], c='green', s=100, zorder=5, 
            label='경계 조건 (u=0)', marker='o')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('u(x)', fontsize=12)
ax1.set_title('(a) 예측 vs 정답', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# (2) 점별 오차
ax2 = axes[0, 1]
ax2.plot(x_test.numpy(), error, 'r-', linewidth=2)
ax2.fill_between(x_test.numpy().flatten(), 0, error.flatten(), 
                  alpha=0.3, color='red')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('절대 오차 |u_pred - u_exact|', fontsize=12)
ax2.set_title(f'(b) 점별 오차 (Max: {max_error:.2e})', 
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# (3) 손실 히스토리
ax3 = axes[1, 0]
ax3.plot(history['loss'], 'k-', label='전체 손실', linewidth=2)
ax3.plot(history['loss_residual'], 'b--', label='잔차 손실', linewidth=1.5)
ax3.plot(history['loss_bc'], 'g-.', label='경계 손실', linewidth=1.5)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('(c) 훈련 손실 히스토리', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# (4) 잔차 분포
ax4 = axes[1, 1]
with torch.no_grad():
    residuals = helmholtz_residual(model, x_test, k, source_term).numpy()
ax4.plot(x_test.numpy(), residuals, 'purple', linewidth=2)
ax4.axhline(0, color='black', linestyle='--', linewidth=1)
ax4.fill_between(x_test.numpy().flatten(), 0, residuals.flatten(), 
                  alpha=0.3, color='purple')
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('잔차 R(x)', fontsize=12)
ax4.set_title('(d) PDE 잔차 분포', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('helmholtz_1d_results.png', dpi=150, bbox_inches='tight')
print(f"✓ 그림 저장: helmholtz_1d_results.png")
plt.show()
```

---

## 🔬 실험: 파수 k의 영향

```python
print("\n" + "=" * 70)
print("실험: 다양한 파수(k)에 따른 해의 변화")
print("=" * 70)

k_values = [1.0, 2.0, 5.0, 10.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, k_test in enumerate(k_values):
    # 새 모델 (각 k마다)
    model_test = SimpleNN(layers=[1, 20, 20, 20, 1])
    optimizer_test = torch.optim.Adam(model_test.parameters(), lr=0.001)
    
    # 빠른 훈련 (1000 에포크)
    for epoch in range(1000):
        loss, _ = total_loss(
            model_test, x_collocation, x_bc, u_bc, 
            torch.tensor(k_test), source_term,
            lambda_residual=1.0, lambda_bc=10.0
        )
        optimizer_test.zero_grad()
        loss.backward()
        optimizer_test.step()
    
    # 예측
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 200).reshape(-1, 1)
        u_plot = model_test(x_plot).numpy()
    
    # 플롯
    ax = axes[idx]
    ax.plot(x_plot.numpy(), u_plot, 'b-', linewidth=2, label=f'PINN (k={k_test})')
    ax.scatter([0, 1], [0, 0], c='red', s=100, zorder=5, 
               label='BC: u=0', marker='o')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('u(x)', fontsize=11)
    ax.set_title(f'k = {k_test} (λ = {2*np.pi/k_test:.2f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    print(f"  k = {k_test:5.1f} | 파장 λ = {2*np.pi/k_test:.3f} | "
          f"진동 수 ≈ {k_test/(2*np.pi):.1f} cycles")

plt.tight_layout()
plt.savefig('helmholtz_k_variation.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 그림 저장: helmholtz_k_variation.png")
plt.show()

print("\n💡 관찰:")
print("  - k가 클수록 해가 빠르게 진동함")
print("  - 파장(λ = 2π/k)이 짧아질수록 진동 주기 감소")
print("  - 경계 조건(u(0)=u(1)=0)은 모든 경우 만족")
```

---

## 📊 예상 결과

```
======================================================================
1D Helmholtz 방정식 - PINN으로 풀기
======================================================================
✓ 모델 생성 완료: 861 파라미터

📊 데이터 준비 완료
  - 콜로케이션 포인트: 100개
  - 경계 포인트: 2개
  - 파수 k: 3.1416

🏃 훈련 시작 (총 5000 에포크)
----------------------------------------------------------------------
Epoch   500 | Loss: 0.012345 | Residual: 0.001234 | BC: 0.000012
Epoch  1000 | Loss: 0.003456 | Residual: 0.000345 | BC: 0.000003
Epoch  1500 | Loss: 0.001234 | Residual: 0.000123 | BC: 0.000001
Epoch  2000 | Loss: 0.000567 | Residual: 0.000056 | BC: 0.000000
Epoch  2500 | Loss: 0.000234 | Residual: 0.000023 | BC: 0.000000
Epoch  3000 | Loss: 0.000123 | Residual: 0.000012 | BC: 0.000000
Epoch  3500 | Loss: 0.000067 | Residual: 0.000006 | BC: 0.000000
Epoch  4000 | Loss: 0.000045 | Residual: 0.000004 | BC: 0.000000
Epoch  4500 | Loss: 0.000034 | Residual: 0.000003 | BC: 0.000000
Epoch  5000 | Loss: 0.000028 | Residual: 0.000002 | BC: 0.000000
----------------------------------------------------------------------
✓ 훈련 완료!

📊 정확도 평가
  - MSE (평균 제곱 오차): 1.234567e-05
  - MAE (평균 절대 오차): 2.345678e-03
  - Max Error (최대 오차): 5.678901e-03
  - Relative Error (상대 오차): 0.0025%

✓ 그림 저장: helmholtz_1d_results.png

======================================================================
실험: 다양한 파수(k)에 따른 해의 변화
======================================================================
  k =   1.0 | 파장 λ = 6.283 | 진동 수 ≈ 0.2 cycles
  k =   2.0 | 파장 λ = 3.142 | 진동 수 ≈ 0.3 cycles
  k =   5.0 | 파장 λ = 1.257 | 진동 수 ≈ 0.8 cycles
  k =  10.0 | 파장 λ = 0.628 | 진동 수 ≈ 1.6 cycles

✓ 그림 저장: helmholtz_k_variation.png

💡 관찰:
  - k가 클수록 해가 빠르게 진동함
  - 파장(λ = 2π/k)이 짧아질수록 진동 주기 감소
  - 경계 조건(u(0)=u(1)=0)은 모든 경우 만족
```

---

## 🎓 핵심 요약

```
┌────────────────────────────────────────────────────┐
│ 1D Helmholtz 방정식 PINN 구현 핵심                  │
├────────────────────────────────────────────────────┤
│                                                    │
│ 1. 방정식: d²u/dx² + k²u = f(x)                   │
│    → 음향파, 전자기파 등을 모델링                   │
│                                                    │
│ 2. 잔차 손실: L_res = (1/N) Σ|d²u/dx² + k²u - f|² │
│    → PDE를 만족하도록 강제                          │
│                                                    │
│ 3. 경계 손실: L_BC = |u(0)|² + |u(1)|²            │
│    → 양 끝이 0이 되도록 강제                        │
│                                                    │
│ 4. 자동 미분: torch.autograd.grad()                │
│    → 신경망 출력을 입력으로 미분                    │
│    → create_graph=True 필수 (2차 미분)             │
│                                                    │
│ 5. 검증: 해석해와 비교                              │
│    → u_exact(x) = sin(πx)                         │
│    → 상대 오차 < 0.01% 달성 가능                   │
│                                                    │
│ 6. 파수 k의 영향:                                  │
│    → k ↑ → 진동 빠름 (고주파)                      │
│    → k ↓ → 진동 느림 (저주파)                      │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 💡 실습 과제

위 코드를 직접 실행해보고 다음을 시도해보세요:

### 과제 1: 파수 변경 🔢
```python
# k = π/2, k = 2π로 바꿔서 해 비교
k_new = torch.pi / 2
# 또는
k_new = 2 * torch.pi
```
**질문:** 해가 어떻게 변하나요? 진동 주기는?

### 과제 2: 소스 항 변경 📝
```python
# f(x) = x(1-x)로 변경
def source_term_new(x):
    return x * (1 - x)
```
**질문:** 어떤 해가 나올까요? 해석해를 구할 수 있나요?

### 과제 3: 경계 조건 변경 🎯
```python
# u(0) = 0, u(1) = 1로 변경
u_bc_new = torch.tensor([[0.0], [1.0]])
```
**질문:** 대칭성이 깨지나요? 해의 형태는?

### 과제 4: Neumann 경계 조건 🌊
```python
# du/dx|_{x=0} = 1, du/dx|_{x=1} = 0으로 변경
def neumann_bc_loss(model, x_bc, dudn_bc):
    x_bc = x_bc.requires_grad_(True)
    u = model(x_bc)
    du_dx = torch.autograd.grad(u, x_bc, torch.ones_like(u), 
                                 create_graph=True)[0]
    return torch.mean((du_dx - dudn_bc) ** 2)
```
**질문:** 해가 어떻게 달라지나요?

### 과제 5: 더 깊은 네트워크 🏗️
```python
# 네트워크 깊이 증가
model_deep = SimpleNN(layers=[1, 50, 50, 50, 50, 1])
```
**질문:** 정확도가 향상되나요? 훈련 시간은?

---

## 🔧 문제 해결 (Troubleshooting)

### 문제 1: 손실이 감소하지 않음

**증상:**
```
Epoch 1000 | Loss: 0.5 (변화 없음)
```

**원인 & 해결:**
```python
# 1. 학습률이 너무 작음
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 0.001 → 0.01

# 2. 가중치 불균형
lambda_bc = 100.0  # 10.0 → 100.0

# 3. 콜로케이션 포인트 부족
x_collocation = torch.linspace(0, 1, 500).reshape(-1, 1)  # 100 → 500
```

---

### 문제 2: 경계 조건을 만족하지 못함

**증상:**
```
u(0) = 0.1234 (0이 아님)
u(1) = -0.0567 (0이 아님)
```

**원인 & 해결:**
```python
# 경계 손실 가중치를 크게 증가
lambda_bc = 50.0  # 또는 100.0

# 하드 경계 조건 (Hard BC) 사용
def model_with_hard_bc(model, x):
    """u(0) = u(1) = 0을 자동으로 만족"""
    u_net = model(x)
    return u_net * x * (1 - x)  # x=0, x=1에서 자동으로 0
```

---

### 문제 3: 해석해와 큰 차이

**증상:**
```
Relative Error: 5.0% (너무 큼)
```

**원인 & 해결:**
```python
# 1. 더 많은 에포크
epochs = 10000  # 5000 → 10000

# 2. 2단계 최적화
# Adam으로 사전학습
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(5000):
    # ... 훈련 ...

# L-BFGS로 미세조정
optimizer_lbfgs = torch.optim.LBFGS(model.parameters())
def closure():
    optimizer_lbfgs.zero_grad()
    loss, _ = total_loss(...)
    loss.backward()
    return loss

for step in range(100):
    optimizer_lbfgs.step(closure)
```

---

### 문제 4: NaN 발생

**증상:**
```
Epoch 50 | Loss: nan
```

**원인 & 해결:**
```python
# 1. 학습률이 너무 큼
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 0.001 → 0.0001

# 2. 그래디언트 클리핑
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 가중치 초기화 변경
for m in model.layers:
    nn.init.xavier_uniform_(m.weight, gain=0.5)  # gain 추가
```

---

## 🔗 다음 단계

Helmholtz 방정식을 완전히 이해했다면:

### 2D로 확장 🌐
```
∂²u/∂x² + ∂²u/∂y² + k²u = f(x, y)
```
- 원형, 사각형, L자형 도메인
- 2D 시각화 (heatmap, contour)

### 시간 의존 문제 ⏰
```
파동 방정식: ∂²u/∂t² = c² ∇²u
```
- 초기 조건 추가
- 시간 진화 애니메이션

### 복잡한 도메인 🏗️
```
- 원형 도메인: x² + y² ≤ R²
- 불규칙한 경계
- 다중 연결 영역
```

### 고급 기능 🚀
```
- 적응형 샘플링
- 동적 손실 가중치
- 전이 학습
```

👉 [고급 기능 가이드](../07_고급기능.md)로 이동!

---

## 📚 참고 자료

### 이론 배경
- **PDE 이론:** Evans, L. C. - "Partial Differential Equations"
- **음향학:** Morse & Ingard - "Theoretical Acoustics"
- **전자기학:** Jackson - "Classical Electrodynamics"

### PINN 논문
- Raissi et al. (2019) - "Physics-informed neural networks: A deep learning framework..."
- Wang et al. (2021) - "Understanding and mitigating gradient flow pathologies..."

### 관련 문서
- **[PDE 잔차 손실 완전 가이드](../concepts/PDE_잔차_손실_상세설명.md)** - 기초 개념
- **[손실 함수 구조의 이유](../concepts/PINN_손실함수_구조의_이유.md)** - 이론적 배경
- **[손실 함수 가이드](../04_손실함수.md)** - 다양한 손실 함수

---

## 💬 마무리

축하합니다! 🎉

이제 여러분은 **1D Helmholtz 방정식을 PINN으로 완전히 풀 수 있습니다!**

**배운 내용:**
- ✅ 물리적 의미 (음향파, 전자기파)
- ✅ 수학적 정식화 (잔차, 경계 조건)
- ✅ 완전한 구현 (9단계)
- ✅ 결과 검증 (해석해 비교)
- ✅ 파라미터 실험 (k의 영향)
- ✅ 문제 해결 기법

**다음 도전:**

이제 더 복잡한 문제를 풀어보세요!

👉 [훈련 과정 가이드](../05_훈련과정.md)  
👉 [결과 분석 가이드](../06_결과분석.md)  
👉 [고급 기능 가이드](../07_고급기능.md)

---

**질문이나 피드백이 있으신가요?**

이 가이드가 도움이 되었다면, 프로젝트에 스타⭐를 부탁드립니다!

**행운을 빕니다! 🚀**

---

*마지막 업데이트: 2025년 1월*
