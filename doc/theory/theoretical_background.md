# Scaled-cPIKAN 이론적 배경 문서

**작성일**: 2025-10-25  
**버전**: 1.0  
**목적**: Scaled-cPIKAN 프로젝트의 수학적, 물리적, 알고리즘적 이론을 상세히 설명

---

## 📋 목차

1. [소개](#소개)
2. [Physics-Informed Neural Networks (PINN)](#physics-informed-neural-networks-pinn)
3. [Kolmogorov-Arnold Networks (KAN)](#kolmogorov-arnold-networks-kan)
4. [Chebyshev 다항식 이론](#chebyshev-다항식-이론)
5. [도메인 스케일링](#도메인-스케일링)
6. [PINN 손실 함수](#pinn-손실-함수)
7. [위상 재구성 이론](#위상-재구성-이론)
8. [최적화 알고리즘](#최적화-알고리즘)
9. [Latin Hypercube Sampling](#latin-hypercube-sampling)
10. [자동 미분](#자동-미분)
11. [참고문헌](#참고문헌)

---

## 소개

### 프로젝트 개요

Scaled-cPIKAN은 **물리 정보 신경망(Physics-Informed Neural Networks, PINN)**과 **Kolmogorov-Arnold Networks(KAN)**를 결합한 하이브리드 아키텍처입니다. 이 시스템은 편미분방정식(PDE)을 풀거나 물리적 제약 조건 하에서 역문제를 해결하는 데 특화되어 있습니다.

### 핵심 혁신

1. **Chebyshev 기반 KAN**: 전통적인 MLP의 고정 활성화 함수를 학습 가능한 체비쇼프 다항식으로 대체
2. **도메인 스케일링**: 물리적 도메인을 체비쇼프 다항식의 최적 구간 [-1, 1]로 자동 변환
3. **2단계 최적화**: Adam으로 빠른 사전학습, L-BFGS로 정밀한 미세조정
4. **준-몬테카를로 샘플링**: Latin Hypercube Sampling으로 효율적인 콜로케이션 포인트 배치

### 응용 분야

- 유체 역학 시뮬레이션
- 열전달 방정식 풀이
- 3D 표면 높이 재구성 (간섭계 데이터)
- 역 설계 문제
- 파라메트릭 PDE 풀이

---

## Physics-Informed Neural Networks (PINN)

### PINN의 기본 개념

**핵심 아이디어**: 신경망을 사용하여 PDE의 해를 근사하되, 물리 법칙(PDE와 경계/초기 조건)을 **손실 함수에 직접 임베딩**합니다.

전통적인 방법:
```
PDE → 이산화 (FDM, FEM, ...) → 선형 시스템 → 해
```

PINN 방식:
```
PDE → 손실 함수 설계 → 신경망 훈련 → 해 (연속 함수)
```

### 수학적 정식화

**문제 설정**:

주어진 PDE:
$$
\\mathcal{N}[u](\\mathbf{x}) = f(\\mathbf{x}), \\quad \\mathbf{x} \\in \\Omega
$$

경계 조건 (Boundary Condition):
$$
\\mathcal{B}[u](\\mathbf{x}) = g(\\mathbf{x}), \\quad \\mathbf{x} \\in \\partial\\Omega
$$

초기 조건 (Initial Condition, 시간 의존 문제):
$$
u(\\mathbf{x}, t=0) = u_0(\\mathbf{x})
$$

여기서:
- $\\mathcal{N}$은 미분 연산자 (예: $\\nabla^2$, $\\partial_t + \\mathbf{v} \\cdot \\nabla$)
- $\\Omega$는 물리적 도메인
- $\\partial\\Omega$는 경계

**PINN 근사**:

신경망 $u_\\theta(\\mathbf{x})$로 해를 근사:
$$
u(\\mathbf{x}) \\approx u_\\theta(\\mathbf{x})
$$

### PINN의 장점

1. **메쉬 프리**: 격자 생성 불필요 (복잡한 기하학에 유리)
2. **고차원 확장성**: 차원의 저주 완화
3. **노이즈 내성**: 관측 데이터가 불완전해도 작동
4. **파라메트릭 해**: 한 번 훈련하면 여러 파라미터에 재사용 가능
5. **연속 표현**: 임의 위치에서 해를 평가 가능

---

## Kolmogorov-Arnold Networks (KAN)

### Kolmogorov-Arnold 표현 정리

**정리 (1957)**:

연속 함수 $f: [0,1]^n \\to \\mathbb{R}$에 대해, 다음과 같이 표현할 수 있습니다:

$$
f(\\mathbf{x}) = f(x_1, x_2, \\ldots, x_n) = \\sum_{q=0}^{2n} \\Phi_q \\left( \\sum_{p=1}^{n} \\phi_{q,p}(x_p) \\right)
$$

여기서 $\\Phi_q$와 $\\phi_{q,p}$는 단변수 연속 함수입니다.

**의미**: 고차원 함수를 단변수 함수들의 합성으로 분해 가능!

### KAN vs MLP

**전통적 MLP**:
$$
y = \\sigma(W_L \\cdot \\sigma(W_{L-1} \\cdot (\\ldots \\sigma(W_1 \\mathbf{x} + \\mathbf{b}_1) \\ldots) + \\mathbf{b}_{L-1}) + \\mathbf{b}_L)
$$

- 고정된 활성화 함수 $\\sigma$ (ReLU, Tanh, ...)
- 학습 가능한 가중치 $W$, 편향 $\\mathbf{b}$

**KAN 구조**:
$$
y = \\sum_{i} \\Phi_i \\left( \\sum_{j} \\phi_{ij}(x_j) \\right)
$$

- **학습 가능한 활성화 함수** $\\phi_{ij}$
- 가중치는 함수 자체

---

## Chebyshev 다항식 이론

### 정의 및 성질

**제1종 체비쇼프 다항식** $T_n(x)$는 다음과 같이 정의됩니다:

$$
T_n(\\cos\\theta) = \\cos(n\\theta)
$$

또는 구간 $[-1, 1]$에서:

$$
T_n(x) = \\cos(n \\arccos(x))
$$

**재귀 관계**:
$$
\\begin{align}
T_0(x) &= 1 \\\\
T_1(x) &= x \\\\
T_{n+1}(x) &= 2x \\cdot T_n(x) - T_{n-1}(x)
\\end{align}
$$

**예시**:
$$
\\begin{align}
T_2(x) &= 2x^2 - 1 \\\\
T_3(x) &= 4x^3 - 3x \\\\
T_4(x) &= 8x^4 - 8x^2 + 1 \\\\
T_5(x) &= 16x^5 - 20x^3 + 5x
\\end{align}
$$

### 직교성

체비쇼프 다항식은 가중치 함수 $w(x) = 1/\\sqrt{1-x^2}$에 대해 직교합니다:

$$
\\int_{-1}^{1} T_m(x) T_n(x) \\frac{1}{\\sqrt{1-x^2}} dx = 
\\begin{cases}
0 & m \\neq n \\\\
\\pi & m = n = 0 \\\\
\\pi/2 & m = n \\geq 1
\\end{cases}
$$

**수치적 장점**: 직교성 덕분에 계수 학습 시 상호 간섭 최소화

---

## 도메인 스케일링

### 필요성

**문제**: 물리적 도메인과 체비쇼프 정의역 불일치

- 물리적 도메인: $\\mathbf{x} \\in [x_{\\min}, x_{\\max}] \\times [y_{\\min}, y_{\\max}] \\times \\ldots$
- 체비쇼프 정의역: $x \\in [-1, 1]$

**해결책**: 아핀 변환으로 스케일링

### 아핀 변환 이론

**수식**:

$$
x_{\\text{scaled}} = 2 \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}} - 1
$$

**역변환**:

$$
x = \\frac{x_{\\max} - x_{\\min}}{2} (x_{\\text{scaled}} + 1) + x_{\\min}
$$

---

## PINN 손실 함수

### 손실 구조

PINN의 총 손실은 여러 항의 가중 합:

$$
\\mathcal{L}_{\\text{total}} = \\lambda_{\\text{PDE}} \\mathcal{L}_{\\text{PDE}} + \\lambda_{\\text{BC}} \\mathcal{L}_{\\text{BC}} + \\lambda_{\\text{IC}} \\mathcal{L}_{\\text{IC}} + \\lambda_{\\text{data}} \\mathcal{L}_{\\text{data}}
$$

### 1. PDE 잔차 손실

**목적**: 신경망 출력이 PDE를 만족하도록 강제

**수식**:

$$
\\mathcal{L}_{\\text{PDE}} = \\frac{1}{N_{\\text{PDE}}} \\sum_{i=1}^{N_{\\text{PDE}}} \\left| \\mathcal{F}[u](\\mathbf{x}_i) \\right|^2
$$

여기서 $\\mathbf{x}_i$는 도메인 내부의 **콜로케이션 포인트**

---

## 최적화 알고리즘

### 2단계 최적화 전략

**동기**: PINN 훈련은 비볼록, 고차원, 병태(ill-conditioned) 최적화 문제

**해결책**: 1차 방법(Adam)과 2차 방법(L-BFGS) 결합

```
Phase 1: Adam (빠른 탐색)
    ↓
Phase 2: L-BFGS (정밀 수렴)
```

### Phase 1: Adam 옵티마이저

**알고리즘**:

Adam (Adaptive Moment Estimation)은 각 파라미터에 적응적 학습률을 적용합니다.

**업데이트 규칙**:

$$
\\begin{align}
m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) g_t & \\text{(1차 모멘트)} \\\\
v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 & \\text{(2차 모멘트)} \\\\
\\hat{m}_t &= \\frac{m_t}{1 - \\beta_1^t} & \\text{(편향 보정)} \\\\
\\hat{v}_t &= \\frac{v_t}{1 - \\beta_2^t} \\\\
\\theta_{t+1} &= \\theta_t - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}
\\end{align}
$$

---

## Latin Hypercube Sampling

### 몬테카를로 vs 준-몬테카를로

**문제**: PINN에서 도메인 샘플링

**순수 랜덤 (Monte Carlo)**:

$$
\\mathbf{x}_i \\sim \\text{Uniform}(\\Omega)
$$

**문제점**:
- 클러스터링 (같은 영역에 여러 점)
- 빈 공간 (샘플이 없는 영역)
- 높은 분산

**준-몬테카를로 (Quasi-Monte Carlo)**:

더 균일한 분포를 보장하는 저불일치 수열(low-discrepancy sequence) 사용

---

## 자동 미분

### PyTorch Autograd

**핵심 개념**: 계산 그래프(computational graph)를 구축하고 역전파로 그래디언트 계산

**예시**:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7.0
```

### PINN에서의 고차 미분

**문제**: PDE는 2차 이상의 미분 필요

**해결책**: `create_graph=True`로 미분 그래프 유지

---

## 참고문헌

### 핵심 논문

**PINN**:
1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707.

**KAN**:
2. Liu, Z., Wang, Y., Vaidya, S., et al. (2024). *KAN: Kolmogorov-Arnold Networks*. arXiv:2404.19756.

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-10-25  
**작성자**: Scaled-cPIKAN 개발팀  
**관련 문서**: [클래스 다이어그램](../implementation/class_diagram_implementation.md)
