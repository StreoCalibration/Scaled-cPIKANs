# Scaled-cPIKAN: 이론과 구현 가이드

부제: Chebyshev 기반 물리 정보 신경망의 도메인 스케일링 기법

작성일: 2025-10-25
버전: 1.2
최종 업데이트: 2025-10-26
목적: 논문 "Scaled-cPIKANs: Domain Scaling in Chebyshev-based Physics-informed Kolmogorov-Arnold Networks"의 방법론을 이해하기 쉽게 설명하고, 실제 구현 관점에서 필요한 요소를 일목요연하게 정리합니다.

---

## 1. 소개 및 개요

Scaled-cPIKAN은 물리 정보 신경망(Physics-Informed Neural Networks, PINN)에 Kolmogorov-Arnold Networks(KAN)를 결합하고, Chebyshev 다항식의 정의역 제약(입력 ∈ [-1, 1])을 만족시키기 위해 ‘아핀 도메인 스케일링(affine domain scaling)’을 첫 연산으로 삽입한 하이브리드 아키텍처입니다.

핵심 문제와 목표:
- 문제: 기존 PINN의 MLP 백본은 스펙트럼 편향(spectral bias)으로 인해 고주파(진동) 해, 경직된 비선형 해를 학습하기 어렵습니다.
- 목표: MLP를 Chebyshev 기반 KAN(cKAN)으로 대체하고, 입력을 [-1, 1]로 스케일링해 Chebyshev 기저의 이점을 살려 정확도와 수렴 속도를 개선합니다.

핵심 혁신:
1) Chebyshev 기반 KAN: 엣지(연결선)에 학습 가능한 활성화 함수 배치, Chebyshev 급수로 매개변수화
2) 도메인 스케일링: 물리적 좌표를 [-1, 1]로 아핀 변환해 cKAN의 수학적 제약 충족
3) 2단계 최적화: Adam(탐색) → L-BFGS(정밀 수렴)

적용 예시: 유체/열전달 PDE, Helmholtz, Allen–Cahn, 반응-확산, 간섭계(3D 표면 재구성) 역문제 등

---

## 2. 배경 이론

### 2.1 PINN 기초

물리 정보 신경망(Physics-Informed Neural Networks, PINN)은 2019년 Raissi 등이 제안한 혁신적인 방법론으로, 신경망과 물리 법칙을 결합하여 편미분방정식(PDE)을 풀거나 역문제를 해결합니다.

#### 2.1.1 기본 개념과 동기

**전통적 수치해법의 한계**:
- **유한요소법(FEM)**, **유한차분법(FDM)**: 도메인을 메쉬로 분할해야 하며, 고차원에서 계산 비용이 기하급수적으로 증가(차원의 저주)
- **복잡한 형상**: 불규칙한 도메인에서 메쉬 생성이 어렵고 시간 소모적
- **역문제**: 관측 데이터로부터 파라미터를 추정할 때 전통적 방법은 반복적 순방향 해석이 필요

**PINN의 핵심 아이디어**:
PDE와 경계/초기 조건을 손실 함수에 **직접 포함**하여, 신경망이 물리 법칙을 만족하는 함수를 학습하도록 합니다. 즉, 신경망 $u_\theta(\mathbf{x})$가 PDE의 해가 되도록 제약을 가하면서 학습합니다.

**메쉬 프리(Mesh-free) 방법론**:
- 도메인을 격자로 나누지 않고 **콜로케이션 포인트**(collocation points)를 샘플링
- 임의의 위치에서 함수값 평가 가능 → 연속 함수 표현
- 형상 변화에 유연하게 대응

#### 2.1.2 수학적 정식화

**문제 설정** (일반적인 PDE):
$$
\mathcal{N}[u](\mathbf{x}) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega \quad \text{(도메인 내부)}
$$
$$
\mathcal{B}[u](\mathbf{x}) = g(\mathbf{x}), \quad \mathbf{x} \in \partial\Omega \quad \text{(경계 조건)}
$$

시간 의존 문제의 경우:
$$
u(\mathbf{x}, 0) = u_0(\mathbf{x}) \quad \text{(초기 조건)}
$$

여기서:
- $\mathcal{N}[\cdot]$: PDE 연산자 (예: $\nabla^2 u + k^2 u$ for Helmholtz)
- $\mathcal{B}[\cdot]$: 경계 조건 연산자 (Dirichlet, Neumann, Robin 등)
- $f(\mathbf{x})$, $g(\mathbf{x})$: 소스 항 및 경계값
- $\Omega$: 관심 도메인, $\partial\Omega$: 도메인 경계

**PINN 근사**:
신경망 $u_\theta(\mathbf{x})$로 해를 근사합니다:
$$
u(\mathbf{x}) \approx u_\theta(\mathbf{x})
$$

**손실 함수 구성**:
PINN의 핵심은 물리 법칙을 손실 함수로 인코딩하는 것입니다:
$$
\mathcal{L}_{\text{total}} = \lambda_{\text{PDE}}\,\mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}}\,\mathcal{L}_{\text{BC}} + \lambda_{\text{IC}}\,\mathcal{L}_{\text{IC}} + \lambda_{\text{data}}\,\mathcal{L}_{\text{data}}
$$

각 항의 의미:

1. **PDE 잔차 손실** $\mathcal{L}_{\text{PDE}}$:
$$
\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}}\sum_{i=1}^{N_{\text{PDE}}} \big\|\mathcal{N}[u_\theta](\mathbf{x}_i) - f(\mathbf{x}_i)\big\|^2
$$
도메인 내부에서 PDE를 얼마나 잘 만족하는지 측정합니다.

2. **경계 조건 손실** $\mathcal{L}_{\text{BC}}$:
$$
\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}}\sum_{i=1}^{N_{\text{BC}}} \big\|\mathcal{B}[u_\theta](\mathbf{x}_i) - g(\mathbf{x}_i)\big\|^2
$$
경계에서 주어진 조건을 만족하는지 확인합니다.

3. **초기 조건 손실** $\mathcal{L}_{\text{IC}}$ (시간 의존 문제):
$$
\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{IC}}}\sum_{i=1}^{N_{\text{IC}}} \big\|u_\theta(\mathbf{x}_i, 0) - u_0(\mathbf{x}_i)\big\|^2
$$

4. **데이터 손실** $\mathcal{L}_{\text{data}}$ (역문제, 선택적):
$$
\mathcal{L}_{\text{data}} = \frac{1}{N_{\text{data}}}\sum_{i=1}^{N_{\text{data}}} \big\|u_\theta(\mathbf{x}_i) - u_{\text{obs},i}\big\|^2
$$
관측 데이터를 활용하여 해의 정확도를 높입니다.

#### 2.1.3 자동 미분의 역할

PINN의 구현에서 **자동 미분(Automatic Differentiation)**은 필수적입니다.

**왜 중요한가?**
- PDE는 미분 연산자를 포함 (예: $\frac{\partial^2 u}{\partial x^2}$, $\nabla^2 u$)
- 손실 계산 시 신경망 출력의 공간/시간 미분이 필요
- 손실 함수를 파라미터로 미분하여 그래디언트 계산 (역전파)

**PyTorch Autograd 활용**:
```python
import torch

# 입력에 대해 미분 추적 활성화
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
u = model(x)  # 신경망 출력

# 1차 미분 계산
u_x = torch.autograd.grad(u, x, torch.ones_like(u), 
                          create_graph=True)[0]

# 2차 미분 계산 (고차 미분은 create_graph=True 필요)
u_xx = torch.autograd.grad(u_x[:, 0:1], x, torch.ones_like(u_x[:, 0:1]),
                           create_graph=True)[0][:, 0:1]
```

**고차 미분 처리**:
- `create_graph=True`: 미분 계산 자체를 계산 그래프에 포함 → 2차, 3차 미분 가능
- 예: Biharmonic 방정식 ($\nabla^4 u$)은 4차 미분 필요

**계산 효율성**:
- 자동 미분은 수치 미분보다 정확하고 빠름
- 체인 룰을 효율적으로 적용하여 그래디언트 계산

#### 2.1.4 전통적 수치해법과의 비교

| 특성 | FEM/FDM | PINN |
|------|---------|------|
| **메쉬 의존성** | 필수 (메쉬 생성 필요) | 불필요 (메쉬 프리) |
| **차원 확장** | 차원 증가 시 급격한 비용 증가 | 고차원에 비교적 유리 |
| **복잡한 형상** | 메쉬 생성이 어려움 | 도메인 샘플링만으로 처리 |
| **역문제** | 반복적 순방향 해석 필요 | 데이터와 물리 법칙 동시 활용 |
| **연속성** | 이산적 해 (보간 필요) | 연속 함수 (임의 위치 평가) |
| **정확도** | 수렴성 이론 확립됨 | 신경망 최적화에 의존 |
| **계산 비용** | 메쉬 크기에 선형적 | 훈련 시간이 주요 비용 |

**PINN의 장점**:
- **메쉬 프리**: 복잡한 형상에서도 도메인 포인트만 샘플링
- **고차원 적합성**: 3D 이상 문제에서 상대적으로 유리
- **노이즈 내성**: 관측 데이터가 노이즈를 포함해도 물리 법칙으로 정규화
- **연속 표현**: 신경망은 연속 함수 → 임의 위치에서 미분 가능
- **파라메트릭 재사용**: 훈련된 모델을 다양한 초기/경계 조건에 재사용 가능

**PINN의 한계**:
- **스펙트럼 편향(Spectral Bias)**: 표준 MLP는 저주파 성분을 먼저 학습 → 고주파/진동 해 학습 어려움
- **경직된 비선형성**: Allen-Cahn 같은 경직(stiff) 문제에서 수렴 느림
- **하이퍼파라미터 민감성**: 손실 가중치($\lambda$), 학습률, 네트워크 크기 조정 필요
- **수렴 보장 없음**: 전통적 방법처럼 엄밀한 수렴 이론 부족

#### 2.1.5 적용 사례와 도전 과제

**성공적 적용 분야**:
- 유체 역학(Navier-Stokes)
- 열전달 및 확산 방정식
- 파동 방정식 (음향, 전자기파)
- 구조 해석 (탄성, 소성)
- 생물물리 모델 (반응-확산)
- **역문제**: 파라미터 추정, 데이터 동화

**Scaled-cPIKAN의 등장 배경**:
PINN의 스펙트럼 편향과 경직 문제를 해결하기 위해 Chebyshev 기반 KAN을 도입한 것이 본 프로젝트의 핵심 동기입니다.

---

### 2.2 KAN 이론(MLP와의 비교)

Kolmogorov-Arnold Networks(KAN)는 전통적인 다층 퍼셉트론(MLP)의 구조를 근본적으로 재설계한 신경망 아키텍처입니다.

#### 2.2.1 전통적 MLP의 구조와 한계

**MLP 아키텍처**:
표준 MLP는 다음과 같이 구성됩니다:
$$
\mathbf{h}^{(l+1)} = \sigma\!\left(\mathbf{W}^{(l)}\mathbf{h}^{(l)} + \mathbf{b}^{(l)}\right)
$$

여기서:
- $\mathbf{W}^{(l)}$: 가중치 행렬 (학습 가능)
- $\mathbf{b}^{(l)}$: 편향 벡터 (학습 가능)
- $\sigma$: **고정된** 활성화 함수 (ReLU, tanh, sigmoid 등)
- $\mathbf{h}^{(l)}$: $l$번째 층의 활성화

**핵심 특징**:
- **노드(뉴런)에서 계산**: 가중합 → 활성화 함수 적용
- **엣지(연결)는 선형**: 단순히 가중치를 곱함
- **활성화 함수는 고정**: 모든 뉴런이 같은 $\sigma$ 사용

**MLP의 한계**:

1. **고정 활성화 함수의 제약**:
   - ReLU, tanh 등은 모든 뉴런에 동일하게 적용
   - 문제별 최적 활성화를 선택할 수 없음
   - 복잡한 비선형성 표현에 제한

2. **스펙트럼 편향(Spectral Bias)**:
   - MLP는 저주파 성분을 먼저 학습하는 경향
   - 고주파/진동 성분 학습이 매우 느리거나 실패
   - PINN에서 진동하는 PDE 해를 학습하기 어려움

3. **파라미터 비효율성**:
   - 복잡한 함수를 표현하려면 많은 뉴런과 층 필요
   - 과도한 파라미터 → 과적합 위험, 계산 비용 증가

4. **해석 가능성 부족**:
   - 어떤 패턴을 학습했는지 이해하기 어려움
   - 블랙박스 특성

#### 2.2.2 Kolmogorov-Arnold 표현 정리

**정리의 핵심 (1957)**:
모든 다변수 연속 함수는 **단변수 함수들의 합성과 합**으로 정확히 표현할 수 있습니다:
$$
f(\mathbf{x}) = f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q\!\left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)
$$

여기서:
- $\phi_{q,p}: \mathbb{R} \to \mathbb{R}$: **단변수 내부 함수**
- $\Phi_q: \mathbb{R} \to \mathbb{R}$: **단변수 외부 함수**
- $n$: 입력 차원

**의미**:
- 고차원 함수 근사 문제를 단변수 함수들의 조합으로 환원
- MLP처럼 고정 활성화 함수가 아니라, **학습 가능한 단변수 함수** 사용 가능
- 이론적으로 더 효율적인 표현 가능

**MLP와의 연결**:
- MLP도 보편 근사 정리(Universal Approximation Theorem)로 임의의 함수 근사 가능
- 하지만 고정 활성화 + 많은 뉴런 필요
- KAN은 단변수 함수를 학습하여 더 적은 파라미터로 표현

#### 2.2.3 KAN의 핵심 개념

**엣지 기반 활성화 함수**:
KAN은 MLP와 달리 **연결선(엣지)마다** 학습 가능한 활성화 함수를 배치합니다.

**구조적 차이**:
```
MLP:  [입력] --선형--> [노드: 가중합 + σ] --선형--> [노드: 가중합 + σ] --> [출력]
      
KAN:  [입력] --φ₁(·)--> [노드: 합] --φ₂(·)--> [노드: 합] --> [출력]
             --φ₃(·)-->            --φ₄(·)-->
```

**KAN의 층 구조**:
$$
\mathbf{h}^{(l+1)}_j = \sum_{i=1}^{n_l} \phi^{(l)}_{i,j}(h^{(l)}_i)
$$

여기서:
- $\phi^{(l)}_{i,j}(\cdot)$: $i$번째 입력 뉴런에서 $j$번째 출력 뉴런으로의 **학습 가능한 단변수 함수**
- 노드에서는 단순히 **합산**만 수행 (활성화 함수 없음)

**학습 가능한 단변수 함수의 매개변수화**:
Chebyshev 급수 (Scaled-cPIKAN의 선택):
$$
\phi_{i,j}(x) \approx \sum_{k=0}^{K} c^{i,j}_k\, T_k(x)
$$

- $c^{i,j}_k$: 학습 가능한 계수
- $T_k(x)$: Chebyshev 다항식
- 각 엣지마다 독립적인 계수 집합

다른 선택지:
- B-spline (부드러운 곡선)
- Fourier 급수 (주기적 함수)
- RBF (방사 기저 함수)

#### 2.2.4 MLP와 KAN의 심층 비교

| 관점 | MLP | KAN |
|------|-----|-----|
| **계산 위치** | 노드 (가중합 + 활성화) | 엣지 (단변수 함수) |
| **활성화 함수** | 고정 (ReLU, tanh 등) | 학습 가능 (Chebyshev 등) |
| **파라미터** | 가중치 행렬 $\mathbf{W}$ | 함수 계수 $c_{k}$ (엣지별) |
| **유연성** | 제한적 (같은 σ) | 매우 높음 (엣지별 최적화) |
| **표현력** | 층과 뉴런 수에 의존 | 적은 파라미터로 복잡한 함수 표현 |
| **스펙트럼 편향** | 심각함 (고주파 학습 어려움) | 완화됨 (Chebyshev 기저) |
| **해석 가능성** | 낮음 (블랙박스) | 높음 (함수 형태 시각화 가능) |
| **계산 비용** | 행렬 곱셈 (빠름) | 기저 함수 계산 (약간 느림) |

**표현력 비교 (이론적)**:
- MLP: $\mathcal{O}(W \cdot D)$ 파라미터 (폭 $W$, 깊이 $D$)
- KAN: $\mathcal{O}(N \cdot M \cdot K)$ 파라미터 (입력 $N$, 출력 $M$, 기저 차수 $K$)
- 동일한 근사 정확도에 KAN이 **더 적은 파라미터** 필요 (문제 의존적)

**학습 특성**:
- MLP: 경사하강법으로 빠르게 수렴 (단순한 문제)
- KAN: 초기에는 느릴 수 있으나, 복잡한 비선형성에서 최종 정확도 높음
- Chebyshev 기저의 직교성 → 계수 간섭 최소화 → 안정적 학습

#### 2.2.5 KAN의 장점과 도전 과제

**이론적 장점**:
1. **보편 근사 능력**: Kolmogorov-Arnold 정리에 기반한 이론적 근거
2. **파라미터 효율성**: 적은 파라미터로 복잡한 함수 표현
3. **적응적 비선형성**: 각 엣지가 문제에 맞는 활성화 학습

**실용적 이점 (PINN 적용)**:
1. **고주파 표현**: Chebyshev 기저로 진동 해 학습 가능
2. **수치 안정성**: [-1,1] 정의역에서 유계성 보장
3. **물리 정보와의 조화**: PDE 잔차 계산 시 미분 친화적

**구현상 고려사항**:
1. **계산 복잡도**: 기저 함수 계산이 단순 행렬 곱셈보다 느림
   - Chebyshev 재귀 관계로 효율적 계산 가능
2. **메모리 사용**: 엣지별 계수 저장 필요
   - 작은 네트워크로도 효과적이므로 실제로는 문제 없음
3. **하이퍼파라미터**: 기저 차수 $K$, 네트워크 구조 선택 필요
   - $K=3\sim4$가 일반적으로 효과적

**Scaled-cPIKAN에서의 활용**:
- Chebyshev 기반 KAN (cKAN) 사용
- 도메인 스케일링으로 [-1,1] 제약 만족
- PINN 손실과 결합하여 물리 법칙 학습
- 2단계 최적화로 안정적 수렴

---

Chebyshev 다항식은 [-1, 1] 구간에서 정의되는 직교 다항식으로, 수치 해석과 근사 이론에서 매우 중요한 역할을 합니다. Scaled-cPIKAN에서는 이 다항식을 기저 함수로 사용하여 학습 가능한 활성화 함수를 구성합니다.

#### 2.3.1 정의 및 재귀 관계

**기본 정의** (삼각함수 형태):
$$
T_n(\cos\theta) = \cos(n\theta), \qquad T_n(x) = \cos(n\,\arccos x)
$$

이 정의는 Chebyshev 다항식이 코사인 함수의 합성으로 표현될 수 있음을 보여줍니다. 이러한 특성 덕분에 [-1, 1] 구간에서 진동하는 함수를 효과적으로 근사할 수 있습니다.

**재귀 관계** (효율적 계산):
$$
\begin{align}
T_0(x)&=1, \quad T_1(x)=x, \\
T_{n+1}(x) &= 2x\,T_n(x) - T_{n-1}(x)
\end{align}
$$

이 재귀 관계는 구현에서 매우 중요합니다. 처음 두 다항식만 알면 이후의 모든 고차 다항식을 순차적으로 계산할 수 있어 계산 효율성이 뛰어납니다.

#### 2.3.2 명시적 형태 (구체적 예시)

처음 몇 개의 Chebyshev 다항식을 명시적으로 나타내면 다음과 같습니다:

$$
\begin{align}
T_0(x) &= 1 \\
T_1(x) &= x \\
T_2(x) &= 2x^2 - 1 \\
T_3(x) &= 4x^3 - 3x \\
T_4(x) &= 8x^4 - 8x^2 + 1 \\
T_5(x) &= 16x^5 - 20x^3 + 5x
\end{align}
$$

**관찰할 점**:
- $T_0$는 상수, $T_1$은 선형 → 기본적인 근사부터 시작
- $T_2$ 이상은 비선형 항 포함 → 복잡한 패턴 표현 가능
- 차수가 증가할수록 계수도 증가 → 고주파 진동 표현
- 홀수 차수는 홀함수, 짝수 차수는 짝함수 → 대칭성 활용

#### 2.3.3 직교성과 수학적 성질

**직교성** (가중치 함수 $w(x)=1/\sqrt{1-x^2}$):
$$
\int_{-1}^{1} T_m(x)T_n(x)\frac{1}{\sqrt{1-x^2}}dx = \begin{cases}
0 & m\ne n \\
\pi & m=n=0 \\
\pi/2 & m=n\ge 1
\end{cases}
$$

이 직교성은 다음과 같은 중요한 의미를 갖습니다:
- **독립성**: 서로 다른 차수의 다항식들이 서로 간섭하지 않음
- **학습 효율**: 각 계수 $c_k$를 독립적으로 학습 가능
- **안정성**: 수치적으로 안정적인 급수 전개

**주요 수학적 특성**:
1. **유계성**: 모든 $x \in [-1,1]$에 대해 $|T_n(x)| \leq 1$
   - 극값은 정확히 ±1 (수치 안정성 보장)
   
2. **영점(roots)**: $T_n(x)$는 [-1,1]에 정확히 n개의 영점을 가짐
   - 영점 위치: $x_k = \cos\!\left(\frac{2k-1}{2n}\pi\right), \; k=1,\ldots,n$
   
3. **극값점**: n+1개의 극값점에서 $T_n(x) = \pm 1$
   - 극값 위치: $x_j = \cos\!\left(\frac{j\pi}{n}\right), \; j=0,\ldots,n$

4. **균등 근사(Minimax Property)**: 
   - Chebyshev 급수는 최대 오차를 최소화하는 최적 근사
   - 다른 다항식 기저보다 빠르게 수렴

#### 2.3.4 PINN 및 cKAN에서의 이점

Chebyshev 다항식이 Scaled-cPIKAN에서 효과적인 이유:

**1. 고주파 표현 능력**
- 차수가 증가할수록 진동 횟수가 증가 → PDE의 진동 해(Helmholtz 등) 표현에 유리
- 전통적 MLP는 스펙트럼 편향으로 고주파 학습이 어려움
- Chebyshev 기저는 고주파 성분을 자연스럽게 포함

**2. 수치 안정성**
- [-1,1]에서 유계 → 그래디언트 폭발/소실 방지
- 직교성 → 계수 학습 시 상호 간섭 최소화
- 재귀 관계 → 효율적이고 안정적인 계산

**3. 효율적 미분 계산**
- Chebyshev 다항식의 도함수도 Chebyshev 급수로 표현 가능
- PINN의 PDE 잔차 계산 시 자동미분과 결합하여 효율적
- 미분 공식: $\frac{d}{dx}T_n(x) = n\,U_{n-1}(x)$ (Chebyshev 제2종과 연결)

**4. 빠른 수렴**
- 균등 근사 특성으로 적은 항으로 정확한 근사
- cKAN에서 낮은 차수 K(예: 3~4)로도 복잡한 함수 학습 가능
- 파라미터 효율성 향상

**5. 적응적 표현**
- 학습 가능한 계수 $c_k$로 각 엣지별 최적 활성화 함수 구성
- 도메인의 지역적 특성에 맞춰 자동으로 적응
- MLP의 고정 활성화 함수 대비 훨씬 유연함

**구현 예시** (개념):
```python
# ChebyKANLayer에서의 사용
# 입력 x는 이미 [-1,1]로 스케일링됨
T = [torch.ones_like(x), x]  # T0, T1
for k in range(2, K+1):
    T.append(2 * x * T[-1] - T[-2])  # 재귀 관계
# einsum으로 학습된 계수 c와 결합
output = torch.einsum("bik,oik->bo", torch.stack(T, dim=-1), c)
```

---

## 3. Scaled-cPIKAN 방법론

### 3.1 핵심 구성 요소

1) PINN 손실 프레임워크:
$$
\mathcal{L}_{\text{total}} = \lambda_{\text{PDE}}\,\mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}}\,\mathcal{L}_{\text{BC}} + \lambda_{\text{IC}}\,\mathcal{L}_{\text{IC}} + \lambda_{\text{data}}\,\mathcal{L}_{\text{data}}
$$

2) cKAN 엔진(학습 가능한 활성화 함수):
$$
\phi(x) \approx \sum_{k=0}^{K} c_k\, T_k(x) \quad (c_k: \text{학습 파라미터})
$$

3) 아핀 도메인 스케일링:
$$
\hat{x}_i = 2\,\frac{x_i - x_{\min,i}}{x_{\max,i} - x_{\min,i}} - 1
$$
이 변환을 모델의 첫 연산으로 고정 적용해 cKAN의 정의역 제약을 만족시킵니다.

### 3.2 왜 효과적인가?
- Chebyshev 기저는 [-1, 1]에서 수치적으로 안정적이며 직교성으로 계수 학습의 간섭을 줄입니다.
- 엣지별 단변수 함수( cKAN )는 복잡한 국소 패턴과 고주파 성분을 더 잘 포착합니다.
- 스케일링으로 이론과 구현 간 간극을 제거해 Chebyshev의 장점을 그대로 활용합니다.

---

## 4. 아키텍처와 데이터 흐름

### 4.1 고수준 아키텍처(모듈)
- 데이터 생성/처리: 도메인 내부/경계에 콜로케이션 포인트 샘플링 → 아핀 스케일링 적용
- Scaled-cPIKAN 모델: ChebyKANLayer × L, 필요 시 LayerNorm, tanh 등
- 물리 정보 손실: 자동미분으로 PDE 잔차/BC/IC 계산 및 결합
- 최적화: Adam 사전학습 → L-BFGS 미세조정

### 4.2 데이터 흐름(가상/실제)

| 데이터 유형 | 데이터 소스 및 구조 | 데이터 흐름 |
| :-- | :-- | :-- |
| 가상 데이터(순방향) | PDE, 도메인 경계 $[x_{\min},x_{\max}]$, $[t_{\min},t_{\max}]$ | 1) 콜로케이션 포인트 $(x,t)$ 생성 → 2) $(\hat{x},\hat{t})$ 스케일링 → 3) 모델 입력, $u_{\text{pred}}$ 예측 → 4) PDE 제약 기반 손실 계산 |
| 실제 데이터(역방향) | 측정 데이터 $(x_i,t_i,u_i)$ | 1) 가상 데이터와 동일한 스케일링 → 2) $(x_i,t_i)\!\to u_{\text{pred},i}$ → 3) $\mathcal{L}_{\text{data}}=\text{MSE}(u_{\text{pred},i},u_i)$ 추가 |

---

## 5. 손실 함수 상세

예시(도메인 내부 잔차):
$$
\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}}\sum_{i=1}^{N_{\text{PDE}}} \big\|\,\mathcal{F}[u_\theta](\mathbf{x}_i)\,\big\|^2
$$
경계/초기 조건 및 데이터 항은 문제에 맞는 규정식으로 정의합니다(MSE 등). 총합은 3.1의 $\mathcal{L}_{\text{total}}$을 사용합니다.

---

## 6. 최적화 및 훈련 전략

2단계 최적화(권장):
- Phase 1(Adam): 비교적 큰 학습률(예: 1e-3)로 전역 탐색, 필요 시 스케줄러 사용
- Phase 2(L-BFGS): 전 배치로 정밀 수렴, 준-뉴턴 업데이트

Adam 업데이트 규칙(요약):
$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
\hat m_t &= m_t/(1-\beta_1^t),\quad \hat v_t = v_t/(1-\beta_2^t) \\
\theta_{t+1} &= \theta_t - \alpha\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)
\end{align}
$$

하이퍼파라미터 시작점(예):
- 네트워크: [2, 32, 32, 32, 1] (2D 입력, 1D 출력)
- Chebyshev 차수 K: 3~4
- 학습률: Adam 1e-3(스케줄러), L-BFGS는 라인서치 기본

---

## 7. 샘플링과 자동 미분

샘플링: Latin Hypercube(준-몬테카를로)로 저불일치 분포를 사용해 도메인 전체를 균일하게 커버합니다.
$$
\mathbf{x}_i \sim \text{Uniform}(\Omega)\;\; \text{(개념적으로)}
$$

자동 미분(PyTorch Autograd): 계산 그래프 기반으로 도함수/고차 미분을 계산합니다.
```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)  # 2x + 3 = 7.0
```
고차 미분이 필요하면 create_graph=True 옵션으로 그래프를 유지합니다.

---

## 8. 실험과 평가(요지)

- 벤치마크: 확산, Helmholtz(진동 해), Allen–Cahn(경직 비선형), 반응–확산(순/역)
- 지표: 상대적 L2 오차(Relative L2 Error)
- 관찰: 동일/적은 파라미터로 MLP-PINN 대비 더 높은 정확도와 빠른 수렴

---

## 9. 한계와 향후 과제

- 계산 비용: cKAN은 MLP보다 파라미터가 증가할 수 있음
- 고차원 확장성: 3D 이상 문제에서의 체계적 검증 필요
- 형상 일반화: 복잡한 도메인은 추가 매핑/좌표 변환이 요구될 수 있음

---

## 10. 참고문헌(핵심)

1) Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. Journal of Computational Physics, 378, 686–707.

2) Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov–Arnold Networks. arXiv:2404.19756.

(프로젝트 보고서의 확장 참고문헌은 별도 문서를 참조하세요.)

---

문서 버전: 1.2
최종 업데이트: 2025-10-26
변경 이력: 
- v1.2 - PINN 기초(2.1)와 KAN 이론(2.2) 섹션 대폭 확장 (하위 섹션 추가, 상세 설명, 비교 표 등)
- v1.1 - Chebyshev 다항식 섹션 확장 (명시적 형태, 주요 특성, PINN/cKAN 이점 추가)
작성자: Scaled-cPIKAN 개발팀
관련 문서: 모델/손실/데이터 구현 파일 및 실행 예제는 `src/`, `examples/` 디렉토리를 참고하십시오.
