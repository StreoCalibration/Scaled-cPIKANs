# Scaled-cPIKAN: 이론과 구현 가이드

부제: Chebyshev 기반 물리 정보 신경망의 도메인 스케일링 기법

작성일: 2025-10-25
버전: 1.0
목적: 논문 “Scaled-cPIKANs: Domain Scaling in Chebyshev-based Physics-informed Kolmogorov-Arnold Networks”의 방법론을 이해하기 쉽게 설명하고, 실제 구현 관점에서 필요한 요소를 일목요연하게 정리합니다.

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

핵심 아이디어: PDE와 경계/초기 조건을 손실 함수에 직접 포함해 신경망이 물리 법칙을 만족하도록 학습합니다.

문제 설정:
$$
\mathcal{N}[u](\mathbf{x}) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega
$$
$$
\mathcal{B}[u](\mathbf{x}) = g(\mathbf{x}), \quad \mathbf{x} \in \partial\Omega
$$
(시간 의존 문제의 초기 조건) \;\; $u(\mathbf{x}, 0) = u_0(\mathbf{x})$

PINN 근사: 신경망 $u_\theta$로 해를 근사합니다.
$$
u(\mathbf{x}) \approx u_\theta(\mathbf{x})
$$

장점(요약): 메쉬 프리, 고차원에 비교적 유리, 노이즈 내성, 연속 함수 표현, 파라메트릭 재사용 가능

### 2.2 KAN 이론(MLP와의 비교)

Kolmogorov–Arnold 표현 정리(요지): 고차원 연속 함수는 단변수 함수들의 합성과 합으로 표현 가능합니다.
$$
f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi_q\!\left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)
$$

- 전통적 MLP: 고정 활성화 $\sigma$를 층마다 사용
- KAN: 엣지마다 학습 가능한 단변수 함수 $\phi_{ij}(\cdot)$ 사용 → 표현력 향상

### 2.3 Chebyshev 다항식 요약

정의:
$$
T_n(\cos\theta) = \cos(n\theta), \qquad T_n(x) = \cos(n\,\arccos x)
$$
재귀 관계:
$$
\begin{align}
T_0(x)&=1, \quad T_1(x)=x, \\
T_{n+1}(x) &= 2x\,T_n(x) - T_{n-1}(x)
\end{align}
$$
직교성(가중치 $w(x)=1/\sqrt{1-x^2}$):
$$
\int_{-1}^{1} T_m(x)T_n(x)\frac{1}{\sqrt{1-x^2}}dx = \begin{cases}
0 & m\ne n \\
\pi & m=n=0 \\
\pi/2 & m=n\ge 1
\end{cases}
$$

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

문서 버전: 1.0
최종 업데이트: 2025-10-25
작성자: Scaled-cPIKAN 개발팀
관련 문서: 모델/손실/데이터 구현 파일 및 실행 예제는 `src/`, `examples/` 디렉토리를 참고하십시오.
