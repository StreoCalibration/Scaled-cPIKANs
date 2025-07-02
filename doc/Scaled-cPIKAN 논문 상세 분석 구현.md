# **Scaled-cPIKAN의 기술적 분석 및 구현 가이드: 아키텍처, 방법론, 그리고 검증**

### **초록**

본 보고서는 Mostajeran과 Faroughi(arXiv:2501.02762)에 의해 소개된 Scaled-cPIKAN 아키텍처에 대한 포괄적이고 기술적인 분석을 제공한다. 물리 정보 신경망(Physics-Informed Neural Networks, PINNs)과 체비쇼프 기반 콜모고로프-아르놀트 신경망(Chebyshev-based Kolmogorov-Arnold Networks, cKANs)을 포함한 기초 원리를 해체한 후, 아핀 영역 스케일링(affine domain scaling)이라는 핵심 혁신을 상세히 설명한다. 신경망 구성, 하이퍼파라미터 설정, 고급 손실 함수 전략 및 훈련 프로토콜을 포함하는 포괄적인 구현 가이드를 제시한다. 논문의 벤치마크 검증은 재현을 용이하게 하기 위해 체계적으로 요약된다. 마지막으로, 성능 기여도를 분석하기 위한 절제 연구(ablation study)를 제안하고 향후 연구 방향을 논의하며 비판적 분석을 제공한다. 이 문서는 Scaled-cPIKAN 방법론을 구현하고 확장하고자 하는 연구자 및 실무자를 위한 실용적인 동반 자료로 작성되었다.

---

## **제 1장: 물리 정보 머신러닝의 기초 원리**

### **1.1. 물리 정보 신경망(PINNs)의 패러다임**

물리 정보 신경망(Physics-Informed Neural Networks, PINNs)은 편미분방정식(Partial Differential Equations, PDEs)을 풀기 위한 방법론으로 등장했다. PINNs는 데이터뿐만 아니라 주어진 데이터를 지배하는 물리 법칙을 만족하도록 훈련된 심층 신경망이다 \[1, 2, 3, 4\]. 핵심 아이디어는 PDE를 신경망의 손실 함수에 내장하여 '물리 정보' 정규화 항을 만드는 것이다. 이 항, 즉 PDE 잔차(residual)는 도메인 내의 콜로케이션 포인트(collocation points) 집합에서 평가되며, 신경망의 출력이 물리 법칙을 따르도록 강제한다.

일반적으로 PINNs 아키텍처는 시공간 좌표(예: (x, t))를 입력으로 받고 해 필드(예: u(x, t))를 출력하는 표준 다층 퍼셉트론(Multilayer Perceptrons, MLPs)에 기반한다 \[1, 5\]. 손실 함수는 PDE 잔차, 초기 조건(Initial Conditions, ICs), 그리고 경계 조건(Boundary Conditions, BCs)의 평균 제곱 오차(Mean Squared Error, MSE)를 결합한 복합 손실 함수로 구성된다 \[4, 6, 7\]. 이 접근 방식은 PDE 해결 문제를 다중 목표 최적화 문제로 변환한다.

PINNs의 주요 강점은 메쉬가 필요 없는(mesh-free) 특성, 복잡한 기하학적 구조를 다룰 수 있는 능력, 그리고 순방향 및 역방향 문제를 해결하기 위해 희소하고 노이즈가 있는 데이터를 자연스럽게 통합할 수 있다는 점이다 \[4, 8, 9\].

그러나 PINNs는 몇 가지 내재된 한계를 가지고 있다. 첫째, 표준 MLP 기반 PINNs는 넓은 계산 영역에서 강한 진동 동역학(oscillatory dynamics), 경직된(stiff) 또는 다중 스케일(multi-scale) 특성을 가진 PDE를 다룰 때 종종 수렴성이 저하되고 정확도가 감소하는 훈련 병목 현상을 보인다 \[1, 2, 3, 10\]. 둘째, 신경망, 특히 일반적인 활성화 함수를 사용하는 MLP는 '스펙트럼 편향(spectral bias)'을 가지고 있어, 해의 저주파 성분을 먼저 학습하고 고주파 세부 정보를 포착하는 데 어려움을 겪는다 \[11, 12\]. 이는 진동 해를 가진 문제에서 주요 병목 현상이다. 마지막으로, 손실 함수의 여러 구성 요소(PDE, BC, IC)는 매우 다른 크기를 가질 수 있어 역전파 중 불균형한 그래디언트를 유발할 수 있다. 이로 인해 최적화가 특정 지점에서 멈추거나 한 목표를 다른 목표보다 우선시하게 되어 전반적인 수렴을 방해할 수 있다 \[6, 13, 14\].

### **1.2. 콜모고로프-아르놀트 신경망(KANs)의 등장**

콜모고로프-아르놀트 신경망(Kolmogorov-Arnold Networks, KANs)은 콜모고로프-아르놀트 표현 정리에 영감을 받아 개발되었다. 이 정리는 모든 연속 다변수 함수가 연속 단변수 함수들의 유한한 합성 및 덧셈 연산으로 표현될 수 있음을 명시한다 \[15, 16, 17\].

KANs는 MLP와 근본적으로 다른 아키텍처를 가진다. MLP가 노드(뉴런)에 고정된 활성화 함수를 사용하는 반면, KANs는 학습 가능한 단변수 활성화 함수를 엣지(가중치)에 배치한다 \[15, 18\]. 노드는 단순히 들어오는 신호를 합산하는 역할만 한다. 이는 학습 패러다임을 선형 가중치를 조정하는 것에서 활성화 함수의 형태 자체를 학습하는 것으로 근본적으로 변화시킨다.

이러한 구조는 KANs가 데이터의 국소적 구조에 맞게 활성화 함수를 조정할 수 있게 하여, 특히 복잡한 함수에 대해 MLP보다 더 정확하고 파라미터 효율적일 수 있는 잠재력을 제공한다 \[5, 15\]. 최초의 KANs는 이러한 학습 가능한 함수를 B-스플라인(B-splines)으로 매개변수화했는데, 이는 복잡한 PDE 솔버에 사용하기에는 계산 비용이 매우 높은 것으로 입증되었다 \[5, 19\].

### **1.3. 체비쇼프 기반 KANs(cKANs)를 통한 효율성과 표현력의 결합**

B-스플라인 KANs의 계산 비용 문제를 해결하기 위해, 후속 연구에서는 직교 다항식(orthogonal polynomials) 사용을 제안했다. 체비쇼프 다항식(Chebyshev polynomials)은 뛰어난 근사 특성(최소최대 근사(minimax approximation)에 가까움), 빠른 수렴 속도, 그리고 수치적 안정성 때문에 주요 선택지로 부상했다 \[18, 19, 20\]. 이 다항식들은 매끄러운 함수에 대해 간결하고 효율적인 표현을 제공한다.

제1종 체비쇼프 다항식 T\_n(x)는 T\_0(x)=1과 T\_1(x)=x를 초기값으로 하는 점화식 T\_{n+1}(x) \= 2xT\_n(x) \- T\_{n-1}(x)으로 정의된다. 이들은 특정 가중 함수에 대해 구간 \[-1, 1\]에서 직교성을 가진다.

여기서 결정적인 제약 조건은 체비쇼프 다항식이 본질적으로 표준화된 영역 \[-1, 1\]에서 정의된다는 점이다 \[1, 2, 21\]. 이는 체비쇼프 기반 레이어에 대한 모든 입력이 이 범위로 정규화되어야 함을 의미한다. 이 제약 조건이 바로 Scaled-cPIKAN에서 "스케일링" 구성 요소의 직접적인 동기가 된다.

cPIKAN은 학습 가능한 엣지 함수가 체비쇼프 다항식으로 매개변수화된 물리 정보 KAN이다 \[22, 23\]. 그러나 일부 연구에서는 기본적인 cKAN이 랭크 붕괴(rank collapse)와 같은 문제로 인해 표현력이 제한될 수 있다고 지적했으며, 이는 AC-PKAN과 같은 모델에서 어텐션 메커니즘과 같은 추가적인 개선으로 이어졌다 \[19\]. Scaled-cPIKAN 논문은 다른 종류의 개선, 즉 임의의 물리적 도메인에 cKAN을 견고하게 적용하는 방법에 초점을 맞춘다.

이러한 배경을 종합해 볼 때, Scaled-cPIKAN의 개발은 과학적 머신러닝 분야에서 명확하고 연쇄적으로 발생하는 문제들을 해결하기 위한 논리적 귀결이라 할 수 있다. 이는 지배적인 PINN 패러다임의 잘 정의된 실패 모드에 대한 표적화된 솔루션을 나타낸다. 그 논리적 흐름은 다음과 같다:

1. **문제 1:** 표준 MLP 기반 PINNs는 스펙트럼 편향과 훈련 병목 현상으로 인해 넓은 영역에서 복잡하고 진동하는 문제를 해결하는 데 실패한다 \[1, 3\].  
2. **해결책 1:** MLP 백본을 더 표현력 있는 함수 근사기로 교체한다. 강력한 콜모고로프-아르놀트 정리에 기반한 KANs가 유망한 후보이다 \[15, 16\].  
3. **문제 2:** 원래의 B-스플라인 KANs는 복잡한 PDE 솔버에서 실용적으로 사용하기에는 계산 비용이 너무 높다 \[5, 19\].  
4. **해결책 2:** B-스플라인을 더 효율적인 기저 함수로 교체한다. 체비쇼프 다항식(cKANs)이 우수한 근사 특성과 수치적 안정성 때문에 선택된다 \[18, 20, 21\].  
5. **문제 3:** 체비쇼프 다항식은 엄격한 작동 요구사항을 가진다: 입력 도메인이 반드시 \[-1, 1\]이어야 한다 \[1, 2\]. 그러나 물리적 PDE 문제는 임의의 도메인(예: x \\in \[0, L\], t \\in)에서 정의된다. 따라서 cKANs를 직접 적용하는 것은 불가능하다.  
6. **최종 종합 (Scaled-cPIKAN):** 결정적이고 필수적인 단계는 임의의 물리적 도메인을 요구되는 \[-1, 1\]^d 표준화된 도메인으로 매핑하는 **영역 스케일링** 변환을 도입하는 것이다. 이는 cKANs의 이론적 힘과 물리적 문제의 실제적 현실 사이의 간극을 메운다 \[1, 2\]. 따라서 Scaled-cPIKAN에서 "스케일링"은 모델 크기에 관한 것이 아니라 *영역 변환*에 관한 것이다.

---

## **제 2장: Scaled-cPIKAN 아키텍처: 시너지적 통합**

Scaled-cPIKAN 아키텍처는 물리 정보 학습, 효율적인 함수 표현, 그리고 수학적 제약 조건 준수라는 세 가지 핵심 요소를 시너지적으로 통합하여 기존 방법론의 한계를 극복한다. 각 구성 요소는 다른 구성 요소의 약점을 보완하도록 설계되었으며, 특히 '영역 스케일링'은 전체 구조를 지탱하는 핵심적인 역할을 한다.

### **2.1. 아키텍처 청사진**

Scaled-cPIKAN의 전체적인 구조는 입력에서 출력까지 명확한 정보 흐름을 가지는 계층적 다이어그램으로 개념화할 수 있다.

* **입력 레이어 (Input Layer):** 물리적 좌표(예: 공간 변수 x, 시간 변수 t)를 입력으로 받는다.  
* **영역 스케일링 모듈 (Domain Scaling Module):** 입력된 물리적 좌표를 체비쇼프 다항식의 요구사항에 맞게 표준화된 하이퍼큐브 \[-1, 1\]^d로 정규화하는 아핀 변환(affine transformation) 레이어이다.  
* **cKAN 레이어 (cKAN Layers):** 일련의 완전 연결 레이어로 구성되며, 각 연결은 단순한 가중치가 아니라 체비쇼프 다항식으로 매개변수화된 학습 가능한 함수이다. 각 cKAN 레이어의 입력에는 tanh 활성화 함수가 적용되어 입력값이 \[-1, 1\] 범위 내에 있도록 보장하며, LayerNorm은 그래디언트 소실 문제를 방지하기 위해 사용된다 \[21\].  
* **출력 레이어 (Output Layer):** PDE의 근사해 u\_pred(x, t)를 생성한다.  
* **자동 미분 엔진 (Automatic Differentiation Engine):** PDE 잔차를 형성하기 위해 u\_pred를 입력 (x, t)에 대해 필요한 도함수를 계산한다. 이는 최신 딥러닝 프레임워크의 핵심 기능이다.  
* **손실 계산 모듈 (Loss Calculation Module):** PDE 잔차, 경계 조건, 초기 조건 손실을 결합하여 최종 손실 함수를 구성한다.

이 아키텍처의 강력함은 PINNs의 물리 기반 손실 구조, KANs의 표현력 있는 함수 표현(체비쇼프 다항식으로 효율화됨), 그리고 이 두 구성 요소를 호환 가능하게 만드는 필수적인 영역 스케일링 변환을 결합한 데서 비롯된다 \[1, 2\]. 즉, cKANs는 MLP의 스펙트럼 편향 문제를 해결할 표현력을 제공하지만 \[-1, 1\] 도메인 제약이라는 치명적인 단점을 가지고 있다 \[1, 2\]. 영역 스케일링은 바로 이 단점을 해결하여 PINN 프레임워크에서 cKANs의 잠재력을 완전히 발휘할 수 있도록 하는, 단순하지만 필수적인 열쇠이다 \[1, 2\].

### **2.2. 핵심 혁신—아핀 영역 스케일링**

Scaled-cPIKAN의 가장 핵심적인 혁신은 아핀 영역 스케일링으로, 이는 이론적 모델(cKAN)과 실제 물리 문제 사이의 다리 역할을 한다.

#### **문제 정의**

일반적으로 PDE는 d차원의 직사각형 도메인 \\Omega \= \[x\_{min,1}, x\_{max,1}\] \\times... \\times \[x\_{min,d}, x\_{max,d}\]에서 정의된다. 반면, cKAN 아키텍처는 표준화된 도메인 \\hat{\\Omega} \= \[-1, 1\]^d에서의 입력을 요구한다.

#### **수학적 공식화**

이 불일치를 해결하기 위해 각 입력 변수 x\_i에 아핀 변환을 적용한다. 스케일링된 변수를 \\hat{x}\_i라고 할 때, 변환은 다음과 같이 정의된다:

x^i​=2⋅xmax,i​−xmin,i​xi​−xmin,i​​−1  
이 선형 매핑은 x\_i가 x\_{min,i}에서 x\_{max,i}로 변할 때 \\hat{x}\_i가 \-1에서 1로 부드럽게 매핑되도록 보장한다. 이는 체비쇼프 기저 함수를 사용하기 위한 근본적이고 협상 불가능한 단계이다 \[1, 2\].

#### **구현**

이 스케일링은 신경망에 입력 좌표가 공급되기 전에 수행되는 가장 첫 번째 연산이어야 한다. 이것은 학습되지 않는 고정된 전처리 단계이다. 이 변환이 없으면 cKAN 레이어는 정의된 도메인 밖의 입력을 받게 되어 수치적으로 불안정해지거나 완전히 잘못된 결과를 생성하게 된다.

### **2.3. cKAN 엔진—체비쇼프 급수로서의 학습 가능한 함수**

Scaled-cPIKAN의 심장부는 체비쇼프 다항식으로 구동되는 KAN 엔진이다. 이 엔진은 고정된 활성화 함수 대신 학습 가능한 함수를 통해 뛰어난 표현력을 달성한다.

#### **매개변수화**

표준 MLP에서 레이어의 출력은 \\sigma(W \\cdot x \+ b) 형태이다. cKAN 레이어에서는 개념이 다르다. 레이어 l의 뉴런 i와 레이어 l+1의 뉴런 j 사이의 연결은 학습 가능한 함수 \\phi\_{j,i}^{(l)}(x)로 표현된다. 이 함수는 체비쇼프 다항식의 유한 급수로 근사된다:

ϕj,i(l)​(x)≈k=0∑K​cj,i,k(l)​Tk​(x)  
여기서 K는 체비쇼프 다항식의 차수(degree)이고, 계수 c\_{j,i,k}^{(l)}가 바로 신경망의 **학습 가능한 파라미터**이다 \[18, 19\]. 신경망은 최적의 계수 집합 c를 학습하여 엣지의 활성화 함수 \\phi의 형태를 만들며, 이를 통해 고정된 활성화 함수를 가진 MLP보다 목표 함수(PDE 해)에 훨씬 유연하게 적응할 수 있다.

#### **cKAN 노드의 출력**

레이어 l+1의 뉴런 j는 이전 레이어 뉴런들의 출력에 학습 가능한 함수를 적용한 결과를 합산하여 자신의 출력 x\_j^{(l+1)}을 계산한다:

$$ x\_j^{(l+1)} \= \\sum\_{i} \\phi\_{j,i}^{(l)}(x\_i^{(l)}) \= \\sum\_{i} \\sum\_{k=0}^{K} c\_{j,i,k}^{(l)} T\_k(x\_i^{(l)}) $$

이러한 구조는 신경망이 국소적으로 함수의 복잡성에 적응할 수 있게 하여, 특히 진동이나 급격한 변화가 있는 해를 모델링하는 데 유리하다.

구현 관점에서 볼 때, 초기 영역 스케일링은 첫 번째 레이어의 입력을 \[-1, 1\] 범위로 가져온다. 그러나 중간 레이어의 출력이 이 범위를 유지한다는 보장은 없다. 따라서 각 후속 cKAN 레이어의 입력에 tanh 활성화 함수를 적용하여 신호를 다시 정규화하는 것이 필요하며, 이는 체비쇼프 기저가 항상 올바르게 적용되도록 보장한다. 그 후 LayerNorm은 tanh가 유발할 수 있는 그래디언트 소실 문제를 해결하여 안정적인 훈련을 보장하는 데 추가된다 \[21\]. 이는 심층 Scaled-cPIKAN의 미묘하지만 필수적인 구현 세부사항을 보여준다.

---

## **제 3장: 구현 가이드: 이론에서 코드로**

이 장에서는 연구자나 실무자가 Scaled-cPIKAN 모델을 실제로 구축할 수 있도록 단계별 가이드를 제공한다. 이는 이론적 개념을 구체적인 구현 지침으로 변환하는 데 중점을 둔다.

### **3.1. 신경망 구성 및 파라미터화**

Scaled-cPIKAN 모델은 PyTorch나 TensorFlow와 같은 표준 딥러닝 프레임워크에서 구현할 수 있다. 핵심은 사용자 정의 레이어를 생성하고 자동 미분을 효과적으로 관리하는 능력에 있다.

#### **ChebyKAN 레이어 구현**

사용자 정의 ChebyKANLayer를 생성하는 것이 구현의 핵심이다.

1. **입력:** 이전 레이어의 출력을 입력으로 받는다.  
2. 연산 순서:  
   a. (권장) 안정성을 위해 입력에 LayerNorm을 적용한다 \[21\].  
   b. 입력을 \[-1, 1\] 범위로 제한하기 위해 tanh 활성화 함수를 적용한다 \[21\].  
   c. 지정된 차수 K까지 체비쇼프 다항식 T\_k(x)를 점화식을 사용하여 재귀적으로 계산한다.  
   d. 레이어의 학습 가능한 가중치는 (in\_features, out\_features, K+1) 형태의 계수 c 텐서가 된다.  
   e. 계산된 체비쇼프 다항식 값 T와 계수 c를 사용하여 출력을 계산한다. 효율적인 계산을 위해 einsum 연산(예: output \= torch.einsum('bik,ijk-\>bj', T, c))을 사용하는 것이 좋다.

#### **전체 신경망 조립**

1. 입력 레이어를 정의한다 (예: (x,t)에 대해 2개).  
2. 아핀 영역 스케일링을 입력 데이터에 대한 전처리 단계로 구현한다.  
3. 사용자 정의 ChebyKANLayer를 원하는 깊이만큼 쌓는다.  
4. 출력 레이어를 정의한다 (예: u(x,t)에 대해 1개).

#### **하이퍼파라미터 설정**

성공적인 구현을 위해서는 하이퍼파라미터에 대한 명확한 지침이 필수적이다. 다음 표는 논문 및 관련 연구를 바탕으로 한 권장 시작점을 제공하여, 사용자가 시행착오에 드는 시간을 크게 줄일 수 있도록 돕는다 \[21, 24, 25, 26, 27, 28\].

**표 1: Scaled-cPIKAN 하이퍼파라미터 구성**

| 하이퍼파라미터 | 기호 | 설명 | 권장 값 / 전략 | 출처/근거 |
| :---- | :---- | :---- | :---- | :---- |
| 신경망 아키텍처 | \[n\_in,..., n\_out\] | 각 레이어의 뉴런 수를 정의한다. | 2D 문제의 경우 \`\`로 시작. 더 복잡한 PDE의 경우 더 깊거나 넓게 설정. | 일반적인 PINN 아키텍처 관행 \[27, 29\]. |
| 체비쇼프 차수 | K | 급수 전개의 최대 다항식 차수. K가 높을수록 더 복잡한 함수 표현이 가능하지만 파라미터가 증가한다. | K=3 또는 K=4로 시작. 해의 복잡성에 따라 조정. \[21, 29\]. |  |
| 초기 학습률 | η | 옵티마이저의 스텝 크기. 안정성과 수렴에 매우 중요하다. | Adam의 경우 1e-3, 민감한 네트워크의 경우 더 낮은 값(1e-4) 사용. 학습률 탐색기를 사용하거나 작은 값에서 시작. \[21, 24, 27\]. |  |
| 옵티마이저 | \- | 경사 하강법에 사용되는 알고리즘. | **2단계:** 1\. 초기 빠른 수렴을 위해 Adam 사용. 2\. 최소값 근처에서 미세 조정을 위해 L-BFGS 사용. \[27, 30\]. |  |
| 학습률 스케줄 | \- | 훈련 중 학습률을 감소시키는 전략. | 지수적 감소 또는 단계적 감소. 예: 2000 에포크마다 0.8배 감소. \[24, 31\]. |  |
| 손실 가중치 | λ\_pde, λ\_bc, λ\_ic | 손실 구성 요소의 균형을 맞추기 위한 정적 또는 동적 가중치. | 모두 1.0으로 시작. 훈련이 불안정하면 동적/자체 적응 방식 구현. \[7, 12\]. |  |
| 배치 크기 | B | 미니배치 경사 하강법 사용 시 그래디언트 업데이트 당 샘플 수. | L-BFGS의 경우 전체 배치(full-batch)가 일반적. Adam의 경우 작은 배치(예: 128, 256\) 사용 가능. \[26\]. |  |

### **3.2. 물리 정보 손실 함수: 고급 전략**

#### **기본 공식**

총 손실 L\_total은 PDE 잔차 손실 L\_pde, 경계 조건 손실 L\_bc, 초기 조건 손실 L\_ic의 가중 합으로 구성된다.

$$ L\_{\\text{total}} \= \\lambda\_{\\text{pde}} \\cdot L\_{\\text{pde}} \+ \\lambda\_{\\text{bc}} \\cdot L\_{\\text{bc}} \+ \\lambda\_{\\text{ic}} \\cdot L\_{\\text{ic}} $$

여기서 각 L 항은 해당 제약 조건에 대한 콜로케이션 포인트 집합에서의 평균 제곱 오차(MSE)이다 \[4, 32\].

#### **고급 주제: 동적 및 자체 적응 손실 가중치**

정적 가중치(λ가 고정 상수)는 각 손실 항에서 발생하는 그래디언트 크기가 크게 달라 훈련 불균형을 초래할 수 있기 때문에 종종 실패한다 \[6, 13\]. 견고한 구현을 위해서는 동적 가중치 기법을 고려해야 한다.

1. **불확실성 가중치 (Uncertainty Weighting):** 각 손실을 가우시안 우도(likelihood)로 취급하고 각 손실 항에 대한 노이즈 파라미터 σ\_i를 학습한다. i번째 항의 손실은 (1/2σ\_i^2) \\cdot L\_i \+ \\log(σ\_i)가 된다 \[33, 34\]. 이는 분산이 큰(어려운) 작업을 자동으로 다운웨이트한다.  
2. **그래디언트 정규화 (GradNorm):** 각 손실 항의 그래디언트 놈(norm)이 비슷한 크기를 갖도록 가중치를 동적으로 조정한다 \[34\].  
3. **신경망 탄젠트 커널(NTK) 가중치:** 각 손실 항의 NTK 행렬의 트레이스(trace)를 균등화하여 손실 항의 균형을 맞추는 고급 기법이다. 이는 다른 구성 요소들의 학습 속도를 정렬한다 \[11, 12, 35\].  
4. **자체 적응 (SA-PINN) / 잔차 기반 가중치:** 개별 콜로케이션 포인트나 손실 항에 학습 가능한 가중치를 할당하고, 이를 역전파를 통해 업데이트한다. 종종 모델은 손실을 최소화하려 하고 가중치는 '어려운' 포인트에 대해 손실을 최대화하려는 최소최대(min-max) 게임 형태로 진행된다 \[9, 32, 36, 37\].

**구현 조언:** 첫 구현에서는 정적 가중치로 시작한다. 만약 수렴에 실패하거나 특정 손실 항이 지배적이라면 동적 가중치 기법을 구현한다. 불확실성 가중치 \[33\]는 비교적 간단하면서도 좋은 출발점이 될 수 있다.

### **3.3. 최적화 및 훈련 프로토콜**

#### **데이터 생성**

PDE 잔차를 위한 콜로케이션 포인트는 시공간 도메인 내부에서 무작위로 샘플링한다. BC와 IC를 위한 포인트는 각각 경계와 초기 시간 슬라이스에서 샘플링한다.

#### **2단계 최적화**

1. **1단계 (Adam):** Adam 옵티마이저를 사용하여 많은 에포크(예: 20,000-50,000) 동안 신경망을 훈련시킨다. Adam은 견고하며 좋은 최소값의 대략적인 위치를 찾는 데 효과적이다 \[27\]. 학습률 감소 스케줄을 함께 사용하는 것이 좋다.  
2. **2단계 (L-BFGS):** Adam 단계 이후, L-BFGS 옵티마이저로 전환한다. L-BFGS는 해에 이미 가까워졌을 때 더 빠르고 더 날카로운 최소값으로 수렴하는 경향이 있는 준-뉴턴(quasi-Newton) 방법이다 \[27, 30\]. 일반적으로 각 업데이트에 전체 데이터셋(full-batch)을 사용한다.

#### **수렴 모니터링**

훈련 과정 동안 총 손실과 각 개별 손실(L\_pde, L\_bc, L\_ic)을 모니터링한다. 손실이 충분히 낮은 값에서 안정되면 모델이 수렴한 것으로 간주한다. 또한, 분석해가 알려진 경우 별도의 검증 세트에서 상대적 L2 오차를 모니터링하여 모델의 정확도를 평가한다.

결론적으로, Scaled-cPIKAN의 성공적인 구현은 논문의 핵심 아이디어를 직접 번역하는 것을 넘어, PINN 및 KAN 관련 연구에서 축적된 모범 사례들을 통합해야 한다. 예를 들어, 심층 cKAN을 위한 LayerNorm \[21\], PINN을 위한 2단계 Adam+L-BFGS 최적화 \[27\], 그리고 그래디언트 불균형 문제를 해결하기 위한 동적 손실 가중치 \[7, 12\]와 같은 "불문율"을 통합하는 것이 중요하다. 이러한 지식의 합성은 보고서를 단순한 요약에서 진정한 구현 가이드로 격상시킨다.

---

## **제 4장: 검증 및 성능 분석**

이 장에서는 Scaled-cPIKAN 아키텍처의 효과를 검증하고 그 성능을 정량적으로 분석하는 프레임워크를 제시한다. 논문에서 제안된 벤치마크 문제군, 성능 평가 지표, 그리고 기존 모델과의 비교 분석 방법을 상세히 기술하여 결과의 재현성과 신뢰성을 확보하고자 한다.

### **4.1. 벤치마크 문제군**

Scaled-cPIKAN의 성능은 기존 PINN이 어려움을 겪는 특정 유형의 문제들을 해결하는 능력을 평가하기 위해 신중하게 선택된 벤치마크 문제군을 통해 입증되었다 \[1, 2, 38\]. 각 벤치마크는 표준 PINN의 특정 약점을 시험하도록 설계되었다.

* **확산 방정식 (Diffusion Equation):** 기본적인 정확도와 수렴성을 테스트하기 위한 포물선형(parabolic) PDE이다. 이는 모델의 기본적인 함수 근사 능력을 평가하는 기준선 역할을 한다.  
* **헬름홀츠 방정식 (Helmholtz Equation):** 종종 진동하는 해를 갖는 타원형(elliptic) PDE로, 모델이 고주파 함수를 처리하는 능력을 직접적으로 시험한다 \[39, 40, 41\]. 이 문제에서의 성공은 MLP의 스펙트럼 편향을 cKAN 기저의 표현력으로 극복했음을 입증한다.  
* **앨런-칸 방정식 (Allen-Cahn Equation):** 경직된(stiff) 거동과 날카로운 이동 경계면을 특징으로 하는 비선형 반응-확산 방정식이다 \[27, 42, 43\]. 이 문제는 모델이 비선형성과 급격한 변화를 처리하는 능력을 평가하며, cKAN의 학습 가능한 활성화 함수가 국소적으로 적응하는 능력을 보여준다.  
* **반응-확산 방정식 (Reaction-Diffusion Equation, 순방향 및 역방향):** 이 문제는 PDE를 푸는 순방향 문제와 데이터로부터 미지의 매개변수를 추론하는 역방향 문제를 모두 포함하여 모델의 다재다능함을 시험한다. 특히 노이즈가 포함된 데이터를 사용한 실험은 실제 데이터에 대한 모델의 견고성을 평가한다.

이러한 실험 설계는 단순한 성능 비교를 넘어, 제안된 아키텍처가 기존 방법들이 실패했던 바로 그 지점에서 성공하는지를 체계적으로 검증하는 형태의 표적 검증(targeted validation)이다.

### **4.2. 성능 평가 지표**

모델의 성능을 정량적으로 평가하기 위해 표준화된 지표들이 사용된다.

* **주요 지표: 상대적 L2 오차 (Relative L2 Error):** 정확도를 측정하는 핵심 지표로, 다음과 같이 계산된다:Error=∥uexact​∥2​∥upred​−uexact​∥2​​  
  여기서 \\| \\cdot \\|\_2는 L2 놈(norm)을 의미하며, 평가는 테스트 포인트 그리드 상에서 수행된다. 이는 수치 해석 및 회귀 문제에 대한 머신러닝 분야에서 표준적으로 사용되는 지표이다 \[44, 45\].  
* **보조 지표:**  
  * **수렴 그래프 (Convergence Plots):** 훈련 에포크에 따른 손실 함수 값의 변화를 그려 수렴 속도와 안정성을 시각화한다.  
  * **계산 비용 (Computational Cost):** 훈련 시간과 총 학습 가능 파라미터 수를 보고하여 효율성을 평가한다.  
  * **점별 오차 플롯 (Pointwise Error Plots):** 절대 오차 |u\_{pred} \- u\_{exact}|를 도메인 전체에 걸쳐 시각화하여 모델이 어려움을 겪는 영역을 식별한다.

### **4.3. 비교 분석—Scaled-cPIKAN 대 표준 PINNs**

Scaled-cPIKAN의 우월성 주장을 검증하기 위해서는 표준 MLP 기반 PINN과의 엄격한 비교가 필수적이다 \[1, 2\].

#### **방법론**

1. **베이스라인 모델:** 표준 MLP 기반 PINN을 구현한다.  
2. **공정한 비교:** 아키텍처의 이점을 분리하기 위해, 베이스라인 PINN은 Scaled-cPIKAN 모델과 비슷한 수의 학습 가능 파라미터 또는 비슷한 훈련 시간을 갖도록 구성해야 한다 \[23\].  
3. **실행:** 동일한 초기 상태(랜덤 시드)에서 Scaled-cPIKAN과 베이스라인 PINN을 모든 벤치마크 문제에 대해 훈련시킨다.  
4. **데이터 수집:** 두 모델에 대해 모든 평가 지표를 기록한다.

#### **비교 성능 결과표**

다음 표는 검증 실험의 최종 결과를 구조화된 형식으로 제시하기 위한 템플릿이다. 이를 통해 제안된 모델과 베이스라인을 한눈에 명확하게 비교할 수 있으며, 이는 기술 논문에서 우월성을 입증하기 위한 표준적인 관행이다 \[1\].

**표 2: 벤치마크 문제에 대한 비교 성능 템플릿**

| 벤치마크 문제 | 모델 | 최종 L2 오차 | 훈련 시간 (초) | 총 파라미터 수 | 수렴 에포크 (Adam) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **확산 방정식** | Scaled-cPIKAN |  |  |  |  |
|  | PINN (MLP) |  |  |  |  |
| **헬름홀츠 방정식** | Scaled-cPIKAN |  |  |  |  |
|  | PINN (MLP) |  |  |  |  |
| **앨런-칸 방정식** | Scaled-cPIKAN |  |  |  |  |
|  | PINN (MLP) |  |  |  |  |
| **반응-확산 방정식** | Scaled-cPIKAN |  |  |  |  |
|  | PINN (MLP) |  |  |  |  |

이러한 체계적인 검증 절차를 통해 Scaled-cPIKAN이 "몇 배 더 높은 정확도와 빠른 수렴 속도를 달성한다"는 논문의 핵심 주장을 객관적으로 평가할 수 있다 \[1\]. 성공적인 결과는 단순히 "더 나은 성능"을 의미하는 것이 아니라, "기존 방법들이 실패했던 바로 그 유형의 문제에서 더 나은 성능"을 의미하므로, 제안된 아키텍처의 근본적인 가치를 입증하게 된다.

---

## **제 5장: 비판적 분석 및 향후 연구 방향**

Scaled-cPIKAN은 편미분방정식 해결을 위한 강력한 프레임워크를 제시하지만, 그 성능 기여도에 대한 심층적인 이해와 잠재적 한계, 그리고 미래 발전 가능성을 탐색하는 것은 매우 중요하다. 이 장에서는 제안된 모델에 대한 비판적 분석을 수행하고, 향후 연구를 위한 구체적인 방향을 제시한다.

### **5.1. 제안된 절제 연구: 성능 향상 요인 해부**

복잡한 모델의 성능 향상이 어떤 구성 요소에서 비롯되었는지를 과학적으로 검증하기 위해서는 절제 연구(ablation study)가 필수적이다 \[46, 47\]. Scaled-cPIKAN의 핵심 주장은 cKAN과 영역 스케일링의 조합이 성능의 핵심이라는 것이다. 이 주장은 각 구성 요소를 제거하여 테스트해야 한다.

#### **방법론**

1. **모델 1 (전체 모델):** 완전한 Scaled-cPIKAN 모델. 이는 성능의 기준선이 된다.  
2. **모델 2 (스케일링 제거):** 초기 영역 스케일링 없이 cKAN을 사용하는 모델. 입력 좌표가 신경망에 직접 공급된다. 이 모델은 성능이 매우 저조하거나 훈련에 실패할 것으로 예상되며, 이는 스케일링 단계의 필요성을 증명한다.  
3. **모델 3 (cKAN 제거):** 영역 스케일링 전처리를 포함하지만, cKAN 대신 표준 MLP를 사용하는 PINN 모델. 이 모델은 성능 향상이 스케일링 자체에서 얼마나 비롯되었는지, 그리고 cKAN 아키텍처가 추가적으로 얼마나 기여하는지를 시험한다.

#### **분석**

이 세 가지 모델의 L2 오차와 수렴성을 대표적인 벤치마크(예: 헬름홀츠 방정식)에서 비교한다. 이 결과는 각 구성 요소의 기여도를 정량화하여 Scaled-cPIKAN의 성공 요인을 명확히 밝혀줄 것이다.

### **5.2. 한계 및 미해결 과제**

* **계산 비용:** cKAN은 B-스플라인 KAN보다 효율적이지만, 각 다항식 항에 대한 계수 때문에 동일한 크기의 MLP보다 여전히 더 많은 파라미터를 가질 수 있다. 파라미터 수와 메모리 사용량 대 성능 간의 상세한 트레이드오프 분석이 필요하다.  
* **고차원으로의 확장성:** 논문은 1D 및 2D 문제에 초점을 맞추고 있다. 모든 신경망 방법론의 주요 과제인 '차원의 저주'에 대해 Scaled-cPIKAN이 3D 또는 더 높은 차원의 문제에서 어떻게 작동하는지는 아직 해결되지 않은 문제이다.  
* **비직사각형 도메인:** 아핀 영역 스케일링은 직사각형 도메인에 대해서는 간단하다. 복잡하고 비직사각형인 기하학적 구조에 이 방법을 적용하는 것(예를 들어, 더 복잡한 매핑이 필요함)은 향후 연구의 핵심 영역이다.  
* **이론적 보증:** 경험적 결과는 강력하지만 \[1\], 이 논문은 주로 경험적 연구이다. 다른 PIKAN에 대해 수행된 것처럼 NTK 이론을 사용하여 Scaled-cPIKAN 아키텍처의 수렴 특성과 최적화 환경을 이해하기 위한 더 엄격한 이론적 분석이 향후 연구 과제로 남아있다 \[23, 48\].

### **5.3. 향후 연구 방향**

* **하이브리드 아키텍처:** Scaled-cPIKAN을 어텐션 메커니즘(예: AC-PKAN \[19\])이나 영역 분할 방법(예: XPINN \[4\])과 같은 다른 고급 기법과 결합하여 훨씬 더 복잡한 문제를 해결하는 방안을 탐색할 수 있다.  
* **적응형 다항식 차수:** 네트워크의 다른 부분이나 다른 문제에 대해 체비쇼프 차수 K를 동적으로 또는 적응적으로 선택하는 방법을 연구하여, 표현력과 복잡성 간의 트레이드오프를 최적화할 수 있다.  
* **광범위한 응용:** Scaled-cPIKAN 프레임워크를 계산 유체 역학, 고체 역학(저자들이 다른 연구에서 다룬 EPi-cKANs처럼 \[49, 50, 51\]), 그리고 양자 역학과 같은 더 넓은 범위의 도전적인 과학 및 공학 문제에 적용할 수 있다.

결론적으로, Scaled-cPIKAN은 효과적인 방법론을 제시하면서도 물리 정보 모델을 위한 새로운 "설계 공간"을 열어준다. 이 연구의 진정한 장기적 가치는 Scaled-cPIKAN 모델 자체뿐만 아니라, 그것이 검증하는 원리, 즉 **선택된 기저 함수의 수학적 제약 조건을 존중하기 위해 신경망 아키텍처와 입력 변환을 공동으로 설계하는 것**에 있다.

이러한 관점에서 볼 때, 향후 연구는 단순히 Scaled-cPIKAN을 개선하는 것을 넘어, 그 핵심 원리를 일반화하여 완전히 새로운 종류의 고도로 전문화되고 효과적인 물리 정보 모델군을 창출하는 방향으로 나아갈 수 있다. 예를 들어, 진동 문제에 강력한 푸리에 급수와 같은 다른 수학적 기저를 선택하고, 그에 맞는 입력 변환 및 네트워크 레이어를 설계하여 PINN 프레임워크 내에서 효율적으로 구현하는 "기저 정보 신경망 설계(Basis-Informed Network Design)" 패러다임을 탐색할 수 있다. Scaled-cPIKAN은 이 광범위한 패러다임에 대한 최초의 성공적인 개념 증명으로 볼 수 있으며, 이는 향후 과학적 머신러닝 분야의 발전에 중요한 이정표가 될 것이다.

#### **참고 자료**

1. \[2501.02762\] Scaled-cPIKANs: Domain Scaling in Chebyshev-based Physics-informed Kolmogorov-Arnold Networks \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/abs/2501.02762](https://arxiv.org/abs/2501.02762)  
2. Scaled-cPIKANs: Domain Scaling in Chebyshev-based Physics-informed Kolmogorov-Arnold Networks \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/html/2501.02762v1](https://arxiv.org/html/2501.02762v1)  
3. The basic idea of PINNs-WE framework in solving problems with strong nonlinear discontinuities \- ResearchGate, 7월 1, 2025에 액세스, [https://www.researchgate.net/figure/The-basic-idea-of-PINNs-WE-framework-in-solving-problems-with-strong-nonlinear\_fig2\_376484041](https://www.researchgate.net/figure/The-basic-idea-of-PINNs-WE-framework-in-solving-problems-with-strong-nonlinear_fig2_376484041)  
4. Physics-informed neural networks \- Wikipedia, 7월 1, 2025에 액세스, [https://en.wikipedia.org/wiki/Physics-informed\_neural\_networks](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)  
5. (PDF) A comprehensive and FAIR comparison between MLP and KAN representations for differential equations and operator networks \- ResearchGate, 7월 1, 2025에 액세스, [https://www.researchgate.net/publication/385455718\_A\_comprehensive\_and\_FAIR\_comparison\_between\_MLP\_and\_KAN\_representations\_for\_differential\_equations\_and\_operator\_networks](https://www.researchgate.net/publication/385455718_A_comprehensive_and_FAIR_comparison_between_MLP_and_KAN_representations_for_differential_equations_and_operator_networks)  
6. Physics-Informed Neural Networks: Minimizing Residual Loss with Wide Networks and Effective Activations \- IJCAI, 7월 1, 2025에 액세스, [https://www.ijcai.org/proceedings/2024/0647.pdf](https://www.ijcai.org/proceedings/2024/0647.pdf)  
7. \[2505.11117\] Dual-Balancing for Physics-Informed Neural Networks \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/abs/2505.11117](https://arxiv.org/abs/2505.11117)  
8. \[D\] What is the point of physics-informed neural networks if you need to know the actual physics? \- Reddit, 7월 1, 2025에 액세스, [https://www.reddit.com/r/MachineLearning/comments/12lzzv6/d\_what\_is\_the\_point\_of\_physicsinformed\_neural/](https://www.reddit.com/r/MachineLearning/comments/12lzzv6/d_what_is_the_point_of_physicsinformed_neural/)  
9. Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/html/2009.04544v5](https://arxiv.org/html/2009.04544v5)  
10. Scaling physics-informed neural networks to large domains by using domain decomposition, 7월 1, 2025에 액세스, [https://openreview.net/forum?id=o1WiAZiw\_CE](https://openreview.net/forum?id=o1WiAZiw_CE)  
11. When and why PINNs fail to train: A Neural Tangent Kernel perspective \- ResearchGate, 7월 1, 2025에 액세스, [https://www.researchgate.net/publication/355304640\_When\_and\_why\_PINNs\_fail\_to\_train\_A\_Neural\_Tangent\_Kernel\_perspective](https://www.researchgate.net/publication/355304640_When_and_why_PINNs_fail_to_train_A_Neural_Tangent_Kernel_perspective)  
12. when and why pinns fail to train: a neural tangent \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/pdf/2007.14527](https://arxiv.org/pdf/2007.14527)  
13. Dynamic Weight Strategy of Physics-Informed Neural Networks for the 2D Navier–Stokes Equations \- PMC, 7월 1, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9497516/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9497516/)  
14. Dynamic Weight Strategy of Physics-Informed Neural Networks for the 2D Navier–Stokes Equations \- ResearchGate, 7월 1, 2025에 액세스, [https://www.researchgate.net/publication/363343067\_Dynamic\_Weight\_Strategy\_of\_Physics-Informed\_Neural\_Networks\_for\_the\_2D\_Navier-Stokes\_Equations](https://www.researchgate.net/publication/363343067_Dynamic_Weight_Strategy_of_Physics-Informed_Neural_Networks_for_the_2D_Navier-Stokes_Equations)  
15. Kolmogorov-Arnold Networks (KANs): A Guide With Implementation | DataCamp, 7월 1, 2025에 액세스, [https://www.datacamp.com/tutorial/kolmogorov-arnold-networks](https://www.datacamp.com/tutorial/kolmogorov-arnold-networks)  
16. Understanding the Kolmogorov-Arnold Network | by NeuroCortex.AI | Medium, 7월 1, 2025에 액세스, [https://medium.com/@theagipodcast/understanding-the-kolmogorov-arnold-network-52e7232f8749](https://medium.com/@theagipodcast/understanding-the-kolmogorov-arnold-network-52e7232f8749)  
17. adithyarao3103/PINN: Physics Informed Neural Networks \- GitHub, 7월 1, 2025에 액세스, [https://github.com/adithyarao3103/PINN](https://github.com/adithyarao3103/PINN)  
18. Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/pdf/2405.07200](https://arxiv.org/pdf/2405.07200)  
19. \[Literature Review\] AC-PKAN: Attention-Enhanced and Chebyshev Polynomial-Based Physics-Informed Kolmogorov-Arnold Networks \- Moonlight | AI Colleague for Research Papers, 7월 1, 2025에 액세스, [https://www.themoonlight.io/en/review/ac-pkan-attention-enhanced-and-chebyshev-polynomial-based-physics-informed-kolmogorov-arnold-networks](https://www.themoonlight.io/en/review/ac-pkan-attention-enhanced-and-chebyshev-polynomial-based-physics-informed-kolmogorov-arnold-networks)  
20. Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/html/2405.07200v1](https://arxiv.org/html/2405.07200v1)  
21. SynodicMonth/ChebyKAN: Kolmogorov-Arnold Networks ... \- GitHub, 7월 1, 2025에 액세스, [https://github.com/SynodicMonth/ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)  
22. Representation Meets Optimization: Training PINNs and PIKANs for Gray-Box Discovery in Systems Pharmacology \- PubMed, 7월 1, 2025에 액세스, [https://pubmed.ncbi.nlm.nih.gov/40297233/](https://pubmed.ncbi.nlm.nih.gov/40297233/)  
23. \[Literature Review\] Neural Tangent Kernel Analysis to Probe Convergence in Physics-informed Neural Solvers: PIKANs vs. PINNs \- Moonlight | AI Colleague for Research Papers, 7월 1, 2025에 액세스, [https://www.themoonlight.io/en/review/neural-tangent-kernel-analysis-to-probe-convergence-in-physics-informed-neural-solvers-pikans-vs-pinns](https://www.themoonlight.io/en/review/neural-tangent-kernel-analysis-to-probe-convergence-in-physics-informed-neural-solvers-pikans-vs-pinns)  
24. What is Learning Rate in Machine Learning? \- IBM, 7월 1, 2025에 액세스, [https://www.ibm.com/think/topics/learning-rate](https://www.ibm.com/think/topics/learning-rate)  
25. Using an Appropriate Scale to Pick Hyperparameters, Hyperparameters Tuning in Practice: Pandas vs. Caviar | by Ahmet Taşdemir | Medium, 7월 1, 2025에 액세스, [https://medium.com/@ahmettsdmr1312/using-an-appropriate-scale-to-pick-hyperparameters-hyperparameters-tuning-in-practice-pandas-vs-52b6f555c886](https://medium.com/@ahmettsdmr1312/using-an-appropriate-scale-to-pick-hyperparameters-hyperparameters-tuning-in-practice-pandas-vs-52b6f555c886)  
26. Mastering Hyperparameters: Learning Rate, Batch Size, and More | by Sanjay Dutta, PhD, 7월 1, 2025에 액세스, [https://medium.com/@sanjay\_dutta/mastering-hyperparameters-learning-rate-batch-size-and-more-e3b4df6624dc](https://medium.com/@sanjay_dutta/mastering-hyperparameters-learning-rate-batch-size-and-more-e3b4df6624dc)  
27. Allen-Cahn equation — DeepXDE 1.14.1.dev8+gb944422 documentation, 7월 1, 2025에 액세스, [https://deepxde.readthedocs.io/en/latest/demos/pinn\_forward/allen.cahn.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html)  
28. Hyperparameter Tuning Techniques in Machine Learning Engineering \- DataScienceCentral.com, 7월 1, 2025에 액세스, [https://www.datasciencecentral.com/hyperparameter-tuning-techniques-in-machine-learning-engineering/](https://www.datasciencecentral.com/hyperparameter-tuning-techniques-in-machine-learning-engineering/)  
29. Allen-Cahn equation: Comparison of predictive and analytical solutions... \- ResearchGate, 7월 1, 2025에 액세스, [https://www.researchgate.net/figure/Allen-Cahn-equation-Comparison-of-predictive-and-analytical-solutions-of-Case-1-and-Case\_fig18\_363257082](https://www.researchgate.net/figure/Allen-Cahn-equation-Comparison-of-predictive-and-analytical-solutions-of-Case-1-and-Case_fig18_363257082)  
30. okada39/pinn\_wave: Physics Informed Neural Network (PINN) for the wave equation. \- GitHub, 7월 1, 2025에 액세스, [https://github.com/okada39/pinn\_wave](https://github.com/okada39/pinn_wave)  
31. Fine-tuning Models: Hyperparameter Optimization \- Encord, 7월 1, 2025에 액세스, [https://encord.com/blog/fine-tuning-models-hyperparameter-optimization/](https://encord.com/blog/fine-tuning-models-hyperparameter-optimization/)  
32. Constrained Self-Adaptive Physics-Informed Neural Networks with ResNet Block-Enhanced Network Architecture \- MDPI, 7월 1, 2025에 액세스, [https://www.mdpi.com/2227-7390/11/5/1109](https://www.mdpi.com/2227-7390/11/5/1109)  
33. murnanedaniel/Dynamic-Loss-Weighting: A small collection of tools to manage deep learning with multiple sources of loss \- GitHub, 7월 1, 2025에 액세스, [https://github.com/murnanedaniel/Dynamic-Loss-Weighting](https://github.com/murnanedaniel/Dynamic-Loss-Weighting)  
34. Self-adaptive weight balanced physics-informed neural networks for solving complex coupling equations \- SPIE Digital Library, 7월 1, 2025에 액세스, [https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13555/135553X/Self-adaptive-weight-balanced-physics-informed-neural-networks-for-solving/10.1117/12.3064996.full](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13555/135553X/Self-adaptive-weight-balanced-physics-informed-neural-networks-for-solving/10.1117/12.3064996.full)  
35. "When and why physics-informed neural networks fail to train" by Paris Perdikaris \- YouTube, 7월 1, 2025에 액세스, [https://www.youtube.com/watch?v=xvOsV106kuA](https://www.youtube.com/watch?v=xvOsV106kuA)  
36. Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism \- CEUR-WS, 7월 1, 2025에 액세스, [https://ceur-ws.org/Vol-2964/article\_68.pdf](https://ceur-ws.org/Vol-2964/article_68.pdf)  
37. BO-SA-PINNs: Self-adaptive physics-informed neural networks based on Bayesian optimization for automatically designing PDE solvers \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/html/2504.09804v1](https://arxiv.org/html/2504.09804v1)  
38. 数值分析/PDE分析/经典分析与ODE/泛函分析2025\_1\_7, 7월 1, 2025에 액세스, [https://www.arxivdaily.com/thread/62850](https://www.arxivdaily.com/thread/62850)  
39. Helmholtz equation \- Wikipedia, 7월 1, 2025에 액세스, [https://en.wikipedia.org/wiki/Helmholtz\_equation](https://en.wikipedia.org/wiki/Helmholtz_equation)  
40. Helmhurts \- Almost looks like work, 7월 1, 2025에 액세스, [https://jasmcole.com/2014/08/25/helmhurts/](https://jasmcole.com/2014/08/25/helmhurts/)  
41. Lecture Notes 5 ; Helmholtz Equation and High Frequency Approximations, 7월 1, 2025에 액세스, [https://www.csc.kth.se/utbildning/kth/kurser/DN2255/ndiff12/Lecture5.pdf](https://www.csc.kth.se/utbildning/kth/kurser/DN2255/ndiff12/Lecture5.pdf)  
42. High Accuracy Benchmark Problems for Allen-Cahn and Cahn-Hilliard Dynamics, 7월 1, 2025에 액세스, [https://users.math.msu.edu/users/promislo/Preprints/CH\_Benchmark\_accepted.pdf](https://users.math.msu.edu/users/promislo/Preprints/CH_Benchmark_accepted.pdf)  
43. Allen–Cahn equation \- Wikipedia, 7월 1, 2025에 액세스, [https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn\_equation](https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation)  
44. Evaluation Metrics in Machine Learning \- GeeksforGeeks, 7월 1, 2025에 액세스, [https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/](https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/)  
45. Forecasting Time Series \- Evaluation Metrics \- AutoGluon 1.3.2 documentation, 7월 1, 2025에 액세스, [https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-metrics.html](https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-metrics.html)  
46. Walking into the journey of a Ph.D. researcher (5): The ablation study | by Hajar Zankadi, 7월 1, 2025에 액세스, [https://medium.com/@hajar.zankadi/walking-into-the-journey-of-a-ph-d-researcher-5-the-ablation-study-ed6a1824e741](https://medium.com/@hajar.zankadi/walking-into-the-journey-of-a-ph-d-researcher-5-the-ablation-study-ed6a1824e741)  
47. Ablation Programming for Machine Learning \- DiVA portal, 7월 1, 2025에 액세스, [https://www.diva-portal.org/smash/get/diva2:1349978/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1349978/FULLTEXT01.pdf)  
48. Neural Tangent Kernel Analysis to Probe Convergence in Physics-informed Neural Solvers: PIKANs vs. PINNs \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/html/2506.07958v1](https://arxiv.org/html/2506.07958v1)  
49. EPi-cKANs: Elasto-Plasticity Informed Kolmogorov-Arnold Networks Using Chebyshev Polynomials | AI Research Paper Details \- AIModels.fyi, 7월 1, 2025에 액세스, [https://www.aimodels.fyi/papers/arxiv/epi-ckans-elasto-plasticity-informed-kolmogorov-arnold](https://www.aimodels.fyi/papers/arxiv/epi-ckans-elasto-plasticity-informed-kolmogorov-arnold)  
50. EPi-cKANs: Elasto-Plasticity Informed Kolmogorov-Arnold Networks Using Chebyshev Polynomials \- ResearchGate, 7월 1, 2025에 액세스, [https://www.researchgate.net/publication/384938811\_EPi-cKANs\_Elasto-Plasticity\_Informed\_Kolmogorov-Arnold\_Networks\_Using\_Chebyshev\_Polynomials](https://www.researchgate.net/publication/384938811_EPi-cKANs_Elasto-Plasticity_Informed_Kolmogorov-Arnold_Networks_Using_Chebyshev_Polynomials)  
51. \[2410.10897\] EPi-cKANs: Elasto-Plasticity Informed Kolmogorov-Arnold Networks Using Chebyshev Polynomials \- arXiv, 7월 1, 2025에 액세스, [https://arxiv.org/abs/2410.10897](https://arxiv.org/abs/2410.10897)