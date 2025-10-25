# Scaled-cPIKANs 문서 가이드

본 폴더는 Scaled-cPIKANs 프로젝트의 상세 문서를 포함하고 있습니다.

---

## 📂 폴더 구조

```
doc/
├── 📕 manual.md              # 사용자 매뉴얼 (설치, 실행, 예제)
├── 📝 TODO.md                # 작업 계획 및 진행 상황
│
├── 📘 theory/                # 이론 및 설계 문서
│   ├── theoretical_background.md
│   │   └─→ PINN, KAN, Chebyshev 다항식, 도메인 스케일링 등 수학 이론
│   ├── Scaled-cPIKAN 구현 명세서.md
│   │   └─→ 알고리즘 수학적 정의, 구현 요구사항, 하이퍼파라미터 명세
│   └── Technical Design Document for Scaled-cPIKAN Algorithm Implementation.md
│       └─→ 고수준 아키텍처 설계 (영문)
│
├── 📗 implementation/        # 구현 상세 문서
│   └── class_diagram_implementation.md
│       └─→ 클래스 다이어그램, 14개 클래스 상세 설명, Mermaid 다이어그램
│
└── 📙 reports/               # 작업 완료 보고서 (아카이브)
    ├── P0_작업_완료_보고서.md
    ├── P1_작업_완료_보고서.md
    ├── P2_COMPLETION_SUMMARY.md
    └── 프로젝트_정리_완료_보고서.md
```

---

## 🎯 문서 사용 가이드

### 🚀 **처음 시작하시나요?**
1. **[manual.md](manual.md)** - 설치 및 실행 방법
2. **[theory/theoretical_background.md](theory/theoretical_background.md)** - 이론 학습

### 🔬 **알고리즘을 이해하고 싶으신가요?**
1. **[theory/Scaled-cPIKAN 구현 명세서.md](theory/Scaled-cPIKAN%20구현%20명세서.md)** - 구현 가이드
2. **[implementation/class_diagram_implementation.md](implementation/class_diagram_implementation.md)** - 코드 구조

### 👨‍💻 **개발에 참여하시나요?**
1. **[TODO.md](TODO.md)** - 현재 작업 계획
2. **[reports/](reports/)** - 과거 작업 기록

---

## 📖 주요 문서 설명

### 1️⃣ **manual.md** - 사용자 매뉴얼
- 설치 방법 (requirements.txt)
- 예제 실행 방법 (4가지 예제)
- 하이퍼파라미터 설정 가이드
- 전체 파이프라인 실행 방법
- 테스트 실행 방법

**대상**: 초보자, 사용자

---

### 2️⃣ **theory/theoretical_background.md** - 이론적 배경
**최신 작성** (2025-10-25)

상세 내용:
- **PINN**: Physics-Informed Neural Networks 이론
- **KAN**: Kolmogorov-Arnold Networks 개념
- **Chebyshev 다항식**: 정의, 직교성, 근사 이론, 수치적 안정성
- **도메인 스케일링**: 아핀 변환, 그래디언트 정규화
- **위상 재구성**: 간섭계 원리, 다중 파장 기법
- **최적화**: Adam vs L-BFGS, 학습률 스케줄링
- **Latin Hypercube Sampling**: 준-몬테카를로 방법
- **자동 미분**: PyTorch Autograd

**특징**:
- 📐 풍부한 수학 공식 (KaTeX)
- 📊 시각적 설명 (ASCII 그래프, 표)
- 📚 참고문헌 포함
- 🔗 구현 문서와 링크

**대상**: 연구자, 이론 학습자

---

### 3️⃣ **theory/Scaled-cPIKAN 구현 명세서.md** - 구현 명세
논문 충실 구현 가이드

상세 내용:
- 알고리즘 수학적 정의
- 구현 요구사항 (ChebyKANLayer, Scaled_cPIKAN, 손실 함수)
- 하이퍼파라미터 명세 (논문 권장값)
- 훈련 프로토콜 (2단계 최적화)
- 구현 검증 (단위 테스트, 통합 테스트, 벤치마크)

**특징**:
- ✅ 논문과의 일치성 체크리스트
- 📝 구체적인 코드 스니펫
- 🧪 검증 기준 명시

**대상**: 개발자, 구현자

---

### 4️⃣ **implementation/class_diagram_implementation.md** - 클래스 다이어그램
**최신 작성** (2025-10-25)

상세 내용:
- 전체 아키텍처 개요 (ASCII 다이어그램)
- **Mermaid 클래스 다이어그램** (시각적)
- 14개 클래스 상세 설명:
  - 모델: `ChebyKANLayer`, `Scaled_cPIKAN`, `UNet`
  - 손실 함수: `PhysicsInformedLoss`, `PinnReconstructionLoss`
  - 훈련: `Trainer`
  - 데이터: `LatinHypercubeSampler`, `WaferPatchDataset`
- 각 클래스의 속성, 메서드, 이론 연결
- Sequence 다이어그램 (워크플로우)

**특징**:
- 📊 풍부한 다이어그램
- 🔗 이론 문서로의 링크
- 📦 코드 참조 인덱스

**대상**: 개발자, 코드 리뷰어

---

### 5️⃣ **TODO.md** - 작업 계획
활발히 업데이트되는 문서

내용:
- P0 (Critical): 논문 일치성 (2/3 완료)
- P1 (High): 테스트 및 검증 (4/4 완료 ✅)
- P2 (Medium): 코드 정리 (1/7 완료)
- P3 (Low): 고급 기능 (0/3)

**전체 진행률**: 59% (10/17)

**대상**: 개발 팀, 프로젝트 매니저

---

### 6️⃣ **reports/** - 작업 완료 보고서
과거 작업 기록 보관

- **P0_작업_완료_보고서.md** (2025-10-21)
  - 학습률 스케줄러 추가
  - 입력 범위 검증 추가
  - 14개 신규 테스트

- **P1_작업_완료_보고서.md** (2025-10-23)
  - Chebyshev/아핀 스케일링 단위 테스트
  - Poisson 통합 테스트
  - Helmholtz 벤치마크 재현

- **P2_COMPLETION_SUMMARY.md** (2025-10-23)
  - 데이터 모듈 재구성
  - 문서 업데이트
  - Docstring 개선

- **프로젝트_정리_완료_보고서.md** (2025-10-21)
  - TODO 리스트 생성
  - 불필요한 파일 삭제

**대상**: 작업 이력 추적, 회고

---

## 🔗 문서 간 링크

문서들은 서로 연결되어 있습니다:

```
theory/theoretical_background.md
        ↕️ (상호 참조)
implementation/class_diagram_implementation.md

각 클래스 설명 → 이론 섹션
각 이론 섹션 → 구현 클래스
```

**링크 규칙**:
- `implementation/` → `../theory/`
- `theory/` → `../implementation/`
- `reports/` → `../TODO.md`

---

## 📝 문서 작성 규칙

### 언어
- **한글 위주** 작성
- 기술 용어/함수명은 원문 유지 (예: `ChebyKANLayer`, `einsum`)

### 형식
- Markdown 표준 준수
- 수식: KaTeX 형식 (`$...$`, `$$...$$`)
- 다이어그램: Mermaid 권장
- 코드 블록: 언어 명시 (```python)

### 구조
- 목차 포함 (앵커 링크)
- 섹션 번호 사용
- 예시 코드 포함
- 참고문헌 명시

---

## 🆘 도움이 필요하신가요?

| 질문 | 문서 |
|------|------|
| "어떻게 설치하나요?" | [manual.md](manual.md) |
| "PINN이 뭔가요?" | [theory/theoretical_background.md](theory/theoretical_background.md#physics-informed-neural-networks-pinn) |
| "ChebyKANLayer는 어떻게 구현되나요?" | [implementation/class_diagram_implementation.md](implementation/class_diagram_implementation.md#chebykan-layer) |
| "현재 작업 상황은?" | [TODO.md](TODO.md) |
| "논문과 일치하는지 확인하려면?" | [theory/Scaled-cPIKAN 구현 명세서.md](theory/Scaled-cPIKAN%20구현%20명세서.md#6-현재-구현-상태-분석) |

---

## 📅 최종 업데이트

- **날짜**: 2025-10-25
- **버전**: 1.0
- **작성자**: Scaled-cPIKAN 개발팀

---

**프로젝트 홈**: [GitHub - StreoCalibration/Scaled-cPIKANs](https://github.com/StreoCalibration/Scaled-cPIKANs)
