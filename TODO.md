# Scaled-cPIKAN 프로젝트 TODO 리스트

**목표**: 논문 "Scaled-cPIKANs: Domain Scaling in Chebyshev-based Physics-informed Kolmogorov-Arnold Networks"의 알고리즘을 최대한 충실하게 구현

**관련 문서**: `doc/Scaled-cPIKAN 구현 명세서.md` 참조

---

## 우선순위 체계

- **P0 (Critical)**: 논문과의 일치성을 위한 필수 수정 사항
- **P1 (High)**: 구현 검증 및 품질 보증
- **P2 (Medium)**: 코드 품질 및 유지보수성 개선
- **P3 (Low)**: 선택적 기능 및 최적화

---

## P0: 핵심 알고리즘 수정 (Critical)

### 1. 학습률 스케줄러 추가 ✅
- **파일**: `src/train.py`
- **작업**: Trainer 클래스에 ExponentialLR 스케줄러 추가
- **논문 설정**: `gamma=0.9995`
- **수정 내용**:
  ```python
  # Trainer._train_adam() 메서드에 추가
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
  # 각 에포크 후 scheduler.step() 호출
  ```
- **검증**: Adam 훈련 중 학습률이 지수적으로 감소하는지 확인 ✅
- **테스트**: `tests/test_p0_implementation.py::TestLearningRateScheduler`
- **예상 소요**: 1시간
- **상태**: [x] 완료

### 2. 입력 범위 검증 추가 ✅
- **파일**: `src/models.py` - `ChebyKANLayer.forward()`
- **작업**: 입력이 [-1, 1] 범위를 벗어나면 경고 또는 오류 발생
- **수정 내용**:
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      # 디버그 모드에서 검증
      if torch.is_grad_enabled():  # 훈련 중에만 체크
          if not torch.all((x >= -1.0 - 1e-6) & (x <= 1.0 + 1e-6)):
              warnings.warn(f"Input to ChebyKANLayer outside [-1,1]: min={x.min():.6f}, max={x.max():.6f}")
  ```
- **검증**: 스케일링 없이 모델 실행 시 경고 발생 확인 ✅
- **테스트**: `tests/test_p0_implementation.py::TestInputRangeValidation`
- **예상 소요**: 30분
- **상태**: [x] 완료

### 3. 하이퍼파라미터 통일
- **파일**: 모든 `examples/*.py`
- **작업**: 논문 권장 설정으로 기본값 변경
- **변경 사항**:
  - `layers_dims=[2, 32, 32, 32, 1]` (2D 문제)
  - `cheby_order=3`
  - `adam_epochs=20000` (또는 10000 for quick tests)
  - `adam_lr=1e-3`
- **영향받는 파일**:
  - [ ] `examples/solve_helmholtz_1d.py`
  - [ ] `examples/solve_reconstruction_pinn.py`
  - [ ] `examples/solve_reconstruction_from_buckets.py`
  - [ ] `examples/run_pipeline.py`
- **예상 소요**: 2시간
- **상태**: [ ] 미완료

---

## P1: 테스트 및 검증 (High Priority)

### 4. 체비쇼프 다항식 단위 테스트
- **파일**: `tests/test_models.py` (새 테스트 추가)
- **작업**: 체비쇼프 다항식 계산 정확성 검증
- **테스트 케이스**:
  ```python
  def test_chebyshev_basis():
      """T_0=1, T_1=x, T_2=2x²-1, T_3=4x³-3x 검증"""
      # 구현 상세는 명세서 부록 A.1 참조
  ```
- **검증 기준**: 모든 차수에서 수학적 정의와 일치
- **예상 소요**: 2시간
- **상태**: [ ] 미완료

### 5. 아핀 스케일링 단위 테스트
- **파일**: `tests/test_models.py` (새 테스트 추가)
- **작업**: 아핀 스케일링 정확성 검증
- **테스트 케이스**:
  ```python
  def test_affine_scaling():
      """x_min→-1, x_max→1, midpoint→0 검증"""
      # 명세서 섹션 5.1.2 참조
  ```
- **검증 기준**: 경계값과 역변환 정확성
- **예상 소요**: 1시간
- **상태**: [ ] 미완료

### 6. Poisson 방정식 통합 테스트
- **파일**: `tests/test_integration.py` (새 테스트 추가)
- **작업**: 간단한 PDE로 end-to-end 테스트
- **문제**: 1D Poisson `u''(x) = -1`, `u(0)=0`, `u(1)=0`
- **분석 해**: `u(x) = x(1-x)/2`
- **검증 기준**: Relative L2 error < 1e-3
- **예상 소요**: 3시간
- **상태**: [ ] 미완료

### 7. Helmholtz 벤치마크 재현
- **파일**: `examples/solve_helmholtz_1d.py` 개선
- **작업**: 논문 Table 1 결과 재현
- **설정**: `k=4π`, Adam 20k epochs + L-BFGS 5 steps
- **목표**: Relative L2 error < 1e-4
- **검증**: 결과를 논문과 비교하여 로그에 기록
- **예상 소요**: 4시간 (실험 시간 포함)
- **상태**: [ ] 미완료

---

## P2: 코드 정리 및 리팩토링 (Medium Priority)

### 8. 불필요한 파일 삭제 ✅
- **작업**: 사용하지 않는 classical reconstruction 알고리즘 제거
- **삭제 대상**:
  - [x] `reconstruction/reconstruction.py` ✅ 삭제 완료
  - [x] `reconstruction/main.py` ✅ 삭제 완료
  - [x] `reconstruction/README.md` ✅ 삭제 완료
- **보존**: `reconstruction/data_generator.py` (예제에서 사용 중)
- **검증**: 테스트 통과 확인 완료 (9 tests OK)
- **상태**: [x] 완료

### 9. 데이터 생성 모듈 재구성
- **작업**: `reconstruction/data_generator.py` → `src/data_generator.py`로 이동
- **이유**: 논리적으로 `src/` 패키지에 포함되는 것이 적절
- **영향받는 파일**:
  - [ ] `examples/solve_reconstruction_pinn.py` - import 수정
  - [ ] `examples/solve_reconstruction_from_buckets.py` - import 수정
  - [ ] `examples/run_pipeline.py` - import 확인
- **수정 후 임포트**:
  ```python
  from src.data_generator import DEFAULT_WAVELENGTHS, generate_synthetic_data
  ```
- **예상 소요**: 1시간
- **상태**: [ ] 미완료

### 10. .gitignore 업데이트
- **파일**: `.gitignore`
- **작업**: 생성된 데이터 및 결과 디렉토리 추가
- **추가 항목**:
  ```
  # Generated data
  synthetic_data/
  reconstruction_data/
  reconstruction_from_buckets_results/
  reconstruction_pinn_results/
  
  # Generated outputs
  *.png
  *.npy
  models/
  
  # Python
  __pycache__/
  *.pyc
  .pytest_cache/
  ```
- **예상 소요**: 15분
- **상태**: [ ] 미완료

### 11. 중복 예제 통합 검토
- **작업**: 유사한 기능의 예제 스크립트 통합 가능성 검토
- **검토 대상**:
  - `train_bucket_pinn.py` vs `run_pipeline.py` - 기능 중복 여부 확인
  - `infer_bucket_pinn.py` - 독립 실행 가능 여부 확인
- **결정 기준**: 각 스크립트의 명확한 목적이 있으면 유지
- **예상 소요**: 2시간
- **상태**: [ ] 미완료

---

## P2: 문서 업데이트 (Medium Priority)

### 12. README.md 업데이트
- **파일**: `README.md`
- **작업**: 변경사항 반영 및 예제 명령어 업데이트
- **업데이트 항목**:
  - [ ] 하이퍼파라미터 기본값 설명 수정
  - [ ] 벤치마크 결과 추가 (Helmholtz 등)
  - [ ] 프로젝트 구조 설명에서 reconstruction/ 역할 명확화
  - [ ] 테스트 실행 방법 강조
- **예상 소요**: 2시간
- **상태**: [ ] 미완료

### 13. manual.md 업데이트
- **파일**: `doc/manual.md`
- **작업**: 사용자 매뉴얼에 변경사항 반영
- **업데이트 항목**:
  - [ ] 학습률 스케줄러 관련 설명 추가
  - [ ] 하이퍼파라미터 설정 가이드 업데이트
  - [ ] 새로운 테스트 실행 방법 추가
- **예상 소요**: 1.5시간
- **상태**: [ ] 미완료

### 14. 예제 docstring 개선
- **파일**: 모든 `examples/*.py`
- **작업**: 각 예제의 목적과 사용법을 명확히 문서화
- **형식**:
  ```python
  """
  Example: [명확한 제목]
  
  Purpose: [이 예제가 보여주는 것]
  
  Usage:
      python examples/script_name.py [options]
  
  Expected output:
      [예상되는 출력 설명]
  """
  ```
- **예상 소요**: 2시간
- **상태**: [ ] 미완료

---

## P3: 선택적 고급 기능 (Low Priority / Future Work)

### 15. 동적 손실 가중치 구현
- **파일**: `src/loss.py` (새 클래스 추가)
- **작업**: GradNorm 또는 uncertainty weighting 구현
- **참조**: 명세서 섹션 2.5
- **이점**: 손실 항 간 균형 자동 조정
- **예상 소요**: 8시간
- **상태**: [ ] 미완료 (Phase 3)

### 16. 적응형 콜로케이션 샘플링
- **파일**: `src/data.py` (새 클래스 추가)
- **작업**: 잔차가 큰 영역에 더 많은 포인트 샘플링
- **이점**: 효율적인 훈련
- **예상 소요**: 10시간
- **상태**: [ ] 미완료 (Phase 3)

### 17. 3D 문제로 확장
- **파일**: 새 예제 추가 `examples/solve_3d_problem.py`
- **작업**: 3D PDE에 대한 Scaled-cPIKAN 적용
- **문제 예시**: 3D Poisson, 3D Heat equation
- **검증**: 논문의 확장성 한계 검증
- **예상 소요**: 12시간
- **상태**: [ ] 미완료 (Phase 3)

---

## 진행 상황 요약

### 완료된 작업
- [x] 구현 명세서 작성 (`doc/Scaled-cPIKAN 구현 명세서.md`)
- [x] 불필요한 파일 삭제 (reconstruction/ 정리)
- [x] 학습률 스케줄러 추가 (P0-1)
- [x] 입력 범위 검증 추가 (P0-2)
- [x] P0 작업 테스트 작성 (14개 테스트)

### 진행 중
- [ ] P0 작업 (3개 항목)
- [ ] P1 작업 (4개 항목)

### 전체 진행률
- **P0 (Critical)**: 2/3 (67%) - 학습률 스케줄러 ✅, 입력 검증 ✅, 하이퍼파라미터 통일 ⏳
- **P1 (High)**: 0/4 (0%)
- **P2 (Medium)**: 1/7 (14%)
- **P3 (Low)**: 0/3 (0%)

**전체**: 3/17 (18%)

---

## 다음 단계 (권장 순서)

1. **이번 주**: P0 작업 완료 (학습률 스케줄러, 입력 검증, 하이퍼파라미터 통일)
2. **다음 주**: P1 테스트 작성 및 벤치마크 재현
3. **그 다음 주**: P2 코드 정리 및 문서 업데이트
4. **장기**: P3 고급 기능 (필요시)

---

## 기여 가이드

각 작업을 시작할 때:
1. 해당 항목을 `[ ]`에서 `[진행중]`으로 변경
2. 브랜치 생성: `git checkout -b feature/task-name`
3. 작업 완료 후 `[x]`로 변경하고 커밋
4. 테스트 실행: `python -m unittest discover tests`
5. PR 생성 또는 main에 머지

---

**마지막 업데이트**: 2025-10-21
**다음 리뷰 예정일**: 2025-10-28
