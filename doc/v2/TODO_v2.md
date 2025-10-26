# Scaled-cPIKAN 성능 개선 TODO (v2)

**버전**: v2 (성능 개선)  
**현재 Phase**: v2-P0 완료 ✅  
**다음 Phase**: v2-P1 (슬라이딩 윈도우 추론)  
**최종 업데이트**: 2025년 10월 26일  
**참조 문서**: `doc/v2/성능_개선_연구조사_보고서.md`  
**핵심 목표**: 🎯 **9344×7000급 초대형 PSI 영상 실시간 처리**

**Phase 체계**:
- v2-P0: 고급 기능 기본 활성화 (동적 가중치 + 적응형 샘플링) ✅ 완료
- v2-P1: 대규모 영상 처리 인프라 (슬라이딩 윈도우 + AMP)
- v2-P2: 데이터 파이프라인 최적화
- v2-P3: 훈련 전략 고도화
- v2-P4+: 선택적 기능 및 실험

---

## 📋 진행 상황 요약

### ✅ 완료된 항목 (v2-P0)
- [x] 그래디언트 클리핑 구현
- [x] Mixed Precision Training (AMP) 적용
- [x] Early Stopping 구현
- [x] 성능 개선 연구 조사 완료
- [x] 동적 손실 가중치 (`DynamicWeightedLoss`) 기본 활성화 ⭐
- [x] 적응형 콜로케이션 샘플링 (`AdaptiveResidualSampler`) 통합 ⭐

**📄 v2-P0 완료 보고서**: `doc/v2/reports/P0_작업_완료_보고서.md`

### 🔴 진행 중
(없음 - 다음 우선순위 작업 준비 중)

---

## 🔴 v2-P1: 최우선 과제 (즉시 착수, 1-2일 이내)

### 1. 슬라이딩 윈도우 추론 + 오버랩 블렌딩
**우선순위**: 🔴🔴🔴 CRITICAL  
**예상 시간**: 1-2일  
**담당**: 핵심 개발팀

**작업 항목**:
- [ ] `src/utils/tiling.py` 신규 생성
  - [ ] `tile_image()` 함수 구현 (512×512 타일, 128px 오버랩)
  - [ ] `blend_tiles()` 함수 구현 (Hanning 윈도우 블렌딩)
  - [ ] 성능 테스트 (9344×7000 이미지)
- [ ] `examples/run_psi_pipeline.py` 플래그 추가
  - [ ] `--tiled-infer` 플래그
  - [ ] `--tile-size` (기본값: 512)
  - [ ] `--tile-overlap` (기본값: 128)
- [ ] 단위 테스트 작성 (`tests/test_tiling.py`)
- [ ] 벤치마크 및 문서화

**성공 지표**:
- 9344×7000 이미지 처리 가능 (OOM 없음)
- GPU 메모리 사용량 < 6GB
- 전체 이미지 상대 L2 오차 < 5e-3

**참고**: 보고서 §2.1, U-Net (Ronneberger et al., 2015)

---

### 2. AMP 기본 활성화 + 체크포인팅/그래디언트 누적 토글
**우선순위**: 🔴🔴🔴 CRITICAL  
**예상 시간**: 1-2일  
**담당**: 훈련 파이프라인 팀

**작업 항목**:
- [ ] `src/train.py` 수정
  - [ ] `--amp` 플래그 기본 활성화
  - [ ] `--checkpoint` 옵션 추가 (활성화 체크포인팅)
  - [ ] `--grad-accum-steps` 옵션 추가 (기본값: 1)
  - [ ] GradScaler 통합
- [ ] `examples/run_psi_pipeline.py` 플래그 전파
- [ ] 메모리 사용량 측정 및 비교
- [ ] 문서 업데이트 (`doc/guides/`)

**성공 지표**:
- 메모리 사용량 30-50% 절감
- 훈련 속도 1.5-2배 향상 (Tensor Core 활용)
- 수치 안정성 유지 (NaN/Inf 없음)

**참고**: 보고서 §2.4

---

## 🔴 v2-P2: 고우선순위 (1주 이내)

### 3. 스트리밍/메모리맵 데이터 파이프라인
**우선순위**: 🔴🔴 HIGH  
**예상 시간**: 1일  
**담당**: 데이터 파이프라인 팀

**작업 항목**:
- [ ] `src/data.py` 수정
  - [ ] `MemmapImageDataset` 클래스 구현
  - [ ] `WaferPatchDataset`에 memmap 옵션 추가
  - [ ] DataLoader 최적화 (`pin_memory=True`, `prefetch_factor=2`)
- [ ] 대용량 NPY 파일 테스트 (>10GB)
- [ ] I/O 병목 벤치마크
- [ ] 단위 테스트 작성

**성공 지표**:
- 전체 데이터셋 크기와 무관한 메모리 사용량
- I/O 병목 완화 (DataLoader 효율 >80%)

**참고**: 보고서 §2.5

---

### 4. 패치 경계 일관성 손실
**우선순위**: 🔴🔴 HIGH  
**예상 시간**: 1-2일  
**담당**: 손실 함수 팀

**작업 항목**:
- [ ] `src/loss.py` 수정
  - [ ] `interface_consistency_loss()` 함수 구현
  - [ ] Augmented Lagrangian 옵션 추가
  - [ ] 오버랩 마스크 생성 유틸리티
- [ ] 훈련 루프 통합
- [ ] `--interface-weight` 플래그 추가 (기본값: 0.1)
- [ ] 경계 영역 L2 오차 측정
- [ ] 단위 테스트 작성

**성공 지표**:
- 경계 seam artifact 제거
- 오버랩 영역 L2 오차 < 1e-3
- 전체 일관성 5-10% 개선

**참고**: 보고서 §2.6

---

## 🔴 v2-P3: 중요 (2-3주 이내)

### 5. 멀티해상도 커리큘럼 학습
**우선순위**: 🔴 HIGH  
**예상 시간**: 2-3일  
**담당**: 훈련 전략 팀

**작업 항목**:
- [ ] `examples/train_curriculum.py` 신규 작성
  - [ ] 1/4 → 1/2 → full-res 단계별 학습 파이프라인
  - [ ] 체크포인트 기반 전이 학습
  - [ ] 해상도 전환 스케줄 설정
- [ ] 다중 스케일 손실 옵션 추가
  - [ ] `--loss-multiscale w_low w_mid w_high` 플래그
- [ ] 수렴 속도 벤치마크
- [ ] 문서화 및 튜토리얼

**성공 지표**:
- 수렴 속도 30-50% 향상
- 발산 확률 감소
- 전역-지역 밸런스 개선

**참고**: 보고서 §2.3

---

### 6. 도메인 분해 병렬 학습 (DDP)
**우선순위**: 🔴 HIGH  
**예상 시간**: 3-5일  
**담당**: 병렬화 팀

**작업 항목**:
- [ ] `src/models.py` 수정
  - [ ] `DomainDecomposedcPIKAN` 클래스 구현
  - [ ] 서브도메인 라우팅 로직
  - [ ] 인터페이스 연속성 조건
- [ ] `examples/train_ddp_xpinns.py` 신규 작성
  - [ ] PyTorch DDP 설정
  - [ ] 인터페이스 손실 통합
  - [ ] 멀티-GPU 통신 최적화
- [ ] 2-4 GPU 환경 벤치마크
- [ ] 확장성 테스트 및 문서화

**성공 지표**:
- 4 GPU에서 3-3.5배 속도 향상
- 메모리 분산 (각 GPU는 1/N 서브도메인)
- 전역 일관성 유지

**참고**: 보고서 §2.2, XPINNs (Jagtap & Karniadakis, 2020)

---

## 🟡 v2-P4: 선택 사항 (필요 시)

### 7. DeepSpeed ZeRO-Offload 통합
**우선순위**: 🟡 MEDIUM  
**예상 시간**: 2-3일  
**담당**: 고급 최적화 팀

**작업 항목**:
- [ ] DeepSpeed 의존성 추가 (`requirements.txt`)
- [ ] `examples/train_deepspeed.py` 신규 작성
- [ ] `ds_config.json` 설정 파일 작성
- [ ] ZeRO Stage 2/3 테스트
- [ ] 성능 프로파일링

**성공 지표**:
- 대모델 (>1B 파라미터) 단일 GPU 훈련 가능
- 메모리 한계 극복

**참고**: 보고서 §2.4.4

---

## 🟡 보조 목표: 정확도/속도 개선 (1-2개월)

**참고**: P4 작업 완료! 동적 손실 가중치와 적응형 샘플링이 기본 활성화되었습니다.
- 📄 완료 보고서: `doc/v2/reports/P4_작업_완료_보고서.md`
- 🔧 통합 예제: `examples/solve_poisson_1d_advanced.py`
- 📊 벤치마크 테스트: `tests/test_performance_improvements.py`

---

## 🟡 보조 목표: 정확도/속도 개선 (1-2개월)

### 8. ConFIG 그래디언트 최적화
**우선순위**: 🟡 MEDIUM  
**예상 시간**: 3-5일

**작업 항목**:
- [ ] `src/train.py`에 `ConFIGOptimizer` 클래스 추가
- [ ] Dual Cone 프로젝션 구현
- [ ] 기존 Adam/L-BFGS와 호환성 확보
- [ ] 벤치마크 (안정성 20-30% 향상 측정)

**참고**: 보고서 §3.2.1

---

### 9. 부피 가중 샘플링 (VW-PINNs)
**우선순위**: 🟡 MEDIUM  
**예상 시간**: 2-3일

**작업 항목**:
- [ ] `src/data.py`에 `VolumeWeightedSampler` 클래스 추가
- [ ] KDE 기반 부피 근사
- [ ] `src/loss.py`에 가중 손실 함수 추가
- [ ] 비균일 샘플링 테스트

**참고**: 보고서 §3.1.1, Song et al. (2024)

---

### 10. Fourier Features 레이어
**우선순위**: 🟡 MEDIUM  
**예상 시간**: 2일

**작업 항목**:
- [ ] `src/models.py`에 `FourierFeatureLayer` 추가
- [ ] `Scaled_cPIKAN_FF` 변형 구현
- [ ] 고주파 문제 테스트 (Helmholtz)
- [ ] 하이퍼파라미터 튜닝 (`sigma`, `num_features`)

**참고**: 보고서 §3.3.1

---

### 11. 2차 도함수 벡터화
**우선순위**: 🟡 MEDIUM  
**예상 시간**: 2일

**작업 항목**:
- [ ] `src/loss.py`에 `laplacian_efficient()` 구현
- [ ] torch.func 기반 최적화 (PyTorch 2.x)
- [ ] 기존 Laplacian 계산 대체
- [ ] 성능 벤치마크 (20-40% 절감 확인)

**참고**: 보고서 §3.3.1, §3.5

---

## 🟢 장기 연구 (3-6개월)

### 12. Augmented Lagrangian 기반 BC/IC 강화
**우선순위**: 🟢 LOW  
**예상 시간**: 1주

**작업 항목**:
- [ ] `src/loss.py`에 `AugmentedLagrangianLoss` 클래스
- [ ] Poisson/Helmholtz A/B 테스트
- [ ] Hard Constraints 비교 평가

**참고**: 보고서 §3.5, Zhang et al. (2025)

---

### 13. Deep Ritz 변분 손실
**우선순위**: 🟢 LOW  
**예상 시간**: 1주

**작업 항목**:
- [ ] `--loss-mode=ritz|residual|weak` 플래그
- [ ] 에너지 함수 구현 (Poisson/타원형 PDE)
- [ ] 수치 적분 (Gauss-Chebyshev 사중점)

**참고**: 보고서 §3.6

---

### 14. 전이 학습 프레임워크
**우선순위**: 🟢 LOW  
**예상 시간**: 1주

**작업 항목**:
- [ ] `examples/solve_helmholtz_transfer_learning.py`
- [ ] k=π → k=2π → k=4π 단계별 학습
- [ ] 자동 커리큘럼 생성

**참고**: 보고서 §3.4, Mustajab et al. (2024)

---

### 15. Wavelet-KAN 변형
**우선순위**: 🟢 LOW  
**예상 시간**: 2주

**작업 항목**:
- [ ] `src/models.py`에 `WaveletKANLayer` 구현
- [ ] Mother wavelet 주파수 제어
- [ ] 불연속면/충격파 문제 테스트

**참고**: 보고서 §3.3.2, Meshir et al. (2025)

---

### 16. Neural Operator 통합 (FNO × cPIKAN)
**우선순위**: 🟢 LOW  
**예상 시간**: 2-3주

**작업 항목**:
- [ ] FNO 인코더 + cPIKAN 보정 헤드 아키텍처
- [ ] 하이브리드 훈련 전략
- [ ] 파라메트릭 PDE 벤치마크

**참고**: 보고서 §6.5.4

---

## 📊 성공 지표 (KPIs)

### 🔴 대규모 영상 처리 지표 (최우선)

| 지표 | 현재 (baseline) | Week 2 목표 | Month 1 목표 |
|------|----------------|------------|-------------|
| **처리 가능 최대 해상도** | 2048×2048 (OOM) | 9344×7000 (타일링) | 16K×16K |
| **9344×7000 추론 시간** | N/A (OOM) | < 30초 | < 15초 (DDP) |
| **피크 GPU 메모리** | 16GB (2K×2K) | 4-6GB (512 타일) | 3-4GB (최적화) |
| **경계 artifact (L2)** | N/A | < 1e-3 | < 1e-4 |
| **전체 이미지 상대 L2** | N/A | < 5e-3 | < 1e-3 |

### 🟡 정확도/속도 지표 (보조 목표)

| 지표 | 현재 | 1개월 후 | 3개월 후 |
|------|------|---------|---------|
| Helmholtz 1D 상대 L2 오차 | ~1e-4 | < 5e-5 | < 1e-5 |
| 3D Poisson 수렴 시간 | 실패/느림 | 30분 | 15분 |
| 훈련 안정성 (성공률) | 70% | 90% | 95% |
| 필요 콜로케이션 포인트 | 10,000 | 5,000 | 3,000 |

---

## 🔄 주간 체크포인트

### Week 1 (현재)
- [ ] P0-1: 슬라이딩 윈도우 구현
- [ ] P0-2: AMP 기본 활성화

### Week 2
- [ ] P1-3: 메모리맵 파이프라인
- [ ] P1-4: 패치 경계 일관성 손실
- [ ] 9344×7000 이미지 처리 검증

### Week 3-4
- [ ] P2-5: 멀티해상도 커리큘럼
- [ ] P2-6: 도메인 분해 DDP
- [ ] 성능 벤치마크 및 보고서

### Month 2+
- [ ] 보조 목표 (ConFIG, 부피 가중 샘플링 등)
- [ ] 정확도 개선 실험
- [ ] 논문 작성 준비

---

## 📚 참고 문서

- **주요**: `doc/v2/성능_개선_연구조사_보고서.md`
- **배경**: `doc/v2/대규모_영상에_대한_연구.md`
- **가이드**: `doc/v2/성능_향상_즉시_적용_가이드.md`
- **구현**: `doc/implementation/class_diagram_implementation.md`

---

## 🔗 관련 이슈 및 PR

(작업 시작 시 GitHub 이슈/PR 링크 추가)

---

**최종 업데이트**: 2025년 10월 26일  
**다음 리뷰**: 2025년 11월 2일 (Week 2)