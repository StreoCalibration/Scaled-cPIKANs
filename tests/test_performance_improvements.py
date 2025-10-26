"""
성능 개선 벤치마크 테스트

이 스크립트는 다음 기능들의 성능을 정량적으로 비교합니다:
1. 기존 방식: PhysicsInformedLoss + LatinHypercubeSampler
2. 동적 가중치: DynamicWeightedLoss + LatinHypercubeSampler
3. 적응형 샘플링: PhysicsInformedLoss + AdaptiveResidualSampler
4. 통합 방식: DynamicWeightedLoss + AdaptiveResidualSampler ⭐

측정 지표:
- 수렴 속도 (목표 오차 도달 시간)
- 최종 오차 (L2 relative error)
- 필요한 콜로케이션 포인트 수
- 훈련 시간

테스트 문제: 1D Poisson 방정식
    u''(x) = -π²sin(πx),  x ∈ [0,1]
    u(0) = 0,  u(1) = 0
    분석해: u(x) = sin(πx)
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
import sys
import os

# src 디렉토리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import Scaled_cPIKAN
from src.loss import PhysicsInformedLoss, DynamicWeightedLoss
from src.data import LatinHypercubeSampler, AdaptiveResidualSampler


class PerformanceBenchmark(unittest.TestCase):
    """성능 벤치마크 테스트 클래스"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 환경 초기화"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*70}")
        print(f"🚀 성능 벤치마크 테스트 시작")
        print(f"{'='*70}")
        print(f"디바이스: {cls.device}")
        print(f"PyTorch 버전: {torch.__version__}")
        
        # 결과 저장 디렉토리
        cls.output_dir = Path('outputs/benchmark_results')
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1D Poisson 문제 설정
        cls.domain_min = 0.0
        cls.domain_max = 1.0
        
        # 훈련 설정
        cls.n_initial_points = 50
        cls.n_max_points = 200
        cls.adam_epochs = 1000  # 더 짧게 (빠른 테스트용)
        cls.target_error = 1e-3  # 목표 상대 오차
        
        # 모델 구조 (모든 테스트에서 동일)
        cls.layers_dims = [1, 32, 32, 1]
        cls.cheby_order = 4
        
        # 결과 저장용
        cls.results = {}
    
    def analytical_solution(self, x):
        """분석해: u(x) = sin(πx)"""
        return torch.sin(np.pi * x)
    
    def create_pde_residual_fn(self):
        """PDE 잔차 함수 생성"""
        def pde_residual_fn(model, points):
            """
            1D Poisson: u''(x) = -π²sin(πx)
            잔차: u'' + π²sin(πx) = 0
            """
            points.requires_grad_(True)
            u = model(points)
            
            # 1차 도함수
            u_x = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
            # 2차 도함수
            u_xx = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0]
            
            # 소스 항
            source = np.pi**2 * torch.sin(np.pi * points)
            
            # 잔차
            residual = u_xx + source
            return residual
        
        return pde_residual_fn
    
    def create_bc_fn(self):
        """경계 조건 함수 생성"""
        def bc_fn(model, points):
            """u(0) = 0, u(1) = 0"""
            u = model(points)
            return u  # 경계에서 0이어야 함
        
        return bc_fn
    
    def compute_l2_error(self, model, n_test=1000):
        """
        상대 L2 오차 계산
        
        Returns:
            float: 상대 L2 오차
        """
        model.eval()
        with torch.no_grad():
            x_test = torch.linspace(
                self.domain_min, self.domain_max, n_test, 
                device=self.device
            ).reshape(-1, 1)
            
            u_pred = model(x_test)
            u_exact = self.analytical_solution(x_test)
            
            l2_error = torch.sqrt(torch.mean((u_pred - u_exact)**2))
            l2_norm = torch.sqrt(torch.mean(u_exact**2))
            
            relative_error = (l2_error / l2_norm).item()
        
        return relative_error
    
    def train_model(self, model, loss_fn, sampler, method_name):
        """
        모델 훈련 및 성능 측정
        
        Args:
            model: PINN 모델
            loss_fn: 손실 함수
            sampler: 샘플러 (LatinHypercubeSampler 또는 AdaptiveResidualSampler)
            method_name: 방법 이름
            
        Returns:
            dict: 성능 메트릭
        """
        print(f"\n{'='*70}")
        print(f"📊 방법: {method_name}")
        print(f"{'='*70}")
        
        # 옵티마이저
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 메트릭 기록
        history = {
            'epoch': [],
            'loss': [],
            'error': [],
            'n_points': []
        }
        
        # 경계 포인트 (고정)
        bc_points_0 = torch.tensor([[self.domain_min]], device=self.device)
        bc_points_1 = torch.tensor([[self.domain_max]], device=self.device)
        bc_points_dicts = [
            {'points': bc_points_0},
            {'points': bc_points_1}
        ]
        
        # 훈련 시작 시간
        start_time = time.time()
        
        # 목표 오차 도달 에포크
        convergence_epoch = None
        
        # AdaptiveResidualSampler인 경우 정제 주기 설정
        is_adaptive = isinstance(sampler, AdaptiveResidualSampler)
        refinement_interval = 500 if is_adaptive else None
        
        model.train()
        
        for epoch in range(self.adam_epochs):
            # 샘플 포인트 가져오기
            if is_adaptive:
                pde_points = sampler.get_current_points()
            else:
                pde_points = sampler.sample()
            
            # 순전파
            optimizer.zero_grad()
            total_loss, loss_dict = loss_fn(
                model, pde_points, bc_points_dicts
            )
            
            # 역전파
            total_loss.backward()
            optimizer.step()
            
            # 적응형 샘플링: 잔차 업데이트 및 정제
            if is_adaptive and (epoch + 1) % refinement_interval == 0:
                # 완전히 독립적인 forward pass로 잔차 계산
                model.eval()
                # 현재 포인트의 새로운 복사본 생성
                with torch.enable_grad():  # gradient 계산 활성화
                    pde_points_copy = sampler.get_current_points().clone().detach().requires_grad_(True)
                    pde_residual_fn = self.create_pde_residual_fn()
                    residuals = pde_residual_fn(model, pde_points_copy)
                    # 계산 그래프와 분리
                    sampler.update_residuals(residuals.detach())
                
                refined = sampler.refine()
                if refined:
                    n_points = sampler.get_current_points().shape[0]
                    print(f"  [Epoch {epoch+1}] 정제 완료 → {n_points}개 포인트")
                
                model.train()
            
            # 오차 계산 (매 100 에포크마다)
            if (epoch + 1) % 100 == 0:
                error = self.compute_l2_error(model)
                n_points = pde_points.shape[0]
                
                history['epoch'].append(epoch + 1)
                history['loss'].append(total_loss.item())
                history['error'].append(error)
                history['n_points'].append(n_points)
                
                print(f"  Epoch {epoch+1:5d} | Loss: {total_loss.item():.4e} | "
                      f"Error: {error:.4e} | Points: {n_points}")
                
                # 목표 오차 도달 여부 확인
                if convergence_epoch is None and error < self.target_error:
                    convergence_epoch = epoch + 1
                    print(f"  ✅ 목표 오차 도달! (Epoch {convergence_epoch})")
        
        # 훈련 종료 시간
        end_time = time.time()
        training_time = end_time - start_time
        
        # 최종 오차
        final_error = self.compute_l2_error(model)
        final_n_points = pde_points.shape[0] if is_adaptive else self.n_initial_points
        
        # 결과 정리
        result = {
            'method': method_name,
            'convergence_epoch': convergence_epoch if convergence_epoch else 'Not converged',
            'final_error': final_error,
            'final_n_points': final_n_points,
            'training_time_seconds': training_time,
            'history': history
        }
        
        print(f"\n📈 결과 요약:")
        print(f"  수렴 에포크: {result['convergence_epoch']}")
        print(f"  최종 오차: {result['final_error']:.6e}")
        print(f"  최종 포인트 수: {result['final_n_points']}")
        print(f"  훈련 시간: {training_time:.2f}초")
        
        return result
    
    def test_1_baseline(self):
        """기존 방식: PhysicsInformedLoss + LatinHypercubeSampler"""
        print("\n" + "="*70)
        print("🔵 테스트 1: 기존 방식 (Baseline)")
        print("="*70)
        
        # 모델 생성
        model = Scaled_cPIKAN(
            layers_dims=self.layers_dims,
            cheby_order=self.cheby_order,
            domain_min=torch.tensor([self.domain_min]),
            domain_max=torch.tensor([self.domain_max])
        ).to(self.device)
        
        # 손실 함수
        base_loss_fn = PhysicsInformedLoss(
            pde_residual_fn=self.create_pde_residual_fn(),
            bc_fns=[self.create_bc_fn()],
            loss_weights={'pde': 1.0, 'bc': 10.0}
        )
        
        # 샘플러
        sampler = LatinHypercubeSampler(
            n_points=self.n_initial_points,
            domain_min=[self.domain_min],
            domain_max=[self.domain_max],
            device=self.device
        )
        
        # 훈련 및 측정
        result = self.train_model(model, base_loss_fn, sampler, 'Baseline')
        self.results['baseline'] = result
    
    def test_2_dynamic_weights(self):
        """동적 가중치: DynamicWeightedLoss + LatinHypercubeSampler"""
        print("\n" + "="*70)
        print("🟢 테스트 2: 동적 가중치")
        print("="*70)
        
        # 모델 생성
        model = Scaled_cPIKAN(
            layers_dims=self.layers_dims,
            cheby_order=self.cheby_order,
            domain_min=torch.tensor([self.domain_min]),
            domain_max=torch.tensor([self.domain_max])
        ).to(self.device)
        
        # 기본 손실 함수
        base_loss_fn = PhysicsInformedLoss(
            pde_residual_fn=self.create_pde_residual_fn(),
            bc_fns=[self.create_bc_fn()],
            loss_weights={'pde': 1.0, 'bc': 10.0}
        )
        
        # 동적 가중치 손실 함수
        dynamic_loss_fn = DynamicWeightedLoss(
            base_loss_fn=base_loss_fn,
            loss_names=['loss_pde', 'loss_bc'],
            alpha=1.5,
            learning_rate=0.025
        )
        
        # 샘플러
        sampler = LatinHypercubeSampler(
            n_points=self.n_initial_points,
            domain_min=[self.domain_min],
            domain_max=[self.domain_max],
            device=self.device
        )
        
        # 훈련 및 측정
        result = self.train_model(model, dynamic_loss_fn, sampler, 'Dynamic Weights')
        self.results['dynamic_weights'] = result
    
    def test_3_adaptive_sampling(self):
        """적응형 샘플링: PhysicsInformedLoss + AdaptiveResidualSampler"""
        print("\n" + "="*70)
        print("🟡 테스트 3: 적응형 샘플링")
        print("="*70)
        
        # 모델 생성
        model = Scaled_cPIKAN(
            layers_dims=self.layers_dims,
            cheby_order=self.cheby_order,
            domain_min=torch.tensor([self.domain_min]),
            domain_max=torch.tensor([self.domain_max])
        ).to(self.device)
        
        # 손실 함수
        base_loss_fn = PhysicsInformedLoss(
            pde_residual_fn=self.create_pde_residual_fn(),
            bc_fns=[self.create_bc_fn()],
            loss_weights={'pde': 1.0, 'bc': 10.0}
        )
        
        # 적응형 샘플러
        sampler = AdaptiveResidualSampler(
            n_initial_points=self.n_initial_points,
            n_max_points=self.n_max_points,
            domain_min=[self.domain_min],
            domain_max=[self.domain_max],
            refinement_ratio=0.2,
            residual_threshold_percentile=75.0,
            device=self.device
        )
        
        # 훈련 및 측정
        result = self.train_model(model, base_loss_fn, sampler, 'Adaptive Sampling')
        self.results['adaptive_sampling'] = result
    
    def test_4_combined(self):
        """통합 방식: DynamicWeightedLoss + AdaptiveResidualSampler ⭐"""
        print("\n" + "="*70)
        print("🔴 테스트 4: 통합 방식 (Dynamic + Adaptive) ⭐")
        print("="*70)
        
        # 모델 생성
        model = Scaled_cPIKAN(
            layers_dims=self.layers_dims,
            cheby_order=self.cheby_order,
            domain_min=torch.tensor([self.domain_min]),
            domain_max=torch.tensor([self.domain_max])
        ).to(self.device)
        
        # 기본 손실 함수
        base_loss_fn = PhysicsInformedLoss(
            pde_residual_fn=self.create_pde_residual_fn(),
            bc_fns=[self.create_bc_fn()],
            loss_weights={'pde': 1.0, 'bc': 10.0}
        )
        
        # 동적 가중치 손실 함수
        dynamic_loss_fn = DynamicWeightedLoss(
            base_loss_fn=base_loss_fn,
            loss_names=['loss_pde', 'loss_bc'],
            alpha=1.5,
            learning_rate=0.025
        )
        
        # 적응형 샘플러
        sampler = AdaptiveResidualSampler(
            n_initial_points=self.n_initial_points,
            n_max_points=self.n_max_points,
            domain_min=[self.domain_min],
            domain_max=[self.domain_max],
            refinement_ratio=0.2,
            residual_threshold_percentile=75.0,
            device=self.device
        )
        
        # 훈련 및 측정
        result = self.train_model(model, dynamic_loss_fn, sampler, 'Combined (Dynamic + Adaptive)')
        self.results['combined'] = result
    
    @classmethod
    def tearDownClass(cls):
        """결과 저장 및 요약"""
        print("\n" + "="*70)
        print("📊 최종 성능 비교")
        print("="*70)
        
        # 결과 테이블 출력
        print("\n| 방법 | 수렴 에포크 | 최종 오차 | 포인트 수 | 훈련 시간(초) |")
        print("|------|-------------|-----------|-----------|---------------|")
        
        for key, result in cls.results.items():
            method = result['method']
            conv_epoch = result['convergence_epoch']
            if conv_epoch == 'Not converged':
                conv_epoch_str = "미수렴"
            else:
                conv_epoch_str = f"{conv_epoch}"
            
            error = result['final_error']
            n_points = result['final_n_points']
            time_sec = result['training_time_seconds']
            
            print(f"| {method:30s} | {conv_epoch_str:11s} | {error:.6e} | "
                  f"{n_points:9d} | {time_sec:13.2f} |")
        
        # JSON 저장
        output_path = cls.output_dir / 'benchmark_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cls.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 결과 저장: {output_path}")
        
        # 개선 비율 계산
        if 'baseline' in cls.results and 'combined' in cls.results:
            baseline = cls.results['baseline']
            combined = cls.results['combined']
            
            print("\n" + "="*70)
            print("📈 통합 방식 vs 기존 방식 개선 비율")
            print("="*70)
            
            # 수렴 속도 개선
            if baseline['convergence_epoch'] != 'Not converged' and \
               combined['convergence_epoch'] != 'Not converged':
                speedup = baseline['convergence_epoch'] / combined['convergence_epoch']
                print(f"수렴 속도: {speedup:.2f}배 향상")
            
            # 오차 개선
            error_improvement = (baseline['final_error'] - combined['final_error']) / baseline['final_error'] * 100
            print(f"최종 오차: {error_improvement:.1f}% 개선")
            
            # 시간 비교
            time_ratio = combined['training_time_seconds'] / baseline['training_time_seconds']
            print(f"훈련 시간: {time_ratio:.2f}배")
        
        print("\n" + "="*70)
        print("🎉 벤치마크 테스트 완료!")
        print("="*70)


if __name__ == '__main__':
    # unittest 실행
    unittest.main(verbosity=2)
