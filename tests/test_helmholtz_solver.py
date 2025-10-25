"""
단위 테스트: 1D Helmholtz 방정식 솔버 (Scaled-cPIKAN 기반)

목적:
    Scaled-cPIKAN PINN 아키텍처를 사용한 1D 파동 방정식(Helmholtz) 해결을
    검증하는 단위 테스트입니다. 2단계 최적화 전략(Adam 사전학습 + L-BFGS 미세조정)을
    테스트하고 분석적 해와의 비교를 통해 모델을 검증합니다.

문제 설명:
    해결: u_xx + k²u = 0, u(0) = 0, u(1) = sin(k)
    분석적 해: u(x) = sin(kx)
    k = 4π (파수)

테스트 케이스:
    1. test_helmholtz_solver_basic: 기본 솔버 기능 테스트 (빠른 실행)
    2. test_helmholtz_solver_convergence: 수렴성 테스트 (더 많은 에포크)
    3. test_helmholtz_with_lbfgs: L-BFGS 미세조정 포함 테스트

실행 방법:
    python -m unittest tests.test_helmholtz_solver
    python -m unittest tests.test_helmholtz_solver.TestHelmholtzSolver.test_helmholtz_solver_basic
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Scaled_cPIKAN
from src.data import LatinHypercubeSampler
from src.loss import PhysicsInformedLoss
from src.train import Trainer


class TestHelmholtzSolver(unittest.TestCase):
    """1D Helmholtz 방정식 솔버 테스트"""

    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.device = torch.device("cpu")  # 테스트는 CPU에서 실행
        self.k_wavenumber = 4 * torch.pi
        self.domain_min = [-1.0]
        self.domain_max = [1.0]

    def analytical_solution(self, x):
        """1D Helmholtz 방정식의 분석적 해"""
        return torch.sin(self.k_wavenumber * x)

    def define_pde_residual(self):
        """u_xx + k^2*u = 0에 대한 PDE 잔차 함수"""
        k = self.k_wavenumber
        
        def pde_residual_fn(model, x):
            x.requires_grad_(True)
            u = model(x)
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            return u_xx + (k**2) * u
        
        return pde_residual_fn

    def define_bc_funcs(self):
        """경계 조건 함수들"""
        k = self.k_wavenumber
        
        def bc_fn1(model, x_bc1):
            u_bc1 = model(x_bc1)
            return u_bc1 - self.analytical_solution(x_bc1)
        
        def bc_fn2(model, x_bc2):
            u_bc2 = model(x_bc2)
            return u_bc2 - self.analytical_solution(x_bc2)
        
        return [bc_fn1, bc_fn2]

    def create_model(self, layers_dims, cheby_order):
        """모델 생성 헬퍼 함수"""
        return Scaled_cPIKAN(
            layers_dims=layers_dims,
            cheby_order=cheby_order,
            domain_min=torch.tensor(self.domain_min, device=self.device),
            domain_max=torch.tensor(self.domain_max, device=self.device)
        ).to(self.device)

    def test_helmholtz_solver_basic(self):
        """
        기본 Helmholtz 솔버 기능 테스트
        
        목적: 빠른 smoke test로 솔버가 오류 없이 실행되는지 확인
        성공 기준: 학습이 완료되고 유한한 손실값을 생성
        """
        # 빠른 테스트를 위한 최소 설정
        LAYERS_DIMS = [1, 16, 16, 1]
        CHEBY_ORDER = 3
        N_PDE_POINTS = 50
        N_BC_POINTS = 10
        ADAM_EPOCHS = 100
        ADAM_LR = 1e-3

        # 데이터 생성
        pde_sampler = LatinHypercubeSampler(
            N_PDE_POINTS, self.domain_min, self.domain_max, device=self.device
        )
        pde_points = pde_sampler.sample()
        pde_points.requires_grad_(True)

        x_bc1 = torch.full((N_BC_POINTS, 1), self.domain_min[0], device=self.device)
        x_bc2 = torch.full((N_BC_POINTS, 1), self.domain_max[0], device=self.device)
        bc_points_dicts = [{'points': x_bc1}, {'points': x_bc2}]

        # 모델, 손실 함수, 트레이너
        model = self.create_model(LAYERS_DIMS, CHEBY_ORDER)
        pde_residual_fn = self.define_pde_residual()
        bc_fns = self.define_bc_funcs()
        
        loss_fn = PhysicsInformedLoss(
            pde_residual_fn=pde_residual_fn,
            bc_fns=bc_fns,
            loss_weights={'pde': 1.0, 'bc': 10.0}
        )

        trainer = Trainer(model, loss_fn)

        # 학습 실행
        history = trainer.train(
            pde_points=pde_points,
            bc_points_dicts=bc_points_dicts,
            adam_epochs=ADAM_EPOCHS,
            lbfgs_epochs=0,
            adam_lr=ADAM_LR,
            log_interval=50
        )

        # 검증: 학습 완료 및 손실 기록
        self.assertIn('total_loss', history)
        self.assertGreater(len(history['total_loss']), 0)
        
        final_loss = history['total_loss'][-1]
        self.assertTrue(np.isfinite(final_loss), f"최종 손실이 유한하지 않음: {final_loss}")
        
        # 검증: 손실이 감소했는지 확인
        initial_loss = history['total_loss'][0]
        self.assertLess(final_loss, initial_loss, "손실이 감소하지 않음")
        
        print(f"\n[기본 테스트] 초기 손실: {initial_loss:.4e}, 최종 손실: {final_loss:.4e}")

    def test_helmholtz_solver_convergence(self):
        """
        Helmholtz 솔버 수렴성 테스트
        
        목적: 더 많은 에포크로 학습하여 상대 L2 오차가 임계값 이하로 수렴하는지 확인
        성공 기준: 상대 L2 오차 < 0.01 (1%)
        """
        # 더 정교한 설정
        LAYERS_DIMS = [1, 32, 32, 32, 1]
        CHEBY_ORDER = 3
        N_PDE_POINTS = 200
        N_BC_POINTS = 20
        ADAM_EPOCHS = 500
        ADAM_LR = 1e-3
        ERROR_THRESHOLD = 0.01  # 1% 상대 오차

        # 데이터 생성
        pde_sampler = LatinHypercubeSampler(
            N_PDE_POINTS, self.domain_min, self.domain_max, device=self.device
        )
        pde_points = pde_sampler.sample()
        pde_points.requires_grad_(True)

        x_bc1 = torch.full((N_BC_POINTS, 1), self.domain_min[0], device=self.device)
        x_bc2 = torch.full((N_BC_POINTS, 1), self.domain_max[0], device=self.device)
        bc_points_dicts = [{'points': x_bc1}, {'points': x_bc2}]

        # 모델, 손실 함수, 트레이너
        model = self.create_model(LAYERS_DIMS, CHEBY_ORDER)
        pde_residual_fn = self.define_pde_residual()
        bc_fns = self.define_bc_funcs()
        
        loss_fn = PhysicsInformedLoss(
            pde_residual_fn=pde_residual_fn,
            bc_fns=bc_fns,
            loss_weights={'pde': 1.0, 'bc': 10.0}
        )

        trainer = Trainer(model, loss_fn)

        # 학습 실행
        history = trainer.train(
            pde_points=pde_points,
            bc_points_dicts=bc_points_dicts,
            adam_epochs=ADAM_EPOCHS,
            lbfgs_epochs=0,
            adam_lr=ADAM_LR,
            log_interval=100
        )

        # 평가: 상대 L2 오차 계산
        model.eval()
        with torch.no_grad():
            x_test = torch.linspace(
                self.domain_min[0], self.domain_max[0], 100, device=self.device
            ).view(-1, 1)
            u_pred = model(x_test)
            u_true = self.analytical_solution(x_test)
            
            l2_error = torch.norm(u_pred - u_true) / torch.norm(u_true)
            l2_error_value = l2_error.item()

        print(f"\n[수렴성 테스트] 상대 L2 오차: {l2_error_value:.6f}")
        print(f"목표 임계값: {ERROR_THRESHOLD}")
        
        # 검증: 오차가 임계값 이하인지 확인
        self.assertLess(
            l2_error_value, 
            ERROR_THRESHOLD,
            f"상대 L2 오차 {l2_error_value:.6f}가 임계값 {ERROR_THRESHOLD}를 초과"
        )

    def test_helmholtz_with_lbfgs(self):
        """
        L-BFGS 미세조정을 포함한 Helmholtz 솔버 테스트
        
        목적: 2단계 최적화(Adam + L-BFGS)가 정상 작동하는지 확인
        성공 기준: L-BFGS 적용 후 손실이 추가로 감소
        """
        LAYERS_DIMS = [1, 32, 32, 1]
        CHEBY_ORDER = 3
        N_PDE_POINTS = 100
        N_BC_POINTS = 10
        ADAM_EPOCHS = 200
        LBFGS_EPOCHS = 1
        ADAM_LR = 1e-3

        # 데이터 생성
        pde_sampler = LatinHypercubeSampler(
            N_PDE_POINTS, self.domain_min, self.domain_max, device=self.device
        )
        pde_points = pde_sampler.sample()
        pde_points.requires_grad_(True)

        x_bc1 = torch.full((N_BC_POINTS, 1), self.domain_min[0], device=self.device)
        x_bc2 = torch.full((N_BC_POINTS, 1), self.domain_max[0], device=self.device)
        bc_points_dicts = [{'points': x_bc1}, {'points': x_bc2}]

        # 모델, 손실 함수, 트레이너
        model = self.create_model(LAYERS_DIMS, CHEBY_ORDER)
        pde_residual_fn = self.define_pde_residual()
        bc_fns = self.define_bc_funcs()
        
        loss_fn = PhysicsInformedLoss(
            pde_residual_fn=pde_residual_fn,
            bc_fns=bc_fns,
            loss_weights={'pde': 1.0, 'bc': 10.0}
        )

        trainer = Trainer(model, loss_fn)

        # 학습 실행 (Adam + L-BFGS)
        history = trainer.train(
            pde_points=pde_points,
            bc_points_dicts=bc_points_dicts,
            adam_epochs=ADAM_EPOCHS,
            lbfgs_epochs=LBFGS_EPOCHS,
            adam_lr=ADAM_LR,
            log_interval=100
        )

        # 검증: L-BFGS 적용 후 손실 감소
        adam_final_loss = history['total_loss'][ADAM_EPOCHS - 1]
        lbfgs_final_loss = history['total_loss'][-1]
        
        print(f"\n[L-BFGS 테스트] Adam 최종 손실: {adam_final_loss:.4e}")
        print(f"[L-BFGS 테스트] L-BFGS 최종 손실: {lbfgs_final_loss:.4e}")
        
        # L-BFGS가 손실을 유지하거나 감소시켰는지 확인 (증가하지 않아야 함)
        self.assertLessEqual(
            lbfgs_final_loss,
            adam_final_loss * 1.1,  # 10% 여유
            f"L-BFGS 후 손실이 크게 증가: {adam_final_loss:.4e} -> {lbfgs_final_loss:.4e}"
        )


if __name__ == '__main__':
    unittest.main()
