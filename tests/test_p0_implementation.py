"""
P0 작업에 대한 단위 테스트

이 모듈은 TODO.md P0 작업의 구현을 검증합니다:
1. 학습률 스케줄러 추가
2. 입력 범위 검증
3. 하이퍼파라미터 설정
"""

import unittest
import torch
import torch.nn as nn
import warnings
import sys
import os

# src 디렉토리를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ChebyKANLayer, Scaled_cPIKAN
from src.train import Trainer
from src.loss import PhysicsInformedLoss


class TestLearningRateScheduler(unittest.TestCase):
    """학습률 스케줄러 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.device = torch.device('cpu')
        domain_min = torch.tensor([0.0, 0.0])
        domain_max = torch.tensor([1.0, 1.0])
        
        self.model = Scaled_cPIKAN(
            layers_dims=[2, 8, 8, 1],
            cheby_order=3,
            domain_min=domain_min,
            domain_max=domain_max
        ).to(self.device)

    def test_scheduler_enabled(self):
        """스케줄러가 활성화되면 학습률이 감소하는지 확인"""
        # 간단한 PDE 손실 함수
        def simple_pde(model, points):
            return model(points)
        
        def simple_bc(model, points):
            return model(points)
        
        loss_fn = PhysicsInformedLoss(
            pde_residual_fn=simple_pde,
            bc_fns=[simple_bc],
            loss_weights={'pde': 1.0, 'bc': 1.0}
        )
        
        trainer = Trainer(self.model, loss_fn)
        
        # 테스트 데이터
        pde_points = torch.rand(100, 2).to(self.device)
        bc_points = torch.rand(20, 2).to(self.device)
        
        # 스케줄러 사용하여 짧은 훈련
        history = trainer.train(
            pde_points=pde_points,
            bc_points_dicts=[{'points': bc_points}],
            adam_epochs=50,
            lbfgs_epochs=0,
            adam_lr=1e-3,
            use_scheduler=True,
            scheduler_gamma=0.99,  # 더 빠른 감소로 테스트
            log_interval=100
        )
        
        # 학습률이 기록되었는지 확인
        self.assertIn('learning_rate', history)
        self.assertEqual(len(history['learning_rate']), 50)
        
        # 학습률이 감소했는지 확인
        initial_lr = history['learning_rate'][0]
        final_lr = history['learning_rate'][-1]
        self.assertLess(final_lr, initial_lr)
        
        # 예상 학습률 계산 (gamma^epochs * initial_lr)
        expected_final_lr = initial_lr * (0.99 ** 49)
        self.assertAlmostEqual(final_lr, expected_final_lr, places=8)
        
    def test_scheduler_disabled(self):
        """스케줄러가 비활성화되면 학습률이 일정한지 확인"""
        def simple_pde(model, points):
            return model(points)
        
        def simple_bc(model, points):
            return model(points)
        
        loss_fn = PhysicsInformedLoss(
            pde_residual_fn=simple_pde,
            bc_fns=[simple_bc],
            loss_weights={'pde': 1.0, 'bc': 1.0}
        )
        
        trainer = Trainer(self.model, loss_fn)
        
        pde_points = torch.rand(100, 2).to(self.device)
        bc_points = torch.rand(20, 2).to(self.device)
        
        # 스케줄러 사용하지 않고 훈련
        history = trainer.train(
            pde_points=pde_points,
            bc_points_dicts=[{'points': bc_points}],
            adam_epochs=50,
            lbfgs_epochs=0,
            adam_lr=1e-3,
            use_scheduler=False,
            log_interval=100
        )
        
        # 학습률이 기록되지 않았는지 확인 (스케줄러 비활성화 시)
        self.assertNotIn('learning_rate', history)


class TestInputRangeValidation(unittest.TestCase):
    """입력 범위 검증 테스트"""

    def test_valid_input_range(self):
        """[-1, 1] 범위 내의 입력은 경고 없이 처리됨"""
        layer = ChebyKANLayer(in_features=2, out_features=3, cheby_order=3)
        
        # 유효한 입력
        x = torch.linspace(-1, 1, 10).unsqueeze(1).repeat(1, 2)
        
        # 경고가 발생하지 않아야 함
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = layer(x)
            
            # 경고가 없어야 함
            self.assertEqual(len(w), 0)
            
        # 출력 shape 확인
        self.assertEqual(output.shape, (10, 3))

    def test_invalid_input_range(self):
        """[-1, 1] 범위를 벗어난 입력은 경고 발생"""
        layer = ChebyKANLayer(in_features=2, out_features=3, cheby_order=3)
        
        # 범위를 벗어난 입력
        x = torch.tensor([[2.0, 3.0], [-2.0, -3.0]])  # 범위 초과
        
        # 경고가 발생해야 함
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = layer(x)
            
            # 경고가 있어야 함
            self.assertGreater(len(w), 0)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("범위를 벗어났습니다", str(w[0].message))

    def test_boundary_values(self):
        """경계값 (-1, 1)은 경고 없이 처리됨"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=3)
        
        # 정확히 경계값
        x = torch.tensor([[-1.0], [1.0], [0.0]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = layer(x)
            
            # 경고가 없어야 함
            self.assertEqual(len(w), 0)

    def test_no_warning_in_eval_mode(self):
        """평가 모드에서는 경고가 발생하지 않음"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=3)
        layer.eval()
        
        # 범위를 벗어난 입력이지만 eval 모드
        x = torch.tensor([[2.0], [3.0]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with torch.no_grad():  # grad 비활성화
                output = layer(x)
            
            # 경고가 없어야 함 (eval 모드)
            self.assertEqual(len(w), 0)


class TestAffineScaling(unittest.TestCase):
    """아핀 스케일링 정확성 테스트"""

    def test_scaling_boundaries(self):
        """도메인 경계값이 정확히 [-1, 1]로 매핑되는지 확인"""
        domain_min = torch.tensor([0.0, 0.0])
        domain_max = torch.tensor([10.0, 5.0])
        
        model = Scaled_cPIKAN(
            layers_dims=[2, 8, 1],
            cheby_order=3,
            domain_min=domain_min,
            domain_max=domain_max
        )
        
        # 최소값 테스트
        x_min = torch.tensor([[0.0, 0.0]])
        scaled_min = model._affine_scale(x_min)
        expected_min = torch.tensor([[-1.0, -1.0]])
        self.assertTrue(torch.allclose(scaled_min, expected_min, atol=1e-6))
        
        # 최대값 테스트
        x_max = torch.tensor([[10.0, 5.0]])
        scaled_max = model._affine_scale(x_max)
        expected_max = torch.tensor([[1.0, 1.0]])
        self.assertTrue(torch.allclose(scaled_max, expected_max, atol=1e-6))
        
        # 중간값 테스트
        x_mid = torch.tensor([[5.0, 2.5]])
        scaled_mid = model._affine_scale(x_mid)
        expected_mid = torch.tensor([[0.0, 0.0]])
        self.assertTrue(torch.allclose(scaled_mid, expected_mid, atol=1e-6))

    def test_scaling_invertibility(self):
        """스케일링과 역스케일링이 원래 값을 복원하는지 확인"""
        from src.data import affine_scale, affine_unscale
        
        domain_min = torch.tensor([0.0, 0.0])
        domain_max = torch.tensor([10.0, 5.0])
        
        # 임의의 값
        x_original = torch.tensor([[3.5, 2.1], [7.8, 4.2]])
        
        # 스케일링 후 역스케일링
        x_scaled = affine_scale(x_original, domain_min, domain_max)
        x_recovered = affine_unscale(x_scaled, domain_min, domain_max)
        
        # 원래 값과 일치해야 함
        self.assertTrue(torch.allclose(x_original, x_recovered, atol=1e-6))


class TestChebyshevPolynomials(unittest.TestCase):
    """체비쇼프 다항식 계산 정확성 테스트"""

    def test_chebyshev_basis_T0(self):
        """T_0(x) = 1 검증"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=3)
        
        x = torch.linspace(-1, 1, 50).unsqueeze(1)
        
        # forward를 통해 내부적으로 계산된 기저를 테스트
        # T_0는 항상 1이어야 함
        with torch.no_grad():
            # 계수를 특별히 설정: 첫 번째 계수만 1, 나머지 0
            layer.cheby_coeffs.data.fill_(0)
            layer.cheby_coeffs.data[0, 0, 0] = 1.0  # T_0만 활성화
            
            output = layer(x)
            expected = torch.ones(50, 1)
            
            self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_chebyshev_basis_T1(self):
        """T_1(x) = x 검증"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=3)
        
        x = torch.linspace(-1, 1, 50).unsqueeze(1)
        
        with torch.no_grad():
            # T_1만 활성화
            layer.cheby_coeffs.data.fill_(0)
            layer.cheby_coeffs.data[0, 0, 1] = 1.0
            
            output = layer(x)
            expected = x
            
            self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_chebyshev_basis_T2(self):
        """T_2(x) = 2x² - 1 검증"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=3)
        
        x = torch.linspace(-1, 1, 50).unsqueeze(1)
        
        with torch.no_grad():
            # T_2만 활성화
            layer.cheby_coeffs.data.fill_(0)
            layer.cheby_coeffs.data[0, 0, 2] = 1.0
            
            output = layer(x)
            expected = 2 * x**2 - 1
            
            self.assertTrue(torch.allclose(output, expected, atol=1e-4))

    def test_chebyshev_basis_T3(self):
        """T_3(x) = 4x³ - 3x 검증"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=3)
        
        x = torch.linspace(-1, 1, 50).unsqueeze(1)
        
        with torch.no_grad():
            # T_3만 활성화
            layer.cheby_coeffs.data.fill_(0)
            layer.cheby_coeffs.data[0, 0, 3] = 1.0
            
            output = layer(x)
            expected = 4 * x**3 - 3 * x
            
            self.assertTrue(torch.allclose(output, expected, atol=1e-4))


class TestHyperparameters(unittest.TestCase):
    """하이퍼파라미터 설정 테스트"""

    def test_paper_recommended_architecture(self):
        """논문 권장 아키텍처 [2, 32, 32, 32, 1]이 올바르게 생성되는지 확인"""
        domain_min = torch.tensor([0.0, 0.0])
        domain_max = torch.tensor([1.0, 1.0])
        
        model = Scaled_cPIKAN(
            layers_dims=[2, 32, 32, 32, 1],
            cheby_order=3,
            domain_min=domain_min,
            domain_max=domain_max
        )
        
        # 네트워크 구조 확인
        self.assertEqual(model.layers_dims, [2, 32, 32, 32, 1])
        self.assertEqual(model.cheby_order, 3)
        
        # 레이어 개수 확인 (ChebyKAN + LayerNorm + Tanh)
        # 4개의 ChebyKAN 레이어 + 3개의 (LayerNorm + Tanh) = 4 + 6 = 10
        self.assertEqual(len(model.network), 10)

    def test_forward_pass_with_paper_architecture(self):
        """논문 권장 구조로 순방향 패스가 정상 작동하는지 확인"""
        domain_min = torch.tensor([0.0, 0.0])
        domain_max = torch.tensor([1.0, 1.0])
        
        model = Scaled_cPIKAN(
            layers_dims=[2, 32, 32, 32, 1],
            cheby_order=3,
            domain_min=domain_min,
            domain_max=domain_max
        )
        
        # 배치 입력
        x = torch.rand(10, 2)  # 10 samples, 2D input
        
        # 순방향 패스
        output = model(x)
        
        # 출력 shape 확인
        self.assertEqual(output.shape, (10, 1))
        
        # 출력이 유한한 값인지 확인
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
