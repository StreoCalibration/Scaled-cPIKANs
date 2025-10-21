import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ChebyKANLayer, Scaled_cPIKAN

class TestModels(unittest.TestCase):

    def test_chebykan_layer_forward(self):
        """Tests the forward pass of the ChebyKANLayer."""
        in_features = 4
        out_features = 8
        cheby_order = 3
        batch_size = 16

        layer = ChebyKANLayer(in_features, out_features, cheby_order)
        # Input must be in [-1, 1]
        input_tensor = torch.rand(batch_size, in_features) * 2 - 1

        output = layer(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

    def test_scaled_cpikan_forward(self):
        """Tests the forward pass of the Scaled_cPIKAN model."""
        layers_dims = [2, 16, 16, 1]
        cheby_order = 3
        domain_min = torch.tensor([-5.0, -5.0])
        domain_max = torch.tensor([5.0, 5.0])
        batch_size = 32

        model = Scaled_cPIKAN(layers_dims, cheby_order, domain_min, domain_max)

        # Input is in the physical domain
        input_tensor = torch.rand(batch_size, 2) * 10 - 5

        output = model(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, layers_dims[-1]))

    def test_affine_scaling_in_model(self):
        """Checks that the model correctly scales the input."""
        layers_dims = [1, 8, 1]
        cheby_order = 2
        domain_min = torch.tensor([100.0])
        domain_max = torch.tensor([200.0])

        model = Scaled_cPIKAN(layers_dims, cheby_order, domain_min, domain_max)

        # Test point at the middle of the physical domain
        physical_x = torch.tensor([[150.0]])
        # This should be scaled to 0.0
        scaled_x = model._affine_scale(physical_x)
        self.assertTrue(torch.allclose(scaled_x, torch.tensor([[0.0]])))

        # Test point at the min of the physical domain
        physical_x_min = torch.tensor([[100.0]])
        # This should be scaled to -1.0
        scaled_x_min = model._affine_scale(physical_x_min)
        self.assertTrue(torch.allclose(scaled_x_min, torch.tensor([[-1.0]])))


class TestChebyshevPolynomials(unittest.TestCase):
    """P1 Task 4: 체비쇼프 다항식 계산 정확성 검증"""

    def test_chebyshev_T0(self):
        """T_0(x) = 1을 검증"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=0)
        
        # 다양한 입력값에서 T_0 = 1을 확인
        x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        
        # 수동으로 T_0 = 1 계산
        with torch.no_grad():
            # cheby_coeffs를 [[[1.0]]]로 설정 (out_features=1, in_features=1, cheby_order+1=1)
            layer.cheby_coeffs.data = torch.ones(1, 1, 1)
            output = layer(x)
        
        # 모든 출력이 1이어야 함 (T_0 = 1이므로)
        expected = torch.ones(5, 1)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_chebyshev_T1(self):
        """T_1(x) = x를 검증"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=1)
        
        # 테스트 입력
        x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        
        with torch.no_grad():
            # T_1 계수만 1로 설정: coeffs[0, 0, 1] = 1, coeffs[0, 0, 0] = 0
            layer.cheby_coeffs.data = torch.zeros(1, 1, 2)
            layer.cheby_coeffs.data[0, 0, 1] = 1.0
            output = layer(x)
        
        # T_1(x) = x이므로 출력은 입력과 동일해야 함
        self.assertTrue(torch.allclose(output, x, atol=1e-6))

    def test_chebyshev_T2(self):
        """T_2(x) = 2x² - 1을 검증"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=2)
        
        # 테스트 입력
        x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        
        with torch.no_grad():
            # T_2 계수만 1로 설정
            layer.cheby_coeffs.data = torch.zeros(1, 1, 3)
            layer.cheby_coeffs.data[0, 0, 2] = 1.0
            output = layer(x)
        
        # T_2(x) = 2x² - 1
        expected = 2 * x ** 2 - 1
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_chebyshev_T3(self):
        """T_3(x) = 4x³ - 3x를 검증"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=3)
        
        # 테스트 입력
        x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        
        with torch.no_grad():
            # T_3 계수만 1로 설정
            layer.cheby_coeffs.data = torch.zeros(1, 1, 4)
            layer.cheby_coeffs.data[0, 0, 3] = 1.0
            output = layer(x)
        
        # T_3(x) = 4x³ - 3x
        expected = 4 * x ** 3 - 3 * x
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_chebyshev_recurrence_relation(self):
        """체비쇼프 다항식의 점화식을 검증: T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)"""
        layer = ChebyKANLayer(in_features=1, out_features=1, cheby_order=4)
        
        x = torch.linspace(-1, 1, 11).unsqueeze(1)
        
        with torch.no_grad():
            # T_0(x), T_1(x), T_2(x)를 개별적으로 계산하고 점화식 확인
            # T_0(x) = 1
            layer.cheby_coeffs.data = torch.zeros(1, 1, 5)
            layer.cheby_coeffs.data[0, 0, 0] = 1.0
            T0 = layer(x)
            
            # T_1(x) = x
            layer.cheby_coeffs.data = torch.zeros(1, 1, 5)
            layer.cheby_coeffs.data[0, 0, 1] = 1.0
            T1 = layer(x)
            
            # T_2(x) 직접 계산
            layer.cheby_coeffs.data = torch.zeros(1, 1, 5)
            layer.cheby_coeffs.data[0, 0, 2] = 1.0
            T2_direct = layer(x)
            
            # T_2(x) 점화식으로 계산: T_2(x) = 2x*T_1(x) - T_0(x)
            T2_from_recurrence = 2 * x * T1 - T0
            
            # 두 방법의 결과가 일치하는지 확인
            self.assertTrue(torch.allclose(T2_from_recurrence, T2_direct, atol=1e-5))


class TestAffineScaling(unittest.TestCase):
    """P1 Task 5: 아핀 스케일링 정확성 검증"""

    def test_scaling_boundaries(self):
        """도메인 경계값이 정확히 [-1, 1]로 매핑되는지 확인"""
        domain_min = torch.tensor([0.0, 100.0])
        domain_max = torch.tensor([10.0, 200.0])
        
        model = Scaled_cPIKAN(
            layers_dims=[2, 8, 1],
            cheby_order=2,
            domain_min=domain_min,
            domain_max=domain_max
        )
        
        # 최소값 경계 테스트
        x_min = torch.tensor([[0.0, 100.0]])
        scaled_min = model._affine_scale(x_min)
        expected_min = torch.tensor([[-1.0, -1.0]])
        self.assertTrue(torch.allclose(scaled_min, expected_min, atol=1e-6))
        
        # 최대값 경계 테스트
        x_max = torch.tensor([[10.0, 200.0]])
        scaled_max = model._affine_scale(x_max)
        expected_max = torch.tensor([[1.0, 1.0]])
        self.assertTrue(torch.allclose(scaled_max, expected_max, atol=1e-6))

    def test_scaling_midpoint(self):
        """도메인 중간값이 0으로 매핑되는지 확인"""
        domain_min = torch.tensor([0.0, -5.0, 100.0])
        domain_max = torch.tensor([10.0, 5.0, 300.0])
        
        model = Scaled_cPIKAN(
            layers_dims=[3, 8, 1],
            cheby_order=2,
            domain_min=domain_min,
            domain_max=domain_max
        )
        
        # 중간값 계산
        x_mid = torch.tensor([[5.0, 0.0, 200.0]])
        scaled_mid = model._affine_scale(x_mid)
        expected_mid = torch.tensor([[0.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(scaled_mid, expected_mid, atol=1e-6))

    def test_inverse_transformation(self):
        """스케일링과 역스케일링이 원래 값을 복원하는지 확인"""
        domain_min = torch.tensor([0.0, 0.0])
        domain_max = torch.tensor([1.0, 1.0])
        
        model = Scaled_cPIKAN(
            layers_dims=[2, 8, 1],
            cheby_order=2,
            domain_min=domain_min,
            domain_max=domain_max
        )
        
        # 임의의 물리적 도메인 포인트
        x_physical = torch.tensor([[0.3, 0.7], [0.1, 0.9], [0.5, 0.5]])
        
        # 스케일링 후 역스케일링
        x_scaled = model._affine_scale(x_physical)
        # 역변환: x = (x_scaled + 1) / 2 * (max - min) + min
        x_unscaled = (x_scaled + 1.0) / 2.0 * (domain_max - domain_min) + domain_min
        
        self.assertTrue(torch.allclose(x_physical, x_unscaled, atol=1e-6))

    def test_scaling_linearity(self):
        """아핀 스케일링의 선형성 확인"""
        domain_min = torch.tensor([0.0])
        domain_max = torch.tensor([10.0])
        
        model = Scaled_cPIKAN(
            layers_dims=[1, 8, 1],
            cheby_order=2,
            domain_min=domain_min,
            domain_max=domain_max
        )
        
        # 등간격 포인트들
        x = torch.tensor([[0.0], [2.5], [5.0], [7.5], [10.0]])
        scaled = model._affine_scale(x)
        
        # 스케일링된 값들도 등간격이어야 함
        diffs = scaled[1:] - scaled[:-1]
        expected_diff = torch.tensor([[0.5]])
        
        for diff in diffs:
            self.assertTrue(torch.allclose(diff, expected_diff, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
