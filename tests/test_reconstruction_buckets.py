"""
단위 테스트: 버킷 이미지로부터 3D 높이 재구성 (Scaled-cPIKAN PINN 기반)

목적:
    원시 버킷 강도 이미지로부터 3D 높이 맵을 재구성하는 물리 기반 역문제
    해결을 검증하는 단위 테스트입니다. 사전 계산된 위상 맵 대신 원시 데이터를
    직접 사용하는 더 현실적인 접근 방식을 테스트합니다.

문제 설명:
    주어진: 4개 레이저, k=0..2 위상의 원시 버킷 강도 이미지 Iⱼ(x,y,δⱼ)
    찾기: 관측된 강도를 생성하는 높이 맵 h(x, y)
    제약: 표면 평활도

물리:
    정방향 모델: Iⱼ,ₖ(x,y) = A + B·cos(4π·h(x,y)·λⱼ⁻¹ + δₖ)

테스트 케이스:
    1. test_bucket_loss_computation: 버킷 기반 손실 함수 계산 테스트
    2. test_reconstruction_from_buckets_basic: 기본 재구성 기능 테스트
    3. test_reconstruction_with_known_height: 알려진 높이로 검증 테스트

실행 방법:
    python -m unittest tests.test_reconstruction_buckets
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
from src.models import Scaled_cPIKAN
from src.data_generator import DEFAULT_WAVELENGTHS


class ReconstructionLossFromBuckets(torch.nn.Module):
    """버킷 이미지로부터 3D 재구성을 위한 물리 기반 손실 함수"""
    
    def __init__(self, wavelengths, num_buckets, smoothness_weight=1e-4):
        super().__init__()
        self.register_buffer(
            "wavelengths",
            torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1)
        )
        
        # 위상 시프트 동적 생성
        deltas_np = np.linspace(
            0.0, 2.0 * np.pi, num_buckets, endpoint=False, dtype=np.float32
        )
        deltas = torch.from_numpy(deltas_np)
        self.register_buffer("deltas", deltas.view(1, num_buckets, 1))
        
        self.smoothness_weight = smoothness_weight
        self.mse_loss = torch.nn.MSELoss()
        self.metrics = {}

    def forward(self, model_outputs, coords, targets):
        """
        손실 계산
        
        Args:
            model_outputs: 예측 높이 (형태: [1, 1, H*W])
            coords: 좌표 텐서, requires_grad=True
            targets: 목표 버킷 이미지 (형태: [N_lasers, N_buckets, H*W])
        """
        predicted_height = model_outputs

        # 1. 데이터 충실도 손실
        # 예측 위상: [1, 1, H*W] -> [N_lasers, 1, H*W]
        predicted_phase = (4 * np.pi / self.wavelengths) * predicted_height

        # 위상 시프트 추가: [N_lasers, 1, H*W] -> [N_lasers, N_buckets, H*W]
        phase_with_shifts = predicted_phase + self.deltas

        # 버킷 이미지 시뮬레이션 (데이터 생성기와 동일한 파라미터)
        A = 128
        B = 100
        predicted_buckets = A + B * torch.cos(phase_with_shifts)
        predicted_buckets = predicted_buckets.view(
            predicted_buckets.shape[0], predicted_buckets.shape[1], -1
        )

        loss_data = self.mse_loss(predicted_buckets, targets)

        # 2. 평활도 정규화 손실
        h = predicted_height.squeeze(0).squeeze(0)  # [1,1,H*W] -> [H*W]
        grad_h = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]
        h_x, h_y = grad_h[:, 0], grad_h[:, 1]
        h_xx = torch.autograd.grad(h_x.sum(), coords, create_graph=True)[0][:, 0]
        h_yy = torch.autograd.grad(h_y.sum(), coords, create_graph=True)[0][:, 1]
        laplacian = h_xx + h_yy
        loss_smoothness = self.mse_loss(laplacian, torch.zeros_like(laplacian))

        # 총 손실
        total_loss = loss_data + self.smoothness_weight * loss_smoothness

        self.metrics = {
            'loss_total': total_loss.item(),
            'loss_data': loss_data.item(),
            'loss_smoothness': loss_smoothness.item()
        }
        
        return total_loss


def simulate_bucket_images(height_map, wavelengths, num_buckets=3):
    """
    높이 맵으로부터 버킷 이미지 시뮬레이션
    
    Args:
        height_map: numpy 배열, 형태 (H, W)
        wavelengths: 파장 리스트
        num_buckets: 레이저당 버킷 수
        
    Returns:
        numpy 배열, 형태 (num_lasers, num_buckets, H, W)
    """
    height, width = height_map.shape
    num_lasers = len(wavelengths)
    
    height_map_t = torch.from_numpy(height_map).float().view(1, 1, height, width)
    wavelengths_t = torch.tensor(wavelengths, dtype=torch.float32).view(num_lasers, 1, 1, 1)
    
    deltas = torch.arange(num_buckets, dtype=torch.float32) * (2 * np.pi / num_buckets)
    deltas = deltas.view(1, num_buckets, 1, 1)
    
    phase = (4 * np.pi / wavelengths_t) * height_map_t
    phase_with_shifts = phase + deltas
    
    A, B = 128, 100
    predicted_buckets = A + B * torch.cos(phase_with_shifts)
    
    return predicted_buckets.view(num_lasers, num_buckets, height, width).numpy()


class TestReconstructionFromBuckets(unittest.TestCase):
    """버킷 이미지로부터 3D 재구성 테스트"""

    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.device = torch.device("cpu")
        self.wavelengths = DEFAULT_WAVELENGTHS
        self.num_buckets = 3

    def test_bucket_loss_computation(self):
        """
        버킷 기반 손실 함수 계산 테스트
        
        목적: 손실 함수가 버킷 이미지로부터 올바르게 계산되는지 확인
        성공 기준: 손실이 유한하고 각 구성 요소가 존재
        """
        grid_shape = (16, 16)
        
        # 간단한 높이 맵 생성 (단일 가우시안)
        x = np.linspace(0, 1, grid_shape[0])
        y = np.linspace(0, 1, grid_shape[1])
        xx, yy = np.meshgrid(x, y)
        height_map = 20e-6 * np.exp(-((xx-0.5)**2 + (yy-0.5)**2) / 0.1)
        
        # 버킷 이미지 생성
        bucket_images = simulate_bucket_images(
            height_map, self.wavelengths, self.num_buckets
        )
        bucket_images_t = torch.from_numpy(bucket_images).float().to(self.device)
        
        # 모델 생성
        domain_min = torch.tensor([0.0, 0.0], device=self.device)
        domain_max = torch.tensor([1.0, 1.0], device=self.device)
        
        model = Scaled_cPIKAN(
            layers_dims=[2, 32, 32, 1],
            cheby_order=3,
            domain_min=domain_min,
            domain_max=domain_max
        ).to(self.device)
        
        # 좌표 생성
        x_t = torch.linspace(0, 1, grid_shape[0], device=self.device)
        y_t = torch.linspace(0, 1, grid_shape[1], device=self.device)
        grid_x, grid_y = torch.meshgrid(x_t, y_t, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        # 타겟 준비
        targets = bucket_images_t.view(len(self.wavelengths), self.num_buckets, -1)
        
        # 손실 함수
        loss_fn = ReconstructionLossFromBuckets(
            wavelengths=self.wavelengths,
            num_buckets=self.num_buckets,
            smoothness_weight=1e-7
        ).to(self.device)
        
        # 순전파
        predicted_height = model(coords).view(1, 1, -1)
        loss = loss_fn(predicted_height, coords, targets)
        
        # 검증
        self.assertTrue(torch.isfinite(loss), "손실이 유한하지 않음")
        self.assertIn('loss_total', loss_fn.metrics)
        self.assertIn('loss_data', loss_fn.metrics)
        self.assertIn('loss_smoothness', loss_fn.metrics)
        self.assertGreater(loss.item(), 0, "손실이 0 이하")
        
        print(f"\n[버킷 손실 계산 테스트] 총 손실: {loss_fn.metrics['loss_total']:.4e}")
        print(f"  데이터 손실: {loss_fn.metrics['loss_data']:.4e}")
        print(f"  평활도 손실: {loss_fn.metrics['loss_smoothness']:.4e}")

    def test_reconstruction_from_buckets_basic(self):
        """
        버킷 이미지로부터 기본 재구성 기능 테스트
        
        목적: 재구성 파이프라인이 오류 없이 실행되는지 확인
        성공 기준: 학습이 완료되고 손실이 감소
        """
        grid_shape = (16, 16)
        adam_epochs = 100
        
        # 간단한 높이 맵 생성
        x = np.linspace(0, 1, grid_shape[0])
        y = np.linspace(0, 1, grid_shape[1])
        xx, yy = np.meshgrid(x, y)
        height_map = 20e-6 * np.exp(-((xx-0.5)**2 + (yy-0.5)**2) / 0.1)
        
        # 버킷 이미지 생성
        bucket_images = simulate_bucket_images(
            height_map, self.wavelengths, self.num_buckets
        )
        bucket_images_t = torch.from_numpy(bucket_images).float().to(self.device)
        
        # 모델 생성
        domain_min = torch.tensor([0.0, 0.0], device=self.device)
        domain_max = torch.tensor([1.0, 1.0], device=self.device)
        
        model = Scaled_cPIKAN(
            layers_dims=[2, 32, 32, 1],
            cheby_order=3,
            domain_min=domain_min,
            domain_max=domain_max
        ).to(self.device)
        
        # 좌표 생성
        x_t = torch.linspace(0, 1, grid_shape[0], device=self.device)
        y_t = torch.linspace(0, 1, grid_shape[1], device=self.device)
        grid_x, grid_y = torch.meshgrid(x_t, y_t, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        targets = bucket_images_t.view(len(self.wavelengths), self.num_buckets, -1)
        
        # 손실 함수 및 옵티마이저
        loss_fn = ReconstructionLossFromBuckets(
            wavelengths=self.wavelengths,
            num_buckets=self.num_buckets,
            smoothness_weight=1e-7
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 학습
        loss_history = []
        for epoch in range(adam_epochs):
            model.train()
            optimizer.zero_grad()
            
            predicted_height = model(coords).view(1, 1, -1)
            loss = loss_fn(predicted_height, coords, targets)
            
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss_fn.metrics['loss_total'])
        
        # 검증
        self.assertEqual(len(loss_history), adam_epochs)
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        
        self.assertLess(final_loss, initial_loss, "손실이 감소하지 않음")
        self.assertTrue(np.isfinite(final_loss), "최종 손실이 유한하지 않음")
        
        print(f"\n[버킷 기본 재구성 테스트] 초기 손실: {initial_loss:.4e}, 최종 손실: {final_loss:.4e}")

    def test_reconstruction_with_known_height(self):
        """
        알려진 높이로 재구성 검증 테스트
        
        목적: 작은 그리드에서 알려진 높이를 재구성하고 RMSE 계산
        성공 기준: RMSE < 10e-6 (상대적으로 작은 오차)
        """
        grid_shape = (12, 12)  # 매우 작은 그리드
        adam_epochs = 300
        
        # 간단한 알려진 높이 맵 (평면 + 작은 경사)
        x = np.linspace(0, 1, grid_shape[0])
        y = np.linspace(0, 1, grid_shape[1])
        xx, yy = np.meshgrid(x, y)
        ground_truth_height = 10e-6 * (xx + yy)  # 선형 경사
        
        # 버킷 이미지 생성
        bucket_images = simulate_bucket_images(
            ground_truth_height, self.wavelengths, self.num_buckets
        )
        bucket_images_t = torch.from_numpy(bucket_images).float().to(self.device)
        ground_truth_height_t = torch.from_numpy(ground_truth_height).float().to(self.device)
        
        # 모델 생성
        domain_min = torch.tensor([0.0, 0.0], device=self.device)
        domain_max = torch.tensor([1.0, 1.0], device=self.device)
        
        model = Scaled_cPIKAN(
            layers_dims=[2, 64, 64, 1],
            cheby_order=4,
            domain_min=domain_min,
            domain_max=domain_max
        ).to(self.device)
        
        # 좌표 생성
        x_t = torch.linspace(0, 1, grid_shape[0], device=self.device)
        y_t = torch.linspace(0, 1, grid_shape[1], device=self.device)
        grid_x, grid_y = torch.meshgrid(x_t, y_t, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        targets = bucket_images_t.view(len(self.wavelengths), self.num_buckets, -1)
        
        # 손실 함수 및 옵티마이저
        loss_fn = ReconstructionLossFromBuckets(
            wavelengths=self.wavelengths,
            num_buckets=self.num_buckets,
            smoothness_weight=1e-8  # 평면이므로 평활도 가중치 낮춤
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 학습
        for epoch in range(adam_epochs):
            model.train()
            optimizer.zero_grad()
            
            predicted_height = model(coords).view(1, 1, -1)
            loss = loss_fn(predicted_height, coords, targets)
            
            loss.backward()
            optimizer.step()
        
        # 평가
        model.eval()
        with torch.no_grad():
            predicted_height_t = model(coords).view(grid_shape)
        
        rmse = torch.sqrt(torch.mean((predicted_height_t - ground_truth_height_t)**2))
        rmse_value = rmse.item()
        
        print(f"\n[알려진 높이 검증 테스트] RMSE: {rmse_value:.4e}")
        
        # 검증: RMSE가 임계값 이하 (선형 함수이므로 잘 맞아야 함)
        self.assertLess(rmse_value, 10e-6, f"RMSE {rmse_value:.4e}가 임계값 10e-6을 초과")


if __name__ == '__main__':
    unittest.main()
