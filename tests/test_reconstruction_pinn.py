"""
단위 테스트: 3D 높이 재구성 (Scaled-cPIKAN PINN 기반)

목적:
    Scaled-cPIKAN 모델을 사용한 물리 기반 역문제 해결을 검증하는 단위 테스트입니다.
    다중 파장 위상 측정으로부터 3D 높이 맵 h(x, y)를 재구성하는 기능을 테스트합니다.

문제 설명:
    주어진: 4개 서로 다른 파장의 위상 맵 φ₁, φ₂, φ₃, φ₄
    찾기: 관측된 위상을 생성하는 높이 맵 h(x, y)
    제약: 표면 평활도 (Laplacian 페널티)

테스트 케이스:
    1. test_reconstruction_loss_computation: 손실 함수 계산 테스트
    2. test_reconstruction_basic: 기본 재구성 기능 테스트
    3. test_reconstruction_small_grid: 작은 그리드 빠른 수렴 테스트

실행 방법:
    python -m unittest tests.test_reconstruction_pinn
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
from src.models import Scaled_cPIKAN
from src.data_generator import DEFAULT_WAVELENGTHS, generate_synthetic_data


class ReconstructionLoss(torch.nn.Module):
    """3D 재구성을 위한 물리 기반 손실 함수"""
    
    def __init__(self, wavelengths, smoothness_weight=1e-4):
        super().__init__()
        self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1)
        self.smoothness_weight = smoothness_weight
        self.mse_loss = torch.nn.MSELoss()
        self.metrics = {}

    def forward(self, model_outputs, coords, targets):
        """
        손실 계산
        
        Args:
            model_outputs: 예측 높이 (형태: [1, H*W])
            coords: 좌표 텐서, requires_grad=True
            targets: 목표 위상 맵 (형태: [num_lasers, 1, H*W])
        """
        predicted_height = model_outputs
        self.wavelengths = self.wavelengths.to(predicted_height.device)

        # 1. 데이터 충실도 손실
        true_phase = (4 * np.pi / self.wavelengths) * predicted_height

        target_cos = torch.cos(targets)
        target_sin = torch.sin(targets)
        pred_cos = torch.cos(true_phase)
        pred_sin = torch.sin(true_phase)

        loss_data = self.mse_loss(pred_cos, target_cos) + self.mse_loss(pred_sin, target_sin)

        # 2. 평활도 정규화 손실
        h = predicted_height.squeeze(0)
        grad_h = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]

        h_x = grad_h[:, 0]
        h_y = grad_h[:, 1]

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


class TestReconstructionPINN(unittest.TestCase):
    """3D 재구성 PINN 테스트"""

    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.device = torch.device("cpu")
        self.wavelengths = DEFAULT_WAVELENGTHS

    def test_reconstruction_loss_computation(self):
        """
        재구성 손실 함수 계산 테스트
        
        목적: 손실 함수가 올바르게 계산되는지 확인
        성공 기준: 손실이 유한한 값이고 각 구성 요소가 존재
        """
        grid_shape = (32, 32)
        
        # 합성 데이터 생성 (버킷 이미지)
        ground_truth_height, bucket_images = generate_synthetic_data(
            shape=grid_shape,
            wavelengths=self.wavelengths,
            num_buckets=3,
            save_path=None
        )
        
        # bucket_images는 (num_lasers * num_buckets, H, W) 형태
        # wrapped_phases를 계산하기 위해 각 레이저의 위상 추출
        num_lasers = len(self.wavelengths)
        num_buckets = 3
        
        # wrapped_phases 계산: 각 레이저당 하나의 위상 맵
        wrapped_phases = []
        for i in range(num_lasers):
            # 각 레이저의 버킷 이미지에서 위상 추출 (간단히 첫 번째 버킷 사용)
            bucket_idx = i * num_buckets
            phase = (4 * np.pi * ground_truth_height) / self.wavelengths[i]
            wrapped_phase = np.mod(phase, 2 * np.pi)
            wrapped_phases.append(wrapped_phase)
        
        wrapped_phases = np.array(wrapped_phases)
        
        ground_truth_height_t = torch.from_numpy(ground_truth_height).float().to(self.device)
        wrapped_phases_t = torch.from_numpy(wrapped_phases).float().to(self.device)
        
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
        x = torch.linspace(0, 1, grid_shape[0], device=self.device)
        y = torch.linspace(0, 1, grid_shape[1], device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        # 타겟 준비
        # wrapped_phases_t 형태: (4, H, W), 필요한 형태: (4, 1, H*W)
        H, W = grid_shape
        targets = wrapped_phases_t.view(4, H*W).unsqueeze(1)  # (4, 1, H*W)
        
        # 손실 함수
        loss_fn = ReconstructionLoss(
            wavelengths=self.wavelengths,
            smoothness_weight=1e-5
        )
        
        # 순전파
        predicted_height = model(coords).view(1, -1)
        loss = loss_fn(predicted_height, coords, targets)
        
        # 검증
        self.assertTrue(torch.isfinite(loss), "손실이 유한하지 않음")
        self.assertIn('loss_total', loss_fn.metrics)
        self.assertIn('loss_data', loss_fn.metrics)
        self.assertIn('loss_smoothness', loss_fn.metrics)
        self.assertGreater(loss.item(), 0, "손실이 0 이하")
        
        print(f"\n[손실 계산 테스트] 총 손실: {loss_fn.metrics['loss_total']:.4e}")
        print(f"  데이터 손실: {loss_fn.metrics['loss_data']:.4e}")
        print(f"  평활도 손실: {loss_fn.metrics['loss_smoothness']:.4e}")

    def test_reconstruction_basic(self):
        """
        기본 3D 재구성 기능 테스트
        
        목적: 재구성 파이프라인이 오류 없이 실행되는지 확인
        성공 기준: 학습이 완료되고 손실이 감소
        """
        grid_shape = (32, 32)
        adam_epochs = 100
        
        # 합성 데이터 생성
        ground_truth_height, bucket_images = generate_synthetic_data(
            shape=grid_shape,
            wavelengths=self.wavelengths,
            num_buckets=3,
            save_path=None
        )
        
        # wrapped_phases 계산
        num_lasers = len(self.wavelengths)
        wrapped_phases = []
        for i in range(num_lasers):
            phase = (4 * np.pi * ground_truth_height) / self.wavelengths[i]
            wrapped_phase = np.mod(phase, 2 * np.pi)
            wrapped_phases.append(wrapped_phase)
        
        wrapped_phases = np.array(wrapped_phases)
        wrapped_phases_t = torch.from_numpy(wrapped_phases).float().to(self.device)
        
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
        x = torch.linspace(0, 1, grid_shape[0], device=self.device)
        y = torch.linspace(0, 1, grid_shape[1], device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        # 타겟 준비: wrapped_phases_t 형태 (4, H, W) -> (4, 1, H*W)
        H, W = grid_shape
        targets = wrapped_phases_t.view(4, H*W).unsqueeze(1)
        
        # 손실 함수 및 옵티마이저
        loss_fn = ReconstructionLoss(
            wavelengths=self.wavelengths,
            smoothness_weight=1e-5
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 학습
        loss_history = []
        for epoch in range(adam_epochs):
            model.train()
            optimizer.zero_grad()
            
            predicted_height = model(coords).view(1, -1)
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
        
        print(f"\n[기본 재구성 테스트] 초기 손실: {initial_loss:.4e}, 최종 손실: {final_loss:.4e}")

    def test_reconstruction_small_grid(self):
        """
        작은 그리드 빠른 수렴 테스트
        
        목적: 작은 그리드에서 빠르게 수렴하는지 확인
        성공 기준: RMSE < 0.5 (합성 데이터에서)
        """
        grid_shape = (16, 16)  # 매우 작은 그리드
        adam_epochs = 500
        
        # 합성 데이터 생성
        ground_truth_height, bucket_images = generate_synthetic_data(
            shape=grid_shape,
            wavelengths=self.wavelengths,
            num_buckets=3,
            save_path=None
        )
        
        # wrapped_phases 계산
        num_lasers = len(self.wavelengths)
        wrapped_phases = []
        for i in range(num_lasers):
            phase = (4 * np.pi * ground_truth_height) / self.wavelengths[i]
            wrapped_phase = np.mod(phase, 2 * np.pi)
            wrapped_phases.append(wrapped_phase)
        
        wrapped_phases = np.array(wrapped_phases)
        
        ground_truth_height_t = torch.from_numpy(ground_truth_height).float().to(self.device)
        wrapped_phases_t = torch.from_numpy(wrapped_phases).float().to(self.device)
        
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
        x = torch.linspace(0, 1, grid_shape[0], device=self.device)
        y = torch.linspace(0, 1, grid_shape[1], device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        # 타겟 준비: wrapped_phases_t 형태 (4, H, W) -> (4, 1, H*W)
        H, W = grid_shape
        targets = wrapped_phases_t.view(4, H*W).unsqueeze(1)
        
        # 손실 함수 및 옵티마이저
        loss_fn = ReconstructionLoss(
            wavelengths=self.wavelengths,
            smoothness_weight=1e-6
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 학습
        for epoch in range(adam_epochs):
            model.train()
            optimizer.zero_grad()
            
            predicted_height = model(coords).view(1, -1)
            loss = loss_fn(predicted_height, coords, targets)
            
            loss.backward()
            optimizer.step()
        
        # 평가
        model.eval()
        with torch.no_grad():
            predicted_height_t = model(coords).view(grid_shape)
            predicted_height = predicted_height_t.cpu().numpy()
        
        rmse = np.sqrt(np.mean((predicted_height - ground_truth_height)**2))
        
        print(f"\n[작은 그리드 수렴 테스트] RMSE: {rmse:.6f}")
        
        # 검증: RMSE가 임계값 이하
        self.assertLess(rmse, 0.5, f"RMSE {rmse:.6f}가 임계값 0.5를 초과")


if __name__ == '__main__':
    unittest.main()
