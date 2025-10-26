"""
src/utils/tiling.py 모듈의 단위 테스트

타일링 및 블렌딩 기능의 정확성을 검증합니다.
"""

import unittest
import numpy as np
import torch
import sys
import os

# src 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.tiling import (
    tile_image,
    create_hanning_window,
    blend_tiles,
    infer_with_tiling,
    estimate_memory_usage
)


class TestTileImage(unittest.TestCase):
    """tile_image 함수 테스트"""
    
    def test_basic_tiling_3d(self):
        """기본 3D 이미지 타일링 테스트"""
        img = np.random.rand(3, 1024, 1024)
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        
        self.assertEqual(shape, (3, 1024, 1024))
        self.assertGreater(len(tiles), 0)
        
        # 각 타일 검증
        for tile_data, (y_start, y_end, x_start, x_end) in tiles:
            self.assertEqual(tile_data.shape[0], 3)  # 채널 수
            self.assertEqual(tile_data.shape[1], 512)  # 타일 높이
            self.assertEqual(tile_data.shape[2], 512)  # 타일 너비
    
    def test_basic_tiling_2d(self):
        """기본 2D 이미지 타일링 테스트"""
        img = np.random.rand(1024, 1024)
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        
        self.assertEqual(shape, (1024, 1024))
        self.assertGreater(len(tiles), 0)
        
        # 각 타일 검증
        for tile_data, _ in tiles:
            self.assertEqual(tile_data.shape[1], 512)
            self.assertEqual(tile_data.shape[2], 512)
    
    def test_tile_count(self):
        """예상 타일 수 검증"""
        img = np.random.rand(3, 1024, 1024)
        tiles, _ = tile_image(img, tile_size=512, overlap=128)
        
        # stride = 512 - 128 = 384
        # 1024 / 384 = 2.67 -> 3 타일 (y 방향)
        # 1024 / 384 = 2.67 -> 3 타일 (x 방향)
        # 총 3 * 3 = 9 타일
        self.assertEqual(len(tiles), 9)
    
    def test_edge_case_small_image(self):
        """타일 크기보다 작은 이미지"""
        img = np.random.rand(3, 256, 256)
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        
        # 이미지가 작으면 1개의 패딩된 타일만 생성
        self.assertEqual(len(tiles), 1)
        tile_data, _ = tiles[0]
        self.assertEqual(tile_data.shape, (3, 512, 512))
    
    def test_invalid_parameters(self):
        """잘못된 파라미터 처리"""
        img = np.random.rand(3, 1024, 1024)
        
        # tile_size <= overlap
        with self.assertRaises(ValueError):
            tile_image(img, tile_size=128, overlap=128)
        
        # 잘못된 차원
        img_4d = np.random.rand(2, 3, 1024, 1024)
        with self.assertRaises(ValueError):
            tile_image(img_4d, tile_size=512, overlap=128)
    
    def test_large_image_tiling(self):
        """대규모 이미지 타일링 (9344×7000)"""
        img = np.random.rand(16, 1024, 1024)  # 메모리 절약을 위해 축소
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        
        self.assertEqual(shape[0], 16)
        self.assertGreater(len(tiles), 0)


class TestHanningWindow(unittest.TestCase):
    """create_hanning_window 함수 테스트"""
    
    def test_window_shape(self):
        """윈도우 형태 검증"""
        window = create_hanning_window(512, 128, device='cpu')
        self.assertEqual(window.shape, (512, 512))
        self.assertEqual(window.dtype, torch.float32)
    
    def test_window_values(self):
        """윈도우 값 범위 검증"""
        window = create_hanning_window(512, 128, device='cpu')
        
        # 값은 0과 1 사이
        self.assertTrue(torch.all(window >= 0))
        self.assertTrue(torch.all(window <= 1))
        
        # 중앙 영역은 1에 가까움
        center_region = window[128:-128, 128:-128]
        self.assertTrue(torch.all(center_region > 0.99))
    
    def test_window_symmetry(self):
        """윈도우 대칭성 검증"""
        window = create_hanning_window(512, 128, device='cpu')
        
        # 수평 대칭
        self.assertTrue(torch.allclose(window, window.flip(0), atol=1e-6))
        # 수직 대칭
        self.assertTrue(torch.allclose(window, window.flip(1), atol=1e-6))
    
    def test_no_overlap(self):
        """오버랩이 없는 경우"""
        window = create_hanning_window(512, 0, device='cpu')
        # 모든 값이 1이어야 함
        self.assertTrue(torch.all(window == 1.0))


class TestBlendTiles(unittest.TestCase):
    """blend_tiles 함수 테스트"""
    
    def test_perfect_reconstruction(self):
        """완벽한 재구성 테스트 (동일한 값의 타일)"""
        # 균일한 이미지 생성
        img = np.ones((3, 1024, 1024)) * 0.5
        
        # 타일링
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        
        # 블렌딩
        reconstructed = blend_tiles(tiles, shape, tile_size=512, overlap=128, device='cpu')
        
        # 재구성 검증 (균일한 이미지는 완벽하게 재구성되어야 함)
        self.assertTrue(np.allclose(reconstructed, img, atol=1e-4))
    
    def test_shape_preservation(self):
        """형태 보존 테스트"""
        img = np.random.rand(3, 1024, 1024)
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        reconstructed = blend_tiles(tiles, shape, tile_size=512, overlap=128, device='cpu')
        
        self.assertEqual(reconstructed.shape, img.shape)
    
    def test_2d_reconstruction(self):
        """2D 이미지 재구성 테스트"""
        img = np.random.rand(1024, 1024)
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        reconstructed = blend_tiles(tiles, shape, tile_size=512, overlap=128, device='cpu')
        
        self.assertEqual(reconstructed.shape, img.shape)
    
    def test_smooth_transition(self):
        """부드러운 전환 검증 (seam artifact 없음)"""
        # 격자 패턴 이미지 (seam 테스트에 적합)
        img = np.ones((3, 1024, 1024))
        img[:, :512, :512] = 0.2
        img[:, 512:, 512:] = 0.8
        
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        reconstructed = blend_tiles(tiles, shape, tile_size=512, overlap=128, device='cpu')
        
        # 경계 영역에서 큰 불연속이 없어야 함
        # 오버랩 영역의 그래디언트 측정
        diff_y = np.abs(np.diff(reconstructed[0, :, 512], axis=0))
        diff_x = np.abs(np.diff(reconstructed[0, 512, :], axis=0))
        
        # 최대 그래디언트가 작아야 함 (부드러운 전환)
        self.assertLess(np.max(diff_y), 0.1)
        self.assertLess(np.max(diff_x), 0.1)


class TestInferWithTiling(unittest.TestCase):
    """infer_with_tiling 함수 테스트"""
    
    def setUp(self):
        """테스트용 간단한 모델 생성"""
        # 간단한 항등 모델 (입력을 그대로 출력)
        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                # 입력의 평균을 1채널로 반환
                return x.mean(dim=1, keepdim=True)
        
        self.model = IdentityModel()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_basic_inference(self):
        """기본 추론 테스트"""
        img = np.random.rand(3, 512, 512)
        
        result = infer_with_tiling(
            img,
            self.model,
            tile_size=256,
            overlap=64,
            device=self.device,
            verbose=False
        )
        
        # 출력 형태 검증 (1채널)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 512)
        self.assertEqual(result.shape[2], 512)
    
    def test_large_image_inference(self):
        """대규모 이미지 추론 테스트"""
        img = np.random.rand(3, 1024, 1024)
        
        result = infer_with_tiling(
            img,
            self.model,
            tile_size=512,
            overlap=128,
            device=self.device,
            batch_size=2,
            verbose=False
        )
        
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 1024)
        self.assertEqual(result.shape[2], 1024)
    
    def test_consistency_with_no_tiling(self):
        """타일링 없는 추론과의 일관성 검증"""
        img = np.random.rand(3, 512, 512)
        
        # 타일링 없이 추론
        self.model.eval()
        with torch.no_grad():
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
            direct_result = self.model(img_tensor).cpu().numpy().squeeze()
        
        # 타일링 추론 (타일 크기 = 이미지 크기, 오버랩 0)
        tiled_result = infer_with_tiling(
            img,
            self.model,
            tile_size=512,
            overlap=0,
            device=self.device,
            verbose=False
        ).squeeze()
        
        # 결과가 거의 동일해야 함
        self.assertTrue(np.allclose(direct_result, tiled_result, atol=1e-5))


class TestEstimateMemoryUsage(unittest.TestCase):
    """estimate_memory_usage 함수 테스트"""
    
    def test_memory_estimation(self):
        """메모리 사용량 추정 테스트"""
        memory = estimate_memory_usage(
            (16, 9344, 7000),
            tile_size=512,
            overlap=128,
            model_params=1_000_000,
            batch_size=1,
            dtype='float32'
        )
        
        # 필수 키 존재 확인
        self.assertIn('num_tiles', memory)
        self.assertIn('input_tile_mb', memory)
        self.assertIn('output_tile_mb', memory)
        self.assertIn('model_mb', memory)
        self.assertIn('total_mb', memory)
        
        # 양수 값 확인
        self.assertGreater(memory['total_mb'], 0)
        self.assertGreater(memory['num_tiles'], 0)
    
    def test_fp16_vs_fp32(self):
        """FP16과 FP32 메모리 차이 검증"""
        memory_fp32 = estimate_memory_usage(
            (16, 1024, 1024),
            tile_size=512,
            overlap=128,
            dtype='float32'
        )
        
        memory_fp16 = estimate_memory_usage(
            (16, 1024, 1024),
            tile_size=512,
            overlap=128,
            dtype='float16'
        )
        
        # FP16이 FP32의 약 절반
        ratio = memory_fp32['input_tile_mb'] / memory_fp16['input_tile_mb']
        self.assertAlmostEqual(ratio, 2.0, places=1)
    
    def test_batch_size_scaling(self):
        """배치 크기에 따른 메모리 스케일링"""
        memory_batch1 = estimate_memory_usage(
            (16, 1024, 1024),
            tile_size=512,
            overlap=128,
            batch_size=1
        )
        
        memory_batch4 = estimate_memory_usage(
            (16, 1024, 1024),
            tile_size=512,
            overlap=128,
            batch_size=4
        )
        
        # 배치 크기가 4배면 타일 메모리도 약 4배
        ratio = memory_batch4['input_tile_mb'] / memory_batch1['input_tile_mb']
        self.assertAlmostEqual(ratio, 4.0, places=1)


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def test_end_to_end_pipeline(self):
        """전체 파이프라인 테스트 (타일링 → 추론 → 블렌딩)"""
        # 테스트 이미지
        img = np.random.rand(3, 1024, 1024)
        
        # 간단한 모델
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                # 채널 평균
                return x.mean(dim=1, keepdim=True)
        
        model = SimpleModel()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 전체 파이프라인
        result = infer_with_tiling(
            img,
            model,
            tile_size=512,
            overlap=128,
            device=device,
            batch_size=2,
            verbose=True
        )
        
        # 검증
        self.assertEqual(result.shape, (1, 1024, 1024))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
    
    def test_reconstruction_accuracy(self):
        """재구성 정확도 테스트"""
        # 간단한 패턴 이미지
        img = np.zeros((3, 1024, 1024))
        img[0, :512, :] = 1.0
        img[1, :, :512] = 1.0
        img[2, 512:, 512:] = 1.0
        
        # 타일링 및 재구성 (추론 없이)
        tiles, shape = tile_image(img, tile_size=512, overlap=128)
        reconstructed = blend_tiles(tiles, shape, tile_size=512, overlap=128, device='cpu')
        
        # L2 오차 측정
        l2_error = np.sqrt(np.mean((reconstructed - img) ** 2))
        relative_l2 = l2_error / np.sqrt(np.mean(img ** 2))
        
        # 상대 L2 오차 < 1e-3 (목표)
        self.assertLess(relative_l2, 1e-3)
        
        print(f"\n재구성 상대 L2 오차: {relative_l2:.6e}")


if __name__ == '__main__':
    unittest.main()
