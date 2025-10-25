"""
적응형 잔차 샘플링 테스트

AdaptiveResidualSampler 클래스의 기능을 검증합니다:
1. 초기화 및 샘플 생성
2. 잔차 업데이트
3. 적응형 정제 (refinement)
4. 최대 포인트 수 제한
"""

import unittest
import torch
import numpy as np
from src.data import AdaptiveResidualSampler


class TestAdaptiveResidualSampler(unittest.TestCase):
    """AdaptiveResidualSampler의 기본 기능 테스트"""
    
    def test_initialization(self):
        """초기화 및 초기 샘플 생성 테스트"""
        sampler = AdaptiveResidualSampler(
            n_initial_points=100,
            n_max_points=500,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0]
        )
        
        points = sampler.get_current_points()
        
        # 형태 검증
        self.assertEqual(points.shape, (100, 2))
        
        # 도메인 범위 검증
        self.assertTrue(torch.all(points >= 0.0))
        self.assertTrue(torch.all(points <= 1.0))
    
    def test_residual_update(self):
        """잔차 업데이트 테스트"""
        sampler = AdaptiveResidualSampler(
            n_initial_points=50,
            n_max_points=200,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0]
        )
        
        # 가짜 잔차 생성
        residuals = torch.rand(50, 1)
        
        # 잔차 업데이트
        sampler.update_residuals(residuals)
        
        # 잔차가 저장되었는지 확인
        self.assertIsNotNone(sampler.residuals)
        self.assertEqual(len(sampler.residuals), 50)
    
    def test_refinement(self):
        """적응형 정제 테스트"""
        sampler = AdaptiveResidualSampler(
            n_initial_points=100,
            n_max_points=500,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            refinement_ratio=0.2
        )
        
        initial_count = sampler.get_current_points().shape[0]
        
        # 비균일한 잔차 생성 (일부 포인트에 높은 잔차)
        residuals = torch.rand(100, 1)
        residuals[:10] = 10.0  # 처음 10개 포인트에 높은 잔차
        
        sampler.update_residuals(residuals)
        
        # 정제 수행
        refined = sampler.refine()
        
        self.assertTrue(refined, "정제가 수행되지 않았습니다.")
        
        # 포인트 수가 증가했는지 확인
        new_count = sampler.get_current_points().shape[0]
        self.assertGreater(new_count, initial_count)
        
        # 추가된 포인트 수가 예상 범위 내인지 확인
        expected_additional = int(initial_count * 0.2)
        self.assertAlmostEqual(new_count - initial_count, expected_additional, delta=5)
    
    def test_max_points_limit(self):
        """최대 포인트 수 제한 테스트"""
        sampler = AdaptiveResidualSampler(
            n_initial_points=90,
            n_max_points=100,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            refinement_ratio=0.2
        )
        
        # 잔차 업데이트 및 정제
        residuals = torch.rand(90, 1)
        residuals[:10] = 10.0
        
        sampler.update_residuals(residuals)
        refined = sampler.refine()
        
        # 최대값에 도달했으므로 더 이상 정제되지 않아야 함
        self.assertTrue(refined, "첫 정제는 성공해야 합니다.")
        
        current_count = sampler.get_current_points().shape[0]
        self.assertLessEqual(current_count, 100)
        
        # 다시 정제 시도
        residuals = torch.rand(current_count, 1)
        sampler.update_residuals(residuals)
        refined_again = sampler.refine()
        
        # 최대값에 도달했으므로 False 반환
        self.assertFalse(refined_again, "최대 포인트 수에 도달하면 False를 반환해야 합니다.")
    
    def test_3d_sampling(self):
        """3D 도메인에서의 샘플링 테스트"""
        sampler = AdaptiveResidualSampler(
            n_initial_points=50,
            n_max_points=200,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0]
        )
        
        points = sampler.get_current_points()
        
        # 3D 검증
        self.assertEqual(points.shape, (50, 3))
        self.assertTrue(torch.all(points >= 0.0))
        self.assertTrue(torch.all(points <= 1.0))
        
        # 잔차 업데이트 및 정제
        residuals = torch.rand(50, 1)
        residuals[:5] = 10.0
        
        sampler.update_residuals(residuals)
        refined = sampler.refine()
        
        self.assertTrue(refined)
        
        # 새 포인트도 3D
        new_points = sampler.get_current_points()
        self.assertEqual(new_points.shape[1], 3)
    
    def test_reset(self):
        """리셋 기능 테스트"""
        sampler = AdaptiveResidualSampler(
            n_initial_points=100,
            n_max_points=500,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0]
        )
        
        # 정제 수행
        residuals = torch.rand(100, 1)
        sampler.update_residuals(residuals)
        sampler.refine()
        
        refined_count = sampler.get_current_points().shape[0]
        self.assertGreater(refined_count, 100)
        
        # 리셋
        sampler.reset()
        
        # 초기 상태로 돌아왔는지 확인
        reset_count = sampler.get_current_points().shape[0]
        self.assertEqual(reset_count, 100)
        self.assertIsNone(sampler.residuals)


class TestAdaptiveRefinementStrategy(unittest.TestCase):
    """적응형 정제 전략의 효과성 테스트"""
    
    def test_high_residual_region_refinement(self):
        """높은 잔차 영역에 포인트가 집중되는지 테스트"""
        sampler = AdaptiveResidualSampler(
            n_initial_points=100,
            n_max_points=500,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            residual_threshold_percentile=75.0
        )
        
        points = sampler.get_current_points()
        
        # 특정 영역 (0.4~0.6, 0.4~0.6)에 높은 잔차 할당
        x, y = points[:, 0], points[:, 1]
        in_high_residual_region = (x >= 0.4) & (x <= 0.6) & (y >= 0.4) & (y <= 0.6)
        
        residuals = torch.ones(100, 1)
        residuals[in_high_residual_region] = 10.0
        
        sampler.update_residuals(residuals)
        sampler.refine()
        
        # 새 포인트 가져오기
        new_points = sampler.get_current_points()
        added_points = new_points[100:]  # 추가된 포인트만
        
        # 추가된 포인트 중 높은 잔차 영역 근처에 있는 비율 계산
        x_new, y_new = added_points[:, 0], added_points[:, 1]
        near_high_residual = (x_new >= 0.3) & (x_new <= 0.7) & (y_new >= 0.3) & (y_new <= 0.7)
        
        ratio_near_high_residual = near_high_residual.float().mean().item()
        
        # 적어도 일부는 높은 잔차 영역 근처에 있어야 함
        # 가우시안 노이즈로 인해 완벽하지 않을 수 있음
        self.assertGreater(
            ratio_near_high_residual,
            0.15,  # 15% 이상이면 적응형 샘플링이 작동하는 것으로 간주
            f"추가된 포인트의 {ratio_near_high_residual*100:.1f}%만이 높은 잔차 영역 근처에 있습니다."
        )


if __name__ == '__main__':
    unittest.main()
