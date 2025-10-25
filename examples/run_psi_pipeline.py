"""
위상천이간섭법(PSI) 기반 3D 높이 재구성 파이프라인

이 스크립트는 합성 bucket 이미지로부터 3D 표면 높이를 재구성하는
전체 파이프라인을 제공합니다.

단계:
    1. generate: 합성 bucket 이미지 데이터 생성
    2. pretrain: UNet 모델 사전학습
    3. finetune: PINN 모델 미세조정 (선택적)
    4. inference: 새로운 데이터에 대한 추론
    5. test: 성능 평가 및 시각화
    6. all: 전체 파이프라인 실행 (1→2→4→5)

사용법:
    # 전체 파이프라인 실행
    python examples/run_psi_pipeline.py all
    
    # 개별 단계 실행
    python examples/run_psi_pipeline.py generate
    python examples/run_psi_pipeline.py pretrain
    python examples/run_psi_pipeline.py inference
    python examples/run_psi_pipeline.py test
    
    # 옵션 지정
    python examples/run_psi_pipeline.py all --num-samples 50 --epochs 100 --device cuda
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# src 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import UNet, Scaled_cPIKAN
from src.loss import UNetPhysicsLoss, PinnReconstructionLoss
from src.data_generator import generate_synthetic_data, DEFAULT_WAVELENGTHS
from src.data import WaferPatchDataset


class PSIPipeline:
    """위상천이간섭법 기반 3D 재구성 파이프라인"""
    
    def __init__(self, config):
        """
        Args:
            config: argparse Namespace 객체 (설정 값들)
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # 경로 설정
        self.output_dir = Path(config.output_dir)
        self.data_dir = self.output_dir / 'synthetic_data'
        self.train_dir = self.data_dir / 'train'
        self.test_dir = self.data_dir / 'test'
        self.model_dir = self.output_dir / 'models'
        self.result_dir = self.output_dir / 'results'
        self.viz_dir = self.result_dir / 'visualizations'
        
        # 디렉토리 생성
        for dir_path in [self.train_dir, self.test_dir, self.model_dir, 
                        self.result_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # PSI 설정
        self.wavelengths = DEFAULT_WAVELENGTHS
        self.num_buckets = 4
        self.num_channels = len(self.wavelengths) * self.num_buckets  # 16
        
        print(f"🔧 파이프라인 초기화 완료")
        print(f"   디바이스: {self.device}")
        print(f"   출력 디렉토리: {self.output_dir}")
        print(f"   파장: {self.wavelengths}")
        print(f"   Bucket 수: {self.num_buckets}")
        print(f"   입력 채널: {self.num_channels}")
    
    def generate_data(self):
        """합성 bucket 이미지 데이터 생성"""
        print("\n" + "="*70)
        print("📊 1단계: 합성 데이터 생성")
        print("="*70)
        
        num_train = self.config.num_train_samples
        num_test = self.config.num_test_samples
        img_size = self.config.image_size
        
        print(f"훈련 샘플: {num_train}, 테스트 샘플: {num_test}")
        print(f"이미지 크기: {img_size}x{img_size}")
        
        # 재현성을 위한 시드 고정
        np.random.seed(42)
        
        # 훈련 데이터 생성
        print(f"\n📁 훈련 데이터 생성 중... ({self.train_dir})")
        for i in tqdm(range(num_train), desc="훈련 데이터"):
            # 재현성을 위한 시드 설정
            np.random.seed(42 + i)
            
            sample_dir = self.train_dir / f"sample_{i:04d}"
            sample_dir.mkdir(exist_ok=True)
            
            height_map, buckets = generate_synthetic_data(
                shape=(img_size, img_size),
                wavelengths=self.wavelengths,
                num_buckets=self.num_buckets,
                save_path=None  # 수동 저장
            )
            
            # Ground truth 저장
            np.save(sample_dir / "ground_truth.npy", height_map)
            
            # Bucket 이미지 저장 (BMP 형식)
            for laser_idx in range(len(self.wavelengths)):
                for bucket_idx in range(self.num_buckets):
                    channel_idx = laser_idx * self.num_buckets + bucket_idx
                    bucket_img = buckets[laser_idx, bucket_idx]
                    img = Image.fromarray(bucket_img.astype(np.uint8))
                    img.save(sample_dir / f"bucket_{channel_idx:02d}.bmp")
        
        # 테스트 데이터 생성
        print(f"\n📁 테스트 데이터 생성 중... ({self.test_dir})")
        for i in tqdm(range(num_test), desc="테스트 데이터"):
            # 테스트는 다른 시드 사용
            np.random.seed(10000 + i)
            
            sample_dir = self.test_dir / f"sample_{i:04d}"
            sample_dir.mkdir(exist_ok=True)
            
            height_map, buckets = generate_synthetic_data(
                shape=(img_size, img_size),
                wavelengths=self.wavelengths,
                num_buckets=self.num_buckets,
                save_path=None  # 수동 저장
            )
            
            np.save(sample_dir / "ground_truth.npy", height_map)
            
            for laser_idx in range(len(self.wavelengths)):
                for bucket_idx in range(self.num_buckets):
                    channel_idx = laser_idx * self.num_buckets + bucket_idx
                    bucket_img = buckets[laser_idx, bucket_idx]
                    img = Image.fromarray(bucket_img.astype(np.uint8))
                    img.save(sample_dir / f"bucket_{channel_idx:02d}.bmp")
        
        print(f"\n✅ 데이터 생성 완료!")
        print(f"   훈련: {num_train}개 샘플")
        print(f"   테스트: {num_test}개 샘플")
    
    def train_unet(self):
        """UNet 모델 사전학습"""
        print("\n" + "="*70)
        print("🎓 2단계: UNet 사전학습")
        print("="*70)
        
        # 데이터셋 및 데이터로더 준비
        train_dataset = WaferPatchDataset(
            data_dir=str(self.train_dir),
            patch_size=self.config.patch_size,
            num_channels=self.num_channels,
            output_format='bmp'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Windows에서는 0 권장
        )
        
        print(f"훈련 데이터: {len(train_dataset)}개 패치")
        print(f"배치 크기: {self.config.batch_size}")
        
        # 모델 생성
        model = UNet(
            n_channels=self.num_channels,
            n_classes=1,
            bilinear=True
        ).to(self.device)
        
        print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
        
        # 손실 함수 및 옵티마이저
        criterion = UNetPhysicsLoss(
            wavelengths=self.wavelengths,
            num_buckets=self.num_buckets,
            smoothness_weight=self.config.smoothness_weight
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.config.epochs // 3, 
            gamma=0.5
        )
        
        # 학습
        print(f"\n🚀 학습 시작 (에포크: {self.config.epochs})")
        best_loss = float('inf')
        history = {'epoch': [], 'loss': [], 'loss_data': [], 'loss_smoothness': []}
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_metrics = {'loss_data': 0.0, 'loss_smoothness': 0.0}
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for batch_idx, (input_buckets, target_height) in enumerate(pbar):
                input_buckets = input_buckets.to(self.device)
                target_height = target_height.to(self.device)
                
                optimizer.zero_grad()
                
                # 순전파
                predicted_height = model(input_buckets)
                loss = criterion(predicted_height, input_buckets)
                
                # 역전파
                loss.backward()
                optimizer.step()
                
                # 메트릭 누적
                epoch_loss += loss.item()
                for key in ['loss_data', 'loss_smoothness']:
                    epoch_metrics[key] += criterion.metrics[key]
                
                # 진행 표시줄 업데이트
                pbar.set_postfix({
                    'loss': f"{loss.item():.4e}",
                    'data': f"{criterion.metrics['loss_data']:.4e}",
                    'smooth': f"{criterion.metrics['loss_smoothness']:.4e}"
                })
            
            # 에포크 평균
            avg_loss = epoch_loss / len(train_loader)
            for key in epoch_metrics:
                epoch_metrics[key] /= len(train_loader)
            
            # 기록
            history['epoch'].append(epoch)
            history['loss'].append(avg_loss)
            history['loss_data'].append(epoch_metrics['loss_data'])
            history['loss_smoothness'].append(epoch_metrics['loss_smoothness'])
            
            # 학습률 스케줄러
            scheduler.step()
            
            # 체크포인트 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': vars(self.config)
                }
                torch.save(checkpoint, self.model_dir / 'unet_best.pth')
                print(f"✨ 최고 모델 저장 (loss: {best_loss:.6e})")
        
        # 최종 모델 저장
        torch.save(model.state_dict(), self.model_dir / 'unet_final.pth')
        
        # 학습 이력 저장
        with open(self.result_dir / 'unet_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # 학습 곡선 시각화
        self._plot_training_curve(history, 'UNet')
        
        print(f"\n✅ UNet 학습 완료!")
        print(f"   최고 손실: {best_loss:.6e}")
        print(f"   모델 저장: {self.model_dir / 'unet_best.pth'}")
    
    def finetune_pinn(self):
        """PINN 모델 미세조정 (선택적)"""
        print("\n" + "="*70)
        print("🔬 3단계: PINN 미세조정 (선택적)")
        print("="*70)
        print("⚠️  이 기능은 고급 사용자를 위한 것입니다.")
        print("    대부분의 경우 UNet만으로 충분합니다.")
        print("    구현 예정...")
    
    def inference(self):
        """추론 실행"""
        print("\n" + "="*70)
        print("🔮 4단계: 추론")
        print("="*70)
        
        # 모델 로드
        model = UNet(
            n_channels=self.num_channels,
            n_classes=1,
            bilinear=True
        ).to(self.device)
        
        checkpoint_path = self.model_dir / 'unet_best.pth'
        if not checkpoint_path.exists():
            print(f"❌ 모델 파일이 없습니다: {checkpoint_path}")
            print("   먼저 'pretrain' 단계를 실행하세요.")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ 모델 로드 완료: {checkpoint_path}")
        print(f"   학습 에포크: {checkpoint['epoch']}")
        print(f"   학습 손실: {checkpoint['loss']:.6e}")
        
        # 테스트 데이터셋 준비
        test_dataset = WaferPatchDataset(
            data_dir=str(self.test_dir),
            patch_size=self.config.patch_size,
            num_channels=self.num_channels,
            output_format='bmp'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\n🔍 추론 실행 중... ({len(test_dataset)}개 샘플)")
        
        # 추론 결과 저장
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for idx, (input_buckets, target_height) in enumerate(tqdm(test_loader, desc="추론")):
                input_buckets = input_buckets.to(self.device)
                
                # 예측
                predicted_height = model(input_buckets)
                
                # CPU로 이동 및 저장
                pred = predicted_height.cpu().numpy().squeeze()
                gt = target_height.numpy().squeeze()
                
                predictions.append(pred)
                ground_truths.append(gt)
                
                # 처음 몇 개만 시각화 저장
                if idx < self.config.num_visualize:
                    self._save_inference_visualization(
                        pred, gt, idx, 
                        input_buckets.cpu().numpy().squeeze()
                    )
        
        # 결과 저장
        results = {
            'predictions': [p.tolist() for p in predictions],
            'ground_truths': [g.tolist() for g in ground_truths]
        }
        
        with open(self.result_dir / 'inference_results.json', 'w') as f:
            json.dump(results, f)
        
        print(f"\n✅ 추론 완료!")
        print(f"   결과 저장: {self.result_dir / 'inference_results.json'}")
        print(f"   시각화: {self.viz_dir} (상위 {self.config.num_visualize}개)")
    
    def test(self):
        """성능 평가"""
        print("\n" + "="*70)
        print("📊 5단계: 성능 평가")
        print("="*70)
        
        # 추론 결과 로드
        results_path = self.result_dir / 'inference_results.json'
        if not results_path.exists():
            print(f"❌ 추론 결과가 없습니다: {results_path}")
            print("   먼저 'inference' 단계를 실행하세요.")
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        predictions = [np.array(p) for p in results['predictions']]
        ground_truths = [np.array(g) for g in results['ground_truths']]
        
        print(f"평가 샘플 수: {len(predictions)}")
        
        # 메트릭 계산
        rmse_list = []
        mae_list = []
        mape_list = []
        
        for pred, gt in zip(predictions, ground_truths):
            # RMSE
            rmse = np.sqrt(np.mean((pred - gt) ** 2))
            rmse_list.append(rmse)
            
            # MAE
            mae = np.mean(np.abs(pred - gt))
            mae_list.append(mae)
            
            # MAPE (평균 절대 백분율 오차)
            # 0으로 나누기 방지
            mask = np.abs(gt) > 1e-10
            if mask.any():
                mape = np.mean(np.abs((pred[mask] - gt[mask]) / gt[mask])) * 100
                mape_list.append(mape)
        
        # 통계
        metrics = {
            'rmse_mean': float(np.mean(rmse_list)),
            'rmse_std': float(np.std(rmse_list)),
            'rmse_min': float(np.min(rmse_list)),
            'rmse_max': float(np.max(rmse_list)),
            'mae_mean': float(np.mean(mae_list)),
            'mae_std': float(np.std(mae_list)),
            'mape_mean': float(np.mean(mape_list)) if mape_list else None,
            'mape_std': float(np.std(mape_list)) if mape_list else None,
            'num_samples': len(predictions)
        }
        
        # 결과 출력
        print("\n📈 평가 결과:")
        print(f"   RMSE: {metrics['rmse_mean']:.6e} ± {metrics['rmse_std']:.6e}")
        print(f"         (min: {metrics['rmse_min']:.6e}, max: {metrics['rmse_max']:.6e})")
        print(f"   MAE:  {metrics['mae_mean']:.6e} ± {metrics['mae_std']:.6e}")
        if metrics['mape_mean'] is not None:
            print(f"   MAPE: {metrics['mape_mean']:.2f}% ± {metrics['mape_std']:.2f}%")
        
        # 메트릭 저장
        with open(self.result_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 메트릭 시각화
        self._plot_metrics(rmse_list, mae_list, mape_list)
        
        print(f"\n✅ 평가 완료!")
        print(f"   메트릭 저장: {self.result_dir / 'metrics.json'}")
    
    def run_all(self):
        """전체 파이프라인 실행"""
        print("\n" + "="*70)
        print("🚀 전체 파이프라인 실행")
        print("="*70)
        
        self.generate_data()
        self.train_unet()
        self.inference()
        self.test()
        
        print("\n" + "="*70)
        print("🎉 전체 파이프라인 완료!")
        print("="*70)
        print(f"결과 디렉토리: {self.output_dir}")
    
    def _plot_training_curve(self, history, model_name):
        """학습 곡선 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 총 손실
        axes[0].plot(history['epoch'], history['loss'], 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title(f'{model_name} Training Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # 세부 손실
        axes[1].plot(history['epoch'], history['loss_data'], 'r-', label='Data Loss', linewidth=2)
        axes[1].plot(history['epoch'], history['loss_smoothness'], 'g-', label='Smoothness Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'{model_name} Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.result_dir / f'{model_name.lower()}_training_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_inference_visualization(self, pred, gt, idx, buckets):
        """추론 결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 예측 높이
        im1 = axes[0, 0].imshow(pred, cmap='viridis')
        axes[0, 0].set_title('예측 높이')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Ground truth
        im2 = axes[0, 1].imshow(gt, cmap='viridis')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 절대 오차
        error = np.abs(pred - gt)
        im3 = axes[0, 2].imshow(error, cmap='hot')
        axes[0, 2].set_title('절대 오차')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Bucket 이미지 샘플 (처음 3개)
        for i in range(3):
            if i < buckets.shape[0]:
                axes[1, i].imshow(buckets[i], cmap='gray')
                axes[1, i].set_title(f'Bucket {i}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'inference_{idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics(self, rmse_list, mae_list, mape_list):
        """메트릭 분포 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # RMSE 히스토그램
        axes[0].hist(rmse_list, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('RMSE')
        axes[0].set_ylabel('빈도')
        axes[0].set_title(f'RMSE 분포\n평균: {np.mean(rmse_list):.6e}')
        axes[0].grid(True, alpha=0.3)
        
        # MAE 히스토그램
        axes[1].hist(mae_list, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('MAE')
        axes[1].set_ylabel('빈도')
        axes[1].set_title(f'MAE 분포\n평균: {np.mean(mae_list):.6e}')
        axes[1].grid(True, alpha=0.3)
        
        # MAPE 히스토그램
        if mape_list:
            axes[2].hist(mape_list, bins=20, color='red', alpha=0.7, edgecolor='black')
            axes[2].set_xlabel('MAPE (%)')
            axes[2].set_ylabel('빈도')
            axes[2].set_title(f'MAPE 분포\n평균: {np.mean(mape_list):.2f}%')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='위상천이간섭법(PSI) 기반 3D 높이 재구성 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 전체 파이프라인 실행 (기본 설정)
  python %(prog)s all
  
  # 개별 단계 실행
  python %(prog)s generate
  python %(prog)s pretrain
  python %(prog)s inference
  python %(prog)s test
  
  # 옵션 지정
  python %(prog)s all --num-train-samples 100 --epochs 50 --device cuda
  python %(prog)s pretrain --batch-size 8 --learning-rate 0.001
        """
    )
    
    # 필수 인자: 명령어
    parser.add_argument(
        'command',
        choices=['generate', 'pretrain', 'finetune', 'inference', 'test', 'all'],
        help='실행할 단계 선택'
    )
    
    # 경로 설정
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='출력 디렉토리 경로 (기본값: outputs)'
    )
    
    # 데이터 생성 옵션
    parser.add_argument(
        '--num-train-samples',
        type=int,
        default=20,
        help='훈련 샘플 수 (기본값: 20)'
    )
    parser.add_argument(
        '--num-test-samples',
        type=int,
        default=5,
        help='테스트 샘플 수 (기본값: 5)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=256,
        help='생성할 이미지 크기 (기본값: 256)'
    )
    
    # 학습 옵션
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='학습 에포크 수 (기본값: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='배치 크기 (기본값: 4)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='학습률 (기본값: 0.001)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=256,
        help='패치 크기 (기본값: 256)'
    )
    parser.add_argument(
        '--smoothness-weight',
        type=float,
        default=1e-4,
        help='평활도 손실 가중치 (기본값: 1e-4)'
    )
    
    # 추론/평가 옵션
    parser.add_argument(
        '--num-visualize',
        type=int,
        default=5,
        help='시각화할 결과 수 (기본값: 5)'
    )
    
    # 디바이스 설정
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='연산 디바이스 (기본값: cuda)'
    )
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # 파이프라인 생성
    pipeline = PSIPipeline(args)
    
    # 명령어에 따라 실행
    if args.command == 'generate':
        pipeline.generate_data()
    elif args.command == 'pretrain':
        pipeline.train_unet()
    elif args.command == 'finetune':
        pipeline.finetune_pinn()
    elif args.command == 'inference':
        pipeline.inference()
    elif args.command == 'test':
        pipeline.test()
    elif args.command == 'all':
        pipeline.run_all()


if __name__ == '__main__':
    main()
