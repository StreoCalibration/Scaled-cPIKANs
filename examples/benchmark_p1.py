"""
v2-P1 성능 벤치마크 스크립트

타일링 추론 및 AMP의 정량적 성과를 측정합니다.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import sys
import os
from pathlib import Path

# src 디렉토리 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import UNet
from src.utils.tiling import infer_with_tiling, estimate_memory_usage, tile_image, blend_tiles


def get_gpu_memory_mb():
    """현재 GPU 메모리 사용량 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0


def reset_gpu_memory():
    """GPU 메모리 통계 초기화"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


class SimpleUNet(nn.Module):
    """벤치마크용 간단한 UNet"""
    def __init__(self, in_channels=16, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def benchmark_tiling_inference(device='cuda'):
    """타일링 추론 벤치마크"""
    print("\n" + "="*70)
    print("📊 벤치마크 1: 타일링 추론 성능")
    print("="*70)
    
    results = {}
    
    # 모델 생성
    model = SimpleUNet(in_channels=16, out_channels=1).to(device)
    model.eval()
    
    # 테스트 이미지 크기
    test_sizes = [
        (16, 512, 512, "작은 이미지"),
        (16, 1024, 1024, "중간 이미지"),
        (16, 2048, 2048, "큰 이미지"),
    ]
    
    for channels, height, width, label in test_sizes:
        print(f"\n테스트: {label} ({channels}×{height}×{width})")
        img = np.random.rand(channels, height, width).astype(np.float32)
        
        # 1. 직접 추론 (가능한 경우)
        if height <= 1024:  # 메모리 제약
            print("  방법 1: 직접 추론")
            reset_gpu_memory()
            
            try:
                start_time = time.time()
                with torch.no_grad():
                    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
                    _ = model(img_tensor)
                    torch.cuda.synchronize()
                direct_time = time.time() - start_time
                direct_memory = get_gpu_memory_mb()
                
                print(f"    시간: {direct_time:.3f}초")
                print(f"    메모리: {direct_memory:.2f} MB")
                
                results[f"{label}_direct"] = {
                    "time_sec": direct_time,
                    "memory_mb": direct_memory
                }
            except RuntimeError as e:
                print(f"    실패: {str(e)}")
                results[f"{label}_direct"] = {"error": "OOM"}
        else:
            print("  방법 1: 직접 추론 - 건너뜀 (메모리 부족 예상)")
            results[f"{label}_direct"] = {"skipped": True}
        
        # 2. 타일링 추론
        print("  방법 2: 타일링 추론 (512×512, 오버랩 128)")
        reset_gpu_memory()
        
        start_time = time.time()
        result = infer_with_tiling(
            img,
            model,
            tile_size=512,
            overlap=128,
            device=device,
            batch_size=1,
            verbose=False
        )
        torch.cuda.synchronize()
        tiling_time = time.time() - start_time
        tiling_memory = get_gpu_memory_mb()
        
        print(f"    시간: {tiling_time:.3f}초")
        print(f"    메모리: {tiling_memory:.2f} MB")
        
        results[f"{label}_tiling"] = {
            "time_sec": tiling_time,
            "memory_mb": tiling_memory,
            "output_shape": result.shape
        }
        
        # 비교
        if f"{label}_direct" in results and "time_sec" in results[f"{label}_direct"]:
            speedup = results[f"{label}_direct"]["time_sec"] / tiling_time
            memory_reduction = (1 - tiling_memory / results[f"{label}_direct"]["memory_mb"]) * 100
            print(f"    비교: 속도 {speedup:.2f}배, 메모리 {memory_reduction:.1f}% 절감")
    
    return results


def benchmark_reconstruction_accuracy():
    """재구성 정확도 벤치마크"""
    print("\n" + "="*70)
    print("📊 벤치마크 2: 타일링 재구성 정확도")
    print("="*70)
    
    results = {}
    
    # 테스트 이미지: 균일한 값
    print("\n테스트 1: 균일한 이미지 (완벽한 재구성 기대)")
    img = np.ones((3, 1024, 1024)) * 0.5
    tiles, shape = tile_image(img, tile_size=512, overlap=128)
    reconstructed = blend_tiles(tiles, shape, tile_size=512, overlap=128, device='cpu')
    
    l2_error = np.sqrt(np.mean((reconstructed - img) ** 2))
    relative_l2 = l2_error / (np.sqrt(np.mean(img ** 2)) + 1e-10)
    max_error = np.max(np.abs(reconstructed - img))
    
    print(f"  L2 오차: {l2_error:.6e}")
    print(f"  상대 L2 오차: {relative_l2:.6e}")
    print(f"  최대 절대 오차: {max_error:.6e}")
    
    results["uniform_image"] = {
        "l2_error": float(l2_error),
        "relative_l2_error": float(relative_l2),
        "max_abs_error": float(max_error)
    }
    
    # 테스트 이미지: 그래디언트
    print("\n테스트 2: 그래디언트 이미지")
    x = np.linspace(0, 1, 1024)
    y = np.linspace(0, 1, 1024)
    xx, yy = np.meshgrid(x, y)
    img = np.stack([xx, yy, xx + yy], axis=0)
    
    tiles, shape = tile_image(img, tile_size=512, overlap=128)
    reconstructed = blend_tiles(tiles, shape, tile_size=512, overlap=128, device='cpu')
    
    l2_error = np.sqrt(np.mean((reconstructed - img) ** 2))
    relative_l2 = l2_error / (np.sqrt(np.mean(img ** 2)) + 1e-10)
    max_error = np.max(np.abs(reconstructed - img))
    
    print(f"  L2 오차: {l2_error:.6e}")
    print(f"  상대 L2 오차: {relative_l2:.6e}")
    print(f"  최대 절대 오차: {max_error:.6e}")
    
    results["gradient_image"] = {
        "l2_error": float(l2_error),
        "relative_l2_error": float(relative_l2),
        "max_abs_error": float(max_error)
    }
    
    # 테스트 이미지: 랜덤
    print("\n테스트 3: 랜덤 이미지")
    np.random.seed(42)
    img = np.random.rand(3, 1024, 1024)
    
    tiles, shape = tile_image(img, tile_size=512, overlap=128)
    reconstructed = blend_tiles(tiles, shape, tile_size=512, overlap=128, device='cpu')
    
    l2_error = np.sqrt(np.mean((reconstructed - img) ** 2))
    relative_l2 = l2_error / (np.sqrt(np.mean(img ** 2)) + 1e-10)
    max_error = np.max(np.abs(reconstructed - img))
    
    print(f"  L2 오차: {l2_error:.6e}")
    print(f"  상대 L2 오차: {relative_l2:.6e}")
    print(f"  최대 절대 오차: {max_error:.6e}")
    
    results["random_image"] = {
        "l2_error": float(l2_error),
        "relative_l2_error": float(relative_l2),
        "max_abs_error": float(max_error)
    }
    
    return results


def benchmark_memory_estimation():
    """메모리 추정 정확도"""
    print("\n" + "="*70)
    print("📊 벤치마크 3: 메모리 추정 vs 실제 사용량")
    print("="*70)
    
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("CUDA를 사용할 수 없어 건너뜁니다.")
        return {"skipped": True}
    
    model = SimpleUNet(in_channels=16, out_channels=1).to(device)
    model.eval()
    model_params = sum(p.numel() for p in model.parameters())
    
    test_cases = [
        (16, 512, 512),
        (16, 1024, 1024),
        (16, 2048, 2048),
    ]
    
    for channels, height, width in test_cases:
        label = f"{channels}×{height}×{width}"
        print(f"\n테스트: {label}")
        
        # 추정
        estimated = estimate_memory_usage(
            (channels, height, width),
            tile_size=512,
            overlap=128,
            model_params=model_params,
            batch_size=1,
            dtype='float32'
        )
        
        # 실제 측정
        img = np.random.rand(channels, height, width).astype(np.float32)
        reset_gpu_memory()
        
        _ = infer_with_tiling(
            img,
            model,
            tile_size=512,
            overlap=128,
            device=device,
            batch_size=1,
            verbose=False
        )
        
        actual_mb = get_gpu_memory_mb()
        
        print(f"  추정 메모리: {estimated['total_mb']:.2f} MB")
        print(f"  실제 메모리: {actual_mb:.2f} MB")
        print(f"  오차: {abs(estimated['total_mb'] - actual_mb):.2f} MB ({abs(estimated['total_mb'] - actual_mb) / actual_mb * 100:.1f}%)")
        
        results[label] = {
            "estimated_mb": estimated['total_mb'],
            "actual_mb": float(actual_mb),
            "error_mb": abs(estimated['total_mb'] - actual_mb),
            "error_percent": abs(estimated['total_mb'] - actual_mb) / actual_mb * 100
        }
    
    return results


def benchmark_amp_training(device='cuda'):
    """AMP 훈련 성능 비교"""
    print("\n" + "="*70)
    print("📊 벤치마크 4: AMP vs FP32 훈련 성능")
    print("="*70)
    
    if device == 'cpu' or not torch.cuda.is_available():
        print("CUDA를 사용할 수 없어 건너뜁니다.")
        return {"skipped": True}
    
    results = {}
    
    # 간단한 훈련 루프
    def train_epochs(model, optimizer, use_amp, num_epochs=10):
        model.train()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # 더미 데이터
        data = torch.randn(4, 16, 256, 256, device=device)
        target = torch.randn(4, 1, 256, 256, device=device)
        
        reset_gpu_memory()
        start_time = time.time()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = nn.functional.mse_loss(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = nn.functional.mse_loss(output, target)
                loss.backward()
                optimizer.step()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        memory = get_gpu_memory_mb()
        
        return elapsed, memory
    
    # FP32 훈련
    print("\n테스트 1: FP32 훈련")
    model_fp32 = SimpleUNet(in_channels=16, out_channels=1).to(device)
    optimizer_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=1e-3)
    
    fp32_time, fp32_memory = train_epochs(model_fp32, optimizer_fp32, use_amp=False, num_epochs=20)
    print(f"  시간: {fp32_time:.3f}초")
    print(f"  메모리: {fp32_memory:.2f} MB")
    
    results["fp32"] = {
        "time_sec": fp32_time,
        "memory_mb": float(fp32_memory)
    }
    
    # AMP 훈련
    print("\n테스트 2: AMP (FP16) 훈련")
    model_amp = SimpleUNet(in_channels=16, out_channels=1).to(device)
    optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=1e-3)
    
    amp_time, amp_memory = train_epochs(model_amp, optimizer_amp, use_amp=True, num_epochs=20)
    print(f"  시간: {amp_time:.3f}초")
    print(f"  메모리: {amp_memory:.2f} MB")
    
    results["amp"] = {
        "time_sec": amp_time,
        "memory_mb": float(amp_memory)
    }
    
    # 비교
    speedup = fp32_time / amp_time
    memory_reduction = (1 - amp_memory / fp32_memory) * 100
    
    print(f"\n✅ AMP 성능 향상:")
    print(f"  속도: {speedup:.2f}배 빠름")
    print(f"  메모리: {memory_reduction:.1f}% 절감")
    
    results["improvement"] = {
        "speedup": speedup,
        "memory_reduction_percent": memory_reduction
    }
    
    return results


def main():
    """메인 벤치마크 실행"""
    print("\n" + "="*70)
    print("🚀 v2-P1 성능 벤치마크 시작")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"디바이스: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
    
    all_results = {}
    
    # 벤치마크 실행
    try:
        all_results["tiling_inference"] = benchmark_tiling_inference(device)
    except Exception as e:
        print(f"❌ 타일링 추론 벤치마크 실패: {e}")
        all_results["tiling_inference"] = {"error": str(e)}
    
    try:
        all_results["reconstruction_accuracy"] = benchmark_reconstruction_accuracy()
    except Exception as e:
        print(f"❌ 재구성 정확도 벤치마크 실패: {e}")
        all_results["reconstruction_accuracy"] = {"error": str(e)}
    
    try:
        all_results["memory_estimation"] = benchmark_memory_estimation()
    except Exception as e:
        print(f"❌ 메모리 추정 벤치마크 실패: {e}")
        all_results["memory_estimation"] = {"error": str(e)}
    
    try:
        all_results["amp_training"] = benchmark_amp_training(device)
    except Exception as e:
        print(f"❌ AMP 훈련 벤치마크 실패: {e}")
        all_results["amp_training"] = {"error": str(e)}
    
    # 결과 저장
    output_dir = Path("outputs/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "p1_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ 벤치마크 완료!")
    print(f"결과 저장: {output_file}")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    main()
