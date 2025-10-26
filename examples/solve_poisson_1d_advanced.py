"""
고급 기능을 사용한 1D Poisson 방정식 해결

이 스크립트는 Scaled-cPIKAN의 고급 기능들을 시연합니다:
1. 동적 손실 가중치 (DynamicWeightedLoss)
2. 적응형 콜로케이션 샘플링 (AdaptiveResidualSampler)

문제 설정:
    u''(x) = -π²sin(πx),  x ∈ [0,1]
    u(0) = 0,  u(1) = 0
    
분석해:
    u(x) = sin(πx)

사용법:
    # 기본 실행 (모든 고급 기능 활성화)
    python examples/solve_poisson_1d_advanced.py
    
    # 동적 가중치만 사용
    python examples/solve_poisson_1d_advanced.py --use-dynamic-weights --no-adaptive-sampling
    
    # 적응형 샘플링만 사용
    python examples/solve_poisson_1d_advanced.py --no-dynamic-weights --use-adaptive-sampling
    
    # 모든 고급 기능 비활성화 (기존 방식)
    python examples/solve_poisson_1d_advanced.py --no-dynamic-weights --no-adaptive-sampling
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# src 디렉토리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import Scaled_cPIKAN
from src.loss import PhysicsInformedLoss, DynamicWeightedLoss
from src.data import LatinHypercubeSampler, AdaptiveResidualSampler


def analytical_solution(x):
    """분석해: u(x) = sin(πx)"""
    return torch.sin(np.pi * x)


def create_pde_residual_fn():
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


def create_bc_fn():
    """경계 조건 함수 생성"""
    def bc_fn(model, points):
        """u(0) = 0, u(1) = 0"""
        u = model(points)
        return u  # 경계에서 0이어야 함
    
    return bc_fn


def compute_l2_error(model, device, domain_min=0.0, domain_max=1.0, n_test=1000):
    """상대 L2 오차 계산"""
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(domain_min, domain_max, n_test, device=device).reshape(-1, 1)
        u_pred = model(x_test)
        u_exact = analytical_solution(x_test)
        
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact)**2))
        l2_norm = torch.sqrt(torch.mean(u_exact**2))
        relative_error = (l2_error / l2_norm).item()
    
    return relative_error


def train(model, loss_fn, sampler, args, device):
    """모델 훈련"""
    print(f"\n{'='*70}")
    print(f"🚀 훈련 시작")
    print(f"{'='*70}")
    print(f"에포크: {args.epochs}")
    print(f"동적 가중치: {'✅' if args.use_dynamic_weights else '❌'}")
    print(f"적응형 샘플링: {'✅' if args.use_adaptive_sampling else '❌'}")
    
    # 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 경계 포인트 (고정)
    bc_points_0 = torch.tensor([[args.domain_min]], device=device)
    bc_points_1 = torch.tensor([[args.domain_max]], device=device)
    bc_points_dicts = [
        {'points': bc_points_0},
        {'points': bc_points_1}
    ]
    
    # 메트릭 기록
    history = {
        'epoch': [],
        'loss': [],
        'error': [],
        'n_points': []
    }
    
    # 가중치 기록 (동적 가중치 사용 시)
    if args.use_dynamic_weights:
        history['weight_pde'] = []
        history['weight_bc'] = []
    
    # 적응형 샘플링 설정
    is_adaptive = args.use_adaptive_sampling
    refinement_interval = args.refinement_interval if is_adaptive else None
    
    model.train()
    
    for epoch in range(args.epochs):
        # 샘플 포인트 가져오기
        if is_adaptive:
            pde_points = sampler.get_current_points()
        else:
            pde_points = sampler.sample()
        
        # 순전파
        optimizer.zero_grad()
        total_loss, loss_dict = loss_fn(model, pde_points, bc_points_dicts)
        
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
                pde_residual_fn = create_pde_residual_fn()
                residuals = pde_residual_fn(model, pde_points_copy)
                # 계산 그래프와 분리
                sampler.update_residuals(residuals.detach())
            
            refined = sampler.refine()
            if refined:
                n_points = sampler.get_current_points().shape[0]
                print(f"  [Epoch {epoch+1}] 📈 정제 완료 → {n_points}개 포인트")
            
            model.train()
        
        # 메트릭 기록 (매 log_interval 마다)
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            error = compute_l2_error(model, device, args.domain_min, args.domain_max)
            n_points = pde_points.shape[0]
            
            history['epoch'].append(epoch + 1)
            history['loss'].append(total_loss.item())
            history['error'].append(error)
            history['n_points'].append(n_points)
            
            # 동적 가중치 기록
            if args.use_dynamic_weights and 'weights' in loss_dict:
                weights = loss_dict['weights']
                history['weight_pde'].append(weights.get('loss_pde', 1.0))
                history['weight_bc'].append(weights.get('loss_bc', 1.0))
            
            # 출력
            log_str = f"  Epoch {epoch+1:5d} | Loss: {total_loss.item():.4e} | Error: {error:.4e} | Points: {n_points}"
            
            if args.use_dynamic_weights and 'weights' in loss_dict:
                weights = loss_dict['weights']
                log_str += f" | w_pde: {weights['loss_pde']:.3f} | w_bc: {weights['loss_bc']:.3f}"
            
            print(log_str)
    
    print(f"\n✅ 훈련 완료!")
    
    return history


def visualize_results(model, history, args, device):
    """결과 시각화"""
    print(f"\n{'='*70}")
    print(f"📊 결과 시각화")
    print(f"{'='*70}")
    
    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 예측 vs 분석해
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(args.domain_min, args.domain_max, 1000, device=device).reshape(-1, 1)
        u_pred = model(x_test).cpu().numpy()
        u_exact = analytical_solution(x_test).cpu().numpy()
        x_test_np = x_test.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) 예측 vs 분석해
    axes[0, 0].plot(x_test_np, u_exact, 'b-', linewidth=2, label='분석해')
    axes[0, 0].plot(x_test_np, u_pred, 'r--', linewidth=2, label='예측')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u(x)')
    axes[0, 0].set_title('예측 vs 분석해')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) 절대 오차
    error = np.abs(u_pred - u_exact)
    axes[0, 1].plot(x_test_np, error, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('|u_pred - u_exact|')
    axes[0, 1].set_title('절대 오차')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # (c) 손실 및 오차 히스토리
    axes[1, 0].plot(history['epoch'], history['loss'], 'b-', linewidth=2, label='Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('훈련 손실')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    ax2 = axes[1, 0].twinx()
    ax2.plot(history['epoch'], history['error'], 'r--', linewidth=2, label='Error')
    ax2.set_ylabel('Relative L2 Error', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')
    
    # (d) 콜로케이션 포인트 수
    if args.use_adaptive_sampling:
        axes[1, 1].plot(history['epoch'], history['n_points'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('콜로케이션 포인트 수')
        axes[1, 1].set_title('적응형 샘플링 진행')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '적응형 샘플링 미사용', 
                       ha='center', va='center', fontsize=12, 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
    
    # 가중치 히스토리 (동적 가중치 사용 시)
    if args.use_dynamic_weights and 'weight_pde' in history:
        # 새 figure 생성
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['epoch'], history['weight_pde'], 'b-', linewidth=2, label='w_pde')
        ax.plot(history['epoch'], history['weight_bc'], 'r-', linewidth=2, label='w_bc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('가중치')
        ax.set_title('동적 손실 가중치 변화')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2.savefig(output_dir / 'weights_history.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  가중치 히스토리 저장: {output_dir / 'weights_history.png'}")
    
    plt.tight_layout()
    fig.savefig(output_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  결과 시각화 저장: {output_dir / 'results.png'}")
    
    # 최종 메트릭 출력
    final_error = history['error'][-1]
    print(f"\n📈 최종 메트릭:")
    print(f"  상대 L2 오차: {final_error:.6e}")
    print(f"  콜로케이션 포인트: {history['n_points'][-1]}개")


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='고급 기능을 사용한 1D Poisson 방정식 해결',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 고급 기능 플래그
    parser.add_argument(
        '--use-dynamic-weights',
        action='store_true',
        default=True,
        help='동적 손실 가중치 사용 (기본: True)'
    )
    parser.add_argument(
        '--no-dynamic-weights',
        action='store_false',
        dest='use_dynamic_weights',
        help='동적 손실 가중치 비활성화'
    )
    parser.add_argument(
        '--use-adaptive-sampling',
        action='store_true',
        default=True,
        help='적응형 콜로케이션 샘플링 사용 (기본: True)'
    )
    parser.add_argument(
        '--no-adaptive-sampling',
        action='store_false',
        dest='use_adaptive_sampling',
        help='적응형 샘플링 비활성화'
    )
    
    # 도메인 설정
    parser.add_argument(
        '--domain-min',
        type=float,
        default=0.0,
        help='도메인 최솟값 (기본: 0.0)'
    )
    parser.add_argument(
        '--domain-max',
        type=float,
        default=1.0,
        help='도메인 최댓값 (기본: 1.0)'
    )
    
    # 모델 설정
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[32, 32],
        help='은닉층 차원 (기본: [32, 32])'
    )
    parser.add_argument(
        '--cheby-order',
        type=int,
        default=4,
        help='Chebyshev 다항식 차수 (기본: 4)'
    )
    
    # 샘플링 설정
    parser.add_argument(
        '--n-initial-points',
        type=int,
        default=50,
        help='초기 콜로케이션 포인트 수 (기본: 50)'
    )
    parser.add_argument(
        '--n-max-points',
        type=int,
        default=200,
        help='최대 콜로케이션 포인트 수 (적응형 샘플링 시, 기본: 200)'
    )
    parser.add_argument(
        '--refinement-interval',
        type=int,
        default=500,
        help='적응형 정제 주기 (에포크, 기본: 500)'
    )
    
    # 손실 가중치 설정
    parser.add_argument(
        '--pde-weight',
        type=float,
        default=1.0,
        help='PDE 잔차 손실 초기 가중치 (기본: 1.0)'
    )
    parser.add_argument(
        '--bc-weight',
        type=float,
        default=10.0,
        help='경계 조건 손실 초기 가중치 (기본: 10.0)'
    )
    parser.add_argument(
        '--gradnorm-alpha',
        type=float,
        default=1.5,
        help='GradNorm alpha 파라미터 (기본: 1.5)'
    )
    parser.add_argument(
        '--gradnorm-lr',
        type=float,
        default=0.025,
        help='GradNorm 가중치 학습률 (기본: 0.025)'
    )
    
    # 훈련 설정
    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='훈련 에포크 수 (기본: 5000)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='학습률 (기본: 0.001)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='로그 출력 간격 (기본: 100)'
    )
    
    # 출력 설정
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/poisson_1d_advanced',
        help='출력 디렉토리 (기본: outputs/poisson_1d_advanced)'
    )
    
    # 디바이스 설정
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='연산 디바이스 (기본: cuda if available)'
    )
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"🚀 고급 1D Poisson 해법")
    print(f"{'='*70}")
    print(f"디바이스: {args.device}")
    print(f"동적 가중치: {'✅' if args.use_dynamic_weights else '❌'}")
    print(f"적응형 샘플링: {'✅' if args.use_adaptive_sampling else '❌'}")
    
    device = torch.device(args.device)
    
    # 모델 생성
    layers_dims = [1] + args.hidden_dims + [1]
    model = Scaled_cPIKAN(
        layers_dims=layers_dims,
        cheby_order=args.cheby_order,
        domain_min=torch.tensor([args.domain_min]),
        domain_max=torch.tensor([args.domain_max])
    ).to(device)
    
    print(f"\n모델 구조:")
    print(f"  레이어: {layers_dims}")
    print(f"  Chebyshev 차수: {args.cheby_order}")
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 기본 손실 함수
    base_loss_fn = PhysicsInformedLoss(
        pde_residual_fn=create_pde_residual_fn(),
        bc_fns=[create_bc_fn()],
        loss_weights={'pde': args.pde_weight, 'bc': args.bc_weight}
    )
    
    # 동적 가중치 적용 여부
    if args.use_dynamic_weights:
        loss_fn = DynamicWeightedLoss(
            base_loss_fn=base_loss_fn,
            loss_names=['loss_pde', 'loss_bc'],
            alpha=args.gradnorm_alpha,
            learning_rate=args.gradnorm_lr
        )
        print(f"\n✅ 동적 가중치 활성화")
        print(f"  Alpha: {args.gradnorm_alpha}")
        print(f"  학습률: {args.gradnorm_lr}")
    else:
        loss_fn = base_loss_fn
        print(f"\n❌ 고정 가중치 사용")
        print(f"  PDE 가중치: {args.pde_weight}")
        print(f"  BC 가중치: {args.bc_weight}")
    
    # 샘플러 선택
    if args.use_adaptive_sampling:
        sampler = AdaptiveResidualSampler(
            n_initial_points=args.n_initial_points,
            n_max_points=args.n_max_points,
            domain_min=[args.domain_min],
            domain_max=[args.domain_max],
            refinement_ratio=0.2,
            residual_threshold_percentile=75.0,
            device=device
        )
        print(f"\n✅ 적응형 샘플링 활성화")
        print(f"  초기 포인트: {args.n_initial_points}")
        print(f"  최대 포인트: {args.n_max_points}")
        print(f"  정제 주기: {args.refinement_interval} 에포크")
    else:
        sampler = LatinHypercubeSampler(
            n_points=args.n_initial_points,
            domain_min=[args.domain_min],
            domain_max=[args.domain_max],
            device=device
        )
        print(f"\n❌ 고정 샘플링 사용")
        print(f"  포인트 수: {args.n_initial_points}")
    
    # 훈련
    history = train(model, loss_fn, sampler, args, device)
    
    # 결과 시각화
    visualize_results(model, history, args, device)
    
    print(f"\n{'='*70}")
    print(f"🎉 완료!")
    print(f"{'='*70}")
    print(f"결과 디렉토리: {args.output_dir}")


if __name__ == '__main__':
    main()
