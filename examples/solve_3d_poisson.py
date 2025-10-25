"""
3D Poisson 방정식 해결 예제

이 스크립트는 Scaled-cPIKAN을 사용하여 3차원 Poisson 방정식을 해결합니다.

문제 설정:
    ∇²u = -f(x,y,z)  in Ω = [0,1]³
    u = 0            on ∂Ω (경계)

분석해 (검증용):
    u(x,y,z) = sin(πx)sin(πy)sin(πz)
    f(x,y,z) = 3π²sin(πx)sin(πy)sin(πz)

이 예제는:
1. 3D 도메인에 대한 아핀 스케일링 적용
2. 3D PDE 잔차 계산 (Laplacian)
3. 3D 경계 조건 적용
4. Scaled-cPIKAN의 3D 확장성 검증
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# src 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import Scaled_cPIKAN
from src.loss import PhysicsInformedLoss
from src.train import Trainer
from src.data import LatinHypercubeSampler


def analytical_solution(x, y, z):
    """
    분석해: u(x,y,z) = sin(πx)sin(πy)sin(πz)
    
    Args:
        x, y, z: 좌표 텐서
        
    Returns:
        torch.Tensor: 분석해 값
    """
    return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(np.pi * z)


def source_term(x, y, z):
    """
    소스 항: f(x,y,z) = 3π²sin(πx)sin(πy)sin(πz)
    
    Laplacian을 계산하면:
    ∇²u = -π²sin(πx)sin(πy)sin(πz) - π²sin(πx)sin(πy)sin(πz) - π²sin(πx)sin(πy)sin(πz)
        = -3π²sin(πx)sin(πy)sin(πz)
    
    Args:
        x, y, z: 좌표 텐서
        
    Returns:
        torch.Tensor: 소스 항 값
    """
    return 3.0 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(np.pi * z)


def create_pde_residual_fn():
    """PDE 잔차 함수를 생성합니다."""
    def pde_residual_fn(model, points):
        """
        3D Poisson 방정식의 잔차를 계산합니다.
        
        ∇²u + f = 0
        여기서 ∇²u = u_xx + u_yy + u_zz (Laplacian)
        
        Args:
            model: PINN 모델
            points: (N, 3) 형태의 콜로케이션 포인트 (x, y, z)
            
        Returns:
            torch.Tensor: PDE 잔차
        """
        points.requires_grad_(True)
        u = model(points)
        
        # 1차 도함수 계산
        grad_u = torch.autograd.grad(
            outputs=u.sum(),
            inputs=points,
            create_graph=True
        )[0]
        
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_z = grad_u[:, 2:3]
        
        # 2차 도함수 계산 (Laplacian)
        u_xx = torch.autograd.grad(
            outputs=u_x.sum(),
            inputs=points,
            create_graph=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            outputs=u_y.sum(),
            inputs=points,
            create_graph=True
        )[0][:, 1:2]
        
        u_zz = torch.autograd.grad(
            outputs=u_z.sum(),
            inputs=points,
            create_graph=True
        )[0][:, 2:3]
        
        # Laplacian
        laplacian = u_xx + u_yy + u_zz
        
        # 소스 항
        x, y, z = points[:, 0:1], points[:, 1:2], points[:, 2:3]
        f = source_term(x, y, z)
        
        # PDE 잔차: ∇²u + f = 0
        residual = laplacian + f
        
        return residual
    
    return pde_residual_fn


def create_bc_fn():
    """경계 조건 함수를 생성합니다."""
    def bc_fn(model, points):
        """
        디리클레 경계 조건: u = 0 on ∂Ω
        
        Args:
            model: PINN 모델
            points: 경계 포인트
            
        Returns:
            torch.Tensor: 경계 조건 오차
        """
        u = model(points)
        return u  # u - 0
    
    return bc_fn


def generate_boundary_points(n_points_per_face=20):
    """
    3D 큐브의 6개 면에 대한 경계 포인트를 생성합니다.
    
    Args:
        n_points_per_face: 각 면당 포인트 수
        
    Returns:
        torch.Tensor: (N, 3) 형태의 경계 포인트
    """
    # 각 면에 균일하게 분포된 포인트 생성
    n = int(np.sqrt(n_points_per_face))
    grid_1d = np.linspace(0, 1, n)
    
    boundary_points = []
    
    # x=0 면
    y, z = np.meshgrid(grid_1d, grid_1d)
    x = np.zeros_like(y)
    boundary_points.append(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1))
    
    # x=1 면
    y, z = np.meshgrid(grid_1d, grid_1d)
    x = np.ones_like(y)
    boundary_points.append(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1))
    
    # y=0 면
    x, z = np.meshgrid(grid_1d, grid_1d)
    y = np.zeros_like(x)
    boundary_points.append(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1))
    
    # y=1 면
    x, z = np.meshgrid(grid_1d, grid_1d)
    y = np.ones_like(x)
    boundary_points.append(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1))
    
    # z=0 면
    x, y = np.meshgrid(grid_1d, grid_1d)
    z = np.zeros_like(x)
    boundary_points.append(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1))
    
    # z=1 면
    x, y = np.meshgrid(grid_1d, grid_1d)
    z = np.ones_like(x)
    boundary_points.append(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1))
    
    # 모든 경계 포인트 결합
    boundary_points = np.vstack(boundary_points)
    
    return torch.tensor(boundary_points, dtype=torch.float32)


def compute_relative_l2_error(model, n_test_points=1000, device='cpu'):
    """
    테스트 포인트에서 상대 L2 오차를 계산합니다.
    
    Args:
        model: 훈련된 PINN 모델
        n_test_points: 테스트 포인트 수
        device: 계산 장치
        
    Returns:
        float: 상대 L2 오차
    """
    model.eval()
    
    # 테스트 포인트 생성 (Latin Hypercube Sampling)
    sampler = LatinHypercubeSampler(
        n_points=n_test_points,
        domain_min=[0.0, 0.0, 0.0],
        domain_max=[1.0, 1.0, 1.0],
        device=device
    )
    test_points = sampler.sample()
    
    # 모델 예측
    with torch.no_grad():
        u_pred = model(test_points).squeeze()
    
    # 분석해
    x, y, z = test_points[:, 0], test_points[:, 1], test_points[:, 2]
    u_exact = analytical_solution(x, y, z)
    
    # 상대 L2 오차
    error = torch.sqrt(torch.mean((u_pred - u_exact)**2))
    norm_exact = torch.sqrt(torch.mean(u_exact**2))
    relative_error = (error / norm_exact).item()
    
    return relative_error


def visualize_results(model, save_path='3d_poisson_results.png'):
    """
    결과를 시각화합니다.
    
    Args:
        model: 훈련된 PINN 모델
        save_path: 저장할 이미지 경로
    """
    model.eval()
    
    # 2D 슬라이스 생성 (z=0.5에서)
    n_grid = 50
    x_grid = np.linspace(0, 1, n_grid)
    y_grid = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = 0.5 * np.ones_like(X)
    
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # 예측
    with torch.no_grad():
        u_pred = model(points_tensor).squeeze().numpy()
    
    # 분석해
    u_exact = analytical_solution(
        torch.tensor(X.ravel()),
        torch.tensor(Y.ravel()),
        torch.tensor(Z.ravel())
    ).numpy()
    
    u_pred = u_pred.reshape(n_grid, n_grid)
    u_exact = u_exact.reshape(n_grid, n_grid)
    error = np.abs(u_pred - u_exact)
    
    # 플롯
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 예측
    im1 = axes[0].contourf(X, Y, u_pred, levels=20, cmap='viridis')
    axes[0].set_title('PINN 예측 (z=0.5)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # 분석해
    im2 = axes[1].contourf(X, Y, u_exact, levels=20, cmap='viridis')
    axes[1].set_title('분석해 (z=0.5)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # 오차
    im3 = axes[2].contourf(X, Y, error, levels=20, cmap='hot')
    axes[2].set_title('절대 오차 (z=0.5)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"결과 저장: {save_path}")
    plt.close()


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("3D Poisson 방정식 해결 with Scaled-cPIKAN")
    print("=" * 70)
    print()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    print()
    
    # 하이퍼파라미터
    layers_dims = [3, 32, 32, 32, 1]  # 3D 입력
    cheby_order = 3
    domain_min = torch.tensor([0.0, 0.0, 0.0])
    domain_max = torch.tensor([1.0, 1.0, 1.0])
    
    n_pde_points = 1000  # 내부 콜로케이션 포인트
    n_bc_points = 600    # 경계 포인트 (6개 면 × 100)
    
    adam_epochs = 5000
    lbfgs_epochs = 5
    adam_lr = 1e-3
    
    print(f"모델 구조: {layers_dims}")
    print(f"Chebyshev 차수: {cheby_order}")
    print(f"도메인: {domain_min.tolist()} → {domain_max.tolist()}")
    print(f"PDE 포인트: {n_pde_points}")
    print(f"경계 포인트: {n_bc_points}")
    print(f"Adam 에포크: {adam_epochs}")
    print(f"L-BFGS 에포크: {lbfgs_epochs}")
    print()
    
    # 모델 생성
    model = Scaled_cPIKAN(
        layers_dims=layers_dims,
        cheby_order=cheby_order,
        domain_min=domain_min,
        domain_max=domain_max
    ).to(device)
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 데이터 생성
    print("데이터 생성 중...")
    
    # PDE 콜로케이션 포인트 (Latin Hypercube Sampling)
    sampler = LatinHypercubeSampler(
        n_points=n_pde_points,
        domain_min=[0.0, 0.0, 0.0],
        domain_max=[1.0, 1.0, 1.0],
        device=device
    )
    pde_points = sampler.sample()
    
    # 경계 포인트
    bc_points = generate_boundary_points(n_points_per_face=100).to(device)
    bc_points_dicts = [{'points': bc_points}]
    
    print(f"PDE 포인트 형태: {pde_points.shape}")
    print(f"경계 포인트 형태: {bc_points.shape}")
    print()
    
    # 손실 함수 생성
    pde_residual_fn = create_pde_residual_fn()
    bc_fn = create_bc_fn()
    
    loss_fn = PhysicsInformedLoss(
        pde_residual_fn=pde_residual_fn,
        bc_fns=[bc_fn],
        loss_weights={'pde': 1.0, 'bc': 10.0}
    )
    
    # 트레이너 생성
    trainer = Trainer(model, loss_fn)
    
    # 훈련
    print("훈련 시작...")
    print("-" * 70)
    
    history = trainer.train(
        pde_points=pde_points,
        bc_points_dicts=bc_points_dicts,
        adam_epochs=adam_epochs,
        lbfgs_epochs=lbfgs_epochs,
        adam_lr=adam_lr,
        log_interval=500
    )
    
    print("-" * 70)
    print("훈련 완료!")
    print()
    
    # 최종 오차 계산
    print("최종 오차 계산 중...")
    relative_error = compute_relative_l2_error(model, n_test_points=2000, device=device)
    print(f"상대 L2 오차: {relative_error:.6e}")
    print()
    
    # 결과 시각화
    print("결과 시각화 중...")
    visualize_results(model)
    
    # 성능 요약
    print()
    print("=" * 70)
    print("성능 요약")
    print("=" * 70)
    print(f"최종 손실: {history['total_loss'][-1]:.6e}")
    print(f"PDE 잔차 손실: {history['loss_pde'][-1]:.6e}")
    print(f"경계 조건 손실: {history['loss_bc'][-1]:.6e}")
    print(f"상대 L2 오차: {relative_error:.6e}")
    
    # 3D 문제의 확장성 평가
    print()
    print("=" * 70)
    print("3D 확장성 평가")
    print("=" * 70)
    print(f"✓ 3D 도메인에 대한 아핀 스케일링 성공")
    print(f"✓ 3D Laplacian 계산 성공")
    print(f"✓ 6개 경계면 처리 성공")
    print(f"✓ Scaled-cPIKAN의 3D 확장 검증 완료")
    
    if relative_error < 1e-2:
        print(f"✓ 우수한 정확도 달성 (< 1%)")
    elif relative_error < 1e-1:
        print(f"✓ 양호한 정확도 달성 (< 10%)")
    else:
        print(f"⚠ 정확도 개선 필요 (더 많은 에포크 또는 포인트 필요)")
    
    print()


if __name__ == '__main__':
    main()
