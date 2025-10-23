import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Scaled_cPIKAN
from src.data import LatinHypercubeSampler
from src.loss import PhysicsInformedLoss
from src.train import Trainer

def get_device():
    """사용 가능한 경우 CUDA 장치를, 그렇지 않은 경우 CPU를 반환합니다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analytical_solution(x, k):
    """1D 헬름홀츠 방정식의 분석적 해."""
    return torch.sin(k * x)

def define_pde_residual(k):
    """u_xx + k^2*u = 0에 대한 PDE 잔차를 계산하는 함수를 반환합니다."""
    def pde_residual_fn(model, x):
        x.requires_grad_(True)
        u = model(x)

        # 1차 미분: du/dx
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # 2차 미분: d^2u/dx^2
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        return u_xx + (k**2) * u
    return pde_residual_fn

def define_bc_funcs(k, analytical_sol_fn):
    """경계 조건을 위한 함수들을 반환합니다."""
    # x = -1에서의 경계 조건
    def bc_fn1(model, x_bc1):
        u_bc1 = model(x_bc1)
        return u_bc1 - analytical_sol_fn(x_bc1, k)

    # x = 1에서의 경계 조건
    def bc_fn2(model, x_bc2):
        u_bc2 = model(x_bc2)
        return u_bc2 - analytical_sol_fn(x_bc2, k)

    return [bc_fn1, bc_fn2]

def main():
    """
    Example: 1D Helmholtz Equation Solver
    
    Purpose: Demonstrates the Scaled-cPIKAN PINN for solving the 1D Helmholtz equation
    
    Problem: u_xx + k²u = 0, u(-1) = sin(-k), u(1) = sin(k)
             where k = 4π (high-frequency test case)
    
    Expected output: Final relative L2 error < 1e-4 (after Adam + L-BFGS)
    
    This is a benchmark test case from the paper "Scaled-cPIKANs".
    """
    # --- 1. 설정 및 구성 ---
    DEVICE = get_device()
    print(f"사용 장치: {DEVICE}")

    # 문제 매개변수
    K_WAVENUMBER = 4 * torch.pi
    DOMAIN_MIN = [-1.0]
    DOMAIN_MAX = [1.0]

    # 모델 하이퍼파라미터 (논문 권장 설정)
    LAYERS_DIMS = [1, 32, 32, 32, 1]  # 스케일링된 모델
    CHEBY_ORDER = 3

    # 훈련 하이퍼파라미터 (논문 권장 설정)
    N_PDE_POINTS = 1000
    N_BC_POINTS = 100
    ADAM_EPOCHS = 20000  # 논문 설정
    ADAM_LR = 1e-3  # 논문 설정
    LOSS_WEIGHTS = {'pde': 1.0, 'bc': 10.0}  # 균형잡힌 가중치

    print("\n" + "="*60)
    print("1D Helmholtz Equation Solver - Scaled-cPIKAN PINN")
    print("="*60)
    print(f"Wavenumber k: {K_WAVENUMBER/torch.pi:.1f}π")
    print(f"Domain: [{DOMAIN_MIN[0]}, {DOMAIN_MAX[0]}]")
    print(f"Model architecture: {LAYERS_DIMS}")
    print(f"Chebyshev order: {CHEBY_ORDER}")
    print(f"Training epochs (Adam): {ADAM_EPOCHS}")
    print(f"Learning rate (Adam): {ADAM_LR}")
    print(f"Loss weights: PDE={LOSS_WEIGHTS['pde']}, BC={LOSS_WEIGHTS['bc']}")
    print("="*60 + "\n")

    # --- 2. 데이터 샘플러 및 포인트 생성 ---
    pde_sampler = LatinHypercubeSampler(N_PDE_POINTS, DOMAIN_MIN, DOMAIN_MAX, device=DEVICE)
    pde_points = pde_sampler.sample()
    pde_points.requires_grad_(True) # PDE 잔차에 대한 그래디언트 계산 보장

    x_bc1 = torch.full((N_BC_POINTS, 1), DOMAIN_MIN[0], device=DEVICE)
    x_bc2 = torch.full((N_BC_POINTS, 1), DOMAIN_MAX[0], device=DEVICE)
    bc_points_dicts = [{'points': x_bc1}, {'points': x_bc2}]

    # --- 3. 모델, 손실 함수, 트레이너 인스턴스화 ---
    model = Scaled_cPIKAN(
        layers_dims=LAYERS_DIMS,
        cheby_order=CHEBY_ORDER,
        domain_min=torch.tensor(DOMAIN_MIN, device=DEVICE),
        domain_max=torch.tensor(DOMAIN_MAX, device=DEVICE)
    ).to(DEVICE)

    pde_residual_fn = define_pde_residual(K_WAVENUMBER)
    bc_fns = define_bc_funcs(K_WAVENUMBER, analytical_solution)

    loss_fn = PhysicsInformedLoss(
        pde_residual_fn=pde_residual_fn,
        bc_fns=bc_fns,
        loss_weights=LOSS_WEIGHTS
    )

    trainer = Trainer(model, loss_fn)

    # --- 4. 훈련 실행 ---
    print("Starting training...")
    history = trainer.train(
        pde_points=pde_points,
        bc_points_dicts=bc_points_dicts,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_epochs=0,  # Skip L-BFGS for this demo
        adam_lr=ADAM_LR,
        log_interval=5000
    )
    print("Training completed!\n")

    # --- 5. 결과 평가 및 벤치마크 비교 ---
    model.eval()

    # 손실 기록 플롯
    plt.figure(figsize=(10, 5))
    epochs = history.get('epoch', list(range(len(history['total_loss']))))
    
    for key in ['loss_pde', 'loss_bc', 'total_loss']:
        if key in history:
            plt.plot(epochs, history[key], label=key, linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('1D Helmholtz - Training Loss History', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("helmholtz_loss_history.png", dpi=150)
    print("✓ 손실 기록 플롯을 helmholtz_loss_history.png에 저장했습니다.")

    # 해답 플롯 및 오차 분석
    with torch.no_grad():
        x_plot = torch.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], 500).view(-1, 1).to(DEVICE)
        u_pred = model(x_plot).cpu().numpy()
        u_true = analytical_solution(x_plot.cpu(), K_WAVENUMBER).numpy()

        # 상대 L2 오차 계산
        l2_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        
        # 점별 오차
        pointwise_error = np.abs(u_pred - u_true)
        max_pointwise_error = np.max(pointwise_error)
        mean_pointwise_error = np.mean(pointwise_error)

        # 결과 출력
        print("="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Final relative L2 error: {l2_error:.4e}")
        print(f"Max pointwise error: {max_pointwise_error:.4e}")
        print(f"Mean pointwise error: {mean_pointwise_error:.4e}")
        print("="*60)
        print("\nTarget from paper (Helmholtz k=4π):")
        print("  - Relative L2 error < 1e-4 expected")
        
        if l2_error < 1e-4:
            print(f"✓ BENCHMARK PASSED! Error {l2_error:.4e} < 1e-4")
        else:
            print(f"⚠ Error {l2_error:.4e} did not meet 1e-4 target.")
            print("  (This may be due to limited epochs or hyperparameter tuning)")
        print("="*60 + "\n")

        # 해답 플롯
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_plot.cpu().numpy(), u_true, 'b-', label='Analytical solution', linewidth=2)
        plt.plot(x_plot.cpu().numpy(), u_pred, 'r--', label='Scaled-cPIKAN prediction', linewidth=2)
        plt.title(f'1D Helmholtz Solution (k={K_WAVENUMBER/torch.pi:.1f}π)', fontsize=12)
        plt.xlabel('x', fontsize=11)
        plt.ylabel('u(x)', fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.semilogy(x_plot.cpu().numpy(), pointwise_error + 1e-16, 'g-', linewidth=2)
        plt.title(f'Pointwise Absolute Error (L2={l2_error:.2e})', fontsize=12)
        plt.xlabel('x', fontsize=11)
        plt.ylabel('|u_pred - u_analytical|', fontsize=11)
        plt.grid(True, alpha=0.3, which="both")
        
        plt.tight_layout()
        plt.savefig("helmholtz_solution.png", dpi=150)
        print("✓ 해답 및 오차 플롯을 helmholtz_solution.png에 저장했습니다.\n")

if __name__ == "__main__":
    main()
