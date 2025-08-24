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
    # --- 1. 설정 및 구성 ---
    DEVICE = get_device()
    print(f"사용 장치: {DEVICE}")

    # 문제 매개변수
    K_WAVENUMBER = 4 * torch.pi
    DOMAIN_MIN = [-1.0]
    DOMAIN_MAX = [1.0]

    # 모델 하이퍼파라미터
    LAYERS_DIMS = [1, 64, 64, 1]
    CHEBY_ORDER = 4

    # 훈련 하이퍼파라미터
    N_PDE_POINTS = 1000
    N_BC_POINTS = 100
    ADAM_EPOCHS = 2000
    LBFGS_EPOCHS = 1 # 이 파라미터는 수정된 트레이너에서 실제로 사용되지 않음
    ADAM_LR = 1e-3
    LOSS_WEIGHTS = {'pde': 1.0, 'bc': 20.0}

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
    history = trainer.train(
        pde_points=pde_points,
        bc_points_dicts=bc_points_dicts,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_epochs=LBFGS_EPOCHS,
        adam_lr=ADAM_LR,
        log_interval=500
    )

    # --- 5. 결과 시각화 ---
    model.eval()

    # 손실 기록 플롯
    plt.figure(figsize=(10, 5))
    for key in ['loss_pde', 'loss_bc', 'total_loss']:
        if key in history:
            plt.plot(history['epoch'], history[key], label=key)
    plt.yscale('log')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.title('훈련 손실 기록')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("helmholtz_loss_history.png")
    print("\n손실 기록 플롯을 helmholtz_loss_history.png에 저장했습니다.")

    # 해답 플롯
    with torch.no_grad():
        x_plot = torch.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], 500).view(-1, 1).to(DEVICE)
        u_pred = model(x_plot).cpu().numpy()
        u_true = analytical_solution(x_plot.cpu(), K_WAVENUMBER).numpy()

        # 상대 L2 오차 계산
        l2_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        print(f"최종 상대 L2 오차: {l2_error:.4e}")

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot.cpu().numpy(), u_true, 'b-', label='분석적 해')
        plt.plot(x_plot.cpu().numpy(), u_pred, 'r--', label=f'Scaled-cPIKAN 예측')
        plt.title(f'1D 헬름홀츠 해 (k={K_WAVENUMBER/torch.pi:.1f}π) - L2 오차: {l2_error:.2e}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True)
        plt.savefig("helmholtz_solution.png")
        print("해답 플롯을 helmholtz_solution.png에 저장했습니다.")

if __name__ == "__main__":
    main()
