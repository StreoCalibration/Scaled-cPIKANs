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
    print("=" * 70)
    print("1D Helmholtz 방정식 벤치마크 (논문 Table 1 재현)")
    print("=" * 70)

    # 문제 매개변수 (논문 설정)
    K_WAVENUMBER = 4 * torch.pi
    DOMAIN_MIN = [-1.0]
    DOMAIN_MAX = [1.0]

    # 모델 하이퍼파라미터 (논문 권장 설정)
    LAYERS_DIMS = [1, 32, 32, 32, 1]  # 논문 권장 구조
    CHEBY_ORDER = 3  # 논문 권장 차수

    # 훈련 하이퍼파라미터 (논문 설정)
    N_PDE_POINTS = 1000
    N_BC_POINTS = 100
    ADAM_EPOCHS = 20000  # 논문 권장: 20k epochs
    LBFGS_EPOCHS = 5     # 논문 권장: 5 L-BFGS steps
    ADAM_LR = 1e-3       # 논문 설정
    LOSS_WEIGHTS = {'pde': 1.0, 'bc': 20.0}
    
    print(f"\n문제 설정:")
    print(f"  PDE: u_xx + k²u = 0, k = {K_WAVENUMBER/torch.pi:.1f}π")
    print(f"  도메인: [{DOMAIN_MIN[0]}, {DOMAIN_MAX[0]}]")
    print(f"  경계 조건: u(-1) = sin(-k), u(1) = sin(k)")
    print(f"  분석 해: u(x) = sin(kx)")
    print(f"\n모델 구조:")
    print(f"  아키텍처: {LAYERS_DIMS}")
    print(f"  체비쇼프 차수: {CHEBY_ORDER}")
    print(f"\n훈련 설정:")
    print(f"  PDE 포인트: {N_PDE_POINTS}")
    print(f"  경계 포인트: {N_BC_POINTS}")
    print(f"  Adam 에포크: {ADAM_EPOCHS}")
    print(f"  L-BFGS 에포크: {LBFGS_EPOCHS}")
    print(f"  학습률: {ADAM_LR}")
    print(f"  손실 가중치: {LOSS_WEIGHTS}")
    print("=" * 70)

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
    print("\n훈련 시작...")
    import time
    start_time = time.time()
    
    history = trainer.train(
        pde_points=pde_points,
        bc_points_dicts=bc_points_dicts,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_epochs=LBFGS_EPOCHS,
        adam_lr=ADAM_LR,
        log_interval=2000  # 20k epochs이므로 로그 간격 증가
    )
    
    training_time = time.time() - start_time
    print(f"\n훈련 완료! 소요 시간: {training_time:.2f}초")

    # --- 5. 결과 평가 및 논문 비교 ---
    model.eval()
    
    # 정확도 평가
    with torch.no_grad():
        x_test = torch.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], 1000).view(-1, 1).to(DEVICE)
        u_pred = model(x_test).cpu().numpy().flatten()
        u_true = analytical_solution(x_test.cpu(), K_WAVENUMBER).numpy().flatten()
        
        # 다양한 오차 메트릭 계산
        l2_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        linf_error = np.max(np.abs(u_pred - u_true))
        mse = np.mean((u_pred - u_true) ** 2)
        
    print("\n" + "=" * 70)
    print("벤치마크 결과 (논문 Table 1과 비교)")
    print("=" * 70)
    print(f"Relative L2 error: {l2_error:.6e}")
    print(f"L∞ error:          {linf_error:.6e}")
    print(f"MSE:               {mse:.6e}")
    print(f"Final total loss:  {history['total_loss'][-1]:.6e}")
    print(f"Final PDE loss:    {history['loss_pde'][-1]:.6e}")
    print(f"Final BC loss:     {history['loss_bc'][-1]:.6e}")
    print(f"Training time:     {training_time:.2f}s")
    print("=" * 70)
    
    # 목표 달성 여부 확인
    target_error = 1e-4
    if l2_error < target_error:
        print(f"✓ 목표 달성! L2 error ({l2_error:.6e}) < 목표 ({target_error:.6e})")
    else:
        print(f"✗ 목표 미달성. L2 error ({l2_error:.6e}) >= 목표 ({target_error:.6e})")
    print("=" * 70)
    
    # 논문과의 비교 정보
    print("\n논문 Table 1 참조:")
    print("  - 이 벤치마크는 논문의 1D Helmholtz 문제 결과와 비교할 수 있습니다.")
    print("  - 논문에서 Scaled-cPIKAN은 L2 error < 1e-4를 달성했습니다.")
    print("  - 실험 조건: k=4π, 도메인=[-1,1], Adam 20k + L-BFGS 5 steps")
    print("=" * 70)

    # --- 6. 결과 시각화 ---
    # --- 6. 결과 시각화 ---
    print("\n시각화 생성 중...")
    
    # 손실 기록 플롯
    plt.figure(figsize=(12, 5))
    
    # 서브플롯 1: 손실 기록 (로그 스케일)
    plt.subplot(1, 2, 1)
    for key in ['loss_pde', 'loss_bc', 'total_loss']:
        if key in history:
            plt.plot(history['epoch'], history[key], label=key, linewidth=2)
    plt.yscale('log')
    plt.xlabel('에포크', fontsize=12)
    plt.ylabel('손실', fontsize=12)
    plt.title('훈련 손실 기록', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # 서브플롯 2: 학습률 변화 (스케줄러 사용 시)
    plt.subplot(1, 2, 2)
    if 'learning_rate' in history:
        plt.plot(history['epoch'], history['learning_rate'], 'g-', linewidth=2)
        plt.xlabel('에포크', fontsize=12)
        plt.ylabel('학습률', fontsize=12)
        plt.title('학습률 변화 (ExponentialLR)', fontsize=14)
        plt.grid(True, alpha=0.5)
    else:
        plt.text(0.5, 0.5, 'Learning rate not tracked', 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("helmholtz_loss_history.png", dpi=150)
    print("  손실 기록 플롯: helmholtz_loss_history.png")

    # 해답 플롯
    with torch.no_grad():
        x_plot = torch.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], 500).view(-1, 1).to(DEVICE)
        u_pred = model(x_plot).cpu().numpy().flatten()
        u_true = analytical_solution(x_plot.cpu(), K_WAVENUMBER).numpy().flatten()
        error = np.abs(u_pred - u_true)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 서브플롯 1: 해 비교
        ax1.plot(x_plot.cpu().numpy(), u_true, 'b-', label='분석적 해', linewidth=2)
        ax1.plot(x_plot.cpu().numpy(), u_pred, 'r--', label='Scaled-cPIKAN 예측', linewidth=2)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('u(x)', fontsize=12)
        ax1.set_title(f'1D 헬름홀츠 해 (k={K_WAVENUMBER/torch.pi:.1f}π)\nRelative L2 error: {l2_error:.6e}', 
                     fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 서브플롯 2: 절대 오차
        ax2.plot(x_plot.cpu().numpy(), error, 'g-', linewidth=2)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('절대 오차 |u_pred - u_true|', fontsize=12)
        ax2.set_title(f'절대 오차 분포 (최대: {linf_error:.6e})', fontsize=14)
        ax2.set_yscale('log')
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("helmholtz_solution.png", dpi=150)
        print("  해 비교 플롯: helmholtz_solution.png")
    
    print("\n완료! 결과 파일:")
    print("  - helmholtz_loss_history.png")
    print("  - helmholtz_solution.png")

if __name__ == "__main__":
    main()
