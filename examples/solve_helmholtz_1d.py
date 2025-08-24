import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Scaled_cPIKAN
from src.data import LatinHypercubeSampler
from src.loss import PhysicsInformedLoss
from src.train import Trainer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analytical_solution(x, k):
    """Analytical solution for the 1D Helmholtz equation."""
    return torch.sin(k * x)

def define_pde_residual(k):
    """Returns a function that computes the PDE residual for u_xx + k^2*u = 0."""
    def pde_residual_fn(model, x):
        x.requires_grad_(True)
        u = model(x)

        # First derivative: du/dx
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # Second derivative: d^2u/dx^2
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        return u_xx + (k**2) * u
    return pde_residual_fn

def define_bc_funcs(k, analytical_sol_fn):
    """Returns functions for the boundary conditions."""
    # BC at x = -1
    def bc_fn1(model, x_bc1):
        u_bc1 = model(x_bc1)
        return u_bc1 - analytical_sol_fn(x_bc1, k)

    # BC at x = 1
    def bc_fn2(model, x_bc2):
        u_bc2 = model(x_bc2)
        return u_bc2 - analytical_sol_fn(x_bc2, k)

    return [bc_fn1, bc_fn2]

def main():
    # --- 1. Setup and Configuration ---
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    # Problem parameters
    K_WAVENUMBER = 4 * torch.pi
    DOMAIN_MIN = [-1.0]
    DOMAIN_MAX = [1.0]

    # Model hyperparameters
    LAYERS_DIMS = [1, 64, 64, 1]
    CHEBY_ORDER = 4

    # Training hyperparameters
    N_PDE_POINTS = 1000
    N_BC_POINTS = 100
    ADAM_EPOCHS = 2000
    LBFGS_EPOCHS = 1 # This parameter is not really used in the corrected trainer
    ADAM_LR = 1e-3
    LOSS_WEIGHTS = {'pde': 1.0, 'bc': 20.0}

    # --- 2. Create Data Samplers and Points ---
    pde_sampler = LatinHypercubeSampler(N_PDE_POINTS, DOMAIN_MIN, DOMAIN_MAX, device=DEVICE)
    pde_points = pde_sampler.sample()
    pde_points.requires_grad_(True) # Ensure gradients can be computed for PDE residual

    x_bc1 = torch.full((N_BC_POINTS, 1), DOMAIN_MIN[0], device=DEVICE)
    x_bc2 = torch.full((N_BC_POINTS, 1), DOMAIN_MAX[0], device=DEVICE)
    bc_points_dicts = [{'points': x_bc1}, {'points': x_bc2}]

    # --- 3. Instantiate Model, Loss, and Trainer ---
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

    # --- 4. Run Training ---
    history = trainer.train(
        pde_points=pde_points,
        bc_points_dicts=bc_points_dicts,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_epochs=LBFGS_EPOCHS,
        adam_lr=ADAM_LR,
        log_interval=500
    )

    # --- 5. Visualize Results ---
    model.eval()

    # Plot Loss History
    plt.figure(figsize=(10, 5))
    for key in ['loss_pde', 'loss_bc', 'total_loss']:
        if key in history:
            plt.plot(history['epoch'], history[key], label=key)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("helmholtz_loss_history.png")
    print("\nSaved loss history plot to helmholtz_loss_history.png")

    # Plot Solution
    with torch.no_grad():
        x_plot = torch.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], 500).view(-1, 1).to(DEVICE)
        u_pred = model(x_plot).cpu().numpy()
        u_true = analytical_solution(x_plot.cpu(), K_WAVENUMBER).numpy()

        # Calculate Relative L2 Error
        l2_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        print(f"Final Relative L2 Error: {l2_error:.4e}")

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot.cpu().numpy(), u_true, 'b-', label='Analytical Solution')
        plt.plot(x_plot.cpu().numpy(), u_pred, 'r--', label=f'Scaled-cPIKAN Prediction')
        plt.title(f'1D Helmholtz Solution (k={K_WAVENUMBER/torch.pi:.1f}π) - L2 Error: {l2_error:.2e}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True)
        plt.savefig("helmholtz_solution.png")
        print("Saved solution plot to helmholtz_solution.png")

if __name__ == "__main__":
    main()
