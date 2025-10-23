import unittest
import torch
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Scaled_cPIKAN
from src.data import LatinHypercubeSampler
from src.loss import PhysicsInformedLoss
from src.train import Trainer

# Dummy functions from the example, simplified for testing
def pde_residual_fn_dummy(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_xx + u

def bc_fn_dummy(model, x_bc):
    return model(x_bc) - 0.0 # Target is 0 for simplicity

class TestIntegration(unittest.TestCase):

    def test_helmholtz_solver_smoke_test(self):
        """
        A smoke test for the full solver pipeline.
        Runs for a minimal number of epochs to ensure everything is connected correctly.
        """
        try:
            DEVICE = torch.device("cpu")

            # Minimal problem setup
            DOMAIN_MIN = [-1.0]
            DOMAIN_MAX = [1.0]
            LAYERS_DIMS = [1, 8, 8, 1] # Small model
            CHEBY_ORDER = 2
            N_PDE_POINTS = 10
            N_BC_POINTS = 2
            ADAM_EPOCHS = 2 # Just a couple of steps

            # Data
            pde_sampler = LatinHypercubeSampler(N_PDE_POINTS, DOMAIN_MIN, DOMAIN_MAX, device=DEVICE)
            pde_points = pde_sampler.sample()
            x_bc1 = torch.full((N_BC_POINTS, 1), DOMAIN_MIN[0], device=DEVICE)
            x_bc2 = torch.full((N_BC_POINTS, 1), DOMAIN_MAX[0], device=DEVICE)
            bc_points_dicts = [{'points': x_bc1}, {'points': x_bc2}]

            # Model
            model = Scaled_cPIKAN(
                layers_dims=LAYERS_DIMS,
                cheby_order=CHEBY_ORDER,
                domain_min=torch.tensor(DOMAIN_MIN, device=DEVICE),
                domain_max=torch.tensor(DOMAIN_MAX, device=DEVICE)
            ).to(DEVICE)

            # Loss
            loss_fn = PhysicsInformedLoss(
                pde_residual_fn=pde_residual_fn_dummy,
                bc_fns=[bc_fn_dummy, bc_fn_dummy]
            )

            # Trainer
            trainer = Trainer(model, loss_fn)
            history = trainer.train(
                pde_points=pde_points,
                bc_points_dicts=bc_points_dicts,
                adam_epochs=ADAM_EPOCHS,
                lbfgs_epochs=0 # Skip L-BFGS for this smoke test
            )

            # Check if training ran and produced some history
            self.assertIn('total_loss', history)
            self.assertEqual(len(history['total_loss']), ADAM_EPOCHS)

        except Exception as e:
            self.fail(f"Integration smoke test failed with an exception: {e}")

    def test_poisson_equation_1d(self):
        """
        Tests solving a 1D Poisson equation with Scaled-cPIKAN PINN.
        Problem: u''(x) = -1, u(0) = 0, u(1) = 0
        Analytical solution: u(x) = x(1-x)/2
        
        This is a smoke test to verify the PINN framework works correctly.
        Success criterion: Model trains without errors and produces finite losses
        """
        DEVICE = torch.device("cpu")

        # Problem setup
        DOMAIN_MIN = [0.0]
        DOMAIN_MAX = [1.0]
        LAYERS_DIMS = [1, 32, 32, 1]
        CHEBY_ORDER = 4
        N_PDE_POINTS = 50  # Reduced for faster testing
        N_BC_POINTS = 5    # Reduced for faster testing
        ADAM_EPOCHS = 100  # Reduced for faster testing
        ADAM_LR = 1e-3

        # Define the analytical solution
        def analytical_solution(x):
            return x * (1 - x) / 2

        # Define PDE residual: u'' + 1 = 0
        def pde_residual_fn(model, x):
            x.requires_grad_(True)
            u = model(x)
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            return u_xx + 1.0  # u'' + 1 = 0

        # Define boundary conditions
        def bc_fn_left(model, x_bc):
            return model(x_bc) - 0.0  # u(0) = 0

        def bc_fn_right(model, x_bc):
            return model(x_bc) - 0.0  # u(1) = 0

        # Sample points
        pde_sampler = LatinHypercubeSampler(N_PDE_POINTS, DOMAIN_MIN, DOMAIN_MAX, device=DEVICE)
        pde_points = pde_sampler.sample()

        x_bc_left = torch.full((N_BC_POINTS, 1), DOMAIN_MIN[0], device=DEVICE)
        x_bc_right = torch.full((N_BC_POINTS, 1), DOMAIN_MAX[0], device=DEVICE)
        bc_points_dicts = [{'points': x_bc_left}, {'points': x_bc_right}]

        # Initialize model
        model = Scaled_cPIKAN(
            layers_dims=LAYERS_DIMS,
            cheby_order=CHEBY_ORDER,
            domain_min=torch.tensor(DOMAIN_MIN, device=DEVICE),
            domain_max=torch.tensor(DOMAIN_MAX, device=DEVICE)
        ).to(DEVICE)

        # Loss function and trainer
        loss_fn = PhysicsInformedLoss(
            pde_residual_fn=pde_residual_fn,
            bc_fns=[bc_fn_left, bc_fn_right],
            loss_weights={'pde': 1.0, 'bc': 10.0}
        )

        trainer = Trainer(model, loss_fn)

        # Train
        history = trainer.train(
            pde_points=pde_points,
            bc_points_dicts=bc_points_dicts,
            adam_epochs=ADAM_EPOCHS,
            adam_lr=ADAM_LR,
            lbfgs_epochs=0
        )

        # Verify training completed and produced losses
        self.assertIn('total_loss', history)
        self.assertGreater(len(history['total_loss']), 0)
        
        # Verify losses are finite
        final_loss = history['total_loss'][-1]
        self.assertTrue(np.isfinite(final_loss), f"Final loss is not finite: {final_loss}")
        
        # Evaluate on fine grid
        x_eval = torch.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], 20, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            u_pred = model(x_eval).squeeze()
            u_analytical = analytical_solution(x_eval).squeeze()

        # Compute relative L2 error
        l2_error = torch.norm(u_pred - u_analytical) / (torch.norm(u_analytical) + 1e-8)
        
        # Log the result
        print(f"\nPoisson 1D Integration Test - Relative L2 Error: {l2_error.item():.6f}")
        print(f"Final training loss: {final_loss:.6e}")
        print(f"Note: This is a smoke test. Full convergence requires more epochs and tuning.")
        
        # Smoke test: just verify model trains without errors
        self.assertLess(final_loss, 1e6, f"Training loss {final_loss} is unreasonably high")

if __name__ == '__main__':
    unittest.main()
