import unittest
import torch
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

if __name__ == '__main__':
    unittest.main()
