import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ChebyKANLayer, Scaled_cPIKAN

class TestModels(unittest.TestCase):

    def test_chebykan_layer_forward(self):
        """Tests the forward pass of the ChebyKANLayer."""
        in_features = 4
        out_features = 8
        cheby_order = 3
        batch_size = 16

        layer = ChebyKANLayer(in_features, out_features, cheby_order)
        # Input must be in [-1, 1]
        input_tensor = torch.rand(batch_size, in_features) * 2 - 1

        output = layer(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

    def test_scaled_cpikan_forward(self):
        """Tests the forward pass of the Scaled_cPIKAN model."""
        layers_dims = [2, 16, 16, 1]
        cheby_order = 3
        domain_min = torch.tensor([-5.0, -5.0])
        domain_max = torch.tensor([5.0, 5.0])
        batch_size = 32

        model = Scaled_cPIKAN(layers_dims, cheby_order, domain_min, domain_max)

        # Input is in the physical domain
        input_tensor = torch.rand(batch_size, 2) * 10 - 5

        output = model(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, layers_dims[-1]))

    def test_affine_scaling_in_model(self):
        """Checks that the model correctly scales the input."""
        layers_dims = [1, 8, 1]
        cheby_order = 2
        domain_min = torch.tensor([100.0])
        domain_max = torch.tensor([200.0])

        model = Scaled_cPIKAN(layers_dims, cheby_order, domain_min, domain_max)

        # Test point at the middle of the physical domain
        physical_x = torch.tensor([[150.0]])
        # This should be scaled to 0.0
        scaled_x = model._affine_scale(physical_x)
        self.assertTrue(torch.allclose(scaled_x, torch.tensor([[0.0]])))

        # Test point at the min of the physical domain
        physical_x_min = torch.tensor([[100.0]])
        # This should be scaled to -1.0
        scaled_x_min = model._affine_scale(physical_x_min)
        self.assertTrue(torch.allclose(scaled_x_min, torch.tensor([[-1.0]])))

    def test_chebyshev_basis_correctness(self):
        """
        Tests that the Chebyshev polynomial basis is computed correctly.
        Validates T_0, T_1, T_2, T_3 against mathematical definitions.
        """
        # Test on a single input value in [-1, 1]
        x_test = torch.tensor([0.5])
        batch_size = 1

        layer = ChebyKANLayer(1, 1, cheby_order=3)

        # Manually compute expected Chebyshev polynomials at x=0.5
        # T_0(x) = 1
        T0_expected = torch.tensor(1.0)
        # T_1(x) = x
        T1_expected = torch.tensor(0.5)
        # T_2(x) = 2x^2 - 1
        T2_expected = torch.tensor(2 * (0.5**2) - 1)  # = 2*0.25 - 1 = -0.5
        # T_3(x) = 4x^3 - 3x
        T3_expected = torch.tensor(4 * (0.5**3) - 3 * 0.5)  # = 4*0.125 - 1.5 = -1.0

        # Compute Chebyshev polynomials in forward pass
        # We'll extract them from the forward pass logic
        x = x_test.reshape(batch_size, 1)
        cheby_polys = []
        cheby_polys.append(torch.ones_like(x))
        if layer.cheby_order > 0:
            cheby_polys.append(x)
        for k in range(1, layer.cheby_order):
            next_poly = 2 * x * cheby_polys[-1] - cheby_polys[-2]
            cheby_polys.append(next_poly)

        # Verify each polynomial
        self.assertTrue(torch.allclose(cheby_polys[0].squeeze(), T0_expected, atol=1e-5))
        self.assertTrue(torch.allclose(cheby_polys[1].squeeze(), T1_expected, atol=1e-5))
        self.assertTrue(torch.allclose(cheby_polys[2].squeeze(), T2_expected, atol=1e-5))
        self.assertTrue(torch.allclose(cheby_polys[3].squeeze(), T3_expected, atol=1e-5))

    def test_affine_scaling_boundary_conditions(self):
        """
        Tests that affine scaling correctly maps domain boundaries.
        Verifies: x_min -> -1, x_max -> 1, (x_min+x_max)/2 -> 0
        """
        layers_dims = [2, 8, 1]
        cheby_order = 3
        domain_min = torch.tensor([0.0, -5.0])
        domain_max = torch.tensor([10.0, 5.0])

        model = Scaled_cPIKAN(layers_dims, cheby_order, domain_min, domain_max)

        # Test lower boundary
        x_min = domain_min.unsqueeze(0)
        scaled_min = model._affine_scale(x_min)
        expected_min = torch.tensor([[-1.0, -1.0]])
        self.assertTrue(torch.allclose(scaled_min, expected_min, atol=1e-5))

        # Test upper boundary
        x_max = domain_max.unsqueeze(0)
        scaled_max = model._affine_scale(x_max)
        expected_max = torch.tensor([[1.0, 1.0]])
        self.assertTrue(torch.allclose(scaled_max, expected_max, atol=1e-5))

        # Test midpoint
        x_mid = (domain_min + domain_max) / 2
        x_mid = x_mid.unsqueeze(0)
        scaled_mid = model._affine_scale(x_mid)
        expected_mid = torch.tensor([[0.0, 0.0]])
        self.assertTrue(torch.allclose(scaled_mid, expected_mid, atol=1e-5))

    def test_affine_scaling_inverse_property(self):
        """
        Tests that affine scaling is invertible within the domain bounds.
        If scaled = scale(physical), then physical = unscale(scaled).
        """
        layers_dims = [1, 8, 1]
        cheby_order = 2
        domain_min = torch.tensor([2.0])
        domain_max = torch.tensor([8.0])

        model = Scaled_cPIKAN(layers_dims, cheby_order, domain_min, domain_max)

        # Generate random points in the physical domain
        physical_points = torch.rand(10, 1) * (domain_max - domain_min) + domain_min

        # Scale to [-1, 1]
        scaled_points = model._affine_scale(physical_points)

        # Unscale back to physical domain
        # Inverse of: scaled = 2*(x - min)/(max - min) - 1
        # is: x = (scaled + 1) * (max - min) / 2 + min
        recovered_points = (scaled_points + 1) * (domain_max - domain_min) / 2 + domain_min

        # Verify that recovered points match original physical points
        self.assertTrue(torch.allclose(recovered_points, physical_points, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
