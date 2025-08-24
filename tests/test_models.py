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

if __name__ == '__main__':
    unittest.main()
