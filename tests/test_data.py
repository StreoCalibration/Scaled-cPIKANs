import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import affine_scale, affine_unscale, LatinHypercubeSampler

class TestDataUtils(unittest.TestCase):

    def test_affine_scale_unscale(self):
        """Tests that scaling and then unscaling returns the original tensor."""
        domain_min = torch.tensor([-10.0, 0.0])
        domain_max = torch.tensor([10.0, 5.0])
        original_tensor = torch.tensor([[0.0, 2.5], [-5.0, 1.0], [10.0, 5.0]])

        scaled_tensor = affine_scale(original_tensor, domain_min, domain_max)

        # Check that scaled values are in [-1, 1]
        self.assertTrue(torch.all(scaled_tensor >= -1.0))
        self.assertTrue(torch.all(scaled_tensor <= 1.0))

        # Check some specific values
        # [0.0, 2.5] should scale to [0.0, 0.0]
        self.assertTrue(torch.allclose(scaled_tensor[0], torch.tensor([0.0, 0.0])))
        # [-5.0, 1.0] should scale to [-0.5, -0.6]
        self.assertTrue(torch.allclose(scaled_tensor[1], torch.tensor([-0.5, -0.6])))

        unscaled_tensor = affine_unscale(scaled_tensor, domain_min, domain_max)

        # Check that unscaling restores the original tensor
        self.assertTrue(torch.allclose(original_tensor, unscaled_tensor))

    def test_latin_hypercube_sampler(self):
        """Tests the LatinHypercubeSampler."""
        n_points = 100
        domain_min = [-1.0, 0.0, -5.0]
        domain_max = [1.0, 10.0, 5.0]
        dimensions = len(domain_min)

        sampler = LatinHypercubeSampler(n_points, domain_min, domain_max)
        samples = sampler.sample()

        # Check shape
        self.assertEqual(samples.shape, (n_points, dimensions))

        # Check that samples are within the domain bounds
        self.assertTrue(torch.all(samples >= torch.tensor(domain_min)))
        self.assertTrue(torch.all(samples <= torch.tensor(domain_max)))

if __name__ == '__main__':
    unittest.main()
