import unittest
import torch
import numpy as np
import os
import shutil
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import UNet
from src.data import WaferPatchDataset
from src.loss import UNetPhysicsLoss

class TestUNetPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory with dummy data for testing."""
        self.test_dir = "temp_test_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a dummy data sample
        sample_dir = os.path.join(self.test_dir, "sample_0000")
        os.makedirs(sample_dir, exist_ok=True)

        # Dummy data arrays (12 channels for input, 512x512)
        self.H, self.W = 512, 512
        dummy_buckets = np.random.rand(12, self.H, self.W).astype(np.float32)
        dummy_gt = np.random.rand(self.H, self.W).astype(np.float32)

        np.save(os.path.join(sample_dir, "bucket_images.npy"), dummy_buckets)
        np.save(os.path.join(sample_dir, "ground_truth.npy"), dummy_gt)

    def tearDown(self):
        """Remove the temporary directory after tests are done."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_unet_forward_pass(self):
        """Tests the forward pass of the UNet model."""
        n_channels = 12
        n_classes = 1
        batch_size = 2
        patch_size = 256

        model = UNet(n_channels=n_channels, n_classes=n_classes)
        input_tensor = torch.rand(batch_size, n_channels, patch_size, patch_size)

        output = model(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, n_classes, patch_size, patch_size))

    def test_wafer_patch_dataset(self):
        """Tests the WaferPatchDataset."""
        patch_size = 256

        dataset = WaferPatchDataset(data_dir=self.test_dir, patch_size=patch_size)

        # Check dataset length
        self.assertEqual(len(dataset), 1)

        # Get one sample
        input_tensor, target_tensor = dataset[0]

        # Check tensor shapes
        self.assertEqual(input_tensor.shape, (12, patch_size, patch_size))
        self.assertEqual(target_tensor.shape, (1, patch_size, patch_size))

        # Check tensor types
        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertIsInstance(target_tensor, torch.Tensor)

    def test_unet_physics_loss(self):
        """Tests the UNetPhysicsLoss calculation."""
        batch_size = 2
        patch_size = 128 # Use smaller patch for faster test

        # Dummy model output and input
        predicted_height = torch.rand(batch_size, 1, patch_size, patch_size)
        input_buckets = torch.rand(batch_size, 12, patch_size, patch_size)

        wavelengths = [635e-9, 525e-9, 450e-9, 405e-9]
        num_buckets = 3
        loss_fn = UNetPhysicsLoss(wavelengths=wavelengths, num_buckets=num_buckets)

        loss = loss_fn(predicted_height, input_buckets)

        # Check if loss is a scalar tensor
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.numel(), 1)

        # Check that metrics were recorded
        self.assertIn("loss_total", loss_fn.metrics)
        self.assertIn("loss_data", loss_fn.metrics)
        self.assertIn("loss_smoothness", loss_fn.metrics)

if __name__ == '__main__':
    unittest.main()
