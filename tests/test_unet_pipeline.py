import unittest
import torch
import numpy as np
import os
import shutil
import sys
from PIL import Image

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
        self.num_channels = 12
        self.output_format = 'bmp'

        # Create a dummy data sample
        sample_dir = os.path.join(self.test_dir, "sample_0000")
        os.makedirs(sample_dir, exist_ok=True)

        # Dummy data arrays (12 channels for input, 64x64)
        self.H, self.W = 64, 64 # Use smaller images for faster tests
        dummy_buckets = np.random.randint(0, 256, size=(self.num_channels, self.H, self.W), dtype=np.uint8)
        dummy_gt = np.random.rand(self.H, self.W).astype(np.float32)

        # Save dummy buckets as individual BMP files
        for i in range(self.num_channels):
            img = Image.fromarray(dummy_buckets[i])
            img.save(os.path.join(sample_dir, f"bucket_{i:02d}.{self.output_format}"))

        # Ground truth is still saved as .npy for precision
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
        patch_size = 32 # Smaller patch size for test

        dataset = WaferPatchDataset(
            data_dir=self.test_dir,
            patch_size=patch_size,
            num_channels=self.num_channels,
            output_format=self.output_format
        )

        # Check dataset length
        self.assertEqual(len(dataset), 1)

        # Get one sample
        input_tensor, target_tensor = dataset[0]

        # Check tensor shapes
        self.assertEqual(input_tensor.shape, (self.num_channels, patch_size, patch_size))
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
