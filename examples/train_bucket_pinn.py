import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import PinnPatchDataset
from src.models import Scaled_cPIKAN


class ReconstructionLossFromBuckets(torch.nn.Module):
    """Loss computing bucket intensity consistency and smoothness."""
    def __init__(self, wavelengths, num_buckets, smoothness_weight=1e-7):
        super().__init__()
        self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1, 1)
        deltas = torch.arange(num_buckets, dtype=torch.float32) * (2 * np.pi / num_buckets)
        self.deltas = deltas.view(1, num_buckets, 1, 1)
        self.smoothness_weight = smoothness_weight
        self.mse_loss = torch.nn.MSELoss()
        self.metrics = {}

    def forward(self, predicted_height, coords, targets):
        self.wavelengths = self.wavelengths.to(predicted_height.device)
        self.deltas = self.deltas.to(predicted_height.device)

        predicted_phase = (4 * np.pi / self.wavelengths) * predicted_height
        phase_with_shifts = predicted_phase.unsqueeze(1) + self.deltas
        A, B = 128, 100
        predicted_buckets = A + B * torch.cos(phase_with_shifts)
        loss_data = self.mse_loss(predicted_buckets, targets)

        # Reshape h and coords for gradient calculation
        # predicted_height is (B, 1, 1, N), h should be (B*N)
        # coords is (B*N, 2)
        h = predicted_height.view(-1)

        # We need to compute gradients per-sample in the batch. This is tricky.
        # A simpler approach for now is to compute the laplacian on the coordinates
        # assuming a regular grid, which is true for our patches.
        # This part of the loss is an approximation and might need refinement.
        # For now, let's keep the smoothness loss calculation simple.
        # A full per-sample gradient is complex with batched inputs.
        # Let's compute an aggregate smoothness loss.
        grad_h = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]
        h_x, h_y = grad_h[:, 0], grad_h[:, 1]

        # The second derivative calculation is also tricky in batch mode.
        # The original implementation assumed a single large tensor.
        # We will compute the sum of all laplacians.
        h_xx_sum = torch.autograd.grad(h_x.sum(), coords, create_graph=True)[0][:, 0]
        h_yy_sum = torch.autograd.grad(h_y.sum(), coords, create_graph=True)[0][:, 1]
        laplacian = h_xx_sum + h_yy_sum

        loss_smoothness = self.mse_loss(laplacian, torch.zeros_like(laplacian))

        total_loss = loss_data + self.smoothness_weight * loss_smoothness
        self.metrics = {
            "loss_total": total_loss.item(),
            "loss_data": loss_data.item(),
            "loss_smoothness": loss_smoothness.item(),
        }
        return total_loss


def main():
    parser = argparse.ArgumentParser(description="Train Scaled-cPIKAN using bucket images with patching.")
    parser.add_argument("--data-dir", type=str, default="reconstruction_data",
                        help="Directory containing bucket image data.")
    parser.add_argument("--model-path", type=str, default="bucket_model.pth",
                        help="Path to save trained model weights.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for Adam.")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Size of the square patches to train on.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of patches per batch.")
    parser.add_argument("--steps-per-epoch", type=int, default=200,
                        help="Number of batches to process per epoch.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    # The dataset loads the entire image into RAM and serves random patches.
    dataset = PinnPatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        output_format='npy',
        epoch_length=args.steps_per_epoch * args.batch_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # Shuffle is implicitly handled by random sampling in dataset
        num_workers=os.cpu_count() // 2, # Use half of the available CPU cores
        pin_memory=True,
    )

    # --- Model ---
    domain_min = torch.tensor([0.0, 0.0], device=device)
    domain_max = torch.tensor([1.0, 1.0], device=device)
    model = Scaled_cPIKAN(layers_dims=[2, 64, 64, 64, 1],
                          cheby_order=8,
                          domain_min=domain_min,
                          domain_max=domain_max).to(device)

    # --- Loss and Optimizer ---
    wavelengths = np.load(os.path.join(args.data_dir, "wavelengths.npy"))
    num_buckets = dataset.bucket_images.shape[1] # Get from loaded data
    loss_fn = ReconstructionLossFromBuckets(wavelengths, num_buckets, smoothness_weight=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = defaultdict(list)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for i, (coords, bucket_targets, gt_patch) in enumerate(progress_bar):
            coords = coords.to(device).requires_grad_(True)
            # Dataloader batches items, so coords is (B, N, 2). Reshape to (B*N, 2)
            coords_flat = coords.view(-1, 2)

            # bucket_targets is (B, C, N). Reshape to (B, C, 1, N) for loss fn
            bucket_targets = bucket_targets.to(device).unsqueeze(2)

            optimizer.zero_grad()

            # predicted_height is (B*N, 1). Reshape to (B, 1, 1, N) for loss
            predicted_height = model(coords_flat).view(args.batch_size, 1, 1, -1)

            loss = loss_fn(predicted_height, coords_flat, bucket_targets)
            loss.backward()
            optimizer.step()

            for k, v in loss_fn.metrics.items():
                epoch_losses[k].append(v)

            progress_bar.set_postfix({k: f"{v:.4e}" for k, v in loss_fn.metrics.items()})

        # --- Logging epoch results ---
        log_str = f"Epoch [{epoch+1}/{args.epochs}]"
        for k, v_list in epoch_losses.items():
            log_str += f" - Avg {k}: {np.mean(v_list):.4e}"
        print(log_str)

    torch.save(model.state_dict(), args.model_path)
    print(f"\nTraining complete. Saved model weights to {args.model_path}")


if __name__ == "__main__":
    main()
