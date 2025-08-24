import argparse
import os
from collections import defaultdict

import numpy as np
import torch

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

        h = predicted_height.squeeze(0).squeeze(0)
        grad_h = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]
        h_x, h_y = grad_h[:, 0], grad_h[:, 1]
        h_xx = torch.autograd.grad(h_x.sum(), coords, create_graph=True)[0][:, 0]
        h_yy = torch.autograd.grad(h_y.sum(), coords, create_graph=True)[0][:, 1]
        laplacian = h_xx + h_yy
        loss_smoothness = self.mse_loss(laplacian, torch.zeros_like(laplacian))

        total_loss = loss_data + self.smoothness_weight * loss_smoothness
        self.metrics = {
            "loss_total": total_loss.item(),
            "loss_data": loss_data.item(),
            "loss_smoothness": loss_smoothness.item(),
        }
        return total_loss


def main():
    parser = argparse.ArgumentParser(description="Train Scaled-cPIKAN using bucket images.")
    parser.add_argument("--data-dir", type=str, default="reconstruction_data",
                        help="Directory containing bucket_images.npy and ground_truth_height.npy")
    parser.add_argument("--model-path", type=str, default="bucket_model.pth",
                        help="Path to save trained model weights")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of Adam epochs (default: 1000)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam (default: 1e-3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bucket_images = np.load(os.path.join(args.data_dir, "bucket_images.npy"))
    wavelengths = np.load(os.path.join(args.data_dir, "wavelengths.npy"))
    num_lasers, num_buckets, H, W = bucket_images.shape
    bucket_images_t = torch.from_numpy(bucket_images).float().to(device).view(num_lasers, num_buckets, -1)

    x = torch.linspace(0, 1, H, device=device)
    y = torch.linspace(0, 1, W, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    coords.requires_grad_(True)

    domain_min = torch.tensor([0.0, 0.0], device=device)
    domain_max = torch.tensor([1.0, 1.0], device=device)
    model = Scaled_cPIKAN(layers_dims=[2, 64, 64, 64, 1],
                          cheby_order=8,
                          domain_min=domain_min,
                          domain_max=domain_max).to(device)

    loss_fn = ReconstructionLossFromBuckets(wavelengths, num_buckets)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = defaultdict(list)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        predicted_height = model(coords).view(1, 1, -1)
        loss = loss_fn(predicted_height, coords, bucket_images_t)
        loss.backward()
        optimizer.step()

        for k, v in loss_fn.metrics.items():
            history[k].append(v)
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            log_str = f"Epoch [{epoch+1}/{args.epochs}]"
            for k, v in loss_fn.metrics.items():
                log_str += f" - {k}: {v:.4e}"
            print(log_str)

    torch.save(model.state_dict(), args.model_path)
    print(f"Saved model weights to {args.model_path}")


if __name__ == "__main__":
    main()
