"""
Example: 3D Height Reconstruction from Bucket Images using a Scaled-cPIKAN PINN.

This script demonstrates how to solve a physics-informed inverse problem
using the Scaled-cPIKAN model. The goal is to reconstruct a 3D height map, h(x, y),
from a set of raw "bucket" intensity images from four different laser wavelengths.

This approach is more direct than using pre-calculated wrapped phase maps.

The physics-informed loss function enforces two main constraints:
1.  Data Fidelity: The reconstructed height h(x, y), when used to simulate
    the intensity images, must match the observed bucket images.
2.  Smoothness Prior: The reconstructed surface is assumed to be smooth. This is
    enforced by penalizing the Laplacian of the height map.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collections import defaultdict
from src.models import Scaled_cPIKAN
from reconstruction.data_generator import (
    DEFAULT_WAVELENGTHS,
    generate_synthetic_data,
)

def main():
    """Main function to set up and run the 3D reconstruction from buckets experiment."""

    parser = argparse.ArgumentParser(
        description="3D reconstruction from bucket images using Scaled-cPIKAN"
    )
    parser.add_argument(
        "--num-lasers",
        type=int,
        default=None,
        help="Number of lasers. Defaults to len(wavelengths).",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=3,
        help="Number of bucket images per laser.",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=DEFAULT_WAVELENGTHS,
        help="Wavelength for each laser.",
    )
    args = parser.parse_args()
    if args.num_lasers is None:
        args.num_lasers = len(args.wavelengths)
    elif args.num_lasers != len(args.wavelengths):
        raise ValueError(
            "num_lasers must match the number of provided wavelengths"
        )

    print("--- Starting 3D Reconstruction from Buckets with Scaled-cPIKAN PINN ---")

    # 1. --- Configuration ---
    output_dir = "reconstruction_from_buckets_results"
    os.makedirs(output_dir, exist_ok=True)

    grid_shape = (128, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. --- Data Generation ---
    print("\nStep 1: Generating synthetic data...")
    ground_truth_height, _, bucket_images = generate_synthetic_data(
        shape=grid_shape,
        wavelengths=args.wavelengths,
        num_buckets=args.num_buckets,
        save_path=None,  # Don't save files for this script
    )

    # Convert numpy arrays to torch tensors
    ground_truth_height_t = torch.from_numpy(ground_truth_height).float().to(device)
    bucket_images_t = torch.from_numpy(bucket_images).float().to(device)

    # Plot and save the ground truth and a sample of bucket images
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Input Data for Bucket-Based PINN Training", fontsize=16)

    im = axs[0, 0].imshow(ground_truth_height, cmap='viridis')
    axs[0, 0].set_title("Ground Truth Height")
    fig.colorbar(im, ax=axs[0, 0])

    # Show the 3 buckets for the first laser
    for i in range(3):
        im = axs[0, i+1 if i < 2 else 2].imshow(bucket_images[0, i], cmap='gray')
        axs[0, i+1 if i < 2 else 2].set_title(f"Laser 1, Bucket {i+1}")
        fig.colorbar(im, ax=axs[0, i+1 if i < 2 else 2])

    # Show one bucket image for each of the other lasers
    for i in range(1, 4):
        im = axs[1, i-1].imshow(bucket_images[i, 0], cmap='gray')
        axs[1, i-1].set_title(f"Laser {i+1}, Bucket 1")
        fig.colorbar(im, ax=axs[1, i-1])


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "01_input_bucket_data.png"))
    print(f"Saved input data visualization to '{os.path.join(output_dir, '01_input_bucket_data.png')}'")


    # 3. --- Define the Physics-Informed Loss Function ---
    class ReconstructionLossFromBuckets(torch.nn.Module):
        def __init__(self, wavelengths, smoothness_weight=1e-4):
            super().__init__()
            self.register_buffer("wavelengths", torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1))
            # Phase shifts for the 3 buckets
            self.register_buffer("deltas", torch.tensor([0, 2 * np.pi / 3, 4 * np.pi / 3], dtype=torch.float32).view(1, 3, 1))
            self.smoothness_weight = smoothness_weight
            self.mse_loss = torch.nn.MSELoss()
            self.metrics = {}

        def forward(self, model_outputs, coords, targets):
            predicted_height = model_outputs

            # --- 1. Data Fidelity Loss ---
            # Enforce that the predicted height, when used to form bucket images,
            # matches the target bucket images.

            # Predicted phase: shape [1, 1, H*W] -> [N_lasers, 1, H*W]
            predicted_phase = (4 * np.pi / self.wavelengths) * predicted_height

            # Add phase shifts for buckets: [N_lasers, 1, H*W] -> [N_lasers, N_buckets, H*W]
            phase_with_shifts = predicted_phase + self.deltas

            # Simulate bucket images
            # These are fixed parameters from the data generator
            A = 128
            B = 100
            predicted_buckets = A + B * torch.cos(phase_with_shifts)
            predicted_buckets = predicted_buckets.view(predicted_buckets.shape[0], predicted_buckets.shape[1], -1)

            # targets are the bucket_images_t, shape [N_lasers, N_buckets, H*W]
            loss_data = self.mse_loss(predicted_buckets, targets)

            # --- 2. Smoothness Regularization Loss ---
            h = predicted_height.squeeze(0).squeeze(0) # Shape from [1,1,H*W] to [H*W]
            grad_h = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]
            h_x, h_y = grad_h[:, 0], grad_h[:, 1]
            h_xx = torch.autograd.grad(h_x.sum(), coords, create_graph=True)[0][:, 0]
            h_yy = torch.autograd.grad(h_y.sum(), coords, create_graph=True)[0][:, 1]
            laplacian = h_xx + h_yy
            loss_smoothness = self.mse_loss(laplacian, torch.zeros_like(laplacian))

            # --- Total Loss ---
            total_loss = loss_data + self.smoothness_weight * loss_smoothness

            self.metrics = {
                'loss_total': total_loss.item(),
                'loss_data': loss_data.item(),
                'loss_smoothness': loss_smoothness.item()
            }
            return total_loss

    # 4. --- Model Initialization ---
    print("\nStep 2: Initializing the Scaled-cPIKAN model...")
    domain_min = torch.tensor([0.0, 0.0], device=device)
    domain_max = torch.tensor([1.0, 1.0], device=device)
    layers_dims = [2, 64, 64, 64, 1]
    model = Scaled_cPIKAN(
        layers_dims=layers_dims,
        cheby_order=8,
        domain_min=domain_min,
        domain_max=domain_max
    ).to(device)
    print(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count}")

    # 5. --- Training ---
    print("\nStep 3: Setting up and running the training...")
    x = torch.linspace(0, 1, grid_shape[0], device=device)
    y = torch.linspace(0, 1, grid_shape[1], device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    coords.requires_grad_(True)

    # Reshape targets (bucket images) to match flattened coordinates
    # Shape: [num_lasers, num_buckets, H, W] -> [num_lasers, num_buckets, H*W]
    num_lasers = args.num_lasers
    num_buckets = args.num_buckets

    targets = bucket_images_t.view(num_lasers, num_buckets, -1)

    # Instantiate the loss function
    loss_fn = ReconstructionLossFromBuckets(
        wavelengths=args.wavelengths,
        smoothness_weight=1e-7,  # NOTE: This weight may need significant tuning
    )

    # --- Adam Optimization ---
    adam_epochs = 10000
    adam_lr = 1e-3
    log_frequency = 500
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    loss_history = defaultdict(list)

    print(f"\nTraining with Adam for {adam_epochs} epochs (lr={adam_lr})...")
    for epoch in range(adam_epochs):
        model.train()
        optimizer_adam.zero_grad()

        # Predicted height shape: [1, H*W] -> [1, 1, H*W] for broadcasting
        predicted_height = model(coords).view(1, 1, -1)
        total_loss = loss_fn(predicted_height, coords, targets)

        total_loss.backward()
        optimizer_adam.step()

        for key, value in loss_fn.metrics.items():
            loss_history[key].append(value)

        if (epoch + 1) % log_frequency == 0 or epoch == adam_epochs - 1:
            log_str = f"[Adam] Epoch [{epoch+1}/{adam_epochs}]"
            for key, value in loss_fn.metrics.items():
                log_str += f" - {key}: {value:.4e}"
            print(log_str)

    # --- L-BFGS Optimization ---
    lbfgs_epochs = 1
    print(f"\nFine-tuning with L-BFGS for {lbfgs_epochs} step(s)...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), max_iter=50, history_size=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        predicted_height = model(coords).view(1, 1, -1)
        total_loss = loss_fn(predicted_height, coords, targets)
        total_loss.backward()
        for key, value in loss_fn.metrics.items():
            loss_history[key].append(value)
        return total_loss

    for i in range(lbfgs_epochs):
        print(f"[L-BFGS] Step {i+1}/{lbfgs_epochs}")
        optimizer_lbfgs.step(closure)
        log_str = f"  Completed L-BFGS step."
        for key, value in loss_fn.metrics.items():
            log_str += f" - {key}: {value:.4e}"
        print(log_str)

    print("\n--- Training Complete ---")
    print("\nStep 4: Evaluating model and visualizing results...")
    model.eval()
    with torch.no_grad():
        predicted_height_t = model(coords).view(grid_shape).cpu()

    predicted_height = predicted_height_t.numpy()
    rmse = np.sqrt(np.mean((predicted_height - ground_truth_height)**2))
    print(f"\nReconstruction complete. Final RMSE: {rmse:.6f}")

    # Plotting the results
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("PINN 3D Reconstruction from Buckets Results", fontsize=16)
    im1 = axes[0].imshow(ground_truth_height, cmap='viridis')
    axes[0].set_title("Ground Truth Height")
    fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.2)
    im2 = axes[1].imshow(predicted_height, cmap='viridis')
    axes[1].set_title("PINN Reconstructed Height")
    fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.2)
    error = np.abs(predicted_height - ground_truth_height)
    im3 = axes[2].imshow(error, cmap='hot')
    axes[2].set_title(f"Absolute Error (RMSE: {rmse:.4f})")
    fig.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.2)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path = os.path.join(output_dir, "02_reconstruction_results.png")
    plt.savefig(output_path)
    print(f"Saved final result visualization to '{output_path}'")

    # Plot loss history
    fig, ax = plt.subplots(figsize=(12, 7))
    total_steps = len(loss_history['loss_total'])
    ax.plot(range(total_steps), loss_history['loss_total'], label='Total Loss')
    ax.plot(range(total_steps), loss_history['loss_data'], label='Data Fidelity Loss', linestyle='--')
    ax.plot(range(total_steps), loss_history['loss_smoothness'], label='Smoothness Loss', linestyle='--')
    ax.set_yscale('log')
    ax.set_title('Training Loss History')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    output_path = os.path.join(output_dir, "03_loss_history.png")
    plt.savefig(output_path)
    print(f"Saved loss history plot to '{output_path}'")

    print("\n--- 3D Reconstruction from Buckets Complete ---")

if __name__ == "__main__":
    main()
