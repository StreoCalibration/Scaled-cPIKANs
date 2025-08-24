"""
Example: 3D Height Reconstruction using a Scaled-cPIKAN PINN.

This script demonstrates how to solve a physics-informed inverse problem
using the Scaled-cPIKAN model. The goal is to reconstruct a 3D height map, h(x, y),
from a set of four wrapped phase measurements, each from a different laser wavelength.

This problem is representative of multi-wavelength phase-shifting profilometry.

The physics-informed loss function enforces two main constraints:
1.  Data Fidelity: The reconstructed height h(x, y) must be consistent with the
    four observed wrapped phase maps. This is enforced using a differentiable
    cos/sin loss formulation.
2.  Smoothness Prior: The reconstructed surface is assumed to be smooth. This is
    enforced by penalizing the Laplacian of the height map, encouraging a
    physically plausible solution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the Python path
# This allows us to import modules from the 'src' and 'reconstruction' directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collections import defaultdict

from src.models import Scaled_cPIKAN
# from src.train import Trainer # Not using the generic trainer for this problem
# from src.loss import PhysicsInformedLoss # Not using the generic loss for this problem
from reconstruction.data_generator import DEFAULT_WAVELENGTHS, generate_synthetic_data

def main():
    """
    Main function to set up and run the 3D reconstruction experiment.
    """
    print("--- Starting 3D Reconstruction with Scaled-cPIKAN PINN ---")

    # 1. --- Configuration ---
    # Define parameters for data generation and model training
    output_dir = "reconstruction_pinn_results"
    os.makedirs(output_dir, exist_ok=True)

    grid_shape = (128, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wavelengths = DEFAULT_WAVELENGTHS

    # 2. --- Data Generation ---
    # Generate synthetic data for the problem
    print("\nStep 1: Generating synthetic data...")
    ground_truth_height, wrapped_phases = generate_synthetic_data(
        shape=grid_shape,
        wavelengths=wavelengths,
        save_path=None,  # Don't save individual files for this script
    )

    # Convert numpy arrays to torch tensors
    ground_truth_height_t = torch.from_numpy(ground_truth_height).float().to(device)
    wrapped_phases_t = torch.from_numpy(np.array(wrapped_phases)).float().to(device)

    # Plot and save the ground truth and phase maps for inspection
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Generated Input Data for PINN Training", fontsize=16)

    im = axs[0, 0].imshow(ground_truth_height, cmap='viridis')
    axs[0, 0].set_title("Ground Truth Height")
    fig.colorbar(im, ax=axs[0, 0])

    for i in range(4):
        row, col = (i + 1) // 3, (i + 1) % 3
        im = axs[row, col].imshow(wrapped_phases[i], cmap='twilight_shifted')
        axs[row, col].set_title(
            f"Wrapped Phase (Laser {i+1}, $\lambda={wavelengths[i]:.2f}$ um)"
        )
        fig.colorbar(im, ax=axs[row, col])

    # Hide the last subplot as it's empty
    axs[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "01_input_data.png"))
    print(f"Saved input data visualization to '{os.path.join(output_dir, '01_input_data.png')}'")


    # 3. --- Define the Physics-Informed Loss Function ---
    # The loss function is the core of the PINN. It enforces that the
    # network's output h(x, y) is consistent with the observed phase maps
    # and that the solution is smooth.

    class ReconstructionLoss(torch.nn.Module):
        def __init__(self, wavelengths, smoothness_weight=1e-4):
            super().__init__()
            self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1)
            self.smoothness_weight = smoothness_weight
            self.mse_loss = torch.nn.MSELoss()
            self.metrics = {} # To store individual losses for logging

        def forward(self, model_outputs, coords, targets):
            predicted_height = model_outputs
            self.wavelengths = self.wavelengths.to(predicted_height.device)

            # --- 1. Data Fidelity Loss ---
            # Enforce that the predicted height matches the wrapped phase measurements.
            # We use cos/sin to create a differentiable loss function.
            true_phase = (4 * np.pi / self.wavelengths) * predicted_height

            # targets are the wrapped_phases_t
            target_cos = torch.cos(targets)
            target_sin = torch.sin(targets)

            pred_cos = torch.cos(true_phase)
            pred_sin = torch.sin(true_phase)

            loss_data = self.mse_loss(pred_cos, target_cos) + self.mse_loss(pred_sin, target_sin)

            # --- 2. Smoothness Regularization Loss ---
            # Enforce that the solution is smooth by penalizing the Laplacian.
            h = predicted_height.squeeze(0) # Remove channel dim for differentiation
            grad_h = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]

            # Unpack gradients
            h_x = grad_h[:, 0]
            h_y = grad_h[:, 1]

            # Compute second derivatives (Laplacian)
            h_xx = torch.autograd.grad(h_x.sum(), coords, create_graph=True)[0][:, 0]
            h_yy = torch.autograd.grad(h_y.sum(), coords, create_graph=True)[0][:, 1]

            laplacian = h_xx + h_yy
            loss_smoothness = self.mse_loss(laplacian, torch.zeros_like(laplacian))

            # --- Total Loss ---
            total_loss = loss_data + self.smoothness_weight * loss_smoothness

            # Store individual losses for logging
            self.metrics = {
                'loss_total': total_loss.item(),
                'loss_data': loss_data.item(),
                'loss_smoothness': loss_smoothness.item()
            }

            return total_loss

    # 4. --- Model Initialization ---
    # Create the Scaled-cPIKAN model instance.
    print("\nStep 2: Initializing the Scaled-cPIKAN model...")

    # The physical domain for x and y is [0, 1]
    domain_min = torch.tensor([0.0, 0.0], device=device)
    domain_max = torch.tensor([1.0, 1.0], device=device)

    # Correctly define the network architecture in a single list
    layers_dims = [2, 64, 64, 64, 1] # Input(x,y) -> 3 hidden layers -> Output(h)

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
    # Prepare the data and run the training process.
    print("\nStep 3: Setting up and running the training...")

    # Create a grid of coordinates for training
    x = torch.linspace(0, 1, grid_shape[0], device=device)
    y = torch.linspace(0, 1, grid_shape[1], device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    coords.requires_grad_(True)

    # The model's output at these coords should map to the phase targets
    # We need to reshape the targets to match the flattened coordinates
    # Target shape: [num_lasers, height, width] -> [num_lasers, height*width] -> [num_lasers, 1, height*width] for broadcasting
    targets = wrapped_phases_t.view(4, -1).unsqueeze(1)

    # Instantiate the loss function
    loss_fn = ReconstructionLoss(
        wavelengths=wavelengths,
        smoothness_weight=1e-5,  # This is a key hyperparameter to tune
    )

    # 5. --- Training ---
    # We will implement a custom training loop here instead of using the generic
    # Trainer class, because our loss function has a different signature.
    print("\nStep 3: Setting up and running the training...")

    # Create a grid of coordinates for training
    x = torch.linspace(0, 1, grid_shape[0], device=device)
    y = torch.linspace(0, 1, grid_shape[1], device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    coords.requires_grad_(True)

    # The model's output at these coords should map to the phase targets
    # Reshape targets to match flattened coordinates for loss calculation
    targets = wrapped_phases_t.view(4, 1, -1) # Shape: [4, 1, H*W]

    # Instantiate the loss function
    loss_fn = ReconstructionLoss(
        wavelengths=wavelengths,
        smoothness_weight=1e-5,  # This is a key hyperparameter to tune
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

        predicted_height = model(coords).view(1, -1) # Shape: [1, H*W]
        total_loss = loss_fn(predicted_height, coords, targets)

        total_loss.backward()
        optimizer_adam.step()

        # Log history
        for key, value in loss_fn.metrics.items():
            loss_history[key].append(value)

        if (epoch + 1) % log_frequency == 0 or epoch == adam_epochs - 1:
            log_str = f"[Adam] Epoch [{epoch+1}/{adam_epochs}]"
            for key, value in loss_fn.metrics.items():
                log_str += f" - {key}: {value:.4e}"
            print(log_str)

    # --- L-BFGS Optimization ---
    lbfgs_epochs = 1 # L-BFGS is often run for just a few "epochs" or steps
    print(f"\nFine-tuning with L-BFGS for {lbfgs_epochs} step(s)...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=50,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        predicted_height = model(coords).view(1, -1)
        total_loss = loss_fn(predicted_height, coords, targets)
        total_loss.backward()
        # Log metrics from the closure
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

    # Convert back to numpy for analysis and plotting
    predicted_height = predicted_height_t.numpy()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((predicted_height - ground_truth_height)**2))
    print(f"\nReconstruction complete. Final RMSE: {rmse:.6f}")

    # Plotting the results
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("PINN 3D Reconstruction Results", fontsize=16)

    # Plot Ground Truth
    im1 = axes[0].imshow(ground_truth_height, cmap='viridis')
    axes[0].set_title("Ground Truth Height")
    fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.2)

    # Plot PINN Reconstruction
    im2 = axes[1].imshow(predicted_height, cmap='viridis')
    axes[1].set_title("PINN Reconstructed Height")
    fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.2)

    # Plot Error Map
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
    adam_epochs_len = len(loss_history['loss_total'])
    ax.plot(range(adam_epochs_len), loss_history['loss_total'], label='Total Loss')
    ax.plot(range(adam_epochs_len), loss_history['loss_data'], label='Data Fidelity Loss', linestyle='--')
    ax.plot(range(adam_epochs_len), loss_history['loss_smoothness'], label='Smoothness Loss', linestyle='--')

    ax.set_yscale('log')
    ax.set_title('Training Loss History')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    output_path = os.path.join(output_dir, "03_loss_history.png")
    plt.savefig(output_path)
    print(f"Saved loss history plot to '{output_path}'")

    print("\n--- 3D Reconstruction with PINN Complete ---")

if __name__ == "__main__":
    main()
