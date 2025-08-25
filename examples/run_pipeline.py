import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data import WaferPatchDataset
from src.models import UNet
from src.loss import UNetPhysicsLoss

# --- Data Generation Logic ---

def generate_spherical_bump(grid_x, grid_y, center_x, center_y, diameter, height):
    radius = diameter / 2
    distance_sq = (grid_x - center_x)**2 + (grid_y - center_y)**2
    bump = np.zeros_like(grid_x)
    mask = distance_sq < radius**2
    bump[mask] = height * (1 - np.sqrt(distance_sq[mask]) / radius)
    return bump

def simulate_bucket_images(height_map, wavelengths, num_buckets=3):
    height, width = height_map.shape
    num_lasers = len(wavelengths)
    height_map_t = torch.from_numpy(height_map).float().view(1, 1, height, width)
    wavelengths_t = torch.tensor(wavelengths, dtype=torch.float32).view(num_lasers, 1, 1, 1)
    deltas = torch.arange(num_buckets, dtype=torch.float32) * (2 * np.pi / num_buckets)
    deltas = deltas.view(1, num_buckets, 1, 1)
    phase = (4 * np.pi / wavelengths_t) * height_map_t
    phase_with_shifts = phase + deltas
    A, B = 128, 100
    predicted_buckets = A + B * torch.cos(phase_with_shifts)
    return predicted_buckets.view(num_lasers * num_buckets, height, width).numpy()

def generate_data(output_dir, num_samples=2):
    """Generates a dataset."""
    if os.path.exists(output_dir):
        print(f"Data directory {output_dir} already exists. Skipping generation.")
        return

    print(f"Generating {num_samples} data samples in {output_dir}...")
    image_size = (512, 512)
    num_bumps_per_sample = 5
    bump_diameter = 50
    bump_height = 40e-6
    wavelengths = [635e-9, 525e-9, 450e-9, 405e-9]
    num_buckets = 3

    os.makedirs(output_dir, exist_ok=True)
    H, W = image_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    for i in range(num_samples):
        ground_truth_height = np.zeros(image_size, dtype=np.float32)
        for _ in range(num_bumps_per_sample):
            center_x = np.random.randint(bump_diameter, W - bump_diameter)
            center_y = np.random.randint(bump_diameter, H - bump_diameter)
            bump = generate_spherical_bump(grid_x, grid_y, center_x, center_y, bump_diameter, bump_height)
            ground_truth_height += bump

        bucket_images = simulate_bucket_images(ground_truth_height, wavelengths, num_buckets)

        sample_dir = os.path.join(output_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        np.save(os.path.join(sample_dir, "ground_truth.npy"), ground_truth_height)
        np.save(os.path.join(sample_dir, "bucket_images.npy"), bucket_images)
    print(f"Data generation complete for {output_dir}.")

# --- Main Pipeline ---

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Generate Datasets ---
    generate_data(args.pretrain_data_dir, num_samples=args.num_pretrain_samples)
    generate_data(args.finetune_data_dir, num_samples=args.num_finetune_samples)

    # --- Model ---
    model = UNet(n_channels=12, n_classes=1).to(device)

    # ===================================================================
    #                            PRE-TRAINING
    # ===================================================================
    print("\n" + "="*50)
    print(" " * 20 + "PRE-TRAINING")
    print("="*50)

    pretrain_dataset = WaferPatchDataset(
        data_dir=args.pretrain_data_dir,
        patch_size=args.patch_size,
        use_augmentation=True
    )
    pretrain_dataloader = DataLoader(
        pretrain_dataset, batch_size=args.pretrain_batch_size, shuffle=True, num_workers=2
    )

    pretrain_criterion = nn.MSELoss()
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)

    for epoch in range(args.pretrain_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(pretrain_dataloader, desc=f"Pre-train Epoch {epoch+1}/{args.pretrain_epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = pretrain_criterion(outputs, targets)
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / len(pretrain_dataloader)
        print(f"Pre-train Epoch {epoch+1}/{args.pretrain_epochs}, Average Loss: {avg_loss:.6f}")

    print("Pre-training finished.")

    # ===================================================================
    #                             FINE-TUNING
    # ===================================================================
    print("\n" + "="*50)
    print(" " * 20 + "FINE-TUNING")
    print("="*50)

    finetune_dataset = WaferPatchDataset(
        data_dir=args.finetune_data_dir,
        patch_size=args.patch_size,
        use_augmentation=True,
        real_data=True
    )
    finetune_dataloader = DataLoader(
        finetune_dataset, batch_size=args.finetune_batch_size, shuffle=True, num_workers=2
    )

    wavelengths = [635e-9, 525e-9, 450e-9, 405e-9]
    finetune_criterion = UNetPhysicsLoss(
        wavelengths=wavelengths, num_buckets=3, smoothness_weight=args.smoothness_weight
    )
    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)

    for epoch in range(args.finetune_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(finetune_dataloader, desc=f"Finetune Epoch {epoch+1}/{args.finetune_epochs}")
        for inputs, _ in progress_bar:
            inputs = inputs.to(device)
            predicted_height = model(inputs)
            loss = finetune_criterion(predicted_height, inputs)
            finetune_optimizer.zero_grad()
            loss.backward()
            finetune_optimizer.step()
            metrics = finetune_criterion.metrics
            epoch_loss += metrics['loss_total']
            progress_bar.set_postfix(loss=metrics['loss_total'])
        avg_loss = epoch_loss / len(finetune_dataloader)
        print(f"Finetune Epoch {epoch+1}/{args.finetune_epochs}, Average Loss: {avg_loss:.6f}")

    print("Fine-tuning finished.")

    # --- Save Final Model ---
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nFinal fine-tuned model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full pre-training and fine-tuning pipeline for wafer inspection.")

    # Data args
    parser.add_argument("--pretrain-data-dir", type=str, default="synthetic_data/train")
    parser.add_argument("--finetune-data-dir", type=str, default="real_data/train")
    parser.add_argument("--num-pretrain-samples", type=int, default=10)
    parser.add_argument("--num-finetune-samples", type=int, default=5)

    # Model saving
    parser.add_argument("--save-path", type=str, default="models/unet_final.pth")

    # Training args
    parser.add_argument("--patch-size", type=int, default=256)

    # Pre-training args
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--pretrain-batch-size", type=int, default=4)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)

    # Fine-tuning args
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-batch-size", type=int, default=2)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--smoothness-weight", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
