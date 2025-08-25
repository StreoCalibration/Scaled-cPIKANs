import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from src.data import PinnPatchDataset
from src.models import Scaled_cPIKAN
from src.loss import PinnReconstructionLoss

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

def generate_data(output_dir, num_samples, image_size, num_buckets, wavelengths, output_format='npy'):
    """Generates a dataset."""
    if os.path.exists(output_dir):
        print(f"Data directory {output_dir} already exists. Skipping generation.")
        return

    print(f"Generating {num_samples} data samples in {output_dir} with format {output_format}...")
    num_bumps_per_sample = 5
    bump_diameter = 50
    bump_height = 40e-6

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

        # Always save ground truth as .npy for precision
        np.save(os.path.join(sample_dir, "ground_truth.npy"), ground_truth_height)

        # Save bucket images based on the specified format
        if output_format == 'npy':
            np.save(os.path.join(sample_dir, "bucket_images.npy"), bucket_images)
        elif output_format in ['bmp', 'png']:
            # Normalize to 0-255 for standard image formats
            bucket_images_uint8 = np.clip(bucket_images, 0, 255).astype(np.uint8)
            for j in range(bucket_images_uint8.shape[0]):
                img = Image.fromarray(bucket_images_uint8[j])
                img.save(os.path.join(sample_dir, f"bucket_{j:02d}.{output_format}"))
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Data generation complete for {output_dir}.")

# --- Main Pipeline ---

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    image_size = (args.image_size, args.image_size)
    wavelengths = args.wavelengths

    # --- Generate Datasets ---
    # Pre-training data is always generated synthetically
    generate_data(args.pretrain_data_dir, args.num_pretrain_samples, image_size, args.num_buckets, wavelengths, args.output_format)

    # Fine-tuning data can be user-provided (default) or generated (if flag is set)
    if args.generate_finetune_data:
        generate_data(args.finetune_data_dir, args.num_finetune_samples, image_size, args.num_buckets, wavelengths, args.output_format)
    else:
        print(f"Skipping fine-tuning data generation. Expecting user-provided data in {args.finetune_data_dir}.")

    # --- Model ---
    domain_min = torch.tensor([0.0, 0.0], device=device)
    domain_max = torch.tensor([1.0, 1.0], device=device)
    model = Scaled_cPIKAN(
        layers_dims=[2, 128, 128, 128, 1],
        cheby_order=8,
        domain_min=domain_min,
        domain_max=domain_max
    ).to(device)

    # ===================================================================
    #                            PRE-TRAINING
    # ===================================================================
    print("\n" + "="*50)
    print(" " * 20 + "PRE-TRAINING")
    print("="*50)

    num_channels = len(args.wavelengths) * args.num_buckets
    pretrain_dataset = PinnPatchDataset(
        data_dir=args.pretrain_data_dir,
        patch_size=args.patch_size,
        full_image_size=image_size,
        output_format=args.output_format,
        real_data=False,
        num_channels=num_channels
    )
    # Note: batch_size for PINN training is typically 1, as we process a full patch of points
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=1, shuffle=True, num_workers=2)

    pretrain_criterion = nn.MSELoss()
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)

    for epoch in range(args.pretrain_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(pretrain_dataloader, desc=f"Pre-train Epoch {epoch+1}/{args.pretrain_epochs}")
        for coords, _, gt_patch in progress_bar:
            coords, gt_patch = coords.squeeze(0).to(device), gt_patch.squeeze(0).to(device)

            predicted_height = model(coords).view(1, -1)
            loss = pretrain_criterion(predicted_height, gt_patch)

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

    finetune_dataset = PinnPatchDataset(
        data_dir=args.finetune_data_dir,
        patch_size=args.patch_size,
        full_image_size=image_size,
        output_format=args.output_format,
        real_data=True, # In fine-tuning we don't assume access to GT height
        num_channels=num_channels
    )
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=1, shuffle=True, num_workers=2)

    finetune_criterion = PinnReconstructionLoss(
        wavelengths=wavelengths,
        num_buckets=args.num_buckets,
        smoothness_weight=args.smoothness_weight
    )
    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)

    for epoch in range(args.finetune_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(finetune_dataloader, desc=f"Finetune Epoch {epoch+1}/{args.finetune_epochs}")
        for coords, bucket_patch, _ in progress_bar:
            coords, bucket_patch = coords.squeeze(0).to(device), bucket_patch.squeeze(0).to(device)
            coords.requires_grad_(True)

            predicted_height = model(coords).view(1, -1)

            loss = finetune_criterion(predicted_height, coords, bucket_patch)

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
    parser.add_argument("--pretrain-data-dir", type=str, default="synthetic_data/train", help="Directory for pre-training dataset.")
    parser.add_argument("--finetune-data-dir", type=str, default="real_data/train", help="Directory for fine-tuning dataset.")
    parser.add_argument("--generate-finetune-data", action="store_true", help="Flag to generate synthetic data for fine-tuning. If not set, user-provided data is expected.")
    parser.add_argument("--num-pretrain-samples", type=int, default=10, help="Number of synthetic samples for pre-training.")
    parser.add_argument("--num-finetune-samples", type=int, default=5, help="Number of synthetic samples for fine-tuning (if generated).")
    parser.add_argument("--image-size", type=int, default=512, help="Size of the full synthetic images.")
    parser.add_argument("--num-buckets", type=int, default=3, help="Number of buckets per laser.")
    parser.add_argument('--wavelengths', type=float, nargs='+', default=[635e-9, 525e-9, 450e-9, 405e-9],
                        help='List of laser wavelengths in meters.')
    parser.add_argument('--output-format', type=str, default='bmp', choices=['bmp', 'png'],
                        help='Output format for generated bucket images. Default is bmp.')

    # Model saving
    parser.add_argument("--save-path", type=str, default="models/pinn_final.pth")

    # Training args
    parser.add_argument("--patch-size", type=int, default=64, help="Size of the square patches to train on.")

    # Pre-training args
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)

    # Fine-tuning args
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--smoothness-weight", type=float, default=1e-7)

    args = parser.parse_args()
    main(args)
