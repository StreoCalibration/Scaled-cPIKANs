import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from src.models import Scaled_cPIKAN


def main():
    parser = argparse.ArgumentParser(description="Run patched inference with a trained Scaled-cPIKAN model.")
    parser.add_argument("--model-path", type=str, default="bucket_model.pth",
                        help="Path to trained model weights")
    parser.add_argument("--output", type=str, default="reconstructed_height.npy",
                        help="File to save reconstructed height map")
    parser.add_argument("--height", type=int, required=True,
                        help="Full height of the image to reconstruct.")
    parser.add_argument("--width", type=int, required=True,
                        help="Full width of the image to reconstruct.")
    parser.add_argument("--patch-size", type=int, default=512,
                        help="Size of patches to process at a time to manage memory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    domain_min = torch.tensor([0.0, 0.0], device=device)
    domain_max = torch.tensor([1.0, 1.0], device=device)
    model = Scaled_cPIKAN(layers_dims=[2, 64, 64, 64, 1],
                          cheby_order=8,
                          domain_min=domain_min,
                          domain_max=domain_max).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- Patched Inference ---
    H, W = args.height, args.width
    patch_size = args.patch_size
    reconstructed_height = np.zeros((H, W), dtype=np.float32)

    # Calculate number of patches for tqdm progress bar
    n_patches_h = (H + patch_size - 1) // patch_size
    n_patches_w = (W + patch_size - 1) // patch_size
    total_patches = n_patches_h * n_patches_w

    print(f"Starting reconstruction of {H}x{W} image with {patch_size}x{patch_size} patches...")
    with torch.no_grad(), tqdm(total=total_patches, desc="Reconstructing") as pbar:
        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                # Define patch boundaries, handling image edges
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                patch_H = y_end - y
                patch_W = x_end - x

                # Generate coordinates for the current patch, normalized to [0, 1]
                # relative to the full image dimensions. Note the use of (W-1) and (H-1)
                # to ensure the coordinate range is exactly [0, 1].
                x_coords = torch.linspace(x / (W - 1), (x_end - 1) / (W - 1), patch_W, device=device)
                y_coords = torch.linspace(y / (H - 1), (y_end - 1) / (H - 1), patch_H, device=device)
                grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
                coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

                # Run model on the patch coordinates
                pred_patch = model(coords).view(patch_H, patch_W).cpu().numpy()

                # Place the reconstructed patch into the full image array
                reconstructed_height[y:y_end, x:x_end] = pred_patch
                pbar.update(1)

    np.save(args.output, reconstructed_height)
    print(f"\nReconstruction complete. Saved height map to {args.output}")


if __name__ == "__main__":
    main()
