import argparse
import os

import numpy as np
import torch

from src.models import Scaled_cPIKAN


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Scaled-cPIKAN model on bucket data.")
    parser.add_argument("--data-dir", type=str, default="reconstruction_data",
                        help="Directory containing bucket_images.npy")
    parser.add_argument("--model-path", type=str, default="bucket_model.pth",
                        help="Path to trained model weights")
    parser.add_argument("--output", type=str, default="reconstructed_height.npy",
                        help="File to save reconstructed height map")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bucket_images = np.load(os.path.join(args.data_dir, "bucket_images.npy"))
    H, W = bucket_images.shape[2:]

    x = torch.linspace(0, 1, H, device=device)
    y = torch.linspace(0, 1, W, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    domain_min = torch.tensor([0.0, 0.0], device=device)
    domain_max = torch.tensor([1.0, 1.0], device=device)
    model = Scaled_cPIKAN(layers_dims=[2, 64, 64, 64, 1],
                          cheby_order=8,
                          domain_min=domain_min,
                          domain_max=domain_max).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_height = model(coords).view(H, W).cpu().numpy()

    np.save(args.output, pred_height)
    print(f"Saved reconstructed height to {args.output}")


if __name__ == "__main__":
    main()
