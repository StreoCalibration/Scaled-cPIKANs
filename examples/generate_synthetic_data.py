import os
import numpy as np
import torch

def generate_spherical_bump(grid_x, grid_y, center_x, center_y, diameter, height):
    """
    Generates a single spherical bump on a grid.
    """
    radius = diameter / 2
    distance_sq = (grid_x - center_x)**2 + (grid_y - center_y)**2
    bump = np.zeros_like(grid_x)
    mask = distance_sq < radius**2
    bump[mask] = height * (1 - np.sqrt(distance_sq[mask]) / radius)
    return bump

def simulate_bucket_images(height_map, wavelengths, num_buckets=3):
    """
    Simulates bucket images from a height map.
    """
    height, width = height_map.shape
    num_lasers = len(wavelengths)
    bucket_images = np.zeros((num_lasers * num_buckets, height, width))

    # Use torch for broadcasting and calculation
    height_map_t = torch.from_numpy(height_map).float().view(1, 1, height, width)
    wavelengths_t = torch.tensor(wavelengths, dtype=torch.float32).view(num_lasers, 1, 1, 1)
    deltas = torch.arange(num_buckets, dtype=torch.float32) * (2 * np.pi / num_buckets)
    deltas = deltas.view(1, num_buckets, 1, 1)

    # Physical simulation formula
    phase = (4 * np.pi / wavelengths_t) * height_map_t
    phase_with_shifts = phase + deltas

    # Intensity formula (assuming A=128, B=100 from example)
    A, B = 128, 100
    predicted_buckets = A + B * torch.cos(phase_with_shifts)

    # Reshape to (num_lasers * num_buckets, H, W)
    return predicted_buckets.view(num_lasers * num_buckets, height, width).numpy()

def main():
    # --- Configuration ---
    output_dir = "synthetic_data/train"
    image_size = (512, 512)
    num_samples = 10  # Number of data pairs to generate
    num_bumps_per_sample = 5
    bump_diameter = 50  # pixels
    bump_height = 40e-6  # 40 micrometers

    # Wavelengths for 4 lasers (in meters)
    wavelengths = [635e-9, 525e-9, 450e-9, 405e-9]
    num_buckets = 3

    # --- Generation ---
    os.makedirs(output_dir, exist_ok=True)

    H, W = image_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")

        # Create a flat ground truth height map
        ground_truth_height = np.zeros(image_size, dtype=np.float32)

        # Add random bumps
        for _ in range(num_bumps_per_sample):
            center_x = np.random.randint(bump_diameter, W - bump_diameter)
            center_y = np.random.randint(bump_diameter, H - bump_diameter)
            bump = generate_spherical_bump(grid_x, grid_y, center_x, center_y, bump_diameter, bump_height)
            ground_truth_height += bump

        # Simulate the bucket images from the height map
        bucket_images = simulate_bucket_images(ground_truth_height, wavelengths, num_buckets)

        # Save the data
        sample_dir = os.path.join(output_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        np.save(os.path.join(sample_dir, "ground_truth.npy"), ground_truth_height)
        np.save(os.path.join(sample_dir, "bucket_images.npy"), bucket_images)

    print(f"Successfully generated {num_samples} data samples in '{output_dir}'.")

if __name__ == "__main__":
    main()
