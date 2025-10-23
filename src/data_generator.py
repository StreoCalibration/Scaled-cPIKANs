import numpy as np
import os
from PIL import Image

# --- Configuration ---
# Default wavelengths for four lasers (in micrometers, for example).
# Users can supply their own list to change the number of lasers.
DEFAULT_WAVELENGTHS = [5.0, 5.5, 6.05, 6.655]

def generate_synthetic_data(
    shape=(128, 128),
    wavelengths=None,
    num_buckets=3,
    save_path="reconstruction_data",
    output_format='bmp'
):
    """
    Generates and optionally saves all synthetic data for the 3D reconstruction problem.

    This function creates a ground truth height map and then simulates a
    phase-shifting bucket measurement process for each laser wavelength to
    produce the raw bucket images.

    Args:
        shape (tuple[int, int]): The (height, width) of the data grids to generate.
        wavelengths (list[float] | None): Wavelength for each laser. If None,
            ``DEFAULT_WAVELENGTHS`` is used.
        num_buckets (int): Number of phase-shifted bucket images per laser.
        save_path (str | None): Directory to save generated files. If ``None``
            data is not saved to disk.
        output_format (str): The format for the output bucket images ('bmp' or 'png').

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - ground_truth_height (np.ndarray): The 2D ground truth height map.
            - bucket_images (np.ndarray): An array of shape
              ``(len(wavelengths) * num_buckets, H, W)`` containing the bucket
              images.
    """
    print("--- Starting Synthetic Data Generation ---")
    height, width = shape

    if wavelengths is None:
        wavelengths = DEFAULT_WAVELENGTHS
    num_lasers = len(wavelengths)

    # --- Ground Truth Height Map Generation ---
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    peak = 15.0 * np.exp(-((xx - 0.2)**2 + (yy + 0.3)**2) / 0.2)
    plane = 5.0 * xx + 2.5 * yy
    cone = 10.0 * np.sqrt((xx + 0.5)**2 + (yy - 0.4)**2)
    ground_truth_height = peak + plane + cone
    print(f"Generated ground truth height map of size {shape}.")

    # --- Simulation of Phase-Shift Imaging for each laser ---
    all_bucket_images = []
    deltas = np.linspace(0, 2 * np.pi, num_buckets, endpoint=False)

    for i, wavelength in enumerate(wavelengths):
        phase = (4 * np.pi * ground_truth_height) / wavelength
        A = 128
        B = 100
        buckets = [A + B * np.cos(phase + delta) for delta in deltas]
        laser_buckets = np.stack(buckets, axis=0)
        all_bucket_images.append(laser_buckets)
        print(f"Generated {num_buckets} bucket images for Laser {i+1} (Wavelength: {wavelength} um)...")

    # Reshape to (total_buckets, H, W)
    bucket_images_np = np.concatenate(all_bucket_images, axis=0)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(f"\nSaving data to directory: {save_path}")
        gt_path = os.path.join(save_path, "ground_truth_height.npy")
        np.save(gt_path, ground_truth_height)
        print(f"Saved ground truth height to {gt_path}")

        # Save bucket images as individual files
        bucket_images_uint8 = np.clip(bucket_images_np, 0, 255).astype(np.uint8)
        for i in range(bucket_images_uint8.shape[0]):
            img = Image.fromarray(bucket_images_uint8[i])
            img_path = os.path.join(save_path, f"bucket_{i:02d}.{output_format}")
            img.save(img_path)
        print(f"Saved {len(bucket_images_np)} bucket images as .{output_format} files.")

        # Save wavelengths as a simple text file or numpy array
        wl_path = os.path.join(save_path, "wavelengths.txt")
        np.savetxt(wl_path, np.asarray(wavelengths, dtype=np.float32))
        print(f"Saved wavelengths to {wl_path}")

    print("\n--- Synthetic Data Generation Complete ---")
    # Return the ground truth and the bucket images with the correct shape
    return ground_truth_height, bucket_images_np.reshape(num_lasers, num_buckets, height, width)


if __name__ == "__main__":
    # Use a smaller shape for testing to avoid memory issues.
    generate_synthetic_data(shape=(128, 128), save_path="reconstruction_data", output_format='bmp')
