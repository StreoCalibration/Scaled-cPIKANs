import numpy as np
import os

# --- Configuration ---
# Default wavelengths for four lasers (in micrometers, for example).
# Users can supply their own list to change the number of lasers.
DEFAULT_WAVELENGTHS = [5.0, 5.5, 6.05, 6.655]

# Backward compatibility alias
Wavelengths = DEFAULT_WAVELENGTHS


def generate_synthetic_data(
    shape=(128, 128),
    wavelengths=None,
    num_buckets=3,
    save_path="reconstruction_data",
):
    """
    Generates and optionally saves all synthetic data for the 3D reconstruction problem.

    This function creates a ground truth height map and then simulates a
    phase-shifting bucket measurement process for each laser wavelength to
    produce both the raw bucket images and the final wrapped phase maps.

    Args:
        shape (tuple[int, int]): The (height, width) of the data grids to generate.
        wavelengths (list[float] | None): Wavelength for each laser. If None,
            ``DEFAULT_WAVELENGTHS`` is used.
        num_buckets (int): Number of phase-shifted bucket images per laser.
        save_path (str | None): Directory to save generated ``.npy`` files. If
            ``None`` data is not saved to disk.

    Returns:
        tuple[np.ndarray, list[np.ndarray], np.ndarray]: A tuple containing:
            - ground_truth_height (np.ndarray): The 2D ground truth height map.
            - wrapped_phases (list[np.ndarray]): A list of ``len(wavelengths)`` wrapped
              phase maps.
            - bucket_images (np.ndarray): An array of shape
              ``(len(wavelengths), num_buckets, H, W)`` containing the bucket
              images for each laser.
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
    wrapped_phases = []
    deltas = np.linspace(0, 2 * np.pi, num_buckets, endpoint=False)

    for i, wavelength in enumerate(wavelengths):
        phase = (4 * np.pi * ground_truth_height) / wavelength

        A = 128
        B = 100

        buckets = [A + B * np.cos(phase + delta) for delta in deltas]
        laser_buckets = np.stack(buckets, axis=0)
        all_bucket_images.append(laser_buckets)

        sum_sin = np.sum(laser_buckets * np.sin(deltas)[:, None, None], axis=0)
        sum_cos = np.sum(laser_buckets * np.cos(deltas)[:, None, None], axis=0)
        wrapped_phase = np.arctan2(-sum_sin, sum_cos)
        wrapped_phases.append(wrapped_phase)
        print(
            f"Generated {num_buckets} bucket images and wrapped phase for Laser {i+1} (Wavelength: {wavelength} um)..."
        )

    bucket_images_np = np.stack(all_bucket_images, axis=0)  # Shape: (num_lasers, num_buckets, H, W)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(f"\nSaving data to directory: {save_path}")
        gt_path = os.path.join(save_path, "ground_truth_height.npy")
        np.save(gt_path, ground_truth_height)
        print(f"Saved ground truth height to {gt_path}")

        for i, wrapped_phase in enumerate(wrapped_phases):
            phase_path = os.path.join(save_path, f"wrapped_phase_laser_{i+1}.npy")
            np.save(phase_path, wrapped_phase)
            print(f"Saved wrapped phase for laser {i+1} to {phase_path}")

        buckets_path = os.path.join(save_path, "bucket_images.npy")
        np.save(buckets_path, bucket_images_np)
        print(f"Saved all bucket images to {buckets_path}")

        wl_path = os.path.join(save_path, "wavelengths.npy")
        np.save(wl_path, np.asarray(wavelengths, dtype=np.float32))
        print(f"Saved wavelengths to {wl_path}")

    print("\n--- Synthetic Data Generation Complete ---")
    return ground_truth_height, wrapped_phases, bucket_images_np


if __name__ == "__main__":
    generate_synthetic_data(shape=(512, 512), save_path="reconstruction_data")
