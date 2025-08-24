import numpy as np
import os

# --- Configuration ---
# Wavelengths for the 4 lasers (in micrometers, for example)
# Chosen to be non-integer multiples to aid in unwrapping.
# Renamed from WAVELENGTHS to Wavelengths to match import in the PINN script.
Wavelengths = [5.0, 5.5, 6.05, 6.655]

def generate_synthetic_data(shape=(128, 128), save_path="reconstruction_data"):
    """
    Generates and optionally saves all synthetic data for the 3D reconstruction problem.

    This function creates a ground truth height map and then simulates the
    3-bucket measurement process for each of the 4 lasers to produce
    wrapped phase maps.

    Args:
        shape (tuple[int, int]): The (height, width) of the data grids to generate.
        save_path (str | None): The directory to save the generated .npy files.
                                If None, data is not saved to disk.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: A tuple containing:
            - ground_truth_height (np.ndarray): The 2D ground truth height map.
            - wrapped_phases (list[np.ndarray]): A list of 4 wrapped phase maps.
    """
    print("--- Starting Synthetic Data Generation ---")
    height, width = shape

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
    wrapped_phases = []
    for i, wavelength in enumerate(Wavelengths):
        # Phase is proportional to height and inversely proportional to wavelength
        phase = (4 * np.pi * ground_truth_height) / wavelength

        # Simulate intensity images with some background illumination (A) and modulation (B)
        A = 128
        B = 100
        # Phase shifts for the 3 buckets
        delta_0 = 0
        delta_1 = 2 * np.pi / 3
        delta_2 = 4 * np.pi / 3
        I0 = A + B * np.cos(phase + delta_0)
        I1 = A + B * np.cos(phase + delta_1)
        I2 = A + B * np.cos(phase + delta_2)

        # Calculate wrapped phase from the three bucket images
        numerator = np.sqrt(3) * (I2 - I1)
        denominator = 2 * I0 - I1 - I2
        wrapped_phase = np.arctan2(numerator, denominator)
        wrapped_phases.append(wrapped_phase)
        print(f"Generated wrapped phase for Laser {i+1} (Wavelength: {wavelength} um)...")

    # --- Save data if a path is provided ---
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(f"\nSaving data to directory: {save_path}")
        # Save ground truth
        gt_path = os.path.join(save_path, "ground_truth_height.npy")
        np.save(gt_path, ground_truth_height)
        print(f"Saved ground truth height to {gt_path}")
        # Save phase maps
        for i, wrapped_phase in enumerate(wrapped_phases):
            phase_path = os.path.join(save_path, f"wrapped_phase_laser_{i+1}.npy")
            np.save(phase_path, wrapped_phase)
            print(f"Saved wrapped phase for laser {i+1} to {phase_path}")

    print("\n--- Synthetic Data Generation Complete ---")
    return ground_truth_height, wrapped_phases


if __name__ == "__main__":
    # To maintain the original behavior of running this file as a script
    generate_synthetic_data(shape=(512, 512), save_path="reconstruction_data")
