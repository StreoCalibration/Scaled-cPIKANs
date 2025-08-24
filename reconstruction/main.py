import numpy as np
import os
from .data_generator import generate_all_data, WAVELENGTHS, OUTPUT_DIR
from .reconstruction import hierarchical_phase_unwrapping_and_reconstruction

def calculate_rmse(ground_truth, reconstructed):
    """Calculates the Root Mean Squared Error between two images."""
    return np.sqrt(np.mean((ground_truth - reconstructed)**2))

def main():
    """
    Main function to run the entire simulation and evaluation pipeline.
    """
    # --- 1. Data Generation ---
    # First, ensure the synthetic data is generated.
    # The data_generator script is written to be executed directly,
    # but we call its main function here for orchestration.
    generate_all_data()

    # --- 2. Load Data ---
    print("\n--- Loading Data for Reconstruction ---")

    # Load ground truth height
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth_height.npy")
    ground_truth_height = np.load(gt_path)
    print(f"Loaded ground truth height from {gt_path}")

    # Load the 4 wrapped phase maps
    wrapped_phases = []
    for i in range(4):
        phase_path = os.path.join(OUTPUT_DIR, f"wrapped_phase_laser_{i+1}.npy")
        wrapped_phases.append(np.load(phase_path))
        print(f"Loaded phase map from {phase_path}")

    # --- 3. Reconstruction ---
    # Reconstruct the height from the loaded phase maps
    reconstructed_height = hierarchical_phase_unwrapping_and_reconstruction(wrapped_phases, WAVELENGTHS)

    # Save the reconstructed height for inspection
    reconstructed_path = os.path.join(OUTPUT_DIR, "reconstructed_height.npy")
    np.save(reconstructed_path, reconstructed_height)
    print(f"\nSaved final reconstructed height to {reconstructed_path}")

    # --- 4. Evaluation ---
    print("\n--- Evaluating Reconstruction Accuracy ---")

    # Debug prints
    print(f"Ground Truth Stats (Raw): Min={np.min(ground_truth_height):.2f}, Max={np.max(ground_truth_height):.2f}, Mean={np.mean(ground_truth_height):.2f}")
    print(f"Reconstructed Stats (Raw): Min={np.min(reconstructed_height):.2f}, Max={np.max(reconstructed_height):.2f}, Mean={np.mean(reconstructed_height):.2f}")

    # To get a more meaningful error of the shape (ignoring global piston/offset error),
    # we can center both height maps by subtracting their mean before calculating RMSE.
    gt_centered = ground_truth_height - np.mean(ground_truth_height)
    recon_centered = reconstructed_height - np.mean(reconstructed_height)

    rmse = calculate_rmse(gt_centered, recon_centered)

    print(f"\n==============================================")
    print(f"Reconstruction Complete!")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"==============================================")

if __name__ == "__main__":
    main()
