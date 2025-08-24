import numpy as np
import os

# --- Configuration ---
# Start with smaller dimensions for development
IMG_WIDTH = 512
IMG_HEIGHT = 512
# Final dimensions
# IMG_WIDTH = 9377
# IMG_HEIGHT = 7000

OUTPUT_DIR = "reconstruction_data"

# Wavelengths for the 4 lasers (in micrometers, for example)
# Chosen to be non-integer multiples to aid in unwrapping.
WAVELENGTHS = [5.0, 5.5, 6.05, 6.655]

# --- Ground Truth Height Map Generation ---
def create_ground_truth_height(width, height):
    """
    Generates a non-trivial ground truth height map for simulation.

    This creates a surface with a combination of a Gaussian peak and a tilted plane.

    Args:
        width (int): The width of the height map.
        height (int): The height of the height map.

    Returns:
        np.ndarray: A 2D numpy array representing the height map.
    """
    print(f"Generating ground truth height map of size ({height}, {width})...")
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)

    # A Gaussian peak
    peak = 15.0 * np.exp(-((xx - 0.2)**2 + (yy + 0.3)**2) / 0.2)

    # A tilted plane
    plane = 5.0 * xx + 2.5 * yy

    # A smooth cone
    cone = 10.0 * np.sqrt((xx + 0.5)**2 + (yy - 0.4)**2)

    height_map = peak + plane + cone
    print("Ground truth height map generated.")
    return height_map

# --- Simulation of Phase-Shift Imaging ---
def simulate_bucket_images(height_map, wavelength):
    """
    Simulates the 3-bucket phase-shifting measurement process for a single laser.

    Args:
        height_map (np.ndarray): The ground truth height map.
        wavelength (float): The wavelength of the laser.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of 3 intensity images (I0, I1, I2).
    """
    # Phase is proportional to height and inversely proportional to wavelength
    phase = (4 * np.pi * height_map) / wavelength

    # Simulate intensity images with some background illumination (A) and modulation (B)
    A = 128  # Background intensity
    B = 100  # Amplitude of the cosine term

    # Phase shifts for the 3 buckets
    delta_0 = 0
    delta_1 = 2 * np.pi / 3
    delta_2 = 4 * np.pi / 3

    I0 = A + B * np.cos(phase + delta_0)
    I1 = A + B * np.cos(phase + delta_1)
    I2 = A + B * np.cos(phase + delta_2)

    return I0, I1, I2

def calculate_wrapped_phase(I0, I1, I2):
    """
    Calculates the wrapped phase from the three bucket images.

    Args:
        I0, I1, I2 (np.ndarray): The three intensity bucket images.

    Returns:
        np.ndarray: The calculated wrapped phase map in the range [-pi, pi].
    """
    # Formula for 3-step phase-shifting with 2pi/3 shifts
    numerator = np.sqrt(3) * (I2 - I1)
    denominator = 2 * I0 - I1 - I2

    wrapped_phase = np.arctan2(numerator, denominator)
    return wrapped_phase

def generate_all_data():
    """
    Main function to generate and save all synthetic data.
    """
    print("--- Starting Synthetic Data Generation ---")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 1. Generate and save ground truth height
    ground_truth = create_ground_truth_height(IMG_WIDTH, IMG_HEIGHT)
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth_height.npy")
    np.save(gt_path, ground_truth)
    print(f"Saved ground truth height to {gt_path}")

    # 2. Generate and save phase maps for each laser
    for i, wavelength in enumerate(WAVELENGTHS):
        print(f"\nProcessing Laser {i+1} (Wavelength: {wavelength} um)...")

        # Simulate bucket images
        I0, I1, I2 = simulate_bucket_images(ground_truth, wavelength)

        # Calculate wrapped phase
        wrapped_phase = calculate_wrapped_phase(I0, I1, I2)

        # Save the phase map
        phase_path = os.path.join(OUTPUT_DIR, f"wrapped_phase_laser_{i+1}.npy")
        np.save(phase_path, wrapped_phase)
        print(f"Saved wrapped phase for laser {i+1} to {phase_path}")

    print("\n--- Synthetic Data Generation Complete ---")

if __name__ == "__main__":
    generate_all_data()
