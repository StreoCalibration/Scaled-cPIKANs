import numpy as np
from skimage.restoration import unwrap_phase

def calculate_synthetic_wavelength(lambda1, lambda2):
    """Calculates the synthetic wavelength for two given wavelengths."""
    return (lambda1 * lambda2) / np.abs(lambda1 - lambda2)

def reconstruct_height_from_unwrapped_phase(unwrapped_phase, wavelength):
    """Reconstructs height from a single unwrapped phase map."""
    return (unwrapped_phase * wavelength) / (4 * np.pi)

def _refine_height(reference_height, ambiguous_height, wavelength):
    """
    Refines a height map by correcting its fringe order based on a reference.
    """
    # The ambiguity in height is half the wavelength
    ambiguity_step = wavelength / 2.0
    k = np.round((reference_height - ambiguous_height) / ambiguity_step)
    refined_height = ambiguous_height + k * ambiguity_step
    return refined_height

def hierarchical_phase_unwrapping_and_reconstruction(wrapped_phases, wavelengths):
    """
    Reconstructs height from multiple wrapped phase maps using a robust hierarchical
    approach based on a sorted order of synthetic wavelengths.

    Args:
        wrapped_phases (list[np.ndarray]): A list of 4 wrapped phase maps.
        wavelengths (list[float]): A list of 4 corresponding wavelengths.

    Returns:
        np.ndarray: The final reconstructed height map.
    """
    if len(wrapped_phases) != 4 or len(wavelengths) != 4:
        raise ValueError("Requires exactly 4 phase maps and 4 wavelengths.")

    print("--- Starting Robust Hierarchical Reconstruction ---")

    # Step 1: Individually unwrap all 4 phase maps.
    print("Step 1: Individually unwrapping all 4 phase maps...")
    unwrapped_phases = [unwrap_phase(p) for p in wrapped_phases]
    print("Individual unwrapping complete.")

    # Step 2: Generate all possible synthetic wavelengths and sort them from coarsest to finest.
    synthetic_pairs = []
    for i in range(len(wavelengths)):
        for j in range(i + 1, len(wavelengths)):
            l1, l2 = wavelengths[i], wavelengths[j]
            p1, p2 = unwrapped_phases[i], unwrapped_phases[j]
            synth_lambda = calculate_synthetic_wavelength(l1, l2)
            # Store the synthetic wavelength and the corresponding phase difference
            synthetic_pairs.append({'lambda': synth_lambda, 'phase_diff': p1 - p2, 'l1': l1, 'l2': l2})

    # Sort by wavelength, descending (coarsest first)
    synthetic_pairs.sort(key=lambda x: x['lambda'], reverse=True)

    print("\nStep 2: Processing synthetic wavelengths in order from coarse to fine...")

    # Step 3: Start with the coarsest synthetic wavelength for the initial estimate.
    coarsest_pair = synthetic_pairs[0]
    unwrapped_coarsest_phase_diff = unwrap_phase(coarsest_pair['phase_diff'])
    current_height_estimate = reconstruct_height_from_unwrapped_phase(unwrapped_coarsest_phase_diff, coarsest_pair['lambda'])
    print(f"  - Initial coarse estimate using L_synth = {coarsest_pair['lambda']:.2f} um (from {coarsest_pair['l1']}, {coarsest_pair['l2']})")

    # Step 4: Iteratively refine the height using progressively finer synthetic wavelengths.
    for i in range(1, len(synthetic_pairs)):
        pair = synthetic_pairs[i]
        unwrapped_phase_diff = unwrap_phase(pair['phase_diff'])
        ambiguous_height = reconstruct_height_from_unwrapped_phase(unwrapped_phase_diff, pair['lambda'])

        current_height_estimate = _refine_height(current_height_estimate, ambiguous_height, pair['lambda'])
        print(f"  - Refined height using L_synth = {pair['lambda']:.2f} um (from {pair['l1']}, {pair['l2']})")

    # Step 5: Final refinement using the shortest original wavelength for highest precision.
    shortest_wavelength = min(wavelengths)
    shortest_wavelength_idx = wavelengths.index(shortest_wavelength)
    finest_unwrapped_phase = unwrapped_phases[shortest_wavelength_idx]

    height_finest_ambiguous = reconstruct_height_from_unwrapped_phase(finest_unwrapped_phase, shortest_wavelength)

    final_height = _refine_height(current_height_estimate, height_finest_ambiguous, shortest_wavelength)
    print(f"\nStep 3: Final refinement using shortest wavelength L = {shortest_wavelength:.2f} um.")

    print("\n--- Hierarchical Reconstruction Complete ---")
    return final_height

# This file is intended to be used as a module, so no __main__ block for now.
