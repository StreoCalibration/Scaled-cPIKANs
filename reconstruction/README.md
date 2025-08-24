# Multi-Wavelength 3D Reconstruction Simulation

This module provides a complete pipeline for simulating and evaluating a 3D height reconstruction process based on multi-wavelength phase-shifting profilometry. It generates synthetic data for a 4-laser, 3-bucket setup and then uses a hierarchical algorithm to reconstruct the height from the generated phase maps.

## Overview

The core idea is to use multiple laser wavelengths to resolve the phase ambiguity inherent in single-wavelength phase-shifting techniques. By combining the information from different wavelengths, we can create "synthetic wavelengths" that allow for robust unwrapping over a large height range, leading to a precise final height measurement.

This simulation serves as a development and testing environment for the reconstruction algorithm.

## File Structure

- `data_generator.py`: Contains all the logic for generating synthetic data. It creates a ground truth height map and then simulates the 3-bucket measurement process for each of the 4 lasers to produce wrapped phase maps.
- `reconstruction.py`: Implements the core height reconstruction algorithm. It uses a hierarchical approach, leveraging synthetic wavelengths to unwrap and combine the phase information from the different lasers.
- `main.py`: The main orchestration script. It runs the entire pipeline: it calls the data generator, feeds the data to the reconstruction algorithm, and evaluates the result by calculating the Root Mean Squared Error (RMSE) against the ground truth.
- `reconstruction_data/`: This directory is created when `main.py` is run. It stores the output of the data generator, including the ground truth height map (`.npy`) and the wrapped phase maps (`.npy`) for each laser.

## How to Run

### Dependencies

The simulation requires the following Python libraries:
- `numpy`
- `scikit-image`
- `matplotlib` (optional, but good to have)

You can install them using pip:
```bash
pip install numpy scikit-image matplotlib
```

### Execution

To run the entire simulation and evaluation pipeline, execute the `main.py` script as a module from the root directory of the project:

```bash
python3 -m reconstruction.main
```

The script will first generate the synthetic data (or overwrite existing data in `reconstruction_data/`), then run the reconstruction algorithm, and finally print the calculated RMSE between the ground truth height and the reconstructed height to the console.
