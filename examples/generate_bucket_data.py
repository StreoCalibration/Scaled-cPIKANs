import argparse
import numpy as np
from reconstruction.data_generator import generate_synthetic_data, DEFAULT_WAVELENGTHS


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bucket data for reconstruction experiments.")
    parser.add_argument("--shape", type=int, nargs=2, default=[128, 128], metavar=("H", "W"),
                        help="Height and width of generated images (default: 128 128)")
    parser.add_argument("--output", type=str, default="reconstruction_data",
                        help="Directory to save generated numpy arrays (default: reconstruction_data)")
    parser.add_argument("--num-lasers", type=int, default=4,
                        help="Number of lasers to simulate (default: 4)")
    parser.add_argument("--num-buckets", type=int, default=3,
                        help="Number of phase-shifted bucket images per laser (default: 3)")
    parser.add_argument("--wavelengths", type=float, nargs='+', default=None,
                        help="Custom wavelengths for lasers in micrometers; overrides --num-lasers")
    args = parser.parse_args()

    if args.wavelengths is not None:
        wavelengths = args.wavelengths
    else:
        if args.num_lasers <= len(DEFAULT_WAVELENGTHS):
            wavelengths = DEFAULT_WAVELENGTHS[:args.num_lasers]
        else:
            wavelengths = np.linspace(DEFAULT_WAVELENGTHS[0], DEFAULT_WAVELENGTHS[-1], args.num_lasers).tolist()

    generate_synthetic_data(
        shape=tuple(args.shape),
        wavelengths=wavelengths,
        num_buckets=args.num_buckets,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
