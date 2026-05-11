# generate_sine_data.py
import numpy as np
import argparse

def generate_sine_data(
    n_series: int = 10,
    length: int = 1000,
    amplitude_range: tuple = (0.5, 2.0),
    freq_range: tuple = (0.01, 0.05),
    phase_range: tuple = (0, 2*np.pi),
    noise_std: float = 0.0,
    random_seed: int = 42,
    output_file: str = "sine_data.npz"
):
    """
    Generate N independent sine waves with varying amplitude, frequency, phase.
    Shape of positions: (n_series, length)
    """
    np.random.seed(random_seed)

    # Sample parameters for each series
    amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], n_series)
    frequencies = np.random.uniform(freq_range[0], freq_range[1], n_series)
    phases = np.random.uniform(phase_range[0], phase_range[1], n_series)

    # Time vector (same for all series)
    t = np.arange(length)

    positions = np.zeros((n_series, length))
    for i in range(n_series):
        pos = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i])
        if noise_std > 0:
            pos += np.random.normal(0, noise_std, length)
        positions[i, :] = pos

    # Save in the same format as alanine_psi_test.npz (only 'positions' key)
    np.savez(output_file, positions=positions)
    print(f"Saved {n_series} series of length {length} to {output_file}")
    print("Parameter ranges used:")
    print(f"  Amplitude: {amplitude_range}")
    print(f"  Frequency: {freq_range}")
    print(f"  Phase: {phase_range}")
    print(f"  Noise std: {noise_std}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sine wave data with varying amplitude.")
    parser.add_argument("--n_series", "-n", type=int, default=10, help="Number of time series (default: 10)")
    parser.add_argument("--length", "-L", type=int, default=1000, help="Length of each time series (default: 1000)")
    parser.add_argument("--amp_min", type=float, default=0.5, help="Minimum amplitude (default: 0.5)")
    parser.add_argument("--amp_max", type=float, default=2.0, help="Maximum amplitude (default: 2.0)")
    parser.add_argument("--freq_min", type=float, default=0.01, help="Minimum frequency (cycles per step) (default: 0.01)")
    parser.add_argument("--freq_max", type=float, default=0.05, help="Maximum frequency (default: 0.05)")
    parser.add_argument("--phase_min", type=float, default=0, help="Minimum phase (rad) (default: 0)")
    parser.add_argument("--phase_max", type=float, default=2*np.pi, help="Maximum phase (rad) (default: 2π)")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Standard deviation of Gaussian noise (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", "-o", type=str, default="sine_data.npz", help="Output .npz file (default: sine_data.npz)")
    args = parser.parse_args()

    generate_sine_data(
        n_series=args.n_series,
        length=args.length,
        amplitude_range=(args.amp_min, args.amp_max),
        freq_range=(args.freq_min, args.freq_max),
        phase_range=(args.phase_min, args.phase_max),
        noise_std=args.noise_std,
        random_seed=args.seed,
        output_file=args.output
    )

if __name__ == "__main__":
    main()