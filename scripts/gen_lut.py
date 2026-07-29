#!/usr/bin/env python3
"""
Generate lookup tables for Struve-Neumann decay modes S₀(z) and S₁(z).

Mathematical definitions:
  S_ν(z) = (π/2) * [H_ν(z) - Y_ν(z)]
where H_ν is the Struve function and Y_ν is the Bessel function of the second kind.

The derivative relationship:
  S₀'(z) = 1 - S₁(z)

These LUTs are used by the GPU shader to evaluate the analytical Laplace-domain
gravity kernel without computing special functions on the GPU.
"""

import numpy as np
from scipy.special import struve, yv
import struct
import json

# LUT configuration
Z_MAX = 50.0          # Maximum argument for the LUT
N_SAMPLES = 4096      # Number of samples (power of 2 for GPU efficiency)
OUTPUT_BIN = "lut_struve.bin"
OUTPUT_NPY = "lut_struve.npy"
OUTPUT_JSON = "lut_config.json"

def compute_struve_neumann(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute S₀(z) and S₁(z) over the input array z.

    S_ν(z) = (π/2) * [H_ν(z) - Y_ν(z)]

    Returns:
        (S0, S1): tuple of numpy arrays
    """
    # Struve functions H₀ and H₁
    H0 = struve(0, z)
    H1 = struve(1, z)

    # Bessel functions of second kind Y₀ and Y₁
    Y0 = yv(0, z)
    Y1 = yv(1, z)

    # Struve-Neumann modes
    S0 = (np.pi / 2.0) * (H0 - Y0)
    S1 = (np.pi / 2.0) * (H1 - Y1)

    return S0, S1

def main():
    print(f"Generating Struve-Neumann LUT: z ∈ [0, {Z_MAX}], N = {N_SAMPLES}")

    # Generate sample points (uniform spacing)
    z = np.linspace(0, Z_MAX, N_SAMPLES, dtype=np.float64)

    # Handle z=0 singularity: S₀(0) = 0, S₁(0) = 0
    z[0] = 1e-12  # Small epsilon to avoid division by zero in special functions

    # Compute S₀ and S₁
    print("Computing S₀(z) and S₁(z)...")
    S0, S1 = compute_struve_neumann(z)

    # Reset z[0] to exact zero
    z[0] = 0.0
    S0[0] = 0.0
    S1[0] = 0.0

    # Verify derivative relationship: S₀'(z) ≈ 1 - S₁(z)
    dS0_dz = np.gradient(S0, z)
    derivative_error = np.abs(dS0_dz - (1.0 - S1))
    max_error = np.max(derivative_error[10:-10])  # Exclude boundaries
    print(f"Derivative identity check: max|S₀'(z) - (1-S₁(z))| = {max_error:.6e}")

    # Export as binary (interleaved f32: [S0[0], S1[0], S0[1], S1[1], ...])
    print(f"Writing binary LUT to {OUTPUT_BIN}...")
    interleaved = np.empty(2 * N_SAMPLES, dtype=np.float32)
    interleaved[0::2] = S0.astype(np.float32)
    interleaved[1::2] = S1.astype(np.float32)

    with open(OUTPUT_BIN, 'wb') as f:
        f.write(interleaved.tobytes())

    # Export as .npy for easy Python reload
    print(f"Writing .npy LUT to {OUTPUT_NPY}...")
    np.save(OUTPUT_NPY, np.stack([S0, S1], axis=0).astype(np.float32))

    # Export metadata
    config = {
        "z_max": Z_MAX,
        "n_samples": N_SAMPLES,
        "dz": Z_MAX / (N_SAMPLES - 1),
        "format": "interleaved_f32",
        "layout": "[S0[0], S1[0], S0[1], S1[1], ...]"
    }

    print(f"Writing config to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n✓ LUT generation complete!")
    print(f"  - Binary: {OUTPUT_BIN} ({2*N_SAMPLES*4} bytes)")
    print(f"  - NumPy:  {OUTPUT_NPY}")
    print(f"  - Config: {OUTPUT_JSON}")
    print(f"\nSample values:")
    for i in [0, N_SAMPLES//4, N_SAMPLES//2, 3*N_SAMPLES//4, N_SAMPLES-1]:
        print(f"  z={z[i]:6.2f}: S₀={S0[i]:10.6f}, S₁={S1[i]:10.6f}")

if __name__ == "__main__":
    main()
