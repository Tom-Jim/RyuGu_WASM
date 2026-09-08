#!/usr/bin/env python3
"""Independent continuous-density volume check for radial source cells.

Input is JSON containing either a list of cells or an object with a ``cells``
list. Each cell is ``[dx, dy, dz, solid_angle, r_inner, r_outer, density]``.
The geometry is the same star-shaped angular-cell model used by the browser,
but the calculation is independent: SciPy integrates the continuous Newton
kernel with adaptive Gauss-Kronrod quadrature and reports refinement deltas.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec

G = 6.67430e-11


def load_cells(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    cells = payload.get("cells", payload) if isinstance(payload, dict) else payload
    values = np.asarray(cells, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 7 or not np.isfinite(values).all():
        raise ValueError("expected finite cells with at least 7 columns")
    return values[:, :7]


def integrate_cell(cell: np.ndarray, position: np.ndarray, subdivisions: int) -> np.ndarray:
    direction = cell[:3] / np.linalg.norm(cell[:3])
    solid_angle, inner, outer, density = cell[3:7]
    if solid_angle <= 0 or outer < inner or density < 0:
        raise ValueError("invalid angular/radial cell")
    edges = np.linspace(inner, outer, subdivisions + 1)

    def integrand(radius: float) -> np.ndarray:
        displacement = direction * radius - position
        radius_squared = max(float(np.dot(displacement, displacement)), 1.0e-24)
        volume_weight = solid_angle * radius * radius
        scale = G * density * volume_weight / (radius_squared * math.sqrt(radius_squared))
        potential = G * density * volume_weight / math.sqrt(radius_squared)
        return np.array([scale * displacement[0], scale * displacement[1], scale * displacement[2], potential])

    total = np.zeros(4, dtype=np.float64)
    for left, right in zip(edges[:-1], edges[1:]):
        value, _ = quad_vec(integrand, float(left), float(right), epsabs=1.0e-20, epsrel=1.0e-11)
        total += value
    return total


def evaluate(cells: np.ndarray, position: np.ndarray, subdivisions: int) -> np.ndarray:
    return np.sum([integrate_cell(cell, position, subdivisions) for cell in cells], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", type=Path, help="JSON radial-cell export")
    parser.add_argument("--position", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()
    cells = load_cells(args.cells)
    position = np.asarray(args.position, dtype=np.float64)
    previous = None
    for subdivisions in args.subdivisions:
        if subdivisions < 1:
            raise ValueError("subdivisions must be positive")
        value = evaluate(cells, position, subdivisions)
        delta = None if previous is None else np.linalg.norm(value - previous)
        print(json.dumps({
            "subdivisions": subdivisions,
            "acceleration_mps2": value[:3].tolist(),
            "potential_m2ps2": float(value[3]),
            "refinement_delta": delta,
        }, sort_keys=True))
        previous = value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
