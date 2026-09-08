#!/usr/bin/env python3
"""Independent refined tetrahedral volume integration using SciPy quadrature.

Input JSON: vertices (metres), facets (zero-based outward triangles), mass,
and optional epsilon (metres). The closed mesh must be star-shaped about zero.
This checks angular AND radial discretization, unlike radial ray refinement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre

G = 6.67430e-11


def integrate(vertices, facets, mass, epsilon, position, order, constant):
    nodes, weights = roots_legendre(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    r, u, v = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    wr, wu, wv = np.meshgrid(weights, weights, weights, indexing="ij")
    r, u, v = r.ravel(), u.ravel(), v.ravel()
    weight = (wr * wu * wv).ravel() * r * r * u
    field = np.zeros(4)
    normalization = 0.0
    for facet in facets:
        a, b, c = vertices[facet]
        determinant = float(np.linalg.det(np.column_stack((a, b, c))))
        if determinant <= 0:
            raise ValueError("Facets must face outward from the star-shaped origin")
        point = r[:, None] * ((1 - u[:, None]) * a
                             + (u * (1 - v))[:, None] * b + (u * v)[:, None] * c)
        density = np.ones(len(r)) if constant else np.log1p(np.linalg.norm(point, axis=1) / epsilon)
        dm = determinant * weight * density
        displacement = point - position
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(distance == 0):
            raise ValueError("Target coincides with a quadrature node")
        field[:3] += np.sum(dm[:, None] * displacement / distance[:, None] ** 3, axis=0)
        field[3] += np.sum(dm / distance)
        normalization += np.sum(dm)
    return G * mass * field / normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--position", nargs=3, type=float, required=True)
    parser.add_argument("--orders", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--constant", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.mesh.read_text())
    vertices = np.asarray(data["vertices"], dtype=float)
    facets = np.asarray(data["facets"], dtype=int)
    mass, epsilon = float(data["mass"]), float(data.get("epsilon", 1.0))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Expected finite vertices with three coordinates")
    if facets.ndim != 2 or facets.shape[1] != 3 or np.any(facets < 0) or np.any(facets >= len(vertices)):
        raise ValueError("Invalid triangular facet indices")
    if not np.isfinite([mass, epsilon, *args.position]).all() or mass <= 0 or epsilon <= 0:
        raise ValueError("Invalid mass, epsilon, or target")
    if len(args.orders) < 2 or any(n < 2 for n in args.orders) or args.orders != sorted(set(args.orders)):
        raise ValueError("Specify at least two strictly increasing quadrature orders")
    edges = {}
    for face in facets:
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            count, orientation = edges.get(key, (0, 0))
            edges[key] = count + 1, orientation + (1 if a < b else -1)
    if any(count != 2 or orientation != 0 for count, orientation in edges.values()):
        raise ValueError("Mesh must be closed with consistently oriented facets")
    previous = None
    results = []
    for order in args.orders:
        value = integrate(vertices, facets, mass, epsilon, np.asarray(args.position), order, args.constant)
        results.append({"order": order, "acceleration_mps2": value[:3].tolist(),
                        "positive_potential_m2ps2": float(value[3]),
                        "relative_acceleration_refinement_delta": None if previous is None else
                        float(np.linalg.norm(value[:3] - previous[:3]) / max(np.linalg.norm(value[:3]), 1e-30))})
        previous = value
    print(json.dumps({"density": "constant" if args.constant else "logarithmic",
                      "results": results, "convergence_is_not_an_error_bound": True}, indent=2))


if __name__ == "__main__":
    main()
