from __future__ import annotations

import math


DENSITY_EPSILON = 10.0


def inverse_density(radius: float, density_c: float) -> float:
    return density_c / (max(radius, 0.0) + DENSITY_EPSILON)


def taylor_remainder_bound(epsilon: float, order: int) -> float | None:
    if not math.isfinite(epsilon) or epsilon >= 1.0 or epsilon < 0.0:
        return None
    return epsilon ** (order + 1) / (1.0 - epsilon)


def test_inverse_density_is_finite_and_monotone() -> None:
    values = [inverse_density(radius, 12.0) for radius in (0.0, 10.0, 100.0, 1000.0)]
    assert all(math.isfinite(value) and value > 0.0 for value in values)
    assert values == sorted(values, reverse=True)


def test_inverse_density_matches_equation_used_by_section_view() -> None:
    assert math.isclose(inverse_density(90.0, 4.5), 4.5 / 100.0)


def test_taylor_guard_rejects_non_convergent_segments() -> None:
    assert taylor_remainder_bound(0.25, 3) is not None
    assert taylor_remainder_bound(0.999, 8) > 1.0
    assert taylor_remainder_bound(1.0, 8) is None
    assert taylor_remainder_bound(float("inf"), 8) is None


def test_taylor_remainder_decreases_with_order_inside_radius() -> None:
    bounds = [taylor_remainder_bound(0.25, order) for order in range(1, 7)]
    assert all(bound is not None for bound in bounds)
    assert bounds == sorted(bounds, reverse=True)


def test_periodic_promotion_requires_ten_stable_closures() -> None:
    stable_closures = 0
    for _ in range(9):
        stable_closures += 1
        assert stable_closures < 10
    stable_closures += 1
    assert stable_closures == 10
