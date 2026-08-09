"""Exact terminal recovery, finite-noise non-identifiability, and filters.

The output is a deterministic JSON artifact consumed by the paper. Exact
claims use SymPy rationals. Floating-point values appear only in the explicit
Chebyshev stress test and are regenerated from tracked input parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.polynomial import Chebyshev, Polynomial


def rational_string(value: sp.Expr) -> str:
    return str(sp.factor(value))


def exact_flat_extension() -> dict[str, object]:
    atoms = [sp.Rational(1, 5), sp.Rational(1, 2), sp.Rational(4, 5)]
    transfer = sp.diag(*atoms)
    probes = sp.Matrix([[1, 0], [1, 1], [0, 1]])
    moments = [sp.simplify(probes.T * (transfer**n) * probes) for n in range(6)]

    def block_hankel(degree: int, shift: int = 0) -> sp.Matrix:
        return sp.Matrix.vstack(
            *[
                sp.Matrix.hstack(*[moments[i + j + shift] for j in range(degree + 1)])
                for i in range(degree + 1)
            ]
        )

    h1, h2 = block_hankel(1), block_hankel(2)
    g1 = block_hankel(1, shift=1)
    pivots = list(h1.rref()[1])
    hq = h1.extract(pivots, pivots)
    gq = g1.extract(pivots, pivots)
    variable = sp.Symbol("lambda")
    pencil = sp.factor((gq - variable * hq).det())
    recovered = sorted(sp.solve(sp.Eq(pencil, 0), variable))

    return {
        "atoms": [rational_string(value) for value in atoms],
        "probe_matrix": [[str(value) for value in row] for row in probes.tolist()],
        "moments_B0_to_B5": [
            [[rational_string(value) for value in row] for row in moment.tolist()]
            for moment in moments
        ],
        "rank_H0": int(block_hankel(0).rank()),
        "rank_H1": int(h1.rank()),
        "rank_H2": int(h2.rank()),
        "flat_at_degree_1": bool(h1.rank() == h2.rank()),
        "quotient_pivot_indices": pivots,
        "generalized_characteristic_polynomial": rational_string(pencil),
        "recovered_visible_atoms": [rational_string(value) for value in recovered],
        "recovery_exact": recovered == atoms,
        "recovered_visible_edge": rational_string(max(recovered)),
        "recovered_visible_mass": rational_string(-sp.log(max(recovered))),
    }


def hidden_atom_indistinguishability(
    max_moment: int = 40, noise_radius: float = 1e-8
) -> dict[str, object]:
    atoms = np.array([0.25, 0.55, 0.72])
    weights = np.array([0.2, 0.5, 0.3])
    base = np.array([np.sum(weights * atoms**n) for n in range(max_moment + 1)])
    contaminated = (1.0 - noise_radius) * base + noise_radius
    differences = np.abs(contaminated - base)
    return {
        "base_atoms": atoms.tolist(),
        "base_weights": weights.tolist(),
        "hidden_atom_location": 1.0,
        "hidden_atom_weight": noise_radius,
        "max_moment": max_moment,
        "max_absolute_moment_change": float(differences.max()),
        "within_componentwise_noise_radius": bool(
            np.all(differences <= noise_radius * (1 + 1e-12))
        ),
        "base_transfer_edge": float(atoms.max()),
        "contaminated_transfer_edge": 1.0,
        "base_mass_lower_bound": float(-np.log(atoms.max())),
        "contaminated_mass_lower_bound": 0.0,
    }


def chebyshev_visibility_test(
    theta: float = 0.8,
    separation: float = 0.1,
    visible_mass: float = 1e-2,
    moment_error: float = 1e-14,
    max_degree: int = 14,
) -> dict[str, object]:
    """Evaluate the constructive visibility theorem on a hard atomic case."""
    low_atoms = np.array([0.15, 0.45, 0.78])
    low_weights = (1.0 - visible_mass) * np.array([0.2, 0.3, 0.5])
    atoms = np.append(low_atoms, theta + separation)
    weights = np.append(low_weights, visible_mass)
    max_moment = 2 * max_degree + 1
    moments = np.array([np.sum(weights * atoms**n) for n in range(max_moment + 1)])

    rows: list[dict[str, object]] = []
    first_analytic_degree = None
    first_robust_visibility_degree = None
    first_noise_safe_degree = None
    for degree in range(max_degree + 1):
        basis = Chebyshev.basis(degree, domain=[0.0, theta])
        polynomial = basis.convert(kind=Polynomial)
        normalizer = float(polynomial(theta + separation))
        polynomial = Polynomial(polynomial.coef / normalizer)
        coefficients = polynomial.coef
        squared = np.convolve(coefficients, coefficients)
        exact_localizer = float(
            theta * np.dot(squared, moments[: len(squared)])
            - np.dot(squared, moments[1 : len(squared) + 1])
        )
        alpha = 1.0 / abs(normalizer)
        analytic_upper_bound = theta * alpha**2 - separation * visible_mass
        error_radius = (1.0 + theta) * moment_error * float(
            np.sum(np.abs(coefficients)) ** 2
        )
        analytic_detects = analytic_upper_bound < 0.0
        robust_visibility_upper_bound = analytic_upper_bound + 2.0 * error_radius
        realized_measure_upper_bound = exact_localizer + 2.0 * error_radius
        robust_visibility_detects = robust_visibility_upper_bound < 0.0
        noise_safe = realized_measure_upper_bound < 0.0
        if analytic_detects and first_analytic_degree is None:
            first_analytic_degree = degree
        if robust_visibility_detects and first_robust_visibility_degree is None:
            first_robust_visibility_degree = degree
        if noise_safe and first_noise_safe_degree is None:
            first_noise_safe_degree = degree
        rows.append(
            {
                "degree": degree,
                "chebyshev_normalizer": normalizer,
                "stopband_alpha": alpha,
                "analytic_localizer_upper_bound": analytic_upper_bound,
                "actual_localizer_value": exact_localizer,
                "componentwise_moment_error": moment_error,
                "worst_case_polynomial_error_radius": error_radius,
                "analytic_bound_detects_outlier": analytic_detects,
                "robust_visibility_certificate_upper_bound": robust_visibility_upper_bound,
                "robust_visibility_bound_detects_outlier": robust_visibility_detects,
                "realized_measure_certificate_upper_bound": realized_measure_upper_bound,
                "certificate_survives_moment_error": noise_safe,
            }
        )

    return {
        "claimed_edge_theta": theta,
        "outlier_separation_delta": separation,
        "minimum_visible_mass_gamma": visible_mass,
        "atoms": atoms.tolist(),
        "weights": weights.tolist(),
        "max_degree": max_degree,
        "first_degree_from_analytic_bound": first_analytic_degree,
        "first_degree_from_robust_visibility_bound": first_robust_visibility_degree,
        "first_degree_with_noise_safe_certificate": first_noise_safe_degree,
        "degrees": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("certificate_artifacts.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = {
        "exact_flat_extension": exact_flat_extension(),
        "finite_noise_hidden_atom": hidden_atom_indistinguishability(),
        "chebyshev_visibility": chebyshev_visibility_test(),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"artifact mismatch: regenerate {args.output}")
        print(f"artifact verified: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")


if __name__ == "__main__":
    main()
