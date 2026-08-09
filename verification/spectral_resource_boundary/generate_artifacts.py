"""Generate deterministic certificates for the spectral-resource-boundary paper.

The exact layer uses only fractions.Fraction.  Floating-point arithmetic is
restricted to the continuous-time asymptotic slope check, where the tracked
artifact records both the fitted value and its analytic target.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_bridge.json"


def frac(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def moment(atoms: list[Fraction], weights: list[Fraction], n: int) -> Fraction:
    return sum((w * r**n for r, w in zip(atoms, weights)), start=Fraction(0))


def moment_matrix(moments: list[Fraction], order: int) -> list[list[Fraction]]:
    return [[moments[i + j] for j in range(order + 1)] for i in range(order + 1)]


def localizing_matrix(
    moments: list[Fraction], theta: Fraction, order: int
) -> list[list[Fraction]]:
    return [
        [theta * moments[i + j] - moments[i + j + 1] for j in range(order + 1)]
        for i in range(order + 1)
    ]


def quadratic_form(matrix: list[list[Fraction]], vector: list[Fraction]) -> Fraction:
    return sum(
        (
            vector[i] * matrix[i][j] * vector[j]
            for i in range(len(vector))
            for j in range(len(vector))
        ),
        start=Fraction(0),
    )


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom
    return slope, ybar - slope * xbar


def build_artifact() -> dict[str, object]:
    # Transfer spectrum below the vacuum eigenvalue 1.  The weights sum to 1.
    atoms = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 5)]
    weights = [Fraction(2, 7), Fraction(3, 7), Fraction(2, 7)]
    moments = [moment(atoms, weights, n) for n in range(7)]
    order = 2
    hankel = moment_matrix(moments, order)

    theta_good = Fraction(1, 2)
    theta_bad = Fraction(2, 5)
    local_good = localizing_matrix(moments, theta_good, order)
    local_bad = localizing_matrix(moments, theta_bad, order)

    # p(v) = 1 - 8v + 15v^2 vanishes at 1/3 and 1/5, but not at 1/2.
    witness = [Fraction(1), Fraction(-8), Fraction(15)]
    witness_value = quadratic_form(local_bad, witness)
    assert witness_value < 0

    # Exact positivity decompositions: each matrix is a sum of rank-one Gram terms.
    good_coefficients = [w * (theta_good - r) for r, w in zip(atoms, weights)]
    assert all(c >= 0 for c in good_coefficients)

    # Continuous spectral model H = diag(0, 2, 3, 5) on the excited sector.
    energies = [2.0, 3.0, 5.0]
    float_weights = [float(w) for w in weights]

    def correlation(t: float) -> float:
        return sum(w * math.exp(-energy * t) for w, energy in zip(float_weights, energies))

    grid = [i / 4 for i in range(0, 33)]
    correlations = [correlation(t) for t in grid]
    gap_envelope_margins = [math.exp(-2.0 * t) - c for t, c in zip(grid, correlations)]
    assert min(gap_envelope_margins) >= -2e-16

    false_gap = 2.1
    false_gap_time = 30.0
    false_gap_ratio = correlation(false_gap_time) / math.exp(-false_gap * false_gap_time)
    assert false_gap_ratio > 1.0

    fit_times = [20.0 + i for i in range(21)]
    log_rates = [2.0 * math.log(correlation(t)) for t in fit_times]
    fitted_slope, fitted_intercept = linear_fit(fit_times, log_rates)
    analytic_rate_slope = -4.0
    assert abs(fitted_slope - analytic_rate_slope) < 1e-8

    # At x = 1 the Stieltjes resolvent is exact for this rational spectrum.
    exact_energies = [Fraction(2), Fraction(3), Fraction(5)]
    x = Fraction(1)
    resolvent = sum(
        (w / (energy + x) for w, energy in zip(weights, exact_energies)),
        start=Fraction(0),
    )
    resolvent_bound = Fraction(1, 3)  # ||psi||^2 / (m + x), m = 2.
    assert resolvent <= resolvent_bound

    # Two-sided operational horizon in an exactly exponential toy lane.
    budget = 1e-6
    rate_exponent = 4.0
    a_minus = 1.0
    a_plus = 4.0
    impossible_horizon = math.log(a_minus / budget) / rate_exponent
    affordable_horizon = math.log(a_plus / budget) / rate_exponent

    return {
        "schema_version": 1,
        "claim_scope": (
            "Exact finite spectral/moment certificates and deterministic toy-model "
            "checks only; not a Yang-Mills mass-gap or Riemann-hypothesis proof."
        ),
        "transfer_moment_certificate": {
            "atoms": [frac(v) for v in atoms],
            "weights": [frac(v) for v in weights],
            "moments_b0_to_b6": [frac(v) for v in moments],
            "hankel_H2": [[frac(v) for v in row] for row in hankel],
            "theta_good": frac(theta_good),
            "localizing_L2_theta_good": [[frac(v) for v in row] for row in local_good],
            "gram_coefficients_theta_good": [frac(v) for v in good_coefficients],
            "theta_bad": frac(theta_bad),
            "localizing_L2_theta_bad": [[frac(v) for v in row] for row in local_bad],
            "negative_witness": [frac(v) for v in witness],
            "negative_witness_value": frac(witness_value),
            "status_good": "PSD_BY_EXACT_ATOMIC_GRAM_DECOMPOSITION",
            "status_bad": "NOT_PSD_BY_EXACT_RATIONAL_WITNESS",
        },
        "continuous_spectral_triangle": {
            "energies": energies,
            "weights": float_weights,
            "true_gap": 2.0,
            "minimum_gap_envelope_margin_on_grid": min(gap_envelope_margins),
            "false_gap_claim": false_gap,
            "false_gap_test_time": false_gap_time,
            "false_gap_violation_ratio": false_gap_ratio,
            "resolvent_x": frac(x),
            "resolvent_value": frac(resolvent),
            "resolvent_gap_bound": frac(resolvent_bound),
        },
        "rate_squaring_check": {
            "fit_time_start": fit_times[0],
            "fit_time_end": fit_times[-1],
            "fitted_log_rate_slope": fitted_slope,
            "fitted_intercept": fitted_intercept,
            "analytic_log_rate_slope": analytic_rate_slope,
            "absolute_slope_error": abs(fitted_slope - analytic_rate_slope),
        },
        "two_horizon_example": {
            "budget": budget,
            "rate_exponent": rate_exponent,
            "lower_power_factor": a_minus,
            "upper_power_factor": a_plus,
            "impossibility_horizon": impossible_horizon,
            "affordability_horizon": affordable_horizon,
            "uncertainty_width": affordable_horizon - impossible_horizon,
        },
    }


def encoded_artifact() -> bytes:
    return (json.dumps(build_artifact(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = encoded_artifact()
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != payload:
            raise SystemExit(f"artifact drift: regenerate {args.output}")
        print(f"PASS deterministic artifact: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
