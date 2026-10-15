"""Independent exact re-check of the tracked spectral bridge artifact."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "spectral_bridge.json"


def f(value: str) -> Fraction:
    return Fraction(value)


def qform(matrix: list[list[Fraction]], vector: list[Fraction]) -> Fraction:
    return sum(
        (
            vector[i] * matrix[i][j] * vector[j]
            for i in range(len(vector))
            for j in range(len(vector))
        ),
        start=Fraction(0),
    )


def main() -> None:
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    cert = data["transfer_moment_certificate"]
    atoms = [f(x) for x in cert["atoms"]]
    weights = [f(x) for x in cert["weights"]]
    moments = [f(x) for x in cert["moments_b0_to_b6"]]
    assert sum(weights) == 1
    for n, claimed in enumerate(moments):
        actual = sum((w * r**n for r, w in zip(atoms, weights)), start=Fraction(0))
        assert actual == claimed

    hankel = [[f(x) for x in row] for row in cert["hankel_H2"]]
    for i in range(3):
        for j in range(3):
            assert hankel[i][j] == moments[i + j]

    theta_good = f(cert["theta_good"])
    local_good = [[f(x) for x in row] for row in cert["localizing_L2_theta_good"]]
    gram_coefficients = [f(x) for x in cert["gram_coefficients_theta_good"]]
    assert gram_coefficients == [w * (theta_good - r) for r, w in zip(atoms, weights)]
    assert all(c >= 0 for c in gram_coefficients)
    for i in range(3):
        for j in range(3):
            reconstructed = sum(
                (c * r ** (i + j) for c, r in zip(gram_coefficients, atoms)),
                start=Fraction(0),
            )
            assert local_good[i][j] == reconstructed

    theta_bad = f(cert["theta_bad"])
    local_bad = [[f(x) for x in row] for row in cert["localizing_L2_theta_bad"]]
    for i in range(3):
        for j in range(3):
            assert local_bad[i][j] == theta_bad * moments[i + j] - moments[i + j + 1]
    witness = [f(x) for x in cert["negative_witness"]]
    witness_value = f(cert["negative_witness_value"])
    assert qform(local_bad, witness) == witness_value < 0

    continuous = data["continuous_spectral_triangle"]
    assert f(continuous["resolvent_value"]) <= f(continuous["resolvent_gap_bound"])
    assert continuous["false_gap_violation_ratio"] > 1.0
    rate = data["rate_squaring_check"]
    assert rate["absolute_slope_error"] < 1e-8
    horizons = data["two_horizon_example"]
    assert horizons["impossibility_horizon"] < horizons["affordability_horizon"]
    print("PASS independent exact and numerical artifact verification")


if __name__ == "__main__":
    main()
