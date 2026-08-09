"""Finite-window gap certificates in an interacting ANNNI spin chain.

The experiment deliberately keeps the transfer operator implicit.  It builds
Euclidean correlation matrices

    B_n[a,b] = <psi_a, exp(-n*tau*(H-E0)) psi_b>

and uses only these matrices to construct block Hankel and localizing pencils.
The parity-even X probe is blind to the lowest parity-odd excitation, whereas
the two-channel (X,Z) family sees it.  This is the concrete blindness mechanism
behind the finite-window identifiability theorem in the accompanying paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply


I2 = sparse.csr_matrix(np.eye(2, dtype=float))
X2 = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
Z2 = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))


def site_operator(op: sparse.csr_matrix, site: int, length: int) -> sparse.csr_matrix:
    result = sparse.csr_matrix([[1.0]])
    for j in range(length):
        result = sparse.kron(result, op if j == site else I2, format="csr")
    return result


def product_operator(
    factors: dict[int, sparse.csr_matrix], length: int
) -> sparse.csr_matrix:
    result = sparse.csr_matrix([[1.0]])
    for j in range(length):
        result = sparse.kron(result, factors.get(j, I2), format="csr")
    return result


def annni_hamiltonian(length: int, j1: float, j2: float, hx: float) -> sparse.csr_matrix:
    """Open-boundary ANNNI Hamiltonian preserving global spin-flip parity."""
    dim = 1 << length
    hamiltonian = sparse.csr_matrix((dim, dim), dtype=float)
    for j in range(length - 1):
        hamiltonian -= j1 * product_operator({j: Z2, j + 1: Z2}, length)
    for j in range(length - 2):
        hamiltonian -= j2 * product_operator({j: Z2, j + 2: Z2}, length)
    for j in range(length):
        hamiltonian -= hx * site_operator(X2, j, length)
    return hamiltonian


def centered_probe(
    ground: np.ndarray, operator: sparse.csr_matrix
) -> np.ndarray:
    expectation = float(np.vdot(ground, operator @ ground).real)
    return np.asarray(operator @ ground - expectation * ground)


def correlation_moments(
    hamiltonian: sparse.csr_matrix,
    ground_energy: float,
    probes: np.ndarray,
    tau: float,
    max_moment: int,
) -> np.ndarray:
    """Return B_0,...,B_max using sparse imaginary-time propagation."""
    dim = hamiltonian.shape[0]
    shifted_generator = -(hamiltonian - ground_energy * sparse.eye(dim, format="csr"))
    evolved = expm_multiply(
        shifted_generator,
        probes,
        start=0.0,
        stop=tau * max_moment,
        num=max_moment + 1,
        endpoint=True,
        traceA=ground_energy * dim,
    )
    moments = np.empty((max_moment + 1, probes.shape[1], probes.shape[1]))
    for n in range(max_moment + 1):
        moments[n] = np.real_if_close(probes.conj().T @ evolved[n]).real
        moments[n] = (moments[n] + moments[n].T) / 2
    return moments


def block_hankel(moments: np.ndarray, degree: int, shift: int = 0) -> np.ndarray:
    d = moments.shape[1]
    answer = np.empty(((degree + 1) * d, (degree + 1) * d))
    for i in range(degree + 1):
        for j in range(degree + 1):
            answer[i * d : (i + 1) * d, j * d : (j + 1) * d] = moments[
                i + j + shift
            ]
    return (answer + answer.T) / 2


def largest_ritz(moments: np.ndarray, degree: int, rtol: float = 1e-11) -> tuple[float, int]:
    hankel = block_hankel(moments, degree)
    shifted = block_hankel(moments, degree, shift=1)
    values, vectors = np.linalg.eigh(hankel)
    cutoff = rtol * max(1.0, float(values[-1]))
    keep = values > cutoff
    whitening = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    compressed = whitening.T @ shifted @ whitening
    return float(np.linalg.eigvalsh((compressed + compressed.T) / 2)[-1]), int(keep.sum())


def localizer_minimum(moments: np.ndarray, degree: int, edge_claim: float) -> float:
    hankel = block_hankel(moments, degree)
    shifted = block_hankel(moments, degree, shift=1)
    localizer = edge_claim * hankel - shifted
    return float(np.linalg.eigvalsh((localizer + localizer.T) / 2)[0])


def analyze_length(
    length: int,
    max_degree: int,
    tau: float,
    j1: float,
    j2: float,
    hx: float,
    false_gap_factor: float,
) -> dict[str, object]:
    hamiltonian = annni_hamiltonian(length, j1, j2, hx)
    energies, states = eigsh(hamiltonian, k=6, which="SA", tol=1e-12)
    order = np.argsort(energies)
    energies, states = energies[order], states[:, order]
    ground = states[:, 0]

    center = length // 2
    x_probe = centered_probe(ground, site_operator(X2, center, length))
    z_probe = centered_probe(ground, site_operator(Z2, center, length))
    probe_matrix = np.column_stack([x_probe, z_probe])

    parity = product_operator({j: X2 for j in range(length)}, length)
    parities = [float(np.vdot(states[:, k], parity @ states[:, k]).real) for k in range(6)]
    overlaps = np.abs(states[:, 1:6].conj().T @ probe_matrix) ** 2

    max_moment = 2 * max_degree + 1
    matrix_moments = correlation_moments(
        hamiltonian, float(energies[0]), probe_matrix, tau, max_moment
    )
    x_moments = matrix_moments[:, :1, :1]

    gap = float(energies[1] - energies[0])
    false_gap = false_gap_factor * gap
    true_edge = float(np.exp(-tau * gap))
    claimed_edge = float(np.exp(-tau * false_gap))

    rows: list[dict[str, object]] = []
    for degree in range(max_degree + 1):
        x_ritz, x_rank = largest_ritz(x_moments, degree)
        block_ritz, block_rank = largest_ritz(matrix_moments, degree)
        rows.append(
            {
                "degree": degree,
                "x_only_ritz": x_ritz,
                "block_ritz": block_ritz,
                "x_only_hankel_rank": x_rank,
                "block_hankel_rank": block_rank,
                "x_only_false_gap_localizer_min": localizer_minimum(
                    x_moments, degree, claimed_edge
                ),
                "block_false_gap_localizer_min": localizer_minimum(
                    matrix_moments, degree, claimed_edge
                ),
            }
        )

    return {
        "length": length,
        "parameters": {"j1": j1, "j2": j2, "hx": hx, "tau": tau},
        "energies": [float(value) for value in energies],
        "low_state_parities": parities,
        "probe_overlap_weights": {
            "rows_are_excited_states_1_to_5": overlaps.tolist(),
            "columns": ["X_center_even", "Z_center_odd"],
        },
        "true_gap": gap,
        "true_transfer_edge": true_edge,
        "false_gap_factor": false_gap_factor,
        "false_gap_claim": false_gap,
        "claimed_transfer_edge": claimed_edge,
        "finite_window": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--j2", type=float, default=0.37)
    parser.add_argument("--hx", type=float, default=2.2)
    parser.add_argument("--false-gap-factor", type=float, default=1.35)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("annni_results.json"),
    )
    args = parser.parse_args()

    results = [
        analyze_length(
            length,
            args.max_degree,
            args.tau,
            args.j1,
            args.j2,
            args.hx,
            args.false_gap_factor,
        )
        for length in args.lengths
    ]
    payload = {
        "model": "open-boundary quantum ANNNI chain",
        "convention": "H=-J1 sum ZZ-J2 sum Z_i Z_(i+2)-hx sum X",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
