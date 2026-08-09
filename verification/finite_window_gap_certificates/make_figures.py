"""Regenerate manuscript figures from tracked local and Colab artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    local = json.loads((HERE / "annni_local_results.json").read_text())
    colab = json.loads((HERE / "colab_run_summary.json").read_text())

    target = next(result for result in local["results"] if result["length"] == 12)
    rows = target["finite_window"]
    degrees = np.array([row["degree"] for row in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(degrees, [row["x_only_ritz"] for row in rows], "o-", label="even $X$ probe")
    ax.plot(degrees, [row["block_ritz"] for row in rows], "s-", label="$(X,Z)$ block probe")
    ax.axhline(target["true_transfer_edge"], color="black", linestyle="--", label="exact visible edge")
    ax.axhline(target["claimed_transfer_edge"], color="#b51f1f", linestyle=":", label="false claimed edge")
    ax.set(xlabel="Krylov degree $N$", ylabel="largest Ritz transfer eigenvalue")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "annni_ritz_convergence")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(degrees, [row["x_only_false_gap_localizer_min"] for row in rows], "o-", label="even $X$ probe")
    ax.plot(degrees, [row["block_false_gap_localizer_min"] for row in rows], "s-", label="$(X,Z)$ block probe")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(xlabel="localizer degree $N$", ylabel="smallest eigenvalue of $\\theta H_N-G_N$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "annni_localizer_witness")

    combined: dict[int, dict[str, float]] = {}
    for result in local["results"]:
        combined[result["length"]] = {
            "gap": result["true_gap"],
            "error": abs(result["finite_window"][-1]["block_ritz"] - result["true_transfer_edge"]),
        }
    for result in colab["summary"]:
        combined[result["L"]] = {"gap": result["true_gap"], "error": result["block_N6_abs_error"]}
    lengths = np.array(sorted(combined))
    gaps = np.array([combined[length]["gap"] for length in lengths])
    errors = np.array([combined[length]["error"] for length in lengths])

    fig, left = plt.subplots(figsize=(6.4, 4.0))
    right = left.twinx()
    left.plot(lengths, gaps, "o-", color="#1f5aa6", label="exact finite-volume gap")
    right.semilogy(lengths, errors, "s--", color="#b54b1f", label="$N=6$ edge error")
    left.set(xlabel="chain length $L$", ylabel="energy gap")
    right.set_ylabel("absolute transfer-edge error")
    left.grid(alpha=0.25)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center right")
    save_figure(fig, "annni_scaling")


if __name__ == "__main__":
    main()
