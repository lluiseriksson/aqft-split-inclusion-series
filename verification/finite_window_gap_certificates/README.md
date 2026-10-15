# Finite-window gap certificates

This directory contains the frontier verification layer for the finite-window
gap-identifiability paper.

`spin_chain_experiment.py` constructs an interacting open-boundary quantum
ANNNI chain, computes low energies independently, and then forgets the
Hamiltonian: the certificate layer receives only Euclidean correlation
matrices.  It compares a parity-even probe, which is exactly blind to the
lowest parity-odd state, with the two-channel `(X,Z)` family.

Quick run:

```bash
python verification/finite_window_gap_certificates/spin_chain_experiment.py \
  --lengths 6 8 10 12 --max-degree 6
```

The tracked large run is executed in Colab with the same script and includes
larger Hilbert spaces.  The resulting JSON records Hamiltonian parameters,
independent low-energy benchmarks, symmetry parities, overlap weights, Hankel
ranks, Ritz edges, and localizer witnesses at every window depth.

`exact_and_robust_certificates.py` supplies three independent deterministic
checks: exact rational block-Hankel flatness and visible-spectrum recovery; the
hidden-atom construction proving finite-noise non-identifiability; and the
explicit Chebyshev visibility/noise tradeoff.

```bash
python verification/finite_window_gap_certificates/exact_and_robust_certificates.py
python verification/finite_window_gap_certificates/exact_and_robust_certificates.py --check
```

To audit every theorem-level claim used by the paper (including the tracked
local and Colab ANNNI records) in one command, run:

```bash
python verification/finite_window_gap_certificates/verify_all.py
```

The large Colab run is recorded in `colab_run_summary.json`, including the
source commit, numerical environment, full-result SHA-256, and the complete
summary printed by the notebook. `make_figures.py` regenerates every plot used
by the manuscript from the tracked artifacts.
