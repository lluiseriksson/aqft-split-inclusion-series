# AQFT / Operational Coherence Series — five v2 papers + exact verification suites

Author: Lluis Eriksson (lluiseriksson@gmail.com). July 2026 (v2 revisions); v1s: December 2025.

Five companion papers that cite one another (ai.viXra 2512 series), each at
version 2, with every correction driven or confirmed by exact numerical
verification. Scripts and reference logs are included; nothing in the checks
relies on the approximations being tested.

| Paper | Directory | Verification | Role in the series |
|---|---|---|---|
| **0060** — *Clustering, Recovery, and Locality in AQFT* | `papers/0060-clustering-recovery/` | `verification/0060/` | Kinematics: split inclusions, collar-suppressed vacuum correlations (Bessel/Combes–Thomas), reconstruction-fidelity theorem |
| **0061** — *The Conditional Maintenance Work Theorem* | `papers/0061-maintenance-work/` | `verification/0061/` | Thermodynamics: battery-accounted power floors for maintaining states/coherence; Type III blueprint uses 0060's split machinery |
| **0064** — *The Heisenberg Cut as a Resource Boundary* | `papers/0064-heisenberg-cut/` | `verification/0064/` | Dynamics/interpretation: exact rate inheritance through gapped buffers; grounds its maintenance inequality in 0061 (v2) and its geometry in 0060 |
| **0070** — *Stress Testing the Rate Inheritance Principle* | `papers/0070-rate-inheritance/` | `verification/0070/` | Stress test: Davies-generator regime where rate inheritance fails; ceiling/floor envelopes; resource-horizon no-go |
| **0071** — *Operational Coherence Maintenance (closure note)* | `papers/0071-program-closure/` | none of its own — its verification is *inherited* from the 0060/0061/0064/0070 suites | The map: proved core / conditional interfaces / hinge status, updated to the v2 series. Upload last: it is the closure map of the other four |

Cross-citation graph: 0060 ⇄ 0061, 0060 ⇄ 0064, 0061 ⇄ 0064, 0070 → {0060, 0061, 0064}, 0064 → {0070, 0071}, 0071 → {0060, 0061, 0064, 0070} (each bibliography
points to the others' directories in this repository). The same Combes–Thomas
exponent appears in all three: `arccosh(1+m²/2)` for the 0060 lattice vacuum,
`arccosh(μ/2t)` in the 0064 oracle, `cosh q(ω) = (μ²+4−ω²)/(4μ)` in 0064's
frequency-resolved rate law.

## What each v2 fixes (all verified here)

**0060**: (1) v1's Prop. 2.14 misidentified its Gaussian reconstruction with the
Petz map (for a pure reference the Petz map returns the reference for *every*
input); restated for the conditional reattachment with a no-steering
admissibility hypothesis, which numerics show is sharp. (2) Constant 3/8 → 7/16.
(3) Symplectic-gap hypothesis added to the fidelity lemma (vacuum-vs-thermal
counterexample). Key numbers: repaired bound passes at every admissible point
(margins 46–68×); 1916 random gapped fidelity pairs, 0 violations; lattice
η_vac rate 0.954 vs analytic 0.962.

**0061**: (1) Work-sign error (v1's Appendix A flipped its final inequality;
numerically W−ΔF_S reaches −1.12 with v1's sign, min +0.48 with the corrected
consumption sign over 200 random energy-conserving unitaries). (2) v1's
"assumption-free" extra-power definition was vacuous (unlinked-pairs infimum =
−∞); replaced by ε-efficient baselines (proved), difference-of-infima under
explicit Landauer achievability (conditional), and a genuinely assumption-free
free-baseline case (pure dephasing). (3) New unconditional floor
P_min ≥ k_B T σ(ρ) via Spohn, with the rate-Pythagoras split verified to 1e−13.

**0064** (v2 received; revised here): rate inheritance proved exactly in the
quasi-free class, κ(ε) = P(ε)e^(−2q(ω_b)ε), slopes matching the analytic law
(fitted 0.8106 vs 0.8109 at ω₀=0; validation of the covariance solver at
1e−16/1e−15); v1's windowed-proxy evidence withdrawn after a critical-point
control. Revision here: the maintenance inequality is now grounded in 0061
(v2)'s free-baseline theorem (pointer-basis dephasing is exactly that case),
companion references updated to v2, contact email added, and script paths made
portable.

**0070** (v2 written here, replaces v1): (1) v1's envelope κ(ε) is a *supremum*
(ceiling) but its no-go needs a *floor*; both are now defined and computed —
they genuinely differ (σ^x: ceiling falls 3.7→2.3 while the floor rises;
σ^z: ceiling saturates ~1.1 **and** the floor persists ~0.17, which is what
rescues the corrected no-go within the model class). (2) New Davies-locality
caveat: the ω≈0 Bohr component of the local coupling carries ~90% of its
weight beyond the coupling site at N=6 — the saturation is a property of the
(nonlocal) secular model class, bracketed against 0064's exact local-sink
suppression. (3) v1's figure shipped as a compile-time placeholder
(`nedladdning2.png`); replaced by script-generated table+figure. (4) Imported
maintenance inequality re-anchored to 0061 v2.

**0071** (v2 written here, replaces v1): the closure note's imported theorem
reproduced the vacuous v1 form of the 0061 extra-power bound — restated per
0061 v2 (unconditional entropy-production floor; efficient baselines;
free-baseline case). The rate-inheritance hinge is updated from "open" to
"partially resolved": proved in the quasi-free local-sink class (0064 v2),
failure scenario realized within the Davies class (0070 v2, with the secular
nonlocality caveat), open exactly for interacting gapped buffers. A new
proxy-discipline remark distills the windowed-proxy withdrawal and the
ceiling/floor correction. 0070 added to its references (absent in v1).

## Hardening round (pre-upload review, applied)

All four papers passed an external pre-upload review; the following surgical
patches are incorporated in the shipped versions. **0060**: the C_k/6 form of
the fidelity lemma now carries its smallness condition (C_k <= 2 or
C_k ||K||^2 <= 2; the main theorem always uses the safe two-term bound), and
the pure-reference Petz counterexample is stated with support/Moore-Penrose /
faithful-regularization semantics. **0061**: the geometric interface
assumptions are now two-sided (upper bounds alone cannot yield the "in
particular" lower scalings); the free-baseline proposition explicitly assumes
Delta[L(rho)] = 0 alongside L(Delta[rho]) = 0; the battery-burning remark
notes the dense-spectrum work-sink proviso. **0064**: "established exactly" is
rephrased as "derived in an exactly solvable quasi-free model and verified
against exact Liouvillian rapidities"; "cannot distinguish rho from
Delta[rho]" is rephrased as "cannot maintain, exploit, or reverse the
coherence over the relevant operational timescale"; author name unified.
**0070**: numerical floors are explicitly *projected* envelopes on the tested
Pauli/nearest-neighbour subspace; the no-go is quadratic-proxy unless the
stated MLSI-type hypothesis holds; a sensitivity remark (N=5, dOmega x2,
kernel threshold 1e-6..1e-10) is included with the extra script
`verification/0070/sensitivity_0070.py`; the delocalization diagnostic
w_{>d}(omega) is given in closed form; the "bracket the physics, not a
contradiction" framing is stated in the introduction.

### Reproducibility micro-patches (post-review)

`verification/0064/modelA.py` now reports the renormalized bound-state
frequency omega_b and both analytic slopes 2q(omega_b) / 2q(omega_0), exactly
reproducing Table 1 of the 0064 paper (e.g. omega_0=0.5: fit 0.6921 vs
2q(omega_b)=0.6921). `verification/0070/sensitivity_0070.py` is now
path-independent (runs from the repo root or from its own directory).

## Run

```
pip install -r requirements.txt
# 0060 (seconds each unless noted)
python verification/0060/gauss_fidelity.py
python verification/0060/verify_covariance.py
python verification/0060/part2_banchi.py
python verification/0060/control_petz.py        # ~1 min
python verification/0060/verify_repaired.py     # exact Fock suite, long
python verification/0060/verify_0060.py         # v1 falsification suite, long
# 0061 (~1 min)
python verification/0061/verify_0061.py
# 0064
python verification/0064/modelA.py              # exact rapidities, ~30 s
python verification/0064/ct_oracle.py           # Combes–Thomas oracle, seconds
python verification/0064/modelB.py              # windowed-proxy control, long (N=10 exact)
# 0070 (~1 min; also regenerates its figure data)
python verification/0070/verify_0070.py
```

Reference outputs and data in `results/`. In the 0060 Fock suite, rows with
`synth_err > 0.05` are superseded by the exact covariance-level scripts.
`verification/0064/brick_P45_combes_thomas.md` is a formalization note
connecting the Combes–Thomas machinery to a Lean brick in a separate project.

## Honesty statement

- **0060**: Theorem 4.7 (v2) proved for the reattachment reconstruction under
  (a)–(g). Numerically supported, not yet proved: reattachment→Petz convergence
  in the strongly-mixed regime (route: Lami–Das–Wilde, J. Phys. A 51 (2018)
  125301). Holographic corollary conditional on Conjecture 5.1.
- **0061**: P_min and efficient-baseline theorems proved in the stroboscopic
  finite-dimensional battery model; difference-of-infima corollary needs
  classical Landauer achievability (explicit); general-E/Type III conditional.
- **0064**: exact rate inheritance proved in the quasi-free class only; the
  interacting-buffer version is an explicit conjecture; the maintenance
  inequality is a theorem in 0061's battery model (free-baseline case) and an
  interface statement beyond it; no Born-rule claims.
- **0070**: all envelopes are exact-diagonalization results *within the Davies
  model class* at N=6 (secular nonlocality quantified in its Section 7); the
  spectral-floor → entropic-rate step is stated as an explicit MLSI-shaped
  hypothesis, not claimed; the no-go theorem's hypothesis is explicit.
