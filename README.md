# AQFT / Operational Coherence Series — twenty-five v2 papers + verification suites

Author: Lluis Eriksson (lluiseriksson@gmail.com). July 2026 (v2 revisions); v1s: December 2025.

Twenty-five companion papers that cite one another (ai.viXra 2512 series), each at
version 2, with corrections backed by scoped numerical or symbolic checks where
applicable. Scripts and reference logs are included; each check's limits are
spelled out in the honesty statement.

| Paper | Directory | Verification | Role in the series |
|---|---|---|---|
| **0060** — *Clustering, Recovery, and Locality in AQFT* | `papers/0060-clustering-recovery/` | `verification/0060/` | Kinematics: split inclusions, collar-suppressed vacuum correlations (Bessel/Combes–Thomas), reconstruction-fidelity theorem |
| **0061** — *The Conditional Maintenance Work Theorem* | `papers/0061-maintenance-work/` | `verification/0061/` | Thermodynamics: battery-accounted power floors for maintaining states/coherence; Type III blueprint uses 0060's split machinery |
| **0064** — *The Heisenberg Cut as a Resource Boundary* | `papers/0064-heisenberg-cut/` | `verification/0064/` | Dynamics/interpretation: exact rate inheritance through gapped buffers; grounds its maintenance inequality in 0061 (v2) and its geometry in 0060 |
| **0070** — *Stress Testing the Rate Inheritance Principle* | `papers/0070-rate-inheritance/` | `verification/0070/` | Stress test: Davies-generator regime where rate inheritance fails; ceiling/floor envelopes; resource-horizon no-go |
| **0071** — *Operational Coherence Maintenance (closure note)* | `papers/0071-program-closure/` | none of its own — its verification is *inherited* from the 0060/0061/0064/0070 suites | The map: proved core / conditional interfaces / hinge status, updated to the v2 series. Upload last: it is the closure map of the others |
| **0072** — *The Rate Inheritance Principle (framing note)* | `papers/0072-rip-principle/` | `verification/0072/` (figure script; evidence inherited from 0064/0070) | Formulates RIP: weak form as a proved lemma, strong form with updated status (derived quasi-free with squared exponent; Davies-class failure) |
| **0073** — *Finite-Dimensional Davies Interface Lemmas and TFIM Witness Tests* | `papers/0073-davies-core/` | `verification/0073/` | The technical core (P7): exact omega=0 identity, corrected modular-weighted Bohr-block identity, witness protocols, linear envelope lemma |
| **0081** — *Operational Coherence Maintenance and the Quantum-Classical Boundary* | `papers/0081-quantum-classical-boundary/` | protocol-level; verification inherited from the 0061/0064/0070/0073 suites (its MCWF stress-test parameters are fully declared in-paper) | The integrative, protocol-facing paper: corrected maintenance hierarchy, RIP status, falsifiable protocols, speculative FEP outlook |
| **0084** — *The Maintenance Constraint: How Resource Boundaries Shape Cognitive Availability* | `papers/0084-maintenance-constraint/` | none — philosophy-of-mind paper with no theorems of its own; its single technical input (the maintenance inequality) is imported as an anchor | The outermost paper: maintainability→availability bridge, maintenance-feasibility bias, FEP/Global-Workspace connections. Explicitly speculative; no phenomenology claims |
| **0085** — *Geometry, Membranes, and Life as a Resource Boundary* | `papers/0085-geometry-membranes-life/` | `verification/0085/` (two checkable claims: fidelity lemma + Davies upper-envelope direction) | The synthesis/pipeline paper: composes the whole program (maintenance law → static leakage → CMI recoverability → Davies interface → ceiling/floor hinge). Anticipated the κ↑/κ↓ distinction. Speculative "life" framing, operational only |
| **0091** — *Technical Appendix: Heat Kernel, Fermions, and the Sign of Induced Gravity* | `papers/0091-heat-kernel-induced-gravity/` | `verification/0091/` (symbolic check of all 7 Seeley–DeWitt coefficients + induced-Newton sign) | Topically-adjacent support piece (induced-gravity sign bookkeeping), not part of the coherence core. **The only paper in the program with no error in v1** — v2 is enhancement (verification + updated refs) |
| **0101** — *Beyond Gaussianity: Extending the Clustering–Recovery Bridge* | `papers/0101-beyond-gaussianity/` | `verification/0101/` (Markov-product identity; full Petz slope table; rotated-Petz subset + crossover figure) | Direct companion of 0060: non-Gaussian extension of its Conjecture 5.1 (collar geometry, Fawzi–Renner CMI route, TFIM Gibbs numerics). v2 corrects the normalized Markov-product identity (missing log Z, Lieb) |
| **0102** — *Heat Kernel Methods and the Sign of Induced Gravity* | `papers/0102-induced-gravity-sign/` | `verification/0102/` (exact S^4 spectra: scalar, Dirac, Hodge 1-forms; A1 table + induced-Newton sign; 11/11) | Conventions companion of 0091: where the Seeley–DeWitt a1 coefficients are fixed and verified against exact S^4 spectra, anchored to the classical counting 1/G_ind ∝ N0 + 2N_{1/2} − 4N1. Complementary to 0091 (compact bookkeeping), not redundant |
| **0105** — *Prefix-Path Bell Transport on IBM Quantum Hardware* | `papers/0105-prefix-path-bell-transport/` | `verification/0105/` (`--demo` validates the offline re-analysis code; raw on-device JSONs are pending data revision) | Experimental companion for the RIP thread: documents a geometry-dependent transport protocol, a negative static-dynamic association test with power bound, and explicit reproducibility limits |
| **2601.0007** — *Quantitative Recovery Bounds from Vacuum Clustering* | `papers/2601-0007-gaussian-recovery-bounds/` | `verification/2601-0007/` (BBP Gaussian fidelity in numpy/scipy; 450 sampled draws; collar sweep) | Finite-mode Gaussian bridge between 0060's lattice Gaussian layer and 0101's non-Gaussian/CMI route; includes the v2 Petz-to-conditional-reattachment wording correction |
| **2601.0020** — *Geometric Markov Bounds and Rate Inheritance Modulo Fixed Points* | `papers/2601-0020-geometric-markov-rip/` | `verification/2601-0020/` (exact Ising-Z enumeration; corrected Fawzi-Renner factor; A.4-v1 counterexamples and A.4-v2 check) | Static-to-dynamic interface paper: CMI/RIP bridge with fixed-point caveats, finite-chain diagnostics, and explicit correction of the v1 diagonal Dirichlet comparison |
| **2601.0022** — *Operational Influence Proxies in a TFIM Surrogate* | `papers/2601-0022-influence-proxies-tfim/` | `verification/2601-0022/` (regenerable witness ED + power test; TEBD/MCWF trajectories not included) | TFIM surrogate companion for the omega=0/RIP thread: quantified no-floor power limits, positive witness benchmark for S=Z, and S=X null case |
| **2601.0023** — *Finite-Dimensional Davies Interface Lemmas* | `papers/2601-0023-davies-interface-lemmas/` | `verification/2601-0023/` (finite exact identities; v1 counterexamples; positivity pinning; optional `--with-tfim`) | Davies-interface companion for the omega=0 witness: corrected Bohr decomposition, corrected KMS multiplication constants, and finite pinning check |
| **2601.0031** — *From Static Recoverability to Maintenance Power* | `papers/2601-0031-typed-pipeline/` | `verification/2601-0031/` (finite GNS/KMS bridge; v1 work-sign counterexample; corrected sign check; N-trend) | Typed-pipeline companion for the 2601 block: static-to-dynamic-to-thermodynamic bookkeeping with explicit finite diagnostics |
| **2601.0034** — *Modular Recovery from Split Inclusions* | `papers/2601-0034-modular-recovery-split/` | `verification/2601-0034/` (finite-dimensional dictionary: CMI reduction, N-dependence demo, corrected Fawzi-Renner factor) | Split-inclusion-facing dictionary note for the 2601 block; v2 corrects the Fawzi-Renner factor in the finite anchor |
| **2601.0035** — *A Non-Gaussian Clustering-Recovery Bridge via CMI* | `papers/2601-0035-nongaussian-bridge/` | `verification/2601-0035/` (finite ED benchmark; corrected Fawzi-Renner factor; v1 overshoot artifact check) | Non-Gaussian finite-chain bridge companion for 0101/0034; regenerates the Petz-vs-FR benchmark on the fast grid |
| **2601.0038** — *Operational Signatures of Criticality from Petz Recovery* | `papers/2601-0038-criticality-petz-distance/` | `verification/2601-0038/` (CI smoke + manual finite ED benchmark; Petz-distance signal + CMI companion; chunkable) | Finite criticality-diagnostic companion for the recovery block; v2 adds the C-size caveat, window note, and CMI diagnostic |
| **2601.0040** — *Finite-Size Scaling of Petz Recovery Length in the TFIM* | `papers/2601-0040-petz-scaling/` | `verification/2601-0040/` (CI smoke + manual chunked finite ED benchmark; baseline and enhancement diagnostics) | Scaling sequel to 0038; v2 quantifies the C-size/baseline confound and functional-form ambiguity |
| **2601.0042** — *Emergent Information Distance from Petz Recovery* | `papers/2601-0042-emergent-distance/` | `verification/2601-0042/` (CI smoke + manual chunked finite ED benchmark; beta and perturbation sweep) | Closes the finite d_eff mini-block 0038/0040/0042; v2 replaces a corrupted reproducibility paragraph with a real suite |
| **2601.0043** — *Recoverability Geometry* | `papers/2601-0043-recoverability-geometry/` | `verification/2601-0043/` (CI smoke embedding demo + manual chunked finite ED control) | Geometry-facing finite diagnostic for the d_eff block; v2 flags the unstable row and adds a first in-model conjecture check |

Cross-citation graph: 0060 ⇄ 0061, 0060 ⇄ 0064, 0061 ⇄ 0064, 0070 → {0060, 0061, 0064}, 0064 → {0070, 0071}, 0071 → {0060, 0061, 0064, 0070, 0072}, 0072 → all five, 0073 → all six, 0081 → all seven, 0084 → {0081, 0061, 0064}, 0085 → {0060, 0061, 0064, 0070, 0072, 0073}, 0091 → all ten (context only, non-load-bearing), 0105 → {0070, 0072} (experimental companion), 2601.0007 → {0060, 0101} (finite-mode Gaussian bridge), 2601.0020 → {0060, 0064, 0070, 0072, 0101, 2601.0007} (static-to-dynamic interface), 2601.0022 → {0064, 0070, 0105, 2601.0020}, 2601.0023 → {0061, 0064, 0070, 2601.0020, 2601.0022}, 2601.0031 → {0060, 0061, 0064, 0070, 2601.0007, 2601.0020, 2601.0022, 2601.0023}, 2601.0034 → {0060, 0101, 2601.0007, 2601.0020, 2601.0031}, 2601.0035 → {0060, 0101, 2601.0007, 2601.0020, 2601.0034}, 2601.0038 → {0060, 0101, 2601.0035} (finite Petz/CMI criticality diagnostic), 2601.0040 → {2601.0038, 2601.0035, 0101} (finite-size Petz scaling diagnostic), 2601.0042 → {2601.0038, 2601.0040, 2601.0035, 0101} (finite emergent-distance diagnostic), 2601.0043 → {2601.0038, 2601.0040, 2601.0042, 2601.0035, 0101} (finite recoverability-geometry diagnostic) (each bibliography
points to the others' directories in this repository). The same Combes–Thomas
exponent appears in all three: `arccosh(1+m²/2)` for the 0060 lattice vacuum,
`arccosh(μ/2t)` in the 0064 oracle, `cosh q(ω) = (μ²+4−ω²)/(4μ)` in 0064's
frequency-resolved rate law.

*Series-history note:* 0071 v2 (the closure map, submitted 2026-07-04) lists
the series as 0060/0061/0064/0070/0072 — the inventory at its submission
time; 0073 and 0081 joined the v2 series immediately afterwards and cite all
predecessors. This README is the live inventory.

## What each v2 fixes (scoped checks here)

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

**0072** (v2 written here, replaces v1): v1's surrogate decay-curve evidence
(quantum-trajectory proxy on a TFIM buffer chain, no critical control, plus a
broken figure placeholder `nedladdning.png`) is withdrawn — it belongs to the
proxy class whose failure the 0064 v2 critical control exposed — and replaced
by exact Liouvillian-rapidity evidence generated from the 0064 verification
data (`verification/0072/make_rip_figure.py`). The weak form is promoted to a
proved lemma with explicit hypotheses; the strong form's status is updated
(derived in the quasi-free local-sink class with the squared-amplitude
exponent v1 anticipated; failure realized within the Davies class per 0070);
secular delocalization added as failure mode; the falsification protocol
gains the proxy-validation clause; maintenance inequality re-anchored to
0061 v2.

**0073** (v2 written here, replaces v1): (1) v1's Lemma 2 (Bohr-block
commutator decomposition of the KMS Dirichlet form) is FALSE for omega != 0 —
measured 30-40% deviation at beta=1; replaced by the exact modular-weighted
identity, machine-verified to 1.9e-10 (Lemma 1 at omega=0 is the weight-1
special case, so all witness numerics survive). (2) v1's quadratic envelope
Lemma 3 is withdrawn as proved (broken proof; exact tail-reduction shortcut
numerically false, defect ~5e-3) and replaced by a linear-envelope lemma with
explicit hypotheses; the quadratic case is open in the Davies class and exact
in 0064's local-sink class. (3) Imported theorem re-anchored to 0061 v2 (v1
imported the vacuous form verbatim). (4) Witness Table site convention fixed
(k = j0 - eps; values reproduced exactly: 0.134648/0.0885503/0.0684289 at
N=8). (5) Compile-time figure remark removed; program map updated; inline
Colab listings moved to `verification/0073/`.

**0081** (v2 written here, replaces v1): v1's Theorem 1 ("proved here") used
the vacuous unlinked-pairs P_extra and its Appendix A Step 4 subtracted two
lower bounds (invalid). v2 keeps the valid core of that appendix (Steps
0-3+5), which proves the *unconditional* entropy-production bound, and
imports the coherence/extra-power bounds from 0061 v2. Bonus finding: the
paper's own Protocol B uses a dephasing dissipator — exactly the
free-baseline case, so its headline inequality is a theorem with no further
assumptions in its own experimental setting; and its numerics were already
proxy-compliant (fixed-horizon metric + unitary-baseline subtraction). RIP
status updated; series references added (v1 cited only external literature);
work sign fixed to consumption; v1's unnecessary regularity assumption (A5)
dropped.

**0084** (v2 written here, replaces v1): the outermost, philosophy-of-mind
paper. Its single load-bearing technical input — the maintenance inequality
P_extra >= k_B T Cdot_loss — was cited in v1 in the unqualified extra-power
form whose companion-v1 formulation was later found vacuous. v2 cites it in
the corrected form (efficient-baseline incremental bound + unconditional
entropy-production floor, per 0081/0061 v2), and observes that the paper's
own passive-stability special case is exactly the corrected inequality's
free-baseline regime — so the correction sharpens rather than weakens the
argument. Full series references added (v1 cited only the single anchor
preprint); non-claims retained verbatim. No verification suite: it is
philosophy, and claims nothing numerical.

**0085** (v2 written here, replaces v1): the synthesis/pipeline paper. (1) Its
imported core law (Theorem 1) pointed to 0061 v1's Theorem 4.12 — the vacuous
unlinked-pairs extra-power bound; re-anchored to the corrected 0061 v2
hierarchy, work sign fixed to consumption. (2) Notably, this paper ALREADY
distinguished upper/lower rate envelopes (κ↑/κ↓, RIP-U/RIP-L) "to avoid
sign/quantifier errors" — anticipating the exact ceiling/floor correction
that 0070 needed in v2; v2 connects these to the now-proved companion results
(exact RIP-U in 0064, Davies floors in 0070, RIP lemma in 0072, corrected
Bohr-block identity in 0073 — whose linear upper-envelope lemma IS this
paper's Davies interface lemma). (3) CMI/Fawzi-Renner recoverability layer
tied to 0060 v2 (independent of its Petz→reattachment correction). (4) Two
checkable claims verified and script distributed: the elementary fidelity
lemma (1−F ≤ −log F) and the Davies upper-envelope direction (ω=0 witness
rate decays with separation, exactly reproducing the 0073 table).

**0091** (v2 written here, replaces v1): a self-contained heat-kernel /
induced-gravity appendix (Seeley–DeWitt a1 coefficients; sign of the induced
Newton constant per matter species), only heuristically linked to the
"resource boundary" program. **This is the first and only paper in the program
whose v1 contained no mathematical error** — all seven coefficients
(1/6−ξ, 1/6, 1/3, 1/3, 1/6, 1/6, −2/3) and the induced-Newton sign column are
verified symbolically (SymPy) and match standard heat-kernel references
(Vassilevich; Gilkey). v2 is therefore an *enhancement*, not a rectification:
verification script added, companion references updated to v2, scope note
tightened so the cut-program link is explicitly heuristic and non-load-bearing.

**0101** (v2 written here, replaces v1): the non-Gaussian companion of 0060.
(1) v1's "CMI as relative entropy" identity equated I(A:C|B) with
D(rho || sigma_MP) for the *normalized* Markov product state; the exact
identity uses the unnormalized product, and the normalized version differs by
-log Z >= 0 with Z <= 1 by Lieb's triple-matrix inequality — corrected and
verified numerically (machine precision on the unnormalized identity).
(2) Full independent re-implementation of the TFIM pipeline: all twenty Petz
slope rows reproduced (4-decimal agreement for beta >= 0.6; ~2% at beta = 0.3
where fits touch the 1e-13 arithmetic floor — caveat added in v2);
rotated-Petz subset and the crossover w* = 3 confirmed in all eight
disagreement cases; the v1 Colab PNG figure replaced by a script-generated
figure + CSV.

**0102** (v2 written here, replaces v1): the conventions/verification companion
of 0091. Same species table and signs; v2 adds (1) a corollary anchoring to the
classical counting 1/G_ind = (Λ²/12π)(N0 + 2N_{1/2} − 4N1), with A1 = 1/6 ×
{1, 2, −4}; (2) a gauge remark (the −2/3 is Feynman-gauge; off-shell
gauge/parametrization dependence declared, Kabat contact term); (3) a
Weyl/Majorana caveat (the halving is of the local a1 only; chiral-determinant
phases tracked separately); (4) a corrected Euclidean Wick weight
(iS^(L) → −S^(E)); (5) an exact-spectrum numerical verification on S^4 (scalar,
Dirac, Hodge–de Rham 1-forms) giving tr a1 = {+R/6, −R/3, −R/3} to <1e−5 and
A1(vector+ghosts) = −2/3. Verified 11/11 (verification/0102/).

**0105** (v2-LITE package integrated here): the IBM-hardware prefix-path Bell
transport note is included with its paper figures and an offline re-analysis
script. Because the processed on-device JSON artifacts are not preserved in
this repository, the shipped verification is a synthetic-data `--demo` of the
analysis machinery: fit-variant robustness, null scale-up association,
preregistered-test permutation statistic, power bound and proxy agreement. A
future data revision should recover raw job results from IBM account records,
export the canonical JSONs documented in `results/0105/README.md`, rerun the
same script on those files and then recompile the paper.

**2601.0007** (v2 package integrated here): the finite-mode Gaussian recovery
paper keeps its v1 theorems but corrects the language around Petz versus
conditional reattachment, aligning with the repaired 0060 framing. Its
verification suite implements zero-mean Gaussian fidelity via the
Banchi-Braunstein-Pirandola formula in numpy/scipy, checks closed-form anchors,
samples finite-mode admissible draws, records an empirical sampled-domain
constant for the local quadratic benchmark, and runs a collar-suppression
sweep. The suite is numerical finite-mode evidence and regression testing, not
a replacement for the paper's hypotheses or for 0060/0101's separate layers.

**2601.0020** (v2 package integrated here): the geometric Markov/RIP interface
paper makes two explicit corrections. First, in the squared-fidelity
convention, the Fawzi-Renner arithmetic is `I >= -log F`, so the v1 factor two
is removed. Second, the v1 diagonal Dirichlet comparison A.4 is false as
stated; the suite gives finite-chain counterexamples and checks the corrected
`B+2r` formulation. The verification is exact enumeration of a finite
classical Ising-Z chain plus random finite-dimensional Petz arithmetic tests;
it is a finite diagnostic/interface suite, not a general AQFT theorem.

**2601.0022** (v2 package integrated here): the TFIM influence-proxy paper adds
a declared finite ED witness benchmark (`N=10`, `J=1`, `h=1.05`, center-site
`S=Z`, `gamma0=0.1`), the corresponding `S=X` null case, and a synthetic power
analysis showing that a five-point no-floor design has weak floor-detection
power. No v1 result is retracted here, but the v2 package records that the
original TEBD/MCWF trajectories are not included. The shipped suite regenerates
the witness and power diagnostics only; it is not an interacting-buffer proof.
The companion 2601.0023 suite also exposes an appendix-declared `h=1.5` TFIM
witness run behind `--with-tfim`; that does not restore the missing TEBD/MCWF
trajectory data.

**2601.0023** (v2 package integrated here): the finite-dimensional Davies
interface paper records three corrections and finite checks. The v1 Bohr-block
Dirichlet decomposition is refuted on a small TFIM Davies model and replaced by
the corrected KMS-weighted expression; the v1 KMS submultiplicativity step is
refuted by a one-qubit counterexample and replaced with the stated
`c_sigma = (lambda_max/lambda_min)^(1/4)` bound; and positivity pinning checks
that the far-supported quadratic form scales as `delta^2` in the finite model.
These are finite-dimensional interface diagnostics, not an interacting-buffer
or continuum proof.

**2601.0031** (v2 package integrated here): the typed-pipeline note records two
repairs and finite diagnostics. The v1 work-cost sign in Eq. 25 is refuted by
finite energy-conserving-unitary draws and replaced with the battery
free-energy decrease convention; the static layer is aligned with the
conditional-reattachment wording used in 0060/2601.0007; and the suite records
a finite GNS-vs-KMS convention bridge plus the Figure 1 witness trend for
`N = 6, 8, 10`. The verification is typed bookkeeping and finite-model
regression testing, not a theorem beyond the paper's stated hypotheses.

**2601.0034** (v2 package integrated here): the modular-recovery/split note
records the same Fawzi-Renner factor correction family as 0101 and 2601.0020:
with squared fidelity, the finite-dimensional anchor is `-log F <= I`, not
`-log F <= I/2`. The suite checks the standard Type I CMI reduction, demonstrates
dependence on the chosen finite split identification, verifies the `beta_0`
normalization, checks the corrected Petz-recovery anchor on random finite
states, and records the corrected arithmetic constant. These checks are a
finite-dimensional dictionary/regression layer, not a Type III theorem.

**2601.0035** (v2 package integrated here): the non-Gaussian bridge note records
the same squared-fidelity Fawzi-Renner factor correction family. The suite
regenerates a finite TFIM exact-diagonalization benchmark at `N=9`, checks CMI
decay slopes on the finite grid, and records that the v1 Petz "overshoot" was
against the half-scale; against the corrected scale the finite grid has
`-log F / I < 1`. The `--N11` run is available for the paper-size comparison
but is left out of fast CI. These are finite ED benchmarks, not a general
non-Gaussian AQFT theorem.

**2601.0038** (v2 package integrated here): the criticality/Petz-distance note
keeps the v1 numerical signal but adds three finite-scope qualifications: the
absolute recovery-length reading is confounded by shrinking `|C|` at fixed
`N`, the resolved window depends on `(epsilon, beta, N)`, and a CMI companion
diagnostic shows the same qualitative finite-grid signal. CI runs an explicit
`--smoke` grid that exercises the finite Petz/CMI code path; the default
suite regenerates the longer `N=9` Petz and CMI tables, while `--N11` and
`--chunk hz beta` are manual support modes. These are finite ED diagnostics,
not a continuum criticality theorem.

**2601.0040** (v2 package integrated here): the finite-size Petz-scaling note
keeps the v1 numerical table where the tested sizes overlap, but adds two
finite-scope caveats. The off-critical baseline also grows with `N`, so raw
peak-height exponents mix recovery behavior with geometry; the cleaner finite
diagnostic is the peak-minus-baseline enhancement. The paper also records that
power, logarithmic, and linear fits are indistinguishable on the short
`N = 9..12` window. CI runs only an explicit `--smoke` code-path check; the
chunked `N = 8, 9, 10` cache-regeneration run and final report are manual.
These are finite ED diagnostics, not a scaling-law theorem.

**2601.0042** (v2 package integrated here): the emergent-distance note replaces
a corrupted v1 reproducibility paragraph with a real finite ED suite. The
manual chunked benchmark reproduces the finite beta/perturbation sweep at
`N = 9`, including the high-temperature/perturbed-regime comparisons and PSD
projection sensitivity diagnostic. CI runs only an explicit `--smoke` grid;
the `--chunk 0.0`, `--chunk 0.5`, and final report sequence is manual. These
are finite operational-distance diagnostics, not continuum geometry or
general recovery-length results.

**2601.0043** (v2 package integrated here): the recoverability-geometry note
adds a finite sparse-Lanczos verification layer for the fixed-target geometry
protocol used by the d_eff block. The manual chunked benchmark regenerates the
Table 1 control at `L = 14`, flags the unstable `g = 0.5` three-point fit under
the paper's own stability policy, and compares `xi_rec` with a same-state
connected-correlation length as a first finite in-model conjecture check. CI
runs only the `--smoke`/`--demo` embedding illustration at `L = 12`; the
`--chunk 0.5`, `--chunk 1.0`, `--chunk 2.0`, and final report sequence is
manual. These are finite diagnostics for a geometry protocol, not continuum
geometry or a general recoverability theorem.

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
# 0072 (regenerates the exact RIP figure)
python verification/0072/make_rip_figure.py
# 0073 (Davies corrected identities and witness tables)
python verification/0073/verify_corrected.py
python verification/0073/witness_tfim.py
python verification/0073/verify_0073.py
# 0085 (fidelity bound + Davies upper-envelope direction)
python verification/0085/verify_0085.py
# 0091 (symbolic Seeley-DeWitt coefficient/sign table)
python verification/0091/verify_0091.py
# 0101 (petz ~3 min; rot/fig8 longer)
python verification/0101/verify_0101.py petz
python verification/0101/verify_0101.py fig8
# 0102 (exact S^4 spectra, seconds)
python verification/0102/verify_0102.py
# 0105 (synthetic demo of offline re-analysis machinery; raw JSON data pending)
python verification/0105/reanalyze_0105.py --demo --output-dir /tmp/aqft-0105-demo
# 2601.0007 (finite-mode Gaussian recovery numerical suite)
python verification/2601-0007/verify_2601_0007.py
# 2601.0020 (finite Ising-Z Markov/RIP interface checks)
python verification/2601-0020/verify_2601_0020.py
# 2601.0022 (finite TFIM influence-proxy witness + no-floor power diagnostic)
python verification/2601-0022/verify_2601_0022.py
# 2601.0023 (finite Davies-interface identities/counterexamples/pinning)
python verification/2601-0023/verify_2601_0023.py
# Optional appendix-declared TFIM witness comparison, manual/longer
python verification/2601-0023/verify_2601_0023.py --with-tfim
# 2601.0031 (finite typed-pipeline checks)
python verification/2601-0031/verify_2601_0031.py
# 2601.0034 (finite split-recovery dictionary checks)
python verification/2601-0034/verify_2601_0034.py
# 2601.0035 (finite non-Gaussian ED bridge benchmark)
python verification/2601-0035/verify_2601_0035.py
# Optional paper-size comparison, slower
python verification/2601-0035/verify_2601_0035.py --N11
# 2601.0038 (finite Petz-distance/CMI criticality diagnostic)
python verification/2601-0038/verify_2601_0038.py --smoke
# Longer/manual finite grids
python verification/2601-0038/verify_2601_0038.py
python verification/2601-0038/verify_2601_0038.py --N11
python verification/2601-0038/verify_2601_0038.py --chunk 0.0 12.0
# 2601.0040 (finite Petz scaling diagnostic)
python verification/2601-0040/verify_2601_0040.py --smoke
# Longer/manual chunked finite grid
python verification/2601-0040/verify_2601_0040.py --chunk 8
python verification/2601-0040/verify_2601_0040.py --chunk 9
python verification/2601-0040/verify_2601_0040.py --chunk 10
python verification/2601-0040/verify_2601_0040.py
# 2601.0042 (finite emergent-distance diagnostic)
python verification/2601-0042/verify_2601_0042.py --smoke
# Longer/manual chunked finite grid
python verification/2601-0042/verify_2601_0042.py --chunk 0.0
python verification/2601-0042/verify_2601_0042.py --chunk 0.5
python verification/2601-0042/verify_2601_0042.py
# 2601.0043 (finite recoverability-geometry diagnostic)
python verification/2601-0043/verify_2601_0043.py --smoke
# Longer/manual sparse-Lanczos chunks and report
python verification/2601-0043/verify_2601_0043.py --chunk 0.5
python verification/2601-0043/verify_2601_0043.py --chunk 1.0
python verification/2601-0043/verify_2601_0043.py --chunk 2.0
python verification/2601-0043/verify_2601_0043.py
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
- **0105**: verification/0105 currently validates the re-analysis code on
  synthetic data only. The on-device IBM JSON artifacts are explicitly pending
  data revision; this repository does not claim fully offline reproduction of
  the experimental numbers until those raw or processed records are added.
- **2601.0007**: verification/2601-0007 is a finite-mode Gaussian numerical
  suite with sampled admissible draws and closed-form anchors. It supports the
  artifact's regression story; it is not a continuum AQFT result and does not
  remove the explicit hypotheses of the paper or of companion 0060/0101.
- **2601.0020**: verification/2601-0020 is an exact finite-chain/classical
  enumeration suite plus finite-dimensional Petz arithmetic checks. It records
  v1 counterexamples and v2 finite diagnostics; it is not a continuum or
  interacting-buffer proof of rate inheritance.
- **2601.0022**: verification/2601-0022 is a finite TFIM surrogate/witness ED
  suite plus synthetic power analysis. The TEBD/MCWF trajectories behind the
  older proxy figures are not shipped, so those figures are artifacts here,
  not fully regenerated data products. No continuum or interacting-buffer claim
  is made by this repository.
- **2601.0023**: verification/2601-0023 checks finite-dimensional Davies
  identities, finite counterexamples to v1 proof steps, and a finite positivity
  pinning diagnostic. The optional `--with-tfim` flag reuses the finite TFIM
  witness core only as an appendix comparison. No continuum, Type III, or
  interacting-buffer claim is made by this repository.
- **2601.0031**: verification/2601-0031 checks finite typed-pipeline
  bookkeeping: GNS/KMS convention differences, a finite work-sign
  counterexample/repair, pinching Pythagoras, secular covariance, and a finite
  witness trend. It does not claim a continuum, Type III, or interacting-buffer
  theorem.
- **2601.0034**: verification/2601-0034 checks a finite-dimensional dictionary:
  CMI reduction under natural Type I factorization, dependence on split
  identification, beta_0 normalization, and the corrected finite
  Fawzi-Renner/Petz anchor. It does not prove a Type III, continuum, or
  modular-recovery theorem beyond the stated finite checks.
- **2601.0035**: verification/2601-0035 is a finite exact-diagonalization
  benchmark for a non-Gaussian TFIM bridge. It checks the corrected
  Fawzi-Renner scale and finite-grid CMI decay; it does not prove a continuum,
  Type III, or general non-Gaussian clustering-recovery theorem.
- **2601.0038**: verification/2601-0038 is a finite exact-diagonalization
  Petz-distance/CMI diagnostic on fixed finite chains. CI runs only the
  explicit smoke grid; the `N=9`, `--N11`, and chunked grids are longer manual
  runs. The package reports a finite criticality-associated signal and its
  `|C|` and window caveats; it does not prove a continuum criticality,
  Type III, or general recovery-length theorem.
- **2601.0040**: verification/2601-0040 is a finite exact-diagonalization
  diagnostic for Petz recovery length versus finite chain size. CI runs only
  the explicit smoke grid; the chunked `N = 8, 9, 10` report is manual. The
  package reports baseline/enhancement and fit-form caveats; it does not prove
  a continuum scaling law, Type III statement, or general recovery-length
  theorem.
- **2601.0042**: verification/2601-0042 is a finite exact-diagonalization
  diagnostic for beta/perturbation dependence of an operational
  Petz-distance. CI runs only the explicit smoke grid; the two-regime chunked
  report is manual. The package reports finite operational-distance behavior;
  it does not prove continuum geometry, Type III structure, or a general
  emergent-distance theorem.
- **2601.0043**: verification/2601-0043 is a finite sparse-Lanczos diagnostic
  for recoverability geometry on fixed finite chains. CI runs only the
  embedding smoke/demo; the `g = 0.5, 1.0, 2.0` chunked control report is
  manual. The package reports finite geometry-protocol behavior and an
  unstable-row flag; it does not prove continuum geometry, Type III structure,
  or a general recoverability theorem.
