# Brick P4.5 unblocking note: Combes–Thomas by exponential conjugation

Target (from `docs/PHYSICAL-OPERATOR-VERTICAL-SLICE.md`, Section 8):

```lean
theorem physicalCovarianceKernel_expDecay :
  ExpDecay bondDistance covarianceAmp covarianceRate
    (physicalCovarianceKernel Ubar Omega)
```

Status in tree: no Lean file mentions Combes–Thomas or this target yet. This note
gives the complete three-lemma chain in the interface language of the brick,
with explicit volume-uniform constants, so it can be formalized without any new
"lattice covariance operator" primitive — only bounded-operator API that Mathlib
already has (operator norm, Schur-type bound via `opNorm_le_of_...`, Neumann
series / invertibility of `1 - x` for `‖x‖ < 1`, and matrix-entry ≤ opNorm via
`abs_inner_le_norm` on basis vectors).

## Setting

`K : E →L[ℝ] E` self-adjoint on `E = EuclideanSpace ℝ (Bond × Fin lieDim)`
(finite volume, any size), with:

* **(H1) coercivity** `c > 0`: `∀ v, c * ‖v‖² ≤ ⟪v, K v⟫`
  (this is `physicalCoercivityConstant_pos` + `physicalGaugeFixedPrecision_coercive`,
  already in the vertical-slice plan);
* **(H2) finite range** `R`: `kernel K p q = 0` when `bondDistance p q > R`;
* **(H3) Schur bound** `S`: `∀ p, ∑ q, |kernel K p q| ≤ S`.

`c`, `R`, `S` must not depend on the volume — for the physical Hessian they do
not (locality of the action gives H2/H3, coercivity is the standing hypothesis).
`C := K⁻¹` is `physicalCovariance`, and `kernel C = physicalCovarianceKernel`.

## The three lemmas

**Lemma 1 (conjugated kernel identity + defect bound).** For `θ ≥ 0` and a
fixed `q₀`, let `f p := bondDistance p q₀` (1-Lipschitz for the bond metric) and
`M_θ := diagonal (exp (θ * f ·))`. Then

```
kernel (M_θ ∘ K ∘ M_θ⁻¹) p q = exp (θ * (f p - f q)) * kernel K p q,
```

and since `|f p - f q| ≤ bondDistance p q ≤ R` on the support of `kernel K`,

```
conjugationDefect θ := ‖M_θ ∘ K ∘ M_θ⁻¹ - K‖ ≤ (exp (θ*R) - 1) * S.
```

*Proof:* entrywise `|e^{θ(f p−f q)} − 1| ≤ e^{θR} − 1` on the support; the
entrywise-dominated operator has norm ≤ its Schur bound (row-sum test; for a
self-adjoint dominating kernel the symmetric Schur test with both row and
column sums ≤ `(e^{θR}−1)S` suffices). No structure of `K` beyond H2–H3 used.

**Lemma 2 (inversion under small defect).** If `conjugationDefect θ < c`, then
`K_θ := M_θ ∘ K ∘ M_θ⁻¹` is invertible and

```
‖K_θ⁻¹‖ ≤ (c - conjugationDefect θ)⁻¹.
```

*Proof:* H1 gives `‖K⁻¹‖ ≤ c⁻¹`. Write `K_θ = K ∘ (1 + K⁻¹ ∘ D)` with
`D = K_θ - K`, `‖K⁻¹ D‖ ≤ d/c < 1`; Neumann series inverts the bracket with
norm `≤ (1 − d/c)⁻¹`, hence `‖K_θ⁻¹‖ ≤ c⁻¹ (1 − d/c)⁻¹ = (c − d)⁻¹`. This is
exactly the inequality block written in the brick:

```
‖M_θ C M_θ⁻¹‖ ≤ (coercivityConstant − conjugationDefect θ)⁻¹,
```

because `M_θ C M_θ⁻¹ = (M_θ K M_θ⁻¹)⁻¹ = K_θ⁻¹` (conjugation commutes with
inversion) — the "exact inverse, not a majorized formal series" requirement is
met by construction.

**Lemma 3 (kernel extraction ⇒ ExpDecay).** For any `p, q` take `q₀ := q` in
Lemma 1–2. Then

```
|kernel C p q| = exp (−θ * bondDistance p q) * |kernel (M_θ C M_θ⁻¹) p q|
              ≤ exp (−θ * bondDistance p q) * (c − conjugationDefect θ)⁻¹,
```

using `f q = 0`, `f p = bondDistance p q`, and matrix entry ≤ operator norm.
Hence, for **any** `θ` with `(e^{θR} − 1) S < c`, i.e. any

```
θ < R⁻¹ * log (1 + c / S),
```

the headline holds with `covarianceRate := θ` and
`covarianceAmp := (c − (e^{θR} − 1) S)⁻¹` — both **uniform in volume** because
`c, R, S` are, which is the Appendix-F compatibility requirement stated in the
brick. A convenient canonical instantiation: `θ := R⁻¹ log(1 + c/(2S))`, giving
`amp = 2/c`.

## Calibration and guardrails (from the exactly solvable slice)

For the solvable 1D instance `K = μ·1 − t(shift + shiftᵀ)` (range `R = 1`,
`c = μ − 2t`, `S = μ + 2t`) the *sharp* kernel decay rate is the evanescent
branch `q_sharp = arccosh(μ / 2t)`; the conjugation route certifies
`θ_adm = log(1 + c/S)`. At `μ = 3, t = 1`:

| quantity | value |
|---|---|
| conjugation rate `θ_adm` | 0.1823 |
| sharp rate `arccosh(3/2)` | 0.9624 |
| measured kernel fit (`N = 400`, exact inverse) | 0.9624 |
| slack factor | ≈ 5.3× |

The slack is expected and harmless: any strictly positive volume-uniform rate
feeds the P3/P4 → `hRpoly` pipeline; sharpness is a refinement (relevant to
`GAP-REFINEMENT-CHALLENGE.md`, where the arccosh envelope is the ceiling no
conjugation choice can beat). Mandatory sanity guardrail, in the spirit of the
repo's adversarial audits: both rates must **close as `c → 0`**
(`θ_adm ~ c/S`, `q_sharp ~ √(c/t)`); a stated `ExpDecay` lemma that survives the
gapless limit is wrong. `ct_oracle.py` (companion script) checks, for given
`(μ, t, N)`: the certified `(θ, amp)` pair pointwise against the exact inverse
kernel, the sharp-rate match, and the critical closing — run it before freezing
the constants in the Lean statement, matching the existing Oracle pattern
(`OracleC97/98`, `ym-lattice-numerics`).

## What this does and does not unblock

Unblocks: Brick P4.5 as stated (the decay proof and its interface
inequalities), and the "raw pointwise decay" input shape consumed by the
HRPOLY P3/P4 rows — Lemmas 1–3 are pure bounded-operator arguments, no polymer
animal model or Gaussian-measure primitive required. Does **not** touch: the
cluster expansion with holes, the fluctuation integral, M4/M5, or anything in
`YangMills/RG/` — the Clay-distance disclaimer in `HRPOLY-CAMPAIGN-PLAN.md`
stands exactly as written.

*Provenance: derived and numerically validated in the session of 2026-07-03/04
(exact quasi-free Lindblad solvers, ai.viXra:2512.0064v2 companion work);
oracle cross-checked to 1e-15 against brute-force inversion.*
