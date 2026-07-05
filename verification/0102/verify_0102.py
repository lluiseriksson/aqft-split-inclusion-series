#!/usr/bin/env python3
# verify_0102.py -- Exact-spectrum numerical verification of the a1 coefficients used in
# "Heat Kernel Methods and the Sign of Induced Gravity" (2512.0102, v2).
#
# Method: on the round unit sphere S^4 (V = 8 pi^2/3, R = 12) the spectra of
# the relevant Laplace-type operators are known in closed form, so the
# small-s expansion  Tr e^{-sP} ~ V/(16 pi^2 s^2) [tr a0 + s tr a1 + O(s^2)]
# can be checked against exact eigenvalue sums, with Richardson extrapolation
# s -> 0. The left-hand side is an exact spectral sum: nothing in this check
# relies on the expansion being tested.
#
# Checks:
#   PART 1  scalar  P = -Lap              : tr a1 = R/6   -> +2 on S^4
#   PART 2  Dirac   P_D = -D^2 = -Lap+R/4 : tr a1 = -R/3  -> -4 on S^4
#   PART 3  1-forms P_1 = -Lap + Ric      : tr a1 = -R/3  -> -4 on S^4
#           vector + ghosts combination   : A1_eff = -2/3
#   PART 4  bookkeeping: species table, G_ind sign examples, equivalence with
#           the classic counting 1/G_ind = (Lambda^2/12 pi)(N0 + 2N_1/2 - 4N1).
#
# Expected: every line ends OK; exit code 0.

import numpy as np
import sys

V = 8 * np.pi**2 / 3      # Vol(S^4), unit radius
R = 12.0                  # scalar curvature of unit S^4
LMAX = 4000
TOL_SPEC = 5e-4           # tolerance on extrapolated tr a1
TOL_ALG = 1e-12           # tolerance on exact algebra

l = np.arange(0, LMAX, dtype=float)

# ----- exact spectra on unit S^4 -----------------------------------------
# scalar Laplacian: eigenvalues l(l+3), mult (2l+3)(l+1)(l+2)/6, l >= 0
def trace_scalar(s):
    lam = l * (l + 3)
    mult = (2*l + 3) * (l + 1) * (l + 2) / 6
    return float((mult * np.exp(-s * lam)).sum())

# Dirac: P_D = -D^2, eigenvalues (l+2)^2, total mult (4/3)(l+1)(l+2)(l+3), l >= 0
def trace_dirac(s):
    lam = (l + 2)**2
    mult = 4 * (l + 1) * (l + 2) * (l + 3) / 3
    return float((mult * np.exp(-s * lam)).sum())

# 1-forms, Hodge-de Rham Delta_1 = -Lap + Ric (Weitzenboeck; Ric = 3g on S^4):
#   exact part   : d(phi_l), eigenvalues l(l+3),  mult (2l+3)(l+1)(l+2)/6, l >= 1
#   co-exact part: eigenvalues (l+1)(l+2),        mult l(l+3)(2l+3)/2,     l >= 1
#   (no harmonic 1-forms: b_1(S^4) = 0)
# Anchor: co-exact l=1 -> eigenvalue 6, mult 10 = dim SO(5) (Killing 1-forms).
def trace_oneform(s):
    m = l[1:]
    lex = m * (m + 3)
    mex = (2*m + 3) * (m + 1) * (m + 2) / 6
    lco = (m + 1) * (m + 2)
    mco = m * (m + 3) * (2*m + 3) / 2
    return float((mex * np.exp(-s * lex)).sum() + (mco * np.exp(-s * lco)).sum())

def extract_a1(trace_fn, a0):
    """Richardson-extrapolate  (16 pi^2 s^2 Tr/V - a0)/s  to s -> 0."""
    s1, s2 = 0.004, 0.002
    g = lambda s: (trace_fn(s) * 16 * np.pi**2 * s**2 / V - a0) / s
    return 2 * g(s2) - g(s1)

failures = 0
def check(label, got, want, tol):
    global failures
    ok = abs(got - want) <= tol
    if not ok:
        failures += 1
    print(f"  {label}: got {got:+.6f}  expected {want:+.6f}  "
          f"|diff|={abs(got-want):.2e}  {'OK' if ok else 'FAIL'}")

print("== PART 1: scalar on S^4, P = -Lap (X = 0) ==")
a1_s = extract_a1(trace_scalar, a0=1.0)
check("tr a1(scalar) = R/6", a1_s, R / 6, TOL_SPEC)

print("== PART 2: Dirac on S^4, P_D = -Lap + R/4 (Lichnerowicz) ==")
a1_d = extract_a1(trace_dirac, a0=4.0)
check("tr a1(Dirac) = -R/3", a1_d, -R / 3, TOL_SPEC)

print("== PART 3: 1-forms on S^4, P_1 = -Lap + Ric (Feynman gauge) ==")
a1_v = extract_a1(trace_oneform, a0=4.0)
check("tr a1(1-form) = -R/3", a1_v, -R / 3, TOL_SPEC)
# vector + ghosts: W = (1/2) ln det P_1 - ln det P_gh, ghost = complex scalar
#   => effective combination: tr a1(P_1) - 2 tr a1(scalar) = A1_eff * R
A1_vg = (a1_v - 2 * a1_s) / R
check("A1_eff(vector+ghosts) = -2/3", A1_vg, -2.0/3.0, 3 * TOL_SPEC / R)

print("== PART 4: bookkeeping (exact algebra) ==")
A1_scalar = lambda xi: 1.0/6.0 - xi
A1_dirac = 1.0/3.0
A1_vec = -2.0/3.0
check("conformal scalar xi=1/6 -> A1 = 0", A1_scalar(1.0/6.0), 0.0, TOL_ALG)
check("Dirac = 2 x minimal scalar", A1_dirac, 2 * A1_scalar(0), TOL_ALG)
check("vector = -4 x minimal scalar", A1_vec, -4 * A1_scalar(0), TOL_ALG)

# G_ind = 2 pi / (A1 Lambda^2); classic: 1/G = (Lambda^2/12 pi)(N0 + 2N1/2 - 4N1)
Lam = 1.0e3
N0, N12, N1 = 3, 2, 1          # arbitrary spectrum for the identity check
A1_tot = N0 * A1_scalar(0) + N12 * A1_dirac + N1 * A1_vec
invG_paper = A1_tot * Lam**2 / (2 * np.pi)
invG_classic = (Lam**2 / (12 * np.pi)) * (N0 + 2 * N12 - 4 * N1)
check("1/G: unified A1 == (N0+2N1/2-4N1)/12pi", invG_paper, invG_classic,
      TOL_ALG * Lam**2)

# sign examples of Sec. 8
check("sign: minimal scalar -> G>0", np.sign(2*np.pi/(A1_scalar(0)*Lam**2)), 1.0, TOL_ALG)
check("sign: xi=1/4 scalar -> G<0", np.sign(2*np.pi/(A1_scalar(0.25)*Lam**2)), -1.0, TOL_ALG)
check("sign: vector+ghosts -> G<0", np.sign(2*np.pi/(A1_vec*Lam**2)), -1.0, TOL_ALG)

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else str(failures) + ' CHECK(S) FAILED'}")
sys.exit(0 if failures == 0 else 1)
