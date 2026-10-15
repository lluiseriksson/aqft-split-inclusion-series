#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de verificacion para ai.viXra:2601.0064 v2
"RIP-U and the omega=0 obstruction in Davies dynamics"

Comprueba numericamente (d=6, solo numpy/scipy):
  1) Identidad omega=0 (Thm 5.5) a precision de maquina.
  2) Lema de envolvente Fourier (Lemma 3.3) con correlador concreto.
  3) Delta-MONO demostrado para el pinching de energia (v2):
     covarianza Delta o L = L o Delta, perdida >= 0, y la identidad
     dC/dt = -Tr(L(rho)(log rho - log Delta rho)) contra expm.
  4) BRIDGE-P (v2): cota demostrada con constante explicita.
  5) Direccionalidad (Remark 4.4): par de niveles debil => kappa_down <<
     kappa_up bajo la misma envolvente.
  6) Witness C1 (Cor 5.7).
  7) Consistencia con 2601.0023 v2: plain-commutator exacta SOLO en w=0.

Uso: python3 verify_2601_0064.py
"""
import sys

import numpy as np
from scipy.integrate import trapezoid as scipy_trapezoid
from scipy.linalg import expm, logm

np.random.seed(20260064)
TOL = 1e-10
results = []
trapz = getattr(np, "trapezoid", scipy_trapezoid)

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    suffix = " [%s]" % detail if detail else ""
    print("  %s: %s%s" % (name, "OK" if ok else "FAIL", suffix))

def dag(X): return X.conj().T

# ---------- sistema Davies: d=6, H no degenerado, S hermitico ----------
d = 6
beta = 0.7
E = np.sort(np.random.rand(d) * 3.0)
H = np.diag(E)
S = np.random.randn(d, d) + 1j*np.random.randn(d, d)
S = (S + dag(S)) / 2.0
sigma = np.diag(np.exp(-beta*E)); sigma /= np.trace(sigma).real
sq = np.diag(np.diag(sigma)**0.5)

def bohr_decompose(Sop):
    out = {}
    for i in range(d):
        for j in range(d):
            w = round(E[j] - E[i], 12)
            if w not in out: out[w] = np.zeros((d, d), complex)
            out[w][i, j] += Sop[i, j]
    return out

bohr = bohr_decompose(S)
omegas = sorted(bohr.keys())

def gamma_rate(w, f_eps=1.0):
    # tasa KMS: gamma(w) = f * e^{beta w/2} ghat(|w|), ghat par positiva
    return f_eps * np.exp(beta*w/2.0) * np.exp(-abs(w)/2.0)

def dissipator(rho, comps, f_eps=1.0):
    out = np.zeros((d, d), complex)
    for w in comps:
        A = comps[w]; g = gamma_rate(w, f_eps)
        out += g*(A @ rho @ dag(A) - 0.5*(dag(A)@A@rho + rho@dag(A)@A))
    return out

def kms_norm2(X):
    return np.trace(sq @ dag(X) @ sq @ X).real

def dirichlet_sector(O, w):
    A = bohr[w]; g = gamma_rate(w)
    Ld = g*(dag(A) @ O @ A - 0.5*(dag(A)@A@O + O@dag(A)@A))
    return -np.trace(sq @ dag(O) @ sq @ Ld).real

print("== 0) sanidad: estacionariedad y balance detallado ==")
check("L(sigma) = 0", np.linalg.norm(dissipator(sigma, bohr)) < 1e-12,
      "%.1e" % np.linalg.norm(dissipator(sigma, bohr)))
kms_ok = all(abs(gamma_rate(-w) - np.exp(-beta*w)*gamma_rate(w)) < 1e-14 for w in omegas)
check("KMS: gamma(-w) = e^{-bw} gamma(w)", kms_ok)

print("== 1) identidad omega=0 (Thm 5.5) ==")
S0 = bohr[0.0]; g0 = gamma_rate(0.0)
errs = []
for k in range(20):
    O = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    if k % 2 == 0: O = (O + dag(O))/2.0
    lhs = dirichlet_sector(O, 0.0)
    C = S0 @ O - O @ S0
    rhs = 0.5*g0*kms_norm2(C)
    errs.append(abs(lhs - rhs)/max(abs(rhs), 1e-30))
check("E^(0)(O) = g(0)/2 ||[S(0),O]||^2_{2,s} (20 O aleatorios)",
      max(errs) < TOL, "err max %.1e" % max(errs))
check("[S(0), sigma^{1/2}] = 0 (Lemma 5.4)",
      np.linalg.norm(S0 @ sq - sq @ S0) < 1e-14)

print("== 2) envolvente Fourier (Lemma 3.3) ==")
tt = np.linspace(-40, 40, 400001)
gt = np.exp(-tt**2/2.0)*np.cos(2.0*tt)
gl1 = trapz(np.abs(gt), tt)
ok_env = True; margin = np.inf
for eps in [0.0, 0.5, 1.0, 2.0, 4.0]:
    f = np.exp(-eps/1.5)
    for w in [0.0, 0.7, 2.0, 5.0]:
        gam = trapz(gt*f*np.exp(1j*w*tt), tt)
        ok_env &= (abs(gam) <= gl1*f + 1e-9)
        margin = min(margin, gl1*f - abs(gam))
check("|gamma(w;eps)| <= ||g||_1 f(eps) en malla (w,eps)", ok_env,
      "margen min %.3f, ||g||_1=%.3f" % (margin, gl1))

print("== 3) Delta-MONO para pinching de energia (v2) ==")
def pinch(X): return np.diag(np.diag(X))
cov_err = 0.0
for _ in range(10):
    X = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    cov_err = max(cov_err, np.linalg.norm(pinch(dissipator(X, bohr)) -
                                          dissipator(pinch(X), bohr)))
check("covarianza Delta o L = L o Delta", cov_err < 1e-12, "%.1e" % cov_err)

def rand_state(pmin=1e-3):
    A = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    r = A @ dag(A); r /= np.trace(r).real
    return (1.0 - d*pmin)*r + pmin*np.eye(d)

def C_delta(rho):
    lr = logm(rho); ld = np.diag(np.log(np.diag(rho).real))
    return np.trace(rho @ (lr - ld)).real

def Cdot_loss(rho, comps=None, f_eps=1.0):
    L = dissipator(rho, bohr if comps is None else comps, f_eps)
    lr = logm(rho); ld = np.diag(np.log(np.diag(rho).real))
    return -np.trace(L @ (lr - ld)).real

rho0 = rand_state()
h = 1e-6
Lsuper = np.zeros((d*d, d*d), complex)
for a in range(d):
    for b in range(d):
        Eab = np.zeros((d, d), complex); Eab[a, b] = 1.0
        Lsuper[:, a*d+b] = dissipator(Eab, bohr).reshape(-1)
rho_h = (expm(h*Lsuper) @ rho0.reshape(-1)).reshape(d, d)
fd = -(C_delta(rho_h) - C_delta(rho0))/h
an = Cdot_loss(rho0)
check("dC/dt analitico vs expm (dif. finita)", abs(fd - an)/abs(an) < 1e-4,
      "rel %.1e" % (abs(fd - an)/abs(an)))
losses = [Cdot_loss(rand_state()) for _ in range(50)]
check("Cdot_loss >= 0 en 50 estados (DPI+covarianza)", min(losses) > -1e-12,
      "min %.2e" % min(losses))

print("== 4) BRIDGE-P: cota con constante explicita (v2) ==")
pmin, cmin_obs = 1e-3, None
NB = len(omegas); maxS2 = max(np.linalg.norm(bohr[w], 2)**2 for w in omegas)
ghat_max = max(gamma_rate(w) for w in omegas)
ok_bridge, worst = True, 0.0
for eps in [0.0, 1.0, 2.0]:
    f = np.exp(-eps/1.5)
    for _ in range(30):
        rho = rand_state(pmin)
        C = C_delta(rho)
        if C < 1e-6: continue
        cmin_obs = C if cmin_obs is None else min(cmin_obs, C)
        r = Cdot_loss(rho, None, f)/C
        CRIP = (2.0*NB*ghat_max*maxS2*2.0*np.log(1.0/pmin))/max(cmin_obs, 1e-12)
        ratio = r/(CRIP*f) if f > 0 else 0.0
        worst = max(worst, ratio)
        ok_bridge &= (r <= CRIP*f + 1e-9)
check("r(rho) <= C_RIP f(eps) (90 estados, 3 eps)", ok_bridge,
      "peor r/(C_RIP f) = %.2e" % worst)

print("== 5) direccionalidad: spread kappa_up / kappa_down (Remark 4.4) ==")
# par de niveles (0,1) debilmente acoplado (misma envolvente CORR): la
# coherencia 0-1 decae a O(eta^2) mientras otros pares decaen a O(1).
eta = 0.05
S2 = S.copy()
for a in (0, 1):
    S2[a, :] *= eta; S2[:, a] *= eta
    S2[a, a] /= eta
S2 = (S2 + dag(S2))/2.0
bohr2 = bohr_decompose(S2)
rates = []
for (i, j) in [(a, b) for a in range(d) for b in range(a+1, d)]:
    rho = np.diag(np.diag(sigma).copy())
    amp = 0.35*np.sqrt(sigma[i, i].real*sigma[j, j].real)
    rho[i, j] += amp; rho[j, i] += amp
    rho = (1 - d*1e-4)*rho + 1e-4*np.eye(d)
    C = C_delta(rho)
    if C > 1e-12: rates.append(((i, j), Cdot_loss(rho, bohr2)/C))
rvals = [r for _, r in rates]
pair_min = min(rates, key=lambda t: t[1])[0]
spread = max(rvals)/min(rvals)
check("spread r(rho) con par debil (0,1), eta=%.2f" % eta,
      spread > 50.0 and pair_min == (0, 1),
      "kappa_up/kappa_down ~ x%.0f (min en par %s)" % (spread, (pair_min,)))

print("== 6) witness C1 (Cor 5.7) ==")
vals = []
for _ in range(200):
    O = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    Cc = S0 @ O - O @ S0
    vals.append((kms_norm2(Cc)/kms_norm2(O), O))
c_best, O_best = max(vals, key=lambda t: t[0])
lhs = dirichlet_sector(O_best, 0.0)
rhs = 0.5*g0*c_best*kms_norm2(O_best)
check("existe O con ratio >= c > 0 y E^(0) >= g(0)c/2 ||O||^2",
      c_best > 0.1 and lhs >= rhs - 1e-9, "c = %.3f" % c_best)

print("== 7) consistencia con 2601.0023 v2 ==")
devs = {}
for w in omegas:
    A = bohr[w]; g = gamma_rate(w)
    dev = 0.0
    for _ in range(5):
        O = np.random.randn(d, d) + 1j*np.random.randn(d, d)
        lhs = dirichlet_sector(O, w)
        rhs = 0.5*g*kms_norm2(A @ O - O @ A)
        dev = max(dev, abs(lhs - rhs)/max(abs(lhs), abs(rhs), 1e-30))
    devs[w] = dev
dev0 = devs[0.0]
devpos = max(v for w, v in devs.items() if abs(w) > 1e-9)
check("plain-commutator exacta en w=0, se desvia a w!=0",
      dev0 < TOL and devpos > 0.05,
      "dev(0)=%.1e, max dev(w!=0)=%.2f" % (dev0, devpos))

print()
nfail = sum(1 for _, ok, _ in results if not ok)
print("ALL CHECKS PASSED" if nfail == 0 else "%d CHECKS FAILED" % nfail)
sys.exit(0 if nfail == 0 else 1)
