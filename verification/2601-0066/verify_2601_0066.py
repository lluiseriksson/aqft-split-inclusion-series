#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de verificacion para ai.viXra:2601.0066 v2
"Typed Pipeline for Recoverability-Rate-Power Links"

Instancia numericamente todo lo instanciable del contrato:
  1) L1 (Lemma 4.3): 1 - F <= -log F en malla fina de F in (0,1].
  2) Disciplina de convenciones (Remark 4.5 / Apendice A): con F cuadrada,
     -log F = -2 log f_raiz => c_FR = 1 en la convencion fijada; usar la
     raiz sin ajustar la constante produce el factor 2 que la serie
     corrigio cuatro veces.
  3) Example 6.1: estado Markov exacto (rho = rho_{A Bl} (x) rho_{Br C}):
     I(A:C|B) = 0 y reconstruccion Petz-type con F = 1, ambos a precision
     de maquina.
  4) Carril cerrado A-CMI + T2 => T3 instanciado: TFIM abierto gapped
     (g=1.5), A=2 sitios, C=3 fijos, collar B de anchura w=1..4:
     I(A:C|B) ~ K e^{-alpha w} (alpha > 0), reconstruccion Petz-type con
     E_rec <= c_FR I (c_FR = 1) en cada w del dataset generado, y la
     cadena completa 1 - F <= E_rec <= c_FR K e^{-alpha w}.
  5) Example 6.2: semigrupo de dephasing puro (d=3, tasas por par
     distintas), Delta el pinching de la misma base: C_Delta(T_t rho)
     monotona decreciente, Cdot_loss >= 0, y envolventes kappa_up /
     kappa_down con spread >> 1 sobre la familia de pares (la
     direccionalidad del Remark 3.10: los floors no se heredan de los
     ceilings).

Uso: python3 verify_2601_0066.py
"""
import sys

import numpy as np
from scipy.linalg import eigh, sqrtm

np.random.seed(20260066)
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    suffix = " [%s]" % detail if detail else ""
    print("  %s: %s%s" % (name, "OK" if ok else "FAIL", suffix))

def dag(X): return X.conj().T

def entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return float(-np.sum(ev*np.log(ev)))

def ptrace(rho, dims, keep):
    n = len(dims)
    keep = sorted(keep)
    rho = rho.reshape(dims + dims)
    out_axes = [i for i in range(n) if i not in keep]
    for ax in reversed(out_axes):
        rho = np.trace(rho, axis1=ax, axis2=ax + rho.ndim//2)
    dk = int(np.prod([dims[i] for i in keep]))
    return rho.reshape(dk, dk)

def fidelity_sq(rho, sig):
    # F = ||sqrt(rho) sqrt(sig)||_1^2 (convencion cuadrada D4.1)
    sr = sqrtm(rho).astype(complex)
    M = sr @ sig @ sr
    ev = np.linalg.eigvalsh((M + dag(M))/2)
    ev = np.clip(ev, 0, None)
    return float(np.sum(np.sqrt(ev))**2)

def petz_reattach(rho_AB, rho_B, rho_BC, dA, dB, dC, delta=1e-12):
    evB, VB = eigh(rho_B)
    evB = np.clip(evB, delta, None)
    Bm = VB @ np.diag(evB**-0.5) @ dag(VB)
    sBC = sqrtm(rho_BC).astype(complex)
    core = np.kron(np.eye(dA), Bm) @ rho_AB @ np.kron(np.eye(dA), Bm)
    big = np.kron(core, np.eye(dC))
    W = np.kron(np.eye(dA), sBC)
    out = W @ big @ W
    out = (out + dag(out))/2
    ev, V = eigh(out)
    ev = np.clip(ev, 0, None)
    out = V @ np.diag(ev) @ dag(V)
    return out/np.trace(out).real

print("== 1) L1: 1 - F <= -log F (Lemma 4.3) ==")
Fs = np.linspace(1e-6, 1.0, 200001)
gap = -np.log(Fs) - (1 - Fs)
check("1 - F <= -log F en 2e5 puntos de (0,1]", gap.min() >= -1e-12,
      "min gap %.1e (en F=1)" % gap.min())

print("== 2) disciplina de convenciones: c_FR = 1 con F cuadrada ==")
# -log F = -2 log f con f = sqrt(F): identico; el drift aparece al mezclar
Fs2 = np.random.rand(1000)*0.999 + 1e-4
lhs = -np.log(Fs2); rhs = -2.0*np.log(np.sqrt(Fs2))
check("-log F = -2 log f_raiz (identidad de alineacion)",
      np.max(np.abs(lhs - rhs)) < 1e-12, "%.1e" % np.max(np.abs(lhs - rhs)))
check("mezclar convenciones produce factor 2 (trampa documentada)",
      np.allclose(-np.log(np.sqrt(Fs2)), 0.5*lhs), "0.5x exacto")

print("== 3) Example 6.1: Markov exacto => recovery perfecto ==")
# B = Bl (x) Br ; rho = rho_{A Bl} (x) rho_{Br C}, dims: A=2, Bl=2, Br=2, C=2
def rand_dm(d):
    A = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    r = A @ dag(A); return r/np.trace(r).real
rho_ABl = rand_dm(4); rho_BrC = rand_dm(4)
rho = np.kron(rho_ABl, rho_BrC)          # orden A, Bl, Br, C
dims = [2, 2, 2, 2]
S_AB = entropy(ptrace(rho, dims, [0,1,2])); S_BC = entropy(ptrace(rho, dims, [1,2,3]))
S_B = entropy(ptrace(rho, dims, [1,2])); S_ABC = entropy(rho)
cmi = S_AB + S_BC - S_B - S_ABC
rAB = ptrace(rho, dims, [0,1,2]); rB = ptrace(rho, dims, [1,2]); rBC = ptrace(rho, dims, [1,2,3])
rho_t = petz_reattach(rAB, rB, rBC, 2, 4, 2)
F1 = fidelity_sq(rho, rho_t)
check("I(A:C|B) = 0 (estructura HJPW)", abs(cmi) < 1e-10, "I=%.1e" % abs(cmi))
check("Petz-type F = 1 (recovery perfecto)", abs(F1 - 1) < 1e-9,
      "1-F=%.1e" % abs(1 - F1))

print("== 4) carril cerrado instanciado: TFIM g=1.5, A=2, C=3, B=w ==")
def tfim_ground(n, g=1.5):
    dim = 2**n
    Z = np.array([[1, 0], [0, -1]], float); Xp = np.array([[0, 1], [1, 0]], float)
    H = np.zeros((dim, dim))
    def op(o, i):
        M = np.eye(1)
        for k in range(n):
            M = np.kron(M, o if k == i else np.eye(2))
        return M
    for i in range(n-1):
        H -= op(Z, i) @ op(Z, i+1)
    for i in range(n):
        H -= g*op(Xp, i)
    ev, V = eigh(H)
    return V[:, 0]

nA, nC = 2, 3
cmis, erecs, fids = {}, {}, {}
ok_fr, ok_l1chain = True, True
for w in [1, 2, 3, 4]:
    n = nA + w + nC
    dims = [2]*n
    psi = tfim_ground(n)
    rho = np.outer(psi, psi.conj())
    A_ = list(range(nA)); B_ = list(range(nA, nA+w)); C_ = list(range(nA+w, n))
    rAB = ptrace(rho, dims, A_+B_); rB = ptrace(rho, dims, B_); rBC = ptrace(rho, dims, B_+C_)
    cmi = entropy(rAB) + entropy(rBC) - entropy(rB)   # S_ABC = 0
    rho_t = petz_reattach(rAB, rB, rBC, 2**nA, 2**w, 2**nC)
    F = float(np.real(psi.conj() @ rho_t @ psi))
    er = -np.log(max(F, 1e-300))
    cmis[w] = cmi; erecs[w] = er; fids[w] = F
    ok_fr &= (er <= cmi + 1e-9)
    ok_l1chain &= (1 - F <= er + 1e-12)
    print("    w=%d: I=%.3e  E_rec=%.3e  1-F=%.3e" % (w, cmi, er, 1-F))
check("T2 con c_FR=1: E_rec <= I(A:C|B) en w=1..4 (dataset generado)", ok_fr)
ws = np.array(sorted(cmis)); Iv = np.array([cmis[w] for w in ws])
a_fit, logK = np.polyfit(ws, np.log(Iv), 1)
alpha = -a_fit; K = np.exp(logK)
check("A-CMI instanciada: I ~ K e^{-alpha w}, alpha > 0", alpha > 0.5,
      "alpha=%.2f K=%.2e" % (alpha, K))
ok_t3 = all(erecs[w] <= K*np.exp(-alpha*w)*(1+1e-6) + 1e-12 for w in ws)
check("T3: E_rec <= c_FR K e^{-alpha w}", ok_t3)
check("cadena L1: 1 - F <= E_rec <= c_FR K e^{-alpha w}", ok_l1chain)

print("== 5) Example 6.2: dephasing puro, envolventes kappa ==")
d = 3
gam = {(0, 1): 2.0, (0, 2): 0.02, (1, 2): 0.5}   # tasas por par, muy distintas
def T_t(rho, t):
    out = rho.copy().astype(complex)
    for (i, j), g in gam.items():
        out[i, j] *= np.exp(-g*t); out[j, i] *= np.exp(-g*t)
    return out
def C_delta(rho):
    lr_ev, V = eigh(rho)
    lr_ev = np.clip(lr_ev, 1e-16, None)
    lr = V @ np.diag(np.log(lr_ev)) @ dag(V)
    ld = np.diag(np.log(np.clip(np.diag(rho).real, 1e-16, None)))
    return float(np.trace(rho @ (lr - ld)).real)
# monotonia a lo largo de la trayectoria (10 estados, 20 tiempos)
mono_ok = True
for _ in range(10):
    r0 = rand_dm(d)
    ts = np.linspace(0, 3, 21)
    vals = [C_delta(T_t(r0, t)) for t in ts]
    mono_ok &= all(vals[k+1] <= vals[k] + 1e-10 for k in range(20))
check("C_Delta(T_t rho) monotona decreciente (10 estados)", mono_ok)
# Cdot_loss >= 0 y envolventes sobre la familia de pares de coherencia
h = 1e-6
rates = {}
for (i, j) in gam:
    r0 = np.diag([1.0/d]*d).astype(complex)
    r0[i, j] = 0.25; r0[j, i] = 0.25
    C0 = C_delta(r0)
    Cd = -(C_delta(T_t(r0, h)) - C0)/h
    rates[(i, j)] = Cd/C0
kup = max(rates.values()); kdn = min(rates.values())
check("Cdot_loss >= 0 en la familia", kdn > 0, "min r=%.2e" % kdn)
check("spread kappa_up/kappa_down >> 1 (direccionalidad, Remark 3.10)",
      kup/kdn > 20.0, "x%.0f (%.2e .. %.2e)" % (kup/kdn, kdn, kup))

print()
nfail = sum(1 for _, ok, _ in results if not ok)
print("ALL CHECKS PASSED" if nfail == 0 else "%d CHECKS FAILED" % nfail)
sys.exit(0 if nfail == 0 else 1)
