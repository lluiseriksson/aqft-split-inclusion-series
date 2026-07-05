#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de verificacion para ai.viXra:2601.0065 v2
"Split-regularized recoverability in Type III AQFT"

Instancia el contrato completo en el regimen Type I (Remark 10.1 / Lemma
10.2 de v2), donde todo es computable:

  1) Instancia CE (Prop 10.3 v2): M = B(H_N (x) H_K), omega0 producto.
     E(X) = Tr_K[(1 (x) rho_K) X] (x) 1_K es CP unital idempotente y
     omega0-preservante (Takesaki satisfecho constructivamente).
  2) Correccion de direccion (Remark 6.6 v2): el adjunto GNS E^# respecto
     de omega0 es la INCLUSION iota: N -> M (a precision de maquina); su
     predual es restriccion, no recovery. El candidato operacional es el
     predual (E)_*: S(N) -> S(M) (reattachment).
  3) D7a = D7b en dimension finita (Lemma 10.2 v2): CMI por combinacion
     de entropias vs diferencia de entropias relativas (regla de cadena
     I(A:BC) - I(A:B)), a precision de maquina.
  4) CMI-DECAY instanciado: TFIM abierto gapped (g=1.5), A=2 sitios,
     C=4 sitios fijos, collar B de anchura w=1..4 (n=7..10): decaimiento
     exponencial con ajuste alpha > 0.
  5) FR-SPLIT con c_FR = 1 en las convenciones D3/D4 (F_B al cuadrado,
     E_rec = -log F_B): Petz reattachment cumple E_rec <= I(A:C|B) en
     el dataset generado, para cada w.
  6) Cadena T1: E_rec <= c_FR K e^{-alpha w} con K, alpha ajustados, y
     la forma de distancia purificada P = sqrt(1 - e^{-E_rec}).

Uso: python3 verify_2601_0065.py
"""
import sys

import numpy as np
from scipy.linalg import eigh, sqrtm

np.random.seed(20260065)
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

def logm_h(rho, floor=1e-14):
    ev, V = eigh(rho)
    ev = np.clip(ev, floor, None)
    return V @ np.diag(np.log(ev)) @ dag(V)

def rel_ent(rho, sig):
    return float(np.trace(rho @ (logm_h(rho) - logm_h(sig))).real)

def ptrace(rho, dims, keep):
    n = len(dims)
    keep = sorted(keep)
    rho = rho.reshape(dims + dims)
    out_axes = [i for i in range(n) if i not in keep]
    for ax in reversed(out_axes):
        rho = np.trace(rho, axis1=ax, axis2=ax + rho.ndim//2)
    dk = int(np.prod([dims[i] for i in keep]))
    return rho.reshape(dk, dk)

# ================= Parte 1: instancia CE y adjunto GNS =================
print("== 1) instancia CE con omega0 producto (Prop 10.3 v2) ==")
dN, dK = 2, 4
dM = dN*dK
rN = np.random.rand(dN); rN /= rN.sum()
rK = np.random.rand(dK); rK /= rK.sum()
rhoN = np.diag(rN); rhoK = np.diag(rK)
rho0 = np.kron(rhoN, rhoK)   # omega0 = Tr[rho0 . ], fiel

def E_ce(X):
    # E(X) = Tr_K[(1 (x) rho_K) X] (x) 1_K  : M -> N c M
    Xr = X.reshape(dN, dK, dN, dK)
    core = np.einsum('l k, i k j l -> i j', rhoK, Xr.transpose(0,1,2,3)) if False else \
           np.einsum('ikjl,kl->ij', Xr, rhoK.T)
    return np.kron(core, np.eye(dK))

# propiedades: unital, idempotente, omega0-preservante, positividad (Choi)
X = np.random.randn(dM, dM) + 1j*np.random.randn(dM, dM)
check("E unital", np.linalg.norm(E_ce(np.eye(dM)) - np.eye(dM)) < 1e-12)
check("E idempotente", np.linalg.norm(E_ce(E_ce(X)) - E_ce(X)) < 1e-12,
      "%.1e" % np.linalg.norm(E_ce(E_ce(X)) - E_ce(X)))
check("omega0 o E = omega0",
      abs(np.trace(rho0 @ E_ce(X)) - np.trace(rho0 @ X)) < 1e-12)
# CP via Choi de E como mapa M -> M
Choi = np.zeros((dM*dM, dM*dM), complex)
for a in range(dM):
    for b in range(dM):
        Eab = np.zeros((dM, dM), complex); Eab[a, b] = 1.0
        Choi += np.kron(Eab, E_ce(Eab))
evC = np.linalg.eigvalsh((Choi + dag(Choi))/2)
check("E completamente positiva (Choi >= 0)", evC.min() > -1e-12,
      "lambda_min %.1e" % evC.min())

print("== 2) adjunto GNS E^# = inclusion iota (Remark 6.6 v2) ==")
# <E(X), Y>_omega0 = <X, iota(Y)>_omega0 para X en M, Y en N
worst = 0.0
for _ in range(30):
    X = np.random.randn(dM, dM) + 1j*np.random.randn(dM, dM)
    YN = np.random.randn(dN, dN) + 1j*np.random.randn(dN, dN)
    Y = np.kron(YN, np.eye(dK))
    lhs = np.trace(rho0 @ dag(E_ce(X)) @ Y)
    rhs = np.trace(rho0 @ dag(X) @ Y)
    worst = max(worst, abs(lhs - rhs))
check("<E(X),Y>_w0 = <X, iota(Y)>_w0 (30 pares)", worst < 1e-12, "%.1e" % worst)
# el predual de E es reattachment: (E)_*(phi_N) = phi_N o E es estado en M
phiN = np.random.rand(dN); phiN /= phiN.sum()
phiN = np.diag(phiN)
# representacion densidad del predual: buscar rho_ext con Tr[rho_ext X] = Tr[(phiN(x)1/dK-like) E(X)]
# como E(X) = E_N(X) (x) 1_K:  phi(E(X)) = Tr[phiN E_N(X)] => rho_ext = ?
# E_N(X) = Tr_K[(1(x)rhoK)X]  =>  Tr[phiN E_N(X)] = Tr[(phiN (x) rhoK) X]
rho_ext = np.kron(phiN, rhoK)
worst2 = 0.0
for _ in range(10):
    X = np.random.randn(dM, dM) + 1j*np.random.randn(dM, dM)
    XN = ptrace(np.kron(phiN, rhoK), [dN, dK], [0])  # no usado; check directo:
    lhs = np.trace(rho_ext @ X)
    core = np.einsum('ikjl,kl->ij', X.reshape(dN,dK,dN,dK), rhoK.T)
    rhs = np.trace(phiN @ core)
    worst2 = max(worst2, abs(lhs - rhs))
check("(E)_*(phi_N) = phi_N (x) rho_K (reattachment explicito)",
      worst2 < 1e-12, "%.1e" % worst2)

# ================= Parte 2: D7a = D7b en dimension finita ===============
print("== 3) D7a = D7b (Lemma 10.2 v2) ==")
# estado aleatorio mixto en 3 qubits (A|B|C = 1|1|1)
dims3 = [2, 2, 2]
Aq = np.random.randn(8, 8) + 1j*np.random.randn(8, 8)
rho3 = Aq @ dag(Aq); rho3 /= np.trace(rho3).real
S_AB = entropy(ptrace(rho3, dims3, [0, 1])); S_BC = entropy(ptrace(rho3, dims3, [1, 2]))
S_B = entropy(ptrace(rho3, dims3, [1])); S_ABC = entropy(rho3)
cmi_a = S_AB + S_BC - S_B - S_ABC
rA = ptrace(rho3, dims3, [0]); rBC = ptrace(rho3, dims3, [1, 2])
rAB = ptrace(rho3, dims3, [0, 1]); rB = ptrace(rho3, dims3, [1])
cmi_b = rel_ent(rho3, np.kron(rA, rBC)) - rel_ent(rAB, np.kron(rA, rB))
check("CMI entropias = CMI diferencia-Araki (estado aleatorio)",
      abs(cmi_a - cmi_b) < 1e-10, "dif %.1e" % abs(cmi_a - cmi_b))

# =========== Partes 3-6: TFIM, CMI-decay, Petz, cadena T1 ==============
print("== 4-6) contrato instanciado: TFIM g=1.5, A=2, C=4, B=w ==")
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

def petz_reattach(rho_AB, rho_B, rho_BC, dA, dB, dC, delta=1e-12):
    # rho~ = (1_A (x) rho_BC^{1/2})(rho_B^{-1/2} rho_AB rho_B^{-1/2} (x) 1_C)(1_A (x) rho_BC^{1/2})
    evB, VB = eigh(rho_B)
    evB = np.clip(evB, delta, None)
    Bm = VB @ np.diag(evB**-0.5) @ dag(VB)
    sBC = sqrtm(rho_BC).astype(complex)
    core = np.kron(np.eye(dA), Bm) @ rho_AB @ np.kron(np.eye(dA), Bm)
    big = np.kron(core, np.eye(dC))
    W = np.kron(np.eye(dA), sBC)
    # reordenar: core esta en A(x)B, big en A(x)B(x)C ; W en A(x)(BC) -- ordenes coinciden
    out = W @ big @ W
    out = (out + dag(out))/2
    ev, V = eigh(out)
    ev = np.clip(ev, 0, None)
    out = V @ np.diag(ev) @ dag(V)
    return out/np.trace(out).real

nA, nC = 2, 4
cmis, erecs = {}, {}
ok_fr = True
for w in [1, 2, 3, 4]:
    n = nA + w + nC
    dims = [2]*n
    psi = tfim_ground(n)
    rho = np.outer(psi, psi.conj())
    A_ = list(range(nA)); B_ = list(range(nA, nA+w)); C_ = list(range(nA+w, n))
    rAB = ptrace(rho, dims, A_+B_); rB = ptrace(rho, dims, B_)
    rBC = ptrace(rho, dims, B_+C_)
    S_AB = entropy(rAB); S_BC = entropy(rBC); S_B = entropy(rB)
    cmi = S_AB + S_BC - S_B  # S_ABC = 0 (puro)
    rho_t = petz_reattach(rAB, rB, rBC, 2**nA, 2**w, 2**nC)
    F = float(np.real(psi.conj() @ rho_t @ psi))   # F_B cuadrado, rho puro
    er = -np.log(max(F, 1e-300))
    cmis[w] = cmi; erecs[w] = er
    ok_fr &= (er <= cmi + 1e-9)
    print("    w=%d: I=%.3e  E_rec=%.3e  (E_rec<=I: %s)" % (w, cmi, er, er <= cmi + 1e-9))
check("FR-SPLIT con c_FR=1: E_rec <= I(A:C|B) para w=1..4 (dataset generado)", ok_fr)

ws = np.array(sorted(cmis))
Iv = np.array([cmis[w] for w in ws])
pos = Iv > 1e-13
alpha_fit, logK = np.polyfit(ws[pos], np.log(Iv[pos]), 1)
alpha = -alpha_fit; K = np.exp(logK)
check("CMI-DECAY: decaimiento exponencial (alpha > 0)", alpha > 0.5,
      "alpha=%.2f K=%.2e" % (alpha, K))
ok_t1 = all(erecs[w] <= K*np.exp(-alpha*w)*(1 + 1e-6) + 1e-12 for w in ws)
check("cadena T1: E_rec <= c_FR K e^{-alpha w} (c_FR=1)", ok_t1)
Pw = np.sqrt(1 - np.exp(-erecs[4]))
F4 = np.exp(-erecs[4])
check("distancia purificada: P^2 = 1 - F_B y decae con w",
      abs(Pw**2 - (1 - F4)) < 1e-12 and Pw < np.sqrt(1 - np.exp(-erecs[1])),
      "P(w=4)=%.1e" % Pw)

print()
nfail = sum(1 for _, ok, _ in results if not ok)
print("ALL CHECKS PASSED" if nfail == 0 else "%d CHECKS FAILED" % nfail)
sys.exit(0 if nfail == 0 else 1)
