#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de verificacion para ai.viXra:2601.0115 v2
"Algebraic Entropy and CMI in a Tiny Gauge-Invariant Truncated
Hilbert Space"

v1 dependia de un descs.pkl NUNCA especificado (irreproducible tal como
se publico). v2 fija una BASE CANONICA autocontenida, impresa en el
paper y regenerada aqui: L=4 celdas, 3 cortes con b_s in {0,1},
x_s = (b_s, b_s) (top=bot), mu_1=(b_1,), mu_2=(b_1 xor b_2,),
mu_3=(b_2 xor b_3,), mu_4=(b_3,). dim = 8.

Checks:
  1) Inyectividad desc -> (alpha, kR, kC) para las 6 regiones usadas
     => S_alg es entropia de von Neumann genuina de una descomposicion
     en sectores (lema v2); norma sum w = 1 en todo el sweep.
  2) Lema del estado uniforme: S_alg exactas (log2, 2log2, 2log2,
     log2, log2) y CMI_w1 = CMI_w2 = 0 EXACTOS en psi_uniforme
     (la "casi-cuantizacion" de v1 es exacta en el limite).
  3) Lema strong-mixing: alpha_w=0, grafo kNN conexo, t_mix ->
     infinito => psi_0 -> uniforme; overlap y PR -> dim monotonos en t.
  4) Sweep completo (grid de v1): PR ~ 8 y CMI < 1e-4 en (0, 100);
     S ~ n log 2 a <1e-3; SSA I >= -1e-12 en TODO el sweep; purity
     S(1..4) ~ 0.
  5) Baseline Haar (seccion 9.4.1 de v1, ejecutada): 2000 estados Haar
     en C^8: mediana de I_sum ORDENES por encima del punto
     strong-mixing => el "CMI pequeno" es no trivial.
  6) Ablacion k_nn in {1, 2, 5} (seccion 9.4.2): conclusiones
     estructurales robustas cuando el grafo queda conexo (declarado si
     se desconecta).
  7) Grid fino de t_mix (seccion 9.4.3): transicion desde el regimen
     diagonal (S ~ 0) al Laplaciano-dominado.

Uso: python3 verify_2601_0115.py
"""
import itertools
import sys

import numpy as np
from collections import defaultdict

np.random.seed(20260115)
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    suffix = " [%s]" % detail if detail else ""
    print("  %s: %s%s" % (name, "OK" if ok else "FAIL", suffix))

LOG2 = np.log(2.0)

# ---------------- base canonica (v2, autocontenida) ----------------
class Desc:
    __slots__ = ("x", "mu")
    def __init__(self, x, mu):
        self.x = x; self.mu = mu

def canonical_descs():
    out = []
    for b1, b2, b3 in itertools.product((0, 1), repeat=3):
        x = ((b1, b1), (b2, b2), (b3, b3))
        mu = ((b1,), (b1 ^ b2,), (b2 ^ b3,), (b3,))
        out.append(Desc(x, mu))
    return out

descs = canonical_descs()
dim = len(descs)
check("base canonica: dim = 8", dim == 8)

# ---------------- pipeline identico al de v1 (Apendice B) ----------
def j2_to_jj1(b): return 0.0 if int(b) == 0 else 0.75

def electric_energy_proxy(d):
    E = 0.0
    for xs in d.x:
        E += j2_to_jj1(xs[0]) + j2_to_jj1(xs[1])
    for mup in d.mu:
        for b in mup:
            E += j2_to_jj1(b)
    return E

def flatten_bits(d):
    bits = []
    for xs in d.x:
        bits += [int(xs[0]), int(xs[1])]
    for mup in d.mu:
        bits += [int(b) for b in mup]
    return tuple(bits)

def alpha_of_desc(d, i, j, L=4):
    groups = []
    if i > 1: groups.append(d.x[i-2])
    if j < L: groups.append(d.x[j-1])
    return tuple(groups)

def keyR_of_desc(d, i, j):
    return (d.mu[i-1:j], d.x[i-1:(j-1)])

def keyC_of_desc(d, i, j, L=4):
    mu_L = d.mu[0:(i-1)] if i > 1 else ()
    x_L = d.x[0:(i-2)] if i > 1 else ()
    mu_R = d.mu[j:L] if j < L else ()
    x_R = d.x[j:(L-1)] if j < L else ()
    return (mu_L, x_L, mu_R, x_R)

EPS_SVD = 1e-15

def S_alg_interval(psi, i, j):
    trips = defaultdict(list)
    mapR = defaultdict(dict); mapC = defaultdict(dict)
    for k, d in enumerate(descs):
        a = alpha_of_desc(d, i, j)
        r = mapR[a].setdefault(keyR_of_desc(d, i, j), len(mapR[a]))
        c = mapC[a].setdefault(keyC_of_desc(d, i, j), len(mapC[a]))
        trips[a].append((r, c, psi[k]))
    S, norm = 0.0, 0.0
    for a, tc in trips.items():
        M = np.zeros((len(mapR[a]), len(mapC[a])), complex)
        for r, c, amp in tc:
            M[r, c] += amp
        s = np.linalg.svd(M, compute_uv=False)
        w = s.real**2
        norm += float(np.sum(w))
        w = w[w > EPS_SVD]
        S -= float(np.sum(w*np.log(w)))
    assert abs(norm - 1.0) < 1e-10, "norma Schmidt"
    return float(S)

def cmis(psi):
    S12 = S_alg_interval(psi, 1, 2); S23 = S_alg_interval(psi, 2, 3)
    S2 = S_alg_interval(psi, 2, 2); S13 = S_alg_interval(psi, 1, 3)
    S24 = S_alg_interval(psi, 2, 4); S14 = S_alg_interval(psi, 1, 4)
    return {"S12": S12, "S23": S23, "S2": S2, "S13": S13, "S24": S24,
            "S14": S14, "I1": S12 + S23 - S2 - S13,
            "I2": S13 + S24 - S23 - S14}

print("== 1) inyectividad y estructura vN (lema v2) ==")
inj_ok = True
for (i, j) in [(1, 2), (2, 3), (2, 2), (1, 3), (2, 4), (1, 4)]:
    keys = set()
    for d in descs:
        k = (alpha_of_desc(d, i, j), keyR_of_desc(d, i, j),
             keyC_of_desc(d, i, j))
        inj_ok &= (k not in keys)
        keys.add(k)
check("desc -> (alpha, kR, kC) inyectivo en las 6 regiones", inj_ok,
      "=> S_alg = entropia vN de una descomposicion en sectores")

print("== 2) lema del estado uniforme: cuantizacion y CMI exactas ==")
uni = np.ones(dim)/np.sqrt(dim)
r = cmis(uni)
targets = {"S12": LOG2, "S23": 2*LOG2, "S2": 2*LOG2, "S13": LOG2,
           "S24": LOG2, "S14": 0.0}
ok_q = all(abs(r[k] - t) < 1e-12 for k, t in targets.items())
check("S_alg(uniforme) exactas: (log2, 2log2, 2log2, log2, log2, 0)",
      ok_q, "err max %.1e" % max(abs(r[k]-t) for k, t in targets.items()))
check("CMI(uniforme) = 0 exacta (I1 e I2)",
      abs(r["I1"]) < 1e-12 and abs(r["I2"]) < 1e-12,
      "I1=%.1e I2=%.1e" % (r["I1"], r["I2"]))

# ---------------- Hamiltoniano y sweep --------------------------------
bitstrings = [flatten_bits(d) for d in descs]
D = np.zeros((dim, dim), int)
for i in range(dim):
    for j in range(i+1, dim):
        D[i, j] = D[j, i] = sum(x != y for x, y in
                                zip(bitstrings[i], bitstrings[j]))

def build_knn_laplacian(D, k_nn, alpha_w):
    n = D.shape[0]
    k = max(1, min(k_nn, n-1))
    A = np.zeros((n, n))
    for i in range(n):
        neigh = sorted((int(D[i, j]), j) for j in range(n) if j != i)
        for d_, j in neigh[:k]:
            w = np.exp(-alpha_w*d_)
            A[i, j] = max(A[i, j], w); A[j, i] = max(A[j, i], w)
    L = np.diag(A.sum(1)) - A
    return L, int((A > 0).sum()//2)

def connected(L):
    ev = np.linalg.eigvalsh(L)
    return int(np.sum(ev < 1e-10)) == 1

g = 1.0
HE = np.diag([0.5*g*g*electric_energy_proxy(d) for d in descs])

def ground(alpha_w, t_mix, k_nn=3):
    L, edges = build_knn_laplacian(D, k_nn, alpha_w)
    H = HE + t_mix*L
    ev, V = np.linalg.eigh(H)
    psi = V[:, 0]
    PR = 1.0/np.sum(np.abs(psi)**4)
    return psi, PR, float(ev[1]-ev[0]), edges, L

print("== 3) lema strong-mixing: psi_0 -> uniforme monotono en t ==")
ovs, prs = [], []
for t in [1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]:
    psi, PR, _, _, L3 = ground(0.0, t)
    ovs.append(abs(np.dot(uni, psi)))
    prs.append(PR)
check("grafo kNN (k=3, alpha=0) conexo", connected(L3))
check("overlap con uniforme monotono -> 1",
      all(ovs[i+1] >= ovs[i] - 1e-12 for i in range(len(ovs)-1))
      and ovs[-1] > 0.99999, "ov: %.5f -> %.7f" % (ovs[0], ovs[-1]))
check("PR -> dim", prs[-1] > 7.999, "PR(t=1000)=%.5f" % prs[-1])

print("== 4) sweep completo (grid de v1) ==")
ssa_min, rows = np.inf, []
for alpha_w in [1.0, 0.5, 0.2, 0.1, 0.0]:
    for t_mix in [1.0, 3.0, 10.0, 30.0, 100.0]:
        psi, PR, Delta, edges, _ = ground(alpha_w, t_mix)
        rr = cmis(psi)
        ssa_min = min(ssa_min, rr["I1"], rr["I2"])
        rows.append((alpha_w, t_mix, PR, Delta, rr))
check("purity: S(1..4) ~ 0 en todo el sweep",
      all(abs(rr["S14"]) < 1e-10 for _, _, _, _, rr in rows))
check("SSA empirica: I >= -1e-12 en todo el sweep", ssa_min >= -1e-12,
      "min I = %.1e" % ssa_min)
strong = [row for row in rows if row[0] == 0.0 and row[1] == 100.0][0]
_, _, PRs, Ds, rs = strong
check("punto strong-mixing (0, 100): PR ~ 8 y CMI_sum < 1e-4",
      PRs > 7.99 and (rs["I1"] + rs["I2"]) < 1e-4,
      "PR=%.4f Isum=%.2e" % (PRs, rs["I1"] + rs["I2"]))
check("cuantizacion en (0,100): |S - n log2| < 1e-3",
      all(abs(rs[k] - t) < 1e-3 for k, t in targets.items()))

print("== 5) baseline Haar (seccion 9.4.1 de v1, ejecutada) ==")
Isums = []
for _ in range(2000):
    z = np.random.randn(dim) + 1j*np.random.randn(dim)
    z /= np.linalg.norm(z)
    rh = cmis(z)
    Isums.append(rh["I1"] + rh["I2"])
Isums = np.array(Isums)
med = float(np.median(Isums)); p05 = float(np.percentile(Isums, 5))
ratio = med/(rs["I1"] + rs["I2"])
check("mediana Haar de I_sum >> I_sum(strong-mixing)", ratio > 100,
      "mediana=%.3f, p05=%.3f, ratio x%.0f" % (med, p05, ratio))
check("SSA empirica tambien en Haar (min >= -1e-12)",
      float(Isums.min()) >= -1e-12, "min %.1e" % float(Isums.min()))

print("== 6) ablacion k_nn (seccion 9.4.2) ==")
for k in [1, 2, 5]:
    L_, edges = build_knn_laplacian(D, k, 0.0)
    conn = connected(L_)
    if conn:
        psi, PR, _, _, _ = ground(0.0, 100.0, k)
        rk = cmis(psi)
        okk = PR > 7.9 and (rk["I1"] + rk["I2"]) < 1e-3
        check("k=%d (conexo): PR ~ 8 y CMI pequena" % k, okk,
              "PR=%.3f Isum=%.1e" % (PR, rk["I1"] + rk["I2"]))
    else:
        check("k=%d: grafo NO conexo (declarado; lema no aplica)" % k,
              True, "edges=%d" % edges)

print("== 7) grid fino de t_mix (seccion 9.4.3) ==")
tvals = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
S23s = []
for t in tvals:
    if t == 0.0:
        # estado fundamental de HE diagonal: config todo-ceros
        psi = np.zeros(dim); psi[np.argmin(np.diag(HE))] = 1.0
    else:
        psi, _, _, _, _ = ground(0.0, t)
    S23s.append(cmis(psi)["S23"])
check("transicion: S23 crece de ~0 (diagonal) a ~2log2 (mixing)",
      S23s[0] < 1e-12 and S23s[-1] > 1.2 and
      all(S23s[i+1] >= S23s[i] - 1e-9 for i in range(len(S23s)-1)),
      "S23: %.1e -> %.4f" % (S23s[0], S23s[-1]))

print()
nfail = sum(1 for _, ok, _ in results if not ok)
print("ALL CHECKS PASSED" if nfail == 0 else "%d CHECKS FAILED" % nfail)
sys.exit(0 if nfail == 0 else 1)
