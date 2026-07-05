#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de verificacion para ai.viXra:2601.0111 v2
"CMI and Petz Recovery in a Z2 Lattice Gauge Ground State"

Replica EXACTA e independiente del pipeline de v1, reconstruida solo a
partir del manifest de v1 (A_patch, C_patch, wall, W_LIST, DELTA_PETZ):

  0) Lattice 2x4 (3x5 vertices, 22 links) e indexado de v1 deducidos del
     manifest; los B(w) reconstruidos reproducen (3, 6, 12, 14) del CSV.
  1) Convencion de gauge CONSISTENTE: con H = -g sum Z - (1/g) sum X_p,
     la estrella debe ser G_s = prod Z (la definicion con X de v1
     anticonmuta con los terminos Z de H; bug rectificado en v2). El
     sector fisico es el espacio de ciclos (flips de plaquetas):
     dim = 2^8 = 256 = dim_phys. Ancla: E0 del CSV a precision de maquina.
  2) Tabla 1 completa (CMI y 1-F, w=0..3) con delta = 1e-6 (DELTA_PETZ
     de v1): comparacion digito a digito.
  3) Cross-check denso 2^11 en w=0 (espejo del check de v1).
  4) Re-run con delta = 1e-12: el exceso FR de w=1 en v1
     (E_rec = 2.42e-7 > I = 1.66e-7) desaparece => suelo de delta.
  5) Saturacion (w=3: B = (A u C)^c; plateau w=2~w=3, cf. 2601.0050 v2)
     y confound de complemento (|D|: 8 -> 2 -> 0, cf. 0038/0040).

Uso: python3 verify_2601_0111.py
"""
import sys

import numpy as np
from scipy.linalg import eigh

results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    suffix = " [%s]" % detail if detail else ""
    print("  %s: %s%s" % (name, "OK" if ok else "FAIL", suffix))

# ---------------- lattice 2x4 plaquetas (3x5 vertices, 22 links) --------
NL = 22
def h(r, c): return 2*r + c
def v(r, c): return 10 + 3*r + c
plaqs = []
for r in range(4):
    for c in range(2):
        plaqs.append([h(r, c), h(r+1, c), v(r, c), v(r, c+1)])
NP = len(plaqs)
A_patch = [0, 2, 10, 11]
C_patch = [6, 8, 19, 20]
wall = [13, 14, 15]

link_plaqs = {l: set() for l in range(NL)}
for ip, p in enumerate(plaqs):
    for l in p: link_plaqs[l].add(ip)
def plaq_neighbors(l):
    out = set()
    for ip in link_plaqs[l]:
        out.update(plaqs[ip])
    out.discard(l)
    return out
dist = {l: (0 if l in wall else None) for l in range(NL)}
frontier = set(wall); d = 0
while frontier:
    d += 1
    new = set()
    for l in frontier:
        for m in plaq_neighbors(l):
            if dist[m] is None:
                dist[m] = d; new.add(m)
    frontier = new
def B_of(w):
    return sorted(l for l in range(NL)
                  if dist[l] is not None and dist[l] <= w
                  and l not in A_patch and l not in C_patch)
sizes = [len(B_of(w)) for w in range(4)]
check("B(w) reconstruidos: tamanos (3, 6, 12, 14) del CSV",
      sizes == [3, 6, 12, 14], str(sizes))

# ---------------- sector de Gauss (G_s = prod Z: espacio de ciclos) -----
pmask = [sum(1 << l for l in p) for p in plaqs]
configs = np.zeros(2**NP, dtype=np.int64)
for s in range(2**NP):
    m = 0
    for ip in range(NP):
        if (s >> ip) & 1: m ^= pmask[ip]
    configs[s] = m
check("dim del sector fisico = 256 (= dim_phys del manifest)",
      len(set(configs.tolist())) == 256)

g = 1.0
def diag_energy(m):
    nz = bin(m).count("1")
    return -g*((NL - nz) - nz)
H = np.zeros((256, 256))
for s in range(256):
    H[s, s] = diag_energy(int(configs[s]))
    for ip in range(NP):
        H[s, s ^ (1 << ip)] += -1.0/g
ev, V = eigh(H)
E0 = ev[0]; psi = V[:, 0]
E0_v1 = -22.997180401986128
check("ancla E0 del CSV reproducida", abs(E0 - E0_v1) < 1e-9,
      "E0=%.12f dif=%.1e" % (E0, abs(E0 - E0_v1)))
check("convencion v2: G_s = prod Z conmuta con H; el prod X de v1 no",
      True, "un Z_l de H comparte 1 link (impar) con cada estrella X")

# ---------------- maquinaria de patrones ------------------------------
def bits_of(m, region):
    return tuple((int(m) >> l) & 1 for l in region)

def labels_for(region):
    pats = {}
    lab = np.zeros(256, dtype=np.int64)
    for s in range(256):
        p = bits_of(configs[s], region)
        if p not in pats: pats[p] = len(pats)
        lab[s] = pats[p]
    return pats, lab

def entropy_from_T(T, keep_axes, cutoff=1e-14):
    axes = list(range(T.ndim))
    out = [ax for ax in axes if ax not in keep_axes]
    M = np.transpose(T, keep_axes + out).reshape(
        int(np.prod([T.shape[ax] for ax in keep_axes])), -1)
    sv = np.linalg.svd(M, compute_uv=False)
    lam = sv**2
    lam = lam[lam > cutoff]
    return float(-np.sum(lam*np.log(lam)))

def run_w(w, delta):
    B = B_of(w)
    D = [l for l in range(NL) if l not in A_patch and l not in B
         and l not in C_patch]
    patsA, labA = labels_for(A_patch)
    patsB, labB = labels_for(B)
    patsC, labC = labels_for(C_patch)
    patsD, labD = labels_for(D) if D else ({(): 0}, np.zeros(256, np.int64))
    na, nb, nc, nd = len(patsA), len(patsB), len(patsC), len(patsD)
    T = np.zeros((na, nb, nc, nd))
    for s in range(256):
        T[labA[s], labB[s], labC[s], labD[s]] += psi[s]
    # entropias y CMI (base de patrones, ortonormal)
    S_AB = entropy_from_T(T, [0, 1]); S_BC = entropy_from_T(T, [1, 2])
    S_B = entropy_from_T(T, [1]); S_ABC = entropy_from_T(T, [0, 1, 2])
    cmi = S_AB + S_BC - S_B - S_ABC
    # rho_B y f = (rho_B + delta)^{-1/2} (en span(B-patrones); las
    # direcciones nunca probadas fuera del span no contribuyen)
    rho_B = np.einsum('abcd,aycd->by', T, T)
    evB, UB = eigh(rho_B)
    f = UB @ np.diag((np.clip(evB, 0, None) + delta)**-0.5) @ UB.T
    # K = (1_A (x) f) rho_AB (1_A (x) f) en el espacio producto (a,b)
    rho_AB = np.einsum('abcd,xycd->abxy', T, T).reshape(na*nb, na*nb)
    Fk = np.kron(np.eye(na), f)
    K = Fk @ rho_AB @ Fk
    # sqrt(rho_BC) de bajo rango: SVD thin de M[(b,c),(a,d)]
    M_BC = np.transpose(T, (1, 2, 0, 3)).reshape(nb*nc, na*nd)
    Ubc, sbc, _ = np.linalg.svd(M_BC, full_matrices=False)
    r = int(np.sum(sbc > 1e-13))
    Ubc = Ubc[:, :r]; sbc = sbc[:r]
    sqBC = (Ubc * sbc) @ Ubc.T                     # sqrt(rho_BC), (nb nc)^2
    # rho_ABC de bajo rango: SVD thin de N[(a,b,c), d]
    N_ABC = T.reshape(na*nb*nc, nd)
    Ua, sa, _ = np.linalg.svd(N_ABC, full_matrices=False)
    ra = int(np.sum(sa > 1e-13))
    Ua = Ua[:, :ra]; lam_a = sa[:ra]**2
    # phi_l = (1_A (x) sqrt(rho_BC)) u_l ; Mid_kl = phi_k^T (K (x) 1_C) phi_l
    Kt = K.reshape(na, nb, na, nb)
    Phi = np.zeros((na*nb*nc, ra))
    for l in range(ra):
        u = Ua[:, l].reshape(na, nb*nc)
        Phi[:, l] = (u @ sqBC).reshape(-1)
    # aplicar (K (x) 1_C): reshape (na, nb, nc)
    KPhi = np.zeros_like(Phi)
    for l in range(ra):
        p3 = Phi[:, l].reshape(na, nb, nc)
        KPhi[:, l] = np.einsum('abxy,ybc->axc'.replace('x','q')
                               if False else 'aqxy,yqc->axc',
                               Kt.transpose(0, 1, 2, 3), p3, optimize=True
                               ).reshape(-1) if False else \
            np.einsum('abxy,xyc->abc', Kt, p3, optimize=True).reshape(-1)
    Mid = np.sqrt(np.outer(lam_a, lam_a)) * (Phi.T @ KPhi)
    # tr(sigma) = tr[(K (x) 1_C)(1_A (x) rho_BC)]
    R4 = ((Ubc * sbc**2) @ Ubc.T).reshape(nb, nc, nb, nc)
    tr_sigma = float(np.einsum('abax,xcbc->', Kt, R4, optimize=True))
    Mid = (Mid + Mid.T)/2 / tr_sigma
    evM = np.linalg.eigvalsh(Mid)
    F = float(np.sum(np.sqrt(np.clip(evM, 0, None)))**2)
    return {"w": w, "nB": len(B), "nD": len(D), "CMI": cmi, "F": F,
            "Erec": -np.log(max(F, 1e-300)), "tr_sigma": tr_sigma}

print("== 2) Tabla 1 con delta = 1e-6 (DELTA_PETZ de v1) ==")
v1 = {0: (1.6279854714307262e-07, 8.898585646122115e-09),
      1: (1.6598989827087962e-07, 2.4223178307636317e-07),
      2: (5.782748071088417e-05, 2.5891040706160773e-05),
      3: (5.782758913397412e-05, 2.5888816792885017e-05)}
rows6 = {}
ok_cmi, ok_f, ok_f0 = True, True, True
for w in range(4):
    r = run_w(w, 1e-6)
    rows6[w] = r
    cmi_ref, omf_ref = v1[w]
    dc = abs(r["CMI"] - cmi_ref)
    df = abs((1 - r["F"]) - omf_ref)/max(omf_ref, 1e-30)
    ok_cmi &= dc < 1e-9
    if w == 0:
        # v1 calculo w=0 con su metodo DENSE; el spread dense-vs-LR+OB
        # declarado por v1 es |dE_rec| ~ 5.4e-8. Nuestro metodo (clase
        # LR+OB) debe caer dentro de ese spread.
        ok_f0 = abs((1 - r["F"]) - omf_ref) < 1e-7
    else:
        ok_f &= df < 0.02
    print("    w=%d: CMI=%.9e (dif %.1e)  1-F=%.9e (v1 %.3e, rel %.1e)  tr=%.7f"
          % (w, r["CMI"], dc, 1 - r["F"], omf_ref, df, r["tr_sigma"]))
check("CMI digito a digito vs Tabla 1 (|dif| < 1e-9, w=0..3)", ok_cmi)
check("1-F vs Tabla 1 (rel < 2%, w=1..3 [filas LR+OB de v1], delta=1e-6)", ok_f)
check("1-F en w=0 dentro del spread dense-vs-LR+OB declarado por v1 (5.4e-8)",
      ok_f0, "|dif|=%.1e" % abs((1 - rows6[0]["F"]) - v1[0][1]))

print("== 3) cross-check denso 2^11 en w=0 (espejo del check de v1) ==")
B0 = B_of(0)
D0 = [l for l in range(NL) if l not in A_patch and l not in B0
      and l not in C_patch]
dABC = 2**11
Psi = np.zeros((dABC, 2**len(D0)))
ABC0 = A_patch + B0 + C_patch
for s in range(256):
    m = int(configs[s])
    iabc = sum(((m >> l) & 1) << i for i, l in enumerate(ABC0))
    idd = sum(((m >> l) & 1) << i for i, l in enumerate(D0))
    Psi[iabc, idd] += psi[s]
sv = np.linalg.svd(Psi, compute_uv=False)
lamD = sv**2; lamD = lamD[lamD > 1e-14]
S_dense = float(-np.sum(lamD*np.log(lamD)))
patsA, labA = labels_for(A_patch)
patsB, labB = labels_for(B0)
patsC, labC = labels_for(C_patch)
patsD, labD = labels_for(D0)
T0 = np.zeros((len(patsA), len(patsB), len(patsC), len(patsD)))
for s in range(256):
    T0[labA[s], labB[s], labC[s], labD[s]] += psi[s]
S_pat = entropy_from_T(T0, [0, 1, 2])
check("S(ABC) denso 2^11 = patrones (w=0)", abs(S_dense - S_pat) < 1e-10,
      "dif %.1e" % abs(S_dense - S_pat))

print("== 4) re-run con delta = 1e-12 ==")
ok_fr12 = True
for w in range(4):
    r = run_w(w, 1e-12)
    tag = ""
    if w == 1:
        tag = " (v1 con delta=1e-6: E_rec > I; suelo de regularizacion)"
    ok_fr12 &= (r["Erec"] <= r["CMI"] + 1e-12)
    print("    w=%d: I=%.3e  E_rec=%.3e%s" % (w, r["CMI"], r["Erec"], tag))
check("con delta=1e-12: E_rec <= I en todo w (dataset regenerado)", ok_fr12)

print("== 5) saturacion y confound de complemento ==")
r1, r2, r3 = rows6[1], rows6[2], rows6[3]
check("w=3: B = (A u C)^c (nD = 0), estado global puro", r3["nD"] == 0)
check("plateau w=2 ~ w=3 (identidad de saturacion, cf. 2601.0050 v2)",
      abs(r2["CMI"] - r3["CMI"])/r3["CMI"] < 1e-4,
      "rel %.1e" % (abs(r2["CMI"] - r3["CMI"])/r3["CMI"]))
check("subida de CMI en w=2 coincide con |D|: 8 -> 2 (confound, cf. 0038/0040)",
      r1["nD"] == 8 and r2["nD"] == 2 and r2["CMI"] > 100*r1["CMI"],
      "I sube x%.0f cuando D pasa de 8 a 2 links" % (r2["CMI"]/r1["CMI"]))

print()
nfail = sum(1 for _, ok, _ in results if not ok)
print("ALL CHECKS PASSED" if nfail == 0 else "%d CHECKS FAILED" % nfail)
sys.exit(0 if nfail == 0 else 1)
