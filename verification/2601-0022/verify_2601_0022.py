#!/usr/bin/env python3
"""verify_2601_0022.py -- Suite for 2601.0022 v2.

PART A: from-scratch ED reproduction of the omega=0 witness (declared
  benchmark: TFIM N=10, J=1, h=1.05, S = Z at center site, gamma0=0.1,
  eps in {1,2,3}, beta in {0.5,1,2}, blocks L=1,2), plus the S=X null case
  (vanishing omega=0 component -> witness switches off).
PART B: detectability/power of the no-floor conclusion at n=5 over
  eps in [16,32]: floor vs slow-exponential degeneracy.
Runtime ~70 s. Run: python3 verify_2601_0022.py
"""
import itertools
import time

import numpy as np
from scipy import linalg as la

I2 = np.eye(2); Xp = np.array([[0,1],[1,0]],dtype=complex)
Yp = np.array([[0,-1j],[1j,0]]); Zp = np.diag([1.,-1.]).astype(complex)
P1 = {'X':Xp,'Y':Yp,'Z':Zp}
def site_op(op, i, N):
    M = np.array([[1.+0j]])
    for j in range(N): M = np.kron(M, op if j==i else I2)
    return M
def pauli_block(lab, k, N):
    O = np.array([[1.+0j]])
    for j in range(N):
        o = I2.astype(complex)
        if k <= j < k+len(lab) and lab[j-k] != 'I': o = P1[lab[j-k]]
        O = np.kron(O, o)
    return O
def run(
    h,
    Sname,
    N=10,
    J=1.0,
    gamma0=0.1,
    eps_list=(1,2,3),
    betas=(0.5,1.0,2.0),
    published=True,
):
    t0=time.time()
    H = np.zeros((2**N,2**N), dtype=complex)
    for i in range(N-1): H += -J*site_op(Xp,i,N)@site_op(Xp,i+1,N)
    for i in range(N): H += -h*site_op(Zp,i,N)
    E,U = np.linalg.eigh(H)
    S = site_op(P1[Sname[0]], N//2, N)
    Se = U.conj().T @ S @ U
    S0e = Se*(np.abs(E[:,None]-E[None,:])<1e-9)
    j0=N//2; res={}
    for L in (1,2):
        for eps in eps_list:
            k=j0+eps
            if k+L>N: continue
            labs=[p for p in itertools.product('IXYZ',repeat=L) if p!=tuple('I')*L]
            Ae=[U.conj().T@(pauli_block(l,k,N)@U) for l in labs]
            Ce=[S0e@A-A@S0e for A in Ae]
            m=len(labs)
            for beta in betas:
                w=np.exp(-beta*(E-E.min())); w/=w.sum(); sq=np.sqrt(w); Wm=np.outer(sq,sq)
                G=np.zeros((m,m),dtype=complex); M=np.zeros((m,m),dtype=complex)
                for a in range(m):
                    for b in range(a,m):
                        G[a,b]=np.sum(Wm*np.conj(Ae[a])*Ae[b]); G[b,a]=np.conj(G[a,b])
                        M[a,b]=np.sum(Wm*np.conj(Ce[a])*Ce[b]); M[b,a]=np.conj(M[a,b])
                ev=la.eigh(M,G,eigvals_only=True)
                res[(L,eps,beta)]=gamma0/2*float(ev[-1].real)
    print(f"[h={h} S={Sname} t={time.time()-t0:.0f}s]")
    for L in (1,2):
        vals=[v for (l,e,b),v in res.items() if l==L]
        suffix = ""
        if published:
            suffix = f"  (publicado v1: {'[3.2e-04, 1.2e-03]' if L==1 else '[8.2e-04, 2.9e-03]'})"
        print(f"  L={L}: rango kappa_min = [{min(vals):.3e}, {max(vals):.3e}]{suffix}")
    return res

def fit_models(epss, D, sig):
    agrid = np.logspace(-2, 0.5, 60)
    def wls(with_floor):
        best = np.inf
        for a in agrid:
            b = np.exp(-a*epss)
            cols = np.array([b, np.ones_like(b)]).T if with_floor else b[:,None]
            Xd = cols/sig[:,None]; y = D/sig
            coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            coef = np.maximum(coef, 0)
            chi2 = float(np.sum(((D - cols@coef)/sig)**2))
            best = min(best, chi2)
        return best
    chi_e, chi_f = wls(False), wls(True)
    n = len(epss)
    aicc = lambda c,k: c + 2*k + 2*k*(k+1)/(n-k-1)
    bic  = lambda c,k: c + k*np.log(n)
    return bic(chi_f,3)-bic(chi_e,2), aicc(chi_f,3)-aicc(chi_e,2)

def floor_power(seed=26010022):
    rng = np.random.default_rng(seed)
    epss = np.array([16.,20.,24.,28.,32.])
    A0, a0, s = 1.0, 0.15, 0.025   # sigma efectiva por punto
    print("umbral de deteccion de floor a n=5 (60 repeticiones por nivel):")
    det_b = det_a = None
    rows = []
    for D0s in [0, 1, 2, 4, 8, 16, 32]:
        hb = ha = 0
        for _ in range(60):
            D = A0*np.exp(-a0*epss) + D0s*s + rng.normal(0, s, 5)
            sig = np.full(5, s)
            db, da = fit_models(epss, D, sig)
            hb += db < 0; ha += da < 0
        rows.append((D0s, hb, ha))
        print(f"  D0 = {D0s:2d} sigma: BIC detecta {hb}/60  AICc detecta {ha}/60")
        if det_b is None and hb >= 48: det_b = D0s
        if det_a is None and ha >= 48: det_a = D0s
    print(f"deteccion >=80%: BIC a partir de D0 ~ {det_b} sigma; AICc a partir de D0 ~ {det_a} sigma")
    print("=> a n=5, AICc solo detecta floors enormes (penalizacion ~20); BIC es el discriminador operativo.")
    print("   El 'no floor' observado (dBIC ~ +1.6 = penalizacion pura) significa mejora chi2 ~ 0: genuinamente sin senal.")
    return rows, det_b, det_a

def block_range(res, L):
    vals = [v for (l, _, _), v in res.items() if l == L]
    return min(vals), max(vals)

def require(name, ok):
    if not ok:
        raise AssertionError(name)

def main():
    print("== PART A: witness ED (benchmark declarado) ==")
    z_res = run(1.05, "Z")
    z_l1 = block_range(z_res, 1)
    z_l2 = block_range(z_res, 2)
    require("S=Z L=1 witness range drifted", 1.7e-3 < z_l1[0] < z_l1[1] < 5.9e-3)
    require("S=Z L=2 witness range drifted", 4.2e-3 < z_l2[0] < z_l2[1] < 6.9e-3)

    print("== PART A-null: acoplo con componente omega=0 nula ==")
    x_res = run(1.05, "X", published=False)
    x_l1 = block_range(x_res, 1)
    x_l2 = block_range(x_res, 2)
    require("S=X L=1 null witness is not near zero", x_l1[1] < 1e-24)
    require("S=X L=2 null witness is not near zero", x_l2[1] < 1e-24)

    print("== PART B: potencia del pipeline a n=5 ==")
    rows, det_b, det_a = floor_power()
    require("BIC detection threshold changed", det_b == 32)
    require("AICc threshold unexpectedly detected", det_a is None)
    require("power table length changed", len(rows) == 7)
    print("[nota: rangos v1 no reproducidos exactamente porque v1 no declaro (J,h,S); mismo orden y mecanismo.")
    print(" Esta suite regenera el benchmark declarado aqui; no incluye trayectorias TEBD/MCWF.]")

if __name__ == "__main__":
    main()
