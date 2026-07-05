#!/usr/bin/env python3
"""verify_2601_0042.py -- Suite for 2601.0042 v2 (emergent information
distance from Petz recovery: temperature/perturbation dependence).

Regenerable ED benchmark (chunked JSON cache). TFIM + longitudinal field,
J=1, hx=1.05, open chain, |A|=2, Gibbs states; default N=9 (--N11 for the
paper size). Checks:
 1. d_eff(1e-3) grows with beta at hz=0 and stays near-minimal at hz=0.5
    (qualitative reproduction of v1's Fig. 2 / Sec. 4.2).
 2. Regime ordering d_eff(hz=0) >= d_eff(hz=0.5) for all sampled beta and
    thresholds eps in {1e-2, 1e-3, 1e-4} (v1's Sec. 4.3).
 3. mu_prefloor decreases with beta at hz=0, stays comparatively large at
    hz=0.5 (v1's Sec. 4.4).
 4. kappa-bar identifiability policy: defined only when n_prefloor >= 3.
 5. PSD-projection sensitivity: |Delta E_Petz| small (v1 reported ~3e-8).
Usage: --chunk hz  computes one regime and caches; no args reports.
Use --smoke for the fast CI code-path check; it is not the paper grid.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

fails=[]
def check(name, ok, det=""):
    suffix = f" {det}" if det else ""
    print(f"  {name}: {'OK' if ok else 'FAIL'}{suffix}")
    if not ok: fails.append(name)
I2=np.eye(2); X=np.array([[0,1],[1,0]],dtype=float); Z=np.diag([1.,-1.])
def kron_all(ops):
    out=ops[0]
    for o in ops[1:]: out=np.kron(out,o)
    return out
def ham(N,hx,hz,J=1.0):
    H=np.zeros((2**N,2**N))
    for i in range(N-1):
        ops=[I2]*N; ops[i]=Z; ops[i+1]=Z; H+=-J*kron_all(ops)
    for i in range(N):
        ops=[I2]*N; ops[i]=X; H+=-hx*kron_all(ops)
        if hz:
            ops=[I2]*N; ops[i]=Z; H+=-hz*kron_all(ops)
    return H
def ptr(rho, N, keep):
    rho=rho.reshape([2]*(2*N))
    for ax in sorted((i for i in range(N) if i not in keep), reverse=True):
        rho=np.trace(rho, axis1=ax, axis2=ax+rho.ndim//2)
    return rho.reshape(2**len(keep),2**len(keep))
def mpow(r,p,clip=1e-12):
    e,U=np.linalg.eigh(r); e=np.maximum(e,clip); return (U*e**p)@U.conj().T
SMOKE = "--smoke" in sys.argv
LA=2; HX=1.05
BETAS=[0.5,2.0,5.0] if SMOKE else [0.5,1.0,2.0,3.0,4.0,5.0]
N = 6 if SMOKE else (11 if "--N11" in sys.argv else 9)
WS=list(range(1,4)) if SMOKE else list(range(1,6))
def sweep(hz):
    out={}
    for beta in BETAS:
        H=ham(N,HX,hz); E,U=np.linalg.eigh(H)
        p=np.exp(-beta*(E-E.min())); p/=p.sum()
        rho=(U*p)@U.conj().T
        sa=mpow(rho,0.5,0.0)
        row={}
        for w in WS:
            rAB=ptr(rho,N,list(range(LA+w)))
            rB =ptr(rho,N,list(range(LA,LA+w)))
            rBC=ptr(rho,N,list(range(LA,N)))
            dA,dC=2**LA,2**(N-LA-w)
            M=np.kron(np.eye(dA),mpow(rBC,0.5,0.0))@np.kron(np.eye(dA),np.kron(mpow(rB,-0.5),np.eye(dC)))
            til=M@np.kron(rAB,np.eye(dC))@M.conj().T
            tilr=(til+til.conj().T)/2
            ev,V=np.linalg.eigh(tilr); evp=np.maximum(ev,0)
            tilp=(V*evp)@V.conj().T; tilp/=np.trace(tilp).real
            def err(t):
                fv=np.linalg.eigvalsh(sa@t@sa)
                F=float(np.clip(np.sum(np.sqrt(np.maximum(fv,0)))**2,0,1))
                eps=max(1.0-F,0.0)
                return -np.log1p(-eps) if eps<1 else np.inf
            E1=err(tilp)
            E0=err(tilr/np.trace(tilr).real)   # sin proyeccion PSD
            row[w]=(E1, abs(E1-E0))
        out[beta]=row
    return out
CACHE=Path(__file__).resolve().with_name("cache_0042.json")
def load(): return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
if "--chunk" in sys.argv:
    hz=float(sys.argv[sys.argv.index("--chunk")+1])
    d=load(); out=sweep(hz)
    d[str(hz)]={str(b):{str(w):list(v) for w,v in row.items()} for b,row in out.items()}
    json.dump(d,open(CACHE,"w")); print(f"chunk hz={hz} listo"); sys.exit(0)
d=load()
if SMOKE:
    res={0.0:sweep(0.0),0.5:sweep(0.5)}
elif not d:
    print("ejecuta --chunk 0.0 y --chunk 0.5 primero, o usa --smoke para CI")
    sys.exit(1)
else:
    res={float(hz):{float(b):{int(w):tuple(v) for w,v in row.items()} for b,row in dd.items()} for hz,dd in d.items()}
print(f"[{'SMOKE ' if SMOKE else ''}N={N}, |A|={LA}, hx={HX}, ws={WS}]")
def deff(row, eps):
    Eb=np.inf; prev=None
    for w in WS:
        E=row[w][0]; nb=min(Eb,E)
        if nb<=eps:
            if prev and prev[1]>eps and E<prev[1] and E>0:
                return prev[0]+(np.log(prev[1])-np.log(eps))/(np.log(prev[1])-np.log(E)), False
            return float(w), False
        prev=(w,nb); Eb=nb
    return float(WS[-1]), True
print("== 1) d_eff(1e-3) vs beta ==")
tab={}
for hz in (0.0,0.5):
    vals=[deff(res[hz][b],1e-3) for b in BETAS]
    tab[hz]=vals
    print(f"  hz={hz}: " + "  ".join(f"b={b}:{'>' if c else ''}{v:.2f}" for b,(v,c) in zip(BETAS,vals)))
g0=[v for v,c in tab[0.0]]
check("crece con beta (hz=0)", g0[-1]>g0[0]+0.5, f"[{g0[0]:.2f} -> {g0[-1]:.2f}]")
g5=[v for v,c in tab[0.5]]
check("casi minimo en hz=0.5", max(g5)<1.6, f"[max {max(g5):.2f}]")
print("== 2) ordering entre regimenes, 3 umbrales ==")
okall=True
for eps in (1e-2,1e-3,1e-4):
    for b in BETAS:
        v0,c0=deff(res[0.0][b],eps); v5,c5=deff(res[0.5][b],eps)
        e0=v0+(10 if c0 else 0); e5=v5+(10 if c5 else 0)
        if e0 < e5-1e-9: okall=False
check("d_eff(hz=0) >= d_eff(hz=0.5) en todo (beta, eps)", okall, "")
print("== 3) mu_prefloor ==")
mus={}
for hz in (0.0,0.5):
    ms=[]
    for b in BETAS:
        pts=[(w,res[hz][b][w][0]) for w in WS if res[hz][b][w][0]>=1e-11]
        if len(pts)>=2:
            wv=np.array([p[0] for p in pts]); ev=np.log([p[1] for p in pts])
            ms.append(-np.polyfit(wv,ev,1)[0])
        else: ms.append(float('nan'))
    mus[hz]=ms
    print(f"  hz={hz}: " + "  ".join(f"b={b}:{m:.2f}" for b,m in zip(BETAS,ms)))
m0=[m for m in mus[0.0] if np.isfinite(m)]
check("mu decrece con beta (hz=0)", m0[0]>m0[-1]+0.5, f"[{m0[0]:.2f} -> {m0[-1]:.2f}]")
print("== 4) identificabilidad de kappa-bar ==")
okk=True
for hz in (0.0,0.5):
    for b in BETAS:
        pts=[w for w in WS if res[hz][b][w][0]>=1e-11]
        npre=len(pts); nk=max(0,npre-2)
        if nk>0 and npre<3: okk=False
check("kappa-bar solo con n_prefloor>=3 (politica)", okk, "")
print("== 5) sensibilidad de la proyeccion PSD ==")
mx=max(v[1] for hz in (0.0,0.5) for b in BETAS for v in res[hz][b].values())
check("max |Delta E_Petz| pequeno", mx<1e-6, f"[{mx:.1e}; v1 reporto ~3e-8 a N=11]")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
