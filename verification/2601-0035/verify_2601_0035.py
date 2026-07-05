#!/usr/bin/env python3
"""verify_2601_0035.py -- Suite for 2601.0035 v2 (non-Gaussian bridge).

Fully regenerable ED benchmark (no archived data needed): Gibbs states of
the TFIM with longitudinal field, H = -J sum Z Z - hx sum X - hz sum Z
(J = 1, hx = 1, open chain), shielded tripartition LA = 2.

 1. CMI decay: I(A:C|B) vs collar width w, integrable (hz=0) and
    non-integrable (hz=0.5), with censored fits (>=3 pre-floor points).
 2. Explicit Petz recovery and the FR scale, CORRECTED factor:
    ratio r := -log F(rho, Petz(rho_AB)) / I(A:C|B).
    v2 check: r < 1 on the finite grid (no overshoot); the v1 'overshoot'
    (reported vs I/2) is an artifact: r_half := 2r exceeds 1 exactly where
    v1 reported r ~ 1.23.
 3. Arithmetic: 1 - F <= -log F.
Default N = 9 (fast); --N11 uses the paper's N = 11 (slower).
"""
import sys

import numpy as np

rng = np.random.default_rng(26010035)
fails = []
def check(name, ok, det=""):
    suffix = f" {det}" if det else ""
    print(f"  {name}: {'OK' if ok else 'FAIL'}{suffix}")
    if not ok: fails.append(name)
I2=np.eye(2); X=np.array([[0,1],[1,0]],dtype=float); Z=np.diag([1.,-1.])
def kron_all(ops):
    out=ops[0]
    for o in ops[1:]: out=np.kron(out,o)
    return out
def ham(N,J,hx,hz):
    H=np.zeros((2**N,2**N))
    for i in range(N-1):
        ops=[I2]*N; ops[i]=Z; ops[i+1]=Z; H+=-J*kron_all(ops)
    for i in range(N):
        ops=[I2]*N; ops[i]=X; H+=-hx*kron_all(ops)
        ops=[I2]*N; ops[i]=Z; H+=-hz*kron_all(ops)
    return H
def ptr(rho, dims, keep):
    n=len(dims); rho=rho.reshape(dims+dims)
    for ax in sorted((i for i in range(n) if i not in keep), reverse=True):
        rho=np.trace(rho, axis1=ax, axis2=ax+rho.ndim//2)
    d=int(np.prod([dims[i] for i in keep])) if keep else 1
    return rho.reshape(d,d)
def vn(r):
    e=np.linalg.eigvalsh(r); e=e[e>1e-14]; return float(-(e*np.log(e)).sum())
def mpow(r,p,clip=1e-12):
    e,U=np.linalg.eigh(r); e=np.maximum(e,clip); return (U*e**p)@U.conj().T
def fid(a,b):
    sa=mpow(a,0.5,0.0)
    ev=np.linalg.eigvalsh(sa@b@sa)
    return float(np.clip(np.sum(np.sqrt(np.maximum(ev,0)))**2,0,1))

N = 11 if "--N11" in sys.argv else 9
LA=2; J,hx = 1.0, 1.0
print(f"[N={N}, LA={LA}, J={J}, hx={hx}]")
def run(hz, beta, ws):
    H=ham(N,J,hx,hz); E,U=np.linalg.eigh(H)
    w_=np.exp(-beta*(E-E.min())); w_/=w_.sum()
    rho=(U*w_)@U.conj().T
    out={}
    for w in ws:
        dims=[2**LA, 2**w, 2**(N-LA-w)]
        r3=rho.reshape([2]*(2*N))
        rAB=ptr(rho,[2]*N,list(range(LA+w)))
        rB =ptr(rho,[2]*N,list(range(LA,LA+w)))
        rBC=ptr(rho,[2]*N,list(range(LA,N)))
        I = vn(rAB)+vn(rBC)-vn(rB)-vn(rho)
        # Petz: M = (I_A ox s_BC^{1/2}) (I_A ox s_B^{-1/2} ox I_C)
        dA,dB,dC = 2**LA, 2**w, 2**(N-LA-w)
        M = np.kron(np.eye(dA), mpow(rBC,0.5,0.0)) @ np.kron(np.eye(dA), np.kron(mpow(rB,-0.5), np.eye(dC)))
        til = M @ np.kron(rAB, np.eye(dC)) @ M.conj().T
        til = til/np.trace(til).real
        F = fid(rho, til)
        eps = max(1.0-F, 0.0)
        nlf = -np.log1p(-eps) if eps<1 else np.inf
        out[w]=(I, nlf)
    return out
grid=[]
for hz in (0.0, 0.5):
    for beta in (2.0, 3.0, 5.0):
        ws=(1,2,3,4)
        res=run(hz,beta,ws)
        cmi_pts=[(w,I) for w,(I,nl) in res.items() if I>=1e-12]
        pre=[(w,I,nl) for w,(I,nl) in res.items() if nl>=1e-12 and I>=1e-12]
        if len(cmi_pts)>=3:
            wv=np.array([p[0] for p in cmi_pts]); iv=np.log([p[1] for p in cmi_pts])
            mu=-np.polyfit(wv,iv,1)[0]
        else: mu=float('nan')
        rmax=max((nl/I) for _,I,nl in pre) if pre else float('nan')
        grid.append((hz,beta,mu,rmax))
        print(f"  hz={hz} beta={beta}: mu_CMI={mu:.3f}  max r=-logF/I = {rmax:.3f}"
              f"  (media escala v1: {2*rmax:.3f})")
rall=[g[3] for g in grid if np.isfinite(g[3])]
check("ratio FR corregido r < 1 en todo el grid", max(rall)<1.0, f"[max {max(rall):.3f}]")
check("con la media escala de v1 hay 'overshoots' (artefacto)", any(2*r>1 for r in rall),
      f"[{sum(1 for r in rall if 2*r>1)} casos]")
mus=[g[2] for g in grid if np.isfinite(g[2])]
check("decaimiento CMI exponencial (mu > 0)", all(m>0 for m in mus), f"[min mu {min(mus):.2f}]")
xs=rng.uniform(1e-9,1,1000)
check("1-F <= -log F", np.all(1-xs<=-np.log(xs)+1e-12), "")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
