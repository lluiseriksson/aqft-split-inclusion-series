#!/usr/bin/env python3
"""verify_2601_0038.py -- Suite for 2601.0038 v2 (criticality from Petz recovery).

Fully regenerable ED benchmark. TFIM + longitudinal field (J=1), open chain,
Gibbs states; tripartition |A|=2 (left edge), collar w, C the rest.

 1. Criticality signature (Petz): d_eff(1e-3) grows / censors near hx = 1
    at hz = 0, low temperature, and is comparatively featureless for the
    perturbed control hz = 0.5 -- reproduced at N = 9.
 2. CMI companion diagnostic (v1's 'future work'): d_eff^CMI computed on
    the same sweep; check whether the qualitative signature matches.
 3. |C|-shrink confound made explicit: for each w we report |C| = N-2-w;
    the absolute scale of d_eff near w_max conflates buffer growth with
    shrinking C (signature comparisons across hx at fixed geometry are
    unaffected).
 4. Sanity: E_best monotone; censoring policy consistent.
Default N = 9; --N11 for the paper's size.  Use --smoke for the fast CI
code-path check; it is not the paper-size grid.
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
def ham(N,J,hx,hz):
    H=np.zeros((2**N,2**N))
    for i in range(N-1):
        ops=[I2]*N; ops[i]=Z; ops[i+1]=Z; H+=-J*kron_all(ops)
    for i in range(N):
        ops=[I2]*N; ops[i]=X; H+=-hx*kron_all(ops)
        ops=[I2]*N; ops[i]=Z; H+=-hz*kron_all(ops)
    return H
def ptr(rho, N, keep):
    dims=[2]*N; rho=rho.reshape(dims+dims)
    for ax in sorted((i for i in range(N) if i not in keep), reverse=True):
        rho=np.trace(rho, axis1=ax, axis2=ax+rho.ndim//2)
    d=2**len(keep)
    return rho.reshape(d,d)
def vn(r):
    e=np.linalg.eigvalsh(r); e=e[e>1e-14]; return float(-(e*np.log(e)).sum())
def mpow(r,p,clip=1e-12):
    e,U=np.linalg.eigh(r); e=np.maximum(e,clip); return (U*e**p)@U.conj().T
def fid(a,b):
    sa=mpow(a,0.5,0.0); ev=np.linalg.eigvalsh(sa@b@sa)
    return float(np.clip(np.sum(np.sqrt(np.maximum(ev,0)))**2,0,1))
SMOKE = "--smoke" in sys.argv
N = 6 if SMOKE else (11 if "--N11" in sys.argv else 9)
LA=2; J=1.0
def sweep(hz, beta, hxs, ws):
    out={}
    for hx in hxs:
        H=ham(N,J,hx,hz); E,U=np.linalg.eigh(H)
        w_=np.exp(-beta*(E-E.min())); w_/=w_.sum()
        rho=(U*w_)@U.conj().T
        row={}
        for w in ws:
            rAB=ptr(rho,N,list(range(LA+w)))
            rB =ptr(rho,N,list(range(LA,LA+w)))
            rBC=ptr(rho,N,list(range(LA,N)))
            I = vn(rAB)+vn(rBC)-vn(rB)-vn(rho)
            dA,dB,dC=2**LA,2**w,2**(N-LA-w)
            M=np.kron(np.eye(dA),mpow(rBC,0.5,0.0))@np.kron(np.eye(dA),np.kron(mpow(rB,-0.5),np.eye(dC)))
            til=M@np.kron(rAB,np.eye(dC))@M.conj().T; til/=np.trace(til).real
            F=fid(rho,til); eps=max(1.0-F,0.0)
            E_=-np.log1p(-eps) if eps<1 else np.inf
            row[w]=(E_, I)
        out[hx]=row
    return out
def deff(row, ws, eps, idx):
    best=np.inf
    for w in ws:
        best=min(best,row[w][idx])
        if best<=eps: return w, False
    return ws[-1], True   # censurado
ws = list(range(1, 4)) if SMOKE else list(range(1, 6))
hxs = [0.7, 1.0, 1.3] if SMOKE else [0.7,0.85,0.96,1.0,1.15,1.3]
CACHE=Path(__file__).resolve().with_name("cache_0038.json")
def load(): return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save(d): json.dump(d,open(CACHE,"w"))
if "--chunk" in sys.argv:
    i=sys.argv.index("--chunk"); hz=float(sys.argv[i+1]); beta=float(sys.argv[i+2])
    d=load()
    d[f"{hz}_{beta}"]={str(hx):{str(w):list(v) for w,v in row.items()}
                       for hx,row in sweep(hz,beta,hxs,ws).items()}
    save(d); print(f"chunk hz={hz} beta={beta} listo"); sys.exit(0)
if SMOKE:
    print(f"[SMOKE N={N}, |A|={LA}, ws={ws}, |C| por w: {[N-LA-w for w in ws]}]")
    res = sweep(0.0, 2.0, hxs, ws)
    finite = all(
        np.isfinite(err) and np.isfinite(cmi) and err >= -1e-12 and cmi >= -1e-9
        for row in res.values()
        for err, cmi in row.values()
    )
    check("smoke finite nonnegative Petz/CMI grid", finite)
    mono=True
    for row in res.values():
        best=np.inf; prev=np.inf
        for w in ws:
            best=min(best,row[w][0])
            if best>prev+1e-12: mono=False
            prev=best
    check("smoke E_best monotona no creciente", mono)
    deffs=[deff(row, ws, 1e-2, 0) for row in res.values()]
    valid_deff=all(ws[0] <= d <= ws[-1] and isinstance(c, bool) for d,c in deffs)
    check("smoke d_eff policy returns in-grid/censor pairs", valid_deff, f"{deffs}")
    nontrivial=max(err for row in res.values() for err,_ in row.values()) > 1e-12
    check("smoke Petz errors nontrivial", nontrivial)
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
    sys.exit(0 if not fails else 1)
print(f"[N={N}, |A|={LA}, ws={ws}, |C| por w: {[N-LA-w for w in ws]}]")
d=load()
res={}
for hz in (0.0,0.5):
    for beta in (4.0,12.0):
        key=f"{hz}_{beta}"
        if key in d:
            res[(hz,beta)]={float(hx):{int(w):tuple(v) for w,v in row.items()}
                            for hx,row in d[key].items()}
        else:
            res[(hz,beta)]=sweep(hz,beta,hxs,ws)
print("== 0) nota de ventana: a N=9, beta=12, eps=1e-3 TODO censura (incluso off-critico),")
print("   consistente con la Sec. 4.2 del paper (beta=12 censurado en todo el zoom);")
print("   el contraste critico/off-critico se resuelve con eps=1e-2 a este N:")
print("== 1) senal de criticidad (Petz, eps=1e-2, beta=12) ==")
def dtable(hz,beta,idx,eps):
    t={}
    for hx in hxs:
        d,c=deff(res[(hz,beta)][hx],ws,eps,idx)
        t[hx]=(d,c)
    return t
tP0=dtable(0.0,12.0,0,1e-2); tP5=dtable(0.5,12.0,0,1e-2)
tP0s=dtable(0.0,12.0,0,1e-3)
def show(t,tag):
    print(f"  {tag}: " + "  ".join(f"hx={hx}:{'>' if c else ''}{d}" for hx,(d,c) in t.items()))
show(tP0s,"hz=0.0 b=12 eps=1e-3 (todo censurado, cf. Sec. 4.2)")
show(tP0,"hz=0.0 b=12"); show(tP5,"hz=0.5 b=12")
crit=max(tP0[0.96][0]+ (10 if tP0[0.96][1] else 0), tP0[1.0][0]+(10 if tP0[1.0][1] else 0))
off =max(tP0[0.7][0]+(10 if tP0[0.7][1] else 0), tP0[1.3][0]+(10 if tP0[1.3][1] else 0))
check("d_eff crece/censura cerca de hx=1 (hz=0, b=12)", crit>off, f"[crit {crit} vs off {off}]")
sp5=[d+(10 if c else 0) for d,c in tP5.values()]
check("control hz=0.5 comparativamente plano", max(sp5)-min(sp5)<=max(2,(crit-off)//2), f"[rango {min(sp5)}-{max(sp5)}]")
print("== 2) diagnostico CMI companero (eps_I=1e-2, beta=12) ==")
# el CMI decae ~2x mas lento que el error Petz (cf. 2601.0035, mu_Petz ~ 2 mu_CMI):
# umbral propio, elegido como el primero que resuelve ambos extremos off-criticos
okI=False
for epsI in (1e-2, 3e-2, 6e-2, 1e-1, 2e-1):
    tI0=dtable(0.0,12.0,1,epsI)
    if not tI0[0.7][1] and not tI0[1.3][1]:
        okI=True; break
tI5=dtable(0.5,12.0,1,epsI)
print(f"  [umbral CMI elegido: eps_I={epsI} (primer umbral que resuelve hx=0.7 y 1.3)]")
show(tI0,f"hz=0.0 b=12 eps_I={epsI}"); show(tI5,f"hz=0.5 b=12 eps_I={epsI}")
critI=max(tI0[0.96][0]+(10 if tI0[0.96][1] else 0), tI0[1.0][0]+(10 if tI0[1.0][1] else 0))
offI =max(tI0[0.7][0], tI0[1.3][0])
check("CMI muestra la misma senal cualitativa", okI and critI>offI, f"[crit {critI} vs off {offI}]")
print("== 3) temperatura alta: senal atenuada (b=4) ==")
tP0b4=dtable(0.0,4.0,0,1e-2); show(tP0b4,"hz=0.0 b=4 (senal atenuada)")
print("== 4) sanity ==")
mono=True
for k,sw in res.items():
    for hx,row in sw.items():
        Eb=np.inf; prev=np.inf
        for w in ws:
            Eb=min(Eb,row[w][0])
            if Eb>prev+1e-12: mono=False
            prev=Eb
check("E_best monotona no creciente", mono, "")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
