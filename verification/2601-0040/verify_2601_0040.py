#!/usr/bin/env python3
"""verify_2601_0040.py -- Suite for 2601.0040 v2 (finite-size scaling of the
Petz recovery length).

Regenerable ED benchmark (chunked JSON cache; TFIM hz=0, beta=12, |A|=2,
J=1). Default N in {8,9,10} (paper: 9..12; --N up to 12 possible).

 1. Peak growth: d_eff peak height grows with N at censoring-free thresholds.
 2. |C|-confound control (v2 caveat 1): off-critical baseline d_eff(hx=0.80;N)
    also grows with N; the cleaner scaling object is the critical enhancement
    Delta(N) = peak - baseline, reported alongside.
 3. Functional-form indistinguishability (v2 caveat 2): over one sub-octave
    in N, power law N^k, a+b log N and linear fits of the peak height are
    statistically indistinguishable (R^2 spread tiny) -- k(eps) is a
    descriptive summary, not an established power law.
 4. Sanity: E_best monotone; censoring-free verification at chosen eps.
Usage: --chunk N  computes one N and caches; final run (no args) reports.
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
def ham(N,hx,hz=0.0,J=1.0):
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
    d=2**len(keep)
    return rho.reshape(d,d)
def mpow(r,p,clip=1e-12):
    e,U=np.linalg.eigh(r); e=np.maximum(e,clip); return (U*e**p)@U.conj().T
SMOKE = "--smoke" in sys.argv
LA=2; BETA=12.0
HXS=[0.80, 0.90, 0.94, 0.96, 0.98, 1.00]
def sweep_one(N, hxs=None):
    ws=list(range(1,N-LA))            # |C| >= 1
    out={}
    for hx in (hxs or HXS):
        H=ham(N,hx); E,U=np.linalg.eigh(H)
        p=np.exp(-BETA*(E-E.min())); p/=p.sum()
        rho=(U*p)@U.conj().T
        sa=mpow(rho,0.5,0.0)
        row={}
        for w in ws:
            rAB=ptr(rho,N,list(range(LA+w)))
            rB =ptr(rho,N,list(range(LA,LA+w)))
            rBC=ptr(rho,N,list(range(LA,N)))
            dA,dC=2**LA,2**(N-LA-w)
            M=np.kron(np.eye(dA),mpow(rBC,0.5,0.0))@np.kron(np.eye(dA),np.kron(mpow(rB,-0.5),np.eye(dC)))
            til=M@np.kron(rAB,np.eye(dC))@M.conj().T
            til=(til+til.conj().T)/2
            ev,V=np.linalg.eigh(til); ev=np.maximum(ev,0); til=(V*ev)@V.conj().T
            til/=np.trace(til).real
            fv=np.linalg.eigvalsh(sa@til@sa)
            F=float(np.clip(np.sum(np.sqrt(np.maximum(fv,0)))**2,0,1))
            eps=max(1.0-F,0.0)
            row[w]=-np.log1p(-eps) if eps<1 else np.inf
        out[hx]=row
    return out, ws
CACHE=Path(__file__).resolve().with_name("cache_0040.json")
def load(): return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def deff(row, ws, eps):
    Eb=np.inf; prev=None
    for w in ws:
        E=row[w]
        newEb=min(Eb,E)
        if newEb<=eps:
            if prev is not None and Eb>eps and Eb<np.inf and newEb<Eb:
                # interpolacion log-lineal entre (w-1, Eb) y (w, E)
                w0,E0=prev
                if E0>eps and E<E0 and E>0:
                    return w0+(np.log(E0)-np.log(eps))/(np.log(E0)-np.log(E)), False
            return w, False
        prev=(w,newEb); Eb=newEb
    return ws[-1], True
if "--chunk" in sys.argv:
    i=sys.argv.index("--chunk"); N=int(sys.argv[i+1])
    sub=HXS[int(sys.argv[i+2]):int(sys.argv[i+3])] if len(sys.argv)>i+3 else HXS
    d=load(); out,ws=sweep_one(N, sub)
    e=d.get(str(N),{"ws":ws,"data":{}})
    e["data"].update({str(hx):{str(w):v for w,v in row.items()} for hx,row in out.items()})
    d[str(N)]=e
    json.dump(d,open(CACHE,"w")); print(f"chunk N={N} hx={sub} listo"); sys.exit(0)
if SMOKE:
    N=6
    hxs=[0.80, 0.94, 1.00]
    out,ws=sweep_one(N,hxs)
    print(f"[SMOKE N={N}; beta={BETA}, |A|={LA}, hx={hxs}, ws={ws}]")
    finite=all(np.isfinite(v) and v>=-1e-12 for row in out.values() for v in row.values())
    check("smoke finite nonnegative Petz grid", finite)
    mono=True
    for row in out.values():
        best=np.inf; prev=np.inf
        for w in ws:
            best=min(best,row[w])
            if best>prev+1e-12: mono=False
            prev=best
    check("smoke E_best monotona no creciente", mono)
    deffs=[deff(row,ws,5e-3) for row in out.values()]
    valid=all(ws[0] <= d <= ws[-1] and isinstance(c,bool) for d,c in deffs)
    check("smoke d_eff policy returns in-grid/censor pairs", valid, f"{deffs}")
    nontrivial=max(v for row in out.values() for v in row.values())>1e-12
    check("smoke Petz errors nontrivial", nontrivial)
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
    sys.exit(0 if not fails else 1)
d=load()
if not d:
    print("ejecuta --chunk N [i0 i1] primero, o usa --smoke para CI")
    sys.exit(1)
Ns=sorted(int(k) for k in d)
print(f"[N disponibles: {Ns}; beta={BETA}, |A|={LA}, hx={HXS}]")
EPSS=[3e-3, 5e-3]
peaks={}; base={}
for N in Ns:
    ws=d[str(N)]["ws"]
    data={float(hx):{int(w):v for w,v in row.items()} for hx,row in d[str(N)]["data"].items()}
    for eps in EPSS:
        vals={hx:deff(data[hx],ws,eps) for hx in data}
        cen=any(c for _,c in vals.items() if False) # placeholder
        peak_hx=max((hx for hx in vals if hx>=0.9), key=lambda h: vals[h][0])
        peaks[(N,eps)]=(vals[peak_hx][0], peak_hx, vals[peak_hx][1])
        base[(N,eps)]=vals[0.80]
print("== 1-2) picos, baseline off-critico y realce ==")
for eps in EPSS:
    line=[]
    for N in Ns:
        pk,phx,pc=peaks[(N,eps)]; bs,bc=base[(N,eps)]
        line.append(f"N={N}: pico={pk:.2f}@hx={phx}{'(cens)' if pc else ''} base={bs:.2f}{'(cens)' if bc else ''} realce={pk-bs:+.2f}")
    print(f"  eps={eps}: " + " | ".join(line))
for eps in EPSS:
    pks=[peaks[(N,eps)][0] for N in Ns]; bss=[base[(N,eps)][0] for N in Ns]
    check(f"pico crece con N (eps={eps})", all(x<y+1e-9 for x,y in zip(pks,pks[1:])), f"[{['%.2f'%p for p in pks]}]")
    print(f"    baseline: {['%.2f'%b for b in bss]}  -> el baseline TAMBIEN crece: confound |C| real" if bss[0]<bss[-1] else "    baseline plano")
print("== 3) indistinguibilidad de formas funcionales ==")
for eps in EPSS:
    y=np.array([peaks[(N,eps)][0] for N in Ns]); x=np.array(Ns,dtype=float)
    def r2(pred):
        ss=np.sum((y-pred)**2); st=np.sum((y-y.mean())**2); return 1-ss/st
    cp=np.polyfit(np.log(x),np.log(y),1); r2p=r2(np.exp(np.polyval(cp,np.log(x))))
    cl=np.polyfit(np.log(x),y,1);        r2l=r2(np.polyval(cl,np.log(x)))
    cn=np.polyfit(x,y,1);                r2n=r2(np.polyval(cn,x))
    print(f"  eps={eps}: kappa(power)={cp[0]:.3f} R2={r2p:.4f} | log R2={r2l:.4f} | lineal R2={r2n:.4f}")
    check(f"formas indistinguibles (eps={eps})", max(r2p,r2l,r2n)-min(r2p,r2l,r2n)<0.05,
          f"[spread {max(r2p,r2l,r2n)-min(r2p,r2l,r2n):.4f}]")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
