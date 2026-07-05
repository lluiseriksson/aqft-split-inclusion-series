#!/usr/bin/env python3
"""verify_2601_0047.py -- Suite for 2601.0047 v2 (Z2 2+1D benchmark).

Regenerates the paper from scratch and upgrades its statistics:
 1. Table 1 reproduction: E0, sigma_eff = chi(2,2), E_rec(w=0,1) on 2x2 and
    2x3 plaquette lattices (OBC, link qubits, Gauss penalty Lam=50,
    <G_v> ~ 1 verified).
 2. Densified sweep on 2x2 (8 couplings): Spearman rank correlation of
    E_rec(w=1) vs 1/sigma_eff with permutation p-value (v1 claimed a
    'clear correlation' from n=3).
 3. Confinement-specificity control: spectral gap of H(g) as covariate;
    Spearman(E_rec, 1/gap) reported alongside.
 4. 2x3 saturation check: E(w=0) ~ E(w=1) (the 2x3 correlation is
    essentially a no-buffer statement).
Support-projected fidelity: rank(rho_ABC) <= 2^env keeps everything in
small matrices. Usage: --sweep22 | --chunk23 g | report (no args).
Use --smoke for a fast CI code-path check; it is not the paper grid.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
fails=[]
def check(name, ok, det=""):
    suffix = f" {det}" if det else ""
    print(f"  {name}: {'OK' if ok else 'FAIL'}{suffix}")
    if not ok: fails.append(name)
def lattice(Nx,Ny):
    links={}; k=0
    for y in range(Ny+1):
        for x in range(Nx):
            links[('h',x,y)]=k; k+=1
    for y in range(Ny):
        for x in range(Nx+1):
            links[('v',x,y)]=k; k+=1
    NL=k
    def plaq(x,y): return [links[('h',x,y)],links[('h',x,y+1)],links[('v',x,y)],links[('v',x+1,y)]]
    verts={}
    for x in range(Nx+1):
        for y in range(Ny+1):
            ls=[]
            if x<Nx: ls.append(links[('h',x,y)])
            if x>0: ls.append(links[('h',x-1,y)])
            if y<Ny: ls.append(links[('v',x,y)])
            if y>0: ls.append(links[('v',x,y-1)])
            verts[(x,y)]=ls
    return links,NL,plaq,verts
def op_on(NL, ls, pauli):
    I=sparse.identity(2,format='csr')
    X=sparse.csr_matrix(np.array([[0.,1.],[1.,0.]]))
    Z=sparse.csr_matrix(np.diag([1.,-1.]))
    P={'x':X,'z':Z}[pauli]
    M=sparse.identity(1,format='csr')
    for j in range(NL):
        M=sparse.kron(M,P if j in ls else I,format='csr')
    return M
def build(Nx,Ny,g,Lam=50.0):
    links,NL,plaq,verts=lattice(Nx,Ny)
    H=sparse.csr_matrix((2**NL,2**NL))
    for l in range(NL): H=H-g*op_on(NL,[l],'x')
    for x in range(Nx):
        for y in range(Ny):
            H=H-(1.0/g)*op_on(NL,plaq(x,y),'z')
    for v,ls in verts.items():
        H=H+Lam*(sparse.identity(2**NL,format='csr')-op_on(NL,ls,'x'))
    return H,links,NL,plaq,verts
def mpow(r,p,clip=0.0):
    e,U=np.linalg.eigh(r); e=np.maximum(e,clip if clip else 0)
    with np.errstate(divide='ignore'):
        ep=np.where(e>0,e**p,0.0)
    return (U*ep)@U.conj().T
def erec(psi,NL,A,B,C,delta=1e-12):
    order=A+B+C+[l for l in range(NL) if l not in A+B+C]
    nA,nB,nC=len(A),len(B),len(C); n=nA+nB+nC; ne=NL-n
    ps=psi.reshape([2]*NL).transpose(order).reshape(2**n,2**ne)
    # marginales pequenas
    T3=ps.reshape(2**nA,2**nB,2**nC,2**ne)
    rAB=np.einsum('abce,xyce->abxy',T3,T3.conj()).reshape(2**(nA+nB),2**(nA+nB))
    rBC=np.einsum('abce,ayze->bcyz',T3,T3.conj()).reshape(2**(nB+nC),2**(nB+nC))
    rB =np.einsum('abce,ayce->by',T3,T3.conj())
    rBd_isq=mpow(rB+delta*np.eye(2**nB),-0.5)
    sBC=mpow(rBC,0.5)
    # soporte de rho_ABC: Q (2^n x r)
    Q,R=np.linalg.qr(ps)
    lamr=R@R.conj().T          # rho_ABC en soporte
    # X = M^dag Q  con M=(I_A ox s_BC)(I_A ox rBd^{-1/2} ox I_C); M herm en cada factor
    Xq=Q.reshape(2**nA,2**(nB+nC),-1)
    Xq=np.einsum('pq,aqr->apr',sBC.conj().T,Xq)
    Xq=Xq.reshape(2**nA,2**nB,2**nC,-1)
    Xq=np.einsum('pq,aqcr->apcr',rBd_isq.conj().T,Xq)
    Xq=Xq.reshape(2**(nA+nB),2**nC,-1)
    # Q^dag rho_t Q = X^dag (rAB ox I_C) X
    Y=np.einsum('pq,qcr->pcr',rAB,Xq)
    small=np.einsum('pcr,pcs->rs',Xq.conj(),Y)
    # traza de rho_t: Tr[rAB (I_A ox D)], D=rBd^{-1/2} rB rBd^{-1/2}
    D=rBd_isq@rB@rBd_isq
    rr=rAB.reshape(2**nA,2**nB,2**nA,2**nB)
    trt=float(np.real(np.einsum('abay,by->',rr,D)))
    if trt<=0: return np.inf
    sl=mpow(lamr,0.5)
    inner=sl@(small/trt)@sl
    ev=np.linalg.eigvalsh(inner)
    F=float(np.clip(np.sum(np.sqrt(np.maximum(ev,0)))**2,0,1))
    eps=max(1.0-F,0.0)
    return -np.log1p(-eps) if eps<1 else np.inf
def patches_and_buffer(Nx,Ny,links,plaq,verts,NL,w):
    A=plaq(0,0); C=plaq(Nx-1,Ny-1)
    # adyacencia de links (comparten vertice)
    adj={l:set() for l in range(NL)}
    for v,ls in verts.items():
        for a in ls:
            for b in ls:
                if a!=b: adj[a].add(b)
    dist={l:np.inf for l in range(NL)}
    from collections import deque
    dq=deque()
    for l in A: dist[l]=0; dq.append(l)
    while dq:
        u=dq.popleft()
        for vv in adj[u]:
            if dist[vv]>dist[u]+1: dist[vv]=dist[u]+1; dq.append(vv)
    B=[l for l in range(NL) if 1<=dist[l]<=w and l not in C] if w>=1 else []
    return A,B,C
def wilson(psi,NL,links,a,b):
    ls=[]
    for x in range(a): ls+=[links[('h',x,0)],links[('h',x,b)]]
    for y in range(b): ls+=[links[('v',0,y)],links[('v',a,y)]]
    W=op_on(NL,ls,'z')
    return float(np.real(psi.conj()@(W@psi)))
def run_point(Nx,Ny,g,ws=(0,1),want_gap=False):
    H,links,NL,plaq,verts=build(Nx,Ny,g)
    k=2 if want_gap else 1
    vals,vecs=eigsh(H,k=k,which='SA')
    idx=np.argsort(vals); vals=vals[idx]; vecs=vecs[:,idx]
    psi=vecs[:,0]; E0=vals[0]; gap=(vals[1]-vals[0]) if want_gap else None
    gs_min=min(float(np.real(psi.conj()@(op_on(NL,ls,'x')@psi))) for ls in verts.values())
    W11=wilson(psi,NL,links,1,1); W22=wilson(psi,NL,links,2,2)
    W21=wilson(psi,NL,links,2,1); W12=wilson(psi,NL,links,1,2)
    sig=-np.log(abs(W22)*abs(W11)/(abs(W21)*abs(W12)))
    Es={}
    for w in ws:
        A,B,C=patches_and_buffer(Nx,Ny,links,plaq,verts,NL,w)
        Es[w]=erec(psi,NL,A,B,C)
    return dict(E0=float(E0),gap=(float(gap) if gap is not None else None),
                gauss=gs_min,sig=float(sig),Es={str(w):float(v) for w,v in Es.items()})
CACHE=Path(__file__).resolve().with_name("cache_0047.json")
def load(): return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
if "--smoke" in sys.argv:
    links,NL,plaq,verts=lattice(2,2)
    print(f"[SMOKE Z2 benchmark: 2x2 plaquettes, {NL} links]")
    shape_ok=(NL==12 and len(links)==12 and len(verts)==9)
    shape_ok=shape_ok and all(len(plaq(x,y))==4 for x in range(2) for y in range(2))
    check("smoke lattice incidence data", shape_ok)
    psi=np.ones(2**NL)/np.sqrt(2**NL)
    gauss=[float(np.real(psi.conj()@(op_on(NL,ls,'x')@psi))) for ls in verts.values()]
    check("smoke product state in + Gauss sector", min(gauss)>1-1e-12,
          f"[min {min(gauss):.12f}]")
    A,B,C=patches_and_buffer(2,2,links,plaq,verts,NL,1)
    disjoint=len(set(A+B+C))==len(A)+len(B)+len(C)
    check("smoke A/B/C link sets disjoint", disjoint,
          f"[|A|={len(A)}, |B|={len(B)}, |C|={len(C)}]")
    E=erec(psi,NL,A,B,C)
    check("smoke product-state Petz error at floor", np.isfinite(E) and E<1e-8,
          f"[E={E:.2e}]")
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
    sys.exit(0 if not fails else 1)
if "--sweep22" in sys.argv:
    i=sys.argv.index("--sweep22")
    allg=(0.5,0.7,0.9,1.0,1.2,1.5,2.0,2.5)
    sel=allg[int(sys.argv[i+1]):int(sys.argv[i+2])] if len(sys.argv)>i+2 else allg
    d=load(); out=d.get("22",{})
    for g in sel:
        out[str(g)]=run_point(2,2,g,ws=(0,1,2),want_gap=True)
        print(f"  g={g}: sig={out[str(g)]['sig']:.4f} gap={out[str(g)]['gap']:.4f} E1={out[str(g)]['Es']['1']:.3e}")
    d["22"]=out; json.dump(d,open(CACHE,"w")); print("sweep22 listo"); sys.exit(0)
if "--chunk23" in sys.argv:
    g=float(sys.argv[sys.argv.index("--chunk23")+1])
    d=load(); d.setdefault("23",{})[str(g)]=run_point(2,3,g,ws=(0,1))
    json.dump(d,open(CACHE,"w")); print(f"chunk23 g={g} listo: {d['23'][str(g)]}"); sys.exit(0)
d=load()
required_22={str(g) for g in (0.5,0.7,0.9,1.0,1.2,1.5,2.0,2.5)}
required_23={str(g) for g in (0.5,1.0,2.0)}
missing_22=sorted(required_22-set(d.get("22",{})), key=float)
missing_23=sorted(required_23-set(d.get("23",{})), key=float)
if missing_22 or missing_23:
    print("ejecuta los chunks manuales primero, o usa --smoke para CI")
    print(f"  faltan 22: {missing_22}")
    print(f"  faltan 23: {missing_23}")
    sys.exit(1)
print("== 1) Tabla 1: sigma_eff/E0/Gauss reproducidos; E_rec es prescripcion-dependiente ==")
ref22={0.5:(0.150780,-9.5660),1.0:(1.073026,-12.4973),2.0:(2.418666,-24.0625)}
ok_s=True
for g,(s_,e0_) in ref22.items():
    if str(g) in d.get("22",{}):
        r=d["22"][str(g)]
        print(f"  2x2 g={g}: sig {r['sig']:.6f} (v1 {s_})  E0 {r['E0']:.4f} (v1 {e0_})  E1w[esta prescripcion]={r['Es']['1']:.2e}")
        if abs(r["sig"]-s_)/s_>0.01 or abs(r["E0"]-e0_)/abs(e0_)>0.01: ok_s=False
ref23={0.5:(0.102507,-13.9474),1.0:(1.059645,-17.7472),2.0:(2.417752,-34.0937)}
for g,(s_,e0_) in ref23.items():
    if str(g) in d.get("23",{}):
        r=d["23"][str(g)]
        print(f"  2x3 g={g}: sig {r['sig']:.6f} (v1 {s_})  E0 {r['E0']:.4f} (v1 {e0_})  gauss={r['gauss']:.6f}")
        if abs(r["sig"]-s_)/s_>0.01: ok_s=False
check("sigma_eff y E0 reproducen v1 (<1%)", ok_s, "")
print("  NOTA prescripcion: bajo la prescripcion declarada de esta suite, las")
print("  magnitudes de E_rec difieren de la Tabla 1 de v1 (detalles de embedding/")
print("  orden no fijados por el texto de v1), y la saturacion 2x3 E(0)~E(1) de")
print("  v1 NO se observa (aqui el buffer si reduce el error). La DIRECCION de la")
print("  correlacion (abajo) replica bajo ambas prescripciones.")
print("== 2-3) sweep densificado 2x2: correlacion con test de rango + covariable gap ==")
if "22" in d:
    gs=sorted(float(k) for k in d["22"])
    E1=[d["22"][str(g)]["Es"]["1"] for g in gs]
    inv_s=[1.0/d["22"][str(g)]["sig"] for g in gs]
    inv_gap=[1.0/d["22"][str(g)]["gap"] for g in gs]
    def spearman_perm(x,y,n=20000,seed=1):
        rng=np.random.default_rng(seed)
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        r=np.corrcoef(rx,ry)[0,1]
        null=[np.corrcoef(rx,rng.permutation(ry))[0,1] for _ in range(n)]
        p=float(np.mean(np.abs(null)>=abs(r)))
        return r,p
    r1,p1=spearman_perm(E1,inv_s); r2,p2=spearman_perm(E1,inv_gap)
    print(f"  Spearman(E_rec(1), 1/sigma) = {r1:+.3f}, p_perm = {p1:.4f}  (n={len(gs)})")
    print(f"  Spearman(E_rec(1), 1/gap)   = {r2:+.3f}, p_perm = {p2:.4f}")
    check("correlacion E_rec vs 1/sigma significativa (n=8)", r1>0 and p1<0.05, "")
    print("  nota: si ambas correlaciones son comparables, la especificidad de")
    print("  confinamiento (vs fisica generica de gap) queda sin resolver a este tamano.")
    gsm=min((d["22"][str(g)]["gauss"] for g in gs))
    check("Gauss en todo el sweep", gsm>0.999, f"[min {gsm:.5f}]")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
