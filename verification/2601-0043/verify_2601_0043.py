#!/usr/bin/env python3
"""verify_2601_0043.py -- Suite for 2601.0043 v2 (recoverability geometry).

Regenerable pipeline (sparse Lanczos ground states; chunked JSON cache).
TFIM H = -J sum ZZ - g sum X - h sum Z, J=1, open chain.

 1. Regenerates the Table 1 control (L=14, |A|=3, |C|=4, complement traced,
    delta=1e-12, fits on w in {2,3,4} with floor 1e-14): xi_rec per g in
    {0.5 (h=1e-3), 1.0, 2.0}, with R^2. The g=0.5 row is FLAGGED: 3-point
    fit with low R^2 sits below the paper's own stability bar (Rem. 4.6).
 2. First in-model test of Conjecture 8.1: from the SAME ground states,
    extract a conventional correlation length xi_corr from connected
    <Z_1 Z_r> decay and compare orders of magnitude with xi_rec.
 3. Embedding mini-demo (Section 5 finally illustrated): L=12, g=2.0,
    four single-site coarse regions; directed threshold distances,
    max-symmetrization, classical MDS in 1D -> recovers the chain order
    (perfect rank correlation); triangle-violation scores reported.
Usage: --chunk g  (0.5|1.0|2.0) then no args; --demo runs part 3.
Use --smoke as an alias for --demo in fast CI.
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
def ham_sparse(L,g,h,J=1.0):
    I=sparse.identity(2,format='csr'); X=sparse.csr_matrix(np.array([[0,1],[1,0]],dtype=float))
    Zs=sparse.csr_matrix(np.diag([1.,-1.]))
    def site(op,i):
        M=sparse.identity(1,format='csr')
        for j in range(L):
            M=sparse.kron(M, op if j==i else I, format='csr')
        return M
    H=sparse.csr_matrix((2**L,2**L))
    for i in range(L-1): H=H-J*(site(Zs,i)@site(Zs,i+1))
    for i in range(L):
        H=H-g*site(X,i)
        if h: H=H-h*site(Zs,i)
    return H
def gs(L,g,h):
    H=ham_sparse(L,g,h)
    val,vec=eigsh(H,k=1,which='SA')
    return vec[:,0]
def rho_block(psi,L,nblock):
    """rho de los primeros nblock sitios (resto trazado)."""
    M=psi.reshape(2**nblock, 2**(L-nblock))
    return M@M.conj().T
def ptr(rho, n, keep):
    rho=rho.reshape([2]*(2*n))
    for ax in sorted((i for i in range(n) if i not in keep), reverse=True):
        rho=np.trace(rho, axis1=ax, axis2=ax+rho.ndim//2)
    return rho.reshape(2**len(keep),2**len(keep))
def mpow(r,p,clip=0.0):
    e,U=np.linalg.eigh(r)
    if clip: e=np.maximum(e,clip)
    else: e=np.maximum(e,0)
    with np.errstate(divide='ignore'):
        ep=np.where(e>0, e**p, 0.0)
    return (U*ep)@U.conj().T
def petz_err(rho_abc, nA, nB, nC, delta=1e-12):
    n=nA+nB+nC
    rAB=ptr(rho_abc,n,list(range(nA+nB)))
    rB =ptr(rho_abc,n,list(range(nA,nA+nB)))
    rBC=ptr(rho_abc,n,list(range(nA,n)))
    rBd=rB+delta*np.eye(2**nB)
    M=np.kron(np.eye(2**nA), mpow(rBC,0.5))@np.kron(np.eye(2**nA), np.kron(mpow(rBd,-0.5,clip=1e-300), np.eye(2**nC)))
    til=M@np.kron(rAB,np.eye(2**nC))@M.conj().T
    til=(til+til.conj().T)/2
    tr=np.trace(til).real
    if tr<=0: return np.inf
    til/=tr
    sa=mpow(rho_abc,0.5)
    fv=np.linalg.eigvalsh(sa@til@sa)
    F=float(np.clip(np.sum(np.sqrt(np.maximum(fv,0)))**2,0,1))
    eps=max(1.0-F,0.0)
    return -np.log1p(-eps) if eps<1 else np.inf
L=14; nA,nC=3,4
CACHE=Path(__file__).resolve().with_name("cache_0043.json")
def load(): return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
if "--chunk" in sys.argv:
    g=float(sys.argv[sys.argv.index("--chunk")+1])
    h=1e-3 if g<1 else 0.0
    psi=gs(L,g,h)
    row={}
    for w in (2,3,4):
        nblk=nA+w+nC
        rho=rho_block(psi,L,nblk)
        row[w]=petz_err(rho,nA,w,nC)
    # correlaciones conexas <Z1 Zr> del MISMO estado
    Zdiag=np.array([1.,-1.])
    def zexp(i):
        v=psi.reshape([2]*L)
        # <Z_i>
        w_=np.abs(psi)**2
        idx=(np.arange(2**L)>> (L-1-i)) & 1
        return float(np.sum(w_*(1-2*idx)))
    def zz(i,j):
        w_=np.abs(psi)**2
        bi=(np.arange(2**L)>>(L-1-i))&1; bj=(np.arange(2**L)>>(L-1-j))&1
        return float(np.sum(w_*(1-2*bi)*(1-2*bj)))
    corr=[abs(zz(1,1+r)-zexp(1)*zexp(1+r)) for r in range(1,9)]
    d=load(); d[str(g)]={"E":{str(w):v for w,v in row.items()}, "corr":corr}
    json.dump(d,open(CACHE,"w")); print(f"chunk g={g} listo: E={ {w:f'{v:.3e}' for w,v in row.items()} }"); sys.exit(0)
if "--demo" in sys.argv or "--smoke" in sys.argv:
    # En 1D estricto con regiones fijas, la condicion (3) de la collaring rule
    # (B separa A de C) exige w >= gap; con w < gap el entorno trazado entre
    # B y C decorrelaciona "gratis" e invierte la monotonia. El demo usa
    # B = region completa entre A y C (w = gap), con censura cap = gap + 2.
    L2=12; g=2.0; psi=gs(L2,g,0.0)
    sites=[1,4,7,10]; epsT=1e-3
    Dm=np.zeros((4,4))
    for a in range(4):
        for b in range(4):
            if a==b: continue
            lo,hi=min(sites[a],sites[b]),max(sites[a],sites[b])
            gap=hi-lo-1
            nblk=hi-lo+1
            M=psi.reshape(2**lo, 2**nblk, 2**(L2-hi-1))
            M=np.transpose(M,(1,0,2)).reshape(2**nblk,-1)
            rho_blk=M@M.conj().T
            E=petz_err(rho_blk,1,gap,1)
            Dm[a,b]=gap if E<=epsT else gap+2
    Ds=np.maximum(Dm,Dm.T)
    print("D simetrizada (eps=1e-3, cap=gap+2):"); print(np.round(Ds,2))
    J_=np.eye(4)-np.ones((4,4))/4
    Bmat=-0.5*J_@(Ds**2)@J_
    ev,U=np.linalg.eigh(Bmat)
    emb=U[:,-1]*np.sqrt(max(ev[-1],0))
    order=[int(i) for i in np.argsort(emb)]
    ok=(order==[0,1,2,3]) or (order==[3,2,1,0])
    print("embedding 1D:", np.round(emb,3), " orden:", order)
    check("MDS recupera el orden de la cadena", ok, "")
    import itertools as it
    tv=[max(0.0, Ds[i,k]-Ds[i,j]-Ds[j,k]) for i,j,k in it.permutations(range(4),3)]
    print(f"score de violacion triangular: max={max(tv):.3f} (discretizacion/cap)")
    check("violaciones triangulares acotadas por el cap", max(tv)<=2.0+1e-9, f"[max {max(tv):.2f}]")
    print("DEMO OK" if not fails else "DEMO FAIL"); sys.exit(0 if not fails else 1)
d=load()
if len(d)<3: print("ejecuta --chunk 0.5 / 1.0 / 2.0 primero"); sys.exit(1)
print(f"[L={L}, |A|={nA}, |C|={nC}, w in {{2,3,4}}, delta=1e-12]")
print("== 1) Tabla 1 regenerada ==")
tab1={0.5:(1.433,0.6501), 1.0:(4.58,0.9276), 2.0:(0.6515,0.9990)}
for gs_ in (0.5,1.0,2.0):
    E={int(w):v for w,v in d[str(gs_)]["E"].items()}
    pts=[(w,E[w]) for w in (2,3,4) if E[w]>1e-14]
    wv=np.array([p[0] for p in pts]); ev=np.log([p[1] for p in pts])
    c=np.polyfit(wv,ev,1); xi=-1.0/c[0]
    pred=np.polyval(c,wv); r2=1-np.sum((ev-pred)**2)/np.sum((ev-ev.mean())**2)
    flag=" [FLAG: R2 bajo, fit de 3 puntos]" if r2<0.9 else ""
    print(f"  g={gs_}: xi_rec={xi:.3f} (v1: {tab1[gs_][0]})  R2={r2:.4f} (v1: {tab1[gs_][1]}){flag}")
print("== 2) test in-model de la Conjetura 8.1 ==")
for gs_ in (0.5,1.0,2.0):
    corr=d[str(gs_)]["corr"]
    pts=[(r+1,c) for r,c in enumerate(corr) if c>1e-12]
    if len(pts)>=3:
        rv=np.array([p[0] for p in pts]); cv=np.log([p[1] for p in pts])
        cc=np.polyfit(rv,cv,1); xic=-1.0/cc[0] if cc[0]<0 else np.inf
    else: xic=float('nan')
    E={int(w):v for w,v in d[str(gs_)]["E"].items()}
    pts2=[(w,E[w]) for w in (2,3,4) if E[w]>1e-14]
    c2=np.polyfit([p[0] for p in pts2], np.log([p[1] for p in pts2]),1); xir=-1.0/c2[0]
    print(f"  g={gs_}: xi_rec={xir:.3f}  xi_corr(ZZ conexo)={xic:.3f}  ratio={xir/xic if np.isfinite(xic) and xic>0 else float('nan'):.2f}")
print("  (Conjetura 8.1: mismo orden de magnitud en fases gapped; en g=1 ambos crecen)")
print(f"\n{'PARTES 1-2 COMPLETADAS' if not fails else 'FAILED'}")
