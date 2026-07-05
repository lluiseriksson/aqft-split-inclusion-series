#!/usr/bin/env python3
"""verify_2601_0051.py -- Suite for 2601.0051 v2 (Z2 ED + TN ladders).

PART A (dependency-free): why the MPS-contiguous proxy can invert buffer
  monotonicity. On an exact Z2 ladder ground state (P=4 plaquettes, 13
  links, Gauss penalty), compare E_rec under (i) the geometric admissible
  buffer (B fills the gap between fixed rungs, cf. 2601.0044 v2) vs
  (ii) an MPS-ordering contiguous proxy (A = first sites, B = next |B|,
  C = rest). The proxy can show E growing with |B| -- the same inversion
  as v1's Table 2 -- while the geometric profile decays: collar semantics
  matter (2601.0043/0050 v2 lesson).
PART B (requires tenpy; skipped otherwise): minimal DMRG replication on a
  2-plaquette-tall ladder Lx=4 at chi=32 for two couplings: Gauss check,
  sigma_eff = chi(2,2), E_rec(|B|=1,2) on a central contiguous proxy
  tripartition -- verifying the pipeline and trend direction at reduced
  cost. Usage: --partA | --partB g | report.
"""
import json
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
def mpow(r,p,clip=0.0):
    e,U=np.linalg.eigh(r); e=np.maximum(e,clip if clip else 0)
    with np.errstate(divide='ignore'):
        ep=np.where(e>0,e**p,0.0)
    return (U*ep)@U.conj().T
def petz_err_pure(psi, A_l, B_l, delta=1e-12):
    """ABC = todos los links (estado puro); C = complemento. Todo en rango bajo:
    rank(rho_BC)<=dA, rank(rho_B)<=dA*dC_eff..., rank(rho_AB)<=dC."""
    nA,nB=len(A_l),len(B_l)
    C_l=[l for l in range(NL) if l not in A_l+B_l]; nC=len(C_l)
    order=A_l+B_l+C_l
    ps=psi.reshape([2]*NL).transpose(order).reshape(-1)
    dA,dB,dC=2**nA,2**nB,2**nC
    T3=ps.reshape(dA,dB,dC)
    # s_BC aplicado a psi: rho_BC = Wbc Wbc^dag, Wbc[q,a]=psi[a,q], rango<=dA
    Wbc=T3.reshape(dA,dB*dC).T                      # (dBC x dA)
    G=Wbc.conj().T@Wbc                              # dA x dA
    eG,UG=np.linalg.eigh(G); eG=np.maximum(eG,0)
    with np.errstate(divide='ignore'):
        iG=np.where(eG>1e-300,1/np.sqrt(eG),0.0)
    Gm12=(UG*iG)@UG.conj().T
    # v0 = s_BC psi  (por columnas de A): s_BC = Wbc Gm12 Wbc^dag
    V=T3.reshape(dA,dB*dC)                          # filas=A
    v0=(Wbc@(Gm12@(Wbc.conj().T@V.T))).T            # (dA x dBC)
    # rho_B thin: desde Wbc (dB,dC,dA): rB = sum_{c,a} w w^dag, rango<=dA*dC pero
    # thin factor: Fb = Wbc.reshape(dB,dC*dA) -> rB = Fb Fb^dag
    Fb=Wbc.reshape(dB,dC,dA).reshape(dB,dC*dA)
    Gb=Fb.conj().T@Fb                               # (dC*dA)^2 pequeno si dC chico; si no, usar dB pequeno
    if dB<=Gb.shape[0]:
        rB=Fb@Fb.conj().T
        e,U=np.linalg.eigh(rB); e=np.maximum(e,0)
    else:
        eg,Ug=np.linalg.eigh(Gb); eg=np.maximum(eg,0)
        keep=eg>1e-300
        U=Fb@(Ug[:,keep]/np.sqrt(eg[keep]))         # (dB x r) ortonormal
        e=eg[keep]
        # completar espectro: resto autovalor 0 (no necesitamos base completa)
    lam=e
    f1=1/np.sqrt(lam+delta)-1/np.sqrt(delta) if True else None
    # rBd^{-1/2} x = delta^{-1/2} x + U diag(f1) U^dag x
    def rBd_isq_apply(X):                           # X: (dB x m)
        Y=X/np.sqrt(delta)
        W_=U if dB>Gb.shape[0] else U
        proj=W_.conj().T@X
        coef=(1/np.sqrt(lam+delta)-1/np.sqrt(delta))
        return Y+W_@(coef[:,None]*proj)
    # v1 = (I_A ox rBd^{-1/2} ox I_C) v0
    v1=v0.reshape(dA,dB,dC)
    v1=np.transpose(v1,(1,0,2)).reshape(dB,dA*dC)
    v1=rBd_isq_apply(v1)
    v1=v1.reshape(dB,dA,dC).transpose(1,0,2).reshape(dA*dB,dC)
    # numerador: ||sAB v1||^2 con rho_AB = Tab Tab^dag, Tab=(dA dB x dC)
    Tab=T3.reshape(dA*dB,dC)
    Q,R=np.linalg.qr(Tab)                           # Q:(dAdB x k)
    rsm=R@R.conj().T
    ssm=mpow(rsm,0.5)
    num=float(np.linalg.norm(ssm@(Q.conj().T@v1))**2)
    # traza: Tr[rho_AB (I_A ox D)], D = U diag(lam/(lam+delta)) U^dag
    W_=U
    dcoef=lam/(lam+delta)
    T3b=T3  # (dA,dB,dC)
    M1=np.einsum('abc,bk->akc', T3b, W_.conj())
    tr=float(np.real(np.einsum('akc,k,akc->', M1, dcoef, M1.conj())))
    if tr<=0: return np.inf
    F=float(np.clip(num/tr,0,1))
    eps=max(1.0-F,0.0)
    return -np.log1p(-eps) if eps<1 else np.inf
# escalera Z2 P plaquetas en fila (13 links, como 2601.0044 corregido)
P=4; NL=3*P+1   # 13 links; off-by-one (qubit espectador) corregido, cf. issue #3 del repo / 2601.0044 v2
def IDX():
    idx={}; k=0
    for c in range(P):
        idx[('r',c)]=k; k+=1; idx[('t',c)]=k; k+=1; idx[('b',c)]=k; k+=1
    idx[('r',P)]=k
    return idx
ID=IDX()
CACHE_A=Path(__file__).resolve().with_name("cache_A.json")
CACHE_B=Path(__file__).resolve().with_name("cache_B.json")
def ladder_gs(g,lam=10.0):
    I=sparse.identity(2,format='csr')
    X=sparse.csr_matrix(np.array([[0.,1.],[1.,0.]])); Z=sparse.csr_matrix(np.diag([1.,-1.]))
    def on(ls,p):
        M=sparse.identity(1,format='csr')
        for j in range(NL):
            M=sparse.kron(M,(X if p=='x' else Z) if j in ls else I,format='csr')
        return M
    H=sparse.csr_matrix((2**NL,2**NL))
    for l in range(NL): H=H-g*on([l],'x')
    for c in range(P):
        H=H-(1.0/g)*on([ID[('t',c)],ID[('b',c)],ID[('r',c)],ID[('r',c+1)]],'z')
    VL={}
    for r_ in (0,1):
        for c in range(P+1):
            ls=[ID[('r',c)]]; key='t' if r_==0 else 'b'
            if c>0: ls.append(ID[(key,c-1)])
            if c<P: ls.append(ID[(key,c)])
            VL[(r_,c)]=ls
    for v,ls in VL.items(): H=H-lam*on(ls,'x')
    val,vec=eigsh(H,k=1,which='SA',v0=np.ones(2**NL))
    return vec[:,0]
if "--smoke" in sys.argv:
    print(f"[SMOKE Z2 ladder/TN proxy: P={P}, NL={NL}]")
    named=set(ID.values())
    shape_ok=(NL==13 and len(ID)==13 and named==set(range(NL)))
    check("smoke ladder incidence names exactly 13 links", shape_ok)
    plaquettes=[[ID[('t',c)],ID[('b',c)],ID[('r',c)],ID[('r',c+1)]] for c in range(P)]
    plaq_ok=all(len(p)==4 and all(0<=l<NL for l in p) for p in plaquettes)
    check("smoke plaquette incidence in range", plaq_ok)
    prod=np.ones(2**NL)/np.sqrt(2**NL)
    A=[ID[('r',0)]]
    B=[ID[(k,0)] for k in ('t','b')]+[ID[('r',1)]]
    E=petz_err_pure(prod,A,B)
    check("smoke product-state Petz error at floor", np.isfinite(E) and E<1e-8,
          f"[E={E:.2e}]")
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
    sys.exit(0 if not fails else 1)
if "--partA" in sys.argv:
    gsel=float(sys.argv[sys.argv.index("--partA")+1])
    out=json.load(open(CACHE_A)) if CACHE_A.exists() else {}
    for g in (gsel,):
        psi=ladder_gs(g)
        # (i) geometrico admisible: A=rung0, B=columnas 0..w-1, C=resto
        geo={}
        for w in (1,2,3):
            A=[ID[('r',0)]]
            B=[ID[(k,c)] for c in range(w) for k in ('t','b')]+[ID[('r',c)] for c in range(1,w+1)]
            geo[w]=petz_err_pure(psi,A,B)
        # (ii) proxy contiguo en orden de indices: A=[0..2], B=next |B|, C=resto
        prox={}
        for nb in (1,2,3):
            A=list(range(3)); B=list(range(3,3+nb))
            prox[nb]=petz_err_pure(psi,A,B)
        out[str(g)]=dict(geo={str(k):v for k,v in geo.items()},
                         prox={str(k):v for k,v in prox.items()})
        print(f"  g={g}: geo(w=1,2,3)={['%.2e'%geo[w] for w in (1,2,3)]}  "
              f"proxy(|B|=1,2,3)={['%.2e'%prox[n] for n in (1,2,3)]}")
    json.dump(out,open(CACHE_A,"w")); print("partA listo"); sys.exit(0)
if "--partB" in sys.argv:
    g=float(sys.argv[sys.argv.index("--partB")+1])
    try:
        from tenpy.networks.site import SpinHalfSite
        from tenpy.models.model import CouplingMPOModel
        from tenpy.networks.mps import MPS
        from tenpy.algorithms import dmrg
        import tenpy
        tenpy.tools.misc.setup_logging(to_stdout="WARNING")
    except ImportError:
        print("tenpy no disponible: parte B omitida"); sys.exit(0)
    Lx, Ny = 4, 2   # Lx x Ny plaquetas
    def lat_idx():
        idx={}; k=0
        for y in range(Ny+1):
            for x in range(Lx):
                idx[('h',x,y)]=k; k+=1
        for y in range(Ny):
            for x in range(Lx+1):
                idx[('v',x,y)]=k; k+=1
        return idx,k
    IDX2,NS=lat_idx()
    class Z2Ladder(CouplingMPOModel):
        def init_sites(self, mp):
            return SpinHalfSite(conserve=None)
        def init_terms(self, mp):
            gg=mp.get('g',1.0); lam=mp.get('lam',50.0)
            for i in range(NS):
                self.add_onsite_term(-gg, i, 'Sigmax')
            for x in range(Lx):
                for y in range(Ny):
                    ls=[IDX2[('h',x,y)],IDX2[('h',x,y+1)],IDX2[('v',x,y)],IDX2[('v',x+1,y)]]
                    ls=sorted(ls)
                    self.add_multi_coupling_term(-1.0/gg, ls, ['Sigmaz']*4, ['Id']*3)
            for x in range(Lx+1):
                for y in range(Ny+1):
                    ls=[]
                    if x<Lx: ls.append(IDX2[('h',x,y)])
                    if x>0: ls.append(IDX2[('h',x-1,y)])
                    if y<Ny: ls.append(IDX2[('v',x,y)])
                    if y>0: ls.append(IDX2[('v',x,y-1)])
                    ls=sorted(ls)
                    self.add_multi_coupling_term(-lam, ls, ['Sigmax']*len(ls), ['Id']*(len(ls)-1))
        def init_lattice(self, mp):
            from tenpy.models.lattice import Chain
            return Chain(NS, self.init_sites(mp), bc='open', bc_MPS='finite')
    M=Z2Ladder({'g':g,'lam':50.0})
    psi=MPS.from_product_state(M.lat.mps_sites(), ['up']*NS, bc='finite')
    info=dmrg.run(psi, M, {'trunc_params':{'chi_max':32,'svd_min':1e-10},
                           'max_sweeps':14,'mixer':True})
    E0=info['E']
    def expZ(ls):
        term=[('Sigmaz',i) for i in sorted(ls)]
        return float(np.real(psi.expectation_value_term(term)))
    def expX(ls):
        term=[('Sigmax',i) for i in sorted(ls)]
        return float(np.real(psi.expectation_value_term(term)))
    gmin=min(expX(sorted([IDX2[k] for k in kk])) for kk in [[]] ) if False else None
    gs=[]
    for x in range(Lx+1):
        for y in range(Ny+1):
            ls=[]
            if x<Lx: ls.append(IDX2[('h',x,y)])
            if x>0: ls.append(IDX2[('h',x-1,y)])
            if y<Ny: ls.append(IDX2[('v',x,y)])
            if y>0: ls.append(IDX2[('v',x,y-1)])
            gs.append(expX(ls))
    def wilson(a,b,x0=0,y0=0):
        ls=[]
        for x in range(x0,x0+a): ls+=[IDX2[('h',x,y0)],IDX2[('h',x,y0+b)]]
        for y in range(y0,y0+b): ls+=[IDX2[('v',x0,y)],IDX2[('v',x0+a,y)]]
        return expZ(ls)
    W22=abs(wilson(2,2)); W11=abs(wilson(1,1)); W21=abs(wilson(2,1)); W12=abs(wilson(1,2))
    sig=-np.log(W22*W11/(W21*W12))
    # E_rec proxy contiguo central: A=[8..10], B=next nb, C=next 3
    res={}
    for nb in (1,2):
        i0=8; A=list(range(i0,i0+3)); B=list(range(i0+3,i0+3+nb)); C=list(range(i0+3+nb,i0+6+nb))
        seg=list(range(i0, i0+6+nb))
        rho=psi.get_rho_segment(seg).to_ndarray().reshape(2**len(seg),2**len(seg))
        n=len(seg); nA,nB,nC=3,nb,3
        def ptr(r,keep):
            r=r.reshape([2]*(2*n))
            for ax in sorted((i for i in range(n) if i not in keep),reverse=True):
                r=np.trace(r,axis1=ax,axis2=ax+r.ndim//2)
            return r.reshape(2**len(keep),-1)
        rABC=rho  # todo el segmento = ABC
        rAB=ptr(rho,list(range(nA+nB))); rB=ptr(rho,list(range(nA,nA+nB)))
        rBC=ptr(rho,list(range(nA,n)))
        Mx=np.kron(np.eye(2**nA),mpow(rBC,0.5))@np.kron(np.eye(2**nA),np.kron(mpow(rB+1e-12*np.eye(2**nB),-0.5,clip=1e-300),np.eye(2**nC)))
        til=Mx@np.kron(rAB,np.eye(2**nC))@Mx.conj().T
        til=(til+til.conj().T)/2; til/=np.trace(til).real
        sa=mpow(rABC,0.5)
        F=float(np.clip(np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(sa@til@sa),0)))**2,0,1))
        res[nb]=-np.log1p(-(1-F)) if F<1 else 0.0
    d=json.load(open(CACHE_B)) if CACHE_B.exists() else {}
    d[str(g)]=dict(E0=float(E0),gauss_min=float(min(gs)),sig=float(sig),
                   E={str(k):float(v) for k,v in res.items()})
    json.dump(d,open(CACHE_B,"w"))
    print(f"partB g={g}: E0={E0:.6f} gauss_min={min(gs):.6f} sig={sig:.4f} E={ {k:'%.3e'%v for k,v in res.items()} }")
    sys.exit(0)
# reporte
print("== PART A: semantica del collar (exacto, sin dependencias) ==")
if not CACHE_A.exists():
    print("ejecuta los chunks manuales primero, o usa --smoke para CI")
    print("  falta 0051 cache_A: ['0.6', '1.0']")
    sys.exit(1)
a=json.load(open(CACHE_A))
missing_a=sorted({str(g) for g in (0.6,1.0)}-set(a), key=float)
if missing_a:
    print("ejecuta los chunks manuales primero, o usa --smoke para CI")
    print(f"  faltan 0051 partA: {missing_a}")
    sys.exit(1)
for g,r in a.items():
    geo=[r["geo"][str(w)] for w in (1,2,3)]
    prox=[r["prox"][str(n)] for n in (1,2,3)]
    gmon=all(x>=y-1e-15 for x,y in zip(geo,geo[1:]))
    pinv=any(x<y for x,y in zip(prox,prox[1:]))
    print(f"  g={g}: geometrico decae={gmon}  proxy-invierte={pinv}")
check("buffer geometrico decae; el proxy contiguo puede invertir",
      all(all(x>=y-1e-15 for x,y in zip([a[g]['geo'][str(w)] for w in (1,2,3)],[a[g]['geo'][str(w)] for w in (2,3)])) for g in a), "")
print("== PART B: replicacion DMRG minima (tenpy) ==")
if CACHE_B.exists():
    b=json.load(open(CACHE_B))
    for g in sorted(b,key=float):
        r=b[g]
        print(f"  g={g}: gauss_min={r['gauss_min']:.5f} sig={r['sig']:.4f} E(|B|=1)={r['E']['1']:.3e} E(|B|=2)={r['E']['2']:.3e}")
        check(f"Gauss (g={g})", r['gauss_min']>0.99, "")
    if len(b)>=2:
        gs_=sorted(b,key=float)
        e1=[b[g]['E']['1'] for g in gs_]; invs=[1/b[g]['sig'] for g in gs_]
        check("direccion del trend: E(|B|=1) crece con 1/sigma",
              (e1[0]>e1[-1]) == (invs[0]>invs[-1]), f"[E: {e1[0]:.2e}->{e1[-1]:.2e}; 1/sig: {invs[0]:.2f}->{invs[-1]:.2f}]")
else:
    print("  (sin cache_B: ejecuta --partB g con tenpy instalado)")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
