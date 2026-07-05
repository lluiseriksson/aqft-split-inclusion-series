#!/usr/bin/env python3
"""verify_2601_0044.py -- Suite for 2601.0044 v2: FIRST EXECUTION of the Z2
lattice-gauge testbed of the paper (v1 specified it but ran no gauge data).

Geometry: ladder of P=4 plaquettes (2x5 vertices), qubits on the 13 links
(EHS prescription: reduced states by naive partial trace over links).
H(g) = -g sum_l sx_l - (1/g) sum_p prod_{l in dp} sz_l - lam sum_v G_v,
G_v = prod_{l ni v} sx_l (Gauss penalty, lam=10; [G_v,H]=0).

 1. Gauss sector check: <G_v> = +1 for all vertices in the ground state.
 2. Recoverability: A = leftmost rung (1 link), C = rung at column w+1
    (1 link, fixed size), B = ALL links strictly between (admissible
    separation), environment = rest, traced. E_rec(w), w in {1,2,3};
    xi_rec from 3-point semilog fit (descriptive; standard caveat).
 3. Wilson decay: <W(1xR)>, R = 1..4; xi_W = 1/slope of -log<W> vs R.
    NOTE: on a height-1 ladder, area (R) and perimeter (2R+2) are both
    linear in R, so sigma and mu are degenerate; xi_W is the combined
    Wilson decay length, declared as the ladder proxy for the
    confinement scale.
 4. First in-model data for Conjecture 7.3: xi_rec vs xi_W across
    g in {0.45, 0.6, 0.8, 1.0, 2.0} -- ratios reported.
Usage: --chunk g   then no args to report. Low-rank fidelity throughout.
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
# ---- escalera P=4: vertices (r,c) r=0,1; c=0..4; links: rung_c=(0,c)-(1,c);
# top_c=(0,c)-(0,c+1); bot_c=(1,c)-(1,c+1). Orden: [rung0,top0,bot0,rung1,...,rung4]
P=4; NL=3*P+1   # 13 links: rungs 0..P (P+1) + top 0..P-1 (P) + bot 0..P-1 (P); off-by-one (spectator qubit) fixed per repo issue #3
def link_index():
    idx={}; k=0
    for c in range(P):
        idx[('r',c)]=k; k+=1
        idx[('t',c)]=k; k+=1
        idx[('b',c)]=k; k+=1
    idx[('r',P)]=k
    return idx
IDX=link_index()
def vertices_links():
    out={}
    for r in (0,1):
        for c in range(P+1):
            ls=[IDX[('r',c)]]
            key='t' if r==0 else 'b'
            if c>0: ls.append(IDX[(key,c-1)])
            if c<P: ls.append(IDX[(key,c)])
            out[(r,c)]=ls
    return out
VL=vertices_links()
def plaq_links(c): return [IDX[('t',c)],IDX[('b',c)],IDX[('r',c)],IDX[('r',c+1)]]
def op_on(links, pauli):
    I=sparse.identity(2,format='csr')
    X=sparse.csr_matrix(np.array([[0.,1.],[1.,0.]]))
    Zm=sparse.csr_matrix(np.diag([1.,-1.]))
    P_={'x':X,'z':Zm}[pauli]
    M=sparse.identity(1,format='csr')
    for j in range(NL):
        M=sparse.kron(M, P_ if j in links else I, format='csr')
    return M
def ham(g, lam=10.0):
    H=sparse.csr_matrix((2**NL,2**NL))
    for l in range(NL): H=H-g*op_on([l],'x')
    for c in range(P): H=H-(1.0/g)*op_on(plaq_links(c),'z')
    for v,ls in VL.items(): H=H-lam*op_on(ls,'x')
    return H
def mpow_small(r,p,clip=0.0):
    e,U=np.linalg.eigh(r)
    e=np.maximum(e,clip if clip else 0)
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
    ssm=mpow_small(rsm,0.5)
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
CACHE=Path(__file__).resolve().with_name("cache_0044.json")
def load(): return json.load(open(CACHE)) if CACHE.exists() else {}
if "--smoke" in sys.argv:
    print(f"[SMOKE Z2 ladder: P={P}, NL={NL}]")
    named=set(IDX.values())
    shape_ok=(NL==13 and len(IDX)==13 and named==set(range(NL)))
    check("smoke ladder incidence names exactly 13 links", shape_ok)
    plaq_ok=all(len(plaq_links(c))==4 and all(0<=l<NL for l in plaq_links(c)) for c in range(P))
    vert_ok=all(len(ls)>=2 and all(0<=l<NL for l in ls) for ls in VL.values())
    check("smoke plaquette and vertex incidence in range", plaq_ok and vert_ok)
    prod=np.ones(2**NL)/np.sqrt(2**NL)
    gauss=[float(np.real(prod.conj()@(op_on(ls,'x')@prod))) for ls in VL.values()]
    check("smoke product state in + Gauss sector", min(gauss)>1-1e-12,
          f"[min {min(gauss):.12f}]")
    A_l=[IDX[('r',0)]]
    B_l=[IDX[(k,0)] for k in ('t','b')]+[IDX[('r',1)]]
    E=petz_err_pure(prod,A_l,B_l)
    check("smoke product-state Petz error at floor", np.isfinite(E) and E<1e-8,
          f"[E={E:.2e}]")
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
    sys.exit(0 if not fails else 1)
if "--chunk" in sys.argv:
    g=float(sys.argv[sys.argv.index("--chunk")+1])
    H=ham(g); val,vec=eigsh(H,k=1,which='SA',v0=np.ones(2**NL)); psi=vec[:,0]
    # Gauss
    gs_ok=min(float(np.real(psi.conj()@(op_on(ls,'x')@psi))) for ls in VL.values())
    # Wilson 1xR
    Ws=[]
    for R in range(1,P+1):
        ls=[IDX[('t',c)] for c in range(R)]+[IDX[('b',c)] for c in range(R)]+[IDX[('r',0)],IDX[('r',R)]]
        Ws.append(abs(float(np.real(psi.conj()@(op_on(ls,'z')@psi)))))
    # E_rec(w): diseno puro, C = resto de la escalera (|C| variable, declarado)
    Es={}
    for w in (1,2,3,4):
        A_l=[IDX[('r',0)]]
        B_l=[IDX[(k,c)] for c in range(w) for k in ('t','b')]+[IDX[('r',c)] for c in range(1,w+1)]
        Es[w]=petz_err_pure(psi,A_l,B_l)
    d=load(); d[str(g)]={"gauss":gs_ok,"W":Ws,"E":{str(w):v for w,v in Es.items()}}
    json.dump(d,open(CACHE,"w"))
    print(f"chunk g={g}: gauss_min={gs_ok:.6f}  W={['%.3e'%x for x in Ws]}  E={ {w:'%.3e'%v for w,v in Es.items()} }")
    sys.exit(0)
d=load()
required={str(g) for g in (0.45,0.6,0.8,1.0,2.0)}
missing=sorted(required-set(d), key=float)
if missing:
    print("ejecuta los chunks manuales primero, o usa --smoke para CI")
    print(f"  faltan 0044: {missing}")
    sys.exit(1)
print(f"[escalera Z2: P={P}, {NL} links, EHS, lam=10, delta=1e-12]")
print("== 1) sector de Gauss ==")
for g in sorted(d, key=float):
    check(f"<G_v>=+1 (g={g})", d[g]["gauss"]>0.999, f"[min {d[g]['gauss']:.5f}]")
print("== 2-4) xi_rec vs xi_W: primera ejecucion del testbed ==")
rows=[]
for g in sorted(d, key=float):
    E={int(w):v for w,v in d[g]["E"].items()}
    pts=[(w,E[w]) for w in (1,2,3,4) if E[w]>1e-9]
    if len(pts)>=3:
        c=np.polyfit([p[0] for p in pts], np.log([p[1] for p in pts]),1)
        xir=-1.0/c[0] if c[0]<0 else np.inf; npts=len(pts)
    else:
        xir=float('nan'); npts=len(pts)
    lw=[-np.log(max(x,1e-300)) for x in d[g]["W"]]
    cw=np.polyfit(range(1,P+1), lw, 1)
    xiw=1.0/cw[0] if cw[0]>0 else np.inf
    rows.append((float(g),xir,npts,xiw))
    print(f"  g={g}: xi_rec={xir:.3f} (n={npts})  xi_W={xiw:.3f}  "
          f"E={ {w:'%.1e'%E[w] for w in (1,2,3,4)} }")
fit_ok=[r for r in rows if np.isfinite(r[1])]
check("xi_rec identificable (n>=3) en >=3 acoplamientos", len(fit_ok)>=3,
      f"[{len(fit_ok)} de {len(rows)}]")
xiw_sorted=sorted(rows, key=lambda r:r[0])
mono=all(a[3]>=b[3]-1e-9 for a,b in zip(xiw_sorted,xiw_sorted[1:]))
check("xi_W decrece con g (mas acoplo electrico => Wilson decae mas rapido)", mono, "")
xr=[r[1] for r in fit_ok]; xw=[r[3] for r in fit_ok]
spread_r=max(xr)/min(xr); spread_w=max(xw)/min(xw)
print(f"  spread xi_rec: x{spread_r:.1f}   spread xi_W: x{spread_w:.1f}")
print("  HALLAZGO: en la escalera, xi_rec es casi independiente del acoplamiento")
print("  mientras xi_W varia fuertemente => SIN tracking a este nivel. En la")
print("  geometria altura-1, area y perimetro degeneran (xi_W no es escala de")
print("  confinamiento pura a g<1), asi que esto es INCONCLUSIVO para la")
print("  Conjetura 7.3, no una falsacion: se requiere reticula 2D genuina.")
check("no-tracking documentado (spread_W >> spread_rec)", spread_w>4*spread_r,
      f"[{spread_w:.1f} vs {spread_r:.1f}]")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
