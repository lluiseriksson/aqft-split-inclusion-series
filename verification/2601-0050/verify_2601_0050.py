#!/usr/bin/env python3
"""verify_2601_0050.py -- Suite for 2601.0050 v2 (CMI vs Wilson in Z2 2+1D).

 1. SATURATION LEMMA (new in v2, explains Table 1's exact coincidence
    I(w_sat) = I(w=0)): for a PURE global state, if B = (A u C)^c then
    S(B)=S(AC), S(AB)=S(C), S(BC)=S(A), hence I(A:C|B) = S(A)+S(C)-S(AC)
    = I(A:C|empty). Verified to machine precision on random pure states
    AND on the Z2 2x2 ground state. Consequence: saturated rows carry no
    buffer information; the informative range of v1's Table 1 is w in
    {0,1} only.
 2. Non-monotonicity remark anchor: I(A:C|B) can GROW when B is enlarged
    (no data-processing in that direction) -- exhibited numerically.
 3. CMI-based benchmark cross-linked to 2601.0047: on the same 2x2 sweep
    (8 couplings, BFS patch geometry), Spearman rank-trends of I(A:C|B(1))
    vs 1/sigma_eff and vs 1/gap, with permutation p.
 4. Convention harmonization: v1 (correctly) uses ROOT fidelity with
    I >= -2 log f; companions use SQUARED F with I >= -log F. Identity
    -2 log f = -log F verified, so both are the same statement.
Usage: --sweep g0 g1 (indices) then no args.
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
def S_ent(psi,NL,X):
    order=X+[l for l in range(NL) if l not in X]
    M=psi.reshape([2]*NL).transpose(order).reshape(2**len(X),-1)
    sv=np.linalg.svd(M,compute_uv=False)
    p=sv**2; p=p[p>1e-15]
    return float(-(p*np.log(p)).sum())
def cmi(psi,NL,A,B,C):
    return S_ent(psi,NL,A+B)+S_ent(psi,NL,B+C)-(S_ent(psi,NL,B) if B else 0.0)-S_ent(psi,NL,A+B+C)
def wilson(psi,NL,links,a,b):
    ls=[]
    for x in range(a): ls+=[links[('h',x,0)],links[('h',x,b)]]
    for y in range(b): ls+=[links[('v',0,y)],links[('v',a,y)]]
    return float(np.real(psi.conj()@(op_on(NL,ls,'z')@psi)))
def bfs_buffer(Nx,Ny,links,plaq,verts,NL,w):
    A=plaq(0,0); C=plaq(Nx-1,Ny-1)
    adj={l:set() for l in range(NL)}
    for v,ls in verts.items():
        for a in ls:
            for b in ls:
                if a!=b: adj[a].add(b)
    from collections import deque
    dist={l:np.inf for l in range(NL)}
    dq=deque()
    for l in A: dist[l]=0; dq.append(l)
    while dq:
        u=dq.popleft()
        for vv in adj[u]:
            if dist[vv]>dist[u]+1: dist[vv]=dist[u]+1; dq.append(vv)
    B=[l for l in range(NL) if 1<=dist[l]<=w and l not in C] if w>=1 else []
    return A,B,C
CACHE=Path(__file__).resolve().with_name("cache_0050.json")
def load(): return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
GS=(0.5,0.7,0.9,1.0,1.2,1.5,2.0,2.5)
if "--smoke" in sys.argv:
    print("[SMOKE CMI/Z2 benchmark: finite identity checks]")
    links,NL,plaq,verts=lattice(2,2)
    shape_ok=(NL==12 and len(links)==12 and len(verts)==9)
    shape_ok=shape_ok and all(len(plaq(x,y))==4 for x in range(2) for y in range(2))
    check("smoke lattice incidence data", shape_ok)
    rng=np.random.default_rng(26010050)
    psi=rng.normal(size=2**8)+1j*rng.normal(size=2**8)
    psi/=np.linalg.norm(psi)
    A=[0,1]; C=[6,7]; B=[2,3,4,5]
    err=abs(cmi(psi,8,A,B,C)-cmi(psi,8,A,[],C))
    check("smoke pure-state saturation identity", err<1e-10, f"[err {err:.1e}]")
    A2,B2,C2=bfs_buffer(2,2,links,plaq,verts,NL,1)
    disjoint=len(set(A2+B2+C2))==len(A2)+len(B2)+len(C2)
    check("smoke BFS A/B/C link sets disjoint", disjoint,
          f"[|A|={len(A2)}, |B|={len(B2)}, |C|={len(C2)}]")
    prod=np.ones(2**NL)/np.sqrt(2**NL)
    Iprod=cmi(prod,NL,A2,B2,C2)
    check("smoke product-state CMI at floor", abs(Iprod)<1e-8, f"[I={Iprod:.2e}]")
    f=rng.uniform(0.1,1.0,128)
    check("smoke FR root/squared convention", np.allclose(-2*np.log(f), -np.log(f**2)))
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
    sys.exit(0 if not fails else 1)
if "--sweep" in sys.argv:
    i=sys.argv.index("--sweep")
    sel=GS[int(sys.argv[i+1]):int(sys.argv[i+2])]
    d=load(); out=d.get("22",{})
    for g in sel:
        H,links,NL,plaq,verts=build(2,2,g)
        vals,vecs=eigsh(H,k=2,which='SA')
        idx=np.argsort(vals); psi=vecs[:,idx[0]]; gap=float(vals[idx[1]]-vals[idx[0]])
        W11=wilson(psi,NL,links,1,1); W22=wilson(psi,NL,links,2,2)
        W21=wilson(psi,NL,links,2,1); W12=wilson(psi,NL,links,1,2)
        sig=-np.log(abs(W22)*abs(W11)/(abs(W21)*abs(W12)))
        A,B1,C=bfs_buffer(2,2,links,plaq,verts,NL,1)
        I0=cmi(psi,NL,A,[],C); I1=cmi(psi,NL,A,B1,C)
        gauss=min(float(np.real(psi.conj()@(op_on(NL,ls,'x')@psi))) for ls in verts.values())
        out[str(g)]=dict(sig=float(sig),gap=gap,I0=float(I0),I1=float(I1),gauss=gauss)
        print(f"  g={g}: sig={sig:.4f} gap={gap:.4f} I0={I0:.4e} I1={I1:.4e}")
    d["22"]=out; json.dump(d,open(CACHE,"w")); print("sweep listo"); sys.exit(0)
d=load()
required_22={str(g) for g in GS}
missing_22=sorted(required_22-set(d.get("22",{})), key=float)
if missing_22:
    print("ejecuta los chunks manuales primero, o usa --smoke para CI")
    print(f"  faltan 22: {missing_22}")
    sys.exit(1)
print("== 1) LEMA DE SATURACION ==")
rng=np.random.default_rng(26010050)
# (a) estados puros aleatorios, 8 qubits: A=[0,1], C=[6,7], B=complemento
mx=0
for t in range(6):
    psi=rng.normal(size=2**8)+1j*rng.normal(size=2**8); psi/=np.linalg.norm(psi)
    A=[0,1]; C=[6,7]; B=[2,3,4,5]
    I_sat=cmi(psi,8,A,B,C); I_0=cmi(psi,8,A,[],C)
    mx=max(mx,abs(I_sat-I_0))
check("I(A:C|B=(AuC)^c) = I(A:C|vacio) en estados puros aleatorios", mx<1e-10, f"[err {mx:.1e}]")
# (b) en el estado fundamental Z2 2x2
if "22" in d and "1.0" in d["22"]:
    H,links,NL,plaq,verts=build(2,2,1.0)
    vals,vecs=eigsh(H,k=1,which='SA'); psi=vecs[:,0]
    A,_,C=bfs_buffer(2,2,links,plaq,verts,NL,1)
    Bfull=[l for l in range(NL) if l not in A+C]
    ms=abs(cmi(psi,NL,A,Bfull,C)-cmi(psi,NL,A,[],C))
    check("identidad de saturacion en el GS Z2 2x2", ms<1e-10, f"[err {ms:.1e}]")
    print("  -> explica la coincidencia exacta I(w=2)=I(w=0)=0.4992999 de la Tabla 1 de v1:")
    print("     la fila saturada no aporta informacion de buffer; rango informativo w en {0,1}.")
print("== 2) no-monotonia de I en B ==")
if "22" in d:
    grow=sum(1 for g in d["22"] if d["22"][g]["I1"]>d["22"][g]["I0"])
    print(f"  I(A:C|B(1)) > I(A:C|vacio) en {grow}/{len(d['22'])} acoplamientos del sweep")
    print("  (agrandar B puede AUMENTAR el CMI: no hay data-processing en esa direccion)")
print("== 3) cross-link con 2601.0047: tendencias de rango ==")
if "22" in d:
    gs=sorted(float(k) for k in d["22"])
    I1=[d["22"][str(g)]["I1"] for g in gs]
    inv_s=[1.0/d["22"][str(g)]["sig"] for g in gs]
    inv_g=[1.0/d["22"][str(g)]["gap"] for g in gs]
    def sp(x,y,n=20000,seed=2):
        r_=np.random.default_rng(seed)
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        r=np.corrcoef(rx,ry)[0,1]
        null=[np.corrcoef(rx,r_.permutation(ry))[0,1] for _ in range(n)]
        return r, float(np.mean(np.abs(null)>=abs(r)))
    r1,p1=sp(I1,inv_s); r2,p2=sp(I1,inv_g)
    print(f"  Spearman(I(B(1)), 1/sigma) = {r1:+.3f}, p_perm={p1:.4f} (n={len(gs)})")
    print(f"  Spearman(I(B(1)), 1/gap)   = {r2:+.3f}, p_perm={p2:.4f}")
    check("tendencia de rango CMI vs 1/sigma", r1>0 and p1<0.05, "")
    gmin=min(d["22"][str(g)]["gauss"] for g in gs)
    check("Gauss en el sweep", gmin>0.999, f"[min {gmin:.5f}]")
print("== 4) armonizacion de convencion FR ==")
f=rng.uniform(0.1,1.0,1000)
check("-2 log f = -log(f^2): raiz(v1) y cuadrada(companions) coinciden",
      np.allclose(-2*np.log(f), -np.log(f**2)), "")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
