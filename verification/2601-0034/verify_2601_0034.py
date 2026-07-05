#!/usr/bin/env python3
"""verify_2601_0034.py -- Finite-dimensional dictionary checks for 2601.0034 v2.

 1. Reduction: with the natural Type I factorization, I^N(A:C|B) equals the
    standard CMI (chain-rule identity, machine precision).
 2. N-dependence demo (Remark 3.4 made concrete): rotating the split
    identification of the AB layer by a unitary V(theta) while keeping the
    ABC layer natural makes I^N vary and become NEGATIVE for some theta --
    so non-negativity indeed cannot be claimed for arbitrary fixed N.
 3. beta_0(t) = (pi/4) sech^2(pi t/2) integrates to 1 (the universal
    recovery weight of Junge et al., as quoted in Remark 4.3).
 4. Anchor for the corrected Assumption 4.4: on random 3-qubit states with
    the Petz recovery, I >= -log F (squared fidelity) holds on all draws,
    while the v1 half-form -log F <= I/2 FAILS on most draws: the v1
    assumption was strictly stronger than its finite-dimensional motivation.
    (Same factor-2 correction as 2512.0101 v2 and 2601.0020 v2.)
 5. Theorem 4.5 arithmetic: 1 - x <= -log x on (0,1]; purified-distance
    chain with the corrected constant sqrt(K).
"""
import sys

import numpy as np
from scipy import integrate, linalg as la

rng = np.random.default_rng(26010034)
fails = []
def check(name, ok, det=""):
    suffix = f" {det}" if det else ""
    print(f"  {name}: {'OK' if ok else 'FAIL'}{suffix}")
    if not ok: fails.append(name)

def ptrace(rho, dims, keep):
    n=len(dims); rho=rho.reshape(dims+dims)
    for ax in sorted((i for i in range(n) if i not in keep), reverse=True):
        rho=np.trace(rho, axis1=ax, axis2=ax+rho.ndim//2)
    d=int(np.prod([dims[i] for i in keep])) if keep else 1
    return rho.reshape(d,d)
def vn(r):
    e=np.linalg.eigvalsh(r); e=e[e>1e-14]; return float(-(e*np.log(e)).sum())
def rel(a,b):
    ea,Ua=np.linalg.eigh(a); eb,Ub=np.linalg.eigh(b)
    la_=Ua@np.diag(np.log(np.maximum(ea,1e-300)))@Ua.conj().T
    lb_=Ub@np.diag(np.log(np.maximum(eb,1e-300)))@Ub.conj().T
    return float(np.trace(a@(la_-lb_)).real)
def randstate(d):
    X=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); r=X@X.conj().T
    return r/np.trace(r).real
def mpow(r,p):
    e,U=np.linalg.eigh(r); e=np.maximum(e,1e-300); return (U*e**p)@U.conj().T

print("== 1) reduccion a CMI estandar (N natural) ==")
dims=[2,2,2]; mx=0
for t in range(6):
    rho=randstate(8)
    rA=ptrace(rho,dims,[0]); rB=ptrace(rho,dims,[1]); rBC=ptrace(rho,dims,[1,2])
    rAB=ptrace(rho,dims,[0,1])
    IN = rel(rho, np.kron(rA,rBC)) - rel(rAB, np.kron(rA,rB))
    cmi = vn(rAB)+vn(rBC)-vn(rB)-vn(rho)
    mx=max(mx,abs(IN-cmi))
check("I^N = CMI (identidad de cadena)", mx<1e-10, f"[err {mx:.1e}]")

print("== 2) dependencia de N: identificacion AB rotada ==")
# estado con A ~ descorrelacionado de BC: I(A:BC) ~ O(eps), pero una
# identificacion rotada del corte A|B crea I_rot(A:B) > I(A:BC) => I^N < 0
eps=0.05
plus=np.array([1,1],dtype=complex)/np.sqrt(2)
phi=np.zeros(4,dtype=complex); phi[0]=phi[3]=1/np.sqrt(2)
psi=np.kron(np.outer(plus,plus.conj()), np.outer(phi,phi.conj()))
rho2=(1-eps)*psi + eps*np.eye(8)/8
rA=ptrace(rho2,dims,[0]); rBC=ptrace(rho2,dims,[1,2]); rAB=ptrace(rho2,dims,[0,1])
I_ABC = rel(rho2, np.kron(rA,rBC))
neg=0; vals=[]
for t in range(20):
    G=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    V,_=np.linalg.qr(G)                     # Haar-ish
    rot=V.conj().T@rAB@V
    a=ptrace(rot,[2,2],[0]); b=ptrace(rot,[2,2],[1])
    IN = I_ABC - rel(rot, np.kron(a,b))
    vals.append(IN)
    if IN < -1e-8: neg+=1
print(f"  I(A:BC)={I_ABC:.4f}; I^N sobre 20 identificaciones: "
      f"rango [{min(vals):.4f}, {max(vals):.4f}], negativos: {neg}/20")
check("I^N varia con N y puede ser negativo", (max(vals)-min(vals)>1e-3) and neg>0, "")
print("== 3) normalizacion del peso beta_0 ==")
val,_=integrate.quad(lambda t: np.pi/4/np.cosh(np.pi*t/2)**2, -50, 50)
check("int beta_0 = 1", abs(val-1)<1e-10, f"[{val:.12f}]")

print("== 4) ancla finito-dim de la Assumption 4.4 corregida ==")
viol_corr=0; viol_half=0; ndr=40
for t in range(ndr):
    rho=randstate(8)
    rB=ptrace(rho,dims,[1]); rBC=ptrace(rho,dims,[1,2]); rAB=ptrace(rho,dims,[0,1])
    I = vn(rAB)+vn(rBC)-vn(rB)-vn(rho)
    M=np.kron(np.eye(2), np.kron(mpow(rB,-0.5), np.eye(2)))
    sig=np.kron(np.eye(2), mpow(rBC,0.5))@M@np.kron(rAB,np.eye(2)/1)@M@np.kron(np.eye(2), mpow(rBC,0.5))
    sig=sig/np.trace(sig).real
    sr=mpow(rho,0.5)
    F=float(np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(sr@sig@sr),0)))**2)
    if -np.log(max(F,1e-300)) > I + 1e-9: viol_corr+=1
    if -np.log(max(F,1e-300)) > 0.5*I + 1e-9: viol_half+=1
print(f"  -log F <= I (corregida): {ndr-viol_corr}/{ndr} draws la cumplen")
print(f"  -log F <= I/2 (v1):      {ndr-viol_half}/{ndr} draws la cumplen -> la half-form NO es importable")
check("assumption corregida consistente (Petz)", viol_corr==0, f"[{viol_corr}/{ndr}]")
check("half-form de v1 refutada como ancla", viol_half>ndr//2, f"[{viol_half}/{ndr} violaciones]")

print("== 5) aritmetica del Teorema 4.5 ==")
xs=rng.uniform(1e-6,1,2000)
check("1-x <= -log x", np.all(1-xs <= -np.log(xs)+1e-12), "")
K,alpha,w=0.7,0.9,3.0
IN=K*np.exp(-alpha*w)
check("P <= sqrt(K) e^{-aw/2} (constante corregida)",
      np.sqrt(1-np.exp(-IN)) <= np.sqrt(K)*np.exp(-alpha*w/2)+1e-12, "")

print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
