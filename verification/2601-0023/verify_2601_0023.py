#!/usr/bin/env python3
"""verify_2601_0023.py -- Suite for 2601.0023 v2.

Checks (exact, small-model):
 1. Lemma 4.1 (omega=0 Dirichlet identity): machine precision.
 2. v1's Lemma 4.3 (E = 1/2 sum gamma ||[S(w),O]||^2_KMS) is FALSE in general:
    deviations up to ~22% on a TFIM N=4 Davies model.
 3. Corrected Lemma 4.3: E = sum_w gamma(w) e^{beta w/2} Re<S(w)O,[S(w),O]>_KMS
    -- exact (machine precision); reduces to Lemma 4.1 at w=0.
 4. Pairwise +-w positivity (each KMS pair is itself detailed-balanced), which
    rescues Proposition 4.5 (witness floor) under the corrected form.
 5. KMS multiplication bound: the v1 step ||XO|| <= ||X||inf ||O|| is false
    (1-qubit counterexample, ratio = e^{beta*Delta/4}); corrected bound with
    c_sigma = (lmax/lmin)^{1/4}: 0 violations on 300 random draws.
 6. Positivity pinning: for S = near + delta*tail and far-supported O,
    E_delta(O) = delta^2 * E_tail(O) EXACTLY (DB positivity at every delta
    kills the linear term; the dissipator is quadratic in S). This repairs
    Lemmas 4.12/4.13: the e^{-2 eps/xi} exponent survives; constants change to
    sup_w gamma(w)e^{beta w/2} and pick up c_sigma^2.
 7. (--with-tfim) TFIM N=10 witness at the declared parameters of the
    appendix script (J=1, h=1.5, S=sigma^z center), for comparison with the
    ED ranges reported in the companion 2601.0022 (~1 min).
"""
from pathlib import Path
import sys

import numpy as np

rng = np.random.default_rng(26010023)
fails = []
def check(name, ok, det=""):
    suffix = f" {det}" if det else ""
    print(f"  {name}: {'OK' if ok else 'FAIL'}{suffix}")
    if not ok: fails.append(name)
I2=np.eye(2); Xp=np.array([[0,1],[1,0]],dtype=complex); Zp=np.diag([1.,-1.]).astype(complex)
def kron_all(ops):
    out=ops[0]
    for o in ops[1:]: out=np.kron(out,o)
    return out
def tfim(N,J,h):
    H=np.zeros((2**N,2**N),dtype=complex)
    for i in range(N-1):
        ops=[I2]*N; ops[i]=Xp; ops[i+1]=Xp; H+=-J*kron_all(ops)
    for i in range(N):
        ops=[I2]*N; ops[i]=Zp; H+=-h*kron_all(ops)
    return H
def bohr(Se, E, tol=1e-9):
    diffs=E[:,None]-E[None,:]; freqs=[]
    for f in diffs.flatten():
        if not any(abs(f-g)<tol for g in freqs): freqs.append(f)
    return {om: Se*(np.abs(diffs-om)<tol) for om in freqs
            if np.abs(Se*(np.abs(diffs-om)<tol)).max()>1e-12}

# modelo TFIM N=4 en base propia
N=4; J,h,beta=1.0,1.3,0.7
E,U = np.linalg.eigh(tfim(N,J,h))
w=np.exp(-beta*(E-E.min())); w/=w.sum(); W=np.outer(np.sqrt(w),np.sqrt(w))
kms=lambda A,B: complex(np.sum(W*np.conj(A)*B))
ops=[I2]*N; ops[1]=Zp; Se=U.conj().T@kron_all(ops)@U
Sw=bohr(Se,E)
g=lambda om: np.exp(-om**2/8)*np.exp(beta*om/2)
def Edir(O, comps=None):
    comps = comps if comps is not None else Sw
    tot=0.0
    for om,M in comps.items():
        LO=M.conj().T@O@M-0.5*(M.conj().T@M@O+O@M.conj().T@M)
        tot+=-g(om)*kms(O,LO).real
    return tot
def randherm(d):
    R=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); return (R+R.conj().T)/2
d=2**N
print("== 1-3) identidades de Dirichlet (TFIM N=4, canal unico Davies) ==")
S0=Sw[min(Sw,key=lambda x:abs(x))]
mx1=mxv1=mxc=0
for t in range(8):
    O=randherm(d)
    L0=S0@O@S0-0.5*(S0@S0@O+O@S0@S0)   # S0 herm
    lhs0=-g(0)*kms(O,L0).real; rhs0=0.5*g(0)*kms(S0@O-O@S0,S0@O-O@S0).real
    mx1=max(mx1,abs(lhs0-rhs0)/max(abs(lhs0),1e-14))
    lhs=Edir(O)
    v1=0.5*sum(g(om)*kms(M@O-O@M,M@O-O@M).real for om,M in Sw.items())
    corr=sum(g(om)*np.exp(beta*om/2)*kms(M@O,M@O-O@M).real for om,M in Sw.items())
    mxv1=max(mxv1,abs(lhs-v1)/abs(lhs)); mxc=max(mxc,abs(lhs-corr)/abs(lhs))
check("Lemma 4.1 exacta", mx1<1e-12, f"[err {mx1:.1e}]")
print(f"  forma v1 de Lemma 4.3: desviacion max = {mxv1:.3f} (FALSA como identidad)")
check("Lemma 4.3 corregida exacta", mxc<1e-12, f"[err {mxc:.1e}]")
print("== 4) positividad por pares ==")
neg=0
for om in sorted(o for o in Sw if o>1e-9):
    for t in range(15):
        O=randherm(d); pair=0.0
        for o2 in (om,-om):
            if o2 in Sw:
                M=Sw[o2]; LO=M.conj().T@O@M-0.5*(M.conj().T@M@O+O@M.conj().T@M)
                pair+=-g(o2)*kms(O,LO).real
        if pair<-1e-12: neg+=1
check("pares +-omega no negativos", neg==0, f"[{neg} negativos]")
print("== 5) cota de multiplicacion KMS ==")
lmax,lmin=w.max(),w.min(); cs=(lmax/lmin)**0.25
worst=0; violc=0
for t in range(300):
    Xe=randherm(d); R=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); Oe=R
    nX=np.linalg.norm(Xe,2); nO=np.sqrt(max(kms(Oe,Oe).real,1e-300))
    r=np.sqrt(max(kms(Xe@Oe,Xe@Oe).real,0))/(nX*nO)
    worst=max(worst,r)
    if r>cs+1e-9: violc+=1
bq,D=3.0,2.0
wq=np.array([1,np.exp(-bq*D)]); wq/=wq.sum(); Wq=np.outer(np.sqrt(wq),np.sqrt(wq))
Oq=np.array([[0,0],[1,0]],dtype=complex)
ratio=np.sqrt(np.sum(Wq*np.abs(Xp@Oq)**2))/np.sqrt(np.sum(Wq*np.abs(Oq)**2))
check("contraejemplo 1-qubit = e^{bD/4}", abs(ratio-np.exp(bq*D/4))<1e-9, f"[{ratio:.4f}]")
check("cota corregida c_sigma", violc==0, f"[peor ratio {worst:.3f} <= c_sigma {cs:.3f}]")
print("== 6) positivity pinning: E_delta = delta^2 E_tail ==")
dA,dB=2,4; dd=dA*dB
EA=np.array([0.,1.3]); EB=np.array([0.,0.7,1.9,2.6])
E2=(EA[:,None]+EB[None,:]).flatten()
w2=np.exp(-beta*(E2-E2.min())); w2/=w2.sum(); W2=np.outer(np.sqrt(w2),np.sqrt(w2))
kms2=lambda A,B: complex(np.sum(W2*np.conj(A)*B))
def Edir2(S,O):
    tot=0.0
    for om,M in bohr(S,E2).items():
        LO=M.conj().T@O@M-0.5*(M.conj().T@M@O+O@M.conj().T@M)
        tot+=-g(om)*kms2(O,LO).real
    return tot
SA=randherm(dA); Snear=np.kron(SA,np.eye(dB))
T=randherm(dd); T/=np.linalg.norm(T,2)
OB=randherm(dB); OB-=np.trace(OB)/dB*np.eye(dB); Ofar=np.kron(np.eye(dA),OB)
ET=Edir2(T,Ofar); okpin=True
pin_abs=pin_rel=0.0
for delta in (1e-1,1e-2,1e-3):
    v=Edir2(Snear+delta*T,Ofar)
    expected = delta**2*ET
    err = abs(v-expected)
    scale = max(abs(v), abs(expected), 1e-30)
    pin_abs=max(pin_abs, err)
    pin_rel=max(pin_rel, err/scale)
    if err > 1e-9*max(1.0, abs(v), abs(expected)): okpin=False
check("E_near(Ofar)=0", abs(Edir2(Snear,Ofar))<1e-12, "")
check("E_{N+dT}(Ofar) = d^2 E_T(Ofar) exacta", okpin,
      f"[E_T={ET:.4e}, max_abs={pin_abs:.1e}, max_rel={pin_rel:.1e}]")
if "--with-tfim" in sys.argv:
    print("== 7) TFIM N=10 witness (J=1, h=1.5, S=Z centro; ~1 min) ==")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from witness_core import run
    run(1.5, "Z")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
