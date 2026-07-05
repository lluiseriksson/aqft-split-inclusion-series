#!/usr/bin/env python3
"""verify_2601_0031.py -- Suite for 2601.0031 v2 (typed pipeline).

 1. Lemma 4.1 in the GNS inner product <A,B> = Tr(A^dag sigma B) used by this
    paper: exact omega=0 Dirichlet identity (machine precision).
 2. Convention bridge: R_opt computed in GNS vs KMS conventions on the same
    TFIM witness problem -- both positive, numerically different (the paper's
    numbers are GNS; companions 2601.0022/0023 use KMS).
 3. Witness N-trend (Figure 1): R_opt^(1)(eps; N) decreases with N at fixed
    eps for N in {6,8,10} (GNS, J=1, h=1.5, beta=1).
 4. Lemma 5.2 (pinching Pythagoras): exact on random states.
 5. Lemma 5.1: the v1 work-cost definition (eq 25, W = F(w') - F(w)) makes
    the lemma FALSE (random energy-conserving unitaries violate it); with the
    corrected sign W_spent = F(w) - F(w') (battery free-energy decrease) the
    bound W_spent >= dF_S holds on all draws -- same error class as the
    work-sign bug repaired in 2512.0061 v2.
 6. Assumption 5.1 sanity: Delta o T_t = T_t o Delta for a secular Davies
    generator (random O, machine precision).
"""
import sys

import numpy as np
from scipy import linalg as la

rng = np.random.default_rng(26010031)
fails = []
def check(name, ok, det=""):
    suffix = f" {det}" if det else ""
    print(f"  {name}: {'OK' if ok else 'FAIL'}{suffix}")
    if not ok: fails.append(name)
I2=np.eye(2); Xp=np.array([[0,1],[1,0]],dtype=complex); Zp=np.diag([1.,-1.]).astype(complex)
Yp=np.array([[0,-1j],[1j,0]])
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
def site(P,i,N):
    ops=[I2]*N; ops[i]=P; return kron_all(ops)

print("== 1) Lemma 4.1 en GNS ==")
N=4; J,h,beta=1.0,1.5,1.0
H=tfim(N,J,h); E,U=np.linalg.eigh(H)
w=np.exp(-beta*(E-E.min())); w/=w.sum()
sig=U@np.diag(w)@U.conj().T
S=site(Zp,1,N); Se=U.conj().T@S@U
S0=U@(Se*(np.abs(E[:,None]-E[None,:])<1e-9))@U.conj().T
g0=0.37
def L0(O): return g0*(S0@O@S0-0.5*(S0@S0@O+O@S0@S0))
gns=lambda A,B: complex(np.trace(A.conj().T@sig@B))
mx=0
for t in range(8):
    R=rng.normal(size=(2**N,2**N))+1j*rng.normal(size=(2**N,2**N)); O=(R+R.conj().T)/2
    lhs=-gns(O,L0(O)).real
    C=S0@O-O@S0; rhs=0.5*g0*np.trace(sig@C.conj().T@C).real
    mx=max(mx,abs(lhs-rhs)/max(abs(lhs),1e-14))
check("identidad omega=0 exacta (GNS)", mx<1e-12, f"[err {mx:.1e}]")

print("== 2-3) puente GNS<->KMS y trend con N ==")
def ropt(N, eps, conv, J=1.0, h=1.5, beta=1.0):
    H=tfim(N,J,h); E,U=np.linalg.eigh(H)
    w=np.exp(-beta*(E-E.min())); w/=w.sum()
    Se=U.conj().T@site(Zp,N//2,N)@U
    S0e=Se*(np.abs(E[:,None]-E[None,:])<1e-9)
    k=N//2+eps
    best=0.0
    Wk=np.outer(np.sqrt(w),np.sqrt(w))
    for P in (Xp,Yp,Zp):
        Oe=U.conj().T@site(P,k,N)@U
        C=S0e@Oe-Oe@S0e
        if conv=='kms':
            num=np.sum(Wk*np.abs(C)**2).real; den=np.sum(Wk*np.abs(Oe)**2).real
        else:
            num=np.trace(np.diag(w)@C.conj().T@C).real; den=np.trace(np.diag(w)@Oe.conj().T@Oe).real
        best=max(best,num/den)
    return best
for eps in (1,2):
    rg=ropt(8,eps,'gns'); rk=ropt(8,eps,'kms')
    print(f"  N=8, eps={eps}: R_opt GNS={rg:.4e}  KMS={rk:.4e}  ratio={rg/rk:.3f}")
trend_ok=True
vals={}
for eps in (1,2):
    seq=[ropt(Nn,eps,'gns') for Nn in (6,8,10)]
    vals[eps]=seq
    if not (seq[0]>seq[1]>seq[2]): trend_ok=False
    print(f"  trend eps={eps}: N=6,8,10 -> " + ", ".join(f"{v:.3e}" for v in seq))
check("R_opt decrece con N (eps=1,2)", trend_ok, "")

print("== 4) Pythagoras de pinching ==")
d=6; Ed=np.sort(rng.normal(0,1,d)); sg=np.exp(-Ed); sg/=sg.sum()
def vn_rel(a,b):
    ea,Ua=np.linalg.eigh(a); eb,Ub=np.linalg.eigh(b)
    la_=Ua@np.diag(np.log(np.maximum(ea,1e-300)))@Ua.conj().T
    lb_=Ub@np.diag(np.log(np.maximum(eb,1e-300)))@Ub.conj().T
    return float(np.trace(a@(la_-lb_)).real)
mxp=0
for t in range(6):
    R=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); rho=R@R.conj().T; rho/=np.trace(rho).real
    drho=np.diag(np.diag(rho))
    lhs=vn_rel(rho,np.diag(sg)); rhs=vn_rel(rho,drho)+vn_rel(drho,np.diag(sg))
    mxp=max(mxp,abs(lhs-rhs))
check("S(rho||sigma)=S(rho||Drho)+S(Drho||sigma)", mxp<1e-10, f"[err {mxp:.1e}]")

print("== 5) Lemma 5.1: cota de trabajo con U conservadora ==")
# S: qubit gap 1; B: qubit gap 1; W: qutrit escalera gaps 1
HS=np.diag([0.,1.]); HB=np.diag([0.,1.]); HW=np.diag([0.,1.,2.])
Ht=(np.kron(np.kron(HS,np.eye(2)),np.eye(3))+np.kron(np.kron(np.eye(2),HB),np.eye(3))
    +np.kron(np.kron(np.eye(2),np.eye(2)),HW))
Et=np.diag(Ht).real
bq=0.8
gB=np.exp(-bq*np.diag(HB)); gB/=gB.sum()
gW=np.exp(-bq*np.diag(HW)); gW/=gW.sum()
sS=np.exp(-bq*np.diag(HS)); sS/=sS.sum()
def rel(a,b): return vn_rel(a,b)
viol=0; viol_corr=0
for t in range(100):
    G=rng.normal(size=(12,12))+1j*rng.normal(size=(12,12)); G=(G+G.conj().T)/2
    G=G*(np.abs(Et[:,None]-Et[None,:])<1e-9)   # bloque-diagonal en energia total
    Uu=la.expm(1j*G)
    r=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)); rho=r@r.conj().T; rho/=np.trace(rho).real
    q=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); om=q@q.conj().T; om/=np.trace(om).real
    R0=np.kron(np.kron(rho,np.diag(gB)),om)
    Rf=Uu@R0@Uu.conj().T
    Rf_SW=np.trace(Rf.reshape(2,2,3,2,2,3),axis1=1,axis2=4).reshape(6,6)
    rho_f=np.trace(Rf_SW.reshape(2,3,2,3),axis1=1,axis2=3)
    om_f=np.trace(Rf_SW.reshape(2,3,2,3),axis1=0,axis2=2)
    W_v1  = (rel(om_f,np.diag(gW))-rel(om,np.diag(gW)))/bq   # def (25) de v1
    W_sp  = -W_v1                                             # coste corregido: bajada de F de la bateria
    dFS=(rel(rho_f,np.diag(sS))-rel(rho,np.diag(sS)))/bq
    if W_v1 < dFS - 1e-9: viol+=1
    if W_sp < dFS - 1e-9: viol_corr+=1
print(f"  def v1 (eq 25): {viol}/100 violaciones de la Lemma 5.1 -> FALSA con ese signo")
check("signo corregido W_spent >= Delta F_S", viol_corr==0, f"[{viol_corr}/100]")

print("== 6) Assumption 5.1 (covariancia del drift secular) ==")
Sw={}
diffs=E[:,None]-E[None,:]
freqs=[]
for f in diffs.flatten():
    if not any(abs(f-x)<1e-9 for x in freqs): freqs.append(f)
gam=lambda om: np.exp(-om**2/8)*np.exp(beta*om/2)
def Lstar(rho_e):
    out=np.zeros_like(rho_e)
    for om in freqs:
        M=Se*(np.abs(diffs-om)<1e-9)
        if np.abs(M).max()<1e-12: continue
        out+=gam(om)*(M@rho_e@M.conj().T-0.5*(M.conj().T@M@rho_e+rho_e@M.conj().T@M))
    return out
mxc=0
for t in range(4):
    R=rng.normal(size=(2**N,2**N))+1j*rng.normal(size=(2**N,2**N)); r0=R@R.conj().T; r0/=np.trace(r0).real
    re=U.conj().T@r0@U
    piL=Lstar(re)*(np.abs(diffs)<1e-9)
    Lpi=Lstar(re*(np.abs(diffs)<1e-9))
    mxc=max(mxc,np.abs(piL-Lpi).max())
check("Delta L* = L* Delta (secular)", mxc<1e-12, f"[err {mxc:.1e}]")

print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
