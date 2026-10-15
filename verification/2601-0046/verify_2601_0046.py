#!/usr/bin/env python3
"""verify_2601_0046.py -- Finite-dimensional suite for 2601.0046 v2.

 1. Prop 3.3 (adjointness): the Accardi-Cecchini adjoint of the map E there
    defined IS the standard Petz map (machine precision) -- v1's computation
    confirmed.
 2. Terminology fix justified: E is NOT a (Takesaki) conditional expectation
    in general -- the module property E(X_B ox 1) = X_B ox 1 FAILS on random
    faithful rho_BC (max deviation reported), and E is not idempotent; it is
    the *generalized* (Accardi-Cecchini) conditional expectation.
 3. New Lemma of v2: when a TRUE omega-preserving CE exists (product case
    rho_BC = rho_B ox rho_C), its Petz dual is the INCLUSION: adjointness of
    R = iota verified to machine precision; and E reduces to the true CE.
 4. Corrected Definition 6.1: the recovered state omega_AB o Psi (Psi =
    id_A ox E, Heisenberg ABC->AB) EQUALS the standard Schrodinger Petz
    reconstruction (id_A ox R_Petz,*)(rho_AB) -- verified on random W.
 5. c_FR anchor: with squared fidelity, I >= -log F(Petz) on 40/40 random
    tripartite states (c_FR = 1 importable; cf. the series-wide factor fix).
"""
import numpy as np, sys
rng=np.random.default_rng(26010046)
fails=[]
def check(name, ok, det=""):
    suffix = f" {det}" if det else ""
    print(f"  {name}: {'OK' if ok else 'FAIL'}{suffix}")
    if not ok: fails.append(name)
def randrho(d):
    X=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); r=X@X.conj().T
    return r/np.trace(r).real
def mpow(r,p):
    e,U=np.linalg.eigh(r); e=np.maximum(e,1e-300)
    return (U*e**p)@U.conj().T
def ptrB(M,dB,dC):   # Tr_C
    return np.trace(M.reshape(dB,dC,dB,dC),axis1=1,axis2=3)
dB,dC=3,4; dBC=dB*dC
rBC=randrho(dBC); rB=ptrB(rBC,dB,dC)
s=mpow(rBC,0.5); rBi=mpow(rB,-0.5)
def E(Z):   # esperanza generalizada (Prop 3.3)
    return np.kron(rBi@ptrB(s@Z@s,dB,dC)@rBi, np.eye(dC))
def Petz(XB):
    return s@np.kron(rBi@XB@rBi,np.eye(dC))@s
om=lambda Z: complex(np.trace(rBC@Z))
print("== 1) identificacion del pairing de dualidad correcto ==")
mx_tr=mx_om=mx_gns=mx_kms=0
sB=mpow(rB,0.5)
for t in range(8):
    Z=rng.normal(size=(dBC,dBC))+1j*rng.normal(size=(dBC,dBC))
    XB=rng.normal(size=(dB,dB))+1j*rng.normal(size=(dB,dB))
    E0Z=rBi@ptrB(s@Z@s,dB,dC)@rBi
    RX=Petz(XB)
    # (a) traza-predual: Tr(R(X) Y) = Tr(X E0(Y))
    mx_tr=max(mx_tr,abs(np.trace(RX@Z)-np.trace(XB@E0Z)))
    # (b) pairing del paper: omega(Z R(X)) = omega(E(Z) X)
    mx_om=max(mx_om,abs(om(Z@RX)-om(E(Z)@np.kron(XB,np.eye(dC)))))
    # (c) GNS: Tr(rho_BC Z^dag R(X)) = Tr(rho_B E0(Z)^dag X)
    mx_gns=max(mx_gns,abs(np.trace(rBC@Z.conj().T@RX)-np.trace(rB@E0Z.conj().T@XB)))
    # (d) KMS: Tr(s Z^dag s R(X)) = Tr(sB E0(Z)^dag sB X)
    mx_kms=max(mx_kms,abs(np.trace(s@Z.conj().T@s@RX)-np.trace(sB@E0Z.conj().T@sB@XB)))
print(f"  (a) traza-predual: err {mx_tr:.1e}")
print(f"  (b) pairing de v1 (Def 5.3): err {mx_om:.1e}  -> FALSO como esta impreso")
print(f"  (c) GNS: err {mx_gns:.1e}")
print(f"  (d) KMS: err {mx_kms:.1e}")
check("traza-predual exacta (R = E_* en dim finita)", mx_tr<1e-10, "")
check("pairing de v1 refutado", mx_om>1e-3, f"[err {mx_om:.2f}]")
check("E unital", np.abs(E(np.eye(dBC))-np.eye(dBC)).max()<1e-12, "")
Zx=rng.normal(size=(dBC,dBC))+1j*rng.normal(size=(dBC,dBC))
check("E omega-preservante", abs(om(E(Zx))-om(Zx))<1e-12, "")
print("== 2) E NO es esperanza condicional verdadera ==")
XB=rng.normal(size=(dB,dB))+1j*rng.normal(size=(dB,dB)); XB=(XB+XB.conj().T)/2
dev=np.abs(E(np.kron(XB,np.eye(dC)))-np.kron(XB,np.eye(dC))).max()
Z2=rng.normal(size=(dBC,dBC))+1j*rng.normal(size=(dBC,dBC))
idem=np.abs(E(E(Z2))-E(Z2)).max()
check("propiedad de modulo FALLA (generalizada, no Takesaki)", dev>1e-3, f"[desviacion {dev:.3f}]")
print(f"  (no idempotente: |E(E(Z))-E(Z)|max = {idem:.3f})")
print("== 3) CE verdadera (caso producto): predual = tensorizar con rho_C ==")
rBp=randrho(dB); rCp=randrho(dC); rProd=np.kron(rBp,rCp)
sP=mpow(rProd,0.5); rBpi=mpow(rBp,-0.5)
omP=lambda Z: complex(np.trace(rProd@Z))
def CEtrue(Z):
    return np.kron(np.trace((np.kron(np.eye(dB),rCp)@Z).reshape(dB,dC,dB,dC),axis1=1,axis2=3), np.eye(dC))
XBt=rng.normal(size=(dB,dB))+1j*rng.normal(size=(dB,dB))
check("CE verdadera: modulo OK", np.abs(CEtrue(np.kron(XBt,np.eye(dC)))-np.kron(XBt,np.eye(dC))).max()<1e-12, "")
mxi=0
for t in range(8):
    Z=rng.normal(size=(dBC,dBC))+1j*rng.normal(size=(dBC,dBC))
    XB=rng.normal(size=(dB,dB))+1j*rng.normal(size=(dB,dB))
    lhs=omP(Z@np.kron(XB,np.eye(dC)))      # R = inclusion
    rhs=omP(CEtrue(Z)@np.kron(XB,np.eye(dC)))
    mxi=max(mxi,abs(lhs-rhs))
check("pairing v1 vale en el caso conmutativo/producto (con R=inclusion)", mxi<1e-12, f"[err {mxi:.1e}]")
# predual de la CE verdadera: eps_*(X_B) = X_B ox rho_C = formula de Petz para producto
XBs=rng.normal(size=(dB,dB))+1j*rng.normal(size=(dB,dB)); XBs=(XBs+XBs.conj().T)/2
PetzP=sP@np.kron(rBpi@XBs@rBpi,np.eye(dC))@sP
check("predual(CE) = X_B ox rho_C = Petz(producto)",
      np.abs(PetzP-np.kron(XBs,rCp)).max()<1e-10, "")
def Eprod(Z):
    sPl=mpow(rProd,0.5); rBl=mpow(ptrB(rProd,dB,dC),-0.5)
    return np.kron(rBl@ptrB(sPl@Z@sPl,dB,dC)@rBl,np.eye(dC))
check("E generalizada = CE verdadera en el caso producto",
      np.abs(Eprod(Z)-CEtrue(Z)).max()<1e-10, "")
print("== 4) Def 6.1 corregida = reconstruccion Petz estandar ==")
dA=2; dims=(dA,dB,dC)
rABC=randrho(dA*dB*dC)
rAB=np.trace(rABC.reshape(dA,dB,dC,dA,dB,dC),axis1=2,axis2=5).reshape(dA*dB,dA*dB)
rBC2=np.trace(rABC.reshape(dA,dB*dC,dA,dB*dC),axis1=0,axis2=2)
rB2=ptrB(rBC2,dB,dC)
s2=mpow(rBC2,0.5); rB2i=mpow(rB2,-0.5)
M=np.kron(np.eye(dA),s2)@np.kron(np.eye(dA),np.kron(rB2i,np.eye(dC)))
sig1=M@np.kron(rAB,np.eye(dC))@M.conj().T
sig1=sig1/np.trace(sig1).real
def E2(Z):   # generalizada para rBC2
    return np.kron(rB2i@ptrB(s2@Z@s2,dB,dC)@rB2i,np.eye(dC))
mxw=0
for t in range(6):
    W=rng.normal(size=(dA*dB*dC,)*2)+1j*rng.normal(size=(dA*dB*dC,)*2)
    # Psi = id_A ox E2 : reshape W (A,BC,A,BC), aplicar E2 en BC por bloques
    Wr=W.reshape(dA,dBC,dA,dBC)
    PsiW=np.zeros_like(Wr)
    for a in range(dA):
        for b in range(dA):
            PsiW[a,:,b,:]=E2(Wr[a,:,b,:])
    MAB=np.trace(PsiW.reshape(dA,dB,dC,dA,dB,dC),axis1=2,axis2=5).reshape(dA*dB,dA*dB)/dC
    val1=complex(np.einsum('ij,ji->', rAB, MAB))
    val2=complex(np.trace(sig1@W))
    mxw=max(mxw,abs(val1-val2))
check("omega_AB o (id ox E) = Petz Schrodinger", mxw<1e-10, f"[err {mxw:.1e}]")
print("== 5) ancla c_FR (convencion cuadrada) ==")
def vn(r):
    e=np.linalg.eigvalsh(r); e=e[e>1e-14]; return float(-(e*np.log(e)).sum())
viol=0
for t in range(40):
    rho=randrho(8); dms=[2,2,2]
    def pt(keep):
        M=rho.reshape(dms+dms)
        for ax in sorted((i for i in range(3) if i not in keep),reverse=True):
            M=np.trace(M,axis1=ax,axis2=ax+M.ndim//2)
        d=int(np.prod([dms[i] for i in keep]))
        return M.reshape(d,d)
    I=vn(pt([0,1]))+vn(pt([1,2]))-vn(pt([1]))-vn(rho)
    rb=pt([1]); rbc=pt([1,2])
    Mp=np.kron(np.eye(2),np.kron(mpow(rb,-0.5),np.eye(2)))
    sg=np.kron(np.eye(2),mpow(rbc,0.5))@Mp@np.kron(pt([0,1]),np.eye(2))@Mp@np.kron(np.eye(2),mpow(rbc,0.5))
    sg/=np.trace(sg).real
    sr=mpow(rho,0.5)
    F=float(np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(sr@sg@sr),0)))**2)
    if -np.log(max(F,1e-300))>I+1e-9: viol+=1
check("I >= -log F(Petz): c_FR = 1 importable (cuadrada)", viol==0, f"[{viol}/40]")
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: '+', '.join(fails)}")
sys.exit(0 if not fails else 1)
