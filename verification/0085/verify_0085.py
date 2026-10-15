"""Verification of the two checkable technical claims of 2512.0085.
A. Lemma 1 (elementary fidelity bound): 1 - F <= -log F for F in (0,1].
B. Davies interface direction (RIP-U): if the coupling's spatial tail decays,
   the omega=0 Dirichlet rate seen by a distant witness decays too -- the
   legitimate UPPER-envelope direction (consistent with the linear envelope
   lemma of 2512.0073 v2). TFIM N=8, exact.
"""
import numpy as np
# --- A
Fs = np.linspace(1e-6, 1.0, 100000)
gap = (-np.log(Fs)) - (1 - Fs)
print(f"A. min(-logF - (1-F)) over (0,1] = {gap.min():.3e}  (>=0 required): "
      f"{'HOLDS' if gap.min() >= -1e-12 else 'FAILS'}")

# --- B: exact TFIM omega=0 witness rate vs distance
N=8; J,h,beta=1.0,1.5,1.0
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex); I2=np.eye(2)
def op(o,i):
    M=np.array([[1]],complex)
    for k in range(N): M=np.kron(M,o if k==i else I2)
    return M
HS=sum(-J*op(sx,i)@op(sx,i+1) for i in range(N-1))+sum(-h*op(sz,i) for i in range(N))
w,V=np.linalg.eigh(HS); D=2**N; p=np.exp(-beta*w); p/=p.sum(); sq=np.sqrt(p)
j0=N//2; St=V.conj().T@op(sz,j0)@V; W=w[:,None]-w[None,:]
S0=np.where(np.abs(W)<1e-9,St,0)
def kms(A,B): return np.einsum('i,ij,j,ij->',sq,A.conj(),sq,B)
P={'x':sx,'y':sy,'z':sz}
def ropt(k):
    ops=[V.conj().T@op(P[a],k)@V for a in 'xyz']
    G=np.array([[kms(a,b) for b in ops] for a in ops])
    M=np.array([[kms(S0@a-a@S0,S0@b-b@S0) for b in ops] for a in ops])
    ev=np.linalg.eigvals(np.linalg.solve((G+G.conj().T)/2,(M+M.conj().T)/2))
    return float(np.max(np.real(ev)))
print("B. omega=0 witness rate R_opt vs distance eps (k = j0 - eps):")
vals=[]
for eps in [1,2,3]:
    r=ropt(j0-eps); vals.append(r); print(f"   eps={eps}: R_opt={r:.5f}")
print(f"   monotone decreasing (upper envelope decays): "
      f"{'YES' if all(vals[i]>vals[i+1] for i in range(len(vals)-1)) else 'NO'}")
