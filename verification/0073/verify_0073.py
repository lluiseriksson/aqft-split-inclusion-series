"""Verification for 2512.0073 (Davies interface lemmas + TFIM witnesses).
A. Lemma 2 (Bohr-block decomposition of the KMS Dirichlet form):
   E_sigma(O) =?= (1/2) sum_omega gamma(omega) ||[S(omega),O]||^2_{2,sigma}
   tested against the direct Dirichlet form of the full Davies generator,
   for random observables. (Lemma 1 is the omega=0 special case.)
B. Table 2 reproduction: optimized witness R_opt(eps) for TFIM N=8, beta=1,
   h=1.5, J=1, S=sigma^z at center site.
"""
import numpy as np
N = 8
J, h, beta = 1.0, 1.5, 1.0
gamma0, nuc = 1.0, 10.0
dOm = 1e-9   # exact Bohr grouping (no secular binning: identity is exact per block)
sx = np.array([[0,1],[1,0]], complex); sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]], complex); I2 = np.eye(2)
def op_at(o,i):
    M = np.array([[1]],complex)
    for k in range(N): M = np.kron(M, o if k==i else I2)
    return M
HS = sum(-J*op_at(sx,i)@op_at(sx,i+1) for i in range(N-1)) + sum(-h*op_at(sz,i) for i in range(N))
w, V = np.linalg.eigh(HS); D = 2**N
p = np.exp(-beta*w); p /= p.sum(); sqp = np.sqrt(p)
j0 = N//2
S = op_at(sz, j0)
St = V.conj().T@S@V
W = w[:,None]-w[None,:]
keys = np.unique(np.round(W/1e-9).astype(np.int64))
def grate(om):
    if abs(om) < 1e-8: return gamma0/beta
    Jn = gamma0*abs(om)*np.exp(-abs(om)/nuc)
    nb = 1.0/(np.exp(beta*abs(om))-1.0)
    return Jn*(1+nb) if om>0 else Jn*nb
comps = []
for key in keys:
    A = np.where(np.abs(W - key*1e-9) < 5e-10, St, 0)
    if np.abs(A).max() > 1e-12:
        comps.append((key*1e-9, grate(key*1e-9), A))
def kms(A,B): return np.einsum('i,ij,j,ij->', sqp, A.conj(), sqp, B)
def Ldag(O):
    out = np.zeros_like(O)
    for om,g,A in comps:
        if g < 1e-14: continue
        Ad = A.conj().T
        out += g*(Ad@O@A - 0.5*(Ad@A@O + O@Ad@A))
    return out
rng = np.random.default_rng(3)
print("=== A: Lemma 2 identity test (energy basis, exact Bohr blocks) ===")
worst = 0.0
for t in range(6):
    X = rng.normal(size=(D,D)) + 1j*rng.normal(size=(D,D))
    O = (X + X.conj().T)/2
    lhs = -np.real(kms(O, Ldag(O)))
    rhs = 0.5*sum(g*np.real(kms(A@O-O@A, A@O-O@A)) for om,g,A in comps if g>1e-14)
    rel = abs(lhs-rhs)/max(abs(lhs),1e-300)
    worst = max(worst, rel)
    if t < 3: print(f"  trial {t}: E_direct={lhs:.6e}  sum-blocks={rhs:.6e}  rel.err={rel:.2e}")
print(f"  worst relative deviation over 6 random O: {worst:.2e}")

print("\n=== B: Table 2 witness R_opt(eps), N=8, beta=1 ===")
S0 = np.where(np.abs(W) < 1e-9, St, 0)     # omega=0 block (energy basis)
P = {'x':sx,'y':sy,'z':sz}
for eps in [1,2,3]:
    k = j0 - eps
    best = 0.0
    # optimize over span{sx,sy,sz} at site k: R is a Rayleigh quotient of a 3x3 pencil
    ops = [V.conj().T@op_at(P[a],k)@V for a in 'xyz']
    Gm = np.zeros((3,3),complex); Mm = np.zeros((3,3),complex)
    for a in range(3):
        for b in range(3):
            Gm[a,b] = kms(ops[a],ops[b])
            Ca = S0@ops[a]-ops[a]@S0; Cb = S0@ops[b]-ops[b]@S0
            Mm[a,b] = kms(Ca,Cb)
    ev = np.linalg.eigvals(np.linalg.solve((Gm+Gm.conj().T)/2, (Mm+Mm.conj().T)/2))
    best = float(np.max(np.real(ev)))
    print(f"  eps={eps} (k=j0-eps): R_opt={best:.6f}   (paper: {dict([(1,0.134648),(2,0.0885503),(3,0.0684289)])[eps]})")
