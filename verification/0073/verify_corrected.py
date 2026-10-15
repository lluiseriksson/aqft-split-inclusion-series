"""v2 corrections for 2512.0073, verified exactly (TFIM N=8, beta=1):
A. Corrected Lemma 2 (per-block exact identity with modular weights):
   -Re<O, Ldag_w O>_KMS = gamma(w) e^{bw/2} [ 1/2||S(w)O||^2 + 1/2||O S(w)^dag||^2
                                              - Re<S(w)O, O S(w)> ]   (KMS norms)
   (v1 claimed (gamma/2)||[S(w),O]||^2 -- wrong for w != 0.)
B. NEW exact tail-reduction lemma: for O supported at distance >= eps from the
   coupling region, E_sigma(O; {S(w)}) = E_sigma(O; {tail_eps S(w)}) EXACTLY
   (per KMS-conjugate pair; quadratic-in-tail + positivity argument).
"""
import numpy as np
N = 8; J, h, beta = 1.0, 1.5, 1.0
gamma0, nuc = 1.0, 10.0
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
St = V.conj().T@op_at(sz,j0)@V
W = w[:,None]-w[None,:]
def kms(A,B): return np.einsum('i,ij,j,ij->', sqp, A.conj(), sqp, B)
def grate(om):
    if abs(om) < 1e-8: return gamma0/beta
    Jn = gamma0*abs(om)*np.exp(-abs(om)/nuc)
    nb = 1.0/(np.exp(beta*abs(om))-1.0)
    return Jn*(1+nb) if om>0 else Jn*nb
keys = np.unique(np.round(W/1e-9).astype(np.int64))
blocks = []
for key in keys:
    om = key*1e-9
    A = np.where(np.abs(W-om) < 5e-10, St, 0)
    if np.abs(A).max() > 1e-12: blocks.append((om, grate(om), A))

rng = np.random.default_rng(2)
print("=== A: corrected per-block identity ===")
worst = 0.0
for om,g,L in blocks:
    if abs(om) < 0.05 or om < 0: continue
    for t in range(2):
        X = rng.normal(size=(D,D))+1j*rng.normal(size=(D,D)); O=(X+X.conj().T)/2
        Ld = L.conj().T
        Ed = -np.real(kms(O, g*(Ld@O@L - 0.5*(Ld@L@O + O@Ld@L))))
        rhs = g*np.exp(beta*om/2)*np.real(0.5*kms(L@O,L@O) + 0.5*kms(O@Ld,O@Ld) - kms(L@O,O@L))
        rel = abs(Ed-rhs)/max(abs(Ed),1e-300)
        worst = max(worst, rel)
print(f"worst relative deviation (positive-freq blocks, random O): {worst:.2e}")

print("\n=== B: exact tail reduction (KMS pairs) ===")
# Pauli-string projection: E_{<eps}[X] = component supported within distance < eps of j0
def tail(Xsite, eps):
    # Xsite in SITE basis; project onto strings touching the far region O_eps
    # via partial-trace projection onto near region (complement of far)
    far = [j for j in range(N) if abs(j-j0) >= eps]
    near = [j for j in range(N) if abs(j-j0) < eps]
    # reorder tensor factors: build permutation mapping
    perm = near + far
    dims = [2]*N
    Xr = Xsite.reshape(dims+dims)
    Xr = np.transpose(Xr, perm + [N+q for q in perm])
    dn, df = 2**len(near), 2**len(far)
    Xm = Xr.reshape(dn,df,dn,df)
    Xnear = np.einsum('abcb->ac', Xm)/df
    Enear = np.kron(Xnear, np.eye(df))
    T = Xm.reshape(dn*df,dn*df) - Enear
    # rotate back
    inv = np.argsort(perm)
    Tr4 = T.reshape([2]*N+[2]*N)
    Tr4 = np.transpose(Tr4, list(inv)+[N+q for q in inv])
    return Tr4.reshape(D,D)
eps = 3
far_sites = [j for j in range(N) if abs(j-j0) >= eps]
Ofar = op_at(sx, far_sites[0]) @ op_at(sz, far_sites[-1])   # supported in far region
Ofar_t = V.conj().T@Ofar@V
Ssite = op_at(sz, j0)
def dirichlet(Ofull_t, Sfull_site):
    Sfull_t = V.conj().T@Sfull_site@V
    E = 0.0
    for key in keys:
        om = key*1e-9
        A = np.where(np.abs(W-om) < 5e-10, Sfull_t, 0)
        if np.abs(A).max() < 1e-12: continue
        g = grate(om)
        if g < 1e-14: continue
        Ad = A.conj().T
        E += -np.real(kms(Ofull_t, g*(Ad@Ofull_t@A - 0.5*(Ad@A@Ofull_t + Ofull_t@Ad@A))))
    return E
# NOTE: tails must be taken of the Bohr components S(w), which are NOT local;
# the exact statement uses per-block tails. Test: replace each S(w) by tail(S(w)).
def dirichlet_tails(Ofull_t):
    E = 0.0
    for key in keys:
        om = key*1e-9
        A = np.where(np.abs(W-om) < 5e-10, St, 0)
        if np.abs(A).max() < 1e-12: continue
        g = grate(om)
        if g < 1e-14: continue
        A_site = V@A@V.conj().T
        T_site = tail(A_site, eps)
        Tt = V.conj().T@T_site@V
        Td = Tt.conj().T
        E += -np.real(kms(Ofull_t, g*(Td@Ofull_t@Tt - 0.5*(Td@Tt@Ofull_t + Ofull_t@Td@Tt))))
    return E
E_full = dirichlet(Ofar_t, Ssite)
E_tail = dirichlet_tails(Ofar_t)
print(f"eps={eps}: E(O; S) = {E_full:.10e}   E(O; tail S) = {E_tail:.10e}   "
      f"rel.diff = {abs(E_full-E_tail)/max(abs(E_full),1e-300):.2e}")
