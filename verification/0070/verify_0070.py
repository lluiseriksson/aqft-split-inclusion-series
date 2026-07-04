"""Exact re-examination for 'Stress Testing the Rate Inheritance Principle'
(2512.0070). TFIM + Davies generator, exact diagonalization (N=6).
A. kappa_sup(eps): v1 envelope (largest gen. eigenvalue of Dirichlet form on
   operators at distance >= eps) for sigma^x vs sigma^z coupling.
B. kappa_floor(eps): smallest NONZERO gen. eigenvalue (kernel removed) -- the
   quantity a no-go needs is a floor, not a ceiling.
C. Davies-locality diagnostic: delocalization of the Bohr components S(omega),
   esp. omega ~ 0 (secular nonlocality at finite N).
"""
import numpy as np

N = 6
J, h, beta = 1.0, 1.5, 1.0
gamma0, nuc = 1.0, 10.0
dOm = 0.05
j0 = 0
sx = np.array([[0,1],[1,0]], complex); sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]], complex); I2 = np.eye(2)
def op_at(o,i):
    M = np.array([[1]],complex)
    for k in range(N): M = np.kron(M, o if k==i else I2)
    return M
HS = sum(-J*op_at(sx,i)@op_at(sx,i+1) for i in range(N-1)) + sum(-h*op_at(sz,i) for i in range(N))
w, V = np.linalg.eigh(HS); D = 2**N
gap = w[1]-w[0]
p = np.exp(-beta*w); p /= p.sum()
sqp = np.sqrt(p)
W = w[:,None]-w[None,:]
bins = np.round(W/dOm).astype(int)

def grate(om):
    if abs(om) < dOm/2: return gamma0/beta
    Jn = gamma0*abs(om)*np.exp(-abs(om)/nuc)
    nb = 1.0/(np.exp(beta*abs(om))-1.0)
    return Jn*(1+nb) if om>0 else Jn*nb

def components(S):
    St = V.conj().T@S@V
    out = []
    for key in np.unique(bins):
        A = np.where(bins==key, St, 0)
        if np.abs(A).max() > 1e-12:
            out.append((grate(key*dOm), A))
    return out    # all in ENERGY basis

def kms(A,B):   # A,B in energy basis
    return np.einsum('i,ij,j,ij->', sqp, A.conj(), sqp, B)

def Ldag(O, comps):   # energy basis
    out = 1j*(np.diag(w)@O - O@np.diag(w))
    for g,A in comps:
        if g < 1e-14: continue
        Ad = A.conj().T
        out += g*(Ad@O@A - 0.5*(Ad@A@O + O@Ad@A))
    return out

P = {'x':sx,'y':sy,'z':sz}
def basis(eps):
    ops = []
    for i in range(eps,N):
        for a in 'xyz': ops.append(op_at(P[a],i))
    for i in range(eps,N-1):
        for a in 'xyz':
            for b in 'xyz': ops.append(op_at(P[a],i)@op_at(P[b],i+1))
    return ops

def sizes(eps): return 3*(N-eps) + 9*max(N-1-eps,0)

for name,S in [("sigma_x (energy exchange)",op_at(sx,j0)),
               ("sigma_z (near-zero freq)", op_at(sz,j0))]:
    comps = components(S)
    full = [V.conj().T@O@V for O in basis(1)]
    LO = [Ldag(O,comps) for O in full]
    n = len(full)
    G = np.zeros((n,n),complex); M = np.zeros((n,n),complex)
    for a in range(n):
        for b in range(n):
            G[a,b] = kms(full[a],full[b]); M[a,b] = -kms(full[a],LO[b])
    G = (G+G.conj().T)/2; Ms = (M+M.conj().T)/2
    print(f"\n--- coupling {name}, N={N}, gap={gap:.3f} ---")
    print("eps  kappa_sup    kappa_floor")
    for eps in range(1,N):
        idx = []
        # map: basis(eps) ops are the tail of basis(1)? our ordering: singles for
        # i in [eps,N), then pairs. Rebuild index sets explicitly.
        # singles in full: i from 1..N-1 (eps=1 basis): position (i-1)*3 + a
        for i in range(eps,N):
            for a in range(3): idx.append((i-1)*3+a)
        off = 3*(N-1)
        for i in range(eps,N-1):
            for ab in range(9): idx.append(off + (i-1)*9 + ab)
        idx = np.array(idx)
        Gs, Msub = G[np.ix_(idx,idx)], Ms[np.ix_(idx,idx)]
        gw, gv = np.linalg.eigh(Gs)
        keep = gw > 1e-10*gw.max()
        W12 = gv[:,keep]/np.sqrt(gw[keep])
        ev = np.real(np.linalg.eigvalsh(W12.conj().T@Msub@W12))
        sup = ev.max()
        nz = ev[ev > max(1e-8*ev.max(), 1e-12)]
        print(f"{eps:3d}  {sup:11.4e}  {nz.min() if len(nz) else 0.0:11.4e}")

print("\n=== C: delocalization of Bohr components of sigma^z_1 ===")
St = V.conj().T@op_at(sz,j0)@V
def weight_beyond(A_site, d):
    dims = (2**d, 2**(N-d))
    Ar = A_site.reshape(dims[0],dims[1],dims[0],dims[1])
    Aloc = np.einsum('abcb->ac',Ar)/dims[1]
    Af = np.kron(Aloc,np.eye(dims[1]))
    return np.linalg.norm(A_site-Af)/max(np.linalg.norm(A_site),1e-300)
for key in [0, 1, int(round(2*(h-J)/dOm))]:
    mask = (bins==key) if key==0 else (np.abs(bins)==key)
    A = np.where(mask, St, 0)
    if np.abs(A).max() < 1e-12: continue
    A_site = V@A@V.conj().T
    row = " ".join(f"d={d}:{weight_beyond(A_site,d):.3f}" for d in [1,2,3,4])
    print(f"omega~{key*dOm:+.2f}: rel. weight beyond distance {row}")
