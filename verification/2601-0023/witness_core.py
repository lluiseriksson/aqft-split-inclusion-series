import numpy as np, itertools, sys, time
from scipy import linalg as la
I2 = np.eye(2); Xp = np.array([[0,1],[1,0]],dtype=complex)
Yp = np.array([[0,-1j],[1j,0]]); Zp = np.diag([1.,-1.]).astype(complex)
P1 = {'X':Xp,'Y':Yp,'Z':Zp}
def site_op(op, i, N):
    M = np.array([[1.+0j]])
    for j in range(N): M = np.kron(M, op if j==i else I2)
    return M
def pauli_block(lab, k, N):
    O = np.array([[1.+0j]])
    for j in range(N):
        o = I2.astype(complex)
        if k <= j < k+len(lab) and lab[j-k] != 'I': o = P1[lab[j-k]]
        O = np.kron(O, o)
    return O
def run(h, Sname, N=10, J=1.0, gamma0=0.1, eps_list=(1,2,3), betas=(0.5,1.0,2.0)):
    t0=time.time()
    H = np.zeros((2**N,2**N), dtype=complex)
    for i in range(N-1): H += -J*site_op(Xp,i,N)@site_op(Xp,i+1,N)
    for i in range(N): H += -h*site_op(Zp,i,N)
    E,U = np.linalg.eigh(H)
    S = site_op(P1[Sname[0]], N//2, N)
    Se = U.conj().T @ S @ U
    S0e = Se*(np.abs(E[:,None]-E[None,:])<1e-9)
    j0=N//2; res={}
    for L in (1,2):
        for eps in eps_list:
            k=j0+eps
            if k+L>N: continue
            labs=[p for p in itertools.product('IXYZ',repeat=L) if p!=tuple('I')*L]
            Ae=[U.conj().T@(pauli_block(l,k,N)@U) for l in labs]
            Ce=[S0e@A-A@S0e for A in Ae]
            m=len(labs)
            for beta in betas:
                w=np.exp(-beta*(E-E.min())); w/=w.sum(); sq=np.sqrt(w); Wm=np.outer(sq,sq)
                G=np.zeros((m,m),dtype=complex); M=np.zeros((m,m),dtype=complex)
                for a in range(m):
                    for b in range(a,m):
                        G[a,b]=np.sum(Wm*np.conj(Ae[a])*Ae[b]); G[b,a]=np.conj(G[a,b])
                        M[a,b]=np.sum(Wm*np.conj(Ce[a])*Ce[b]); M[b,a]=np.conj(M[a,b])
                ev=la.eigh(M,G,eigvals_only=True)
                res[(L,eps,beta)]=gamma0/2*float(ev[-1].real)
    print(f"[h={h} S={Sname} t={time.time()-t0:.0f}s]")
    for L in (1,2):
        vals=[v for (l,e,b),v in res.items() if l==L]
        print(f"  L={L}: rango kappa_min = [{min(vals):.3e}, {max(vals):.3e}]  "
              f"(publicado: {'[3.2e-04, 1.2e-03]' if L==1 else '[8.2e-04, 2.9e-03]'})")
    return res
if __name__=="__main__":
    run(float(sys.argv[1]), sys.argv[2])
