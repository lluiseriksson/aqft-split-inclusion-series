"""Repaired Theorem 4.7 over the reattachment reconstruction, exact Gaussian
fidelity (Banchi PRL 2015) at covariance level -- no Fock truncation."""
import numpy as np
from gauss_fidelity import gauss_fid

Om2 = np.array([[0,1],[-1,0]], float)
Om4 = np.block([[Om2, np.zeros((2,2))],[np.zeros((2,2)), Om2]])
Z = np.diag([1.0,-1.0])
def S_tms(s):
    c, h = np.cosh(s), np.sinh(s)
    return np.block([[c*np.eye(2), h*Z],[h*Z, c*np.eye(2)]])
def S_sq1(l):
    return np.diag([np.exp(-l), np.exp(l), 1.0, 1.0])
def opn(M): return np.linalg.norm(M,2)
def hs(M): return np.linalg.norm(M,'fro')
def isqrt(M):
    w,V = np.linalg.eigh(M); return (V/np.sqrt(w))@V.T
def blocks(G): return G[:2,:2], G[:2,2:], G[2:,2:]
def symp_min(G): return np.sort(np.abs(np.linalg.eigvals(1j*Om4@G)))[0]

s_tms, lam = 0.35, 0.25
print("nb   kappa   Ckap    1-F(Banchi)   RHS(simpl)   RHS(full)   ratio   adm_min")
for nb in [0.002, 0.02, 0.1, 0.4, 1.0, 3.0]:
    G0 = S_tms(s_tms) @ ((1+2*nb)*np.eye(4)) @ S_tms(s_tms).T
    Gw = S_sq1(lam) @ G0 @ S_sq1(lam).T
    A0,X0,B0 = blocks(G0); A,X,B = blocks(Gw)
    W = np.linalg.solve(A0,X0); C0 = B0 - X0.T@np.linalg.solve(A0,X0)
    Gt = np.block([[A, A@W],[(A@W).T, C0 + W.T@A@W]])
    adm = np.min(np.linalg.eigvalsh(Gt + 1j*Om4))
    loss = 1 - gauss_fid(Gw, Gt, Om4)
    A0h, B0h = isqrt(A0), isqrt(B0)
    eta = opn(A0h@X0@B0h); eps = 1 - opn(A0h@(A-A0)@A0h)
    delta = opn(A0h@(X-X0)@B0h)
    c1, c2 = np.linalg.eigvalsh(A0)[0], np.linalg.eigvalsh(B0)[0]
    D12 = X - A@np.linalg.solve(A0,X0)
    DG = Gw - Gt
    kap = symp_min(Gw) - 1
    Ck = (1+kap)**2/((1+kap)**2-1)
    den = eps**2*(1-(eta+delta)**2/eps)**2*min(c1,c2)**2
    rhs_s = (Ck*(6+Ck)/16)*hs(D12)**2/den
    gi = opn(np.linalg.inv(Gw))
    rhs_f = (Ck/8)*(gi*hs(DG))**2 + (Ck**2/48)*(gi*hs(DG))**4
    rhs = min(rhs_s, rhs_f)
    print(f"{nb:5.3f} {kap:6.3f} {Ck:7.2f} {loss:12.4e} {rhs_s:11.4e} {rhs_f:11.4e} "
          f"{loss/rhs:7.4f} {adm:8.4f}  {'PASS' if loss<=rhs else 'FAIL'}")
