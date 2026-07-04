"""Covariance-level verification (exact Gaussian fidelity, no Fock truncation):
  A. Repaired Theorem 4.7 sweep (reattachment + C_kappa + admissibility).
  B. Gap-corrected fidelity Lemma on random gapped covariance pairs.
  C. Admissibility / steering threshold sweep.
  D. Lattice Klein-Gordon eta_vac decay rate vs arccosh(1+m^2/2).
"""
import numpy as np
from gauss_fidelity import gauss_fid

Om2 = np.array([[0,1],[-1,0]], float)
Om4 = np.block([[Om2, np.zeros((2,2))],[np.zeros((2,2)), Om2]])
Z = np.diag([1.0,-1.0])
def S_tms(s):
    c, h = np.cosh(s), np.sinh(s)
    return np.block([[c*np.eye(2), h*Z],[h*Z, c*np.eye(2)]])
def opn(M): return np.linalg.norm(M,2)
def hs(M): return np.linalg.norm(M,'fro')
def isqrt(M):
    w,V = np.linalg.eigh(M); return (V/np.sqrt(w))@V.T
def blocks(G): return G[:2,:2], G[:2,2:], G[2:,2:]
def symp_min(G, Om): return np.sort(np.abs(np.linalg.eigvals(1j*Om@G)))[0]
from scipy.linalg import expm

print("=== A: repaired Theorem 4.7 (see also part2_banchi.py) ===")
lam = 0.25
S_sq = np.diag([np.exp(-lam), np.exp(lam), 1.0, 1.0])
for nb in [0.002, 0.02, 0.1, 0.4, 1.0, 3.0]:
    G0 = S_tms(0.35) @ ((1+2*nb)*np.eye(4)) @ S_tms(0.35).T
    Gw = S_sq @ G0 @ S_sq.T
    A0,X0,B0 = blocks(G0); A,X,B = blocks(Gw)
    W = np.linalg.solve(A0,X0); C0 = B0 - X0.T@np.linalg.solve(A0,X0)
    Gt = np.block([[A, A@W],[(A@W).T, C0 + W.T@A@W]])
    adm = np.min(np.linalg.eigvalsh(Gt + 1j*Om4))
    if adm < -1e-9:
        print(f"nb={nb:5.3f}: hypothesis (g) FAILS (adm={adm:+.4f}) -- excluded (steering)")
        continue
    loss = 1 - gauss_fid(Gw, Gt, Om4)
    A0h, B0h = isqrt(A0), isqrt(B0)
    eta = opn(A0h@X0@B0h); eps = 1 - opn(A0h@(A-A0)@A0h)
    delta = opn(A0h@(X-X0)@B0h)
    c1, c2 = np.linalg.eigvalsh(A0)[0], np.linalg.eigvalsh(B0)[0]
    D12 = X - A@W
    kap = symp_min(Gw, Om4) - 1
    Ck = (1+kap)**2/((1+kap)**2-1)
    den = eps**2*(1-(eta+delta)**2/eps)**2*min(c1,c2)**2
    rhs = (Ck*(6+Ck)/16)*hs(D12)**2/den
    print(f"nb={nb:5.3f}: adm={adm:+.4f}  1-F={loss:.4e}  bound={rhs:.4e}  "
          f"{'PASS' if loss <= rhs else 'FAIL'} (margin {rhs/max(loss,1e-300):.0f}x)")

print("\n=== B: gap-corrected fidelity lemma, random gapped pairs ===")
rng = np.random.default_rng(11)
ntot = nfail = 0; worst = 0.0
for _ in range(2000):
    h1 = rng.normal(scale=0.25, size=(4,4)); h1 = h1 + h1.T
    S1 = expm(Om4 @ h1)
    nus = 1 + rng.uniform(0.1, 1.0, 2)
    G1 = S1 @ np.diag(np.repeat(nus, 2)) @ S1.T
    h2 = h1 + rng.normal(scale=0.06, size=(4,4)); h2 = (h2 + h2.T)/2
    S2 = expm(Om4 @ h2)
    nus2 = nus * (1 + rng.uniform(-0.1, 0.1, 2))
    nus2 = np.maximum(nus2, 1.0)
    G2 = S2 @ np.diag(np.repeat(nus2, 2)) @ S2.T
    G1h = isqrt(G1)
    K = G1h @ (G1 - G2) @ G1h
    if opn(K) > 0.5: continue
    kap = symp_min(G1, Om4) - 1
    if kap < 1e-3: continue
    Ck = (1+kap)**2/((1+kap)**2-1)
    F = gauss_fid(G1, G2, Om4)
    bound = (Ck/8)*hs(K)**2 + (Ck**2/48)*hs(K)**4
    ntot += 1
    r = (1-F)/bound if bound > 0 else 0
    worst = max(worst, r); nfail += (1-F) > bound + 1e-12
print(f"samples={ntot}  violations={nfail}  max ratio (1-F)/bound = {worst:.3f}")

print("\n=== C: admissibility / steering threshold ===")
print("s_tms | nb   | min eig(C0 + i*Om2) | min eig(Gt + i*Om4)")
for s in [0.35, 0.8, 1.2]:
    for nb in [0.0, 0.1, 0.4]:
        G0 = S_tms(s) @ ((1+2*nb)*np.eye(4)) @ S_tms(s).T
        Gw = S_sq @ G0 @ S_sq.T
        A0,X0,B0 = blocks(G0); A = blocks(Gw)[0]
        W = np.linalg.solve(A0,X0); C0 = B0 - X0.T@np.linalg.solve(A0,X0)
        Gt = np.block([[A, A@W],[(A@W).T, C0 + W.T@A@W]])
        m1 = np.min(np.linalg.eigvalsh(C0 + 1j*Om2))
        m2 = np.min(np.linalg.eigvalsh(Gt + 1j*Om4))
        print(f"{s:5.2f} | {nb:4.1f} | {m1:+19.4f} | {m2:+19.4f}")

print("\n=== D: lattice Klein-Gordon eta_vac decay (m=1) ===")
N, m = 240, 1.0
M = (m**2 + 2)*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
w, V = np.linalg.eigh(M)
Mm = (V/np.sqrt(w)) @ V.T; Mp = (V*np.sqrt(w)) @ V.T
i1 = np.arange(0, 60)
rs = np.arange(2, 40, 2); etas = []
for r in rs:
    i2 = np.arange(60 + r, N)
    A0 = np.block([[Mm[np.ix_(i1,i1)], np.zeros((60,60))],[np.zeros((60,60)), Mp[np.ix_(i1,i1)]]])
    n2 = len(i2)
    B0 = np.block([[Mm[np.ix_(i2,i2)], np.zeros((n2,n2))],[np.zeros((n2,n2)), Mp[np.ix_(i2,i2)]]])
    X0 = np.block([[Mm[np.ix_(i1,i2)], np.zeros((60,n2))],[np.zeros((60,n2)), Mp[np.ix_(i1,i2)]]])
    etas.append(opn(isqrt(A0) @ X0 @ isqrt(B0)))
etas = np.array(etas)
ok = etas > 1e-12                      # exclude double-precision floor
sl_plain = -np.polyfit(rs[ok], np.log(etas[ok]), 1)[0]
sl_pref = -np.polyfit(rs[ok], np.log(etas[ok]) + 0.5*np.log(rs[ok]), 1)[0]
print(f"plain fit = {sl_plain:.4f} | prefactor-corrected (r^-1/2 e^-mu r) = {sl_pref:.4f} "
      f"| arccosh(1+m^2/2) = {np.arccosh(1+m**2/2):.4f}")
