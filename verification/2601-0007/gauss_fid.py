"""Pure-numpy Gaussian fidelity (Banchi-Braunstein-Pirandola, PRL 115, 260501).
Convention: hbar=1, covariance Gamma with Gamma + (i/2) Omega >= 0 (vacuum = I/2).
Returns squared-fidelity F = ||sqrt(rho) sqrt(sigma)||_1^2 (Uhlmann, squared convention).
Zero-mean states. Ordering (x1..xN, p1..pN), Omega = [[0,I],[-I,0]].
"""
import numpy as np
from scipy import linalg as la

def omega(n):
    I = np.eye(n); Z = np.zeros((n, n))
    return np.block([[Z, I], [-I, Z]])

def fidelity_gauss(G1, G2, hbar=1.0):
    # BBP formula operates directly in hbar=1 units (vacuum = I/2) and
    # returns ROOT fidelity; we square at the end (paper convention).
    V1 = np.asarray(G1) / hbar
    V2 = np.asarray(G2) / hbar
    n = V1.shape[0] // 2
    Om = omega(n)
    Vsum_inv = np.linalg.inv(V1 + V2)
    Vaux = Om.T @ Vsum_inv @ (0.25 * Om + V2 @ Om @ V1)
    M = Vaux @ Om
    W = (la.sqrtm(np.eye(2 * n) + 0.25 * np.linalg.inv(M @ M)) + np.eye(2 * n)) @ Vaux
    Ftot4 = np.linalg.det(2.0 * W)
    F_root = np.real(Ftot4) ** 0.25 / np.real(np.linalg.det(V1 + V2)) ** 0.25
    return float(np.clip(F_root, 0, 1)) ** 2   # squared convention

if __name__ == "__main__":
    # anchors (squared convention)
    I2 = np.eye(2)
    vac = 0.5 * I2
    print("vac|vac:", fidelity_gauss(vac, vac), "(want 1)")
    for r in (0.3, 0.8, 1.5):
        sq = 0.5 * np.diag([np.exp(2*r), np.exp(-2*r)])
        got = fidelity_gauss(vac, sq)
        print(f"vac|sq(r={r}): {got:.10f}  want {1/np.cosh(r):.10f}")
    # thermal x thermal, 2 modes, multiplicativity
    t1 = 0.5 * 1.7 * np.eye(4); t2 = 0.5 * 2.4 * np.eye(4)
    f2 = fidelity_gauss(t1, t2)
    f1 = fidelity_gauss(0.5*1.7*I2, 0.5*2.4*I2)
    print(f"multiplicativity 2 modos: {f2:.10f} vs {f1**2:.10f}")
    # symplectic invariance: random symplectic S: S = exp(Omega A), A symmetric
    rng = np.random.default_rng(1)
    n = 2; Om = omega(n)
    A = rng.normal(0, .2, (2*n, 2*n)); A = A + A.T
    S = la.expm(Om @ A)
    G1 = 0.5*np.eye(2*n) + 0.1*np.eye(2*n)
    B = rng.normal(0, .1, (2*n, 2*n)); G2p = G1 + 0.05*(B+B.T) + 0.2*np.eye(2*n)
    fa = fidelity_gauss(G1, G2p); fb = fidelity_gauss(S@G1@S.T, S@G2p@S.T)
    print(f"invariancia simplectica: {fa:.12f} vs {fb:.12f}")
