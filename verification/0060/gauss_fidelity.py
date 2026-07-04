"""Exact Uhlmann fidelity between zero-mean Gaussian states from covariances.
Convention: Gamma_ij = Tr[rho {R_i,R_j}], vacuum Gamma = 1 (so V := Gamma/2,
V_vac = 1/2, [R,R] = i*Omega). Formula: Banchi-Braunstein-Pirandola, PRL 115,
260501 (2015), zero-mean case."""
import numpy as np
from scipy.linalg import sqrtm, inv, det

def gauss_fid(G1, G2, Om):
    V1, V2 = G1/2, G2/2
    n = G1.shape[0]//2
    S = V1 + V2
    Si = inv(S)
    Vaux = Om.T @ Si @ (Om/4 + V2 @ Om @ V1)
    M = Vaux @ Om
    W = np.eye(2*n) + inv(M @ M)/4          # 1 + (Vaux Om)^{-2}/4
    F4 = np.real(det(2*(sqrtm(W) + np.eye(2*n)) @ Vaux)) / np.real(det(S))
    return float(np.clip(F4, 0, None)**0.25)

if __name__ == "__main__":
    Om2 = np.array([[0,1],[-1,0]], float)
    Om4 = np.block([[Om2, np.zeros((2,2))],[np.zeros((2,2)), Om2]])
    # test 1: vacuum vs thermal, F = (1+nb)^{-1/2}
    for nb in [0.05, 0.1, 0.3, 1.0]:
        F = gauss_fid(np.eye(2), (1+2*nb)*np.eye(2), Om2)
        print(f"vac-thermal nb={nb}: {F:.8f} vs exact {(1+nb)**-0.5:.8f}")
    # test 2: vacuum vs squeezed vacuum, F = 1/sqrt(cosh s)
    for s in [0.3, 0.8]:
        G2 = np.diag([np.exp(2*s), np.exp(-2*s)])
        F = gauss_fid(np.eye(2), G2, Om2)
        print(f"vac-squeezed s={s}: {F:.8f} vs exact {np.cosh(s)**-0.5:.8f}")
    # test 3: two thermals (commuting), Bhattacharyya of geometric distributions
    n1, n2 = 0.2, 0.5
    Fex = 1/(np.sqrt((n1+1)*(n2+1)) - np.sqrt(n1*n2))
    F = gauss_fid((1+2*n1)*np.eye(2), (1+2*n2)*np.eye(2), Om2)
    print(f"thermal-thermal: {F:.8f} vs exact {Fex:.8f}")
