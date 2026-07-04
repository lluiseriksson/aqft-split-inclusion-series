"""
Modelo A: sonda fermionica (energia w0) acoplada por g al extremo de una cadena
de Kitaev (t=Dp=1, mu=2h) de longitud eps, con perdida Markoviana L=sqrt(g_loss)*c
en el ultimo sitio. Todo cuadratico => exacto via ODE de covarianzas (Lyapunov).

kappa(eps) = tasa de decaimiento del modo sonda = -2*Re(lambda) del autovalor de
X = H_maj - 2 Re(M) con mayor peso en los Majoranas de la sonda.

Prediccion analitica sub-gap:  pendiente = 2*q(w0),  cosh q = (4+mu^2-w0^2)/(4 mu).
"""
import numpy as np
from numpy.linalg import eig, eigvals
import json

# ---------------- quadratic (A,B) -> Majorana matrix ----------------
def majorana_matrix(A, B):
    """H_op = sum A_pq c^d_p c_q + 1/2 sum (B_pq c^d_p c^d_q + h.c.)
       -> H_maj real antisym with H_op = (i/4) a^T H_maj a + Tr(A)/2."""
    n = A.shape[0]
    T = np.zeros((2 * n, 2 * n), complex)
    for p in range(n):
        for q in range(n):
            Ap = A[p, q]
            if Ap != 0:
                T[2*p, 2*q]     += Ap / 4
                T[2*p+1, 2*q+1] += Ap / 4
                T[2*p, 2*q+1]   += 1j * Ap / 4
                T[2*p+1, 2*q]   += -1j * Ap / 4
            Bp = B[p, q]
            if Bp != 0:
                T[2*p, 2*q]     += Bp / 8
                T[2*p, 2*q+1]   += -1j * Bp / 8
                T[2*p+1, 2*q]   += -1j * Bp / 8
                T[2*p+1, 2*q+1] += -Bp / 8
                Bc = np.conj(Bp)
                T[2*q, 2*p]     += Bc / 8
                T[2*q, 2*p+1]   += 1j * Bc / 8
                T[2*q+1, 2*p]   += 1j * Bc / 8
                T[2*q+1, 2*p+1] += -Bc / 8
    Hm = -2j * (T - T.T)
    assert np.max(np.abs(Hm.imag)) < 1e-10, "H_maj no es real"
    Hm = Hm.real
    assert np.max(np.abs(Hm + Hm.T)) < 1e-10
    return Hm

def X_Y_from(Hm, wlist):
    """L_mu = sum_j w_j a_j. M_jk = sum_mu conj(w_j) w_k.
       dGamma/dt = X Gamma + Gamma X^T + Y,  X = Hm - 2 Re M, Y = -4 Im M."""
    n2 = Hm.shape[0]
    M = np.zeros((n2, n2), complex)
    for w in wlist:
        M += np.outer(np.conj(w), w)
    return Hm - 2 * M.real, -4 * M.imag

# ---------------- builders ----------------
def build_AB(eps, w0, g, mu, t=1.0, Dp=1.0):
    """mode 0 = probe; modes 1..eps = Kitaev chain."""
    n = 1 + eps
    A = np.zeros((n, n)); B = np.zeros((n, n))
    A[0, 0] = w0
    if eps >= 1:
        A[0, 1] = A[1, 0] = g
    for i in range(1, eps + 1):
        A[i, i] = -mu
    for i in range(1, eps):
        A[i, i+1] = A[i+1, i] = -t
        B[i+1, i] = Dp; B[i, i+1] = -Dp
    return A, B

def loss_vector(n, site, gamma):
    """L = sqrt(gamma) c_site -> Majorana coefficients."""
    w = np.zeros(2 * n, complex)
    w[2*site] = np.sqrt(gamma) / 2
    w[2*site + 1] = 1j * np.sqrt(gamma) / 2
    return w

def kappa_of(eps, w0, g, mu, gamma):
    A, B = build_AB(eps, w0, g, mu)
    Hm = majorana_matrix(A, B)
    w = loss_vector(1 + eps, eps, gamma)   # loss on last chain site
    X, Y = X_Y_from(Hm, [w])
    vals, vecs = eig(X)
    wt = np.abs(vecs[0, :])**2 + np.abs(vecs[1, :])**2
    wt = wt / np.sum(np.abs(vecs)**2, axis=0)
    order = np.argsort(-wt)
    lam = vals[order[:2]]                  # conjugate pair of the probe mode
    omega_b = float(np.max(np.abs(lam.imag)))   # renormalized bound-state frequency
    return -2 * np.max(lam.real), wt[order[0]], omega_b

# ---------------- validation vs brute-force Lindblad ----------------
def fock_c_ops(n):
    sm = np.array([[0, 1], [0, 0]], complex)
    sz = np.diag([1., -1.]).astype(complex)
    I2 = np.eye(2, dtype=complex)
    cs = []
    for j in range(n):
        m = np.array([[1.+0j]])
        for k in range(n):
            m = np.kron(m, sz if k < j else (sm if k == j else I2))
        cs.append(m)
    return cs

def validate():
    # (1) operator identity for random (A,B), n=3
    rng = np.random.default_rng(7)
    n = 3
    Ar = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); Ar = (Ar + Ar.conj().T) / 2
    Br = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); Br = (Br - Br.T) / 2
    cs = fock_c_ops(n)
    Hf = np.zeros((2**n, 2**n), complex)
    for p in range(n):
        for q in range(n):
            Hf += Ar[p, q] * cs[p].conj().T @ cs[q]
            Hf += 0.5 * Br[p, q] * cs[p].conj().T @ cs[q].conj().T
            Hf += 0.5 * np.conj(Br[p, q]) * cs[q] @ cs[p]
    a = []
    for c in cs:
        a += [c + c.conj().T, -1j * (c - c.conj().T)]
    Hm = majorana_matrix(Ar, Br)
    Hmf = np.zeros_like(Hf)
    for j in range(2*n):
        for k in range(2*n):
            if Hm[j, k] != 0:
                Hmf += (1j/4) * Hm[j, k] * (a[j] @ a[k])
    err1 = np.max(np.abs(Hf - Hmf - (np.trace(Ar)/2) * np.eye(2**n)))

    # (2) dynamics: probe + 2 Kitaev sites, loss on site 2 -- exact rho vs covariance
    eps, w0, g, mu, gamma = 2, 0.5, 0.3, 3.0, 0.4
    A, B = build_AB(eps, w0, g, mu)
    Hf = np.zeros((8, 8), complex)
    cs = fock_c_ops(3)
    for p in range(3):
        for q in range(3):
            Hf += A[p, q] * cs[p].conj().T @ cs[q]
            Hf += 0.5 * B[p, q] * cs[p].conj().T @ cs[q].conj().T
            Hf += 0.5 * B[p, q].conjugate() * cs[q] @ cs[p]
    L = np.sqrt(gamma) * cs[2]
    Ld, LdL = L.conj().T, L.conj().T @ L
    psi0 = np.zeros(8, complex)
    psi0[np.argmax(np.abs((cs[0].conj().T)[:, 0]))] = 1.0  # probe occupied, chain empty
    # safer: build |100> explicitly: state index with mode0 occupied
    psi0 = (cs[0].conj().T) @ np.eye(8)[:, 0] if False else psi0
    vac = np.zeros(8, complex); vac[0] = 1.0
    psi0 = cs[0].conj().T @ vac
    rho = np.outer(psi0, psi0.conj())
    def rhs(r):
        return -1j * (Hf @ r - r @ Hf) + L @ r @ Ld - 0.5 * (LdL @ r + r @ LdL)
    dt, T = 2e-3, 6.0
    ts = np.arange(0, T + dt/2, dt)
    n_ex = []
    npop = cs[0].conj().T @ cs[0]
    for _ in ts:
        n_ex.append(np.real(np.trace(npop @ rho)))
        k1 = rhs(rho); k2 = rhs(rho + dt/2*k1); k3 = rhs(rho + dt/2*k2); k4 = rhs(rho + dt*k3)
        rho = rho + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    # covariance
    Hm = majorana_matrix(A, B)
    X, Y = X_Y_from(Hm, [loss_vector(3, 2, gamma)])
    G = np.zeros((6, 6))
    for p in range(3):
        s = +1.0 if p == 0 else -1.0   # occupied: +1 ; empty: -1
        G[2*p, 2*p+1] = s; G[2*p+1, 2*p] = -s
    n_cov = []
    for _ in ts:
        n_cov.append(0.5 * (1 + G[0, 1]))
        k1 = X@G + G@X.T + Y
        G2 = G + dt/2*k1; k2 = X@G2 + G2@X.T + Y
        G3 = G + dt/2*k2; k3 = X@G3 + G3@X.T + Y
        G4 = G + dt*k3;  k4 = X@G4 + G4@X.T + Y
        G = G + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    err2 = np.max(np.abs(np.array(n_ex) - np.array(n_cov)))
    print(f"[validacion] error identidad operador: {err1:.2e} | error dinamica n_probe: {err2:.2e}")
    assert err1 < 1e-9 and err2 < 1e-7

# ---------------- production ----------------
def slope_analytic(w0, mu):
    c = (4 + mu**2 - w0**2) / (4 * mu)
    if c <= 1: return 0.0
    return 2 * np.arccosh(c)

def sweep(w0, mu, g=0.3, gamma=0.4, eps_list=range(2, 27, 2)):
    out = []
    for eps in eps_list:
        k, wt, wb = kappa_of(eps, w0, g, mu, gamma)
        out.append((eps, k, wt, wb))
    return np.array(out)

if __name__ == "__main__":
    validate()
    results = {}
    cases = [
        ("gap_w0.0", 0.0, 3.0), ("gap_w0.5", 0.5, 3.0), ("gap_w0.9", 0.9, 3.0),
        ("gap_w2.0_inband", 2.0, 3.0), ("crit_w0.5", 0.5, 2.0),
    ]
    for name, w0, mu in cases:
        arr = sweep(w0, mu)
        eps, kap = arr[:, 0], arr[:, 1]
        omega_b = float(np.median(arr[:, 3]))   # renormalized frequency (Table 1)
        mask = (kap > 1e-12) & (kap < 1e-2) & (eps >= 6)
        if mask.sum() >= 3:
            sl = np.polyfit(eps[mask], np.log(kap[mask]), 1)[0]
        else:
            sl = np.nan
        results[name] = dict(w0=w0, mu=mu, omega_b=omega_b, eps=eps.tolist(),
                             kappa=kap.tolist(),
                             slope_fit=float(-sl) if np.isfinite(sl) else None,
                             slope_analytic_wb=float(slope_analytic(omega_b, mu)),
                             slope_analytic_w0=float(slope_analytic(w0, mu)))
        print(f"{name:18s} w_b = {omega_b:6.4f} | pendiente ajustada = "
              f"{(-sl if np.isfinite(sl) else float('nan')):7.4f}"
              f" | 2q(w_b) = {slope_analytic(omega_b, mu):7.4f}"
              f" | 2q(w0) bare = {slope_analytic(w0, mu):7.4f}"
              f" | kappa(eps=2..8) = {kap[0]:.3e} {kap[1]:.3e} {kap[2]:.3e} {kap[3]:.3e}")
    np.save("modelA_results.npy", results, allow_pickle=True)
    print("OK modelA")
