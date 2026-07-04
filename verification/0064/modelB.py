"""
Modelo B: protocolo del preprint, exacto (sin trayectorias).
TFIM  H = -J sum sx sx - h sum sz, J=1. Sitio S=0. Disipador sigma^- en eps+1,
gamma>0. Estado inicial |up...up> (z), maxima coherencia-x en cada sitio.
C_x(t) = coherencia de entropia relativa de rho_S en la base sigma^x.
kappa_int(eps) = [ln C(t0) - ln C(t0+tau)]/tau, t0 = 0.8*eps/v, v=2J, tau=2.
Controles: gamma=0 (mismas ventanas), ventana fija absoluta, h=J critico.
N=10 denso exacto (1024x1024); gamma=0 tambien a N=12 (statevector).
"""
import numpy as np
import scipy.sparse as sp
import time, json

J = 1.0
ln2 = np.log(2.0)

def build_H(N, h):
    dim = 1 << N
    diag = np.zeros(dim)
    for j in range(N):
        bit = (np.arange(dim) >> j) & 1
        diag += -h * (1 - 2 * bit)          # sz = +1 en bit 0 (up)
    rows, cols, vals = [np.arange(dim)], [np.arange(dim)], [diag]
    for j in range(N - 1):
        mask = (1 << j) | (1 << (j + 1))
        r = np.arange(dim)
        rows.append(r); cols.append(r ^ mask); vals.append(np.full(dim, -J))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    return H

def dissipator_indices(N, site):
    dim = 1 << N
    idx0 = np.array([i for i in range(dim) if not (i >> site) & 1])  # bit=0 (up)
    return idx0, idx0 + (1 << site)

def CX_from_rhoS(rS):
    Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    T = Hd @ rS @ Hd
    p = np.clip(np.real(np.diag(T)), 1e-300, 1)
    w = np.clip(np.linalg.eigvalsh(rS), 1e-300, 1)
    return float(-(p * np.log(p)).sum() + (w * np.log(w)).sum())

def rhoS_from_rho(rho, N):
    r = rho.reshape(1 << (N - 1), 2, 1 << (N - 1), 2)   # i = 2*rest + b0
    return np.einsum('rbrc->bc', r)

def rhoS_from_psi(psi, N):
    A = psi.reshape(1 << (N - 1), 2)                    # i = 2*rest + b0
    return A.T @ A.conj()

def run_lindblad(N, h, eps, gamma, tmax, dt=0.02, dt_out=0.04):
    dim = 1 << N
    H = build_H(N, h)
    site = eps + 1
    idx0, idx1 = dissipator_indices(N, site)
    Pmask = np.zeros(dim); Pmask[idx0] = 1.0            # P_up = s+ s-
    psi0 = np.zeros(dim, complex); psi0[0] = 1.0        # todos up
    rho = np.outer(psi0, psi0.conj())
    def rhs(r):
        Hr = H @ r
        out = -1j * (Hr - Hr.conj().T)                  # H real sim: rho H = (H rho^H)^H
        if gamma > 0:
            jump = np.zeros_like(r)
            jump[np.ix_(idx1, idx1)] = r[np.ix_(idx0, idx0)]
            out += gamma * (jump - 0.5 * (Pmask[:, None] * r + r * Pmask[None, :]))
        return out
    ts, Cs = [], []
    steps = int(round(tmax / dt)); every = max(1, int(round(dt_out / dt)))
    for s in range(steps + 1):
        if s % every == 0:
            ts.append(s * dt); Cs.append(CX_from_rhoS(rhoS_from_rho(rho, N)))
        if s == steps: break
        k1 = rhs(rho); k2 = rhs(rho + dt/2*k1); k3 = rhs(rho + dt/2*k2); k4 = rhs(rho + dt*k3)
        rho = rho + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return np.array(ts), np.array(Cs)

def run_unitary(N, h, tmax, dt=0.005, dt_out=0.04):
    dim = 1 << N
    H = build_H(N, h)
    psi = np.zeros(dim, complex); psi[0] = 1.0
    ts, Cs = [], []
    steps = int(round(tmax / dt)); every = max(1, int(round(dt_out / dt)))
    for s in range(steps + 1):
        if s % every == 0:
            ts.append(s * dt); Cs.append(CX_from_rhoS(rhoS_from_psi(psi, N)))
        if s == steps: break
        k1 = -1j*(H@psi); k2 = -1j*(H@(psi+dt/2*k1)); k3 = -1j*(H@(psi+dt/2*k2)); k4 = -1j*(H@(psi+dt*k3))
        psi = psi + dt/6*(k1+2*k2+2*k3+k4)
        psi /= np.linalg.norm(psi)
    return np.array(ts), np.array(Cs)

def kappa_int(ts, Cs, t0, tau):
    C0 = np.interp(t0, ts, Cs); C1 = np.interp(t0 + tau, ts, Cs)
    return (np.log(max(C0, 1e-300)) - np.log(max(C1, 1e-300))) / tau

if __name__ == "__main__":
    v, tau = 2.0, 2.0
    eps_list = [1, 2, 3, 4, 5, 6]
    tmax = 0.8 * max(eps_list) / v + tau + 0.9   # 5.3 -> cubre tambien ventana fija [3.0,5.0]
    tmax = max(tmax, 5.2)
    gamma = 0.8
    res = {"eps": eps_list, "gamma": gamma, "tau": tau}
    for h in (1.5, 1.0):
        t0u, Cu = run_unitary(10, h, tmax)
        res[f"h{h}_g0_curve"] = [t0u.tolist(), Cu.tolist()]
        res[f"h{h}_g0_kint"] = [kappa_int(t0u, Cu, 0.8*e/v, tau) for e in eps_list]
        res[f"h{h}_g0_kfix"] = kappa_int(t0u, Cu, 3.0, tau)
        kg, kf = [], []
        for e in eps_list:
            t1 = time.time()
            ts, Cs = run_lindblad(10, h, e, gamma, tmax)
            kg.append(kappa_int(ts, Cs, 0.8*e/v, tau))
            kf.append(kappa_int(ts, Cs, 3.0, tau))
            print(f"h={h} eps={e}: kint={kg[-1]:+.4f} (g0 {res[f'h{h}_g0_kint'][eps_list.index(e)]:+.4f}) "
                  f"| fija {kf[-1]:+.4f} (g0 {res[f'h{h}_g0_kfix']:+.4f}) | {time.time()-t1:.0f}s", flush=True)
        res[f"h{h}_g_kint"] = kg
        res[f"h{h}_g_kfix"] = kf
    # control gamma=0 a N=12 (tamano exacto del paper)
    t12, C12 = run_unitary(12, 1.5, tmax)
    res["h1.5_g0_N12_kint"] = [kappa_int(t12, C12, 0.8*e/v, tau) for e in eps_list]
    res["h1.5_g0_N12_curve"] = [t12.tolist(), C12.tolist()]
    with open("./modelB_results.json", "w") as f:
        json.dump(res, f)
    print("OK modelB")
