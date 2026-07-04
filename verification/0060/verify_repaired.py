"""
Verification of the REPAIRED theorem (v2): the clustering-recovery bound holds
for the Gaussian conditional reattachment channel, with the symplectic-gap
constant C_kappa. Exact truncated-Fock computations (2 modes), no Gaussian
fidelity approximations.

Parts:
  1. Gaussian state synthesis from covariance (Williamson + Gaussian unitary),
     self-validating.
  2. Repaired Theorem 4.7 over the reattachment channel: exact Uhlmann 1-F vs
     the v2 bound with C_kappa(6+C_kappa)/16, sweeping mixedness nb (kappa=2nb).
  3. Petz vs reattachment: pure-rho0 counterexample (Petz returns rho0 for ANY
     input) and dose-response of the covariance gap vs mixedness.
  4. Fidelity Lemma 4.5: (i) analytic vacuum-vs-thermal counterexample to the
     v1 (gap-free) bound; (ii) gap-corrected bound with C_kappa on random
     perturbations of gapped references.
  5. Channel admissibility: eigenvalues of C0 + i*Omega2 (no-steering) vs TMS
     squeezing, and the sufficient condition eta_vac^2*||B0|| <= margin(B0).

Convention: Gamma_ij = Tr[rho {R_i, R_j}], vacuum Gamma = identity,
uncertainty Gamma + i*Omega >= 0, purity iff all symplectic eigenvalues = 1.
"""
import numpy as np
from scipy.linalg import expm, logm, schur, sqrtm

Nc = 24
a1_ = np.diag(np.sqrt(np.arange(1, Nc)), 1)
I1 = np.eye(Nc)
a1 = np.kron(a1_, I1); a2 = np.kron(I1, a1_)
ad1, ad2 = a1.conj().T, a2.conj().T
def quad(A): return (A + A.conj().T)/np.sqrt(2), (A - A.conj().T)/(1j*np.sqrt(2))
x1, p1 = quad(a1); x2, p2 = quad(a2)
R = [x1, p1, x2, p2]
Om4 = np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]], float)
Om2 = Om4[:2, :2]

def herm(M): return (M + M.conj().T)/2
def sqrt_psd(rho):
    w, V = np.linalg.eigh(herm(rho)); w = np.clip(w, 0, None)
    return (V*np.sqrt(w)) @ V.conj().T
def invsqrt_psd(rho, rtol=1e-11):
    w, V = np.linalg.eigh(herm(rho))
    wi = np.where(w > rtol*w.max(), 1/np.sqrt(np.maximum(w, 1e-300)), 0.0)
    return (V*wi) @ V.conj().T
def fidelity(r1, r2):
    s = sqrt_psd(r1)
    w = np.linalg.eigvalsh(herm(s @ r2 @ s))
    return float(np.sum(np.sqrt(np.clip(w, 0, None))))
def ptrace2(rho): return np.einsum('abcb->ac', rho.reshape(Nc, Nc, Nc, Nc))
def gamma_of(rho, Rops):
    n = len(Rops); G = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            G[i, j] = G[j, i] = np.real(np.trace(rho @ (Rops[i]@Rops[j] + Rops[j]@Rops[i])))
    return G
def thermal(nb):
    if nb <= 1e-12:
        d = np.zeros(Nc); d[0] = 1.0
    else:
        d = (nb/(1+nb))**np.arange(Nc); d = d/d.sum()
    return np.diag(d)
def opnorm(M): return np.linalg.norm(M, 2)
def hs(M): return np.linalg.norm(M, 'fro')
def symp_eigs(G):
    return np.sort(np.abs(np.linalg.eigvals(1j*Om4 @ G)))[::2]

# ---------------------------------------------------------------- Part 1
def williamson(V):
    """V = S diag(nu) S^T with S symplectic; returns S, nu (per mode)."""
    Vh = sqrtm(V); Vih = np.linalg.inv(Vh)
    A = Vih @ Om4 @ Vih                      # antisymmetric
    T, O = schur(A)                          # real Schur: 2x2 blocks [[0,k],[-k,0]]
    n = V.shape[0]//2
    kap, cols = [], []
    for b in range(n):
        i = 2*b
        k = T[i, i+1]
        if k < 0:                            # enforce block [[0,+k],[-k,0]]
            O[:, [i, i+1]] = O[:, [i+1, i]]
            k = -k
        kap.append(k); cols += [2*b, 2*b+1]
    kap = np.array(kap); nu = 1/kap
    F = np.diag(np.repeat(np.sqrt(kap), 2))  # D^{-1/2}
    S = Vh @ O @ F
    D = np.diag(np.repeat(nu, 2))
    assert np.max(np.abs(S @ Om4 @ S.T - Om4)) < 1e-8, "S not symplectic"
    assert np.max(np.abs(S @ D @ S.T - V)) < 1e-8, "Williamson mismatch"
    return S, nu

_SIGN = {}
def gaussian_unitary(S):
    """U with gamma(U rho U^dag) = S gamma(rho) S^T; convention fixed by self-test."""
    L = np.real(logm(S))
    def build(h):
        H = sum(0.5*h[i, j]*(R[i]@R[j]) for i in range(4) for j in range(4))
        return expm(-1j*herm(H))
    cands = [Om4 @ L, -Om4 @ L, L @ Om4, -L @ Om4]
    cands = [herm((c + c.T)/2 + 0j).real for c in cands]
    if 'idx' not in _SIGN:
        test = np.kron(thermal(0.2), thermal(0.3))
        G_t = gamma_of(test, R)
        errs = []
        for h in cands:
            U = build(h)
            errs.append(np.max(np.abs(gamma_of(U @ test @ U.conj().T, R) - S @ G_t @ S.T)))
        _SIGN['idx'] = int(np.argmin(errs))
        if min(errs) > 5e-2:
            raise RuntimeError(f"no convention works, errs={errs}")
    return build(cands[_SIGN['idx']])

def gaussian_state(V):
    S, nu = williamson(V)
    rho_D = np.kron(thermal((nu[0]-1)/2), thermal((nu[1]-1)/2))
    U = gaussian_unitary(S)
    rho = herm(U @ rho_D @ U.conj().T)
    err = np.max(np.abs(gamma_of(rho, R) - V))
    return rho, err

# ---------------------------------------------------------------- setup states
s_tms, lam = 0.35, 0.25
U_tms = expm(s_tms*(ad1@ad2 - a1@a2))
U_sq  = expm(0.5*lam*(a1@a1 - ad1@ad1))

def blocks(G): return G[:2, :2], G[:2, 2:], G[2:, 2:]

def petz_true(rho0, sig1):
    r0h = sqrt_psd(rho0)
    m1i = invsqrt_psd(ptrace2(rho0))
    out = r0h @ np.kron(m1i @ sig1 @ m1i, I1) @ r0h
    return herm(out), float(np.real(np.trace(out)))

def reattach_cov(G0, A):
    A0, X0, B0 = blocks(G0)
    W = np.linalg.solve(A0, X0)
    C0 = B0 - X0.T @ np.linalg.solve(A0, X0)
    return np.block([[A, A @ W], [(A @ W).T, C0 + W.T @ A @ W]])

print("=== Part 2: repaired Theorem 4.7 over the REATTACHMENT channel ===")
print("nb   kappa   Ckap   synth_err  1-F(exact)   RHS_v2(simpl)  RHS_v2(full)  OK?  [Kop, cond(i), cond(ii)]")
for nb in [0.02, 0.1, 0.4, 1.0]:
    th = np.kron(thermal(nb), thermal(nb))
    rho0 = herm(U_tms @ th @ U_tms.conj().T)
    rho_w = herm(U_sq @ rho0 @ U_sq.conj().T)
    G0 = gamma_of(rho0, R); Gw = gamma_of(rho_w, R)
    A0, X0, B0 = blocks(G0); A, X, B = blocks(Gw)
    Gt = reattach_cov(G0, A)
    # admissibility
    adm = np.min(np.linalg.eigvalsh(Gt + 1j*Om4))
    rho_t, serr = gaussian_state(Gt)
    Fex = fidelity(rho_w, rho_t); loss = 1 - Fex
    # hypotheses / bound
    A0h = invsqrt_psd(A0); B0h = invsqrt_psd(B0)
    eta = opnorm(A0h @ X0 @ B0h)
    eps = 1 - opnorm(A0h @ (A - A0) @ A0h)
    delta = opnorm(A0h @ (X - X0) @ B0h)
    c1, c2 = np.linalg.eigvalsh(A0)[0], np.linalg.eigvalsh(B0)[0]
    D12 = X - A @ np.linalg.solve(A0, X0)
    DG = Gw - Gt
    kappa = symp_eigs(Gw)[0] - 1
    Ck = (1+kappa)**2 / ((1+kappa)**2 - 1) if kappa > 1e-12 else np.inf
    Gwi = np.linalg.inv(Gw)
    Gwih = invsqrt_psd(Gw)
    K = Gwih @ DG @ Gwih
    Kop, KHS = opnorm(K), hs(K)
    # simplified v2 bound
    den = eps**2 * (1 - (eta + delta)**2/eps)**2 * min(c1, c2)**2
    rhs_simpl = (Ck*(6+Ck)/16) * hs(D12)**2 / den
    # full v2 bound (eq. 52-53): (Ck/8)|K|^2 + (Ck^2/48)|K|^4 via |K| <= |Gw^-1||DG|
    gi = opnorm(Gwi)
    rhs_full = (Ck/8)*(gi*hs(DG))**2 + (Ck**2/48)*(gi*hs(DG))**4
    Cvac = np.sqrt(np.linalg.eigvalsh(B0)[-1]/c1)
    cond_i = (Cvac*eta)**4 * hs(Gw-G0)**2 <= hs(D12)**2
    cond_ii = gi*hs(DG) <= 1
    ok = loss <= min(rhs_simpl, rhs_full) + 1e-12
    print(f"{nb:4.2f} {kappa:6.3f} {Ck:7.2f} {serr:9.1e} {loss:12.3e} {rhs_simpl:13.3e} {rhs_full:13.3e}  "
          f"{'PASS' if ok else 'FAIL'} [{Kop:.3f}, {cond_i}, {cond_ii}]")

print("\n=== Part 3: Petz vs reattachment (dose-response; nb=0 is the pure counterexample) ===")
print("nb   | F(Petz_out, rho0) | ||Tr2(Petz)-sig1||_tr | max|G_petz - G_reatt|")
for nb in [0.0, 0.02, 0.1, 0.4, 1.0]:
    th = np.kron(thermal(nb), thermal(nb))
    rho0 = herm(U_tms @ th @ U_tms.conj().T)
    rho_w = herm(U_sq @ rho0 @ U_sq.conj().T)
    sig1 = herm(ptrace2(rho_w))
    G0 = gamma_of(rho0, R); Gw = gamma_of(rho_w, R)
    A = blocks(Gw)[0]
    rt, tr = petz_true(rho0, sig1); rt = rt/tr
    G_petz = gamma_of(rt, R)
    Gt = reattach_cov(G0, A)
    marg = np.sum(np.abs(np.linalg.eigvalsh(herm(ptrace2(rt) - sig1))))
    print(f"{nb:4.2f} | {fidelity(rt, rho0):17.6f} | {marg:21.4f} | {np.max(np.abs(G_petz - Gt)):20.4f}")

print("\n=== Part 4: Lemma 4.5 -- gap-free counterexample + gap-corrected bound ===")
# (i) analytic: one-mode vacuum vs thermal
for nbv in [0.05, 0.1, 0.2]:
    Fq = (1+nbv)**-0.5
    KHS2 = 8*nbv**2
    print(f"vac-vs-thermal nb={nbv}: 1-F = {1-Fq:.5f}  v1 bound (no gap) = {KHS2/8 + KHS2**2/48:.5f}  "
          f"VIOLATED: {1-Fq > KHS2/8 + KHS2**2/48}")
# (ii) gapped references, random 2-mode perturbations, exact Fock fidelity
rng = np.random.default_rng(7)
worst, nfail, ntot = 0.0, 0, 0
for trial in range(12):
    nb1, nb2 = rng.uniform(0.05, 0.5, 2)
    Q = rng.normal(scale=0.1, size=(4, 4)); Q = Q + Q.T
    Hq = sum(0.5*Q[i, j]*(R[i]@R[j]) for i in range(4) for j in range(4))
    Vq = expm(-1j*herm(Hq))
    r1 = herm(Vq @ np.kron(thermal(nb1), thermal(nb2)) @ Vq.conj().T)
    Q2 = Q + rng.normal(scale=0.05, size=(4, 4)); Q2 = herm(Q2 + Q2.T)
    Hq2 = sum(0.5*Q2[i, j]*(R[i]@R[j]) for i in range(4) for j in range(4))
    Vq2 = expm(-1j*herm(Hq2))
    r2 = herm(Vq2 @ np.kron(thermal(nb1*1.15), thermal(nb2*0.9)) @ Vq2.conj().T)
    G1, G2 = gamma_of(r1, R), gamma_of(r2, R)
    G1h = invsqrt_psd(G1)
    K = G1h @ (G1 - G2) @ G1h
    if opnorm(K) > 0.5: continue
    kap = symp_eigs(G1)[0] - 1
    if kap < 1e-3: continue
    Ck = (1+kap)**2/((1+kap)**2 - 1)
    Fq = fidelity(r1, r2)
    bound = (Ck/8)*hs(K)**2 + (Ck**2/48)*hs(K)**4
    ntot += 1
    worst = max(worst, (1-Fq)/bound)
    nfail += (1-Fq) > bound
print(f"gapped Lemma 4.5 (C_kappa): {ntot} samples, failures={nfail}, max ratio (1-F)/bound = {worst:.3f}")

print("\n=== Part 5: channel admissibility (no-steering) ===")
print("s_tms | nb   | min eig(C0+i*Om2) | eta^2*||B0|| vs margin nu0/(1+nu0)*c2")
for s in [0.35, 0.8, 1.2]:
    Uts = expm(s*(ad1@ad2 - a1@a2))
    for nb in [0.0, 0.1, 0.4]:
        th = np.kron(thermal(nb), thermal(nb))
        r0 = herm(Uts @ th @ Uts.conj().T)
        G0 = gamma_of(r0, R)
        A0, X0, B0 = blocks(G0)
        C0 = B0 - X0.T @ np.linalg.solve(A0, X0)
        me = np.min(np.linalg.eigvalsh(C0 + 1j*Om2))
        eta = opnorm(invsqrt_psd(A0) @ X0 @ invsqrt_psd(B0))
        nu0 = symp_eigs(np.kron(np.eye(2), B0))[0] - 1 if False else np.sort(np.abs(np.linalg.eigvals(1j*Om2 @ B0)))[0] - 1
        c2 = np.linalg.eigvalsh(B0)[0]
        margin = nu0/(1+nu0)*c2 if nu0 > 0 else 0.0
        lhs = eta**2*np.linalg.eigvalsh(B0)[-1]
        print(f"{s:5.2f} | {nb:4.2f} | {me:17.4f} | {lhs:.4f} vs {margin:.4f}  "
              f"(suff.cond {'holds' if lhs <= margin else 'fails'}; admissible: {me > -1e-9})")

print("\nDone.")
