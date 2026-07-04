"""
Verificacion exacta de ai.viXra:2512.0060 (Clustering-Recovery Bridge).
Todo en espacio de Fock truncado (2 modos), sin aproximaciones gaussianas:
  A. Prop 2.14 / Lemma D.2: mapa de Petz exacto vs formula del paper,
     barrido en pureza de rho_0 (limite puro = contraejemplo analitico).
  B. Teorema 4.7: 1-F exacta vs cota, con C=3/8 (enunciada) y C=7/16 (derivada).
  C. Lemma 4.5: fidelidad Uhlmann exacta vs cota (1/8)|K|^2+(1/48)|K|^4,
     y comparacion con la formula (84), que es la fidelidad CLASICA.
  D. eta_vac exacto en Klein-Gordon reticular: tasa vs arccosh(1+m^2/2).
Convenio: Gamma_ij = Tr[rho {R_i,R_j}], vacio Gamma = 1.
"""
import numpy as np
from scipy.linalg import expm

Nc = 24
a1_ = np.diag(np.sqrt(np.arange(1, Nc)), 1)
I1 = np.eye(Nc)
a1 = np.kron(a1_, I1); a2 = np.kron(I1, a1_)
ad1, ad2 = a1.conj().T, a2.conj().T
def quad(A): return (A + A.conj().T)/np.sqrt(2), (A - A.conj().T)/(1j*np.sqrt(2))
x1, p1 = quad(a1); x2, p2 = quad(a2)
R = [x1, p1, x2, p2]
Om = np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]], float)

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
def ptrace2(rho):
    return np.einsum('abcb->ac', rho.reshape(Nc, Nc, Nc, Nc))
def gamma_of(rho, Rops):
    n = len(Rops)
    G = np.zeros((n, n))
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

s_tms, lam = 0.35, 0.25
U_tms = expm(s_tms*(ad1@ad2 - a1@a2))
U_sq  = expm(0.5*lam*(a1@a1 - ad1@ad1))

def petz_true(rho0, sig1):
    r0h = sqrt_psd(rho0)
    m1i = invsqrt_psd(ptrace2(rho0))
    M = np.kron(m1i @ sig1 @ m1i, I1)
    out = r0h @ M @ r0h
    return herm(out), float(np.real(np.trace(out)))

def blocks(G):
    return G[:2, :2], G[:2, 2:], G[2:, 2:]

print("=== A/B: Petz exacto vs Prop 2.14, y Teorema 4.7 ===")
print("nb_th | pure(rho0) | |Gt_true - Gt_paper|_max | |Tr2(rt)-sig1|_tr | F(rt,rho0) | 1-F(w,rt) | RHS(3/8) | RHS(7/16)")
for nb in [0.02, 0.1, 0.4, 1.0]:
    th = np.kron(thermal(nb), thermal(nb))
    rho0 = herm(U_tms @ th @ U_tms.conj().T)
    rho_w = herm(U_sq @ rho0 @ U_sq.conj().T)          # excitacion local Bogoliubov: B=B0
    sig1 = herm(ptrace2(rho_w))
    G0 = gamma_of(rho0, R); Gw = gamma_of(rho_w, R)
    A0, X0, B0 = blocks(G0); A, X, B = blocks(Gw)
    # formula del paper (Prop 2.14)
    A0i = np.linalg.inv(A0)
    Gt_paper = np.block([[A, A @ A0i @ X0],
                         [(A @ A0i @ X0).T, B0 + X0.T @ A0i @ (A - A0) @ A0i @ X0]])
    rt, tr = petz_true(rho0, sig1)
    rt = rt/tr
    Gt_true = gamma_of(rt, R)
    marg_dev = np.sum(np.abs(np.linalg.eigvalsh(herm(ptrace2(rt) - sig1))))
    F_rt_rho0 = fidelity(rt, rho0)
    Fex = fidelity(rho_w, rt); loss = 1 - Fex
    # cota del Teorema 4.7
    A0h = invsqrt_psd(A0); B0h = invsqrt_psd(B0)
    eta = opnorm(A0h @ X0 @ B0h)
    eps = 1 - opnorm(A0h @ (A - A0) @ A0h)
    delta = opnorm(A0h @ (X - X0) @ B0h)
    c1, c2 = np.linalg.eigvalsh(A0)[0], np.linalg.eigvalsh(B0)[0]
    D12 = X - A @ A0i @ X0
    den = eps**2 * (1 - (eta + delta)**2/eps)**2 * min(c1, c2)**2
    rhs38, rhs716 = (3/8)*hs(D12)**2/den, (7/16)*hs(D12)**2/den
    purity = np.real(np.trace(rho0 @ rho0))
    print(f"{nb:5.2f} | {purity:10.4f} | {np.max(np.abs(Gt_true-Gt_paper)):24.4f} | {marg_dev:17.2e} |"
          f" {F_rt_rho0:10.6f} | {loss:9.6f} | {rhs38:8.5f} | {rhs716:8.5f}")
    if nb == 0.4:
        print(f"      [hip: eta={eta:.4f} eps={eps:.4f} delta={delta:.4f} "
              f"cond(d): {(eta+delta)/np.sqrt(eps):.4f}<1 | trace-err Petz {abs(tr-1):.1e}]")

# control del pipeline: sigma1 = marginal del vacio => Petz devuelve rho0 exacto
th = np.kron(thermal(0.4), thermal(0.4))
rho0 = herm(U_tms @ th @ U_tms.conj().T)
rt, tr = petz_true(rho0, herm(ptrace2(rho0)))
print(f"[control] Petz(marginal de rho0) = rho0:  1-F = {1-fidelity(rt/tr, rho0):.2e}")

print("\n=== C: Lemma 4.5 (cota de fidelidad) y Eq.(84) clasica vs cuantica ===")
rng = np.random.default_rng(3)
worst, nviol, nq_ge_cl = 0.0, 0, 0
for trial in range(25):
    Q1 = rng.normal(scale=0.12, size=(4, 4)); Q1 = Q1 + Q1.T
    Q2 = rng.normal(scale=0.12, size=(4, 4)); Q2 = Q2 + Q2.T
    def gstate(Q, nb1, nb2):
        H = sum(0.5*Q[i, j]*(R[i]@R[j]) for i in range(4) for j in range(4))
        V = expm(-1j*herm(H))
        return herm(V @ np.kron(thermal(nb1), thermal(nb2)) @ V.conj().T)
    r1 = gstate(Q1, 0.15, 0.3); r2 = gstate(Q2, 0.2, 0.25)
    G1, G2 = gamma_of(r1, R), gamma_of(r2, R)
    G1h = invsqrt_psd(G1)
    K = G1h @ (G1 - G2) @ G1h
    if opnorm(K) > 0.5:  # fuera del regimen del lema
        continue
    Fq = fidelity(r1, r2)
    bound = (1/8)*hs(K)**2 + (1/48)*hs(K)**4
    Fcl = (4**4*np.linalg.det(G1)*np.linalg.det(G2)/np.linalg.det(G1+G2)**2)**0.25
    worst = max(worst, (1-Fq)/bound)
    nviol += (1-Fq) > bound
    nq_ge_cl += Fq >= Fcl - 1e-12
print(f"muestras validas: ratio max (1-F_q)/cota = {worst:.3f} | violaciones: {nviol} | F_q >= F_clasica en {nq_ge_cl} casos")

print("\n=== D: eta_vac exacto en KG reticular (m=1) ===")
N, m = 240, 1.0
M = (m**2 + 2)*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
w, V = np.linalg.eigh(M)
Mm = (V/np.sqrt(w)) @ V.T; Mp = (V*np.sqrt(w)) @ V.T
i1 = np.arange(0, 60)
rs = np.arange(2, 42, 4)
etas = []
for r in rs:
    i2 = np.arange(60 + r, N)
    def blk(K, ia, ib): return K[np.ix_(ia, ib)]
    A0 = np.block([[blk(Mm,i1,i1), 0*blk(Mm,i1,i1)], [0*blk(Mm,i1,i1), blk(Mp,i1,i1)]])
    B0 = np.block([[blk(Mm,i2,i2), np.zeros((len(i2),)*2)], [np.zeros((len(i2),)*2), blk(Mp,i2,i2)]])
    X0 = np.block([[blk(Mm,i1,i2), np.zeros((len(i1),len(i2)))], [np.zeros((len(i1),len(i2))), blk(Mp,i1,i2)]])
    etas.append(opnorm(invsqrt_psd(A0) @ X0 @ invsqrt_psd(B0)))
sl = -np.polyfit(rs[2:], np.log(etas[2:]), 1)[0]
print(f"pendiente ln(eta_vac) vs r = {sl:.4f} | arccosh(1+m^2/2) = {np.arccosh(1+m**2/2):.4f} | m continuo = {m}")
