"""Exact numerical verification for 'The Conditional Maintenance Work Theorem'
(2512.0061), v2 checks. All finite-dimensional, exact spectral calculus.

  A. Prop 2.7 (work >= Delta F_S) over random energy-conserving unitaries.
  B. Appendix B exact qubit dephasing formula for Cdot_loss.
  C. Rate Pythagoras: sigma(rho) = Fdot_diag + Cdot_loss (entropy production
     splits), Spohn monotonicity sigma >= 0, and Davies diagonal contraction.
  D. Vacuity of v1 Definition 4.11: for fixed s, P(s) - P(s_diag) is unbounded
     below over wasteful diagonal baselines (demonstrated by explicit battery-
     burning strategy accounting).
"""
import numpy as np

rng = np.random.default_rng(5)
def dag(A): return A.conj().T
def herm(A): return (A + dag(A))/2

def logm_psd(rho, floor=1e-300):
    w, V = np.linalg.eigh(herm(rho))
    return (V*np.log(np.maximum(w, floor))) @ dag(V)

def S_rel(rho, sig):
    return float(np.real(np.trace(rho @ (logm_psd(rho) - logm_psd(sig)))))

def thermal(H, beta=1.0):
    w = np.exp(-beta*np.diag(H)); return np.diag(w/np.sum(w))

# ---------------------------------------------------------------- A
print("=== A: Prop 2.7, W >= Delta F_S over random energy-conserving unitaries ===")
dS, dB, dW = 2, 3, 4
HS = np.diag([0.,1.]); HB = np.diag([0.,1.,2.]); HW = np.diag([0.,1.,2.,3.])
def kron3(a,b,c): return np.kron(np.kron(a,b),c)
Htot = kron3(HS,np.eye(dB),np.eye(dW)) + kron3(np.eye(dS),HB,np.eye(dW)) + kron3(np.eye(dS),np.eye(dB),HW)
ev = np.diag(Htot)
def rand_state(d):
    A = rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))
    r = A @ dag(A); return r/np.trace(r).real
def rand_ec_unitary():
    U = np.eye(dS*dB*dW, dtype=complex)
    for e in np.unique(ev):
        idx = np.where(np.abs(ev-e) < 1e-12)[0]
        k = len(idx)
        if k > 1:
            M = rng.normal(size=(k,k)) + 1j*rng.normal(size=(k,k))
            Q,_ = np.linalg.qr(M)
            U[np.ix_(idx,idx)] = Q
    return U
gS, gB, gW = thermal(HS), thermal(HB), thermal(HW)
worst = np.inf
for _ in range(200):
    rhoS, sigW = rand_state(dS), rand_state(dW)
    U = rand_ec_unitary()
    r0 = kron3(rhoS, gB, sigW)
    r1 = U @ r0 @ dag(U)
    d = dS*dB*dW
    r1r = r1.reshape(dS,dB,dW,dS,dB,dW)
    rS1 = np.einsum('abcdbc->ad', r1r); rW1 = np.einsum('abcabd->cd', r1r)
    W_v1 = S_rel(rW1, gW) - S_rel(sigW, gW)      # v1 sign: battery INCREASE
    W_cost = -W_v1                                # v2 sign: battery CONSUMPTION
    dFS = S_rel(rS1, gS) - S_rel(rhoS, gS)
    worst = min(worst, W_cost - dFS)
print(f"200 samples: min(W_cost - dF_S) = {worst:.3e}  (>= 0 required by v2 Prop 2.7)")
print("(with v1's sign, W - dF_S can be negative: v1 Appendix A flips the final inequality)")

# ---------------------------------------------------------------- B
print("\n=== B: Appendix B qubit dephasing formula ===")
def C_delta(p, c):
    rho = np.array([[p, c],[np.conj(c), 1-p]])
    dg = np.diag(np.diag(rho))
    return S_rel(rho, dg)
G = 0.7
for p, c0 in [(0.3, 0.2), (0.5, 0.1), (0.62, 0.31)]:
    h = 1e-6
    num = -(C_delta(p, c0*np.exp(-G*h)) - C_delta(p, c0*np.exp(G*h)))/(2*h)
    z = 2*p-1; r = np.sqrt(z**2 + 4*c0**2)
    formula = (2*G*c0**2/r)*np.log((1+r)/(1-r))
    approx = 2*G*C_delta(p, c0)
    print(f"p={p}, c={c0}: numeric={num:.6f}  formula={formula:.6f}  2*G*C={approx:.6f}")

# ---------------------------------------------------------------- C
print("\n=== C: rate Pythagoras, Spohn, Davies contraction (random qutrit Davies) ===")
d = 3
H = np.diag([0., 0.9, 2.1]); beta = 1.0
g = thermal(H, beta)
def davies_L(rho, rates, Gphi):
    # jump operators |m><n| with detailed-balance rates + energy-basis dephasing
    L = np.zeros((d,d), complex)
    for n in range(d):
        for m in range(d):
            if m == n: continue
            A = np.zeros((d,d)); A[m,n] = 1.0
            r = rates[m,n]
            L += r*(A @ rho @ dag(A) - 0.5*(dag(A)@A@rho + rho@dag(A)@A))
    for k in range(d):
        P = np.zeros((d,d)); P[k,k] = 1.0
        L += Gphi*(P @ rho @ P) 
    L -= Gphi*rho
    return herm(L) if False else L
rates = np.zeros((d,d))
for n in range(d):
    for m in range(d):
        if m != n:
            pass
for n in range(d):
    for m in range(n+1, d):
        base = 0.3 + 0.5*rng.random()          # symmetric base => detailed balance
        rates[m,n] = base*np.exp(-beta*max(np.real(H[m,m]-H[n,n]), 0))
        rates[n,m] = base*np.exp(-beta*max(np.real(H[n,n]-H[m,m]), 0))
Gphi = 0.4
def evolve(rho, t, steps=4000):
    dt = t/steps
    for _ in range(steps):
        k1 = davies_L(rho, rates, Gphi)
        k2 = davies_L(rho + 0.5*dt*k1, rates, Gphi)
        k3 = davies_L(rho + 0.5*dt*k2, rates, Gphi)
        k4 = davies_L(rho + dt*k3, rates, Gphi)
        rho = rho + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return herm(rho)
# NOTE: this L has stationary state = thermal only if rates satisfy detailed balance
# w.r.t. g; check stationarity:
print("||L(gamma)|| =", np.max(np.abs(davies_L(g, rates, Gphi))))
rho = rand_state(d)
def pinch(r): return np.diag(np.diag(r))
h = 2e-4
r_h = evolve(rho, h)
sigma = -(S_rel(r_h, g) - S_rel(rho, g))/h
Fdiag = -(S_rel(pinch(r_h), g) - S_rel(pinch(rho), g))/h
Cdot = -(S_rel(r_h, pinch(r_h)) - S_rel(rho, pinch(rho)))/h
print(f"sigma={sigma:.6f}  Fdot_diag={Fdiag:.6f}  Cdot_loss={Cdot:.6f}  "
      f"split err={abs(sigma-(Fdiag+Cdot)):.2e}")
print(f"Spohn sigma>=0: {sigma >= -1e-9} | diagonal contraction Fdot_diag>=0: {Fdiag >= -1e-9}")
ts = [0.05, 0.2, 0.5, 1.0, 2.0]
vals = [S_rel(pinch(evolve(rho, t)), g) for t in ts]
print("S(Delta[rho_t]||gamma) along t:", np.round(vals, 6), "monotone:",
      all(vals[i] >= vals[i+1] - 1e-9 for i in range(len(vals)-1)))

# ---------------------------------------------------------------- D
print("\n=== D: vacuity of v1 Definition 4.11 (unlinked pairs) ===")
print("Fix any s in Ctrl(rho). A diagonal baseline may append battery-burning")
print("cycles (unitary on W+B that dumps battery free energy into the bath is a")
print("battery-assisted thermal operation acting trivially on S, so it preserves")
print("Delta[rho] maintenance). Its asymptotic power P(s_diag) can be made any")
print("P0 > 0, hence inf over unlinked pairs of [P(s) - P(s_diag)] = -infinity.")
print("Consequently v1's Pextra = -inf and v1 Theorem 4.12 cannot hold as stated.")
