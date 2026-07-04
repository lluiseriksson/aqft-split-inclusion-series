"""Pipeline control: Petz(marginal of rho0) = rho0 exactly."""
import numpy as np
from scipy.linalg import expm
Nc = 20
a1_ = np.diag(np.sqrt(np.arange(1, Nc)), 1); I1 = np.eye(Nc)
a1 = np.kron(a1_, I1); a2 = np.kron(I1, a1_)
ad1, ad2 = a1.conj().T, a2.conj().T
def herm(M): return (M + M.conj().T)/2
def sqrt_psd(r):
    w, V = np.linalg.eigh(herm(r)); w = np.clip(w, 0, None)
    return (V*np.sqrt(w)) @ V.conj().T
def invsqrt_psd(r, rtol=1e-11):
    w, V = np.linalg.eigh(herm(r))
    wi = np.where(w > rtol*w.max(), 1/np.sqrt(np.maximum(w, 1e-300)), 0.0)
    return (V*wi) @ V.conj().T
def fidelity(r1, r2):
    s = sqrt_psd(r1); w = np.linalg.eigvalsh(herm(s @ r2 @ s))
    return float(np.sum(np.sqrt(np.clip(w, 0, None))))
def ptrace2(r): return np.einsum('abcb->ac', r.reshape(Nc,Nc,Nc,Nc))
def thermal(nb):
    d = (nb/(1+nb))**np.arange(Nc); d = d/d.sum(); return np.diag(d)
U = expm(0.35*(ad1@ad2 - a1@a2))
rho0 = herm(U @ np.kron(thermal(0.4), thermal(0.4)) @ U.conj().T)
m1 = herm(ptrace2(rho0))
r0h = sqrt_psd(rho0); m1i = invsqrt_psd(m1)
out = r0h @ np.kron(m1i @ m1 @ m1i, I1) @ r0h
out = herm(out)/np.real(np.trace(out))
print(f"[control] 1 - F(Petz(rho0_marginal), rho0) = {1-fidelity(out, rho0):.2e}")
