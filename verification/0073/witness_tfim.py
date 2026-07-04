import numpy as np
N = 8; J, h, beta = 1.0, 1.5, 1.0
sx = np.array([[0,1],[1,0]], complex); sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]], complex); I2 = np.eye(2)
def op_at(o,i):
    M = np.array([[1]],complex)
    for k in range(N): M = np.kron(M, o if k==i else I2)
    return M
HS = sum(-J*op_at(sx,i)@op_at(sx,i+1) for i in range(N-1)) + sum(-h*op_at(sz,i) for i in range(N))
w, V = np.linalg.eigh(HS); D = 2**N
p = np.exp(-beta*w); p /= p.sum(); sqp = np.sqrt(p)
j0 = N//2
St = V.conj().T@op_at(sz,j0)@V
W = w[:,None]-w[None,:]
def kms(A,B): return np.einsum('i,ij,j,ij->', sqp, A.conj(), sqp, B)

# ---- A: per-block weight discovery
print("=== A: per-block ratio  E_direct / [ (1/2)||[S(w),O]||^2 ]  vs modular weights ===")
rng = np.random.default_rng(1)
keys = np.unique(np.round(W/1e-9).astype(np.int64))
tested = 0
for key in keys:
    om = key*1e-9
    if om < 0.05: continue          # test a few positive-frequency blocks
    A = np.where(np.abs(W - om) < 5e-10, St, 0)
    if np.abs(A).max() < 1e-8: continue
    ratios = []
    for t in range(3):
        X = rng.normal(size=(D,D)) + 1j*rng.normal(size=(D,D)); O = (X+X.conj().T)/2
        Ad = A.conj().T
        LdO = Ad@O@A - 0.5*(Ad@A@O + O@Ad@A)
        Ed = -np.real(kms(O, LdO))
        Cm = A@O - O@A
        half = 0.5*np.real(kms(Cm, Cm))
        ratios.append(Ed/half)
    r = np.mean(ratios); s = np.std(ratios)
    print(f"omega={om:+.4f}: ratio={r:.6f} (std {s:.2e}) | e^(-bw/2)={np.exp(-beta*om/2):.6f} "
          f"| e^(+bw/2)={np.exp(beta*om/2):.6f} | cosh={np.cosh(beta*om/2):.6f}")
    tested += 1
    if tested >= 4: break

# ---- B: Table 2 with k = j0 - eps
print("\n=== B: witness with k = j0 - eps (and j0 + eps) ===")
S0 = np.where(np.abs(W) < 1e-9, St, 0)
P = {'x':sx,'y':sy,'z':sz}
def ropt(k):
    ops = [V.conj().T@op_at(P[a],k)@V for a in 'xyz']
    G = np.zeros((3,3),complex); M = np.zeros((3,3),complex)
    for a in range(3):
        for b in range(3):
            G[a,b] = kms(ops[a],ops[b])
            Ca = S0@ops[a]-ops[a]@S0; Cb = S0@ops[b]-ops[b]@S0
            M[a,b] = kms(Ca,Cb)
    ev = np.linalg.eigvals(np.linalg.solve((G+G.conj().T)/2,(M+M.conj().T)/2))
    return float(np.max(np.real(ev)))
for eps in [1,2,3]:
    print(f"eps={eps}: k=j0-eps -> {ropt(j0-eps):.6f} | k=j0+eps -> {ropt(j0+eps):.6f} "
          f"| paper {dict([(1,0.134648),(2,0.0885503),(3,0.0684289)])[eps]}")
print(f"note: j0={j0} (0-indexed), N={N}; also k=j0 itself: {ropt(j0):.6f}")
