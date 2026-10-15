import numpy as np, sys, json, time, pathlib
np.random.seed(0)
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def partial_trace(rho, dims, keep):
    n = len(dims)
    rho = rho.reshape(dims + dims)
    for ax in sorted((i for i in range(n) if i not in keep), reverse=True):
        rho = np.trace(rho, axis1=ax, axis2=ax + rho.ndim//2)
    d = int(np.prod([dims[i] for i in keep])) if keep else 1
    return rho.reshape(d, d)

def h_fun(rho, f, floor=None):
    ev, U = np.linalg.eigh(rho)
    if floor is not None: ev = np.maximum(ev, floor)
    return (U * f(ev)) @ U.conj().T

def logm_h(rho):  # log on support
    ev, U = np.linalg.eigh(rho)
    ev = np.maximum(ev, 1e-300)
    return (U * np.log(ev)) @ U.conj().T

def vn(rho):
    ev = np.linalg.eigvalsh(rho); ev = ev[ev > 1e-14]
    return float(-np.sum(ev*np.log(ev)))

def cmi(rho, dims):
    rAB = partial_trace(rho, dims, [0,1]); rBC = partial_trace(rho, dims, [1,2])
    rB  = partial_trace(rho, dims, [1])
    return vn(rAB)+vn(rBC)-vn(rB)-vn(rho)

# ---------- PART 0: Eq (15) ----------
out = ["== Eq(15): I(A:C|B) vs D(rho||sigma_MP), random 3-qubit =="]
for trial in range(4):
    X = np.random.randn(8,8) + 1j*np.random.randn(8,8)
    rho = X @ X.conj().T; rho /= np.trace(rho).real
    dims=[2,2,2]
    I = cmi(rho, dims)
    rAB = partial_trace(rho,dims,[0,1]); rBC = partial_trace(rho,dims,[1,2]); rB = partial_trace(rho,dims,[1])
    L = np.kron(logm_h(rAB), np.eye(2)) + np.kron(np.eye(2), logm_h(rBC)) - np.kron(np.eye(2), np.kron(logm_h(rB), np.eye(2)))
    ev, U = np.linalg.eigh((L+L.conj().T)/2)
    M = (U * np.exp(ev)) @ U.conj().T
    Z = np.trace(M).real
    D_un = float(np.trace(rho @ (logm_h(rho) - logm_h(M))).real)
    D_no = float(np.trace(rho @ (logm_h(rho) - logm_h(M/Z))).real)
    lz=np.log(Z)
    out.append(f" trial {trial}: I={I:.6f} D(rho||M)={D_un:.6f} D(rho||M/Z)={D_no:.6f} logZ={lz:+.6f} | I==D(M):{abs(I-D_un):.1e} I==D_norm-logZ:{abs(I-(D_no-lz)):.1e} (wrong +logZ resid:{abs(I-(D_no+lz)):.1e})")
print("\n".join(out)); sys.stdout.flush()

# ---------- TFIM machinery ----------
def tfim_H(N, J, h):
    Xp = np.array([[0,1],[1,0]]); Zp = np.array([[1,0],[0,-1]])
    D = 2**N; H = np.zeros((D,D))
    def op(sites_ops):
        M = np.array([[1.]])
        for i in range(N):
            M = np.kron(M, sites_ops.get(i, np.eye(2)))
        return M
    for i in range(N-1): H += -J*op({i:Xp, i+1:Xp})
    for i in range(N):   H += -h*op({i:Zp})
    return H

def gibbs(evH, UH, beta):
    w = np.exp(-beta*(evH - evH.min())); w /= w.sum()
    return (UH * w) @ UH.conj().T

def petz_reconstruct(rABC_dims, rAB, rB, rBC, dA, dB, dC):
    Bm = h_fun(rB, lambda x: np.maximum(x,1e-12)**-0.5)
    inner = np.kron(np.eye(dA), Bm) @ rAB @ np.kron(np.eye(dA), Bm)
    X = np.kron(inner, np.eye(dC))
    S = h_fun(rBC, lambda x: np.maximum(x,0)**0.5)
    KS = np.kron(np.eye(dA), S)
    rt = KS @ X @ KS
    ev, U = np.linalg.eigh((rt+rt.conj().T)/2)
    ev = np.clip(ev, 1e-14, None)
    rt = (U*ev) @ U.conj().T
    return rt/np.trace(rt).real

def rotated_reconstruct(rAB, rB, rBC, dA, dB, dC, tmax=6.0, nt=61):
    ts = np.linspace(-tmax, tmax, nt)
    wts = 1.0/(np.cosh(np.pi*ts)+1.0)
    trap = np.ones(nt); trap[0]=trap[-1]=0.5
    wts = wts*trap; wts /= wts.sum()
    evB, UB = np.linalg.eigh(rB); evB = np.maximum(evB, 1e-12)
    evS, US = np.linalg.eigh(rBC); evS = np.maximum(evS, 0)
    lgB = np.log(evB); lgS = np.log(np.maximum(evS, 1e-300))
    acc = np.zeros((dA*dB*dC, dA*dB*dC), dtype=complex)
    for t, wk in zip(ts, wts):
        zB = np.exp(-(1+1j*t)/2 * lgB)
        PB = (UB * zB) @ UB.conj().T
        inner = np.kron(np.eye(dA), PB) @ rAB @ np.kron(np.eye(dA), PB.conj().T)
        X = np.kron(inner, np.eye(dC))
        zS = np.exp((1+1j*t)/2 * np.where(evS>0, lgS, -np.inf))
        zS = np.nan_to_num(zS, nan=0.0, posinf=0.0, neginf=0.0)
        QS = (US * zS) @ US.conj().T
        KQ = np.kron(np.eye(dA), QS)
        acc += wk * (KQ @ X @ KQ.conj().T)
    ev, U = np.linalg.eigh((acc+acc.conj().T)/2)
    ev = np.clip(ev.real, 1e-14, None)
    rt = (U*ev) @ U.conj().T
    return rt/np.trace(rt).real

def fit_slope(ws, ys):
    ws = np.array(ws); ys = np.array(ys)
    m = ys > 0
    A = np.vstack([np.ones(m.sum()), -ws[m]]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ys[m]), rcond=None)
    return coef[1]

def run(hs, betas, methods, tag):
    N, J, LA = 10, 1.0, 2
    res = {}
    for h in hs:
        evH, UH = np.linalg.eigh(tfim_H(N,J,h))
        for beta in betas:
            rho = gibbs(evH, UH, beta)
            data = {m: {"CMI":[], "T":[], "HS":[]} for m in methods}
            for w in range(1,7):
                dA, dB, dC = 2**LA, 2**w, 2**(N-LA-w)
                dims = [dA,dB,dC]
                rAB = partial_trace(rho, dims, [0,1]); rBC = partial_trace(rho, dims, [1,2])
                rB  = partial_trace(rho, dims, [1])
                I = cmi(rho, dims)
                for m in methods:
                    if m=="petz": rt = petz_reconstruct(None, rAB, rB, rBC, dA,dB,dC)
                    else: rt = rotated_reconstruct(rAB, rB, rBC, dA,dB,dC)
                    dlt = rho - rt
                    sv = np.linalg.svd(dlt, compute_uv=False)
                    data[m]["T"].append(0.5*sv.sum()); data[m]["HS"].append(np.sqrt((sv**2).sum()))
                    data[m]["CMI"].append(I)
            for m in methods:
                muC = fit_slope(range(1,7), data[m]["CMI"]); muT = fit_slope(range(1,7), data[m]["T"])
                muH = fit_slope(range(1,7), data[m]["HS"])
                res[f"{h}_{beta}_{m}"] = (muC, muT, muH, 2*muT)
                print(f"{tag} h={h} beta={beta} {m}: muCMI={muC:.4f} muT={muT:.4f} muHS={muH:.4f} muT2={2*muT:.4f}", flush=True)
    return res

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv)>1 else "petz"
    t0=time.time()
    if mode=="petz":
        run([1.05,1.10,1.20,1.50],[0.3,0.6,1.0,1.5,2.0],["petz"],"PETZ")
    elif mode=="petz2":
        run([1.20],[0.6,1.0,1.5,2.0],["petz"],"PETZ")
        run([1.50],[0.3,0.6,1.0,1.5,2.0],["petz"],"PETZ")
    elif mode=="rot":
        run([1.05],[0.3,0.6,1.0,1.5,2.0],["rotated"],"ROT")
    elif mode=="rot2":
        run([1.50],[0.3,2.0],["rotated"],"ROT2")
    print(f"[{mode}] done in {time.time()-t0:.1f}s")

def fig8():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    N, J, LA = 10, 1.0, 2
    combos = [(1.05,0.30),(1.05,0.60),(1.10,0.30),(1.10,0.60),(1.20,0.30),(1.20,0.60),(1.50,0.30),(1.50,0.60)]
    fig, axes = plt.subplots(4,2, figsize=(9,12))
    rows=[]
    evH_cache={}
    for idx,(h,beta) in enumerate(combos):
        if h not in evH_cache: evH_cache[h]=np.linalg.eigh(tfim_H(N,J,h))
        evH,UH = evH_cache[h]
        rho = gibbs(evH,UH,beta)
        Tp, Tr = [], []
        for w in range(1,7):
            dA,dB,dC = 2**LA, 2**w, 2**(N-LA-w)
            dims=[dA,dB,dC]
            rAB=partial_trace(rho,dims,[0,1]); rBC=partial_trace(rho,dims,[1,2]); rB=partial_trace(rho,dims,[1])
            for tag,fn in (("p",petz_reconstruct),("r",rotated_reconstruct)):
                rt = fn(None,rAB,rB,rBC,dA,dB,dC) if tag=="p" else rotated_reconstruct(rAB,rB,rBC,dA,dB,dC)
                sv=np.linalg.svd(rho-rt,compute_uv=False)
                (Tp if tag=="p" else Tr).append(0.5*sv.sum())
            rows.append((h,beta,w,Tp[-1],Tr[-1]))
            print(f"FIG h={h} beta={beta} w={w} Tpetz={Tp[-1]:.3e} Trot={Tr[-1]:.3e}", flush=True)
        ws=list(range(1,7))
        ax=axes[idx//2][idx%2]
        ax.semilogy(ws,Tp,'o-',label='Petz')
        ax.semilogy(ws,Tr,'s--',label='Rotated')
        ax.axvline(3,ls=':',c='gray'); ax.set_title(f"h={h}, beta={beta}", fontsize=9)
        ax.set_xlabel('w'); ax.set_ylabel('T(w)'); ax.legend(fontsize=7)
        cross = next((w for w in ws if Tp[w-1] < Tr[w-1] and Tr[0] < Tp[0]), None)
        print(f"FIG crossover h={h} beta={beta}: w*={cross}", flush=True)
    plt.tight_layout()
    outputs = [
        ROOT / "papers" / "0101-beyond-gaussianity" / "fig_crossover_0101.pdf",
        ROOT / "results" / "0101" / "fig_crossover_0101.pdf",
    ]
    for out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
    csv_path = ROOT / "results" / "0101" / "fig8_data.csv"
    np.savetxt(csv_path, np.array(rows), header='h,beta,w,T_petz,T_rotated', delimiter=',')
    print("fig saved to papers/0101-beyond-gaussianity and results/0101", flush=True)

if len(sys.argv)>1 and sys.argv[1]=="fig8":
    fig8()
