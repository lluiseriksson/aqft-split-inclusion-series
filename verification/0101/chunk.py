import numpy as np, sys, time, os, tempfile
t0=time.time()
CACHE_DIR = os.path.join(tempfile.gettempdir(), "v0101")
os.makedirs(CACHE_DIR, exist_ok=True)

def partial_trace(rho, dims, keep):
    n=len(dims); rho=rho.reshape(dims+dims)
    for ax in sorted((i for i in range(n) if i not in keep), reverse=True):
        rho=np.trace(rho, axis1=ax, axis2=ax+rho.ndim//2)
    d=int(np.prod([dims[i] for i in keep]))
    return rho.reshape(d,d)

def vn(rho):
    ev=np.linalg.eigvalsh(rho); ev=ev[ev>1e-14]
    return float(-np.sum(ev*np.log(ev)))

def tfim_H(N,J,h):
    Xp=np.array([[0.,1.],[1.,0.]]); Zp=np.array([[1.,0.],[0.,-1.]])
    D=2**N; H=np.zeros((D,D))
    def op(so):
        M=np.array([[1.]])
        for i in range(N): M=np.kron(M, so.get(i,np.eye(2)))
        return M
    for i in range(N-1): H+=-J*op({i:Xp,i+1:Xp})
    for i in range(N): H+=-h*op({i:Zp})
    return H

def sandwich_blocks(X, Q, dA, dS):
    # (I_A x Q) X (I_A x Q^dag) via BLAS tensordot
    Xr = X.reshape(dA,dS,dA,dS)
    Y = np.tensordot(Q, Xr, axes=([1],[1]))        # (s,a,a',s')
    Y = np.tensordot(Y, Q.conj(), axes=([3],[1]))  # (s,a,a',s'')
    return Y.transpose(1,0,2,3).reshape(dA*dS, dA*dS)

def petz_rec(rAB,rB,rBC,dA,dB,dC):
    evB,UB=np.linalg.eigh(rB); evB=np.maximum(evB,1e-12)
    Bm=(UB*evB**-0.5)@UB.T.conj()
    inner=np.kron(np.eye(dA),Bm)@rAB@np.kron(np.eye(dA),Bm)
    X=np.kron(inner.reshape(dA,dB,dA,dB), np.eye(dC)[None,None,:,:]).reshape(dA*dB*dC,-1) if False else np.kron(inner, np.eye(dC))
    # reorder: kron(inner(AB), I_C) already in A,B,C order
    evS,US=np.linalg.eigh(rBC); evS=np.clip(evS,0,None)
    S=(US*np.sqrt(evS))@US.T.conj()
    rt=sandwich_blocks(X,S,dA,dB*dC)
    ev,U=np.linalg.eigh((rt+rt.T.conj())/2); ev=np.clip(ev,1e-14,None)
    rt=(U*ev)@U.T.conj(); return rt/np.trace(rt).real

def rot_rec(rAB,rB,rBC,dA,dB,dC,tmax=6.0,nt=61):
    ts=np.linspace(-tmax,tmax,nt); wts=1.0/(np.cosh(np.pi*ts)+1.0)
    trap=np.ones(nt); trap[0]=trap[-1]=0.5; wts*=trap; wts/=wts.sum()
    evB,UB=np.linalg.eigh(rB); evB=np.maximum(evB,1e-12); lgB=np.log(evB)
    evS,US=np.linalg.eigh(rBC); evS=np.clip(evS,0,None)
    lgS=np.where(evS>0,np.log(np.maximum(evS,1e-300)),0.0); mask=evS>0
    acc=np.zeros((dA*dB*dC,)*2,dtype=complex)
    IA=np.eye(dA)
    half=[(i,ts[i],wts[i]) for i in range(nt)]
    # symmetry: real rho -> term(-t)=conj(term(t)); use t>=0 and double Re for t>0
    for i,t,wk in half:
        if t<0: continue
        zB=np.exp(-(1+1j*t)/2*lgB)
        PB=(UB*zB)@UB.T.conj()
        inner=np.kron(IA,PB)@rAB@np.kron(IA,PB.T.conj())
        X=np.kron(inner,np.eye(dC))
        zS=np.where(mask,np.exp((1+1j*t)/2*lgS),0.0)
        QS=(US*zS)@US.T.conj()
        term=sandwich_blocks(X,QS,dA,dB*dC)
        if t>0: acc+=2*wk*term.real + 0j  # term + conj(term)
        else: acc+=wk*term
    ev,U=np.linalg.eigh((acc+acc.T.conj())/2); ev=np.clip(ev.real,1e-14,None)
    rt=(U*ev)@U.T.conj(); return rt/np.trace(rt).real

def get_state(h,beta):
    sc=os.path.join(CACHE_DIR, f"rho_{h}_{beta}.npz")
    if os.path.exists(sc):
        z=np.load(sc); return z['rho'], float(z['S'])
    cache=os.path.join(CACHE_DIR, f"eigh_{h}.npz")
    if os.path.exists(cache):
        z=np.load(cache); evH,UH=z['e'],z['u']
    else:
        evH,UH=np.linalg.eigh(tfim_H(10,1.0,h)); np.savez(cache,e=evH,u=UH)
    wgt=np.exp(-beta*(evH-evH.min())); wgt/=wgt.sum()
    rho=(UH*wgt)@UH.T; S=float(-np.sum(wgt*np.log(np.maximum(wgt,1e-300))))
    np.savez(sc, rho=rho, S=S); return rho, S

def get_reduced(h,beta,w,rho,Srho):
    rc=os.path.join(CACHE_DIR, f"red_{h}_{beta}_{w}.npz")
    if os.path.exists(rc):
        z=np.load(rc); return z['rAB'],z['rB'],z['rBC'],float(z['I'])
    LA=2; dA,dB,dC=2**LA,2**w,2**(10-LA-w); dims=[dA,dB,dC]
    rAB=partial_trace(rho,dims,[0,1]); rBC=partial_trace(rho,dims,[1,2]); rB=partial_trace(rho,dims,[1])
    I=vn(rAB)+vn(rBC)-vn(rB)-Srho
    np.savez(rc,rAB=rAB,rB=rB,rBC=rBC,I=I); return rAB,rB,rBC,I

def run_combo(h,beta,methods,ws=range(1,7)):
    LA=2
    rho,Srho=get_state(h,beta)
    with open(os.path.join(CACHE_DIR, 'results.csv'),'a') as f:
        for w in ws:
            dA,dB,dC=2**LA,2**w,2**(10-LA-w)
            rAB,rB,rBC,I=get_reduced(h,beta,w,rho,Srho)
            for m in methods:
                rt=petz_rec(rAB,rB,rBC,dA,dB,dC) if m=='petz' else rot_rec(rAB,rB,rBC,dA,dB,dC)
                sv=np.linalg.svd(rho-rt,compute_uv=False)
                T=0.5*sv.sum(); HS=np.sqrt((sv**2).sum())
                f.write(f"{h},{beta},{w},{m},{I:.16e},{T:.16e},{HS:.16e}\n"); f.flush()
                print(f"{h} {beta} w={w} {m} I={I:.3e} T={T:.3e} [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__":
    h=float(sys.argv[1]); beta=float(sys.argv[2]); methods=sys.argv[3].split('+')
    ws=[int(x) for x in sys.argv[4].split(',')] if len(sys.argv)>4 else range(1,7)
    run_combo(h,beta,methods,ws)
