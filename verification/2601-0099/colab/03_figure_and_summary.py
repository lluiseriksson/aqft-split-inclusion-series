# 03_figure_and_summary.py
import os, json, math
import numpy as np
import matplotlib.pyplot as plt
from tenpy.tools import hdf5_io

PROJECT_DIR = "/content/drive/MyDrive/Ising_Project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_PATH = os.path.join(PROJECT_DIR, "moneyplot_CMI_xilocal.png")  # ROOT
TEX_PATH = os.path.join(PROJECT_DIR, "summary.tex")                # ROOT
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

JSONL_G = os.path.join(DATA_DIR, "cmi_h1p5_chi128_semiinf_STREAM.jsonl")
JSONL_N = os.path.join(DATA_DIR,
                       "cmi_from256_h1p005_chi512_semiinf_STREAM.jsonl")

CKPT_G = os.path.join(CKPT_DIR, "GS_iDMRG_h1p5_chi128.h5")
CKPT_N = os.path.join(CKPT_DIR, "GS_from256_h1p005_chi512.h5")

FLOOR = 1e-14
FIT_WINDOWS = [(6, 11), (6, 12), (7, 12)]  # main choice is first

def read_points(path):
    w_to_I = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("type") != "point":
                continue
            if "w" in o and "I" in o:
                w_to_I[int(o["w"])] = float(o["I"])
    ws = np.array(sorted(w_to_I.keys()), dtype=int)
    Is = np.array([w_to_I[w] for w in ws], dtype=float)
    return ws, Is

def xi_local(ws, Is):
    Is = np.maximum(np.asarray(Is, float), FLOOR)
    ws = np.asarray(ws, int)
    wloc, xiloc = [], []
    for i in range(len(ws) - 1):
        if ws[i+1] != ws[i] + 1:
            continue
        d = np.log(Is[i]) - np.log(Is[i+1])
        if d > 0:
            wloc.append(ws[i])
            xiloc.append(1.0/d)
    return np.array(wloc), np.array(xiloc)

def exp_fit(ws, Is, wmin, wmax):
    ws = np.asarray(ws, float)
    Is = np.asarray(Is, float)
    m = (ws >= wmin) & (ws <= wmax) & (Is > 10*FLOOR)
    ww = ws[m]
    yy = np.log(Is[m])
    if len(ww) < 3:
        return None
    A = np.vstack([np.ones_like(ww), -ww]).T
    logA, inv_xi = np.linalg.lstsq(A, yy, rcond=None)[0]
    if inv_xi <= 0:
        return None
    return {"wmin": int(wmin), "wmax": int(wmax), "xi": float(1.0/inv_xi),
            "logA": float(logA)}

def fit_curve(fit, wgrid):
    wgrid = np.asarray(wgrid, float)
    return np.exp(fit["logA"] - wgrid/fit["xi"])

def xi_range(fits):
    xs = [f["xi"] for f in fits if f is not None]
    return (min(xs), max(xs)) if xs else (None, None)

def load_psi(path):
    obj = hdf5_io.load(path)
    psi = obj["psi"] if isinstance(obj, dict) and "psi" in obj else obj
    psi.canonical_form()
    return psi

def chi_eff(psi):
    chi = getattr(psi, "chi", None)
    return int(np.max(np.asarray(chi))) if chi is not None else None

def xi_corr(psi):
    xi = psi.correlation_length2()
    return float(np.max(np.asarray(xi, dtype=float)))

def fmt(x, nd=3):
    if x is None:
        return "nan"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.{nd}f}"

# Load data
wg, Ig = read_points(JSONL_G)
wn, In = read_points(JSONL_N)

fits_g = [exp_fit(wg, Ig, a, b) for (a, b) in FIT_WINDOWS]
fits_n = [exp_fit(wn, In, a, b) for (a, b) in FIT_WINDOWS]
if fits_g[0] is None or fits_n[0] is None:
    raise RuntimeError("Main fit window failed; check JSONL coverage/values.")

wlg, xlg = xi_local(wg, Ig)
wln, xln = xi_local(wn, In)

# Figure (ROOT)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))

ax1.semilogy(wg, np.maximum(Ig, FLOOR), "s-", ms=4, lw=2.0,
             label="gapped h=1.5")
ax1.semilogy(wn, np.maximum(In, FLOOR), "o-", ms=4, lw=2.0,
             label="near-critical h=1.005 (from256$\\to$512)")

for fit, color in [(fits_g[0], "C1"), (fits_n[0], "C0")]:
    wgrid = np.arange(fit["wmin"], fit["wmax"] + 1)
    ax1.semilogy(wgrid, np.maximum(fit_curve(fit, wgrid), FLOOR), "--",
                 lw=2.0, color=color, alpha=0.85)

ax1.set_xlabel("w")
ax1.set_ylabel(r"$I(A:C\mid B(w))$ [nats]")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend()

ax2.plot(wlg, xlg, "s-", ms=4, lw=2.0, label="gapped")
ax2.plot(wln, xln, "o-", ms=4, lw=2.0, label="near-critical")
ax2.set_xlabel("w")
ax2.set_ylabel(r"$\xi_{\rm local}(w)=[\log I(w)-\log I(w+1)]^{-1}$")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=220)
plt.close(fig)
print("Wrote:", FIG_PATH)

# Load checkpoints for chi_eff and xi_corr
psi_g = load_psi(CKPT_G)
psi_n = load_psi(CKPT_N)

chi_g = chi_eff(psi_g)
chi_n = chi_eff(psi_n)

xi_g = xi_corr(psi_g)
xi_n = xi_corr(psi_n)

g_lo, g_hi = xi_range(fits_g)
n_lo, n_hi = xi_range(fits_n)

TABLE_TEX = rf"""
\begin{{table}}[H]
\centering
\caption{{\textbf{{Program A summary (semi-infinite geometry).}}
$\xi_{{\mathrm{{rec}}}}^{{(\mathrm{{early}})}}$ is extracted from an
exponential fit to $I(A:C\mid B(w))$ on a fixed window (main choice:
{FIT_WINDOWS[0]}).
Window sensitivity is reported across {FIT_WINDOWS}.
$\chi_{{\mathrm{{eff}}}}$ is the effective bond dimension after
canonicalization.
$\xi_{{\mathrm{{corr}}}}$ is the iMPS transfer-matrix correlation
length.}}
\label{{tab:programA}}
\sisetup{{round-mode=places,round-precision=3}}
\begin{{tabular}}{{l S[table-format=1.3] S[table-format=3.0] S[table-format=3.3] S[table-format=2.3]}}
\toprule
Regime & {{$h$}} & {{$\chi_{{\mathrm{{eff}}}}$}} & {{$\xi_{{\mathrm{{corr}}}}$}} & {{$\xi_{{\mathrm{{rec}}}}^{{(\mathrm{{early}})}}$}} \\
\midrule
Gapped & 1.500 & {chi_g} & {fmt(xi_g,3)} & {fmt(fits_g[0]["xi"],3)} \\
Near-critical (from256$\to$512) & 1.005 & {chi_n} & {fmt(xi_n,3)} & {fmt(fits_n[0]["xi"],3)} \\
\bottomrule
\end{{tabular}}

\vspace{{0.4em}}
\begin{{minipage}}{{0.94\linewidth}}
\small
\textbf{{Window sensitivity.}}
Gapped: $\xi_{{\mathrm{{rec}}}}^{{(\mathrm{{early}})}}\in[{fmt(g_lo,3)},{fmt(g_hi,3)}]$.
Near-critical: $\xi_{{\mathrm{{rec}}}}^{{(\mathrm{{early}})}}\in[{fmt(n_lo,3)},{fmt(n_hi,3)}]$.
\end{{minipage}}
\end{{table}}
""".lstrip()

with open(TEX_PATH, "w") as f:
    f.write(TABLE_TEX)
print("Wrote:", TEX_PATH)
