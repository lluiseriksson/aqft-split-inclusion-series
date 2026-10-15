"""Figure for 2512.0072 v2: exact rate-inheritance evidence replacing the
surrogate decay-curve proxy of v1 (which had no critical control and belongs
to the proxy class withdrawn in 2512.0064 v2).

Loads the exact Liouvillian-rapidity sweeps of verification/0064/modelA.py
(run it first if modelA_results.npy is absent) and plots kappa(eps) for
sub-gap probes (exponential suppression, slopes 2q(omega_b)), an in-band
probe and the critical chain (no suppression) -- the RIP weak/strong forms
and their failure modes in one exact panel.
"""
import numpy as np, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MA = HERE.parent / "0064" / "modelA_results.npy"
if not MA.exists():
    subprocess.run([sys.executable, str(HERE.parent / "0064" / "modelA.py")],
                   cwd=str(HERE.parent / "0064"), check=True)
res = np.load(MA, allow_pickle=True).item()
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4.2))
styles = {"gap_w0.0":  ("o-", r"sub-gap $\omega_0=0$ ($2q(\omega_b)=0.811$)"),
          "gap_w0.5":  ("s-", r"sub-gap $\omega_0=0.5$ ($2q(\omega_b)=0.692$)"),
          "gap_w0.9":  ("^-", r"sub-gap $\omega_0=0.9$ ($2q(\omega_b)=0.322$)"),
          "gap_w2.0_inband": ("x--", r"in-band $\omega_0=2$ (no suppression)"),
          "crit_w0.5": ("+--", r"critical $\mu=2$ (no gap, no suppression)")}
for k, (st, lab) in styles.items():
    d = res[k]
    eps = np.array(d["eps"]); kap = np.array(d["kappa"])
    ok = kap > 1e-13
    ax.semilogy(eps[ok], kap[ok], st, label=lab, ms=4)
ax.set_xlabel(r"buffer width $\epsilon$ (sites)")
ax.set_ylabel(r"exact probe rapidity $\kappa(\epsilon)$")
ax.set_title("Exact rate inheritance (model of ai.viXra:2512.0064 v2)\n"
             "gapped buffer $\\mu=3$: exponential; in-band / critical: none", fontsize=9)
ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout()
outputs = [
    HERE / "fig_rip_exact.pdf",
    ROOT / "results" / "0072" / "fig_rip_exact.pdf",
    ROOT / "papers" / "0072-rip-principle" / "fig_rip_exact.pdf",
]
for out in outputs:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
print("fig_rip_exact.pdf written to verification/0072, results/0072, papers/0072-rip-principle; slopes (fit vs analytic):")
for k in ["gap_w0.0", "gap_w0.5", "gap_w0.9"]:
    d = res[k]
    print(f"  {k}: fit={d['slope_fit']:.4f}  2q(omega_b)={d['slope_analytic_wb']:.4f}")
