#!/usr/bin/env python3
r"""
reanalyze_0105.py -- Offline re-analysis for 2512.0105 v2 (prefix-path Bell
transport on ibm_fez). Works entirely from archived JSON artifacts in
results/0105/ -- no IBM account needed.

Canonical JSON schema expected (adapt the three load_* functions if your
archived files use different field names -- everything else is generic):

  rip_comparative_fit_v3_fixB.json:
    {"chains": {"P1": {"L": [...], "F_mean": [...], "F_sem": [...]}, "P2":..., "P3":...}}
  rip_strong_scaleup_v2.json:
    {"chains": [{"id": "...", "S_stat_corr": x, "F_mean_by_L": {"0":f,"2":f,"4":f,"6":f}}, ...]}
  rip_predictive_dynamic.json:
    {"LOW": [mu04 per chain...], "HIGH": [mu04 per chain...]}

Analyses (v2 additions to the paper):
  A) mu fit-variant table per chain: (1) Bfix = mean of two largest-L points
     [paper's v1 choice], (2) free-B 3-parameter fit, (3) Bfix = 1/4
     (depolarizing floor). Robustness = ordering mu1 < mu3 < mu2 preserved and
     minimal pairwise separation in combined sigma across variants.
  B) Scale-up: Spearman rho + two-sided permutation p (n_perm = 20000).
  C) Preregistered test: Delta = mean(HIGH) - mean(LOW), permutation p.
  D) Power: sigma_chain from scale-up mu04 dispersion; minimum detectable
     effect at one-sided alpha=0.05, power 80%: MDE = 2.486 * sigma * sqrt(2/k).
  E) Rate proxies: mu04 (L={0,4}) vs mu_allL (weighted log-linear fit of
     ln(1-F) over all measured L); Spearman between proxies.

Outputs: report to stdout + v2_numbers.tex + table_fit_variants.tex
(LaTeX macros/table consumed by induced paper tex via \input).

--demo runs the full pipeline on synthetic data with known ground truth
(true mu = 0.2224/0.5398/0.3014, B = 0.25; scale-up with zero association)
to validate the machinery end to end.
"""
import json, sys, argparse
from pathlib import Path
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(20260705)

# ---------------- loaders (adapt here if field names differ) ---------------
def load_comparative(path):
    d = json.load(open(path))
    out = {}
    for k, v in d["chains"].items():
        out[k] = (np.asarray(v["L"], float), np.asarray(v["F_mean"], float),
                  np.asarray(v["F_sem"], float))
    return out

def load_scaleup(path):
    d = json.load(open(path))
    S, FbyL = [], []
    for c in d["chains"]:
        S.append(float(c["S_stat_corr"]))
        FbyL.append({int(k): float(v) for k, v in c["F_mean_by_L"].items()})
    return np.asarray(S), FbyL

def load_prereg(path):
    d = json.load(open(path))
    return np.asarray(d["LOW"], float), np.asarray(d["HIGH"], float)

# ---------------- fitting ---------------------------------------------------
def fit_mu(L, F, sem, mode):
    """Return (mu, sigma_mu). Modes: 'fixB2pts' (paper), 'freeB', 'B025'."""
    w = 1.0 / np.maximum(sem, 1e-6)**2
    if mode == "freeB":
        f = lambda L, F0, mu, B: B + (F0 - B) * np.exp(-mu * L)
        p0 = [F.max(), 0.3, max(F.min(), 0.25)]
        try:
            p, cov = optimize.curve_fit(f, L, F, p0=p0, sigma=sem,
                                        absolute_sigma=True, maxfev=20000)
            return p[1], float(np.sqrt(cov[1, 1]))
        except Exception:
            return np.nan, np.nan
    B = {"fixB2pts": float(np.mean(F[np.argsort(L)[-2:]])), "B025": 0.25}[mode]
    f = lambda L, F0, mu: B + (F0 - B) * np.exp(-mu * L)
    p, cov = optimize.curve_fit(f, L, F, p0=[F.max(), 0.3], sigma=sem,
                                absolute_sigma=True, maxfev=20000)
    return p[1], float(np.sqrt(cov[1, 1]))

def mu04(FbyL):
    return 0.25 * np.log((1 - FbyL[4]) / (1 - FbyL[0]))

def mu_allL(FbyL):
    Ls = np.array(sorted(FbyL)); y = np.log(1 - np.array([FbyL[k] for k in Ls]))
    A = np.vstack([Ls, np.ones_like(Ls)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])

# ---------------- statistics ------------------------------------------------
def spearman_perm(x, y, n_perm=20000):
    rho = stats.spearmanr(x, y).statistic
    null = np.array([stats.spearmanr(x, rng.permutation(y)).statistic
                     for _ in range(n_perm)])
    return rho, float(np.mean(np.abs(null) >= abs(rho)))

def group_perm(low, high, n_perm=20000):
    obs = high.mean() - low.mean()
    pool = np.concatenate([low, high]); k = len(low)
    null = []
    for _ in range(n_perm):
        rng.shuffle(pool)
        null.append(pool[k:].mean() - pool[:k].mean())
    null = np.asarray(null)
    return obs, float(np.mean(null >= obs)), float(np.mean(np.abs(null) >= abs(obs)))

def mde(sigma, k=5, alpha=0.05, power=0.80):
    return (stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * sigma * np.sqrt(2 / k)

# ---------------- demo data -------------------------------------------------
def demo_data():
    true = {"P1": 0.2224, "P2": 0.5398, "P3": 0.3014}
    comp = {}
    for k, mu in true.items():
        L = np.arange(0, 13, 2, dtype=float)
        F = 0.25 + 0.72 * np.exp(-mu * L)
        sem = np.full_like(L, 0.012)
        comp[k] = (L, F + rng.normal(0, sem), sem)
    S = rng.normal(0.015, 0.005, 18)
    FbyL = []
    for _ in range(18):
        m = rng.normal(0.33, 0.06)          # no association with S
        FbyL.append({Lv: float(0.25 + 0.72 * np.exp(-m * Lv)
                     + rng.normal(0, 0.01)) for Lv in (0, 2, 4, 6)})
    low = rng.normal(0.335, 0.05, 5); high = rng.normal(0.333, 0.05, 5)
    return comp, (S, FbyL), (low, high)

# ---------------- main -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../results/0105")
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if a.demo:
        print("[demo mode: synthetic data, known ground truth]")
        comp, (S, FbyL), (low, high) = demo_data()
    else:
        comp = load_comparative(f"{a.data_dir}/rip_comparative_fit_v3_fixB.json")
        S, FbyL = load_scaleup(f"{a.data_dir}/rip_strong_scaleup_v2.json")
        low, high = load_prereg(f"{a.data_dir}/rip_predictive_dynamic.json")

    # A) fit variants
    modes = ["fixB2pts", "freeB", "B025"]
    res = {k: {m: fit_mu(*comp[k], m) for m in modes} for k in sorted(comp)}
    print("\n== A) mu fit variants ==")
    rows = []
    for k in sorted(res):
        vals = "  ".join(f"{m}: {res[k][m][0]:.4f}+-{res[k][m][1]:.4f}" for m in modes)
        print(f"  {k}: {vals}")
        rows.append(f"{k} & " + " & ".join(
            f"${res[k][m][0]:.4f} \\pm {res[k][m][1]:.4f}$" for m in modes) + r" \\")
    order_ok = all(res["P1"][m][0] < res["P3"][m][0] < res["P2"][m][0] for m in modes)
    seps = [(res["P2"][m][0] - res[o][m][0]) /
            np.hypot(res["P2"][m][1], res[o][m][1])
            for m in modes for o in ("P1", "P3")]
    print(f"  ordering P1<P3<P2 in all variants: {order_ok}; "
          f"min pairwise separation: {min(seps):.1f} sigma")

    # B) scale-up
    mu4 = np.array([mu04(f) for f in FbyL]); muA = np.array([mu_allL(f) for f in FbyL])
    rho4, p4 = spearman_perm(S, mu4); rhoA, pA = spearman_perm(S, muA)
    rho_proxy = stats.spearmanr(mu4, muA).statistic
    print(f"\n== B) scale-up ==\n  S vs mu04:  rho={rho4:+.3f} p={p4:.3f}"
          f"\n  S vs muAll: rho={rhoA:+.3f} p={pA:.3f}"
          f"\n  proxy agreement mu04 vs muAll: rho={rho_proxy:+.3f}")

    # C) prereg
    d_obs, p1, p2 = group_perm(low.copy(), high.copy())
    print(f"\n== C) preregistered test ==\n  Delta={d_obs:+.4f}  "
          f"p(1-sided)={p1:.3f}  p(2-sided)={p2:.3f}")

    # D) power
    sig = float(np.std(mu4, ddof=1)); m80 = mde(sig)
    print(f"\n== D) power ==\n  sigma_chain(mu04, scale-up)={sig:.4f}"
          f"\n  MDE (alpha=.05 one-sided, 80% power, k=5): {m80:.4f}"
          f"  [obs |Delta|={abs(d_obs):.4f}]")

    # outputs
    table_path = output_dir / "table_fit_variants.tex"
    numbers_path = output_dir / "v2_numbers.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(numbers_path, "w") as f:
        f.write(f"""\\newcommand{{\\VarOrderOK}}{{{'yes' if order_ok else 'NO'}}}
\\newcommand{{\\VarMinSep}}{{{min(seps):.1f}}}
\\newcommand{{\\SigmaChain}}{{{sig:.4f}}}
\\newcommand{{\\MDEeighty}}{{{m80:.4f}}}
\\newcommand{{\\RhoAllL}}{{{rhoA:+.3f}}}
\\newcommand{{\\PAllL}}{{{pA:.3f}}}
\\newcommand{{\\RhoProxy}}{{{rho_proxy:+.3f}}}
""")
    print(f"\nWrote {table_path.as_posix()}, {numbers_path.as_posix()}")

if __name__ == "__main__":
    main()
