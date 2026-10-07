"""
burgers_analysis.py — pilot 10: topological defects (4D Burgers vectors in Z[zeta_12]) across melting.

For every run directory (snapshots snap_*.npz with pos, box; run_info.json with T) and its last --last snapshots:
  * defects per particle (faces of the shoulder-bond graph with non-zero holonomy), integer-norm self-check,
    total Burgers vector of the box (must be 0 on a torus);
  * classification: units zeta^k (1 - zeta)^n with n = 0, 1, 2, 3 (|b_par| = 0.518^n, |b_perp| = 1.932^n, edge units)
    versus non-units (norm > 1);
  * binding: fraction of defects whose nearest defect cancels their Burgers vector exactly;
  * screening: <|sum b_perp|^2> in windows of side R; exponent from a log-log fit (about 1 for bound pairs, 2 for free).
Predictions: (1) the defects are units with n = 1 or 2 (phason-dominated, b_par = core vector or smaller);
(2) melting = unbinding: binding fraction ~ 1 below T_m, drops at T_m; screening exponent rises from ~1 to ~2.
Usage  python burgers_analysis.py melt9/T*_hexlat melt9/T*_dodeca --last 3 --out burgers.json
"""
import argparse, glob, json, os, math, numpy as np
import burgers as BG
UNITS = {}
for n in range(4):
    b = BG.BASIS[0].copy()
    for _ in range(n):
        nb = np.zeros(4, np.int64)
        for j in range(4):
            nb += int(b[j]) * (BG.BASIS[j] - BG.BASIS[(j + 1) % 12])
        b = nb
    UNITS[BG.canon(b)] = n
def classify(b):
    o = BG.canon(b)
    if o in UNITS: return f"n={UNITS[o]}"
    o2 = BG.canon(-np.array(b))
    if o2 in UNITS: return f"n={UNITS[o2]}"
    return "non-unit" if round(BG.norm(b)) > 1 else "unit-other"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--P", type=float, default=0.735); ap.add_argument("--lam", type=float, default=1.93); ap.add_argument("--out", default="burgers.json")
    a = ap.parse_args(); rows = []
    for run in a.runs:
        info = json.load(open(os.path.join(run, "run_info.json"))) if os.path.exists(os.path.join(run, "run_info.json")) else {}
        T = info.get("T", float("nan")); tol_out = 0.10 + 2.0 * T / (a.P * a.lam)
        for f in sorted(glob.glob(os.path.join(run, "snap_*.npz")))[-a.last:]:
            z = np.load(f); P, M = z["pos"], z["box"]
            st = BG.defect_stats(P, M, a.lam, 0.06, tol_out)
            cls = {}
            for d in st["defects"]:
                c = classify(d["b"]); cls[c] = cls.get(c, 0) + 1
            nb = np.array(st["norms"]); nonint = int(np.sum(np.abs(nb - np.round(nb)) > 1e-6)) if len(nb) else 0
            sc = BG.screening(st["defects"], M, Rs=(2, 3, 4, 6, 8))
            expo = float("nan")
            good = [(r, q) for r, p, q in sc if q > 0]
            if len(good) >= 3:
                expo = float(np.polyfit(np.log([g[0] for g in good]), np.log([g[1] for g in good]), 1)[0])
            rows.append(dict(run=run, snap=os.path.basename(f), T=T, N=len(P), defects_per_N=st["n_defects"] / len(P), classes=cls,
                             nonint_norms=nonint, total_b=st["total_b"], frac_paired=st["frac_paired_nn"], screening=sc, screen_exponent=expo))
    rows.sort(key=lambda r: (r["run"].split("_")[-1], r["T"], r["snap"]))
    print(f"{'run':22s} {'T':>6s} {'def/N':>7s} {'n=0':>4s} {'n=1':>4s} {'n=2':>4s} {'n=3':>4s} {'other':>5s} {'paired':>6s} {'expo':>5s} {'nonint':>6s} total_b")
    for r in rows:
        c = r["classes"]; oth = sum(v for k, v in c.items() if k not in ("n=0", "n=1", "n=2", "n=3"))
        print(f"{r['run'][-22:]:22s} {r['T']:6.3f} {r['defects_per_N']:7.4f} {c.get('n=0',0):4d} {c.get('n=1',0):4d} {c.get('n=2',0):4d} {c.get('n=3',0):4d} {oth:5d} "
              f"{r['frac_paired']:6.2f} {r['screen_exponent']:5.2f} {r['nonint_norms']:6d} {r['total_b']}")
    json.dump(rows, open(a.out, "w"), indent=1, default=float)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for start, mk in (("hexlat", "o"), ("dodeca", "s"), ("rows", "^"), ("A", "v"), ("B", "D")):
            sel = [r for r in rows if r["run"].endswith(start)]
            if not sel: continue
            Ts = sorted({r["T"] for r in sel})
            f = lambda key: [np.mean([r[key] for r in sel if r["T"] == t]) for t in Ts]
            ax[0].plot(Ts, f("defects_per_N"), mk + "-", label=start); ax[1].plot(Ts, f("frac_paired"), mk + "-", label=start)
            ax[2].plot(Ts, f("screen_exponent"), mk + "-", label=start)
        ax[0].set_ylabel("defects per particle"); ax[1].set_ylabel("fraction in cancelling NN pairs"); ax[2].set_ylabel("screening exponent")
        for x in ax: x.set_xlabel("T*"); x.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=130)
    except Exception as e:
        print("plot failed:", e)
if __name__ == "__main__":
    main()
