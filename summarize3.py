"""
summarize3.py — summary of pilot-3 outputs.

  python summarize3.py --coex coex/* --fe fe/*.json
Coexistence: initial vs final (last 20 %) number of tiling / crystal bins, trend (bins per 1e5 sweeps), plot of
the x_A profile vs time (coex_*.png).
Free energies: table of beta*g per structure and T, with the tiling value beta*g_hexlat - ln2/3; differences
relative to the lowest; Gibbs-Duhem check between two temperatures: (beta g)(T2) - (beta g)(T1) vs
h_mean * (beta2 - beta1) for each structure (should agree within errors if the integration is consistent).
"""
import argparse, glob, json, math, os
import numpy as np


def coex(run):
    d = np.genfromtxt(os.path.join(run, "timeseries.csv"), delimiter=",", names=True)
    sw = d["sweep"]; k = max(1, len(sw) // 5)
    res = {}
    for col in ("bins_tiling", "bins_B", "bins_A", "bins_other"):
        x = d[col]
        slope = np.polyfit(sw, x, 1)[0] * 1e5 if len(sw) > 2 else float("nan")
        res[col] = dict(first=float(x[:k].mean()), last=float(x[-k:].mean()), trend_per_1e5=float(slope))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        p = np.load(os.path.join(run, "profiles.npz"))
        prof = p["profiles"]; tot = np.maximum(prof.sum(2), 1); xA = prof[:, :, 0] / tot
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(xA.T, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis",
                       extent=[p["sweeps"][0], p["sweeps"][-1], 0, 1])
        ax.set_xlabel("sweep"); ax.set_ylabel("x / Lx"); fig.colorbar(im, label="x_A (tiling 1/3, B 0, A 1)")
        ax.set_title(os.path.basename(os.path.normpath(run))); fig.tight_layout()
        fig.savefig(os.path.join(run, "coex_profile.png"), dpi=120); plt.close(fig)
    except Exception as e:
        res["plot_error"] = str(e)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--coex", nargs="*", default=[]); ap.add_argument("--fe", nargs="*", default=[])
    a = ap.parse_args()
    for run in a.coex:
        r = coex(run)
        print(run)
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"   {k:12s} first {v['first']:6.2f}  last {v['last']:6.2f}  trend/1e5 {v['trend_per_1e5']:+7.2f}")
    if a.fe:
        rows = [json.load(open(f)) for f in a.fe]
        byT = {}
        for r in rows:
            g = r.get("beta_g_tiling", r["beta_g"]); name = "tiling(hexlat-ln2/3)" if "beta_g_tiling" in r else r["kind"]
            byT.setdefault(r["T"], []).append((name + f"[seed {r['seed']}]", g, r["beta_g_err"], r["npt"]["h"], r["kind"], r["beta_g"]))
        for T, lst in sorted(byT.items()):
            gmin = min(x[1] for x in lst)
            print(f"T = {T}")
            for name, g, e, h, kind, graw in sorted(lst, key=lambda x: x[1]):
                print(f"   {name:30s} beta*g = {g:10.5f} +- {e:.5f}   delta = {g - gmin:+.5f}   T*delta = {T * (g - gmin):+.6f} eps   h = {h:.5f}")
        Ts = sorted(byT)
        if len(Ts) >= 2:
            T1, T2 = Ts[0], Ts[-1]
            print(f"Gibbs-Duhem check between T = {T1} and {T2}: (beta g)_2 - (beta g)_1  vs  h_mean (beta_2 - beta_1)")
            for kind in sorted({x[4] for x in byT[T1]} & {x[4] for x in byT[T2]}):
                g1 = np.mean([x[5] for x in byT[T1] if x[4] == kind]); g2 = np.mean([x[5] for x in byT[T2] if x[4] == kind])
                h1 = np.mean([x[3] for x in byT[T1] if x[4] == kind]); h2 = np.mean([x[3] for x in byT[T2] if x[4] == kind])
                print(f"   {kind:8s} {g2 - g1:+.4f}  vs  {0.5 * (h1 + h2) * (1 / T2 - 1 / T1):+.4f}")


if __name__ == "__main__":
    main()
