"""
coex_check.py — re-check pilot-3 coexistence runs (B | hexlat): did the B slab melt, or did only the classifier change?

Usage  python coex_check.py coex/B_T0.06 coex/B_T0.08 coex/B_T0.10 [--lam 1.93]
For each run: (1) from profiles.npz, time series of x_A, x_B, x_X averaged over the B half (bins that were
B-like at the start); (2) from the last snapshot, particles that started in the B slab (labels0 == 0):
pair-distance histogram and the fraction of their Delaunay triangles in each class with the temperature-adapted
tolerance and with a wide tolerance (tol_out = 0.45).  Writes coex_check.png and coex_check.json per run.
Interpretation: x_A ~ 0 throughout and B triangles recovered with the wide tolerance  ->  no melting
(classification effect).  x_A rising / broad disordered g(r)  ->  genuine melting.
"""
import argparse, glob, json, math, os
import numpy as np
from hcss_analysis import analyze, tol_out_auto


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--lam", type=float, default=1.93)
    ap.add_argument("--P", type=float, default=0.735)
    a = ap.parse_args()
    for run in a.runs:
        info = json.load(open(os.path.join(run, "run_info.json")))
        T = info["T"]
        p = np.load(os.path.join(run, "profiles.npz")); prof = p["profiles"].astype(float); sw = p["sweeps"]
        tot = np.maximum(prof.sum(2), 1e-12); xA = prof[:, :, 0] / tot; xB = prof[:, :, 1] / tot; xX = prof[:, :, 4] / tot
        Bbins = xA[0] < 0.05
        series = dict(sweeps=sw.tolist(), xA_B=xA[:, Bbins].mean(1).tolist(), xB_B=xB[:, Bbins].mean(1).tolist(),
                      xX_B=xX[:, Bbins].mean(1).tolist())
        snap = sorted(glob.glob(os.path.join(run, "snap_*.npz")))[-1]
        d = np.load(snap); pos, M, lab0 = d["pos"], d["box"], d["labels0"]
        sel = pos[lab0 == 0]
        res = {}
        for name, to in (("auto", tol_out_auto(T, a.P, a.lam)), ("wide", 0.45)):
            r = analyze(pos, M, lam=a.lam, tol_out=to, sk_nmax=0)
            res[name] = dict(tol_out=to, composition_whole_box=r["composition"])
        # pair distances among particles that started in the B slab
        inv = np.linalg.inv(M); dd = []
        for i in range(0, len(sel), max(1, len(sel) // 300)):
            v = sel - sel[i]; f = v @ inv.T; f -= np.round(f); v = f @ M.T
            r = np.linalg.norm(v, axis=1); dd += list(r[(r > 1e-9) & (r < 3.0)])
        h, e = np.histogram(dd, bins=120, range=(0, 3))
        out = dict(T=T, series_last=dict(xA_B=series["xA_B"][-1], xB_B=series["xB_B"][-1], xX_B=series["xX_B"][-1]),
                   series_first=dict(xA_B=series["xA_B"][0], xB_B=series["xB_B"][0], xX_B=series["xX_B"][0]),
                   tolerance=res, pair_hist=dict(edges=e.tolist(), counts=h.tolist()))
        json.dump(out, open(os.path.join(run, "coex_check.json"), "w"), indent=1)
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
            ax[0].plot(sw, series["xA_B"], label="x_A"); ax[0].plot(sw, series["xB_B"], label="x_B"); ax[0].plot(sw, series["xX_B"], label="x_X")
            ax[0].set_xlabel("sweep"); ax[0].set_title("B half"); ax[0].legend()
            ax[1].plot(0.5 * (e[1:] + e[:-1]), h); ax[1].axvline(1, ls="--", c="k", lw=.7); ax[1].axvline(a.lam, ls="--", c="r", lw=.7)
            ax[1].set_xlabel("r (particles that started in the B slab)")
            fig.suptitle(f"{os.path.basename(os.path.normpath(run))}"); fig.tight_layout()
            fig.savefig(os.path.join(run, "coex_check.png"), dpi=120); plt.close(fig)
        except ImportError:
            pass
        print(run, "B half first->last  x_A %.3f->%.3f  x_B %.3f->%.3f  x_X %.3f->%.3f" % (
            series["xA_B"][0], series["xA_B"][-1], series["xB_B"][0], series["xB_B"][-1], series["xX_B"][0], series["xX_B"][-1]))
        for k, v in res.items():
            c = v["composition_whole_box"]; print(f"   tol_out {v['tol_out']:.2f}: A {c['A']:.3f} B {c['B']:.3f} X {c['X']:.3f}")


if __name__ == "__main__":
    main()
