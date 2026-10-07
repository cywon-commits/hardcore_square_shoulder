"""
op_coex.py — (analysis 2) the A -> tiling transformation (and the frozen B | tiling interface) seen through the order parameters.
For each coexistence run (pilot 3: coex/A_T*, coex/B_T*; snapshots with pos, box, labels0) and each snapshot:
profiles along x of eta (tile counts, exact), psi4, psi6, psi12 (bond texture), and the global E (alpha, beta) from the lift.
Prediction: the tiling that grows into the A slab is triangle-rich (eta < 0), i.e. it must carry charge-6 (beta-type) phason
strain: psi6 rises in the product region while psi4 stays small; and globally det E = eta, |beta|^2 - |alpha|^2 = -eta.
Usage  python op_coex.py coex/A_T0.06 coex/A_T0.08 ... --out coex_op.json      (writes <run>/op_profiles.png)
"""
import argparse, glob, json, os, math, numpy as np
import op
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--nbins", type=int, default=24)
    ap.add_argument("--P", type=float, default=0.735); ap.add_argument("--lam", type=float, default=1.93); ap.add_argument("--out", default="coex_op.json")
    a = ap.parse_args(); res = {}
    for run in a.runs:
        snaps = sorted(glob.glob(os.path.join(run, "snap_*.npz")))
        info = json.load(open(os.path.join(run, "run_info.json"))) if os.path.exists(os.path.join(run, "run_info.json")) else {}
        T = info.get("T", 0.06); tol_out = 0.10 + 2.0 * T / (a.P * a.lam)
        series = []
        for f in snaps:
            z = np.load(f); P, M = z["pos"], z["box"]
            o, (L, lf) = op.order_parameters(P, M, lam=a.lam, tol_in=0.06, tol_out=tol_out)
            prof = op.local_profiles(P, M, L, nbins=a.nbins, lam=a.lam, tol_in=0.06, tol_out=tol_out)
            series.append(dict(snap=os.path.basename(f), eta=o["eta"], detE=o.get("detE"), abs_alpha=o.get("abs_alpha"), abs_beta=o.get("abs_beta"),
                               psi4=o["psi4"], psi6=o["psi6"], psi12=o["psi12"], x_other=o["x_other"], defects=o["n_defect_edges"], profile=prof))
        res[run] = series
        if series:
            s0, s1 = series[0], series[-1]
            print(f"{run}: eta {s0['eta']:+.3f} -> {s1['eta']:+.3f}   detE {s1['detE']}   |alpha| {s1['abs_alpha']}  |beta| {s1['abs_beta']}   "
                  f"psi4 {s1['psi4']:.3f} psi6 {s1['psi6']:.3f}   defects {s1['defects']}")
            try:
                import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
                for k, key in enumerate(("eta", "psi6", "psi4")):
                    Z = np.array([s["profile"][key] for s in series], float)
                    im = ax[k].imshow(Z.T, aspect="auto", origin="lower", cmap="RdBu_r" if key == "eta" else "viridis",
                                      vmin=-1 if key == "eta" else 0, vmax=1 if key == "eta" else 0.6)
                    ax[k].set_title(key + "(x, t)"); ax[k].set_xlabel("snapshot"); ax[k].set_ylabel("x bin"); fig.colorbar(im, ax=ax[k])
                fig.suptitle(os.path.basename(os.path.normpath(run))); fig.tight_layout(); fig.savefig(os.path.join(run, "op_profiles.png"), dpi=120); plt.close(fig)
            except Exception as e:
                print("   plot failed:", e)
    json.dump(res, open(a.out, "w"), default=float)
if __name__ == "__main__":
    main()
