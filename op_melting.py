"""
op_melting.py — (analysis 4) order parameters across melting (pilot-2 temperature scan runs2/T*_*).
For the last snapshots of every run: eta, psi6, psi12 (global), the contact-tile fraction 1 - x_other, the orientational
correlation g12(r) = <psi12_i psi12_j*> of per-particle psi12 (decay at r ~ 10), and the translational order S(k_peak)/N.
Symmetry expectation in 2D: translational order is only quasi-long-ranged and should be lost first; the 12-fold orientational
order (long-ranged) may survive in an intermediate window (a 'dodecatic' analogue of the hexatic phase).
Usage  python op_melting.py runs2/T0.04_rows runs2/T0.06_rows ... --last 2 --out melting_op.json
"""
import argparse, glob, json, os, math, cmath, numpy as np
from scipy.spatial import cKDTree
import op
def local_psi(P, L, n=12):
    acc = np.zeros(len(P), complex); c = np.zeros(len(P))
    for (i, j, k) in L:
        acc[i] += cmath.exp(1j * n * math.pi * k / 6); c[i] += 1
    return acc / np.maximum(c, 1)
def g_corr(P, M, f, rmax=14.0, nb=28, nsample=500):
    inv = np.linalg.inv(M); rng = np.random.default_rng(0); idx = rng.choice(len(P), min(len(P), nsample), replace=False)
    num = np.zeros(nb); den = np.zeros(nb)
    for i in idx:
        d = P - P[i]; fr = d @ inv.T; fr -= np.round(fr); d = fr @ M.T; r = np.linalg.norm(d, axis=1)
        b = (r / rmax * nb).astype(int); m = (b < nb) & (r > 1e-9)
        np.add.at(num, b[m], (f[i] * np.conj(f[m])).real); np.add.at(den, b[m], 1)
    return (np.arange(nb) + 0.5) * rmax / nb, num / np.maximum(den, 1)
def s_peak(P, M, kmax=5.0):
    rec = 2 * math.pi * np.linalg.inv(M).T; nmax = int(math.ceil(kmax / min(np.linalg.norm(rec[:, 0]), np.linalg.norm(rec[:, 1])))) + 1
    m = np.arange(-nmax, nmax + 1); A, B = np.meshgrid(m, m); K = np.stack([A.ravel(), B.ravel()], 1) @ rec.T
    kk = np.linalg.norm(K, axis=1); K = K[(kk > 2.0) & (kk < kmax)]
    S = np.abs(np.exp(1j * (P @ K.T)).sum(0)) ** 2 / len(P)
    return float(S.max() / len(P))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--last", type=int, default=2)
    ap.add_argument("--P", type=float, default=0.735); ap.add_argument("--lam", type=float, default=1.93); ap.add_argument("--out", default="melting_op.json")
    a = ap.parse_args(); rows = []
    for run in a.runs:
        info = json.load(open(os.path.join(run, "run_info.json"))) if os.path.exists(os.path.join(run, "run_info.json")) else {}
        T = info.get("T", float("nan")); tol_out = 0.10 + 2.0 * T / (a.P * a.lam) if T == T else 0.25
        for f in sorted(glob.glob(os.path.join(run, "snap_*.npz")))[-a.last:]:
            z = np.load(f); P, M = z["pos"], z["box"]
            o, (L, lf) = op.order_parameters(P, M, lam=a.lam, tol_in=0.06, tol_out=tol_out)
            r, g = g_corr(P, M, local_psi(P, L, 12))
            rows.append(dict(run=run, snap=os.path.basename(f), T=T, eta=o["eta"], psi6=o["psi6"], psi12=o["psi12"], contact=1 - o["x_other"],
                             g12_r10=float(g[np.argmin(np.abs(r - 10))]), S_peak_over_N=s_peak(P, M), defects=o["n_defect_edges"]))
    rows.sort(key=lambda x: (x["run"].split("_")[-1], x["T"]))
    print(f"{'run':24s} {'T':>5s} {'contact':>7s} {'eta':>7s} {'psi6':>6s} {'psi12':>6s} {'g12(10)':>7s} {'S/N':>7s} {'defects':>7s}")
    for x in rows:
        print(f"{x['run'][-24:]:24s} {x['T']:5.2f} {x['contact']:7.3f} {x['eta']:+7.3f} {x['psi6']:6.3f} {x['psi12']:6.3f} {x['g12_r10']:7.3f} {x['S_peak_over_N']:7.4f} {x['defects']:7d}")
    json.dump(rows, open(a.out, "w"), indent=1, default=float)
if __name__ == "__main__":
    main()
