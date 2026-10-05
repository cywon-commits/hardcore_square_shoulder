"""
reanalyze.py — structural re-analysis of saved snapshots (pilot 1 or 2 run directories).

Usage
  python reanalyze.py runs/T0.15_rows runs/T0.15_A runs/T0.15_B [--lam 1.93] [--last 5]

For the last --last snapshots of each run:
  * g(r) up to r = 4 (periodic), with markers at r = 1 and r = lam
  * tile composition for tol_out in {0.15, 0.25, 0.35} and tol_in in {0.06, 0.10}  (tolerance sensitivity)
  * bond-orientational correlation g6(r) = <psi6(0) psi6*(r)> from Delaunay neighbours, and its decay
    (fluid: exponential; hexatic: algebraic; solid: plateau)
Writes reanalysis.json and reanalysis.png in each run directory and prints a summary.
"""
import argparse, glob, json, math, os
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from hcss_analysis import analyze, _periodic_points, _wrap


def gr(pos, M, rmax=4.0, nb=200):
    P, oid, img = _periodic_points(_wrap(pos, M), M)
    tree = cKDTree(P); N = len(pos); area = abs(np.linalg.det(M))
    cen = cKDTree(_wrap(pos, M))
    d = cen.sparse_distance_matrix(tree, rmax, output_type="ndarray")["v"]
    d = d[d > 1e-9]
    h, e = np.histogram(d, bins=nb, range=(0, rmax))
    r = 0.5 * (e[1:] + e[:-1]); shell = math.pi * (e[1:] ** 2 - e[:-1] ** 2)
    return r, h / (N * (N / area) * shell)


def psi6_local(pos, M):
    P, oid, img = _periodic_points(_wrap(pos, M), M)
    tri = Delaunay(P); N = len(pos)
    acc = np.zeros(N, complex); cnt = np.zeros(N)
    indptr, nbr = tri.vertex_neighbor_vertices
    for v in np.where(img == 0)[0]:
        for w in nbr[indptr[v]:indptr[v + 1]]:
            d = P[w] - P[v]
            acc[oid[v]] += np.exp(6j * math.atan2(d[1], d[0])); cnt[oid[v]] += 1
    return acc / np.maximum(cnt, 1)


def g6(pos, M, rmax=12.0, nb=48):
    psi = psi6_local(pos, M)
    W = _wrap(pos, M); inv = np.linalg.inv(M)
    N = len(pos); idx = np.random.default_rng(0).choice(N, min(N, 600), replace=False)
    num = np.zeros(nb); den = np.zeros(nb)
    for i in idx:
        d = W - W[i]; f = d @ inv.T; f -= np.round(f); d = f @ M.T
        r = np.linalg.norm(d, axis=1)
        k = (r / rmax * nb).astype(int); m = (k < nb) & (r > 1e-9)
        np.add.at(num, k[m], (psi[i] * np.conj(psi[m])).real)
        np.add.at(den, k[m], 1)
    rr = (np.arange(nb) + 0.5) * rmax / nb
    return rr, num / np.maximum(den, 1), float(np.mean(np.abs(psi) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--lam", type=float, default=1.93)
    ap.add_argument("--last", type=int, default=5)
    a = ap.parse_args()
    for run in a.runs:
        snaps = sorted(glob.glob(os.path.join(run, "snap_*.npz")))[-a.last:]
        if not snaps:
            print(run, "no snapshots"); continue
        G = []; tol = {}; G6 = []
        for sp in snaps:
            d = np.load(sp); pos, M = d["pos"], d["box"]
            r, g = gr(pos, M); G.append(g)
            for to in (0.15, 0.25, 0.35):
                for ti in (0.06, 0.10):
                    res = analyze(pos, M, lam=a.lam, tol_in=ti, tol_out=to, sk_nmax=0)
                    tol.setdefault(f"tol_in={ti},tol_out={to}", []).append(res["composition"])
            rr, c6, _ = g6(pos, M); G6.append(c6)
        g = np.mean(G, 0); c6 = np.mean(G6, 0)
        tolsum = {k: {t: float(np.mean([c[t] for c in v])) for t in "ABCDX"} for k, v in tol.items()}
        # first minimum after the core peak and the shoulder peak, for choosing tolerances
        out = dict(snapshots=[os.path.basename(x) for x in snaps], gr_r=r.tolist(), gr=g.tolist(),
                   g6_r=rr.tolist(), g6=c6.tolist(), tolerance_sensitivity=tolsum)
        json.dump(out, open(os.path.join(run, "reanalysis.json"), "w"), indent=1)
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
            ax[0].plot(r, g); ax[0].axvline(1, ls="--", c="k", lw=.7); ax[0].axvline(a.lam, ls="--", c="r", lw=.7)
            ax[0].set_xlabel("r"); ax[0].set_ylabel("g(r)")
            ax[1].semilogy(rr, np.abs(c6) + 1e-6); ax[1].set_xlabel("r"); ax[1].set_ylabel("|g6(r)|")
            fig.suptitle(os.path.basename(os.path.normpath(run))); fig.tight_layout()
            fig.savefig(os.path.join(run, "reanalysis.png"), dpi=120); plt.close(fig)
        except ImportError:
            pass
        print(run)
        for k, v in tolsum.items():
            print(f"   {k:26s}  A {v['A']:.3f}  B {v['B']:.3f}  C {v['C']:.3f}  X {v['X']:.3f}")
        print(f"   g6 at r~2, 6, 11:  {c6[8]:.3f} {c6[24]:.3f} {c6[44]:.3f}")


if __name__ == "__main__":
    main()
