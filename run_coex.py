"""
run_coex.py — direct-coexistence (interface) runs: crystal slab | random-tiling slab (hexlat).

Example
  python run_coex.py --left B --T 0.08 --P 0.735 --sweeps 200000 --flip-per-sweep 0.05 --out coex/B_T0.08

Geometry: slabs periodic in y with the common period ny*L (L = lam*scale; see hcss_mc.slab_state);
two interfaces normal to x.  Box moves change a and c only (no shear) so the slab orientation is kept.
Outputs: timeseries.csv (sweep, density, e/N, number of bins classified as tiling / B / A / other,
flip acceptances), profiles.npz (tile counts per x-bin vs time), snapshots.
Bin classification (tol_out = tol_out_auto(T,P,lam)):  A-crystal: x_A > 0.8;  B-crystal: x_A < 0.05 and x_B > 0.7;
tiling: 0.15 < x_A < 0.55 and x_X < 0.3;  other: everything else (interface, disorder).
"""
import argparse, csv, json, math, os, time
import numpy as np
import hcss_mc as mc
from hcss_analysis import tile_profile, tol_out_auto


def classify(prof):
    tot = np.maximum(prof.sum(1), 1)
    xA, xB, xX = prof[:, 0] / tot, prof[:, 1] / tot, prof[:, 4] / tot
    # classification by x_A only (pilot 3 showed x_B can drop at high T through X-classified triangles while
    # x_A stays exactly 0, i.e. without melting); x_X is reported separately in xstats
    lab = np.full(len(prof), 3)
    lab[xA > 0.8] = 2
    lab[xA < 0.05] = 1
    lab[(xA > 0.15) & (xA < 0.55)] = 0
    return lab   # 0 tiling, 1 B-like (A-free), 2 A, 3 other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="B", choices=["A", "B"])
    ap.add_argument("--right", default="hexlat", choices=["hexlat", "dodeca"])
    ap.add_argument("--Ly-target", type=float, default=50.0, help="target y-period when periods must be matched")
    ap.add_argument("--lam", type=float, default=1.93)
    ap.add_argument("--T", type=float, default=0.08)
    ap.add_argument("--P", type=float, default=0.735)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--nL", type=int, default=0, help="crystal rows (0 = choose to match the tiling slab thickness)")
    ap.add_argument("--nR", type=int, default=16, help="hexagon-lattice repeats in the tiling slab")
    ap.add_argument("--nbins", type=int, default=40)
    ap.add_argument("--sweeps", type=int, default=200000)
    ap.add_argument("--tune-sweeps", type=int, default=5000)
    ap.add_argument("--flip-per-sweep", type=float, default=0.05)
    ap.add_argument("--analyze-every", type=int, default=1000)
    ap.add_argument("--snap-every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    beta = 1.0 / a.T
    v0 = {"tiling": (7 + 4 * math.sqrt(3)) / 6}
    scale = math.sqrt(1 + (2 * a.T / a.P) / v0["tiling"]) * (a.lam / (2 * math.cos(math.radians(15)))) ** 0
    if a.nL == 0 and a.right == "hexlat":
        # thickness per row: B 0.5 L, A 0.866 L; hexagon repeat ~ |T2_x| = 1.93 L * sin(...) -> match numerically
        _, tR = mc._phase_points("hexlat", a.lam * scale, a.ny, a.nR, 0)
        per = 0.5 if a.left == "B" else math.sqrt(3) / 2
        a.nL = max(4, int(round(tR / (per * a.lam * scale))))
    if a.right == "hexlat":
        s, box, labels = mc.slab_state(a.left, "hexlat", a.lam, a.ny, a.nL, a.nR, scale, seed=a.seed)
        mismatch = 0.0; smooth = abs(mc._phase_points("hexlat", a.lam * scale, 1, 2, 0)[1] / 2)
    else:
        nyL, nyR, _ = mc.match_periods(a.left, "dodeca", a.lam, scale, a.Ly_target)
        D = a.lam * scale / math.tan(math.radians(15))
        nR = a.nR if a.nR != 16 else 4          # default: 4 rows of 12-gons
        per = 0.5 if a.left == "B" else math.sqrt(3) / 2
        nL = a.nL if a.nL else max(4, int(round(nR * D * math.sqrt(3) / 2 / (per * a.lam * scale))))
        s, box, labels, mismatch = mc.slab_state2(a.left, "dodeca", a.lam, nyL, nyR, nL, nR, scale, seed=a.seed)
        smooth = D * math.sqrt(3) / 2 / 2
    img = np.zeros((len(s), 2), np.int64); N = len(s)
    n = mc.total_count(s, *box, a.lam)
    if n < 0:
        raise SystemExit("overlap in initial slab state")
    tol_out = tol_out_auto(a.T, a.P, a.lam)
    fts = open(os.path.join(a.out, "timeseries.csv"), "w", newline=""); w = csv.writer(fts)
    w.writerow(["sweep", "density", "e_per_N", "bins_tiling", "bins_B", "bins_A", "bins_other", "flip_acc",
                "xX_mean", "xA_left", "xA_right"])
    rs = np.random.default_rng(a.seed)
    dmax, dbox = 0.03, 0.001; acc_p = acc_b = win = 0; fa = 0
    profs = []; sweeps = []
    n_flip = int(round(a.flip_per_sweep * N)); t0 = time.time()
    tl, tu, tp, tang = 0.08, 0.20, 0.30, math.radians(15)
    for sw in range(1, a.sweeps + 1):
        n, ap_, ab_ = mc.sweep_img(s, img, box, a.lam, beta, a.P, dmax, dbox, 0.0, n, int(rs.integers(1, 2**31 - 1)))
        acc_p += ap_; acc_b += ab_; win += 1
        if n_flip:
            n, _, _, c3 = mc.flip_moves(s, img, box, a.lam, beta, n_flip, n, int(rs.integers(1, 2**31 - 1)), tl, tu, tp, tang)
            fa += c3
        if sw <= a.tune_sweeps and win == 100:
            dmax = min(dmax * (1.1 if acc_p / (win * N) > 0.35 else 0.9), 0.3)
            dbox *= 1.1 if acc_b / win > 0.3 else 0.9
            acc_p = acc_b = win = 0
        if sw % a.analyze_every == 0:
            M = mc.box_matrix(box)
            prof = tile_profile(mc.cart(s, box), M, a.lam, tol_out=tol_out, nbins=a.nbins, smooth=smooth)
            lab = classify(prof)
            tot = np.maximum(prof.sum(1), 1e-12)
            xA = prof[:, 0] / tot; h = len(xA) // 2
            w.writerow([sw, N / (box[0] * box[2]), n / N, int((lab == 0).sum()), int((lab == 1).sum()),
                        int((lab == 2).sum()), int((lab == 3).sum()), fa,
                        float((prof[:, 4] / tot).mean()), float(xA[h // 4: 3 * h // 4].mean()),
                        float(xA[h + h // 4: h + 3 * h // 4].mean())])
            fa = 0; profs.append(prof); sweeps.append(sw); fts.flush()
        if sw % a.snap_every == 0 or sw == a.sweeps:
            M = mc.box_matrix(box)
            np.savez(os.path.join(a.out, f"snap_{sw:08d}.npz"), pos=mc.cart(s, box), box=M,
                     pos_unwrapped=(s + img) @ M.T, labels0=labels)
            np.savez(os.path.join(a.out, "profiles.npz"), sweeps=np.array(sweeps), profiles=np.array(profs))
    el = time.time() - t0
    info = dict(vars(a), N_actual=N, tol_out=tol_out, scale=scale, y_period_mismatch=mismatch, profile_smooth=smooth, elapsed_s=el, sweeps_per_s=a.sweeps / el,
                energy_consistency=(mc.total_count(s, *box, a.lam) == n))
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=2)
    fts.close(); print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
