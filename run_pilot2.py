"""
run_pilot2.py — pilot 2 driver: temperature scan / annealing with optional 30-degree hexagon flips.

Examples
  # fixed temperature, with flips (0.05 flip attempts per particle per sweep)
  python run_pilot2.py --lam 1.93 --T 0.06 --P 0.735 --init rows --N 2000 --sweeps 200000 \
                       --flip-per-sweep 0.05 --out runs2/T0.06_rows
  # slow cooling from a fluid: stages "T:sweeps"
  python run_pilot2.py --init fluid --T-schedule 0.15:40000,0.12:40000,0.10:40000,0.08:40000,0.06:40000,0.05:40000,0.04:40000 \
                       --flip-per-sweep 0.05 --out runs2/anneal

Outputs as in run_pilot.py, plus
  timeseries.csv columns: T, tuning (1 inside a step-size tuning window; exclude from statistics),
                          flip_cand, flip_prop, flip_acc (per logging window)
  snap_*.npz            : pos, box, pos_unwrapped (for MSD), T
"""
import argparse, json, math, os, time, csv
import numpy as np
import hcss_mc as mc
from hcss_analysis import analyze


def parse_schedule(sched, T, sweeps):
    if not sched:
        return [(T, sweeps)]
    out = []
    for part in sched.split(","):
        t, n = part.split(":")
        out.append((float(t), int(n)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.93)
    ap.add_argument("--T", type=float, default=0.06)
    ap.add_argument("--T-schedule", default="")
    ap.add_argument("--P", type=float, default=0.735)
    ap.add_argument("--init", default="rows", choices=["A", "B", "rows", "hexlat", "fluid"])
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--sweeps", type=int, default=200000)
    ap.add_argument("--tune-sweeps", type=int, default=5000, help="tuning window at the start of each stage")
    ap.add_argument("--flip-per-sweep", type=float, default=0.0, help="flip attempts per particle per sweep")
    ap.add_argument("--flip-tol", default="0.08,0.20,0.30,15", help="tl,tu,tp,angle_deg for hexagon detection")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--analyze-every", type=int, default=1000)
    ap.add_argument("--snap-every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="runs2/pilot")
    ap.add_argument("--no-sk", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tl, tu, tp, tang = [float(x) for x in args.flip_tol.split(",")]
    tang = math.radians(tang)
    stages = parse_schedule(args.T_schedule, args.T, args.sweeps)

    s, box = mc.lattice_state(args.init, args.lam, args.N, seed=args.seed)
    img = np.zeros((s.shape[0], 2), np.int64)
    N = s.shape[0]
    n_pairs = mc.total_count(s, box[0], box[1], box[2], args.lam)
    if n_pairs < 0:
        raise SystemExit("initial configuration has core overlaps")
    fts = open(os.path.join(args.out, "timeseries.csv"), "w", newline=""); wts = csv.writer(fts)
    fst = open(os.path.join(args.out, "structure.csv"), "w", newline=""); wst = csv.writer(fst)
    wts.writerow(["sweep", "T", "tuning", "density", "e_per_N", "acc_part", "acc_box",
                  "flip_cand", "flip_prop", "flip_acc", "a", "b", "c"])
    wst.writerow(["sweep", "T", "density", "x_A", "x_B", "x_C", "x_D", "x_X", "e_per_N",
                  "core_coord", "psi6_L", "psi12_L", "ring_h12", "ring_h6", "perp_var"])
    rs = np.random.default_rng(args.seed)
    dmax, dbox, dshear = 0.05, 0.002, 0.02
    n_flip = int(round(args.flip_per_sweep * N))
    sw = 0; t0 = time.time(); stage_info = []
    for (T, nsw) in stages:
        beta = 1.0 / T
        acc_p = acc_b = win = 0; fc = fp = fa = 0
        tune_end = sw + min(args.tune_sweeps, max(nsw // 5, 1))
        tstage = time.time()
        for _ in range(nsw):
            sw += 1
            n_pairs, a_p, a_b = mc.sweep_img(s, img, box, args.lam, beta, args.P, dmax, dbox, dshear,
                                             n_pairs, int(rs.integers(1, 2**31 - 1)))
            acc_p += a_p; acc_b += a_b; win += 1
            if n_flip > 0:
                n_pairs, c1, c2, c3 = mc.flip_moves(s, img, box, args.lam, beta, n_flip, n_pairs,
                                                    int(rs.integers(1, 2**31 - 1)), tl, tu, tp, tang)
                fc += c1; fp += c2; fa += c3
            tuning = sw <= tune_end
            if tuning and sw % 100 == 0:
                f_p = acc_p / (win * N); f_b = acc_b / win
                dmax = min(dmax * (1.1 if f_p > 0.35 else 0.9), 0.5)
                f = 1.1 if f_b > 0.3 else 0.9
                dbox *= f; dshear *= f
                acc_p = acc_b = win = 0
            if sw % args.log_every == 0:
                V = box[0] * box[2]
                wts.writerow([sw, T, int(tuning), N / V, n_pairs / N,
                              (acc_p / (win * N)) if win else "", (acc_b / win) if win else "",
                              fc, fp, fa, box[0], box[1], box[2]])
                if not tuning:
                    acc_p = acc_b = win = 0
                fc = fp = fa = 0
            if sw % args.analyze_every == 0:
                r = analyze(mc.cart(s, box), mc.box_matrix(box), lam=args.lam,
                            sk_nmax=(0 if args.no_sk else None))
                comp = r["composition"]; sk = r["Sk"] or {}
                wst.writerow([sw, T, r["density"], comp["A"], comp["B"], comp["C"], comp["D"], comp["X"],
                              r["energy_per_particle_eps"], r["mean_core_coordination"],
                              r["order"]["psi6_L"][0], r["order"]["psi12_L"][0],
                              sk.get("ring_harmonic_12", ""), sk.get("ring_harmonic_6", ""),
                              r["lift"]["perp_variance"]])
                fts.flush(); fst.flush()
            if sw % args.snap_every == 0:
                M = mc.box_matrix(box)
                np.savez(os.path.join(args.out, f"snap_{sw:08d}.npz"), pos=mc.cart(s, box), box=M,
                         pos_unwrapped=(s + img) @ M.T, T=T)
        stage_info.append(dict(T=T, sweeps=nsw, seconds=time.time() - tstage, end_sweep=sw,
                               density=N / (box[0] * box[2]), e_per_N=n_pairs / N))
        M = mc.box_matrix(box)
        np.savez(os.path.join(args.out, f"stage_end_T{T:.3f}.npz"), pos=mc.cart(s, box), box=M,
                 pos_unwrapped=(s + img) @ M.T, T=T)
    el = time.time() - t0
    check = mc.total_count(s, box[0], box[1], box[2], args.lam)
    info = dict(vars(args), N_actual=N, elapsed_s=el, sweeps_per_s=sw / el, stages=stage_info,
                energy_consistency=(check == n_pairs), steps=dict(dmax=dmax, dbox=dbox, dshear=dshear))
    json.dump(info, open(os.path.join(args.out, "run_info.json"), "w"), indent=2)
    fts.close(); fst.close()
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
