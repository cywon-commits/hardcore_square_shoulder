"""
run_pilot.py — pilot NPT run for the HCSS lam* = 2cos15 deg study.

Example
  python run_pilot.py --lam 1.93 --T 0.15 --P 0.735 --init rows --N 2000 \
                      --sweeps 1000000 --out runs/rows_T015

Outputs (in --out)
  timeseries.csv : every --log-every sweeps  (sweep, density, e_per_N, acc_part, acc_box, a, b, c)
  structure.csv  : every --analyze-every sweeps (tile composition, order, S(k) ring harmonics, perp variance)
  snap_XXXXXXXX.npz : every --snap-every sweeps (pos, box matrix)
  state.npz      : checkpoint (for --restart), written at every snapshot and at the end
  run_info.json  : parameters, throughput (trial moves / s), final acceptance
"""
import argparse, json, math, os, time, csv
import numpy as np
import hcss_mc as mc
from hcss_analysis import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.93)
    ap.add_argument("--T", type=float, default=0.15)
    ap.add_argument("--P", type=float, default=0.735)
    ap.add_argument("--init", default="rows", choices=["A", "B", "rows", "fluid"])
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--sweeps", type=int, default=200000)
    ap.add_argument("--tune-sweeps", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--analyze-every", type=int, default=1000)
    ap.add_argument("--snap-every", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--no-sk", action="store_true", help="skip S(k) in structure analysis (faster)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    beta = 1.0 / args.T
    if args.restart and os.path.exists(os.path.join(args.out, "state.npz")):
        st = np.load(os.path.join(args.out, "state.npz"))
        s, box = st["s"].copy(), st["box"].copy()
        start = int(st["sweep"]); dmax, dbox, dshear = [float(x) for x in st["steps"]]
        rng_seed = int(st["rng"])
    else:
        s, box = mc.lattice_state(args.init, args.lam, args.N)
        start = 0; dmax, dbox, dshear = 0.05, 0.002, 0.02
        rng_seed = args.seed
    N = s.shape[0]
    n_pairs = mc.total_count(s, box[0], box[1], box[2], args.lam)
    if n_pairs < 0:
        raise SystemExit("initial configuration has core overlaps")

    ts_path = os.path.join(args.out, "timeseries.csv")
    st_path = os.path.join(args.out, "structure.csv")
    new_files = not (args.restart and os.path.exists(ts_path))
    fts = open(ts_path, "a", newline=""); wts = csv.writer(fts)
    fst = open(st_path, "a", newline=""); wst = csv.writer(fst)
    if new_files:
        wts.writerow(["sweep", "density", "e_per_N", "acc_part", "acc_box", "a", "b", "c"])
        wst.writerow(["sweep", "density", "x_A", "x_B", "x_C", "x_D", "x_X", "e_per_N",
                      "core_coord", "psi6_L", "psi12_L", "ring_h12", "ring_h6", "perp_var"])

    acc_p = acc_b = 0; win = 0
    t0 = time.time(); trials = 0
    rs = np.random.default_rng(rng_seed)
    for sw in range(start + 1, args.sweeps + 1):
        n_pairs, ap_, ab_ = mc.sweep(s, box, args.lam, beta, args.P, dmax, dbox, dshear,
                                     n_pairs, int(rs.integers(1, 2**31 - 1)))
        acc_p += ap_; acc_b += ab_; win += 1; trials += N + 1
        # step-size tuning only during the tuning window (keeps detailed balance afterwards)
        if sw <= args.tune_sweeps and sw % 100 == 0:
            fp = acc_p / (win * N); fb = acc_b / win
            dmax *= 1.1 if fp > 0.35 else 0.9
            dmax = min(dmax, 0.5)
            f = 1.1 if fb > 0.3 else 0.9
            dbox *= f; dshear *= f
            acc_p = acc_b = 0; win = 0
        if sw % args.log_every == 0:
            V = box[0] * box[2]
            wts.writerow([sw, N / V, n_pairs / N,
                          (acc_p / (win * N)) if win else "", (acc_b / win) if win else "",
                          box[0], box[1], box[2]])
            if sw > args.tune_sweeps:
                acc_p = acc_b = 0; win = 0
        if sw % args.analyze_every == 0:
            r = analyze(mc.cart(s, box), mc.box_matrix(box), lam=args.lam,
                        sk_nmax=(0 if args.no_sk else None))
            comp = r["composition"]; sk = r["Sk"] or {}
            wst.writerow([sw, r["density"], comp["A"], comp["B"], comp["C"], comp["D"], comp["X"],
                          r["energy_per_particle_eps"], r["mean_core_coordination"],
                          r["order"]["psi6_L"][0], r["order"]["psi12_L"][0],
                          sk.get("ring_harmonic_12", ""), sk.get("ring_harmonic_6", ""),
                          r["lift"]["perp_variance"]])
            fts.flush(); fst.flush()
        if sw % args.snap_every == 0 or sw == args.sweeps:
            np.savez(os.path.join(args.out, f"snap_{sw:08d}.npz"),
                     pos=mc.cart(s, box), box=mc.box_matrix(box))
            np.savez(os.path.join(args.out, "state.npz"), s=s, box=box, sweep=sw,
                     steps=np.array([dmax, dbox, dshear]), rng=int(rs.integers(1, 2**31 - 1)))
    el = time.time() - t0
    check = mc.total_count(s, box[0], box[1], box[2], args.lam)
    info = dict(vars(args), N_actual=N, elapsed_s=el, trial_moves_per_s=trials / el if el > 0 else None,
                sweeps_per_s=(args.sweeps - start) / el if el > 0 else None,
                final_density=N / (box[0] * box[2]), final_e_per_N=n_pairs / N,
                energy_consistency=(check == n_pairs), steps=dict(dmax=dmax, dbox=dbox, dshear=dshear))
    json.dump(info, open(os.path.join(args.out, "run_info.json"), "w"), indent=2)
    fts.close(); fst.close()
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
