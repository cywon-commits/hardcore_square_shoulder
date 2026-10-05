"""
msd.py — mean-square displacement from pilot-2 snapshots (needs pos_unwrapped).

Usage  python msd.py runs2/T0.06_rows [...]
Prints MSD(t) relative to the first snapshot after --skip sweeps; a fluid shows MSD growing ~ t,
a solid or frozen tiling shows a plateau of order (thermal gap)^2.
"""
import argparse, glob, os, json
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--skip", type=int, default=0)
    a = ap.parse_args()
    for run in a.runs:
        snaps = sorted(glob.glob(os.path.join(run, "snap_*.npz")))
        sw = [int(os.path.basename(p)[5:13]) for p in snaps]
        snaps = [p for p, k in zip(snaps, sw) if k > a.skip]; sw = [k for k in sw if k > a.skip]
        if len(snaps) < 2 or "pos_unwrapped" not in np.load(snaps[0]).files:
            print(run, "needs >= 2 pilot-2 snapshots"); continue
        X0 = np.load(snaps[0])["pos_unwrapped"]; X0 -= X0.mean(0)
        res = []
        for p, k in zip(snaps[1:], sw[1:]):
            X = np.load(p)["pos_unwrapped"]; X -= X.mean(0)
            res.append((k - sw[0], float(((X - X0) ** 2).sum(1).mean())))
        json.dump(res, open(os.path.join(run, "msd.json"), "w"))
        print(run); [print(f"   dt={t:9d}  MSD={m:9.4f}") for t, m in res]


if __name__ == "__main__":
    main()
