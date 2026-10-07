"""
op_identity.py — (analysis 1) eta (tile counts) versus det E = |alpha|^2 - |beta|^2 (winding loops of the lift), and the
calibration of the bond-texture proxies psi4 ~ alpha, psi6 ~ beta, on ideal structures and on any saved snapshots.
Usage  python op_identity.py [snapshot.npz ...] --out identity.json
Snapshots: npz with 'pos' (N,2) and 'box' (2x2, columns = box vectors), e.g. runs2/*/snap_*.npz, coex/*/snap_*.npz.
"""
import argparse, glob, json, math, numpy as np
import hcss_mc as mc, op
def run(name, P, M, lam, tol_in, tol_out):
    o, _ = op.order_parameters(P, M, lam=lam, tol_in=tol_in, tol_out=tol_out)
    keep = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in o.items() if k not in ("E",)}
    keep["name"] = name; return keep
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snaps", nargs="*"); ap.add_argument("--lam", type=float, default=1.93)
    ap.add_argument("--tol-out", type=float, default=0.25); ap.add_argument("--out", default="identity.json"); a = ap.parse_args()
    rows = []
    for kind in ("A", "B", "rows", "hexlat", "dodeca"):
        s, box = mc.lattice_state(kind, op.LS, 900, scale=1 + 1e-7, seed=1)
        rows.append(run(kind + " (ideal)", mc.cart(s, box), mc.box_matrix(box), op.LS - 1e-6, 0.02, 0.02))
    for f in sorted(glob.glob("approx2*.npz")):
        z = np.load(f); rows.append(run(f + " (ideal)", mc.cart(z["s"], z["box"]), mc.box_matrix(z["box"]), op.LS - 1e-6, 0.02, 0.02))
    for f in a.snaps:
        z = np.load(f); rows.append(run(f, z["pos"], z["box"], a.lam, 0.06, a.tol_out))
    print(f"{'structure':48s} {'eta':>8s} {'detE':>8s} {'|alpha|':>7s} {'|beta|':>7s} {'psi4':>6s} {'psi6':>6s} {'psi12':>6s} {'x_oth':>6s} {'defect':>6s}")
    for r in rows:
        print(f"{r['name'][-48:]:48s} {r['eta']:+8.5f} {r.get('detE', float('nan')):+8.5f} {r.get('abs_alpha', float('nan')):7.4f} "
              f"{r.get('abs_beta', float('nan')):7.4f} {r['psi4']:6.3f} {r['psi6']:6.3f} {r['psi12']:6.3f} {r['x_other']:6.3f} {r['n_defect_edges']:6d}")
    json.dump(rows, open(a.out, "w"), indent=1, default=float)
if __name__ == "__main__":
    main()
