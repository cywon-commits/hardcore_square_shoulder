"""
build_approx.py — second-order approximant of the lam* tiling (phason strain eps_2 = (2-sqrt3)^4), built by inflating the
3.12.12 tiling with ratio 2+sqrt3: inflated edges are the palindromic, zero-area polyline [0, 30, -90, 30, 0] deg; the inflated
triangle has a unique filling (7 A + 6 rhombi, big_triangle.json); the inflated 12-gon uses big_12-gon.json (180 A + 156 rhombi),
rotated by k*30 deg in successive cells so that the six rhombus orientations are equally populated (n1*n2 multiple of 6).
Usage  python build_approx.py --n1 2 --n2 3 --out approx2_2x3.npz
"""
import argparse, json, math, numpy as np
import hcss_mc as mc
LS = mc.LAM_STAR; SIG = 2 + math.sqrt(3)
U = lambda d: np.array([math.cos(math.radians(30 * d)), math.sin(math.radians(30 * d))])
def corners(dirs):
    c = [np.zeros(2)]
    for d in dirs[:-1]: c.append(c[-1] + SIG * U(d))
    return np.array(c)
def rot(P, k):
    a = math.radians(30 * k); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]); return P @ R.T
def build(n1, n2, seed=11):
    big12 = [np.array(t) for t in json.load(open("big_12-gon.json"))[0]]
    bigtri = [np.array(t) for t in json.load(open("big_triangle.json"))[0]]
    C12 = corners(list(range(12))); C12c = C12.mean(0); CT = corners([0, 4, 8]); CTc = CT.mean(0)
    D2 = SIG / math.tan(math.radians(15))
    a1 = np.array([D2, 0.0]); a2 = np.array([D2 / 2, D2 * math.sqrt(3) / 2])
    Mbox = np.column_stack([n1 * a1, n2 * a2]); rng = np.random.default_rng(seed)
    allc = np.vstack([C12 - C12c + ii * a1 + jj * a2 for ii in range(-1, n1 + 1) for jj in range(-1, n2 + 1)])
    pts = []; kk = 0
    for i in range(n1):
        for j in range(n2):
            c = i * a1 + j * a2
            for t in big12: pts.append(rot(t - C12c, kk % 6) + c)
            kk += 1
            for cc in ((a1 + a2) / 3, 2 * (a1 + a2) / 3):
                target = c + cc
                ok = [k for k in range(0, 12, 2) if max(np.min(np.linalg.norm(allc - p, axis=1)) for p in rot(CT - CTc, k) + target) < 1e-6]
                k = ok[rng.integers(len(ok))]
                for t in bigtri: pts.append(rot(t - CTc, k) + target)
    P = np.vstack(pts); inv = np.linalg.inv(Mbox); f = (P @ inv.T) % 1.0
    key = np.round(f * 1e6).astype(np.int64) % 1000000; _, idx = np.unique(key, axis=0, return_index=True)
    P = (f[np.sort(idx)] @ Mbox.T) * LS * (1 + 1e-7); Mb = Mbox * LS * (1 + 1e-7)
    box = np.array([Mb[0, 0], Mb[0, 1], Mb[1, 1]]); box[1] -= round(box[1] / box[0]) * box[0]
    s = np.stack([(P[:, 0] - box[1] * P[:, 1] / box[2]) / box[0], P[:, 1] / box[2]], 1) % 1.0
    assert mc.total_count(s, *box, LS) >= 0
    return s, box
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n1", type=int, default=2); ap.add_argument("--n2", type=int, default=3)
    ap.add_argument("--out", default="approx2_2x3.npz"); a = ap.parse_args()
    s, box = build(a.n1, a.n2); np.savez(a.out, s=s, box=box); print("N", len(s), "->", a.out)
