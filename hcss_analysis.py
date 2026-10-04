"""
hcss_analysis.py — analysis routine for 2D hard-core square-shoulder (HCSS) configurations.

Model convention: core diameter sigma = 1, shoulder U = eps for 1 <= r < lam, 0 for r >= lam.

Main entry:  analyze(pos, box=None, lam=..., ...)  ->  dict of observables
  pos : (N,2) array of positions
  box : None (open boundaries) or 2x2 matrix whose COLUMNS are the box vectors a1, a2
        (positions may be anywhere; they are wrapped internally)

Observables
  1. Tile decomposition (Delaunay triangles classified by edge classes):
        S = "short" pair  (r < lam - tol_in)            -> costs eps (core contact or inside shoulder)
        L = "long"  pair  (lam - tol_in <= r < lam + tol_out)  -> shoulder contact, free
        X = anything longer (triangle is not a contact tile)
     Tiles: A = LLL, B = SLL, C = SSL, D = SSS, X = other.  Composition x_t over tiles,
     energy per particle e/eps = (number of S pairs)/N, density.
  2. Core contacts: pairs with r < 1 + tol_core (short diagonals of thin rhombi at lam*); pairs per particle,
     mean core coordination, orientation histogram (mod 180 deg).
  3. Bond-orientational order psi_n (n = 6, 12) on L bonds and on all tile edges; global |<psi_n>| and local <|psi_n|>.
  4. Structure factor S(k) on the reciprocal lattice of the box (periodic only), plus angular harmonics
     of the first-peak ring (12-fold vs 6-fold weight).
  5. Lift / phason: edges of the chosen class are quantized to multiples of 30 deg; vertices are lifted by
     breadth-first search with the 12-fold Galois-conjugate map k -> exp(i*150deg*k).  Returns perp-space
     variance (random tiling: grows ~ ln N; crystal/approximant: bounded) and the number of edges whose
     perp increment disagrees with the BFS assignment (defects or global phason strain across the box).

Dependencies: numpy, scipy.
"""
import json, math, sys
import numpy as np
from scipy.spatial import Delaunay
from collections import deque, Counter

# ------------------------------------------------------------------ geometry helpers
def _wrap(pos, box):
    if box is None:
        return pos.copy()
    inv = np.linalg.inv(box)
    f = pos @ inv.T
    f -= np.floor(f)
    return f @ box.T

def _min_image(d, box):
    if box is None:
        return d
    inv = np.linalg.inv(box)
    f = d @ inv.T
    f -= np.round(f)
    return f @ box.T

def _periodic_points(pos, box):
    """3x3 images for periodic Delaunay; returns points, original index, image flag (0 = central)."""
    if box is None:
        return pos, np.arange(len(pos)), np.zeros(len(pos), int)
    pts, idx, img = [], [], []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            pts.append(pos + i * box[:, 0] + j * box[:, 1])
            idx.append(np.arange(len(pos)))
            img.append(np.full(len(pos), 0 if (i == 0 and j == 0) else 1))
    return np.vstack(pts), np.concatenate(idx), np.concatenate(img)

# ------------------------------------------------------------------ main analysis
def analyze(pos, box=None, lam=2 * math.cos(math.radians(15)), tol_in=0.06, tol_out=0.15,
            tol_core=0.06, lift_class="L", sk_kmax=5.0, sk_nmax=None):
    pos = np.asarray(pos, float)
    box = None if box is None else np.asarray(box, float)
    N = len(pos)
    pos = _wrap(pos, box)
    area = abs(np.linalg.det(box)) if box is not None else None

    # --- Delaunay on (periodic) points, keep triangles with first vertex in the central cell (each once)
    P, oid, img = _periodic_points(pos, box)
    tri = Delaunay(P)
    simp = tri.simplices
    if box is not None:
        # keep each periodic triangle exactly once: the copy whose smallest-index vertex is in the central cell
        o = oid[simp]
        first = simp[np.arange(len(simp)), np.argmin(o, axis=1)]
        simp = simp[img[first] == 0]
    else:
        # drop the long sliver triangles on the convex hull boundary
        e = np.linalg.norm(P[simp] - P[np.roll(simp, 1, axis=1)], axis=2)
        simp = simp[e.max(axis=1) < lam + tol_out + 1e-9]

    def eclass(r):
        if r < lam - tol_in:
            return "S"
        if r < lam + tol_out:
            return "L"
        return "X"

    tiles = []
    edges = {}  # (i,j) i<j  -> (class, vector i->j)
    for s in simp:
        cls = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = P[s[a]], P[s[b]]
            r = np.linalg.norm(pb - pa)
            c = eclass(r)
            cls.append(c)
            i, j = oid[s[a]], oid[s[b]]
            if c != "X":
                key = (min(i, j), max(i, j))
                vec = (pb - pa) if i < j else (pa - pb)
                edges.setdefault(key, (c, vec))
        cs = "".join(sorted(cls))
        t = {"LLL": "A", "LLS": "B", "LSS": "C", "SSS": "D"}.get(cs, "X")
        tiles.append(t)
    ct = Counter(tiles)
    ntile = len(tiles)
    comp = {k: ct.get(k, 0) / ntile for k in "ABCDX"} if ntile else {}

    # --- all pairs below lam (energy) via neighbour search on images
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    pairs = tree.query_pairs(lam - 1e-12)
    S_pairs = set()
    core_pairs = []
    for a, b in pairs:
        i, j = oid[a], oid[b]
        if i == j:
            continue
        if box is not None and (img[a] == 1 and img[b] == 1):
            continue
        key = (min(i, j), max(i, j))
        if key in S_pairs:
            continue
        S_pairs.add(key)
        d = P[b] - P[a]
        if np.linalg.norm(d) < 1 + tol_core:
            core_pairs.append(d)
    e_per_particle = len(S_pairs) / N
    overlaps = sum(1 for a, b in tree.query_pairs(1 - 1e-6)
                   if oid[a] != oid[b] and not (box is not None and img[a] == 1 and img[b] == 1))

    # --- dimers
    dimer_angles = [math.degrees(math.atan2(d[1], d[0])) % 180 for d in core_pairs]
    hist, _ = np.histogram(dimer_angles, bins=12, range=(0, 180))

    # --- bond order on L bonds and on all tile edges
    def psi(n, which):
        acc = np.zeros(N, complex)
        cnt = np.zeros(N)
        for (i, j), (c, v) in edges.items():
            if which == "L" and c != "L":
                continue
            th = math.atan2(v[1], v[0])
            z = np.exp(1j * n * th)
            acc[i] += z; acc[j] += z * np.exp(1j * n * math.pi)  # reversed bond
            cnt[i] += 1; cnt[j] += 1
        m = cnt > 0
        loc = np.abs(acc[m] / cnt[m])
        glob = abs(acc[m].sum() / cnt[m].sum())
        return float(glob), float(loc.mean()) if m.any() else 0.0
    order = {f"psi{n}_{w}": psi(n, w) for n in (6, 12) for w in ("L", "all")}

    # --- structure factor (periodic)
    sk = None
    if box is not None and (sk_nmax is None or sk_nmax > 0):
        rec = 2 * math.pi * np.linalg.inv(box).T          # columns = reciprocal vectors
        if sk_nmax is None:
            sk_nmax = int(math.ceil(sk_kmax / min(np.linalg.norm(rec[:, 0]), np.linalg.norm(rec[:, 1])))) + 1
        m = np.arange(-sk_nmax, sk_nmax + 1)
        M1, M2 = np.meshgrid(m, m)
        K = np.stack([M1.ravel(), M2.ravel()], 1) @ rec.T
        kk = np.linalg.norm(K, axis=1)
        kmax = sk_kmax
        sel = (kk > 1e-9) & (kk < kmax)
        K, kk = K[sel], kk[sel]
        rho = np.exp(1j * (pos @ K.T)).sum(axis=0)
        Sk = np.abs(rho) ** 2 / N
        i0 = np.argmax(Sk)
        kpk = kk[i0]
        ring = np.abs(kk - kpk) < 0.08 * kpk
        ang = np.arctan2(K[ring, 1], K[ring, 0])
        w = Sk[ring]
        h12 = abs((w * np.exp(12j * ang)).sum()) / w.sum()
        h6 = abs((w * np.exp(6j * ang)).sum()) / w.sum()
        sk = {"k_peak": float(kpk), "S_peak": float(Sk[i0]), "ring_harmonic_12": float(h12),
              "ring_harmonic_6": float(h6), "n_ring": int(ring.sum())}

    # --- lift / phason (12 directions, conjugate map k -> 5k)
    adj = {}
    qerr = []
    for (i, j), (c, v) in edges.items():
        if c != lift_class:
            continue
        th = math.degrees(math.atan2(v[1], v[0])) % 360
        k = int(round(th / 30.0)) % 12
        qerr.append(abs(((th - 30 * k + 180) % 360) - 180))
        z = np.exp(1j * math.radians(150 * k))
        adj.setdefault(i, []).append((j, z))
        adj.setdefault(j, []).append((i, -z))
    perp = {}
    mismatch = 0
    for s0 in adj:
        if s0 in perp:
            continue
        perp[s0] = 0j
        q = deque([s0])
        while q:
            u = q.popleft()
            for w, z in adj[u]:
                if w not in perp:
                    perp[w] = perp[u] + z; q.append(w)
                elif abs(perp[w] - perp[u] - z) > 1e-6:
                    mismatch += 1
    pv = np.array(list(perp.values()))
    perp_var = float(np.mean(np.abs(pv - pv.mean()) ** 2)) if len(pv) else None

    out = {
        "N": N, "lam": lam,
        "density": (N / area) if area else None,
        "tiles_total": ntile, "composition": comp,
        "energy_per_particle_eps": e_per_particle,
        "core_overlaps": overlaps,
        "core_pairs_per_particle": len(core_pairs) / N,
        "mean_core_coordination": 2 * len(core_pairs) / N,
        "dimer_orientation_hist_15deg": hist.tolist(),
        "order": order, "Sk": sk,
        "lift": {"class": lift_class, "vertices": len(perp), "perp_variance": perp_var,
                 "mismatched_edges": mismatch // 2,
                 "max_angle_quantization_error_deg": float(max(qerr)) if qerr else None},
    }
    return out

# ------------------------------------------------------------------ prediction table for comparison
def predictions_2cos15():
    r3 = math.sqrt(3)
    return {"lam_star": 2 * math.cos(math.radians(15)), "P_star_eps": r3 - 1,
            "x_A": 1 / 3, "x_B": 2 / 3, "energy_per_particle_eps": 2 / 3,
            "density_12fold": 42 - 24 * r3, "core_pairs_per_particle": 2 / 3,
            "mean_core_coordination": 4 / 3}

def load(path):
    """npz with arrays 'pos' (N,2) and optional 'box' (2,2, columns = box vectors)."""
    d = np.load(path)
    return d["pos"], (d["box"] if "box" in d.files else None)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--lam", type=float, default=2 * math.cos(math.radians(15)))
    ap.add_argument("--lift", default="L", choices=["L", "S"])
    a = ap.parse_args()
    pos, box = load(a.config)
    print(json.dumps(analyze(pos, box, lam=a.lam, lift_class=a.lift), indent=2))
