"""
tiling_mc.py — T -> 0 random-tiling ensemble of the lam* = 2cos15 tiles (A triangles + 30-degree rhombi), uniform weight.
Moves: (1) 30-degree hexagon flip (hcss_mc.flip_moves, point reflection; acceptance n_old/n_new);
       (2) 12-gon refill: a regular 12-gon of side lam whose 13 interior vertices form one of the 4421 fillings is
           refilled with a uniformly chosen filling; acceptance n_old/n_new (number of such 12-gons).
Positions are ideal (edge lam*(1+1e-7)); no vibrations.
Flips use beta = 1e3: valid tiling flips have dU = 0 and are unaffected, while flips of hexagon-like but non-tiling
local configurations (which would create extra costly pairs) are rejected. (beta = 1e-9 in pilots 8 and earlier
symmetry tests let the tiling degrade; eta drifted.)
"""
import sys, math, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import hcss_mc as mc
from scipy.spatial import cKDTree

LS = mc.LAM_STAR
F = mc.dodecagon_fillings()                       # (4421, 13, 2), unit edge, relative to centre
L = LS * (1 + 1e-7)
R = L / (2 * math.sin(math.radians(15)))          # circumradius
ANG = np.radians(15 + 30 * np.arange(12))
VERT = np.stack([R * np.cos(ANG), R * np.sin(ANG)], 1)
FKEY = {tuple(sorted((round(x, 3), round(y, 3)) for x, y in f * L)): j for j, f in enumerate(F)}


class Tiling:
    def __init__(self, kind, N, seed=0):
        self.s, self.box = mc.lattice_state(kind, LS, N, scale=1 + 1e-7, seed=seed)
        self.img = np.zeros((len(self.s), 2), np.int64)
        self.lam = LS
        self.n_pairs = mc.total_count(self.s, *self.box, self.lam)
        self.rng = np.random.default_rng(seed + 100)
        self.find_all_dodecagons()

    def cart(self):
        return mc.cart(self.s, self.box)

    def _tree(self):
        P = self.cart(); M = mc.box_matrix(self.box)
        imgs = []; idx = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                imgs.append(P + i * M[:, 0] + j * M[:, 1]); idx.append(np.arange(len(P)))
        return cKDTree(np.vstack(imgs)), np.concatenate(idx), P, M

    def dodecagons_near(self, tree, oid, P, M, centres_to_test):
        out = {}
        for c in centres_to_test:
            ok = True
            for v in VERT:
                d, k = tree.query(c + v)
                if d > 1e-4:
                    ok = False; break
            if not ok:
                continue
            inside = tree.query_ball_point(c, R - 0.05)
            if len(inside) != 13:
                continue
            rel = tree.data[inside] - c
            key = tuple(sorted((round(x, 3), round(y, 3)) for x, y in rel))
            if key not in FKEY:
                continue
            # canonical centre (wrapped) as dict key; store interior particle ids
            f = np.linalg.solve(M, c) % 1.0
            ck = (round(f[0], 6) % 1.0, round(f[1], 6) % 1.0)
            out[ck] = (c, [int(oid[i]) for i in inside])
        return out

    def find_all_dodecagons(self):
        tree, oid, P, M = self._tree()
        cands = set()
        for p in P:
            for v in VERT:
                c = p - v
                cands.add((round(c[0], 5), round(c[1], 5)))
        self.dod = self.dodecagons_near(tree, oid, P, M, [np.array(c) for c in cands])
        return len(self.dod)

    def refill_move(self):
        if not self.dod:
            return False
        n_old = len(self.dod)
        keys = list(self.dod.keys())
        ck = keys[self.rng.integers(n_old)]
        c, ids = self.dod[ck]
        j = self.rng.integers(len(F))
        newpos = c + F[j] * L
        # assign new interior positions to the 13 particle ids (any assignment; particles are identical)
        old_s = self.s[ids].copy(); old_img = self.img[ids].copy()
        M = mc.box_matrix(self.box); inv = np.linalg.inv(M)
        f = newpos @ inv.T
        self.s[ids] = f % 1.0
        n_pairs_new = mc.total_count(self.s, *self.box, self.lam)
        if n_pairs_new != self.n_pairs:                  # should never happen for valid fillings
            self.s[ids] = old_s; return False
        n_new = self.find_all_dodecagons()
        if self.rng.random() < min(1.0, n_old / max(n_new, 1)):
            self.img[ids] = old_img                     # image counters are irrelevant here
            return True
        self.s[ids] = old_s
        self.find_all_dodecagons()
        return False

    def flip_sweep(self, n_attempt):
        self.n_pairs, seen, prop, acc = mc.flip_moves(self.s, self.img, self.box, self.lam, 1e3, n_attempt, self.n_pairs,
                                                      int(self.rng.integers(1, 2**31 - 1)), 0.03, 0.03, 0.05, math.radians(3))
        return seen, prop, acc


def diagnostics(P, M, lam=LS, ref_P=None):
    """core-pair (rhombus short diagonal) orientation classes mod 180 (6 classes), shoulder-edge classes (6),
    S(k) at the 3.12.12 superlattice vectors, first-ring 12th / 6th harmonics."""
    inv = np.linalg.inv(M)
    tree = cKDTree(np.vstack([P + i * M[:, 0] + j * M[:, 1] for i in (-1, 0, 1) for j in (-1, 0, 1)]))
    N = len(P)
    core = np.zeros(6); sh = np.zeros(6)
    for i, p in enumerate(P):
        for k in tree.query_ball_point(p, lam * 1.001):
            d = tree.data[k] - p; r = np.linalg.norm(d)
            if r < 1e-9:
                continue
            th = (math.degrees(math.atan2(d[1], d[0])) % 180)
            if abs(r - 1) < 1e-3:
                core[int(round((th - 15) / 30)) % 6] += 0.5
            elif abs(r - lam) < 1e-3:
                sh[int(round(th / 30)) % 6] += 0.5
    # structure factor on the box reciprocal lattice
    rec = 2 * math.pi * inv.T
    nmax = int(math.ceil(6.5 / min(np.linalg.norm(rec[:, 0]), np.linalg.norm(rec[:, 1])))) + 1
    m = np.arange(-nmax, nmax + 1); A, B = np.meshgrid(m, m)
    K = np.stack([A.ravel(), B.ravel()], 1) @ rec.T
    kk = np.linalg.norm(K, axis=1); sel = (kk > 1e-9) & (kk < 6.5); K = K[sel]; kk = kk[sel]
    rho = np.exp(1j * (P @ K.T)).sum(0); S = np.abs(rho) ** 2 / N
    # superlattice of 3.12.12: centre spacing D = L cot 15; shortest reciprocal vectors |G| = 4 pi /(sqrt3 D)
    D = LS / math.tan(math.radians(15)); G = 4 * math.pi / (math.sqrt(3) * D)
    sup = S[np.abs(kk - G) < 0.02].mean() if np.any(np.abs(kk - G) < 0.02) else float("nan")
    # strongest ring (|k| > 2): angular harmonics
    i0 = np.argmax(np.where(kk > 2.0, S, 0)); kp = kk[i0]
    ring = np.abs(kk - kp) < 0.04 * kp; ang = np.arctan2(K[ring, 1], K[ring, 0]); w = S[ring]
    h12 = abs((w * np.exp(12j * ang)).sum()) / w.sum(); h6 = abs((w * np.exp(6j * ang)).sum()) / w.sum()
    out = dict(core_classes=(core / core.sum()).round(3).tolist(), shoulder_classes=(sh / sh.sum()).round(3).tolist(),
               S_superlattice=float(sup), k_peak=float(kp), ring_h12=float(h12), ring_h6=float(h6))
    if ref_P is not None:
        t = cKDTree(ref_P); d, _ = t.query(P); out["frac_unmoved"] = float((d < 1e-3).mean())
    return out, (K, S)
