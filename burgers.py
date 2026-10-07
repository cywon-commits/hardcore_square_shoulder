"""
burgers.py — topological defects of the lam* tilings: 4D Burgers vectors in Z[zeta_12].

Faces of the shoulder-bond graph (Gabriel shoulder bonds, quantised to 30-degree classes exactly as in op.bonds) are
obtained by merging Delaunay triangles across non-bond edges.  In a perfect tiling the faces are A triangles and
rhombi (two B triangles merged across the core diagonal) and every face has zero holonomy.  The holonomy of a face,
    b = sum_{boundary edges, ccw} zeta^k ,
is reduced with zeta^4 = zeta^2 - 1 to an INTEGER vector (b0, b1, b2, b3) in the basis 1, zeta, zeta^2, zeta^3.
Physical part b_par = sum b_j zeta^j, perp part b_perp = sum b_j zeta^{5j} (edge units); the norm
N(b) = |b_par|^2 |b_perp|^2 is a rational integer (self-check).  Units zeta^k (1 - zeta)^n: |b_par| = (2 sin 15)^n,
|b_perp| = (2 sin 75)^n; (1 - zeta) has b_par = the core-contact vector (rhombus short diagonal).
"""
import math, cmath
import numpy as np
from scipy.spatial import cKDTree, Delaunay
import op

Z = cmath.exp(1j * math.pi / 6)
# zeta^k in the basis (1, zeta, zeta^2, zeta^3), using zeta^4 = zeta^2 - 1, zeta^6 = -1
BASIS = []
v = [1, 0, 0, 0]
for k in range(12):
    BASIS.append(np.array(v, dtype=np.int64))
    # multiply by zeta: (a0,a1,a2,a3) -> (0,a0,a1,a2) + a3*zeta^4 = (-a3, a0, a1 + a3, a2)
    v = [-v[3], v[0], v[1] + v[3], v[2]]


def par(b):
    return sum(int(b[j]) * Z ** j for j in range(4))


def perp(b):
    return sum(int(b[j]) * Z ** (5 * j) for j in range(4))


def norm(b):
    return abs(par(b)) ** 2 * abs(perp(b)) ** 2


def rot(b, m=1):
    """multiply by zeta^m (12-fold rotation of the defect)."""
    out = np.zeros(4, np.int64)
    for j in range(4):
        out += int(b[j]) * BASIS[(j + m) % 12] if (j + m) < 12 else int(b[j]) * BASIS[(j + m) % 12]
    return out


def canon(b):
    """orbit label under the 12 rotations and sign: lexicographically smallest representative."""
    reps = []
    for m in range(12):
        r = np.zeros(4, np.int64)
        for j in range(4):
            r += int(b[j]) * BASIS[(j + m) % 12]
        reps.append(tuple(int(x) for x in r))
    return min(reps)


def faces(P, M, lam=1.93, tol_in=0.06, tol_out=0.25, max_dev_deg=12.0):
    """faces of the shoulder-bond graph with their integer Burgers vectors.  Returns list of dicts
    (centroid, b, n_triangles) for faces whose centroid lies in the primary cell, and the bond classes used."""
    Q, oid, _ = op._images(P, M)
    tree = cKDTree(Q)
    L, C, raw = op.bonds(P, M, lam, tol_in, tol_out, max_dev_deg=max_dev_deg, return_raw=True)
    bondset = {}
    for (i, j, k) in L:
        bondset[(i, j)] = k                       # directed class, consistent with op.bonds (both directions stored)
    tri = Delaunay(Q).simplices
    nT = len(tri)
    parent = np.arange(nT)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    edge_owner = {}
    # an edge of the Delaunay triangulation is a "bond" iff its endpoints are a shoulder bond pair at the right length
    def is_bond(u, v):
        a, b = oid[u], oid[v]
        if (a, b) not in bondset:
            return None
        r = np.linalg.norm(Q[v] - Q[u])
        if not (lam - tol_in <= r < lam + tol_out):
            return None
        return bondset[(a, b)]
    for t, s in enumerate(tri):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            u, v = s[a], s[b]
            key = (min(u, v), max(u, v))
            if key in edge_owner:
                if is_bond(u, v) is None:
                    ra, rb = find(t), find(edge_owner[key])
                    if ra != rb: parent[ra] = rb
            else:
                edge_owner[key] = t
    hol = {}; cen = {}; cnt = {}
    for t, s in enumerate(tri):
        r = find(t)
        pts = Q[s]
        # ccw orientation of the triangle
        if (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0]) < 0:
            s = s[[0, 2, 1]]
        h = hol.setdefault(r, np.zeros(4, np.int64))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            k = is_bond(s[a], s[b])
            if k is not None:
                h += BASIS[k]
        cen[r] = cen.get(r, 0) + Q[s].mean(0); cnt[r] = cnt.get(r, 0) + 1
    inv = np.linalg.inv(M)
    out = []
    hull = Delaunay(Q).convex_hull
    for r in hol:
        c = cen[r] / cnt[r]
        f = c @ inv.T
        if 0 <= f[0] < 1 and 0 <= f[1] < 1:
            out.append(dict(c=c, b=hol[r].copy(), ntri=cnt[r]))
    return out


def defect_stats(P, M, lam, tol_in, tol_out):
    F = faces(P, M, lam, tol_in, tol_out)
    D = [f for f in F if np.any(f["b"] != 0)]
    nb = np.array([norm(f["b"]) for f in D]) if D else np.zeros(0)
    orbits = {}
    for f in D:
        o = canon(f["b"]); orbits[o] = orbits.get(o, 0) + 1
    # pairing: nearest other defect (periodic) cancels the Burgers vector?
    paired = 0
    if len(D) > 1:
        inv = np.linalg.inv(M); C = np.array([f["c"] for f in D])
        for i, f in enumerate(D):
            d = C - C[i]; fr = d @ inv.T; fr -= np.round(fr); d = fr @ M.T; r = np.linalg.norm(d, axis=1); r[i] = np.inf
            j = int(np.argmin(r))
            if np.all(D[j]["b"] + f["b"] == 0): paired += 1
    total = np.sum([f["b"] for f in F], axis=0) if F else np.zeros(4)
    return dict(n_faces=len(F), n_defects=len(D), norms=nb.tolist(), orbits=orbits, frac_paired_nn=paired / max(len(D), 1),
                total_b=[int(x) for x in total], defects=D)


def screening(D, M, Rs=(2, 3, 4, 6, 8, 12)):
    """<|sum of b_perp|^2> and <|sum b_par|^2> in square windows of side R (sigma) placed on a grid; returns (R, var_par, var_perp)."""
    if not D:
        return []
    inv = np.linalg.inv(M); C = np.array([f["c"] for f in D]); B = [f["b"] for f in D]
    bp = np.array([par(b) for b in B]); bq = np.array([perp(b) for b in B])
    fr = (C @ inv.T) % 1.0
    out = []
    Lx = np.linalg.norm(M[:, 0]); Ly = abs(np.linalg.det(M)) / Lx
    for R in Rs:
        nx, ny = max(1, int(Lx // R)), max(1, int(Ly // R))
        ix = np.minimum((fr[:, 0] * nx).astype(int), nx - 1); iy = np.minimum((fr[:, 1] * ny).astype(int), ny - 1)
        Sp = np.zeros((nx, ny), complex); Sq = np.zeros((nx, ny), complex)
        np.add.at(Sp, (ix, iy), bp); np.add.at(Sq, (ix, iy), bq)
        out.append((float(R), float(np.mean(np.abs(Sp) ** 2)), float(np.mean(np.abs(Sq) ** 2))))
    return out
