"""
hcss_mc.py — NPT Monte Carlo for the 2D hard-core square-shoulder (HCSS) model.

Units: sigma = 1 (core diameter), eps = 1 (shoulder height), k_B = 1.
Pair potential: r < 1 -> infinite;  1 <= r < lam -> eps;  r >= lam -> 0.

State
  s   : (N,2) fractional coordinates in [0,1)
  h   : box as (a, b, c): box vectors a1 = (a, 0), a2 = (b, c)   (upper-triangular cell matrix)
Moves
  1. single-particle displacement (Metropolis, exp(-beta*dU))
  2. box move in (ln a, ln c, b), symmetric proposal; acceptance
        min(1, exp(-beta*(dU + P*dV) + (N+1)*ln(V'/V)))
     (the extra +1 is the Jacobian of sampling ln a, ln c at fixed b: da dc = V dln a dln c)
Neighbour search: fractional cell list, cells at least lam wide in perpendicular width, >= 3 cells per axis.
Minimum image is valid while both perpendicular widths exceed 2*lam (checked).
"""
import math
import numpy as np
from numba import njit

CAP = 48  # max particles per cell


# ----------------------------------------------------------------------------- box helpers
@njit(cache=True)
def perp_widths(a, b, c):
    area = a * c
    w1 = area / math.sqrt(b * b + c * c)   # distance between the two faces spanned by a2
    w2 = c                                 # distance between the two faces spanned by a1
    return w1, w2


@njit(cache=True)
def frac_to_cart(sx, sy, a, b, c):
    return a * sx + b * sy, c * sy


@njit(cache=True)
def min_image_dist2(dsx, dsy, a, b, c):
    dsx -= math.floor(dsx + 0.5)
    dsy -= math.floor(dsy + 0.5)
    dx = a * dsx + b * dsy
    dy = c * dsy
    # triclinic: also test the neighbouring image along a1 (needed for strong shear)
    best = dx * dx + dy * dy
    for k in (-1, 1):
        ddx = dx + k * a
        d2 = ddx * ddx + dy * dy
        if d2 < best:
            best = d2
    return best


# ----------------------------------------------------------------------------- cell list
@njit(cache=True)
def build_cells(s, a, b, c, lam):
    w1, w2 = perp_widths(a, b, c)
    nx = max(3, int(w1 / lam))
    ny = max(3, int(w2 / lam))
    N = s.shape[0]
    count = np.zeros(nx * ny, np.int64)
    members = np.full((nx * ny, CAP), -1, np.int64)
    cell_of = np.empty(N, np.int64)
    slot_of = np.empty(N, np.int64)
    for i in range(N):
        cx = int(s[i, 0] * nx) % nx
        cy = int(s[i, 1] * ny) % ny
        ci = cx + nx * cy
        k = count[ci]
        if k >= CAP:
            raise ValueError("cell capacity exceeded")
        members[ci, k] = i
        count[ci] = k + 1
        cell_of[i] = ci
        slot_of[i] = k
    return nx, ny, count, members, cell_of, slot_of


@njit(cache=True)
def _move_cell(i, newc, count, members, cell_of, slot_of):
    oc = cell_of[i]
    if oc == newc:
        return
    k = slot_of[i]
    last = count[oc] - 1
    j = members[oc, last]
    members[oc, k] = j
    slot_of[j] = k
    members[oc, last] = -1
    count[oc] = last
    kk = count[newc]
    if kk >= CAP:
        raise ValueError("cell capacity exceeded")
    members[newc, kk] = i
    count[newc] = kk + 1
    cell_of[i] = newc
    slot_of[i] = kk


# ----------------------------------------------------------------------------- energy
@njit(cache=True)
def local_count(i, sx, sy, s, a, b, c, lam, nx, ny, count, members):
    """number of shoulder pairs of particle i placed at (sx,sy); -1 if a core overlap occurs."""
    lam2 = lam * lam
    cx = int(sx * nx) % nx
    cy = int(sy * ny) % ny
    n = 0
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            ci = ((cx + ox) % nx) + nx * ((cy + oy) % ny)
            for m in range(count[ci]):
                j = members[ci, m]
                if j == i:
                    continue
                d2 = min_image_dist2(s[j, 0] - sx, s[j, 1] - sy, a, b, c)
                if d2 < 1.0:
                    return -1
                if d2 < lam2:
                    n += 1
    return n


@njit(cache=True)
def total_count(s, a, b, c, lam):
    """total shoulder pairs; -1 on any overlap."""
    nx, ny, count, members, cell_of, slot_of = build_cells(s, a, b, c, lam)
    tot = 0
    for i in range(s.shape[0]):
        n = local_count(i, s[i, 0], s[i, 1], s, a, b, c, lam, nx, ny, count, members)
        if n < 0:
            return -1
        tot += n
    return tot // 2


# ----------------------------------------------------------------------------- sweeps
@njit(cache=True)
def sweep(s, box, lam, beta, P, dmax, dbox, dshear, n_pairs, seed_state):
    """One MC sweep: N particle trials + 1 box trial. Returns (n_pairs, acc_part, acc_box)."""
    np.random.seed(seed_state)
    a, b, c = box[0], box[1], box[2]
    N = s.shape[0]
    nx, ny, count, members, cell_of, slot_of = build_cells(s, a, b, c, lam)
    acc_p = 0
    for t in range(N):
        i = np.random.randint(N)
        # cartesian displacement converted to fractional
        dx = dmax * (2.0 * np.random.random() - 1.0)
        dy = dmax * (2.0 * np.random.random() - 1.0)
        dsy = dy / c
        dsx = (dx - b * dsy) / a
        nsx = s[i, 0] + dsx
        nsy = s[i, 1] + dsy
        nsx -= math.floor(nsx)
        nsy -= math.floor(nsy)
        nnew = local_count(i, nsx, nsy, s, a, b, c, lam, nx, ny, count, members)
        if nnew < 0:
            continue
        nold = local_count(i, s[i, 0], s[i, 1], s, a, b, c, lam, nx, ny, count, members)
        dU = nnew - nold
        if dU <= 0 or np.random.random() < math.exp(-beta * dU):
            s[i, 0] = nsx
            s[i, 1] = nsy
            newc = (int(nsx * nx) % nx) + nx * (int(nsy * ny) % ny)
            _move_cell(i, newc, count, members, cell_of, slot_of)
            n_pairs += dU
            acc_p += 1
    # box move
    acc_b = 0
    na = a * math.exp(dbox * (2.0 * np.random.random() - 1.0))
    nc = c * math.exp(dbox * (2.0 * np.random.random() - 1.0))
    nb = b + dshear * (2.0 * np.random.random() - 1.0)
    # keep the shear reduced to |b| <= a/2 (lattice reduction keeps the same lattice)
    w1, w2 = perp_widths(na, nb, nc)
    if w1 > 3.0 * lam and w2 > 3.0 * lam and abs(nb) <= 0.5 * na:
        V = a * c
        nV = na * nc
        nn = total_count(s, na, nb, nc, lam)
        if nn >= 0:
            arg = -beta * ((nn - n_pairs) + P * (nV - V)) + (N + 1) * math.log(nV / V)
            if arg >= 0 or np.random.random() < math.exp(arg):
                box[0], box[1], box[2] = na, nb, nc
                n_pairs = nn
                acc_b = 1
    return n_pairs, acc_p, acc_b


# ----------------------------------------------------------------------------- initial states
def lattice_state(kind, lam, N_target, scale=1.003):
    """Return (s, box=(a,b,c)) for kind in {'A', 'B', 'rows', 'fluid'}.
    A    : triangular lattice, spacing lam*scale
    B    : thin-rhombus lattice (30 deg rhombi, side lam*scale)  -> pure B tiles at lam*
    rows : periodic row stacking T R+ R- T R+ R- (triangles : rhombi = 1 : 1, x_A = 1/3,
           same composition and density as the predicted 12-fold state)
    fluid: random sequential addition at density 0.25
    """
    L = lam * scale
    if kind == "A":
        nx = int(round(math.sqrt(N_target / (math.sqrt(3) / 2))))
        ny = int(round(N_target / nx)); ny += ny % 2
        a1 = np.array([L, 0.0]); a2 = np.array([L / 2, L * math.sqrt(3) / 2])
        pts = np.array([i * a1 + j * a2 for j in range(ny) for i in range(nx)])
        A = nx * a1; B2 = ny * a2
        # reduce: shift a2 by multiples of a1 so that |b| <= a/2
        b = B2[0] - round(B2[0] / A[0]) * A[0]
        box = np.array([A[0], b, B2[1]])
    elif kind == "B":
        al = math.radians(30.0)
        a1 = np.array([L, 0.0]); a2 = np.array([L * math.cos(al), L * math.sin(al)])
        ny = int(round(math.sqrt(N_target * 0.5))); ny += ny % 2
        nx = int(round(N_target / ny))
        pts = np.array([i * a1 + j * a2 for j in range(ny) for i in range(nx)])
        A = nx * a1; B2 = ny * a2
        b = B2[0] - round(B2[0] / A[0]) * A[0]
        box = np.array([A[0], b, B2[1]])
    elif kind == "rows":
        c30 = math.cos(math.radians(30.0))
        unit = [("T", L * math.sqrt(3) / 2, L / 2), ("R", L / 2, L * c30), ("R", L / 2, -L * c30)] * 2
        m = max(1, int(round(math.sqrt(N_target / 22.4))))
        n = int(round(N_target / (6 * m)))
        pts = []
        y = 0.0; x0 = 0.0
        for _ in range(m):
            for (_, hgt, shift) in unit:
                for i in range(n):
                    pts.append((x0 + i * L, y))
                y += hgt; x0 += shift
        pts = np.array(pts)
        box = np.array([n * L, (x0 % L) if False else 0.0, y])
        # total shift per 6-row unit is exactly L, so the stack is periodic with zero shear
    elif kind == "fluid":
        rho = 0.25
        side = math.sqrt(N_target / rho)
        box = np.array([side, 0.0, side])
        rng = np.random.default_rng(1)
        pts = []
        while len(pts) < N_target:
            p = rng.random(2) * side
            ok = True
            for q in pts:
                d = p - q
                d -= side * np.round(d / side)
                if d @ d < 1.05:
                    ok = False; break
            if ok:
                pts.append(p)
        pts = np.array(pts)
    else:
        raise ValueError(kind)
    a, b, c = box
    sy = pts[:, 1] / c
    sx = (pts[:, 0] - b * sy) / a
    s = np.stack([sx % 1.0, sy % 1.0], axis=1)
    return s, box.astype(float)


def cart(s, box):
    a, b, c = box
    return np.stack([a * s[:, 0] + b * s[:, 1], c * s[:, 1]], axis=1)


def box_matrix(box):
    a, b, c = box
    return np.array([[a, b], [0.0, c]])  # columns = box vectors
