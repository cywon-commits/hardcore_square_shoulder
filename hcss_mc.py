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
def lattice_state(kind, lam, N_target, scale=1.003, seed=0):
    """Return (s, box=(a,b,c)) for kind in {'A', 'B', 'rows', 'fluid'}.
    A    : triangular lattice, spacing lam*scale
    B    : thin-rhombus lattice (30 deg rhombi, side lam*scale)  -> pure B tiles at lam*
    rows : periodic row stacking T R+ R- T R+ R- (triangles : rhombi = 1 : 1, x_A = 1/3,
           same composition and density as the predicted 12-fold state)
    fluid: random sequential addition at density 0.25
    hexlat: periodic lattice of 30-degree hexagons with random interior positions (a random tiling with
           x_A = 1/3 and density rho_12; every interior particle is flippable)
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
    elif kind == "hexlat":
        # lattice of 30-degree hexagons (edge vectors u, v, w at 0, 30, 60 deg), translation vectors
        # T1 = u + v, T2 = v + w; per cell: vertices 0 and u, interior at v or u + w (random -> random tiling)
        d = math.radians
        u = L * np.array([1.0, 0.0]); v = L * np.array([math.cos(d(30)), math.sin(d(30))])
        w = L * np.array([math.cos(d(60)), math.sin(d(60))])
        T1 = u + v; T2 = v + w
        n1 = max(3, int(round(math.sqrt(N_target / 6.0))))
        n2 = max(6, int(round(N_target / (3 * n1))))
        rng = np.random.default_rng(seed)
        pts = []
        for i in range(n1):
            for j in range(n2):
                O = i * T1 + j * T2
                pts += [O, O + u, O + (v if rng.random() < 0.5 else u + w)]
        pts = np.array(pts)
        A1 = n1 * T1; A2 = n2 * T2
        th = math.atan2(A1[1], A1[0]); R = np.array([[math.cos(-th), -math.sin(-th)], [math.sin(-th), math.cos(-th)]])
        pts = pts @ R.T; A1 = R @ A1; A2 = R @ A2
        b = A2[0] - round(A2[0] / A1[0]) * A1[0]
        box = np.array([A1[0], b, A2[1]])
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


# ============================================================================ pilot 2 additions
# ---------------------------------------------------------------------------- sweep with image tracking
@njit(cache=True)
def sweep_img(s, img, box, lam, beta, P, dmax, dbox, dshear, n_pairs, seed_state):
    """Same as sweep() but also tracks periodic image counters img (N,2) for unwrapped trajectories."""
    np.random.seed(seed_state)
    a, b, c = box[0], box[1], box[2]
    N = s.shape[0]
    nx, ny, count, members, cell_of, slot_of = build_cells(s, a, b, c, lam)
    acc_p = 0
    for t in range(N):
        i = np.random.randint(N)
        dx = dmax * (2.0 * np.random.random() - 1.0)
        dy = dmax * (2.0 * np.random.random() - 1.0)
        dsy = dy / c
        dsx = (dx - b * dsy) / a
        rx = s[i, 0] + dsx
        ry = s[i, 1] + dsy
        fx = math.floor(rx); fy = math.floor(ry)
        nsx = rx - fx; nsy = ry - fy
        nnew = local_count(i, nsx, nsy, s, a, b, c, lam, nx, ny, count, members)
        if nnew < 0:
            continue
        nold = local_count(i, s[i, 0], s[i, 1], s, a, b, c, lam, nx, ny, count, members)
        dU = nnew - nold
        if dU <= 0 or np.random.random() < math.exp(-beta * dU):
            s[i, 0] = nsx; s[i, 1] = nsy
            img[i, 0] += int(fx); img[i, 1] += int(fy)
            newc = (int(nsx * nx) % nx) + nx * (int(nsy * ny) % ny)
            _move_cell(i, newc, count, members, cell_of, slot_of)
            n_pairs += dU
            acc_p += 1
    acc_b = 0
    na = a * math.exp(dbox * (2.0 * np.random.random() - 1.0))
    nc = c * math.exp(dbox * (2.0 * np.random.random() - 1.0))
    nb = b + dshear * (2.0 * np.random.random() - 1.0)
    w1, w2 = perp_widths(na, nb, nc)
    if w1 > 3.0 * lam and w2 > 3.0 * lam and abs(nb) <= 0.5 * na:
        V = a * c; nV = na * nc
        nn = total_count(s, na, nb, nc, lam)
        if nn >= 0:
            arg = -beta * ((nn - n_pairs) + P * (nV - V)) + (N + 1) * math.log(nV / V)
            if arg >= 0 or np.random.random() < math.exp(arg):
                box[0], box[1], box[2] = na, nb, nc
                n_pairs = nn
                acc_b = 1
    return n_pairs, acc_p, acc_b


# ---------------------------------------------------------------------------- 30-degree hexagon flip
# Geometry (lam* = 2cos15): interior particle i with three L-neighbours r0, r2, r4 (relative vectors);
# the other three hexagon vertices sit at r0+r2, r0+r4, r2+r4.  The alternative filling puts i at
# x' = x + (r0 + r2 + r4)  = point reflection of x through the centre of the six boundary particles.
# Proposal: choose i uniformly, choose one of its n_old candidate hexagons uniformly, reflect through the
# centre C computed from the six boundary positions.  Reverse move must find the same boundary set
# (else reject).  Acceptance min(1, n_old/n_new * exp(-beta dU)).  Jacobian of a point reflection = 1.
MAXNB = 96
MAXHEX = 16


@njit(cache=True)
def _neigh(i, x0, x1, s, a, b, c, rmax, nx, ny, count, members, ids, rel):
    """collect particles j != i within rmax of point (x0,x1) [fractional]; relative cartesian vectors."""
    r2 = rmax * rmax
    cx = int(x0 * nx) % nx; cy = int(x1 * ny) % ny
    m = 0
    for ox in range(-2, 3):
        for oy in range(-2, 3):
            ci = ((cx + ox) % nx) + nx * ((cy + oy) % ny)
            for k in range(count[ci]):
                j = members[ci, k]
                if j == i:
                    continue
                dsx = s[j, 0] - x0; dsy = s[j, 1] - x1
                dsx -= math.floor(dsx + 0.5); dsy -= math.floor(dsy + 0.5)
                dx = a * dsx + b * dsy; dy = c * dsy
                best = dx * dx + dy * dy; bx = dx
                for kk in (-1, 1):
                    ddx = dx + kk * a
                    if ddx * ddx + dy * dy < best:
                        best = ddx * ddx + dy * dy; bx = ddx
                if best < r2 and m < MAXNB:
                    ids[m] = j; rel[m, 0] = bx; rel[m, 1] = dy; m += 1
    return m


@njit(cache=True)
def _angdiff(t1, t2):
    d = t2 - t1
    while d < 0:
        d += 2 * math.pi
    while d >= 2 * math.pi:
        d -= 2 * math.pi
    return d


@njit(cache=True)
def find_hexagons(i, x0, x1, s, a, b, c, lam, nx, ny, count, members, tl, tu, tp, tang, hex_ids, hex_C):
    """Candidate flip hexagons for particle i placed at fractional (x0,x1).
    Returns number found; hex_ids[k,:6] = sorted boundary ids, hex_C[k,:] = 2*centre (relative, cartesian)."""
    ids = np.empty(MAXNB, np.int64); rel = np.empty((MAXNB, 2))
    m = _neigh(i, x0, x1, s, a, b, c, math.sqrt(3.0) * lam + tu + tp, nx, ny, count, members, ids, rel)
    L = np.empty(MAXNB, np.int64); nL = 0
    for k in range(m):
        r = math.sqrt(rel[k, 0] ** 2 + rel[k, 1] ** 2)
        if lam - tl <= r < lam + tu:
            L[nL] = k; nL += 1
    nh = 0
    big = math.radians(150.0); small = math.radians(60.0)
    for p in range(nL):
        for q in range(p + 1, nL):
            for t in range(q + 1, nL):
                k3 = np.array([L[p], L[q], L[t]])
                th = np.array([math.atan2(rel[k3[0], 1], rel[k3[0], 0]),
                               math.atan2(rel[k3[1], 1], rel[k3[1], 0]),
                               math.atan2(rel[k3[2], 1], rel[k3[2], 0])])
                order = np.argsort(th)
                k3 = k3[order]; th = th[order]
                g = np.array([_angdiff(th[0], th[1]), _angdiff(th[1], th[2]), _angdiff(th[2], th[0])])
                # need one ~60 deg gap and two ~150 deg gaps
                js = -1
                for z in range(3):
                    if abs(g[z] - small) < tang and abs(g[(z + 1) % 3] - big) < tang and abs(g[(z + 2) % 3] - big) < tang:
                        js = z
                if js < 0:
                    continue
                # gap js runs from k3[js] (= H2) to k3[js+1] (= H4); remaining one is H0
                r2v = rel[k3[js]]; r4v = rel[k3[(js + 1) % 3]]; r0v = rel[k3[(js + 2) % 3]]
                pred = np.empty((3, 2))
                pred[0] = r0v + r2v; pred[1] = r0v + r4v; pred[2] = r2v + r4v
                found = np.full(3, -1, np.int64)
                ok = True
                for z in range(3):
                    cnt = 0
                    for k in range(m):
                        dx = rel[k, 0] - pred[z, 0]; dy = rel[k, 1] - pred[z, 1]
                        if dx * dx + dy * dy < tp * tp:
                            found[z] = k; cnt += 1
                    if cnt != 1:
                        ok = False; break
                if not ok:
                    continue
                bset = np.array([ids[k3[0]], ids[k3[1]], ids[k3[2]], ids[found[0]], ids[found[1]], ids[found[2]]])
                bset.sort()
                dup = False
                for z in range(1, 6):
                    if bset[z] == bset[z - 1]:
                        dup = True
                if dup or nh >= MAXHEX:
                    continue
                Cx = (rel[k3[0], 0] + rel[k3[1], 0] + rel[k3[2], 0] + rel[found[0], 0] + rel[found[1], 0] + rel[found[2], 0]) / 3.0
                Cy = (rel[k3[0], 1] + rel[k3[1], 1] + rel[k3[2], 1] + rel[found[0], 1] + rel[found[1], 1] + rel[found[2], 1]) / 3.0
                hex_ids[nh, :] = bset
                hex_C[nh, 0] = Cx; hex_C[nh, 1] = Cy   # = 2 * centre, i.e. the displacement of i
                nh += 1
    return nh


@njit(cache=True)
def flip_moves(s, img, box, lam, beta, n_attempt, n_pairs, seed_state, tl, tu, tp, tang):
    """n_attempt hexagon-flip attempts. Returns (n_pairs, n_candidates_seen, n_proposed, n_accepted)."""
    np.random.seed(seed_state)
    a, b, c = box[0], box[1], box[2]
    N = s.shape[0]
    nx, ny, count, members, cell_of, slot_of = build_cells(s, a, b, c, lam)
    if nx < 5 or ny < 5:
        return n_pairs, 0, 0, 0
    hid = np.empty((MAXHEX, 6), np.int64); hC = np.empty((MAXHEX, 2))
    hid2 = np.empty((MAXHEX, 6), np.int64); hC2 = np.empty((MAXHEX, 2))
    seen = 0; prop = 0; acc = 0
    for t in range(n_attempt):
        i = np.random.randint(N)
        n_old = find_hexagons(i, s[i, 0], s[i, 1], s, a, b, c, lam, nx, ny, count, members, tl, tu, tp, tang, hid, hC)
        if n_old == 0:
            continue
        seen += 1
        k = np.random.randint(n_old)
        dx = hC[k, 0]; dy = hC[k, 1]
        dsy = dy / c; dsx = (dx - b * dsy) / a
        rx = s[i, 0] + dsx; ry = s[i, 1] + dsy
        fx = math.floor(rx); fy = math.floor(ry)
        nsx = rx - fx; nsy = ry - fy
        nnew = local_count(i, nsx, nsy, s, a, b, c, lam, nx, ny, count, members)
        if nnew < 0:
            continue
        prop += 1
        n_new = find_hexagons(i, nsx, nsy, s, a, b, c, lam, nx, ny, count, members, tl, tu, tp, tang, hid2, hC2)
        rev = False
        for z in range(n_new):
            same = True
            for q in range(6):
                if hid2[z, q] != hid[k, q]:
                    same = False
            if same:
                rev = True
        if not rev:
            continue
        nold = local_count(i, s[i, 0], s[i, 1], s, a, b, c, lam, nx, ny, count, members)
        dU = nnew - nold
        arg = math.log(n_old / n_new) - beta * dU
        if arg >= 0 or np.random.random() < math.exp(arg):
            s[i, 0] = nsx; s[i, 1] = nsy
            img[i, 0] += int(fx); img[i, 1] += int(fy)
            newc = (int(nsx * nx) % nx) + nx * (int(nsy * ny) % ny)
            _move_cell(i, newc, count, members, cell_of, slot_of)
            n_pairs += dU
            acc += 1
    return n_pairs, seen, prop, acc


def hexagon_state(lam, scale=1.002, box_side=40.0, noise=0.0, seed=0):
    """Isolated flippable 30-degree hexagon (7 particles) in a large periodic box, for tests."""
    L = lam * scale
    u = np.array([L, 0.0]); v = L * np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
    w = L * np.array([math.cos(math.radians(60)), math.sin(math.radians(60))])
    O = np.array([box_side / 2 - 2.0, box_side / 2 - 1.5])
    pts = np.array([O, O + u, O + u + v, O + u + v + w, O + v + w, O + w, O + v])  # last = interior
    if noise > 0:
        pts = pts + np.random.default_rng(seed).normal(0, noise, pts.shape)
    box = np.array([box_side, 0.0, box_side])
    s = (pts / box_side) % 1.0
    return s, box


# ============================================================================ pilot 3 additions
# ---------------------------------------------------------------------------- Frenkel-Ladd (Einstein molecule)
@njit(cache=True)
def sweep_fl(s, s0, box, lam, beta, Lam, dmax, n_pairs, seed_state):
    """NVT sweep with harmonic tethers beta*U_spring = Lam * sum_{i>=1} |r_i - r0_i|^2.
    Particle 0 is held fixed (Einstein-molecule reference, Vega & Noya 2007).
    Returns (n_pairs, accepted, sum_{i>=1} |r_i - r0_i|^2 after the sweep)."""
    np.random.seed(seed_state)
    a, b, c = box[0], box[1], box[2]
    N = s.shape[0]
    nx, ny, count, members, cell_of, slot_of = build_cells(s, a, b, c, lam)
    acc = 0
    for t in range(N - 1):
        i = 1 + np.random.randint(N - 1)
        dx = dmax * (2.0 * np.random.random() - 1.0)
        dy = dmax * (2.0 * np.random.random() - 1.0)
        dsy = dy / c; dsx = (dx - b * dsy) / a
        nsx = s[i, 0] + dsx; nsy = s[i, 1] + dsy
        nsx -= math.floor(nsx); nsy -= math.floor(nsy)
        nnew = local_count(i, nsx, nsy, s, a, b, c, lam, nx, ny, count, members)
        if nnew < 0:
            continue
        nold = local_count(i, s[i, 0], s[i, 1], s, a, b, c, lam, nx, ny, count, members)
        d_old = min_image_dist2(s[i, 0] - s0[i, 0], s[i, 1] - s0[i, 1], a, b, c)
        d_new = min_image_dist2(nsx - s0[i, 0], nsy - s0[i, 1], a, b, c)
        arg = -beta * (nnew - nold) - Lam * (d_new - d_old)
        if arg >= 0 or np.random.random() < math.exp(arg):
            s[i, 0] = nsx; s[i, 1] = nsy
            newc = (int(nsx * nx) % nx) + nx * (int(nsy * ny) % ny)
            _move_cell(i, newc, count, members, cell_of, slot_of)
            n_pairs += nnew - nold
            acc += 1
    tot = 0.0
    for i in range(1, N):
        tot += min_image_dist2(s[i, 0] - s0[i, 0], s[i, 1] - s0[i, 1], a, b, c)
    return n_pairs, acc, tot


def einstein_dA1(s0, box, lam, beta, Lam, nsamp=2000, seed=0):
    """beta*dA1 = -ln < exp(-beta U_pair) >_EinsteinMolecule(Lam)  (overlaps -> weight 0).
    Returns (beta_dA1, fraction_overlap_free, mean_pairs)."""
    rng = np.random.default_rng(seed)
    a, b, c = box
    sig = 1.0 / math.sqrt(2.0 * Lam)
    vals = []
    nfree = 0
    for _ in range(nsamp):
        d = rng.normal(0.0, sig, s0.shape); d[0] = 0.0
        dsy = d[:, 1] / c; dsx = (d[:, 0] - b * dsy) / a
        s = np.stack([(s0[:, 0] + dsx) % 1.0, (s0[:, 1] + dsy) % 1.0], axis=1)
        n = total_count(s, a, b, c, lam)
        if n >= 0:
            vals.append(n); nfree += 1
    if not vals:
        return float("inf"), 0.0, float("nan")
    v = np.array(vals, float); m = v.min()
    lnmean = -beta * m + math.log(np.exp(-beta * (v - m)).sum() / nsamp)
    return -lnmean, nfree / nsamp, float(v.mean())


# ---------------------------------------------------------------------------- slab (interface) states
def _rot_to_y(vec):
    th = math.atan2(vec[1], vec[0])
    phi = math.pi / 2 - th
    return np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])


def _phase_points(kind, L, ny, nrep, seed):
    """Points of a slab that is periodic along y with period ny*L. Returns (pts, thickness)."""
    d = math.radians
    rng = np.random.default_rng(seed)
    if kind in ("A", "B"):
        ang = 60.0 if kind == "A" else 30.0
        a1 = np.array([0.0, L]); a2 = L * np.array([math.sin(d(ang)), math.cos(d(ang))])
        pts = np.array([i * a1 + j * a2 for j in range(nrep) for i in range(ny)])
        thick = nrep * a2[0]
    elif kind == "hexlat":
        u = L * np.array([1.0, 0.0]); v = L * np.array([math.cos(d(30)), math.sin(d(30))])
        w = L * np.array([math.cos(d(60)), math.sin(d(60))])
        T1 = u + v; T2 = v + w
        R = _rot_to_y(T1 - T2)
        u, v, w, T1, T2 = (R @ x for x in (u, v, w, T1, T2))
        if T2[0] < 0:   # mirror so the slab grows towards +x
            M = np.diag([-1.0, 1.0]); u, v, w, T1, T2 = (M @ x for x in (u, v, w, T1, T2))
        P = []
        for k in range(nrep):
            for i in range(ny):
                O = k * T2 + i * (T1 - T2)
                P += [O, O + u, O + (v if rng.random() < 0.5 else u + w)]
        pts = np.array(P)
        thick = nrep * T2[0]
    else:
        raise ValueError(kind)
    Ly = ny * L
    pts[:, 1] %= Ly
    pts[:, 0] -= pts[:, 0].min()
    return pts, thick


def slab_state(left, right, lam, ny, nL, nR, scale, seed=0, dmin=1.02):
    """Two slabs (left | right) side by side, periodic in x and y, two interfaces.
    The common y-period is ny*L (L = lam*scale): A and B use the lattice vector of length L, the hexagon
    lattice uses T1 - T2 = u - w (also length L), so the slabs match without strain."""
    L = lam * scale
    pL, tL = _phase_points(left, L, ny, nL, seed)
    pR, tR = _phase_points(right, L, ny, nR, seed + 1)
    Ly = ny * L
    wL = pL[:, 0].max(); wR = pR[:, 0].max()

    def mind(P, Q, Lx):
        best = 1e9
        for sx in (-Lx, 0.0, Lx):
            for sy in (-Ly, 0.0, Ly):
                d = P[:, None, :] - (Q[None, :, :] + np.array([sx, sy]))
                best = min(best, np.sqrt((d ** 2).sum(-1)).min())
        return best
    g = 0.3
    while True:
        Lx = wL + wR + 2 * g + 1e-9
        Q = pR + np.array([wL + g, 0.0])
        if mind(pL, Q, Lx) >= dmin:
            break
        g += 0.02
    pts = np.vstack([pL, pR + np.array([wL + g, 0.0])])
    labels = np.array([0] * len(pL) + [1] * len(pR))
    box = np.array([Lx, 0.0, Ly])
    s = np.stack([(pts[:, 0] / Lx) % 1.0, (pts[:, 1] / Ly) % 1.0], axis=1)
    return s, box, labels


# ============================================================================ pilot 3b additions: 3.12.12 tiling
import os as _os
_FILL = None


def dodecagon_fillings():
    """(4421, 13, 2) interior vertices of all fillings of a unit-edge 12-gon (relative to its centre)."""
    global _FILL
    if _FILL is None:
        p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "dodecagon_fillings.npz")
        _FILL = np.load(p)["interior"]
    return _FILL


def _dodecagon_vertices(L):
    R = L / (2 * math.sin(math.radians(15)))
    return np.array([[R * math.cos(math.radians(15 + 30 * k)), R * math.sin(math.radians(15 + 30 * k))] for k in range(12)])


def dodeca_cell_points(L, centres, rng):
    """All particles for 12-gons at the given centres: 12-gon vertices (to be de-duplicated by the caller)
    and the 13 interior vertices of an independently chosen random filling (s_conf >= ln(4421)/19 per particle)."""
    F = dodecagon_fillings()
    V = _dodecagon_vertices(L)
    out_v, out_i = [], []
    for c in centres:
        out_v.append(V + c)
        out_i.append(F[rng.integers(len(F))] * L + c)
    return np.vstack(out_v), np.vstack(out_i)


def _dedupe_periodic(P, M, tol=1e-6):
    inv = np.linalg.inv(M)
    f = (P @ inv.T) % 1.0
    f[np.abs(f - 1.0) < tol] = 0.0
    key = np.round(f / 1e-7).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


def dodeca_state(lam, N_target, scale=1.003, seed=0):
    """Periodic 3.12.12 tiling (12-gons + triangles, all of edge L = lam*scale) with every 12-gon filled by an
    independent random choice among its 4421 tilings.  19 particles and 14 A + 24 B tiles per 12-gon."""
    L = lam * scale
    D = L / math.tan(math.radians(15))            # centre-centre distance of edge-sharing 12-gons
    a1 = np.array([D, 0.0]); a2 = np.array([D / 2, D * math.sqrt(3) / 2])
    n = max(2, int(round(math.sqrt(N_target / 19.0))))
    M = np.column_stack([n * a1, n * a2])
    rng = np.random.default_rng(seed)
    centres = [i * a1 + j * a2 for i in range(n) for j in range(n)]
    Vb, Vi = dodeca_cell_points(L, centres, rng)
    Vb = _dedupe_periodic(Vb, M)
    pts = np.vstack([Vb, Vi])
    A1 = M[:, 0]; A2 = M[:, 1]
    b = A2[0] - round(A2[0] / A1[0]) * A1[0]
    box = np.array([A1[0], b, A2[1]])
    sy = pts[:, 1] / box[2]; sx = (pts[:, 0] - box[1] * sy) / box[0]
    s = np.stack([sx % 1.0, sy % 1.0], axis=1)
    return s, box


_lattice_state_base = lattice_state


def lattice_state(kind, lam, N_target, scale=1.003, seed=0):
    if kind == "dodeca":
        return dodeca_state(lam, N_target, scale=scale, seed=seed)
    return _lattice_state_base(kind, lam, N_target, scale=scale, seed=seed)


def _phase_points_general(kind, L, ny, nrep, seed):
    """Slab points periodic in y. Returns (pts, natural y-period)."""
    if kind in ("A", "B", "hexlat"):
        pts, _ = _phase_points(kind, L, ny, nrep, seed)
        return pts, ny * L
    if kind == "dodeca":
        D = L / math.tan(math.radians(15))
        a1 = np.array([0.0, D]); a2 = np.array([D * math.sqrt(3) / 2, D / 2])   # a1 along y
        rng = np.random.default_rng(seed)
        centres = [i * a1 + j * a2 for j in range(nrep) for i in range(ny)]
        R = np.array([[0.0, -1.0], [1.0, 0.0]])          # rotate 12-gon/fillings by 90 deg to match the frame
        F = dodecagon_fillings(); V = _dodecagon_vertices(L)
        Pv, Pi = [], []
        for c in centres:
            Pv.append(V @ R.T + c); Pi.append((F[rng.integers(len(F))] * L) @ R.T + c)
        Ly = ny * D
        Pv = np.vstack(Pv); Pv[:, 1] %= Ly
        key = np.round(np.stack([Pv[:, 0], Pv[:, 1] % Ly], 1) / 1e-6).astype(np.int64)
        key[:, 1] %= int(round(Ly / 1e-6))
        _, idx = np.unique(key, axis=0, return_index=True)
        pts = np.vstack([Pv[np.sort(idx)], np.vstack(Pi)])
        pts[:, 1] %= Ly
        pts[:, 0] -= pts[:, 0].min()
        return pts, Ly
    raise ValueError(kind)


def slab_state2(left, right, lam, nyL, nyR, nL, nR, scale, seed=0, dmin=1.02):
    """Like slab_state, but each slab has its own natural y-period; the shorter one is stretched to the longer
    (choose nyL, nyR so the mismatch is < ~0.5 %; stretching only opens gaps, it never creates overlaps)."""
    L = lam * scale
    pL, LyL = _phase_points_general(left, L, nyL, nL, seed)
    pR, LyR = _phase_points_general(right, L, nyR, nR, seed + 1)
    Ly = max(LyL, LyR)
    pL[:, 1] *= Ly / LyL; pR[:, 1] *= Ly / LyR
    wL = pL[:, 0].max(); wR = pR[:, 0].max()

    def mind(P, Q, Lx):
        best = 1e9
        for sx in (-Lx, 0.0, Lx):
            for sy in (-Ly, 0.0, Ly):
                d = P[:, None, :] - (Q[None, :, :] + np.array([sx, sy]))
                best = min(best, np.sqrt((d ** 2).sum(-1)).min())
        return best
    g = 0.3
    while True:
        Lx = wL + wR + 2 * g + 1e-9
        if mind(pL, pR + np.array([wL + g, 0.0]), Lx) >= dmin:
            break
        g += 0.02
    pts = np.vstack([pL, pR + np.array([wL + g, 0.0])])
    labels = np.array([0] * len(pL) + [1] * len(pR))
    box = np.array([Lx, 0.0, Ly])
    s = np.stack([(pts[:, 0] / Lx) % 1.0, (pts[:, 1] / Ly) % 1.0], axis=1)
    return s, box, labels, abs(LyL - LyR) / Ly


def match_periods(left, right, lam, scale, target=50.0, nmax=40):
    """Integers (nyL, nyR) whose natural y-periods agree best, with Ly near `target`."""
    L = lam * scale
    per = {"A": L, "B": L, "hexlat": L, "dodeca": L / math.tan(math.radians(15))}
    best = None
    for nR in range(1, nmax):
        LyR = nR * per[right]
        nL = max(1, int(round(LyR / per[left])))
        mis = abs(nL * per[left] - LyR) / LyR
        score = mis + 0.002 * abs(LyR - target) / target
        if best is None or score < best[0]:
            best = (score, nL, nR, mis)
    return best[1], best[2], best[3]


# ============================================================================ pilot 4: lambda != lambda*
LAM_STAR = 2 * math.cos(math.radians(15))
_lattice_state_3b = lattice_state


def lattice_state(kind, lam, N_target, scale=1.003, seed=0):
    """Pilot-4 dispatcher.
    B      : contact B crystal for any lam (rhombus lattice with apex alpha_B = 2 asin(1/(2 lam)), sides lam, short
             diagonal 1); equals the 30-degree lattice at lam*.
    dodeca, hexlat : 30-degree tilings with edge max(lam, lam*) (below lam* the rhombus short diagonal would be < 1,
             so the tiling can only exist stretched to lam*; above lam* its core pairs are no longer in contact)."""
    if kind == "B":
        L = lam * scale
        al = 2 * math.asin(1 / (2 * lam))
        a1 = np.array([L, 0.0]); a2 = np.array([L * math.cos(al), L * math.sin(al)])
        ny = int(round(math.sqrt(N_target * 0.5))); ny += ny % 2
        nx = int(round(N_target / ny))
        pts = np.array([i * a1 + j * a2 for j in range(ny) for i in range(nx)])
        A = nx * a1; B2 = ny * a2
        b = B2[0] - round(B2[0] / A[0]) * A[0]
        box = np.array([A[0], b, B2[1]])
        sy = pts[:, 1] / box[2]; sx = (pts[:, 0] - box[1] * sy) / box[0]
        return np.stack([sx % 1.0, sy % 1.0], axis=1), box
    if kind in ("dodeca", "hexlat"):
        return _lattice_state_3b(kind, max(lam, LAM_STAR), N_target, scale=scale, seed=seed)
    return _lattice_state_3b(kind, lam, N_target, scale=scale, seed=seed)


# ============================================================================ pilot 6: single-filling 3.12.12 (ideal / jammed)
def dodeca1_state(lam, N_target, scale=1.003, seed=0, jammed=False):
    """3.12.12 tiling with ONE filling (index chosen by seed) in every 12-gon, replicated n x n.
    jammed=False: ideal geometry with edge max(lam, lam*);  jammed=True and lam < lam*: the T = 0 jammed
    (isostatic) geometry of the fixed contact network (core pairs >= 1, all other pairs >= lam), from jam_volume.jam.
    Lengths are finally multiplied by `scale`."""
    import jam_volume as JV
    F = dodecagon_fillings()
    fill = int(np.random.default_rng(seed).integers(len(F)))
    if jammed and lam < LAM_STAR:
        r = JV.jam([fill], lam)
        if not r["success"] or r["newS"] > 0:
            raise RuntimeError("jam optimisation failed")
        P = r["P"]; a, b, c = r["cell"]
    else:
        P, M = JV.cell([fill], max(lam, LAM_STAR), 1)
        P = P - P[0]; a, b, c = M[0, 0], M[0, 1], M[1, 1]
    n = max(2, int(round(math.sqrt(N_target / 19.0))))
    A1 = np.array([a, 0.0]); A2 = np.array([b, c])
    pts = np.vstack([P + i * A1 + j * A2 for i in range(n) for j in range(n)]) * scale
    box = np.array([n * a, n * b, n * c]) * scale
    box[1] = box[1] - round(box[1] / box[0]) * box[0]
    sy = pts[:, 1] / box[2]; sx = (pts[:, 0] - box[1] * sy) / box[0]
    return np.stack([sx % 1.0, sy % 1.0], axis=1), box


_lattice_state_p5 = lattice_state


def lattice_state(kind, lam, N_target, scale=1.003, seed=0):
    if kind == "dodeca1":
        return dodeca1_state(lam, N_target, scale=scale, seed=seed, jammed=False)
    if kind == "dodeca1_jam":
        return dodeca1_state(lam, N_target, scale=scale, seed=seed, jammed=True)
    return _lattice_state_p5(kind, lam, N_target, scale=scale, seed=seed)
