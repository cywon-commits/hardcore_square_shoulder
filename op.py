"""
op.py — order parameters of the lam* = 2cos15 tiling phases (pilot 8).

Theory (see README):
  * phason strain E = dw/dr (2x2), written as w = alpha z + beta conj(z);  alpha has rotational charge 4, beta charge 6
    (12-fold symmetry forces E = 0; the 3.12.12 approximant has beta = eps_1 = (2-sqrt3)^2, alpha = 0).
  * eta = (n_R/2 - sqrt3 n_A/4) / (n_R/2 + sqrt3 n_A/4) = det E = |alpha|^2 - |beta|^2   (exact: perp-area density)
    with n_A = number of A triangles, n_R = number of 30-degree rhombi (= B triangles / 2).
  * bond texture: psi_4 of shoulder bonds pairs with alpha, psi_6 with beta (local proxies; calibrate on approximants).
Lift: shoulder edges (|r - lam| small) are quantised to directions 30k; physical step lam* e^{i pi k/6}, perp step
lam* e^{i 5 pi k/6}.  BFS over the shoulder-edge graph; every edge closing a loop gives a pair (dr, dw); loops that wind
around the periodic box give E by least squares (dw = E dr); short loops with dw != 0 are phason defects.
"""
import math, cmath
import numpy as np
from collections import deque
from scipy.spatial import cKDTree, Delaunay

LS = 2 * math.cos(math.radians(15))
S3 = math.sqrt(3)
ZP = [cmath.exp(1j * math.pi * k / 6) for k in range(12)]       # physical unit directions
ZW = [cmath.exp(1j * 5 * math.pi * k / 6) for k in range(12)]   # perp images (Galois conjugate, zeta -> zeta^5)


def _images(P, M):
    imgs, oid, shift = [], [], []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            imgs.append(P + i * M[:, 0] + j * M[:, 1]); oid.append(np.arange(len(P))); shift.append(np.tile([i, j], (len(P), 1)))
    return np.vstack(imgs), np.concatenate(oid), np.vstack(shift)


def bonds(P, M, lam=1.93, tol_in=0.06, tol_out=0.25, tol_core=0.08, max_dev_deg=12.0, return_raw=False):
    """Shoulder edges (i, j, k): every undirected pair is found ONCE, its raw angle measured once, and quantised to a
    30-degree class k relative to the local orientation (12-fold average of the bond angles at its two end vertices);
    then both (i, j, k) and (j, i, k+6) are stored, so the two directions can never disagree (pilot-8 artefact).
    Also returns core pairs (r < 1 + tol_core) and, if requested, the raw shoulder-bond angles."""
    Q, oid, _ = _images(P, M)
    tree = cKDTree(Q)
    raw = []                                     # (i, j, angle) for i < j (pairs with an image of itself are skipped)
    C = []
    for i, p in enumerate(P):
        for m in tree.query_ball_point(p, lam + tol_out):
            j = oid[m]
            if j <= i:
                continue
            d = Q[m] - p; r = math.hypot(*d)
            if lam - tol_in <= r < lam + tol_out:
                # Gabriel condition: no third particle inside the circle with diameter ij (a shoulder CONTACT has nothing
                # in between; e.g. two collinear core steps, r ~ 2.0, are rejected).  A and B tile edges always pass
                # (opposite angles 60 and 75 degrees < 90).
                mid = p + d / 2
                inside = [x for x in tree.query_ball_point(mid, r / 2 - 0.05) if oid[x] != i and oid[x] != j]
                if not inside:
                    raw.append((i, j, math.atan2(d[1], d[0])))
            elif 1e-9 < r < lam - tol_in:
                C.append((i, j, math.atan2(d[1], d[0])))      # all costly pairs (= energy count; core contacts in a tiling)
    # orientation: global offset theta0 first, then a SMALL local deviation per vertex (no +-15 degree branch ambiguity)
    tot = sum(cmath.exp(12j * a) for (_, _, a) in raw) if raw else 1
    theta0 = cmath.phase(tot) / 12.0
    acc = np.zeros(len(P), complex)
    for (i, j, a) in raw:
        z = cmath.exp(12j * (a - theta0)); acc[i] += z; acc[j] += z
    dev = np.where(np.abs(acc) > 0, np.angle(acc) / 12.0, 0.0)
    L = []
    for (i, j, a) in raw:
        off = theta0 + 0.5 * (dev[i] + dev[j])
        x = (a - off) / (math.pi / 6); k = int(round(x)) % 12
        if abs(x - round(x)) * 30.0 <= max_dev_deg:
            L.append((i, j, k)); L.append((j, i, (k + 6) % 12))
    if return_raw:
        return L, C, raw
    return L, C


def tile_counts(P, M, lam=1.93, tol_in=0.06, tol_out=0.25):
    """counts of Delaunay triangles by class (A = LLL, B = SLL, C = SSL, D = SSS, X = other), periodic, each once."""
    Q, oid, sh = _images(P, M)
    img = (np.abs(sh).sum(1) > 0).astype(int)
    simp = Delaunay(Q).simplices
    first = simp[np.arange(len(simp)), np.argmin(oid[simp], axis=1)]
    simp = simp[img[first] == 0]
    cnt = dict(A=0, B=0, C=0, D=0, X=0); cent = {k: [] for k in cnt}
    for s in simp:
        cls = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            r = np.linalg.norm(Q[s[b]] - Q[s[a]])
            cls.append("S" if r < lam - tol_in else ("L" if r < lam + tol_out else "X"))
        t = {"LLL": "A", "LLS": "B", "LSS": "C", "SSS": "D"}.get("".join(sorted(cls)), "X")
        cnt[t] += 1; cent[t].append(Q[s].mean(0))
    return cnt, {k: np.array(v).reshape(-1, 2) for k, v in cent.items()}


def eta_from_counts(nA, nB):
    nR = nB / 2.0
    den = nR / 2 + S3 * nA / 4
    return (nR / 2 - S3 * nA / 4) / den if den > 0 else float("nan")


def psi_raw(raw, n):
    """bond-orientational order from the UNQUANTISED shoulder-bond angles (each bond once; even n)."""
    if not raw:
        return 0j
    return np.mean([cmath.exp(1j * n * a) for (_, _, a) in raw])


def eta_from_core(N, n_R):
    """exact eta for a periodic tiling: Euler gives 2N triangles = n_A + 2 n_R, so n_A = 2N - 2 n_R; n_R = number of costly
    (core-contact) pairs, i.e. the energy count.  Robust against degenerate Delaunay choices and thermal misclassification."""
    nA = 2 * N - 2 * n_R
    den = n_R / 2 + S3 * nA / 4
    return (n_R / 2 - S3 * nA / 4) / den if den > 0 else float("nan")


def psi(L, n):
    """bond-orientational order of shoulder bonds (each undirected bond counted twice, which is harmless for even n)."""
    if not L:
        return 0j
    return np.mean([cmath.exp(1j * n * math.pi * k / 6) for (_, _, k) in L])


def E_to_alpha_beta(E):
    a, b, c, d = E[0, 0], E[0, 1], E[1, 0], E[1, 1]
    return complex((a + d) / 2, (c - b) / 2), complex((a - d) / 2, (c + b) / 2)


def lift(P, M, L, lam_ideal=LS):
    """BFS lift.  Returns dict with per-vertex unwrapped ideal position r (complex), perp w (complex), global E from
    winding loops, alpha, beta, numbers of winding loops and of defect loops, and the vertices reached."""
    N = len(P)
    adj = [[] for _ in range(N)]
    for (i, j, k) in L:
        adj[i].append((j, k))
    r = np.full(N, np.nan + 0j); w = np.full(N, np.nan + 0j)
    wind_dr, wind_dw = [], []; ndef = 0
    for s0 in range(N):
        if not np.isnan(r[s0].real) or not adj[s0]:
            continue
        r[s0] = 0; w[s0] = 0; q = deque([s0])
        while q:
            u = q.popleft()
            for (v, k) in adj[u]:
                rv = r[u] + lam_ideal * ZP[k]; wv = w[u] + lam_ideal * ZW[k]
                if np.isnan(r[v].real):
                    r[v] = rv; w[v] = wv; q.append(v)
                else:
                    dr = rv - r[v]; dw = wv - w[v]
                    if abs(dr) > 0.5 * lam_ideal:
                        wind_dr.append(dr); wind_dw.append(dw)
                    elif abs(dw) > 0.5 * lam_ideal:
                        ndef += 1
    out = dict(r=r, w=w, n_winding=len(wind_dr), n_defect_edges=ndef, reached=int((~np.isnan(r.real)).sum()))
    if len(wind_dr) >= 2:
        X = np.array([[z.real, z.imag] for z in wind_dr]); Y = np.array([[z.real, z.imag] for z in wind_dw])
        if np.linalg.matrix_rank(X, tol=1e-6) == 2:
            Et, *_ = np.linalg.lstsq(X, Y, rcond=None); E = Et.T
            al, be = E_to_alpha_beta(E)
            out.update(E=E, alpha=al, beta=be, detE=float(np.linalg.det(E)),
                       winding_residual=float(np.abs(X @ Et - Y).max()))
    return out


def order_parameters(P, M, lam=1.93, tol_in=0.06, tol_out=None, T=None, Pr=None):
    """global order parameters of one configuration (periodic box M with box vectors as columns)."""
    if tol_out is None:
        tol_out = 0.10 + 2.0 * T / (Pr * lam) if (T and Pr) else 0.25
    L, C, raw = bonds(P, M, lam, tol_in, tol_out, return_raw=True)
    cnt, _ = tile_counts(P, M, lam, tol_in, tol_out)
    tot = sum(cnt.values())
    eta_tiles = eta_from_counts(cnt["A"], cnt["B"])
    eta = eta_from_core(len(P), len(C))
    lf = lift(P, M, L)
    out = dict(N=len(P), counts=cnt, x_A=cnt["A"] / tot, x_B=cnt["B"] / tot, x_other=(cnt["C"] + cnt["D"] + cnt["X"]) / tot,
               eta=eta, eta_tiles=eta_tiles, n_core_pairs_exact=len(C),
               psi2=abs(psi_raw(raw, 2)), psi4=abs(psi_raw(raw, 4)), psi6=abs(psi_raw(raw, 6)), psi12=abs(psi_raw(raw, 12)),
               arg_psi4=float(np.angle(psi(L, 4))), arg_psi6=float(np.angle(psi(L, 6))),
               n_shoulder_bonds=len(raw),
               lift_reached=lf["reached"], n_winding=lf["n_winding"], n_defect_edges=lf["n_defect_edges"])
    if "E" in lf:
        out.update(detE=lf["detE"], abs_alpha=abs(lf["alpha"]), abs_beta=abs(lf["beta"]), winding_residual=lf["winding_residual"],
                   E=lf["E"].tolist())
    return out, (L, lf)


def phason_spectrum(P, M, lf, nshell=6):
    """<|w_q|^2> for the residual w~ = w - E r (periodic if E is exact) on the smallest reciprocal vectors.
    Returns list of (|q|, w_q (complex), (m1, m2)) with w_q = (1/N) sum_j w~_j exp(-i q.r_j) (continuum normalisation)."""
    if "E" not in lf:
        return []
    r = lf["r"]; w = lf["w"]; ok = ~np.isnan(r.real)
    E = lf["E"]
    R = np.stack([r.real, r.imag], 1); W = np.stack([w.real, w.imag], 1)
    Wr = W - R @ E.T
    wr = Wr[:, 0] + 1j * Wr[:, 1]
    wr = wr[ok]; Pk = P[ok]; wr = wr - wr.mean()
    rec = 2 * math.pi * np.linalg.inv(M).T
    out = []
    for m1 in range(-nshell, nshell + 1):
        for m2 in range(-nshell, nshell + 1):
            if (m1, m2) == (0, 0):
                continue
            q = m1 * rec[:, 0] + m2 * rec[:, 1]
            wq = np.mean(wr * np.exp(-1j * (Pk @ q)))
            out.append((float(np.linalg.norm(q)), complex(wq), (m1, m2)))
    return out


def local_profiles(P, M, L, nbins=24, lam=1.93, tol_in=0.06, tol_out=0.25):
    """profiles along the fractional x coordinate (slab geometry): eta from tile counts, psi4, psi6, psi12 of bonds."""
    inv = np.linalg.inv(M)
    cnt, cen = tile_counts(P, M, lam, tol_in, tol_out)
    nA = np.zeros(nbins); nB = np.zeros(nbins); nX = np.zeros(nbins)
    for k, arr in cen.items():
        if len(arr) == 0:
            continue
        fx = (arr @ inv.T)[:, 0] % 1.0; b = np.minimum((fx * nbins).astype(int), nbins - 1)
        if k == "A": np.add.at(nA, b, 1)
        elif k == "B": np.add.at(nB, b, 1)
        else: np.add.at(nX, b, 1)
    eta = np.array([eta_from_counts(a, b_) if a + b_ > 0 else np.nan for a, b_ in zip(nA, nB)])
    acc = {n: np.zeros(nbins, complex) for n in (4, 6, 12)}; cnts = np.zeros(nbins)
    fx = (P @ inv.T)[:, 0] % 1.0
    for (i, j, k) in L:
        b = min(int(fx[i] * nbins), nbins - 1)
        for n in acc: acc[n][b] += cmath.exp(1j * n * math.pi * k / 6)
        cnts[b] += 1
    prof = dict(eta=eta.tolist(), x_A=(nA / np.maximum(nA + nB + nX, 1)).tolist(), x_X=(nX / np.maximum(nA + nB + nX, 1)).tolist())
    for n in acc:
        prof[f"psi{n}"] = (np.abs(acc[n]) / np.maximum(cnts, 1)).tolist()
    return prof
