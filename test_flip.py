"""
test_flip.py — detailed-balance and bookkeeping tests for the 30-degree hexagon flip.
Run with  python -m pytest -q test_flip.py
"""
import math
import numpy as np
import hcss_mc as mc

LAM = 2 * math.cos(math.radians(15))
TOL = (0.08, 0.20, 0.30, math.radians(15))


def _occupancy(noise, beta, n=4000, local=False, seed=0):
    s, box = mc.hexagon_state(LAM, noise=noise, seed=seed)
    img = np.zeros((len(s), 2), np.int64)
    n_pairs = mc.total_count(s, *box, LAM)
    ref = (s[6] * box[0]).copy()
    states = []
    for k in range(n):
        n_pairs, _, _, _ = mc.flip_moves(s, img, box, LAM, beta, 7, n_pairs, 1 + k, *TOL)
        x = s[6] * box[0]
        d = x - ref; d -= box[0] * np.round(d / box[0])
        states.append(int(np.linalg.norm(d) > 0.6))
    return np.mean(states), n_pairs, mc.total_count(s, *box, LAM)


def test_isolated_hexagon_half_occupancy():
    f, n, chk = _occupancy(0.0, 1e-6)
    assert abs(f - 0.5) < 0.05 and n == chk


def test_noisy_hexagon_reversible():
    # boundary noise 0.01: detection must still be symmetric (reverse move found), occupancy ~ 1/2
    f, n, chk = _occupancy(0.01, 1e-6, seed=3)
    assert abs(f - 0.5) < 0.06 and n == chk


def test_energy_bookkeeping_with_local_moves():
    s, box = mc.lattice_state("rows", LAM - 0.002, 600)
    img = np.zeros((len(s), 2), np.int64)
    n = mc.total_count(s, *box, LAM - 0.002)
    for k in range(300):
        n, _, _ = mc.sweep_img(s, img, box, LAM - 0.002, 1 / 0.06, 0.735, 0.03, 0.001, 0.01, n, 10 + k)
        n, _, _, _ = mc.flip_moves(s, img, box, LAM - 0.002, 1 / 0.06, 30, n, 5000 + k, *TOL)
    assert n == mc.total_count(s, *box, LAM - 0.002)


def test_unwrapped_positions_consistent():
    s, box = mc.lattice_state("fluid", LAM, 200)
    img = np.zeros((len(s), 2), np.int64)
    n = mc.total_count(s, *box, LAM)
    M = mc.box_matrix(box); X0 = (s + img) @ M.T
    for k in range(200):
        n, _, _ = mc.sweep_img(s, img, box, LAM, 1 / 0.5, 0.0, 0.5, 0.0, 0.0, n, 77 + k)  # no box moves
    X1 = (s + img) @ M.T
    # wrapped positions must equal unwrapped modulo the box
    W = mc.cart(s, box)
    d = X1 - W; f = d @ np.linalg.inv(M).T
    assert np.allclose(f, np.round(f), atol=1e-8) and np.abs(X1 - X0).max() > 1.0


def test_hexlat_random_tiling():
    from hcss_analysis import analyze
    s, box = mc.lattice_state("hexlat", LAM - 0.002, 900, seed=1)
    assert mc.total_count(s, *box, LAM - 0.002) >= 0
    r = analyze(mc.cart(s, box), mc.box_matrix(box), lam=LAM - 0.002, sk_nmax=0)
    assert abs(r["composition"]["A"] - 1 / 3) < 1e-9 and abs(r["composition"]["B"] - 2 / 3) < 1e-9
    img = np.zeros((len(s), 2), np.int64)
    n = mc.total_count(s, *box, LAM - 0.002)
    n, seen, prop, acc = mc.flip_moves(s, img, box, LAM - 0.002, 1 / 0.06, 300, n, 9, *TOL)
    assert seen > 50 and acc > 0.5 * seen and n == mc.total_count(s, *box, LAM - 0.002)
