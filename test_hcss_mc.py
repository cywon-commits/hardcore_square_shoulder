"""
test_hcss_mc.py — run with  `python -m pytest -q test_hcss_mc.py`  (or `python test_hcss_mc.py`).

1. total_count agrees with brute-force pair counting (triclinic box).
2. Incremental energy bookkeeping agrees with a full recount after many sweeps.
3. Initial states have the intended tile composition (A, B, rows) and no overlaps.
4. Hard-disk limit (beta*eps -> 0): low-pressure NPT density obeys the second-virial equation
   beta*P = rho + (pi/2) rho^2 within 5 %.
"""
import math
import numpy as np
import hcss_mc as mc
from hcss_analysis import analyze

LAM = 2 * math.cos(math.radians(15)) - 0.002


def brute(s, box, lam):
    pos = mc.cart(s, box); M = mc.box_matrix(box); inv = np.linalg.inv(M)
    n = 0; over = False
    for i in range(len(pos)):
        d = pos[i + 1:] - pos[i]
        f = d @ inv.T; f -= np.round(f); d = f @ M.T
        best = (d ** 2).sum(1)
        for k in (-1, 1):  # neighbouring image along a1
            dd = d + k * M[:, 0]
            best = np.minimum(best, (dd ** 2).sum(1))
        over |= bool((best < 1).any())
        n += int((best < lam * lam).sum())
    return -1 if over else n


def test_total_count_bruteforce():
    s, box = mc.lattice_state("fluid", LAM, 300)
    box = box.copy(); box[1] = 0.3 * box[0]          # add shear
    assert mc.total_count(s, *box, LAM) == brute(s, box, LAM)


def test_incremental_energy():
    s, box = mc.lattice_state("rows", LAM, 400)
    n = mc.total_count(s, *box, LAM)
    for k in range(200):
        n, _, _ = mc.sweep(s, box, LAM, 1 / 0.15, 0.735, 0.08, 0.003, 0.02, n, 1000 + k)
    assert n == mc.total_count(s, *box, LAM)


def test_initial_compositions():
    for kind, exp in (("A", {"A": 1.0}), ("B", {"B": 1.0}), ("rows", {"A": 1 / 3, "B": 2 / 3})):
        s, box = mc.lattice_state(kind, LAM, 600)
        assert mc.total_count(s, *box, LAM) >= 0
        r = analyze(mc.cart(s, box), mc.box_matrix(box), lam=LAM, sk_nmax=0)
        for t, v in exp.items():
            assert abs(r["composition"][t] - v) < 1e-9, (kind, r["composition"])


def test_hard_disk_virial():
    T = 1e6; P = 0.02 * T   # beta*P = 0.02, eps irrelevant
    s, box = mc.lattice_state("fluid", LAM, 200)
    n = mc.total_count(s, *box, LAM)
    rho = []
    for k in range(6000):
        n, _, _ = mc.sweep(s, box, LAM, 1 / T, P, 0.8, 0.05, 0.0, n, 7 + k)
        if k > 1000:
            rho.append(len(s) / (box[0] * box[2]))
    r = np.mean(rho); bp = 0.02
    pred = (-1 + math.sqrt(1 + 4 * (math.pi / 2) * bp)) / (2 * (math.pi / 2))
    assert abs(r - pred) / pred < 0.05, (r, pred)


if __name__ == "__main__":
    for f in (test_total_count_bruteforce, test_incremental_energy, test_initial_compositions, test_hard_disk_virial):
        f(); print("ok", f.__name__)
