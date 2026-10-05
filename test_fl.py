"""
test_fl.py — tests for pilot-3 additions.  python -m pytest -q test_fl.py
"""
import math
import numpy as np
import hcss_mc as mc
from hcss_analysis import tile_profile

LAM = 1.93


def test_harmonic_limit():
    # at large Lam the tethered particles are harmonic: Lam * <sum |dr|^2> = N - 1 (2D)
    s0, box = mc.lattice_state("B", LAM, 300); box = box * 1.03
    s = s0.copy(); n = mc.total_count(s, *box, LAM); vals = []
    for k in range(400):
        n, _, tot = mc.sweep_fl(s, s0, box, LAM, 1 / 0.06, 1e4, 0.004, n, k)
        if k > 100:
            vals.append(tot)
    assert abs(np.mean(vals) * 1e4 / (len(s) - 1) - 1.0) < 0.03


def test_particle0_fixed_and_energy():
    s0, box = mc.lattice_state("hexlat", LAM, 300, seed=2); box = box * 1.03
    s = s0.copy(); n = mc.total_count(s, *box, LAM)
    for k in range(200):
        n, _, _ = mc.sweep_fl(s, s0, box, LAM, 1 / 0.06, 10.0, 0.03, n, k)
    assert np.allclose(s[0], s0[0]) and n == mc.total_count(s, *box, LAM)


def test_dA1_baseline():
    s0, box = mc.lattice_state("B", LAM, 300); box = box * 1.03
    dA1, free, mp = mc.einstein_dA1(s0, box, LAM, 1 / 0.06, 1e6, 200)
    assert free == 1.0 and abs(dA1 - mc.total_count(s0, *box, LAM) / 0.06) < 1e-6


def test_slab_states():
    for left, xa in (("B", 0.0), ("A", 1.0)):
        s, box, lab = mc.slab_state(left, "hexlat", LAM, 12, 24 if left == "B" else 14, 8, 1.035, seed=1)
        assert mc.total_count(s, *box, LAM) >= 0
        prof = tile_profile(mc.cart(s, box), mc.box_matrix(box), LAM, tol_out=0.25, nbins=20)
        xA = prof[:, 0] / np.maximum(prof.sum(1), 1)
        inner_crystal = xA[2:7]; inner_tiling = xA[12:16]
        assert np.allclose(inner_crystal, xa, atol=0.05)
        assert np.all((inner_tiling > 0.2) & (inner_tiling < 0.5))
