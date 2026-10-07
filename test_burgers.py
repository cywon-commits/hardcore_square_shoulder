"""test_burgers.py — integer Burgers vectors: zero for perfect tilings, unit norms, total zero on the torus."""
import math, numpy as np, hcss_mc as mc, burgers as BG, op
def test_basis_and_units():
    for k in range(12):
        assert abs(BG.par(BG.BASIS[k]) - BG.Z ** k) < 1e-12
    u = BG.BASIS[0] - BG.BASIS[1]
    assert abs(BG.norm(u) - 1) < 1e-9 and abs(abs(BG.par(u)) - 2 * math.sin(math.pi / 12)) < 1e-12
def test_perfect_and_thermal():
    for kind in ("hexlat", "dodeca", "A", "B"):
        s, box = mc.lattice_state(kind, op.LS, 400, scale=1 + 1e-7, seed=1)
        st = BG.defect_stats(mc.cart(s, box), mc.box_matrix(box), op.LS - 1e-6, 0.02, 0.02)
        assert st["n_defects"] == 0
    T = 0.13; s, box = mc.lattice_state("hexlat", 1.93, 500, scale=math.sqrt(1 + 2 * T / 0.735 / 2.3), seed=3)
    img = np.zeros((len(s), 2), np.int64); n = mc.total_count(s, *box, 1.93)
    for k in range(1500):
        n, _, _ = mc.sweep_img(s, img, box, 1.93, 1 / T, 0.735, 0.04, 0.002, 0.02, n, 9 + k)
    st = BG.defect_stats(mc.cart(s, box), mc.box_matrix(box), 1.93, 0.06, 0.10 + 2 * T / (0.735 * 1.93))
    assert st["total_b"] == [0, 0, 0, 0]
    assert all(abs(x - round(x)) < 1e-6 for x in st["norms"])
