"""test_op.py — the exact identity eta = det E and the strain types of the ideal structures.  python -m pytest -q test_op.py"""
import numpy as np, hcss_mc as mc, op
def _o(kind):
    s, box = mc.lattice_state(kind, op.LS, 600, scale=1 + 1e-7, seed=1)
    o, _ = op.order_parameters(mc.cart(s, box), mc.box_matrix(box), lam=op.LS - 1e-6, tol_in=0.02, tol_out=0.02); return o
def test_identity_and_types():
    for kind, eta, a, b in (("A", -1, 0, 1), ("B", 1, 2, 3 ** .5), ("dodeca", -(2 - 3 ** .5) ** 4, 0, (2 - 3 ** .5) ** 2)):
        o = _o(kind)
        assert abs(o["eta"] - eta) < 1e-6 and abs(o["detE"] - eta) < 1e-6
        assert abs(o["abs_alpha"] - a) < 1e-6 and abs(o["abs_beta"] - b) < 1e-6
def test_second_order_approximant():
    z = np.load("approx2_sym.npz")
    o, _ = op.order_parameters(mc.cart(z["s"], z["box"]), mc.box_matrix(z["box"]), lam=op.LS - 1e-6, tol_in=0.02, tol_out=0.02)
    assert abs(o["detE"] - o["eta"]) < 1e-9 and abs(o["abs_beta"] - (2 - 3 ** .5) ** 4) < 1e-6 and o["abs_alpha"] < 1e-9


def test_thermal_identity_and_flip_conservation():
    import math
    import tiling_mc as TM
    lam, T, Pr = 1.93, 0.06, 0.735
    for kind in ("B", "hexlat", "A"):
        s, box = mc.lattice_state(kind, lam, 400, scale=math.sqrt(1 + 2 * T / Pr / 2.3), seed=2)
        img = np.zeros((len(s), 2), np.int64); n = mc.total_count(s, *box, lam)
        for k in range(600):
            n, _, _ = mc.sweep_img(s, img, box, lam, 1 / T, Pr, 0.03, 0.002, 0.02, n, 11 + k)
        o, _ = op.order_parameters(mc.cart(s, box), mc.box_matrix(box), lam=lam, tol_in=0.06, tol_out=0.10 + 2 * T / (Pr * lam))
        assert o["n_defect_edges"] == 0 and abs(o["eta"] - o["detE"]) < 1e-6
    z = np.load("approx2_sym.npz")
    class Pre(TM.Tiling):
        def __init__(self, s, box):
            self.s = s.copy(); self.box = box.copy(); self.img = np.zeros((len(s), 2), np.int64); self.lam = TM.LS
            self.n_pairs = mc.total_count(self.s, *self.box, self.lam); self.rng = np.random.default_rng(0); self.dod = {}
    Tl = Pre(z["s"], z["box"]); n0 = Tl.n_pairs
    for _ in range(30):
        Tl.flip_sweep(len(Tl.s))
    assert Tl.n_pairs == n0 == mc.total_count(Tl.s, *Tl.box, Tl.lam)
