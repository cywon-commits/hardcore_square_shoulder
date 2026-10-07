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
