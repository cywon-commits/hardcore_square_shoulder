"""
op_stiffness.py — (analysis 3) entropic phason stiffness of the T -> 0 random tiling and the predicted spinodal window.
Samples the uniform tiling ensemble on a second-order approximant with 30-degree hexagon flips (+ 12-gon refills), measures
<|w_q|^2> of the residual perp field w~ = w - E r, and K_sum = K_alpha + K_beta from  <|w_q|^2> = 8 / (A K_area q^2),
K_particle = K_area * A/N.  The 12-fold state is locally stable for -T K_beta/2 < h < T K_alpha/2 with h = (P - P*) |dv/deta|,
|dv/deta| = 0.634 sigma^2 at lam*, so the spinodal window is  dP_sp = T K_sum / (2 * 0.634).  The free-energy window must obey
dP_FL <= dP_sp  (first-order transitions to the far-away crystals pre-empt the spinodal).
Usage  python op_stiffness.py --approx approx2_2x3.npz --sweeps 3000 --every 20 --out stiff_2x3.json
"""
import argparse, json, math, time, numpy as np
import hcss_mc as mc, op, tiling_mc as TM
DVDETA = 0.634
class Pre(TM.Tiling):
    def __init__(self, s, box, seed=0):
        self.s = s.copy(); self.box = box.copy(); self.img = np.zeros((len(s), 2), np.int64); self.lam = TM.LS
        self.n_pairs = mc.total_count(self.s, *self.box, self.lam); self.rng = np.random.default_rng(seed); self.find_all_dodecagons()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approx", default="approx2_2x3.npz"); ap.add_argument("--sweeps", type=int, default=3000)
    ap.add_argument("--every", type=int, default=20); ap.add_argument("--burn", type=int, default=300); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dPFL", default="0.06:0.067", help="T:free-energy window width pairs, comma separated (pilot 4/7)")
    ap.add_argument("--out", default="stiff.json"); a = ap.parse_args()
    z = np.load(a.approx); T = Pre(z["s"], z["box"], a.seed); N = len(T.s)
    M = mc.box_matrix(T.box); A = abs(np.linalg.det(M)); P0 = T.cart().copy()
    spec = {}; nmeas = 0; moved = np.zeros(N, bool); t0 = time.time(); etas = []
    for sw in range(1, a.sweeps + 1):
        T.flip_sweep(N)
        if T.dod and sw % 5 == 0: T.refill_move()
        if sw > a.burn and sw % a.every == 0:
            P = T.cart(); o, (L, lf) = op.order_parameters(P, M, lam=op.LS - 1e-6, tol_in=0.02, tol_out=0.02)
            etas.append(o["eta"])
            for q, wq in op.phason_spectrum(P, M, lf, nshell=4):
                k = round(q, 4); spec.setdefault(k, []).append(wq)
            nmeas += 1
            d = np.linalg.norm(((T.cart() - P0) @ np.linalg.inv(M).T + 0.5) % 1.0 - 0.5, axis=1); moved |= d > 1e-3
    qs = sorted(spec)
    rows = []
    for q in qs[:12]:
        m = np.mean(spec[q]); K_area = 8.0 / (A * q * q * m); rows.append((q, m, K_area, K_area * A / N))
    Kp = np.median([r[3] for r in rows[:6]])
    pred = {}
    for pair in a.dPFL.split(","):
        Tt, w = (float(x) for x in pair.split(":"))
        pred[Tt] = dict(dP_spinodal=Tt * Kp / (2 * DVDETA), dP_FL=w, inequality_holds=bool(w <= Tt * Kp / (2 * DVDETA)))
    out = dict(approx=a.approx, N=N, sweeps=a.sweeps, measurements=nmeas, frac_vertices_ever_moved=float(moved.mean()),
               eta_mean=float(np.mean(etas)), eta_std=float(np.std(etas)), spectrum=[dict(q=r[0], w2=r[1], K_area=r[2], K_particle=r[3]) for r in rows],
               K_sum_particle=float(Kp), window=pred, seconds=time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "spectrum"}, indent=1))
    for r in rows[:8]: print("   q %.4f  <|w_q|^2> %.3e  K_particle %.3f" % (r[0], r[1], r[3]))
if __name__ == "__main__":
    main()
