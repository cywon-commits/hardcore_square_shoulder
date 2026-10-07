"""
op_stiffness.py — (pilot 9) entropic phason stiffness of the T -> 0 random tiling with a convergence analysis.
Uniform tiling ensemble on a second-order approximant, 30-degree hexagon flips (+ 12-gon refills).  For every small
reciprocal vector q the time series |w_q|^2 of the residual perp field is recorded.  The run starts FLAT (w_q = 0), so
<|w_q|^2> grows towards equilibrium with a relaxation time tau_q ~ 1/q^2; a mode counts as equilibrated if the mean over
the last quarter of the run differs from the mean over the third quarter by less than its error (and the integrated
autocorrelation time is < 1/20 of the run).  K is fitted only from equilibrated modes:
    <|w_q|^2> = 8 / (A K_area q^2),   K_particle = K_area * A / N,   K_particle = K_alpha + K_beta (per particle)
Spinodal window of the 12-fold state:  dP_sp = T K_particle / (2 * 0.634);  must satisfy dP_FL <= dP_sp.
eta (exact, from the costly-pair count) is recorded as a check: flips and refills conserve it.
Usage  python op_stiffness.py --approx approx2_2x3.npz --sweeps 40000 --every 20 --out stiff.json
"""
import argparse, json, math, time, numpy as np
import hcss_mc as mc, op, tiling_mc as TM
DVDETA = 0.634
class Pre(TM.Tiling):
    def __init__(self, s, box, seed=0):
        self.s = s.copy(); self.box = box.copy(); self.img = np.zeros((len(s), 2), np.int64); self.lam = TM.LS
        self.n_pairs = mc.total_count(self.s, *self.box, self.lam); self.rng = np.random.default_rng(seed); self.find_all_dodecagons()
def tau_int(x):
    """integrated autocorrelation time in units of measurements (Sokal window)."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    if n < 8 or np.var(x) == 0: return float("nan")
    f = np.fft.rfft(x, 2 * n); c = np.fft.irfft(f * np.conj(f))[:n]; c = c / c[0]; t = 0.5
    for k in range(1, n // 2):
        t += c[k]
        if k >= 6 * t: break
    return t
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approx", default="approx2_2x3.npz"); ap.add_argument("--sweeps", type=int, default=40000)
    ap.add_argument("--every", type=int, default=20); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--nshell", type=int, default=3)
    ap.add_argument("--dPFL", default="0.06:0.067,0.08:0.088"); ap.add_argument("--out", default="stiff.json"); a = ap.parse_args()
    z = np.load(a.approx); T = Pre(z["s"], z["box"], a.seed); N = len(T.s)
    M = mc.box_matrix(T.box); A = abs(np.linalg.det(M)); t0 = time.time()
    series = {}; etas = []; times = []
    for sw in range(1, a.sweeps + 1):
        T.flip_sweep(N)
        if T.dod and sw % 5 == 0: T.refill_move()
        if sw % a.every == 0:
            P = T.cart(); o, (L, lf) = op.order_parameters(P, M, lam=op.LS - 1e-6, tol_in=0.02, tol_out=0.02)
            etas.append(o["eta"]); times.append(sw)
            for q, wq, mm in op.phason_spectrum(P, M, lf, nshell=a.nshell):
                series.setdefault((round(q, 4), mm), []).append(wq)
    rows = []
    for (q, mm) in sorted(series):
        if (-mm[0], -mm[1]) < mm and ((round(q, 4), (-mm[0], -mm[1])) in series):
            pass
        z = np.array(series[(q, mm)]); n = len(z); h = z[n // 2:]
        x = np.abs(z) ** 2; xs = x[n // 2:]
        static = abs(h.mean()) ** 2                      # frozen part of the mode (not mixed by the moves)
        var = float(np.mean(np.abs(h - h.mean()) ** 2))  # fluctuation part
        tau = tau_int(h.real); neff = (n // 2) / max(2 * tau, 1) if tau == tau else 0
        q3, q4 = x[n // 2: 3 * n // 4], x[3 * n // 4:]
        err = np.std(xs) / math.sqrt(max(neff, 1))
        eq = bool(abs(q4.mean() - q3.mean()) < 2 * err * math.sqrt(2) and tau == tau and tau < n / 40 and static < 0.5 * var)
        Ka = 8.0 / (A * q * q * var)
        growth = [float(x[i * n // 8:(i + 1) * n // 8].mean()) for i in range(8)]
        rows.append(dict(q=q, m=list(mm), var=var, static=float(static), static_fraction=float(static / (static + var)), tau_meas=float(tau),
                         equilibrated=eq, K_area=float(Ka), K_particle=float(Ka * A / N), growth=growth))
    good = [r for r in rows if r["equilibrated"]]
    Kp = float(np.median([r["K_particle"] for r in good])) if good else float("nan")
    win = {}
    for pair in a.dPFL.split(","):
        Tt, w = (float(x) for x in pair.split(":")); dps = Tt * Kp / (2 * DVDETA)
        win[Tt] = dict(dP_spinodal=dps, dP_FL=w, inequality_holds=bool(w <= dps) if dps == dps else None)
    out = dict(approx=a.approx, N=N, sweeps=a.sweeps, eta_min=float(min(etas)), eta_max=float(max(etas)),
               n_modes=len(rows), n_equilibrated=len(good), K_particle_equilibrated=Kp, window=win, modes=rows, seconds=time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.approx} N={N} sweeps={a.sweeps}: eta in [{out['eta_min']:+.6f}, {out['eta_max']:+.6f}]  equilibrated modes {len(good)}/{len(rows)}  K_particle {Kp:.3f}")
    for r in sorted(rows, key=lambda r: r['q'])[:16]:
        print("   q %.4f m %-8s  K_particle %6.3f  static fraction %.2f  tau %6.1f meas  eq %-5s" % (r["q"], str(tuple(r["m"])), r["K_particle"], r["static_fraction"], r["tau_meas"], r["equilibrated"]))
    print("   window:", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in win.items()})
if __name__ == "__main__":
    main()
