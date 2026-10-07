"""
fl_free_energy.py — Gibbs free energy per particle of a fixed structure by Frenkel-Ladd thermodynamic
integration (Einstein-molecule variant, Vega & Noya, J. Chem. Phys. 127, 154113 (2007)), 2D.

Steps
  1. NPT run of the structure (no flips) -> <box> (a, b, c averaged), <e>, <v>, h = <e> + P <v>
  2. Lattice sites = ideal fractional positions in the averaged box; particle 0 fixed.
  3. beta*A = -ln V - (N-1) ln(pi/Lam_max) + beta*dA1 - Int_0^{Lam_max} <sum_{i>=1} |r_i - r0_i|^2>_Lam dLam
     (thermal wavelength = 1; the 1/N! cancels against the N! site permutations).
     The integral is done in x = ln Lam with Gauss-Legendre nodes on [ln Lam_min, ln Lam_max]
     plus Lam_min * <...>_{Lam_min} for the piece below Lam_min.
  4. beta*g = beta*A/N + beta*P*<v>.  For --kind dodeca the bound is ln(4421)/19 (3.12.12 tiling with independently
     filled 12-gons).  For --kind hexlat the configurational entropy of the hexagon-flip
     ensemble, ln(2)/3 per particle (exact lower bound for the random tiling), is reported separately:
     beta*g_tiling = beta*g_hexlat - ln(2)/3.

Usage
  python fl_free_energy.py --kind B --T 0.06 --P 0.735 --N 1000 --out fe/B_T0.06.json
"""
import argparse, json, math, os, time
import numpy as np
import hcss_mc as mc


def block_err(x, nb=10):
    x = np.asarray(x, float)
    m = len(x) // nb
    if m < 1:
        return float("nan")
    b = x[: m * nb].reshape(nb, m).mean(1)
    return float(b.std(ddof=1) / math.sqrt(nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["A", "B", "hexlat", "rows", "dodeca", "dodeca1", "dodeca1_jam"])
    ap.add_argument("--lam", type=float, default=1.93)
    ap.add_argument("--T", type=float, default=0.06)
    ap.add_argument("--P", type=float, default=0.735)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--npt-sweeps", type=int, default=50000)
    ap.add_argument("--n-nodes", type=int, default=14)
    ap.add_argument("--lam-min", type=float, default=0.05)
    ap.add_argument("--lam-max", type=float, default=3.0e4)
    ap.add_argument("--equil", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--dA1-samples", type=int, default=2000)
    ap.add_argument("--sites", default="ideal", choices=["ideal", "mean"],
                    help="Einstein sites: ideal lattice positions in the averaged box (pilots 3-5) or the NPT-averaged "
                         "positions (drift removed); 'mean' falls back to the sampled configuration closest to the mean "
                         "if the mean positions overlap")
    ap.add_argument("--expand-c", type=float, default=None,
                    help="NPT starts from v0 + c * 2T/P.  Default (pilot 7): c = 1 for crystals (as in pilots 3-6) and c = 0.8 "
                         "for the tilings, so that the tiling approaches its equilibrium volume (~ v0 + 0.9 * 2T/P below lam*) "
                         "from below, i.e. in the fast expanding direction (compression from above is very slow, pilot 6). "
                         "Below lam* use kind dodeca1_jam (v0 = jammed volume).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    beta = 1.0 / a.T; t0 = time.time()
    rs = np.random.default_rng(a.seed)

    # ---- 1. NPT
    # start from the thermally expanded lattice  v = v0 + 2T/P  (hard-contact NPT law, pilot-2 check)
    s_ideal, box1 = mc.lattice_state(a.kind, a.lam, a.N, scale=1.0, seed=a.seed)
    v0 = box1[0] * box1[2] / len(s_ideal)
    c_exp = a.expand_c if a.expand_c is not None else (0.8 if a.kind.startswith("dodeca") or a.kind == "hexlat" else 1.0)
    f = math.sqrt(1.0 + c_exp * (2 * a.T / a.P) / v0)
    s_ideal, box = mc.lattice_state(a.kind, a.lam, a.N, scale=f, seed=a.seed)
    s = s_ideal.copy(); img = np.zeros((len(s), 2), np.int64); N = len(s)
    n = mc.total_count(s, *box, a.lam)
    dmax, dbox = 0.03, 0.002
    acc_p = acc_b = win = 0; boxes = []; es = []; vs = []
    u_acc = np.zeros_like(s); u_cnt = 0; u_snaps = []
    tune = a.npt_sweeps // 5
    for k in range(a.npt_sweeps):
        n, ap_, ab_ = mc.sweep_img(s, img, box, a.lam, beta, a.P, dmax, dbox, 0.0, n, int(rs.integers(1, 2**31 - 1)))
        acc_p += ap_; acc_b += ab_; win += 1
        if k < tune and win == 100:
            dmax *= 1.1 if acc_p / (win * N) > 0.35 else 0.9
            dbox *= 1.1 if acc_b / win > 0.3 else 0.9
            acc_p = acc_b = win = 0
        if k >= a.npt_sweeps // 2:
            boxes.append(box.copy()); es.append(n / N); vs.append(box[0] * box[2] / N)
            if a.sites == "mean" and k % 10 == 0:
                u = s + img; u = u - u.mean(0)
                u_acc += u; u_cnt += 1
                if k % 200 == 0:
                    u_snaps.append(u.copy())
    boxm = np.mean(boxes, 0); e_m = float(np.mean(es)); v_m = float(np.mean(vs))
    hq = len(vs) // 2
    npt_drift = float(np.mean(vs[hq:]) - np.mean(vs[:hq]))
    from hcss_analysis import analyze, tol_out_auto
    _r = analyze(mc.cart(s, box), mc.box_matrix(box), lam=a.lam, tol_out=tol_out_auto(a.T, a.P, a.lam), sk_nmax=0)
    npt_structure = dict(composition=_r["composition"], core_coordination=_r["mean_core_coordination"])
    h = e_m + a.P * v_m
    npt = dict(drift_v_last_minus_first_half_of_sampling=npt_drift, start_scale=f, box=boxm.tolist(), e=e_m, e_err=block_err(es), v=v_m, v_err=block_err(vs), h=h,
               h_err=math.hypot(block_err(es), a.P * block_err(vs)))

    # ---- 2.-3. Frenkel-Ladd in the averaged box, sites = ideal fractional positions
    s0 = s_ideal.copy(); V = boxm[0] * boxm[2]
    sites_info = dict(mode=a.sites)
    if a.sites == "mean":
        um = u_acc / u_cnt
        cand = (um + 0.5) % 1.0
        nc = mc.total_count(cand, *boxm, a.lam)
        if nc >= 0:
            s0 = cand; sites_info.update(used="mean", site_pairs_per_N=nc / N)
        else:
            d = [((((x - um) + 0.5) % 1.0 - 0.5) ** 2).sum() for x in u_snaps]
            med = (u_snaps[int(np.argmin(d))] + 0.5) % 1.0
            s0 = med; sites_info.update(used="medoid", site_pairs_per_N=mc.total_count(med, *boxm, a.lam) / N)
        # rms distance between the chosen sites and the ideal lattice sites (in the averaged box)
        dd = ((s0 - s_ideal + 0.5) % 1.0 - 0.5)
        dd = dd - dd.mean(0)
        dc = np.stack([boxm[0] * dd[:, 0] + boxm[1] * dd[:, 1], boxm[2] * dd[:, 1]], 1)
        sites_info["rms_offset_from_ideal"] = float(np.sqrt((dc ** 2).sum(1).mean()))
    # adapt Lam_max so that the Einstein molecule at Lam_max is (mostly) overlap-free
    lam_max = a.lam_max
    for _ in range(8):
        dA1, free_frac, mean_pairs = mc.einstein_dA1(s0, boxm, a.lam, beta, lam_max, a.dA1_samples, a.seed)
        if free_frac >= 0.8:
            break
        lam_max *= 3.0
    a.lam_max = lam_max
    xg, wg = np.polynomial.legendre.leggauss(a.n_nodes)
    x0, x1 = math.log(a.lam_min), math.log(a.lam_max)
    xs = 0.5 * (x1 - x0) * xg + 0.5 * (x1 + x0); ws = 0.5 * (x1 - x0) * wg
    lams = [float(v) for v in np.exp(xs)] + [float(a.lam_min)]
    msd = {}; msd_err = {}
    for idx in sorted(range(len(lams)), key=lambda j: -lams[j]):
        Lam = lams[idx]
        s = s0.copy(); n = mc.total_count(s, *boxm, a.lam)
        dm = min(0.05, 0.6 / math.sqrt(Lam)); acc = 0; cnt = 0; vals = []
        for k in range(a.equil + a.sample):
            n, ac, tot = mc.sweep_fl(s, s0, boxm, a.lam, beta, Lam, dm, n, int(rs.integers(1, 2**31 - 1)))
            if k < a.equil:
                acc += ac; cnt += 1
                if cnt == 100:
                    dm *= 1.1 if acc / (cnt * (N - 1)) > 0.4 else 0.9
                    acc = cnt = 0
            else:
                vals.append(tot)
        msd[idx] = float(np.mean(vals)); msd_err[idx] = block_err(vals)
    K = len(xs)
    I = sum(ws[j] * lams[j] * msd[j] for j in range(K)) + a.lam_min * msd[K]
    I_err = math.sqrt(sum((ws[j] * lams[j] * msd_err[j]) ** 2 for j in range(K)) + (a.lam_min * msd_err[K]) ** 2)
    bA = -math.log(V) - (N - 1) * math.log(math.pi / a.lam_max) + dA1 - I
    bg = bA / N + beta * a.P * v_m
    out = dict(kind=a.kind, seed=a.seed, lam=a.lam, T=a.T, P=a.P, N=N, npt=npt, npt_structure=npt_structure,
               beta_A_per_N=bA / N, beta_g=bg, beta_g_err=math.hypot(I_err / N, beta * a.P * npt["v_err"]),
               components=dict(minus_lnV_over_N=-math.log(V) / N,
                               einstein_term_over_N=-(N - 1) * math.log(math.pi / a.lam_max) / N,
                               beta_dA1_over_N=dA1 / N, dA1_overlap_free_fraction=free_frac,
                               integral_over_N=I / N, integral_err_over_N=I_err / N),
               lam_max_used=a.lam_max, v0=v0, sites=sites_info, msd_nodes={f"{lams[j]:.6g}": msd[j] for j in range(len(lams))},
               seconds=time.time() - t0)
    if a.kind == "hexlat":
        out["s_conf_lower_bound"] = math.log(2) / 3
        out["beta_g_tiling"] = bg - math.log(2) / 3
    if a.kind in ("dodeca", "dodeca1", "dodeca1_jam"):
        # independent fillings of each 12-gon of the 3.12.12 tiling: 4421 per 19 particles (exact lower bound)
        out["s_conf_lower_bound"] = math.log(4421) / 19
        out["beta_g_tiling"] = bg - math.log(4421) / 19
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("kind", "T", "N", "beta_g", "beta_g_err")} | {"h": h}, indent=1))


if __name__ == "__main__":
    main()
