"""
summary7.py — final table of pilot 7 (corrected NPT start, mean Einstein sites).
For every (lam, P, T): beta*g of A, B and the tiling (minus ln4421/19), D_tB, D_tA with 2-sigma errors, winner,
NPT drift check, overlap-free fraction; then, per (lam, T), the pressure range where the tiling wins (if any), and a
least-squares estimate of kappa_eff from all points below lam* (rigid theory with sigma_B, sigma_A from lam = 1.93).
Comparison with earlier pilots is printed if fe4/, fe5/, fe6/ are present.
"""
import glob, json, math, os
import numpy as np
import theory_window as tw

SC = math.log(4421) / 19


def main():
    G = tw.load_group(glob.glob("fe7/*.json"))
    rows = []
    print(f"{'lam':>5s} {'P':>6s} {'T':>5s} | {'D_tB (2s)':>16s} {'D_tA (2s)':>16s} | win | {'drift v':>8s} {'olapfree':>8s}")
    for (l, P, T), d in sorted(G.items()):
        if not all(k in d for k in ("A", "B", "dodeca")):
            continue
        t = d["dodeca"]; gt = t["beta_g"] - SC
        dB = gt - d["B"]["beta_g"]; dA = gt - d["A"]["beta_g"]
        eB = 2 * math.hypot(t["beta_g_err"], d["B"]["beta_g_err"]); eA = 2 * math.hypot(t["beta_g_err"], d["A"]["beta_g_err"])
        win = min((0.0, "tiling"), (-dB, "B"), (-dA, "A"))[1]
        drift = max(abs(x["npt"].get("drift_v_last_minus_first_half_of_sampling", 0.0)) for x in (t, d["A"], d["B"]))
        of = min(x["components"]["dA1_overlap_free_fraction"] for x in (t, d["A"], d["B"]))
        rows.append((l, P, T, dB, eB, dA, eA, win))
        print(f"{l:5.2f} {P:6.3f} {T:5.2f} | {dB:+8.3f} ±{eB:.3f}   {dA:+8.3f} ±{eA:.3f} | {win:6s} | {drift:8.4f} {of:8.3f}")
    print("\n== tiling window per (lam, T) [pressures where the tiling is lowest beyond 2 sigma]")
    by = {}
    for r in rows:
        by.setdefault((r[0], r[2]), []).append(r)
    for (l, T), rs in sorted(by.items()):
        wins = [r[1] for r in rs if r[7] == "tiling" and r[3] + r[4] < 0 and r[5] + r[6] < 0]
        ties = [r[1] for r in rs if r[7] == "tiling" and not (r[3] + r[4] < 0 and r[5] + r[6] < 0)]
        print(f"   lam {l:.2f} T {T:.2f}: tiling wins at P = {wins}   (wins within 2 sigma: {ties})   points: {[ (r[1], r[7]) for r in rs]}")
    # kappa_eff fit
    ref = G.get((1.93, 0.735, 0.06))
    if ref and all(k in ref for k in ("A", "B", "dodeca")):
        sB, sA = tw.sigmas_from(1.93, 0.735, 0.06, ref["dodeca"]["beta_g"] - SC - ref["B"]["beta_g"], ref["A"]["beta_g"] - ref["B"]["beta_g"])
        num = den = 0.0
        tw.KAPPA = 0.0
        for (l, P, T, dB, eB, dA, eA, win) in rows:
            s = tw.LS - l
            if s > 0.005:
                pB, _ = tw.predict(l, P, T, sB, sA)
                x = P * s / T                      # D_meas - D_pred = -kappa * P s / T
                num += -(dB - pB) * x / (eB / 2) ** 2; den += x * x / (eB / 2) ** 2
        if den > 0:
            k = num / den
            print(f"\n== kappa_eff (weighted fit of the t-B deviation below lam*, all T) = {k:.3f} +- {1 / math.sqrt(den):.3f}"
                  f"   [sigma_B = {sB:.4f}, sigma_A = {sA:.4f} from lam = 1.93, P = 0.735, T = 0.06]")
    for old in ("fe4", "fe5", "fe6"):
        if os.path.isdir(old):
            Go = tw.load_group(glob.glob(os.path.join(old, "*.json")))
            for key, d in sorted(Go.items()):
                if key in G and "dodeca" in d and "B" in d and "dodeca" in G[key] and "B" in G[key]:
                    o = d["dodeca"]["beta_g"] - SC - d["B"]["beta_g"]
                    n = G[key]["dodeca"]["beta_g"] - SC - G[key]["B"]["beta_g"]
                    print(f"   {old} vs fe7 at {key}: D_tB {o:+.3f} -> {n:+.3f}")


if __name__ == "__main__":
    main()
