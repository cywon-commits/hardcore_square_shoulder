"""
summary6.py — pilot 6 tables.
(1) validation of NPT-mean Einstein sites against pilot 3b (ideal sites) at lam = 1.93, P = 0.735, T = 0.06;
(2) jammed-start vs ideal-start NPT: volume deficit relative to the ideal (lam*-stretched) tiling,
    d = v_ideal + 2T/P - <v>, compared with 0.9 (lam* - lam) (pilots 4-5) and kappa_jam (lam* - lam), kappa_jam ~ 1.48;
(3) Frenkel-Ladd: beta*g of A, B and the tiling variants (s_conf = ln4421/19 subtracted), D_tB, D_tA and the winner;
    jammed vs ideal basin of the same filling.
"""
import glob, json, math, os
import numpy as np

LS = 2 * math.cos(math.radians(15)); SC = math.log(4421) / 19
V_IDEAL = LS ** 2 * (14 * math.sqrt(3) + 24) / 76
REF3B = {"B": 45.6024, "dodeca_vib": 45.9345}   # pilot 3b, ideal sites, N ~ 931, T = 0.06


def load(f):
    return json.load(open(f))


def main():
    fe = {}
    for f in glob.glob("fe6/*.json"):
        r = load(f); fe[(r["kind"], round(r["lam"], 3), round(r["P"], 3))] = r
    print("== (1) validation, lam = 1.93, P = 0.735 (mean sites vs pilot-3b ideal sites)")
    for k, ref in (("B", REF3B["B"]), ("dodeca", REF3B["dodeca_vib"])):
        r = fe.get((k, 1.93, 0.735))
        if r:
            print(f"   {k:7s} mean-site beta_g {r['beta_g']:.4f} +- {2 * r['beta_g_err']:.4f} (2s)   ideal-site {ref:.4f}   diff {r['beta_g'] - ref:+.4f}")
    print("== (2) jammed-start vs ideal-start NPT")
    for d in sorted(glob.glob("npt6/*")):
        info = load(os.path.join(d, "run_info.json"))
        ts = np.genfromtxt(os.path.join(d, "timeseries.csv"), delimiter=",", names=True)
        lam, P, T = info["lam"], info["P"], info["T"]
        v = 1.0 / ts["density"]; n = len(v)
        q1, q4 = v[n // 8: n // 4].mean(), v[3 * n // 4:].mean()
        s = LS - lam
        dfc = V_IDEAL + 2 * T / P - q4
        print(f"   {os.path.basename(d):22s} lam {lam}  <v> early {q1:.4f} late {q4:.4f}  deficit {dfc:+.4f}  "
              f"deficit/(lam*-lam) {dfc / s:5.2f}  (0.9 = pilots 4-5, ~1.48 = jammed)   e/N late {ts['e_per_N'][3 * n // 4:].mean():.3f}")
    print("== (3) Frenkel-Ladd (mean sites), T = 0.06")
    pts = sorted({(k[1], k[2]) for k in fe})
    for (lam, P) in pts:
        A = fe.get(("A", lam, P)); B = fe.get(("B", lam, P))
        if not (A and B):
            continue
        line = f"   lam {lam} P {P}:  A {A['beta_g']:.3f}  B {B['beta_g']:.3f}"
        best = []
        for kind in ("dodeca", "dodeca1", "dodeca1_jam"):
            t = fe.get((kind, lam, P))
            if t:
                gt = t["beta_g"] - SC
                e = 2 * math.hypot(t["beta_g_err"], B["beta_g_err"])
                line += (f"\n      {kind:12s} tiling {gt:.3f}   D_tB {gt - B['beta_g']:+.3f} +- {e:.3f}   D_tA {gt - A['beta_g']:+.3f}"
                         f"   (overlap-free {t['components']['dA1_overlap_free_fraction']:.2f}, Lmax {t['lam_max_used']:.0e}, sites {t['sites'].get('used')})")
                best.append((gt, kind))
        g = min([(A["beta_g"], "A"), (B["beta_g"], "B")] + best)
        line += f"\n      lowest: {g[1]}"
        j = fe.get(("dodeca1_jam", lam, P)); i_ = fe.get(("dodeca1", lam, P))
        if j and i_:
            line += f"   (jammed - ideal basin, same filling: {j['beta_g'] - i_['beta_g']:+.3f})"
        print(line)


if __name__ == "__main__":
    main()
