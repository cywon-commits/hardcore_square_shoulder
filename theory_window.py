"""
theory_window.py — hard-contact prediction of the tiling stability window, and comparison with Frenkel-Ladd.

Theory (pilot 3/3b): below melting, beta*g_X = beta*h0_X - c_X - s_conf,X + (terms common to all structures), with
h0 = e0 + P v0 the T = 0 contact enthalpy and c_X a T- and P-independent geometric constant.  Hence, relative to B,
    D_tB(lam,P,T) = [ (e_t - 1)  + P (v_t - v_B) ] / T - sigma_B
    D_tA(lam,P,T) = [  e_t       + P (v_t - v_A) ] / T - sigma_A
with  v_A = (sqrt3/2) lam^2,  v_B = sqrt(lam^2 - 1/4),  v_t = max(lam, lam*)^2 (14 sqrt3 + 24)/76  (3.12.12 tiling),
e_t = 12/19, and the two entropy constants sigma_B, sigma_A fixed from ONE reference point (lam0, P0, T0).
The tiling is stable where both D < 0:   P_min < P < P_max,
    P_max = (1 - e_t + T sigma_B)/(v_t - v_B),   P_min = (e_t - T sigma_A)/(v_A - v_t).

Usage
  python theory_window.py --ref fe4/ref_T0.06            # uses <ref>_{A,B,dodeca}.json  (lam0 = 1.93, P0 = 0.735)
                          --compare fe4                  # all fe4/*.json grouped by (lam, P, T)
Writes theory_window.json and theory_window.png (window in the (lam, P) plane at T = 0.06, 0.08 with test points).
"""
import argparse, glob, json, math, os
import numpy as np

LS = 2 * math.cos(math.radians(15))
E_T = 12 / 19


def vA(l): return math.sqrt(3) / 2 * l * l
def vB(l): return math.sqrt(l * l - 0.25)
KAPPA = 0.0      # effective-volume slope for lam < lam* (pilot 4: ~1.7 at T = 0.06); set with --kappa


def vt(l):
    L = max(l, LS); v = L * L * (14 * math.sqrt(3) + 24) / 76
    return v - KAPPA * max(0.0, LS - l)


def sigmas_from(lam0, P0, T0, dg_tB, dg_AB):
    bh = lambda v, e: (e + P0 * v) / T0
    sB = -(dg_tB - (bh(vt(lam0), E_T) - bh(vB(lam0), 1.0)))
    cA = -(dg_AB - (bh(vA(lam0), 0.0) - bh(vB(lam0), 1.0)))
    return sB, sB - cA


def predict(l, P, T, sB, sA):
    return ((E_T - 1 + P * (vt(l) - vB(l))) / T - sB, (E_T + P * (vt(l) - vA(l))) / T - sA)


def window(l, T, sB, sA):
    return (E_T - T * sA) / (vA(l) - vt(l)), (1 - E_T + T * sB) / (vt(l) - vB(l))


def load_group(files):
    G = {}
    for f in files:
        r = json.load(open(f))
        key = (round(r["lam"], 4), round(r["P"], 4), round(r["T"], 4))
        G.setdefault(key, {})[r["kind"]] = r
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="fe4"); ap.add_argument("--lam0", type=float, default=1.93)
    ap.add_argument("--P0", type=float, default=0.735); ap.add_argument("--T0", type=float, default=0.06)
    ap.add_argument("--sigmaB", type=float, default=None); ap.add_argument("--sigmaA", type=float, default=None)
    ap.add_argument("--out", default="theory_window")
    ap.add_argument("--kappa", type=float, default=0.0, help="effective-volume slope for lam < lam*")
    a = ap.parse_args()
    global KAPPA
    KAPPA = a.kappa
    G = load_group(glob.glob(os.path.join(a.ref_dir, "*.json")))
    if a.sigmaB is None:
        ref = G[(round(a.lam0, 4), round(a.P0, 4), round(a.T0, 4))]
        tB = ref["dodeca"]["beta_g_tiling"] - ref["B"]["beta_g"]
        AB = ref["A"]["beta_g"] - ref["B"]["beta_g"]
        sB, sA = sigmas_from(a.lam0, a.P0, a.T0, tB, AB)
    else:
        sB, sA = a.sigmaB, a.sigmaA
    print(f"sigma_B = {sB:.4f}, sigma_A = {sA:.4f}  (reference lam={a.lam0}, P={a.P0}, T={a.T0}), kappa = {KAPPA}")
    rows = []
    print(f"{'lam':>6s} {'P':>7s} {'T':>5s} | {'pred t-B':>9s} {'meas t-B':>15s} | {'pred t-A':>9s} {'meas t-A':>15s} | stable(pred/meas)")
    for (l, P, T), d in sorted(G.items()):
        if not all(k in d for k in ("A", "B", "dodeca")):
            continue
        ptB, ptA = predict(l, P, T, sB, sA)
        mtB = d["dodeca"]["beta_g_tiling"] - d["B"]["beta_g"]; mtA = d["dodeca"]["beta_g_tiling"] - d["A"]["beta_g"]
        etB = math.hypot(d["dodeca"]["beta_g_err"], d["B"]["beta_g_err"]); etA = math.hypot(d["dodeca"]["beta_g_err"], d["A"]["beta_g_err"])
        sp = "tiling" if max(ptB, ptA) < 0 else ("B" if ptB > 0 and ptB - ptA >= 0 else "A")
        winner = min((("tiling", 0.0), ("B", -mtB), ("A", -mtA)), key=lambda x: x[1])[0]
        rows.append(dict(lam=l, P=P, T=T, pred_tB=ptB, meas_tB=mtB, err_tB=2 * etB, pred_tA=ptA, meas_tA=mtA, err_tA=2 * etA,
                         pred_stable=sp, meas_stable=winner))
        print(f"{l:6.3f} {P:7.4f} {T:5.2f} | {ptB:+9.3f} {mtB:+8.3f}±{2*etB:.3f} | {ptA:+9.3f} {mtA:+8.3f}±{2*etA:.3f} | {sp}/{winner}")
    json.dump(dict(sigma_B=sB, sigma_A=sA, rows=rows), open(a.out + ".json", "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        lams = np.linspace(1.895, 1.97, 300)
        for k, T in enumerate((0.06, 0.08)):
            lo = np.array([window(l, T, sB, sA)[0] for l in lams]); hi = np.array([window(l, T, sB, sA)[1] for l in lams])
            m = hi > lo
            ax[k].fill_between(lams[m], lo[m], hi[m], color="tab:orange", alpha=0.35, label="tiling (theory)")
            ax[k].plot(lams, lo, "k--", lw=.8); ax[k].plot(lams, hi, "k-", lw=.8)
            ax[k].axvline(LS, color="gray", lw=.6, ls=":")
            for r in rows:
                if abs(r["T"] - T) < 1e-6:
                    mk = {"tiling": "o", "B": "s", "A": "^"}[r["meas_stable"]]
                    col = "tab:green" if r["meas_stable"] == r["pred_stable"] else "tab:red"
                    ax[k].plot(r["lam"], r["P"], mk, color=col, ms=8)
            ax[k].set_title(f"T* = {T}  (o tiling, s B, ^ A measured; green = agrees)"); ax[k].set_xlabel("lambda")
        ax[0].set_ylabel("P"); ax[0].set_ylim(0.6, 0.8); fig.tight_layout(); fig.savefig(a.out + ".png", dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
