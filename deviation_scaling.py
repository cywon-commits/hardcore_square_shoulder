"""
deviation_scaling.py — pilot 5 analysis: deviation of the measured tiling free energy from the rigid (kappa = 0) theory.

Pilot 4 found, at T = 0.06,  D_meas - D_pred ~ -20 (lam* - lam)  for lam < lam*, independent of P, and interpreted it as
an effective tiling volume  v_t -> v_t - kappa (lam* - lam)  with kappa = 20 T / P ~ 1.7  (static compression of the
stretched network + extra free volume from the slack).  Both parts scale like P/T, so the slope
    S(T) = (D_meas - D_pred) / (lam* - lam)
must scale like 1/T at fixed P (prediction: ~ -15 at T = 0.08), i.e. kappa_eff = -S T / P must be T-independent.

Usage  python deviation_scaling.py fe4 fe5      (directories with Frenkel-Ladd JSONs for A, B, dodeca)
"""
import sys, glob, json, math, os
import theory_window as tw

LS = tw.LS


def main():
    files = []
    for d in sys.argv[1:]:
        files += glob.glob(os.path.join(d, "*.json"))
    G = tw.load_group(files)
    ref = G.get((1.93, 0.735, 0.06))
    tB = ref["dodeca"]["beta_g_tiling"] - ref["B"]["beta_g"]; AB = ref["A"]["beta_g"] - ref["B"]["beta_g"]
    sB, sA = tw.sigmas_from(1.93, 0.735, 0.06, tB, AB)
    print(f"sigma_B = {sB:.4f}  sigma_A = {sA:.4f}  (T = 0.06 reference, kappa = 0 theory)")
    print(f"{'lam':>6s} {'P':>6s} {'T':>5s} {'lam*-lam':>8s} | {'dev t-B':>8s} {'slope S':>8s} {'kappa_eff':>9s} | {'tiling NPT comp (A,B,X)':>26s}")
    for (l, P, T), d in sorted(G.items()):
        if not all(k in d for k in ("A", "B", "dodeca")):
            continue
        ptB, ptA = tw.predict(l, P, T, sB, sA)
        mtB = d["dodeca"]["beta_g_tiling"] - d["B"]["beta_g"]
        dl = LS - l
        comp = d["dodeca"].get("npt_structure", {}).get("composition", {})
        cs = f"{comp.get('A', float('nan')):.3f},{comp.get('B', float('nan')):.3f},{comp.get('X', float('nan')):.3f}"
        if dl > 0.005:
            S = (mtB - ptB) / dl
            print(f"{l:6.3f} {P:6.3f} {T:5.2f} {dl:8.4f} | {mtB - ptB:+8.3f} {S:+8.2f} {-S * T / P:9.3f} | {cs:>26s}")
        else:
            print(f"{l:6.3f} {P:6.3f} {T:5.2f} {dl:8.4f} | {mtB - ptB:+8.3f} {'':>8s} {'':>9s} | {cs:>26s}")


if __name__ == "__main__":
    main()
