"""
pilot_analyze.py — integrated autocorrelation times and drift checks for pilot runs.

Usage
  python pilot_analyze.py runs/rows_T015 [runs/A_T015 ...] [--discard 0.2]

For every run directory: reads timeseries.csv and structure.csv, discards the first fraction
(--discard, default 0.2) as equilibration, and reports for each observable
  mean, std, tau_int (in sweeps, Sokal automatic window c=6), N_eff, and the
  drift between first and second half of the kept data in units of the standard error.
Writes summary.json and timeseries.png into each run directory, and prints a comparison table.
Decision rule for the main scan (see README): tau_int(x_A) < 1e5 sweeps -> optimistic plan,
>= 1e6 -> conservative plan / implement tile-flip moves first.
"""
import sys, json, os, argparse
import numpy as np


def tau_int(x, c=6.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20 or np.var(x) == 0:
        return float("nan")
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    for M in range(1, n):
        tau += acf[M]
        if M >= c * tau:
            break
    return float(tau)


def load_csv(path):
    if not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d if d.size > 1 else None


def summarize(run, discard):
    out = {}
    for fname, cols in (("timeseries.csv", ["density", "e_per_N"]),
                        ("structure.csv", ["x_A", "x_B", "x_C", "x_X", "core_coord",
                                           "psi6_L", "psi12_L", "ring_h12", "ring_h6", "perp_var"])):
        d = load_csv(os.path.join(run, fname))
        if d is None:
            continue
        sw = d["sweep"]
        keep = sw > sw.max() * discard
        dt = np.median(np.diff(sw)) if len(sw) > 1 else 1
        for col in cols:
            if col not in d.dtype.names:
                continue
            x = d[col][keep]
            x = x[np.isfinite(x)]
            if len(x) < 10:
                continue
            t = tau_int(x)
            neff = len(x) / max(2 * t, 1)
            h = len(x) // 2
            se = x.std() / np.sqrt(max(neff, 1))
            drift = (x[h:].mean() - x[:h].mean()) / se if se > 0 else 0.0
            out[col] = dict(mean=float(x.mean()), std=float(x.std()),
                            tau_int_sweeps=float(t * dt), N_eff=float(neff), drift_sigma=float(drift))
    return out


def plot(run):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ts = load_csv(os.path.join(run, "timeseries.csv")); st = load_csv(os.path.join(run, "structure.csv"))
    fig, ax = plt.subplots(4, 1, figsize=(7, 9), sharex=True)
    if ts is not None:
        ax[0].plot(ts["sweep"], ts["density"]); ax[0].set_ylabel("density")
        ax[0].axhline(42 - 24 * 3 ** 0.5, ls="--", lw=0.8, color="k")
        ax[1].plot(ts["sweep"], ts["e_per_N"]); ax[1].set_ylabel("e / N")
        ax[1].axhline(2 / 3, ls="--", lw=0.8, color="k")
    if st is not None:
        for c in ("x_A", "x_B", "x_C", "x_X"):
            ax[2].plot(st["sweep"], st[c], label=c)
        ax[2].axhline(1 / 3, ls="--", lw=0.8, color="k"); ax[2].axhline(2 / 3, ls="--", lw=0.8, color="k")
        ax[2].legend(fontsize=7); ax[2].set_ylabel("tile fraction")
        ax[3].plot(st["sweep"], st["ring_h12"], label="S(k) ring h12")
        ax[3].plot(st["sweep"], st["ring_h6"], label="S(k) ring h6")
        ax[3].plot(st["sweep"], st["psi6_L"], label="psi6 (L bonds)")
        ax[3].legend(fontsize=7); ax[3].set_ylabel("order")
    ax[-1].set_xlabel("MC sweeps")
    fig.suptitle(os.path.basename(os.path.normpath(run)) + "   (dashed: 12-fold predictions)")
    fig.tight_layout(); fig.savefig(os.path.join(run, "timeseries.png"), dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--discard", type=float, default=0.2)
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        s = summarize(run, a.discard)
        info = {}
        p = os.path.join(run, "run_info.json")
        if os.path.exists(p):
            info = json.load(open(p))
        s["_throughput"] = {k: info.get(k) for k in ("trial_moves_per_s", "sweeps_per_s", "N_actual",
                                                     "energy_consistency", "init", "lam", "T", "P")}
        json.dump(s, open(os.path.join(run, "summary.json"), "w"), indent=2)
        plot(run)
        g = lambda k, f: s.get(k, {}).get(f, float("nan"))
        rows.append((run, g("density", "mean"), g("x_A", "mean"), g("x_A", "tau_int_sweeps"),
                     g("density", "tau_int_sweeps"), g("ring_h12", "mean"), g("ring_h6", "mean"),
                     s["_throughput"].get("sweeps_per_s")))
    print(f"{'run':30s} {'rho':>7s} {'x_A':>6s} {'tau(x_A)':>10s} {'tau(rho)':>10s} {'h12':>6s} {'h6':>6s} {'sw/s':>8s}")
    for r in rows:
        print(f"{r[0][-30:]:30s} {r[1]:7.4f} {r[2]:6.3f} {r[3]:10.0f} {r[4]:10.0f} {r[5]:6.3f} {r[6]:6.3f} "
              f"{(r[7] or float('nan')):8.1f}")


if __name__ == "__main__":
    main()
