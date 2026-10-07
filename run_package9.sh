#!/usr/bin/env bash
# Pilot 9: corrected order parameters (exact eta, Gabriel shoulder bonds, raw-angle psi_n), long phason-stiffness runs
# with static/fluctuating mode separation, re-analysis of pilot-2/3 data, and a fine melting scan T = 0.120 ... 0.150.
# Place (or link) runs2/ and coex/ here first.  NPROC=4 bash run_package9.sh
set -euo pipefail
NPROC=${NPROC:-$(nproc)}
mkdir -p op9 melt9
python build_approx.py --n1 2 --n2 3 --out approx2_2x3.npz
python build_approx.py --n1 3 --n2 4 --out approx2_3x4.npz
jobs=()
jobs+=("python op_stiffness.py --approx approx2_2x3.npz --sweeps 60000 --every 20 --nshell 5 --seed 1 --out op9/stiff_2x3_s1.json")
jobs+=("python op_stiffness.py --approx approx2_2x3.npz --sweeps 60000 --every 20 --nshell 5 --seed 2 --out op9/stiff_2x3_s2.json")
jobs+=("python op_stiffness.py --approx approx2_3x4.npz --sweeps 60000 --every 20 --nshell 5 --seed 1 --out op9/stiff_3x4_s1.json")
for T in 0.120 0.125 0.130 0.135 0.140 0.145 0.150; do
  for K in hexlat dodeca; do
    jobs+=("python run_pilot2.py --init $K --lam 1.93 --P 0.735 --T $T --N 931 --sweeps 120000 --analyze-every 20000 --snap-every 20000 --no-sk --out melt9/T${T}_$K")
  done
done
i=0
printf '%s\n' "${jobs[@]}" | while read -r cmd; do echo "$cmd > op9/job$((i++)).log 2>&1"; done | xargs -P "$NPROC" -I{} sh -c "{}"
for f in op9/stiff_*.json; do python - "$f" << 'PY'
import json, sys
d = json.load(open(sys.argv[1])); good = sorted([m for m in d["modes"] if m["equilibrated"]], key=lambda m: m["q"])
print(sys.argv[1], "N", d["N"], "eta range", d["eta_min"], d["eta_max"], "equilibrated", d["n_equilibrated"], "/", d["n_modes"], "K_median", round(d["K_particle_equilibrated"], 3), d["window"])
for m in sorted(d["modes"], key=lambda m: m["q"])[:24]:
    print("   q %.4f m %-9s K %7.3f static %.2f eq %s" % (m["q"], tuple(m["m"]), m["K_particle"], m["static_fraction"], m["equilibrated"]))
PY
done | tee op9/stiffness_summary.txt
SNAPS=$(ls runs2/*/snap_*.npz coex/*/snap_*.npz 2>/dev/null | sort | awk 'NR%5==0') || true
python op_identity.py $SNAPS --out op9/identity.json | tee op9/identity_summary.txt
if ls -d coex/A_T* >/dev/null 2>&1; then python op_coex.py coex/A_T* coex/B_T* --out op9/coex_op.json | tee op9/coex_summary.txt; fi
python op_melting.py melt9/T*_hexlat melt9/T*_dodeca --last 3 --out op9/melting_fine.json | tee op9/melting_fine_summary.txt
if ls -d runs2/T*_* >/dev/null 2>&1; then python op_melting.py runs2/T*_rows runs2/T*_hexlat runs2/T*_A runs2/T*_B --last 2 --out op9/melting_runs2.json | tee op9/melting_runs2_summary.txt; fi
