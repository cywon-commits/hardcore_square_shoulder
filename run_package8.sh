#!/usr/bin/env bash
# Pilot 8: order-parameter analysis of the existing runs + entropic phason stiffness.
# Place (or link) the pilot-2 runs2/ and pilot-3 coex/ directories here before running.  NPROC=4 bash run_package8.sh
set -euo pipefail
NPROC=${NPROC:-$(nproc)}
mkdir -p op8
python build_approx.py --n1 2 --n2 3 --out approx2_2x3.npz
python build_approx.py --n1 3 --n2 4 --out approx2_3x4.npz
# (3) entropic stiffness on two system sizes, two seeds each (parallel)
printf '%s\n' \
  "python op_stiffness.py --approx approx2_2x3.npz --sweeps 6000 --burn 500 --every 20 --seed 1 --dPFL 0.06:0.067,0.08:0.088 --out op8/stiff_2x3_s1.json" \
  "python op_stiffness.py --approx approx2_2x3.npz --sweeps 6000 --burn 500 --every 20 --seed 2 --dPFL 0.06:0.067,0.08:0.088 --out op8/stiff_2x3_s2.json" \
  "python op_stiffness.py --approx approx2_3x4.npz --sweeps 6000 --burn 500 --every 20 --seed 1 --dPFL 0.06:0.067,0.08:0.088 --out op8/stiff_3x4_s1.json" \
  "python op_stiffness.py --approx approx2_3x4.npz --sweeps 6000 --burn 500 --every 20 --seed 2 --dPFL 0.06:0.067,0.08:0.088 --out op8/stiff_3x4_s2.json" \
  | xargs -P "$NPROC" -I{} sh -c "{} > /dev/null 2>&1"
python - << 'PY' | tee op8/stiffness_summary.txt
import json, glob
for f in sorted(glob.glob("op8/stiff_*.json")):
    d = json.load(open(f)); w = d["window"]
    print(f, "N", d["N"], "K_sum/particle %.3f" % d["K_sum_particle"], "moved %.2f" % d["frac_vertices_ever_moved"],
          " ".join(f"T={t}: dP_sp={v['dP_spinodal']:.3f} vs dP_FL={v['dP_FL']} ok={v['inequality_holds']}" for t, v in w.items()))
PY
# (1) identity on ideal structures and on the last snapshots of existing runs
SNAPS=$(ls runs2/*/snap_*.npz coex/*/snap_*.npz 2>/dev/null | awk -F/ '{print $0}' | sort | awk 'NR%5==0') || true
python op_identity.py $SNAPS --out op8/identity.json | tee op8/identity_summary.txt
# (2) A -> tiling transformation and B | tiling interface
if ls -d coex/A_T* >/dev/null 2>&1; then python op_coex.py coex/A_T* coex/B_T* --out op8/coex_op.json | tee op8/coex_summary.txt; fi
# (4) melting
if ls -d runs2/T*_* >/dev/null 2>&1; then python op_melting.py runs2/T*_rows runs2/T*_hexlat runs2/T*_A runs2/T*_B --last 2 --out op8/melting_op.json | tee op8/melting_summary.txt; fi
