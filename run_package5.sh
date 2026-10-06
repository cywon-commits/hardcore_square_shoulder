#!/usr/bin/env bash
# Pilot 5: (2) 1/T scaling of the lam < lam* deviation, (3) lower end of the tiling band (lam = 1.85, 1.80).
# Needs the pilot-4 results in ./fe4 for the T = 0.06 reference and comparison (copy or link them here).
# Override: NPROC=4 bash run_package5.sh
set -euo pipefail
NPROC=${NPROC:-$(nproc)}; N=${N:-931}
mkdir -p fe5 logs5
POINTS=(
  "1.90 0.735 0.08" "1.91 0.728 0.08" "1.92 0.721 0.08" "1.92 0.735 0.08"   # (2) same points as pilot 4, T = 0.08
  "1.85 0.818 0.06" "1.85 0.780 0.06" "1.85 0.860 0.06"                     # (3) predicted window (0.795, 0.841) at 1.85
  "1.80 0.895 0.06" "1.80 0.860 0.06" "1.80 0.930 0.06"                     #     predicted window (0.883, 0.907) at 1.80
)
jobs=()
for p in "${POINTS[@]}"; do
  set -- $p
  for K in A B dodeca; do
    jobs+=("python fl_free_energy.py --kind $K --lam $1 --P $2 --T $3 --N $N --out fe5/${K}_lam$1_P$2_T$3.json")
  done
done
i=0
printf '%s\n' "${jobs[@]}" | while read -r cmd; do echo "$cmd > logs5/job$((i++)).log 2>&1"; done | xargs -P "$NPROC" -I{} sh -c "{}"
{
  echo "== deviation from the rigid theory and its 1/T scaling"; python deviation_scaling.py fe4 fe5
  echo "== effective-volume theory (kappa = 1.7) vs all points"
  mkdir -p tmp_all && cp fe4/*.json fe5/*.json tmp_all/
  python theory_window.py --ref-dir tmp_all --T0 0.06 --kappa 1.7 --out theory_window_kappa
} | tee summary5.txt
