#!/usr/bin/env bash
# Pilot 4: test the hard-contact prediction of the tiling stability window in (lambda, P, T).
# Each point: Frenkel-Ladd for A, B and the 3.12.12 tiling (dodeca, s_conf >= ln4421/19).
# Override: NPROC=4 bash run_package4.sh
set -euo pipefail
NPROC=${NPROC:-$(nproc)}; N=${N:-931}
mkdir -p fe4 logs4
POINTS=(
  "1.93 0.735 0.06" "1.93 0.735 0.08"                         # references (fix sigma_B, sigma_A; check T-independence)
  "1.93 0.66 0.06" "1.93 0.70 0.06" "1.93 0.76 0.06"          # pressure scan:  A | tiling | tiling | B
  "1.93 0.70 0.08" "1.93 0.77 0.08"                           # window widens with T
  "1.90 0.735 0.06" "1.91 0.728 0.06" "1.92 0.721 0.06" "1.92 0.735 0.06"   # lambda < lambda*
  "1.94 0.702 0.06" "1.96 0.676 0.06" "1.96 0.735 0.06"       # lambda > lambda*
)
jobs=()
for p in "${POINTS[@]}"; do
  set -- $p
  for K in A B dodeca; do
    jobs+=("python fl_free_energy.py --kind $K --lam $1 --P $2 --T $3 --N $N --out fe4/${K}_lam$1_P$2_T$3.json")
  done
done
i=0
printf '%s\n' "${jobs[@]}" | while read -r cmd; do echo "$cmd > logs4/job$((i++)).log 2>&1"; done | xargs -P "$NPROC" -I{} sh -c "{}"
{
  echo "== reference T0 = 0.06"; python theory_window.py --ref-dir fe4 --T0 0.06 --out theory_window_T006
  echo "== reference T0 = 0.08 (sigma constancy check)"; python theory_window.py --ref-dir fe4 --T0 0.08 --out theory_window_T008
} | tee summary4.txt
