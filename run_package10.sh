#!/usr/bin/env bash
# Pilot 10: topological defects (4D Burgers vectors in Z[zeta_12]) across melting.  Needs the pilot-9 melting scan melt9/
# (and optionally pilot-2 runs2/) in this folder.  Pure analysis of existing snapshots, plus a longer equilibrated scan.
set -euo pipefail
NPROC=${NPROC:-$(nproc)}
mkdir -p b10 melt10
# (A) existing data
python burgers_analysis.py melt9/T*_hexlat melt9/T*_dodeca --last 3 --out b10/burgers_melt9.json | tee b10/burgers_melt9.txt
if ls -d runs2/T*_* >/dev/null 2>&1; then python burgers_analysis.py runs2/T*_hexlat runs2/T*_rows runs2/T*_A runs2/T*_B --last 2 --out b10/burgers_runs2.json | tee b10/burgers_runs2.txt; fi
# (B) longer runs with dense snapshots below and around melting (pilot 9 was not stationary at T = 0.125-0.14)
jobs=()
for T in 0.100 0.110 0.120 0.125 0.130 0.135 0.140 0.150; do
  for K in hexlat dodeca; do
    jobs+=("python run_pilot2.py --init $K --lam 1.93 --P 0.735 --T $T --N 931 --sweeps 400000 --analyze-every 40000 --snap-every 40000 --no-sk --out melt10/T${T}_$K")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "$NPROC" -I{} sh -c "{} > /dev/null 2>&1"
python burgers_analysis.py melt10/T*_hexlat melt10/T*_dodeca --last 5 --out b10/burgers_melt10.json | tee b10/burgers_melt10.txt
python op_melting.py melt10/T*_hexlat melt10/T*_dodeca --last 5 --out b10/melting_melt10.json | tee b10/melting_melt10.txt
