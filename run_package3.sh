#!/usr/bin/env bash
# Pilot 3: direct coexistence (6 runs) + Frenkel-Ladd free energies (8 runs + 2 finite-size checks).
# Override: NPROC=4 SWEEPS=200000 bash run_package3.sh
set -euo pipefail
LAM=${LAM:-1.93}; P=${P:-0.735}; SWEEPS=${SWEEPS:-200000}; NPROC=${NPROC:-$(nproc)}
mkdir -p coex fe logs3
jobs=()
for T in 0.06 0.08 0.10; do
  for L in B A; do
    jobs+=("python run_coex.py --left $L --T $T --P $P --lam $LAM --sweeps $SWEEPS --out coex/${L}_T${T}")
  done
done
for T in 0.06 0.08; do
  for K in A B; do
    jobs+=("python fl_free_energy.py --kind $K --T $T --P $P --lam $LAM --N 1000 --out fe/${K}_T${T}_N1000.json")
  done
  for S in 1 2; do
    jobs+=("python fl_free_energy.py --kind hexlat --seed $S --T $T --P $P --lam $LAM --N 1000 --out fe/hexlat_s${S}_T${T}_N1000.json")
  done
done
jobs+=("python fl_free_energy.py --kind B --T 0.06 --P $P --lam $LAM --N 2000 --out fe/B_T0.06_N2000.json")
jobs+=("python fl_free_energy.py --kind hexlat --seed 1 --T 0.06 --P $P --lam $LAM --N 2000 --out fe/hexlat_s1_T0.06_N2000.json")
i=0
printf '%s\n' "${jobs[@]}" | while read -r cmd; do echo "$cmd > logs3/job$((i++)).log 2>&1"; done | xargs -P "$NPROC" -I{} sh -c "{}"
python summarize3.py --coex coex/* --fe fe/*_N1000.json | tee summary3.txt
python summarize3.py --fe fe/B_T0.06_N2000.json fe/hexlat_s1_T0.06_N2000.json | tee -a summary3.txt
