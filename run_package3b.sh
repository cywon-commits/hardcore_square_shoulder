#!/usr/bin/env bash
# Pilot 3b: free energy of the 3.12.12 random-filled tiling vs B and hexlat at matched N, B | 3.12.12 coexistence,
# and a re-check of the pilot-3 B | hexlat coexistence runs.
# Override: NPROC=4 SWEEPS=200000 bash run_package3b.sh
set -euo pipefail
LAM=${LAM:-1.93}; P=${P:-0.735}; SWEEPS=${SWEEPS:-200000}; NPROC=${NPROC:-$(nproc)}
mkdir -p fe3b coex3b logs3b
jobs=()
for T in 0.06 0.08; do
  for S in 1 2; do
    jobs+=("python fl_free_energy.py --kind dodeca --seed $S --T $T --P $P --lam $LAM --N 931 --out fe3b/dodeca_s${S}_T${T}_N931.json")
  done
  jobs+=("python fl_free_energy.py --kind B --T $T --P $P --lam $LAM --N 931 --out fe3b/B_T${T}_N931.json")
  jobs+=("python fl_free_energy.py --kind hexlat --seed 1 --T $T --P $P --lam $LAM --N 931 --out fe3b/hexlat_s1_T${T}_N931.json")
done
jobs+=("python fl_free_energy.py --kind dodeca --seed 1 --T 0.06 --P $P --lam $LAM --N 1900 --out fe3b/dodeca_s1_T0.06_N1900.json")
jobs+=("python fl_free_energy.py --kind B --T 0.06 --P $P --lam $LAM --N 1900 --out fe3b/B_T0.06_N1900.json")
for T in 0.06 0.08 0.10; do
  jobs+=("python run_coex.py --left B --right dodeca --T $T --P $P --lam $LAM --sweeps $SWEEPS --out coex3b/B_dodeca_T${T}")
done
i=0
printf '%s\n' "${jobs[@]}" | while read -r cmd; do echo "$cmd > logs3b/job$((i++)).log 2>&1"; done | xargs -P "$NPROC" -I{} sh -c "{}"
{
  echo "== free energies, N ~ 931"; python summarize3.py --fe fe3b/*_N931.json
  echo "== free energies, N ~ 1900"; python summarize3.py --fe fe3b/*_N1900.json
  echo "== B | 3.12.12 coexistence"; python summarize3.py --coex coex3b/*
} | tee summary3b.txt
# re-check of the pilot-3 B | hexlat runs, if their directories are present
if ls coex/B_T0.* >/dev/null 2>&1; then python coex_check.py coex/B_T0.* | tee -a summary3b.txt; fi
python coex_check.py coex3b/* | tee -a summary3b.txt
