#!/usr/bin/env bash
# Pilot 7: final Frenkel-Ladd numbers with the corrected NPT start (v0 for crystals, jammed structure for the tiling
# below lam*), all with NPT-mean Einstein sites; re-computation of the pilot-4/5 points below lam*; lower end of the band.
# Override: NPROC=4 bash run_package7.sh
set -euo pipefail
NPROC=${NPROC:-$(nproc)}; N=${N:-931}
mkdir -p fe7 logs7
jobs=()
pt(){  # lam P T tilingkind
  for K in A B $4; do
    jobs+=("python fl_free_energy.py --sites mean --kind $K --lam $1 --P $2 --T $3 --N $N --out fe7/${K}_lam$1_P$2_T$3.json")
  done
}
# references at lam = 1.93 (random-filling 3.12.12; no compression above/at lam*)
pt 1.93 0.735 0.06 dodeca; pt 1.93 0.735 0.08 dodeca
# pilot-4/5 points below lam*, recomputed (tiling = jammed single filling)
pt 1.90 0.735 0.06 dodeca1_jam; pt 1.91 0.728 0.06 dodeca1_jam; pt 1.92 0.721 0.06 dodeca1_jam; pt 1.92 0.735 0.06 dodeca1_jam
pt 1.90 0.735 0.08 dodeca1_jam; pt 1.92 0.721 0.08 dodeca1_jam
# lower end of the band at T = 0.06: P_AB(lam) = (1 - 0.645 T)/(v_A - v_B) and +- 0.035
pt 1.85 0.778 0.06 dodeca1_jam; pt 1.85 0.813 0.06 dodeca1_jam; pt 1.85 0.848 0.06 dodeca1_jam
pt 1.80 0.858 0.06 dodeca1_jam; pt 1.80 0.893 0.06 dodeca1_jam; pt 1.80 0.928 0.06 dodeca1_jam
pt 1.78 0.893 0.06 dodeca1_jam; pt 1.78 0.928 0.06 dodeca1_jam; pt 1.78 0.963 0.06 dodeca1_jam
pt 1.75 0.951 0.06 dodeca1_jam; pt 1.75 0.986 0.06 dodeca1_jam; pt 1.75 1.021 0.06 dodeca1_jam
# temperature dependence of the lower end: T = 0.08 at P_AB(lam, T)
pt 1.80 0.881 0.08 dodeca1_jam; pt 1.78 0.916 0.08 dodeca1_jam
i=0
printf '%s\n' "${jobs[@]}" | while read -r cmd; do echo "$cmd > logs7/job$((i++)).log 2>&1"; done | xargs -P "$NPROC" -I{} sh -c "{}"
python summary7.py | tee summary7.txt
python theory_window.py --ref-dir fe7 --T0 0.06 --kappa 1.7 --out phase_window_final | tee -a summary7.txt
