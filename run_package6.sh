#!/usr/bin/env bash
# Pilot 6: (1) Frenkel-Ladd with NPT-mean Einstein sites, (2) jammed-start vs ideal-start NPT (kinetic trapping or
# entropic preference?), (3) free energy of the jammed basin and the lower end of the tiling band.
# Override: NPROC=4 bash run_package6.sh
set -euo pipefail
NPROC=${NPROC:-$(nproc)}; N=${N:-931}; T=0.06
mkdir -p fe6 npt6 logs6
jobs=()
fl(){ jobs+=("python fl_free_energy.py --sites mean --kind $1 --lam $2 --P $3 --T $T --N $N --out fe6/$1_lam$2_P$3.json"); }
# validation of the mean-site method against pilot 3b (ideal sites): lam = 1.93, P = 0.735
fl B 1.93 0.735; fl dodeca 1.93 0.735
# lam = 1.90: all variants
for K in A B dodeca dodeca1 dodeca1_jam; do fl $K 1.90 0.735; done
# lam = 1.85 and 1.80: pressure scans with the jammed tiling; ideal-start variants at the central pressure
for P in 0.780 0.818 0.860; do for K in A B dodeca1_jam; do fl $K 1.85 $P; done; done
for K in dodeca dodeca1; do fl $K 1.85 0.818; done
for P in 0.860 0.895 0.930; do for K in A B dodeca1_jam; do fl $K 1.80 $P; done; done
for K in dodeca dodeca1; do fl $K 1.80 0.895; done
# jammed-start vs ideal-start NPT (same filling), no flips, shear moves on
for p in "1.90 0.735" "1.85 0.818" "1.80 0.895"; do
  set -- $p
  for K in dodeca1 dodeca1_jam; do
    jobs+=("python run_pilot2.py --init $K --lam $1 --P $2 --T $T --N $N --sweeps 200000 --flip-per-sweep 0 --analyze-every 5000 --snap-every 50000 --no-sk --out npt6/${K}_lam$1")
  done
done
i=0
printf '%s\n' "${jobs[@]}" | while read -r cmd; do echo "$cmd > logs6/job$((i++)).log 2>&1"; done | xargs -P "$NPROC" -I{} sh -c "{}"
python summary6.py | tee summary6.txt
