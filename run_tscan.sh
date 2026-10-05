#!/usr/bin/env bash
# Pilot 2: temperature scan (4 initial states x 5 temperatures) + cooling/heating runs, in parallel.
# Override: NPROC=8 SWEEPS=200000 N=2000 bash run_tscan.sh
set -euo pipefail
LAM=${LAM:-1.93}; P=${P:-0.735}; N=${N:-2000}; SWEEPS=${SWEEPS:-200000}
NPROC=${NPROC:-$(nproc)}; FLIP=${FLIP:-0.05}
mkdir -p runs2 logs2
jobs=()
for T in 0.04 0.06 0.08 0.10 0.12; do
  for INIT in rows A B hexlat; do
    jobs+=("--T $T --init $INIT --sweeps $SWEEPS --out runs2/T${T}_${INIT}")
  done
done
# cooling from the fluid and heating from the random tiling (hysteresis)
COOL=0.15:40000,0.12:40000,0.10:40000,0.08:40000,0.06:40000,0.05:40000,0.04:40000
HEAT=0.04:40000,0.05:40000,0.06:40000,0.08:40000,0.10:40000,0.12:40000,0.15:40000
jobs+=("--init fluid  --T-schedule $COOL --out runs2/cool_fluid")
jobs+=("--init hexlat --T-schedule $HEAT --out runs2/heat_hexlat")
printf '%s\n' "${jobs[@]}" | xargs -P "$NPROC" -I{} sh -c \
  "python run_pilot2.py --lam $LAM --P $P --N $N --flip-per-sweep $FLIP --log-every 100 --analyze-every 1000 --snap-every 20000 --seed \$\$ {} > logs2/\$(echo '{}' | sed 's/.*--out runs2\///').log 2>&1"
python pilot_analyze.py runs2/T*_* > runs2/tscan_summary.txt
python msd.py runs2/T*_* runs2/cool_fluid runs2/heat_hexlat > runs2/msd_summary.txt
python reanalyze.py runs2/T*_* --last 3 > runs2/reanalysis_summary.txt
cat runs2/tscan_summary.txt
