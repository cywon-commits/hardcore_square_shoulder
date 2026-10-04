#!/usr/bin/env bash
# Pilot matrix: three initial states at one state point near the triple point, run in parallel.
# Override with environment variables, e.g.  SWEEPS=300000 T=0.20 bash run_matrix.sh
set -euo pipefail
LAM=${LAM:-1.93}; T=${T:-0.15}; P=${P:-0.735}; N=${N:-2000}
SWEEPS=${SWEEPS:-1000000}; TUNE=${TUNE:-5000}
mkdir -p runs logs
for INIT in rows A B; do
  OUT=runs/T${T}_${INIT}
  python run_pilot.py --lam $LAM --T $T --P $P --init $INIT --N $N \
      --sweeps $SWEEPS --tune-sweeps $TUNE --log-every 100 --analyze-every 1000 \
      --snap-every 100000 --seed $RANDOM --out $OUT $( [ -f $OUT/state.npz ] && echo --restart ) \
      > logs/T${T}_${INIT}.log 2>&1 &
done
wait
python pilot_analyze.py runs/T${T}_rows runs/T${T}_A runs/T${T}_B
