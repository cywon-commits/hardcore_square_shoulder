# CLAUDE.md — HCSS λ* = 2cos15° pilot 2

## Context
2D hard-core square-shoulder model (sigma = 1, eps = 1, k_B = 1): r < 1 forbidden, 1 <= r < lam costs eps, r >= lam free.
Theory predicts an entropy-stabilized 12-fold random tiling at lam* = 2cos15 deg ~ 1.9319, P* = sqrt(3) - 1 ~ 0.7321.
Pilot 1 (T = 0.15) melted every structure; pilot 2 (README.md) scans T = 0.04-0.12, adds a 30-degree hexagon flip move,
a random-tiling initial state (hexlat), unwrapped trajectories, and re-analysis of pilot-1 snapshots.

## Conventions (do not change without saying so)
- Box = (a, b, c), box vectors a1 = (a, 0), a2 = (b, c); fractional positions s in [0,1); image counters img for unwrapping.
- Box moves sample (ln a, ln c, b); acceptance includes (N+1) ln(V'/V).
- Step sizes are tuned only inside the tuning window at the start of each temperature stage (column `tuning` = 1);
  exclude those rows from statistics.
- Flip move: point reflection through the centre of the six boundary particles; reverse detection must find the same
  boundary set; acceptance min(1, n_old/n_new exp(-beta dU)). Keep this exact.

## Workflow
1. `pip install -r requirements.txt`; `python -m pytest -q test_hcss_mc.py test_flip.py` must give 9 passed, before and after any change.
2. Task (a): `python reanalyze.py <pilot-1 run dirs>`. Tasks (b, c): `NPROC=<cores> bash run_tscan.sh`.
3. Any new move type needs its own detailed-balance test.

## Reporting
- Report in Korean, in `results2.md`, following README sections 5-6; include the tables and the figure paths.
- Give tau_int as a lower bound when it exceeds 1/50 of the run length; state any tolerance or parameter you changed.
- Do not edit the theory text in README.
