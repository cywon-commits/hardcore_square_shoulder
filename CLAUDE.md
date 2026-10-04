# CLAUDE.md — HCSS λ* = 2cos15° pilot

## Context
2D hard-core square-shoulder model (sigma = 1, eps = 1, k_B = 1): r < 1 forbidden, 1 <= r < lam costs eps, r >= lam free.
Theory predicts an entropy-stabilized 12-fold random tiling at lam* = 2cos15 deg ~ 1.9319, P* = sqrt(3) - 1 ~ 0.7321.
The task is the pilot run described in README.md: measure tau_int of the tile fraction x_A at lam = 1.93, T = 0.15, P = 0.735.

## Conventions (do not change without saying so)
- Box = (a, b, c) with box vectors a1 = (a, 0), a2 = (b, c); positions stored as fractional s in [0,1).
- Box moves sample (ln a, ln c, b) symmetrically; acceptance includes (N+1) ln(V'/V). Keep this Jacobian.
- Step sizes are tuned only during --tune-sweeps; never adapt them afterwards (detailed balance).
- Tile classes in hcss_analysis.py: S (r < lam - tol_in), L (near lam), X; tiles A = LLL, B = SLL, C = SSL, D = SSS.

## Workflow
1. `pip install -r requirements.txt`, then `python -m pytest -q test_hcss_mc.py` (must be 4 passed) before and after any code change.
2. Throughput check (README 4.2), then `bash run_matrix.sh`, then `python pilot_analyze.py runs/*`.
3. Long runs: use `--restart` to resume from `state.npz`; run the three initial states in parallel.
4. Any new move type (e.g. the tile-flip move in README section 7) needs its own detailed-balance test.

## Reporting
- Report results in Korean (the user prefers Korean), with the table and decision from README sections 5-6.
- State tolerances or parameters you changed, and give tau_int as a lower bound when it exceeds 1/50 of the run length.
- Do not edit the theory predictions in README; add measured values in a separate results file (results.md).
