# CLAUDE.md — HCSS λ* = 2cos15° pilot 3

## Context
2D hard-core square-shoulder model (sigma = 1, eps = 1, k_B = 1): r < 1 forbidden, 1 <= r < lam costs eps, r >= lam free.
Theory: entropy-stabilized random tiling at lam* = 2cos15 deg ~ 1.9319, P* = sqrt(3) - 1. Pilot 2 showed the tiling, A and
B crystals are all locally stable (dynamically frozen) below T ~ 0.12. Pilot 3 (README.md) decides relative stability by
direct coexistence (run_coex.py) and Frenkel-Ladd free energies (fl_free_energy.py).

## Conventions (do not change without saying so)
- Box = (a, b, c), box vectors a1 = (a, 0), a2 = (b, c); fractional positions; image counters for unwrapping.
- NPT box moves sample (ln a, ln c, b) with the (N+1) ln(V'/V) Jacobian; coexistence runs use no shear moves.
- Step sizes are tuned only inside tuning windows.
- Flip move and Frenkel-Ladd conventions: see docstrings in hcss_mc.py and fl_free_energy.py (particle 0 fixed,
  beta*U_spring = Lam * sum |dr|^2, thermal wavelength 1). Compare beta*g only at equal T, P, N.

## Workflow
1. `pip install -r requirements.txt`; `python -m pytest -q test_hcss_mc.py test_flip.py test_fl.py` must give 13 passed.
2. `NPROC=<cores> bash run_package3.sh`; it writes summary3.txt.
3. If a Frenkel-Ladd run reports dA1 overlap-free fraction < 0.8 or infinite beta_g, rerun with larger --lam-max and say so.

## Reporting
- Report in Korean in `results3.md`, following README sections 5-6, with tables and figure paths.
- State every parameter you changed; give errors (2 sigma) for free-energy differences.
- Do not edit the theory text in README.
