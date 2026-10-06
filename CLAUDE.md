# CLAUDE.md — HCSS λ* = 2cos15° pilot 3b

## Context
2D hard-core square-shoulder model (sigma = 1, eps = 1, k_B = 1): r < 1 forbidden, 1 <= r < lam costs eps, r >= lam free.
Pilot 3: hexlat tiling (s_conf bound ln2/3) loses to the B crystal by ~0.08 k_BT per particle (Frenkel-Ladd); A is far higher;
B did not melt in the coexistence maps (x_A stayed 0); A slabs were consumed by the tiling.
Pilot 3b (README.md): free energy of the 3.12.12 tiling with independently filled 12-gons (s_conf >= ln(4421)/19 = 0.442),
B | 3.12.12 coexistence, and a re-check of the pilot-3 B | hexlat runs.

## Conventions (do not change without saying so)
- Box = (a, b, c), box vectors a1 = (a, 0), a2 = (b, c); fractional positions; image counters for unwrapping.
- NPT box moves sample (ln a, ln c, b) with the (N+1) ln(V'/V) Jacobian; coexistence runs use no shear moves.
- Frenkel-Ladd: particle 0 fixed, beta*U_spring = Lam * sum |dr|^2, thermal wavelength 1; compare beta*g only at equal T, P, similar N.
- dodecagon_fillings.npz must stay next to hcss_mc.py.

## Workflow
1. `pip install -r requirements.txt`; `python -m pytest -q test_hcss_mc.py test_flip.py test_fl.py` must give 15 passed.
2. Copy or link the pilot-3 `coex/` directory here if you want the automatic re-check; then `NPROC=<cores> bash run_package3b.sh`.
3. If a Frenkel-Ladd run reports an overlap-free fraction < 0.8 or an infinite beta_g, rerun with larger --lam-max and say so.

## Reporting
- Report in Korean in `results3b.md` following README sections 5-6, with tables (2-sigma errors) and figure paths.
- Separate the vibrational part (beta_g before subtracting s_conf) from the configurational bound in the tables.
- Do not edit the theory text in README.
