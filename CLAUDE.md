# CLAUDE.md — HCSS pilot 4 (stability window in lambda, P, T)

## Context
2D hard-core square-shoulder model (sigma = 1, eps = 1, k_B = 1): r < 1 forbidden, 1 <= r < lam costs eps, r >= lam free.
Pilot 3b established that the 3.12.12 tiling (s_conf >= ln4421/19) beats the B crystal by ~0.10-0.11 k_BT per particle at
lam = 1.93, P = 0.735 (T = 0.06, 0.08). Pilot 4 (README.md) tests the hard-contact prediction of the tiling window in
(lam, P, T), which uses only two entropy constants fixed at the reference point.

## Conventions (do not change without saying so)
- lattice_state("B", lam): contact B crystal with apex 2 asin(1/(2 lam)); "dodeca"/"hexlat": edge max(lam, lam*).
- Frenkel-Ladd: particle 0 fixed, beta*U_spring = Lam * sum |dr|^2, thermal wavelength 1; compare beta*g only at equal (lam, P, T) and similar N.
- dodecagon_fillings.npz must stay next to hcss_mc.py.

## Workflow
1. `pip install -r requirements.txt`; `python -m pytest -q test_hcss_mc.py test_flip.py test_fl.py` must give 16 passed.
2. `NPROC=<cores> bash run_package4.sh` (writes fe4/*.json, summary4.txt, theory_window_T006/T008.{json,png}).
3. If a Frenkel-Ladd run reports an overlap-free fraction < 0.8 or an infinite beta_g, rerun it with larger --lam-max and say so.

## Reporting
- Report in Korean in `results4.md` following README sections 5-6, with tables (2-sigma errors) and figure paths.
- Do not edit the theory text or the prediction table in README.
