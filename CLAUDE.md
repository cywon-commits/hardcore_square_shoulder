# CLAUDE.md — HCSS pilot 7 (final numbers with corrected NPT start; lower end of the tiling band)

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
1. `pip install -r requirements.txt`; `python -m pytest -q test_hcss_mc.py test_flip.py test_fl.py` must give 17 passed.
2. `NPROC=<cores> bash run_package7.sh` (writes fe7/, summary7.txt, phase_window_final.{json,png}).
3. If a Frenkel-Ladd run reports an overlap-free fraction < 0.8 or an infinite beta_g, rerun it with larger --lam-max and say so.

## Reporting
- Report in Korean in `results7.md` following README sections 5-6, with tables (2-sigma errors) and figure paths.
- Do not edit the theory text or the prediction table in README.

## Pilot-7 method notes
- fl_free_energy.py starts NPT at v0 + c*2T/P with c = 1 for crystals and c = 0.8 for tilings (--expand-c overrides);
  below lam* the tiling is kind dodeca1_jam. All runs use --sites mean. Do not revert to the pre-expanded tiling start.
