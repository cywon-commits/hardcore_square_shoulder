# CLAUDE.md — HCSS pilot 8 (order-parameter analysis)

## Context
2D hard-core square-shoulder model; lam* = 2cos15. The tiling phases differ only by the phason strain E (w = alpha z + beta conj z,
charges 4 and 6). Exact identity: eta (tile counts) = det E = |alpha|^2 - |beta|^2. Pressure is the field conjugate to eta.
The 12-fold state has eta = 0 (triangles : rhombi = 2/sqrt3, x_A = 0.366) — not 1:1 as older READMEs said.

## Conventions
- op.py: shoulder bonds |r - lam| within (tol_in, tol_out), quantised to 30-degree classes after removing the global
  orientation offset; perp step lam* e^{i 5 pi k/6}; E from loops winding around the periodic box.
- Do not change the definitions of eta, alpha, beta or the stiffness normalisation <|w_q|^2> = 8/(A K_area q^2).

## Workflow
1. `pip install -r requirements.txt`; run `python -m pytest -q test_op.py test_hcss_mc.py test_flip.py test_fl.py` (all must pass).
2. Copy/link pilot-2 runs2/ and pilot-3 coex/ here, then `NPROC=<cores> bash run_package8.sh` (outputs in op8/).

## Reporting
- Korean, in `results8.md`, following README section 5 (tables, figure paths, which predictions hold and which do not).
