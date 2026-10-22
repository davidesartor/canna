# F3 (2026-09-30): eval fixes
- **ODE steps 4 → 32**, RK4 loop in `lax.fori_loop`. At 4 steps the plots were ~1000× broader
  than Fisher; integration error alone was 3.5× of that.
- **Fisher overlay** is inverted in the Jacobi-scaled frame (the raw diagonal spans 45 decades)
  and eigen-sampled with each eigenvalue floored at the prior's precision along it. It is a
  Gaussian forecast centred on the truth, not an MCMC: for unmeasured parameters (M_c below 1
  mHz) it shows only the prior width.
- **Restore:** `restore_from` works for a GPU checkpoint restored on the CPU.
