# F5 (2026-10-01): the floor is not numerical
- **ODE:** loudest-source width 0.76 / 0.42 / 0.27 / 0.22 / 0.21 bins at 4 / 8 / 16 / 32 / 64
  RK4 steps; converged by 32.
- **Eval precision:** fp32 compute gives 0.197 bins vs 0.210 in bf16 (16 steps).
- **bf16 input cell:** the f₀ coordinate's cell is 0.012 bins at the window centre to 0.094 at
  the edges. Width does not track it (Spearman ρ = −0.11, p = 0.64, 21 sources above SNR 50),
  nor |coordinate| or SNR.
- **Open:** a network *trained* in bf16 may have learned only features coarser than its
  activation noise. Repeat after E2: if the floor then tracks the bf16 cell, precision is the
  limit.
