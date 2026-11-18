# F71 (2026-11-18): full S=16's prior-started chains repeat narrow S=8 below SNR 7 but are 1.2-1.7x WIDER than the reference at SNR 7-22 and still widening; the crashed full-4 run continues from epoch 44 with a Student-t frame NLL

**Outcome.**
- **Full S=16 prior-started chains** (TREX 18033012: windows 70, 77, 31, 42; 31 and 42 hold only SNR < 3 sources;
  two seeds; 200 000 iterations; 96 GB, chunk 10 000), read label-free like
  [F68](068-reference-stuck-frame-refinement-mock-null.md)'s narrow S=8 (18033013,
  `reports/mcmc/data/convergence-261113/labelfree-prior-full-16.json`):
  - SNR < 3: chain 1.00x the prior's f0 width, flow 0.97x, reference 0.43x (stuck at its Fisher start), so the
    prior is the reference there, as on narrow S=8.
  - SNR 3-7: found share 0.07-0.14 in the prior-started chains (flat over the chain), 0.97-1.00 in the
    Fisher-started reference, 0.61-0.79 for the flow: the same bracket.
  - SNR >= 7: every loud source found, but f0 widths 1.2-1.7x the reference's (median 1.4; narrow S=8: 1.11),
    growing over the chain for 6 of the 9 sources. The full-16 reference is likely too narrow at SNR 7-~20, so
    the full-16 loud ratios of the paper are upper bounds; flow/reference 0.55-1.65 on these sources. This also
    takes part of F70's "Fisher 2.2-2.5x wider than the chains at SNR 5-15" onto the reference. Four windows
    only: not settled; longer prior-started chains cost ~40 CPU h each.
  - Wall time 16.7-20.0 h per 200 000 iterations (2 cores), i.e. 33-40 h per 400 000-iteration chain (paper:
    app:mcmc-cost, now measured instead of extrapolated; import_mcmc_cost.py scales completed chains to 400 000).
- **Student-t frame NLL** (0f1def1 on GitHub oneshot): `frame_nu` (0 = Gaussian) bounds each frame entry's
  gradient, |d/d delta_mu| <= (nu+1)/sqrt(nu), |d/d delta_s| <= 2 nu e^{delta_s}, where the Gaussian's -2k and
  2 e^{delta_s}(1 - k^2) are unbounded (full-4's divergence, F67: frame loss 1e5 -> 1e17 in epochs 48-50).
  The logged frame NLL, catalog loss and best checkpoint stay Gaussian. `continue_from=<run>/<epoch>` starts a
  new run on the current commit from another run's snapshot, on its schedule (resumes are pinned to their commit).
- **`261103-n4-full-B-t44`** (TREX 18085060, eval 18085061 after it): full-4 from snapshot 44 with nu = 4,
  epochs 45-64 and the original cooldown; the forks e4-e32 precede epoch 44 and stay valid.

**Selection note.** The eval-catalogue loss that picks each row's model is noisy (0.02-0.09 between epochs)
against main/e32 gaps of 0.01-0.06, mostly in the frame part. The epoch-median training loss (fresh data, no
overfitting) puts the 64-epoch main runs ~0.02 nats ahead of e32 in 4 of 5 healthy rows (narrow-2 a tie), and
the f0 SBC of e32 is slightly worse on narrow-4 (p ~ 0, 68% coverage 0.64 at SNR >= 7) and full-8 (mean f0
rank 0.528 at SNR >= 7). Proposed (not yet applied in the paper): main runs in every row except full-4.

**First ten epochs of t44 (45-54, read from checkpoint 54's histories).** The logged frame NLL (Gaussian, with k^2),
flow z-NLL and their catalog sum are not comparable across the switch: the t-frame is ~0.6x narrower (catalogue
mean 2 delta_s -0.92..-1.21 at epochs 39-44 -> -1.90..-2.15), its Gaussian k^2 is 2.2-3.2 instead of ~0.9, and the
flow's z-NLL rose -1.0 -> -0.57 because z = (x - mu)/sigma widened with it. The model's own NLL in x, flow + 2 delta_s
(binned, approximate), improved from -1.99..-2.16 to -2.48..-2.74. No blow-up: per-step max gradient norm 0.4-1.9e5
through epochs 48-54 (the Gaussian run: 7e9 -> 9e17 in epochs 48-50). The best-checkpoint rule (catalog frame +
flow) stayed at epoch 45, so the eval reads the last checkpoint instead (TREX 18162618); a frame_nu run needs the
selection on flow + 2 delta_s.
