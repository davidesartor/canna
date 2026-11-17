# F70 (2026-11-18): the MCMC appendix rebuilt from the six packed references; one seed each, the Fisher forecast fair from SNR 15, wider than the chains on the full band at SNR 5-15, and its fdot overflows on faint full-band sources

**Outcome.**
- **The references are one seed per window.** `n_seeds` is 1 in every `reports/mcmc/data/eval-<pid>.npz`
  (Davide's 69b03f7); the appendix said two. R-hat <= 1.073 (full-4; narrow <= 1.005), ESS >= 1097, as
  F67 quoted. Seed agreement and burn-in rest on the narrow-4 long chains (F68).
- **app:mcmc no longer uses the old 64-window chains of the previous prior.** New
  `paper/scripts/mcmc_reference.py` (load + fold the packed references) feeds `numbers_mcmc.py`,
  `fig_mcmc.py` (binned Fisher/MCMC per band and S) and `fig_mcmc-posterior.py` (narrow-4, window 101:
  SNR 3.8/4.6/12.8/19.3). `numbers_mcmc.py` runs whole again. Cost from our own TREX chains
  (`import_mcmc_cost.py`): 400k iterations on 2 cores = 4.3-5.4 h narrow S=4, 8.8-9.5 h S=8, ~34-46 h full
  S=16 (extrapolated from the running prior-started chains).
- **Fisher against the chains, 5376 sources:** from SNR 40 the forecast's median width per parameter is
  0.89-0.96x the chains' (59% of ratios in 0.8-1.25), SNR 15-40 56%. Below SNR 5 the chains spread f0
  like the prior, the forecast does not (f0 0.18x). **Full band, SNR 5-15: forecast 2.2-2.5x WIDER in f0**
  (and 2.3x in fdot) than the chains. Not settled which side is off (Fisher's Gaussian fdot prior vs chains
  near their Fisher start): read the running prior-started full-16 chains there.
- **Fisher fdot overflows on the full band:** 44-45% of eval sources (all SNR < 4) have about half their
  Fisher draws at fdot = inf (exp of log fdot); `pack` gives them scale 1, so their fdot is lost. f0 draws
  finite; narrow band unaffected.
- **Agreement with Davide's 2 Nov probes (royal-ferret narrow-B):** sampler convergence agrees (32 uniform
  midpoint steps converged, power-2/log 1.00 by 32; his Euler 0.958 at 32 vs our 0.78/0.92, his f0
  widths ~27 bins vs our loud sources). Robustness agrees on "noise x2 = confident and wrong"; his empty
  window slots 0.34 SNR vs our 2.7 (different, older model). His catalogue-prior mock ladder README
  matches our rerun except the Gaussian mock's p-values (f0 0.141 / fdot 0.422 vs our 0.042 / 0.255;
  prior mock identical): the Gaussian draws are not reproducible across machines.
- Paper TODOs solved: app:mcmc's TODO(new chains); the mock-ladder table's TODO (F69). New red TODOs:
  Fisher fdot overflow (fix), full-band SNR 5-15 Fisher vs chains (check), Gaussian mock reproducibility
  (check), fisher-diagonal.json Hessian statistics (rerun), full-16 cost (update when the chains end).
