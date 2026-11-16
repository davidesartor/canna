# F68 (2026-11-13): the narrow-8 and full-16 references are stuck at their Fisher start on faint sources, the full band's loud misses above 10 mHz are the flow refining its frame too little, and the Gaussian mock as a null halves what F67 read into the validation offsets

**Outcome.**
- **The joint MCMC references of narrow-8 and full-16 are wrong on faint sources, not the flow.**
  Below SNR 3 the posterior is the prior, so the f0-ordered prior width is the reference. The flow's
  f0 width is 0.84–0.99× the prior's on all six problems; the chains' is 0.84–0.97× on four, but
  **0.19× (narrow-8) and 0.60× (full-16)**. The chains start every walker from the Fisher draws,
  whose f0 width at SNR 2 is ~0.3 bins; with 64 or 128 parameters the ensemble never leaves that
  start for faint sources. A 10k-iteration bench with the faint sources started from the prior
  (narrow-8, window 66, `long_chain_prior.py`) spreads them over 11–27 bins against 0.08–0.36 in the
  reference. F67's "narrow-8 faint f0 5.6×, full-16 2×" measured the reference. The overlap
  hypothesis F67 proposed is also refuted: narrow-8's faint f0 ratio is 6.0 within 1 bin of a
  neighbour and 5.75 at 10–30 bins.
- **Loud full-band sources above 10 mHz: the flow refines its frame too little.** The frame
  (data-only mean and width per source) on loud sources below 10 mHz is 54 / 71 / 91× the MCMC f0
  width (S = 4 / 8 / 16) and the flow narrows it 24 / 17 / 15-fold. Above 10 mHz the frame is about
  twice as wide (121 / 163 / 234×), but the flow narrows it only 5 / 4 / 2-fold, ending 28 / 37 /
  92× the MCMC. Narrow-4, for contrast, narrows a 56× frame 26-fold at SNR 15–40.
- **The Gaussian mock is the null the validation table needed, and it moves F67's reading.** On
  1000 prior injections per problem the mock (Fisher-linearised posterior, approximately
  calibrated) is rejected by every γ test (p < 0.001), coordinates and TARP included, so p-values
  carry no information at this size. Its log-likelihood rank offset grows with the dimension too,
  +0.14 (narrow-2) to +0.41 (full-16): over 8S coordinates a near-Gaussian approximation already
  loses likelihood against the truth. The flows sit +0.07 to +0.20 above their mock. The overlap
  offset is the cleaner signal: mock +0.04–0.05 on every problem, flows +0.15–0.22, **narrow-2
  included** although its widths match the chains, so something the width table leaves out
  (phases most likely) is off.
- **Corner plots** for the six selected models on the closest loud pair of similar SNR
  (`paper/figures/corner-<band>-<n>.pdf`, draws in `paper/data/corners/`): narrow-2 = MCMC; narrow-4
  puts a few % of f0 draws on the neighbour 4.6 bins away; narrow-8 wide in everything with long
  amplitude tails; full-4 peaks on the MCMC with long tails; full-8 multimodal where the MCMC has one
  mode; full-16 (SNR 68 and 51, 0.02 bins apart) offset by ~1 bin in f0 and 0.3 rad in the sky.

- **Sampler steps:** widths at SNR >= 15 relative to 128 steps (uniform midpoint, 128 draws): narrow-4
  0.86 / 0.94 / 0.98 / 1.00 at 8 / 16 / 32 / 64 steps; full-8 0.97 already at 8. The eval's 32 steps
  are within 2% of converged; widths only grow with more steps.
- **Robustness (narrow-4, 64 windows, 128 draws):** robust to the prior's edges (f0 past the window,
  fdot or SNR above the prior, noise halved: 56–79% of SNR >= 7 sources within one width of the truth,
  72% in distribution); **confident and wrong with the noise doubled** — slots claim SNR 8.5 instead
  of 4.8, faint f0 widths shrink to 0.61× while their offsets double (0.8 → 1.6 widths), 38% within
  one width (60% in distribution). A fifth source in a four-slot model: 76% of faint sources still
  matched. Empty window: slots claim SNR 2.7. full-8 (OOMed the laptop GPU, ran on TREX 17673632):
  same failure on its loud sources — noise ×2 widens them only 1.28× while offsets go 0.6 → 1.3 widths,
  43% within one width (70% in distribution), slots claim SNR 5.5 instead of 2.1; other cases 57–72%.
- **Float32 backbone at eval:** f0 width over the MCMC at SNR 15–40 1.23 → 1.22 (narrow-4), 2.58 →
  2.14 (full-8); loud full-8 below 10 mHz 3.48 → 3.38; coverage within 3 points. bf16 is not what
  keeps loud sources wide.

- **Long chains, landed 2026-11-14, read 2026-11-16.** The traces list sources by f0 in every draw, so
  once a faint source wanders past a loud one their ranks swap and the loud rank's width blows up
  (`compare_prior_start.py` read 70-316x the reference for loud sources; the narrow-4 drift test failed
  on every window, worst f0 of the SNR-16 source of window 112 at 64x the tolerance). Read label-free
  (`reports/mcmc/scripts/label_free_widths.py`: a loud source takes the slot nearest its true f0;
  "found" = share of draws with any slot within 0.5 bins of the true f0):
  - **SNR >= 7: the references stand.** narrow-8 prior-start f0 width 1.11x the reference (flow 0.83x
    the chain), narrow-4 long chains 0.93x and stable to 0.97-0.99 between the first and last quarter;
    every loud source found in every draw. F68's "the 7-15 and 15-40 rows are suspect too" is
    withdrawn.
  - **SNR < 3: the prior is the reference.** narrow-8 prior-start 0.32x the prior's f0 width (flow 0.52,
    reference 0.07), found at chance (0.04). narrow-4 Fisher-start chains spread 0.79 -> 1.12x the prior
    (reference 0.73, flow 1.15): even the four-source reference is ~1.5x narrow there.
  - **SNR 3-7: bracketed, not settled.** narrow-8 found share per window: reference (Fisher start)
    0.90-1.00, prior start 0.12-0.25 -> 0.17-0.29 over the chain (still rising), flow 0.84 / 0.10 /
    0.98 / 0.38 (inside the bracket on three windows). narrow-4 from the Fisher start falls 0.80 /
    0.92 / 0.13 / 0.62 -> 0.75 / 0.84 / 0.03 / 0.53 (flow 0.43 / 0.27 / 0.01 / 0.45). Finding vs not
    finding a mid-SNR source is the sampler's slowest direction; neither start reaches the balance in
    400k iterations, so rerunning the references with prior starts would not fix them.
  - **full-16 prior-start chains: OUT_OF_MEMORY** at 32 GB after 1h40 (all 8 tasks, 17673832); the post
    job 17673842 never ran and was cancelled. Not rerun (narrow-8 answers the question).
  - Paper (2026-11-16): app:mcmc-convergence rebuilt on the narrow-4 chains
    (`reports/mcmc/data/convergence-261113/`, figure regenerated, McmcConv*/McmcLabelFree*/McmcPriorStart*
    merged into numbers/mcmc.tex; the rest of that file is from the previous generation, since the
    64-window joint chains it came from are not in this checkout); results.tex faint paragraph with the
    bracket; the tab:results-widths caption now marks narrow-8/full-16 below SNR 7 as measured against a
    stuck reference.
- **Rank-ECDF figure (2026-11-16)** `paper/figures/rank-ecdf.pdf` (`import_ranks.py` -> `data/runs/ranks.npz`,
  `fig_rank-ecdf.py`): log-likelihood and overlap rank ECDF - uniform for the six selected models and
  their Gaussian mocks, with the 95% simultaneous γ band. Every curve leaves the band, the mocks' too;
  the flows' log-lik curves run below the mocks', their overlap curves well below (offsets as in the
  table: overlap flow +0.15-0.22, mock +0.04-0.05). Closes the paper's last "Not Yet Run" item.
  Conclusions (main text, stale fact): the mock fails *all* tests at 1000 injections, not "some".

**Runs.**
- Laptop GPU (`.venv313`, the CUDA one; `.venv` is CPU-only), `outputs/runs/261103-analysis/`:
  `261103-frame-vs-flow.py` on the six best checkpoints (downloaded from TREX, md5 ok), bf16 and,
  for narrow-4 and full-8, float32 at autotune level 0; the robustness probe and the sampler sweep
  on narrow-4 and full-8 (`gpu-queue.sh`). Float32 runs ~10× slower than bf16 here.
- TREX (`<scratch>/canna-oneshot-outputs/analysis-261113/`, ~/canna-oneshot at
  cc93fb2): Gaussian mock per problem on GPU (job 17657536; on CPU nodes the JIT ran out of memory
  maps, vm.max_map_count 65530, job 17649213); its validation statistics (17661734);
  long chains of narrow-4 at windows 16/48/80/112 × 4 seeds, 400k iterations, for the
  reference-convergence study; prior-started chains of narrow-8 (windows 66/81/51/57) and full-16
  (70/77/31/42), 2 seeds each, 400k and 200k iterations. The first submission (17650285, 17657747,
  17657803) segfaulted after ~2 h: every 2000-iteration jexplore chunk compiles anew, and after ~50 the
  CPU JIT runs out of memory maps (vm.max_map_count 65530 on TREX). Resubmitted with --chunk 20000
  (17673807 long-n4, 17673808 prior-n8, 17673832 prior-f16); post jobs 17673833/41 ran the
  convergence analysis and `compare_prior_start.py`; the label-free reading is 17903997/98
  (`labelfree-{prior-narrow-8,long-narrow-4}.json`, local copies in reports/mcmc/data/convergence-261113/).

- **TREX cleanup (2026-11-16):** `outputs/trex-fetch-261116.sh` fetches everything under
  canna-oneshot-outputs (analysis folders, chain traces, all 30 source-count runs' best/ + checkpoints/ +
  snapshots/, ~112 GB at ~1.6 MB/s, ~18 h) into the same paths under outputs/; md5 list from TREX job
  17909894 (`md5-all.txt`); delete on TREX only after `md5sum -c` passes. src/ (in git) and best6.tar
  (the six best/ already local, md5 checked 2026-11-13) are deleted without fetching.

**Next.**
- ~~Read the prior-started chains~~ done 2026-11-16 (above): no prior-started reference rerun; decide how
  SNR < 7 is reported for narrow-8 and full-16 (prior below 3, bracket at 3-7).
- Which parameter the overlap offset sees on narrow-2: rerank with phases (then sky) set to the
  truth.
- ~~The paper's MCMC appendix convergence~~ done 2026-11-16; the rest of app:mcmc (Fisher vs MCMC,
  posterior figure) still from the old chains; the raw joint chains behind the references are not on TREX.
