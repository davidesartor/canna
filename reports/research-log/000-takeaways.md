# canna.lisa takeaways (current state, 2026-10-31, F63)

One `0<nn>-<slug>.md` per finding F<nn>, outcome first. The literature lives in
[../literature/](../literature/): [GW and cosmology
inference](../literature/gw-cosmology-inference/000-takeaways.md), [optimizers and
schedules](../literature/optimizers-schedules/000-takeaways.md), [diffusion and flow
matching](../literature/diffusion-flow-matching/000-takeaways.md) and the others; the findings
keep only its consequences for our code.

- **full-B diverged on a8e3477** ([F63](063-narrow-b-full-b-full-b-diverged.md)): from epoch
  ~45 the frame loss climbed and the gradient norm grew from ~20 to 540, ending in a NaN at
  epoch 93 of 240. Catalog flow was still improving to epoch 77 (−1.57). Only the latest checkpoint
  is kept, and it is mid-runaway. narrow-B trains on, flat since epoch ~100.
- **The oneshot code's first runs** ([F59](059-oneshot-first-runs-narrow-s-full-b.md)):
  narrow-S is the narrow-band model (779 of 840 found, every parameter within 0.8–3.3× of the
  MCMC at SNR 15–100 in the medians, the exceptions close pairs, at 2.2× fewer parameters than
  XS-stft-h); full-B's f₀ is stuck at
  0.4–0.6 bins at every SNR, 10× B-stft-h's, and φ₀ follows (36–70×). The floor is a fixed
  fraction (~1.5 × 10⁻⁴) of the window.
- **Half of full-B's f₀ floor is bf16 rounding** ([F61](061-fp32-backbone-halves-f0-but-overconfident.md)):
  a float32 backbone at eval halves f₀'s width (only f₀'s) but makes it overconfident (in68
  0.5–0.6); the rest is a learned f₀ error. The fix is in training: x̂₁ = x_t + (1 − s)·head(h).
- **full-B's f₀ floor is not the sampler** ([F60](060-sampler-grid-not-the-f0-floor.md)): the eval's 32
  uniform steps are converged, and steps late in the path widen f₀. Suspect: x̂₁ read out
  absolutely from the bf16 backbone, one floor of ~3 × 10⁻⁴ in x_f on both bands.
- **B-long is the B model:** 221 of 240 sources found, calibrated, within 1.0–1.6× of the MCMC
  in A, sky, ψ and ι at SNR 17–78 ([F48](048-b-long-best-b-phi0-broke.md)).
- **φ₀ is learned, settled:** the helicity coordinates (φ₊, φ₋) did it, not the template. B-stft-h
  pins φ₀ ± 2ψ at 1.8–2.3× the MCMC at SNR 17–78 (B-stft 29–101×), f₀ 1.03–1.25× B-stft's,
  215 found ([F56](056-phi0-learned-without-template.md); [F54](054-xs-b-stft-ht-phases-learned.md)
  with the template paid 1.27–1.45× in f₀; [F52](052-phi0-velocity-ignores-data.md)'s mechanism
  corrected).
- **Loud sources' sky** is still ~20× the MCMC on B's loudest source
  ([F51](051-xs-stft-b-stft-results.md), F54, F56), and q0.10 s2's λ sits 2.9 MCMC widths
  off at SNR 61. A per-source Doppler-demodulated band (B-stft-hd) halves the typical λ offset
  but leaves those large ones ([F57](057-b-stft-hd-doppler-bands.md),
  [F58](058-doppler-band-trims-typical-sky-offset.md); not ported, `origin/fml` 6e2f01e).
- **Next:** the loudest sources (φ₋ ~100×, sky ~20×) and the per-source sky offsets. F58's
  suspects: the antenna-pattern part of the response, amortising one network over 0.1–12 mHz,
  or network size and training length.

## The recipe now (F32 onward)

- **Data:** the whitened rfft window of A, E, T ([F32](032-muon-never-trained-backbone.md)),
  cut into 16-bin tokens (8 on the narrow band, 256 on the full band). Each token is its band's
  16 time samples (a critically sampled STFT), encoded as asinh|z| and z/√(1 + |z|²)
  ([F50](050-stft-tokens-launch.md)).
- **Coordinates:** per source f₀, M_c, sky and the helicity amplitudes z± (phases φ₊, φ₋), no
  template ([F56](056-phi0-learned-without-template.md)); φ₀ is referenced at mid-observation
  (`phase_reference="middle"`).
- **Network:** PLE of the data and coordinates in fp32 (sinusoid MLPs for t, f and f₀'s bin),
  bf16 MMDiT; RoPE in frequency, sources at their f₀ and tokens at their centre bins
  ([F42](042-b-floor-not-bf16-relaunch.md)). Sources ordered loudest first.
- **Optimiser:** our schedule-free wrapper around Muon with NorMuon rows on the matrices
  (backbone, the Fourier-feature and conditioning MLPs, the modulation, the head's hidden layer),
  AdamW for the PLE embeddings, slots, qk gains, biases and the head's output layer; decay 0.05
  on z for the Muon matrices only, inner momentum, t·lr² averaging, β 0.9 → 0.965, clip 1, no 1/L1 weighting
  ([F39](039-sf-normuon-schedulefree-plus-fixes.md), [F43](043-fixed-recipe-and-rope.md), F50).
  Eval uses the average x.
- **Reference:** a narrowband MCMC per corner-page source, within 1–4% of the joint MCMC, run
  for every run ([F45](045-narrowband-mcmc-every-run.md)).

## Best models (flow / MCMC width)

| | model | headline |
|---|---|---|
| B | B-long, 600k | loud f₀ 0.031 bins; mid-SNR f₀ 1.6–3.6×; loudest sky 30, 44× (F48) |
| B | B-stft, 200k | loud A, sky, ι 0.6–0.75× B-rope per source; loudest sky 18, 21× (F51) |
| B | B-stft-ht, 200k | φ₀ ± 2ψ 2.1–3.4× at mid SNR; f₀ 1.27–1.45× B-stft's; 213 found (F54) |
| B | B-stft-h, 200k | φ₀ ± 2ψ 1.8–2.3× at mid SNR; f₀ 1.03–1.25× B-stft's; 215 found (F56) |
| XS | R1, WDM era | loud f₀ 0.0067 bins (4× ideal), 6.3× MCMC; 739 of 840 found ([F31](031-r1-final.md)) |
| XS | XS-rope | loud f₀ 0.0106 bins, 11.4× MCMC; 687 found (F43) |
| narrow | narrow-S (oneshot), 650k | all parameters 0.8–3.3× MCMC at SNR 15–100; loud f₀ 0.021 bins, 31×; 779 of 840 found (F59) |
| full | full-B (oneshot), 240k | sky, A, ψ, ι ≤ 2.7× below SNR 300; f₀ 0.41–0.57 bins (38–164× MCMC), φ₀ 36–70× (F59) |

No model is biased: the MCMC median is inside the flow's 95% in 123–127 of 128 pairs on XS
([F27](027-flow-vs-mcmc-xs-windows.md), F43) and 48 of 48 for B-rope at mid SNR (F45).

## Still broken

- **full-B's f₀ (oneshot):** 0.41–0.57 bins at every SNR and every f₀, ~1.5 × 10⁻⁴ of its
  3072-bin window, against B-stft-h's 0.033–0.13; φ₀ widens with it. Suspects: learning rates
  10× and 3×, time_power 3 → 1, the new data stream or RoPE axis (F59); not the sampler's grid,
  and first the bf16 readout of an absolute x̂₁ (F60).
- **TARP was void in F59's scorecards:** its reference was the truth (`fold_in(key, 1)` equals
  `split(key, 4)[1]`), so every cell read −0.50; fixed by `key_tarp` (b6e0218), rescore to use it.
- **φ₀ on the loudest source:** φ₋ is 101× the MCMC on B-stft-h (72× with the template, 52×
  with the Doppler band) and 51× on XS-stft-h, the same floor as its sky (F54, F56, F58). Every
  run F39–F51 ignored the data's phase (F52); the helicity coordinates fixed that at mid SNR
  (F56).
- **XS q0.10 s2** (SNR 34): sky 1.8–2.1× → 16–48× the MCMC in both helicity runs, so the
  coordinates, not the template (F54, F56).
- **Loud sky:** wide (~20× on B's loudest), calibrated on average
  ([F49](049-b-helicity-sky-calibrated.md)). F49 blamed the bf16 data tokens; the finer STFT
  encoding moved it, so not a pure cast floor (F51). Single sources sit off: q0.10 s2's λ by
  2.9 MCMC widths (2.0 of the flow's) with or without the Doppler band; the loudest source's
  sin β by 12.4 (21.9 with it), inside the flow's own width (F58).
- **Loud f₀ on B:** ~0.031–0.035 bins, 12–72× Fisher ([F44](044-b-rope-breaks-3-bin-floor.md),
  F48).
- **XS below SNR 100:** 1.25–1.35× XS-fd's width, ~45 fewer found (F43); XS-stft 1.13–1.23×
  worse again (F51). Suspects: 4 tokens, SNR order.
- **Masking:** detection at SNR 15–40 falls from 98–100% beyond 5 bins to 63% within 1 bin of a
  neighbour ([F28](028-mcmc-two-more-windows.md), [F29](029-crowding-detection.md)).

## Next, most urgent first

- **full-B's f₀ floor (F59–F61):** one full-B run predicting x̂₁ = x_t + (1 − s)·head(h), the
  bf16 backbone kept; with B-stft-h's time_power 3 and learning rates folded in or as the next
  run. A per-f₀ window later.
- **φ₀, analytic extrinsics (training-free):** for each flow draw of (f₀, ḟ, sky), the four
  F-statistic templates make the likelihood Gaussian in four amplitudes; sampling them,
  importance-weighted by the (A, ι, ψ, φ₀) prior, gives φ±, the ridge and the amplitudes
  exactly. Still the route for the loudest source's φ₋ (101×, F56) (F49, F52).
- **φ₀, other options:** matched-filter phasors ⟨y, h_k(x_t)⟩ as inputs (F52), a phase
  reference without the template's f₀ cost (the template helped XS's loudest sources, F56); XS
  with no decay on z, the last optimiser difference from XS-fd (F48, F51; watch for F36's
  runaway).
- **Optimiser, prepared, not submitted: E1–E3**, 1.2M steps each against XS-stft: XS-sfnormuon
  (AdamW lr 1e-2 too, plain fp32 momentum), XS-lowdecay (λ 0.005), XS-cooldown (lr → 0 over the last 20%;
  a diagnostic, score y too). Then one change per run: β₂ 0.999, Polar Express,
  adaptive_grad_clip, the head on AdamW, two EMAs of y, PSGD-Kron or AdEMAMix. Free: LAWA over
  the last k y checkpoints.
- **Loud sky and f₀:** float32 eval of the loudest source (only a positive result is
  informative); a Gauss–Newton step δ = (G + I)⁻¹ Jᵀ(y − h(x_t)) as an input (+9% per step, the
  user's call, F49).
- **Per-source sky offsets:** not a missing Doppler reference (F58); left are the antenna-pattern
  part of the response, amortisation over 0.1–12 mHz, or network size and training length.
- **Later:** masking (more close pairs in training, or a residual pass); variable source count,
  stitching windows, gaps, LDC data.

## Lessons

- **The loss does not see the loud floor** ([F6](006-loud-floor-invisible-to-loss.md)): RoPE
  narrowed B 70× for 0.3% of loss (F44). Judge by scorecards and the MCMC.
- **The floor was never numerics or capacity:** ODE converged
  ([F16](016-integrator-converged-quota.md)), fp32 eval changes nothing
  ([F21](021-t3-fp32-eval-floor.md), F42), width neither ([F19](019-e3-capacity-floor.md),
  [F24](024-c2-cooled-512-net.md)). It moved with the lr cooldown
  ([F23](023-c1-lr-cooldown.md)), token positions (R1, RoPE) and training length (F48). B's
  3-bin floor was sources finding their tokens ([F26](026-b1-3-bin-floor.md), F44).
- **XLA GPU fp32 is wrong at the default autotune level;** A100 fp32 jobs use level 0
  ([F20](020-xla-gpu-float32-autotune-fault.md), [F22](022-autotune-level-3-fp32-a100.md),
  F23).
- **Check optimiser routing:** until F32, 89% of parameters (the backbone) trained on Adam, not
  Muon; the Muon share is now asserted at startup.
- **Schedule-free Muon needs decay on z:** without it z ran away (loss 2.7×10²¹ at z) and only
  y stayed good ([F36](036-schedule-free-z-runaway.md),
  [F37](037-schedule-free-failure-literature.md)). Decay coupled to the 1/L1 weighting shrank
  the weights 3.6× and pinned the clip ([F40](040-xs-sfplus-regressed.md)).
- **One run with six changes hid a regression's cause** (F40); the velocity probe on a
  checkpoint is the cheapest decisive diagnostic (F47, F48, F51, F52).
- **x lives in `opt_state`:** a clean-up that drops it loses the evaluated model (F43).
- **A symmetric chart averages the target away:** the four likelihood-equivalent (ψ, φ₀) copies
  cancel the phase in the flow-matching target; the helicity phases (φ₊, φ₋) have no such copies
  and were learned with no phase reference (F49's argument, confirmed by F56).

## The paper

- **Not new, cite:** an amortised GB posterior with the LISA response (Delmond+26,
  [020](../literature/gw-cosmology-inference/020-delmond26-neural-posterior-estimation-galactic.md)
  (C)); flow matching and WDM input in the LISA band (Mao+25,
  [113](../literature/gw-cosmology-inference/113-mao25-robust-scalable-simulation-based.md)
  (C)); permutation-invariant multi-source NPE (SlotFlow,
  [021](../literature/gw-cosmology-inference/021-houba25-slotflow.md) (C)); permutation
  alignment in flow matching
  ([Klein+23](../literature/diffusion-flow-matching/003-klein23-equivariant-flow-matching.md)
  (A)). Tiers as in [the GW and cosmology
  takeaways](../literature/gw-cosmology-inference/000-takeaways.md#how-much-to-trust-each-source);
  a C paper still sets priority, whatever its results' weight.
- **New, as far as the 2026-10-09 survey found:** a joint posterior of 4 overlapping GBs per
  window (32 parameters, full TDI 1.5 A/E/T, 2 yr; Delmond's largest is 2 sources); survey-like priors (f₀ log-uniform over the band,
  SNR 7–1000); one network for 0.1–12 mHz (B's loud f₀ 0.031 bins, F44, F48); on-the-fly JAX
  simulation; the loud-source floor's mechanism and fix (F6, F15, F23, F19, F24), possibly the
  most general result.
- **Required:** the MCMC comparison (F45); cut the XS scorecards by separation from the nearest
  source, to set them beside Delmond's 31.6- and 12.6-bin pairs.
- **Limitations:** 4 sources, no model selection; stationary Gaussian noise, no foreground, no
  gaps; φ₀ within 1.8–2.3× of the MCMC at mid SNR, 101× on the loudest source (F56).
- **Timing:** Korsakova's group, the SlotFlow group (Giarda, CERN AI4GW, May 2026) and Delmond
  et al. (trans-dimensional next) are all close, so "first joint multi-GB amortised posterior
  with the full LISA response" has a short shelf life.
