# Cosmology and GW inference takeaways for canna.lisa

What we take from the papers in this folder for turning the flow into a posterior we can trust:
which parameters to keep out of the network, how loud sources reach their true width, how to
correct and validate the draws against the exact likelihood, and how to handle several sources
in one window. Network design is in [../network-architecture/](../network-architecture/), the objective and
training recipe in [../diffusion-flow-matching/](../diffusion-flow-matching/), the optimizer in
[../optimizers-schedules/000-takeaways.md](../optimizers-schedules/000-takeaways.md).

The setting all of this is read against: an amortised flow-matching posterior over 4 galactic
binaries per frequency window, ordered loudest first, 8 flat flow coordinates each (f0, Mc, sky
lon and lat, and the two complex helicity amplitudes z± = |z±| e^{i(φ0 ± 2ψ)}), with a
differentiable jaxgb likelihood and narrowband MCMC reference posteriors to compare against.
Three facts drove the choices:

- **The φ± phases came out as the prior** in the earlier phase-circle chart. Their velocity did
  not depend on the data, so the phase draws were tens to thousands of times wider than the
  MCMC. The helicity coordinates fixed this ([F56](../../research-log/056-phi0-learned-without-template.md)):
  1.8–2.3× the MCMC at SNR 17–78, but still about 100× on the loudest source.
- **Loud sources are too wide.** For the loudest sources the sky and f0 widths are tens of
  times the MCMC's, while mid-SNR parameters sit within a few times of it.
- **The likelihood is cheap and exact.** jaxgb gives log L and its gradient on GPU in
  milliseconds, so anything the network cannot do precisely can be handed to the likelihood.

## How much to trust each source

Every paper link carries its evidence tier; the paper file says why under its citation.

- **A, established:** ≥100 citations, or peer-reviewed with ≥30 citations a year.
- **B, credible:** peer-reviewed at a recognised venue, or a preprint with ≥20 citations. One
  result, not yet independently confirmed.
- **C, unvetted:** an unreviewed preprint or workshop paper with <20 citations. A hypothesis.

A C claim is written as "X reports ..." and never alone justifies a recipe choice; the choices
that still rest only on C papers are listed at the end
([Choices resting only on C evidence](#choices-resting-only-on-c-evidence)) with the cheapest
test of each. Counts are Semantic Scholar on 2026-10-19. Recent papers have had little time to
be cited, so C means unvetted, not wrong; GW and astro journals also cite less than ML venues,
so a sound PRD or MNRAS paper is often B with few citations.

## The closest work: amortised posteriors for Galactic binaries

From the 2026-10-09 related-work survey: the INSPIRE-HEP citation graph of Korsakova+24,
Katz+25, Strub+24, Deng+25, Srinivasan+25, Alvey+23 and Dax+23, INSPIRE and arXiv searches
(LISA/Taiji/TianQin/GBs × neural/flow/diffusion/SBI), and the 2026 talk listings (AI4GW at
CERN, Journées LISA France, 3IA Nice). Its bib entries are in `paper/references.bib`.

| work | what it infers | data and model | network, objective | label switching | priors, SNR | validated against |
|---|---|---|---|---|---|---|
| **Delmond+26** ([020](020-delmond26-neural-posterior-estimation-galactic.md) (C), `delmond2026neural`) | 1 GB (8 parameters); 2 GBs (16) as a proof of concept; fixed, known source count | GBGPU (FastGB), 1 yr, A and E in the frequency domain (Re, Im), 64–512 bins per channel, no embedding net, stationary noise, no confusion noise | NSF, ResNet conditioner, 1.2–3.9×10⁸ parameters; NLL on 7×10⁶ precomputed waveforms, the other parameters drawn on the fly; ~1 week per run on one A100 | order f₀₁ ≤ f₀₂ (unordered runs gave one mode per source) | uniform; f₀ in f_c ± 2 μHz, one variant over 0.5 mHz with f_c as input; SNR 10–100 | ptemcee MCMC (JS divergence); P-P on 1000 injections |
| **Mao+25** ([113](113-mao25-robust-scalable-simulation-based.md) (C), `mao2025robust`) | 1 GB-like source, 3 parameters (A, f, ḟ); angles fixed | fastlisaresponse, TDI A, 30 or 90 days, gaps (80% duty cycle) | FMPE (Gaussian base, OT path, residual MLP) with a 1D CNN on the time series or a 2D CNN on the WDM log-modulus; 10⁴–2×10⁴ training signals | — | log-uniform boxes of ±10⁻⁵ to ±10⁻³ around the truth; SNR ~40 | MAF with the same summary net; no MCMC |
| **SlotFlow, Houba+25** ([021](021-houba25-slotflow.md) (C), `houba2025slotflow`) | K ≤ 10 sinusoids (A, φ, f), **unknown K**: a classifier over K, then K slots | toy sinusoids in white noise; no LISA response | a shared conditional NSF per slot, **factorised across slots** given a global context; 8×10⁶ training samples | Hungarian matching on the per-slot NLL | uniform | Eryn RJMCMC at K = 3: amplitude and phase agree, **frequency 2–3× broader** |
| **Korsakova+24** ([114](114-korsakova24-neural-density-estimation-galactic.md) (B), `korsakova2024neural`) | density estimation, not amortised over data: a flow for the Galaxy's spatial distribution and one per verification binary, fit to its MCMC posterior; proposed as priors, proposals, a compact catalogue | the Sangria population; MCMC posteriors of 4 verification binaries | NSF + RealNVP blocks, ResNet conditioners | per source; a joint flow named as needed, not built | — | quality of the fits |
| **Sasli+26** ([061](061-sasli26-learned-proposals-trans-dimensional.md) (C), `sasli2026learned`) | learned birth proposals in RJMCMC: useless while the fit is assembled, optimal at equilibrium | wavelet and pulse models of unknown order (BBH, GW150914, EEG, a Sangria MBHB); no GB benchmark | adaptive NSF (pocoMC/zuko), retrained on the cold chain | — | — | ten-seed benchmark with a stopping rule |

## 1. Take the phases out of the flow

- **The failure is common in GW NPE: 6 of the 8 papers below are A or B.** DINGO "often has
  difficulty resolving the phase parameter" and learns its prior
  ([001](001-dax22-neural-importance-sampling-rapid.md) (A)). The same happens for MBHBs (φ_c
  outside the 90% CI, [063](063-du23-advancing-space-based-gravitational.md) (B); ψ and φ_c
  never determined by SNL, [116](116-vilchez24-efficient-massive-black-hole.md) (B); Sun+26
  report φ_c ±2.1 rad and ψ ±1.0 rad, i.e. the prior,
  [117](117-sun26-robust-parameter-inference-taiji.md) (C)), EMRIs (both initial phases exactly
  the uniform prior, [031](031-cole26-sequential-simulation-based-inference.md) (B)), ET (the
  p-p deviation is "most notably" in φ_c,
  [015](015-santoliquido25-fast-accurate-parameter-estimation.md) (B)), and LISA (phase is the
  worst JS before IS for MBHBs,
  [022](022-spadaro26-accurate-efficient-simulation-based.md) (B);
  SlotFlow reports phase widths ~2× RJMCMC on toy sinusoids, [021](021-houba25-slotflow.md)
  (C)). DINGO, the ET network and Dingo-T1 sample φ_c outside the network
  ([001](001-dax22-neural-importance-sampling-rapid.md) (A),
  [015](015-santoliquido25-fast-accurate-parameter-estimation.md) (B),
  [024](024-kofler25-flexible-gravitational-wave-parameter.md) (B)); Zhang+26 marginalise the
  phases out altogether ([066](066-zhang26-pre-localization-massive-black.md) (C)).
- **Why it fails (our reading).** A phase identifiable only against a coherent reference
  behaves like a parity target: gradients have exponentially small signal-to-noise and the
  network sits at the mean ([060](060-shalevshwartz17-failures-gradient-based-deep.md) (A)
  shows this for parity; Alvey+26 report that a linear model learns nothing from
  prior-symmetric phase information in a PTA toy, [043](043-alvey26-opening-black-box.md) (C)).
  On that reading more steps do not fix it.
- **For GBs the phase block is linear and exact.** The signal is linear in four amplitudes, so
  given (f0, Mc, sky) the likelihood is Gaussian in them: the F-statistic (JKS98,
  [arXiv:gr-qc/9804014](https://arxiv.org/abs/gr-qc/9804014); for LISA TDI, Krolak+04,
  [arXiv:gr-qc/0401108](https://arxiv.org/abs/gr-qc/0401108); for GBs,
  [009](009-prix07-statistic-search-white-dwarf.md) (B),
  [038](038-littenberg11-detection-pipeline-galactic-binaries.md) (B)). Whelan+14's
  circular-polarisation coordinates make the two circles separate: the left and right
  amplitudes carry φ0 ± 2ψ, i.e. our φ±, with weights (1 ± cos ι)²
  ([010](010-whelan14-new-coordinates-amplitude-parameter.md) (B)). Dhurandhar+17's
  B-statistic marginalises with the physical prior instead of maximising, and gives von Mises
  conditionals for the phases
  ([011](011-dhurandhar17-marginalizing-likelihood-function-modeled.md)
  (C)); the von Mises form also follows from 009 and 010 by the algebra below. Analytic
  handling of the ψ–φ0 subspace is standard practice for CW pulsars
  ([012](012-ashok24-bayesian-f-statistic-based.md) (B)) and real-time CBC
  ([014](014-roulet24-fast-marginalization-algorithm-optimizing.md) (B),
  [013](013-islam22-factorized-parameter-estimation-real.md) (C)).
- **The conditional per circle is a von Mises.** With the rest fixed, each source is h = α_L
  G_L + α_R G_R, with α_L = (A/2)(1+s)² e^{−iφ₋}, α_R = (A/2)(1−s)² e^{−iφ₊}, G_L = H₁ − iH₂,
  G_R = H₁ + iH₂ (H₁, H₂ the unit-amplitude edge-on templates at ψ = 0 and π/4, functions of
  f0, Mc and sky only). The log-likelihood is linear in e^{−iφ}, so φ₋ | rest ~ VonMises(arg z,
  2|z|) with z = (A/2)(1+s)² Σ G_L conj(r)/P, and the same for φ₊. The prior on the (φ₊, φ₋)
  torus is uniform, so no weights are needed. κ ≈ ρ², i.e. 1/ρ rad. Coupling between the two
  circles and between overlapping sources goes through the residual r and a few Gibbs sweeps.
- **DINGO's synthetic phase is the template.** The flow models everything else; φ_c is drawn
  from the exact likelihood on a grid, and that draw is part of the IS proposal density
  ([001](001-dax22-neural-importance-sampling-rapid.md) (A)). Heurtel-Depeiges's Gibbs
  diffusion is the same split: a learned sampler for one block, an exact one for the other
  ([086](086-heurteldepeiges24-listening-noise.md) (B)).
- **But do not ask the network for the JKS amplitudes.** Delmond+26 report that predicting the
  four 𝒜^µ directly, whose posteriors are near-Gaussian, gave broader posteriors and worse sky
  than (A, ι, φ0, ψ) ([020](020-delmond26-neural-posterior-estimation-galactic.md) (C), one
  ablation). The A/B case for the split above is that the amplitude block is exact given the
  rest (009, 001), so it should be sampled, not regressed.
- **What we take.**
  - Draw φ± from their von Mises conditionals given the flow's other coordinates, 2–3 Gibbs
    sweeps over (source, L/R), Best–Fisher rejection or a wrapped normal when κ > 10³. Test
    first that Σ α G reproduces the clean signal to ~1e-6.
  - Then train without φ±: 7 coordinates per source, the two circles gone from the manifold.
    The network learns the phase-marginal of the rest, which is the correct target.
  - Optionally, at ρ ≳ 30, sample the full complex pair (α_L, α_R) from its Gaussian and
    importance-weight by the prior's amplitude-space density (∝ 1/(A⁴(1−s²)³)).
  - The conditional is exact only given the flow's f0 and sky: a Δf0 of one bin turns the
    reference by about π. Phase accuracy is bounded by f0 accuracy (§2).

## 2. Loud sources: references, local priors, truncation

- **Every high-SNR GW pipeline heterodynes against a reference.** DINGO-BNS heterodynes by the
  leading PN phase at a proxy chirp mass, trains on local priors (M̃ ± 0.005 M⊙ around a
  hyperprior-drawn centre), conditions on M̃ and learns the offset; it still notes efficiency
  dropping at SNR ~10³ ([003](003-dax24-real-time-gravitational-wave.md) (B)). labrador finds a
  per-event reference by likelihood maximisation in ~1 s, relative-bins and SVD-compresses the
  data, folds up to 16 discrete symmetry sectors and restores them with a classifier; its IS
  efficiency is flat in SNR ([005](005-roulet26-domain-optimized-machine-learning.md) (B)).
  Hu+24 uses a grid of references instead of an optimiser
  ([006](006-hu24-decoding-long-duration-gravitational.md) (B)). Relative binning makes the
  reference-relative data compact and smooth in δθ
  ([077](077-zackay18-relative-binning-fast-likelihood.md) (A)).
- **Sample density near the truth, or the objective?** Kolmus+24 report that NPE accuracy
  follows how often training draws land near the event, and that a short per-event fine-tune
  lifts IS efficiency from ~0% to 10–80% in minutes
  ([004](004-kolmus24-tuning-neural-posterior-estimation.md) (C)). Spadaro+26 see low ε where
  the source sits at a prior edge, which fits, but attribute the fall of ε with SNR to the
  mass-covering NPE objective
  ([022](022-spadaro26-accurate-efficient-simulation-based.md) (B)). Truncated proposals
  (TSNPE, [033](033-deistler22-truncated-proposals-scalable-hassle.md) (B); TMNRE,
  [099](099-miller21-truncated-marginal-neural-ratio.md) (B),
  [098](098-cole21-fast-credible-likelihood-free.md) (B), [039](039-bhardwaj23-peregrine.md)
  (B), [118](118-alvey23-what-do-when-things.md) (C)) are the density remedy without a
  reference, but truncation alone is not enough: even with 6 rounds, Cole+26's EMRI widths stay
  10–1000× too wide ([031](031-cole26-sequential-simulation-based-inference.md) (B)).
- **Narrow posteriors are hard for flows in general.** Dingo-T1's efficiency falls as more
  detectors narrow the posterior
  ([024](024-kofler25-flexible-gravitational-wave-parameter.md) (B)), ε falls with SNR for
  DINGO-BNS and LISA MBHBs ([003](003-dax24-real-time-gravitational-wave.md) (B),
  [022](022-spadaro26-accurate-efficient-simulation-based.md) (B)). Delmond+26 blame the
  prior-to-posterior width ratio (0.5 mHz vs ~10⁻³ µHz) for their broad f0
  ([020](020-delmond26-neural-posterior-estimation-galactic.md) (C)). Mao+25 sidestep it with
  priors far below one frequency bin ([113](113-mao25-robust-scalable-simulation-based.md)
  (C)).
- **Ultra-high SNR exposes per-cent model errors.** At ρ ≈ 1200 a 3% error in ρ² from per-SFT
  approximations shows up as amplitude bias ([012](012-ashok24-bayesian-f-statistic-based.md)
  (B)). At our loudest sources the reference likelihood must be exact too.
- **Fisher is not a safe stand-in for sky.** Fisher underestimates sky area 8–10³× and misses
  degenerate modes ([015](015-santoliquido25-fast-accurate-parameter-estimation.md) (B)); the
  MBHB sky has up to 8 modes ([116](116-vilchez24-efficient-massive-black-hole.md) (B),
  [066](066-zhang26-pre-localization-massive-black.md) (C)).
- **What we take.**
  - For loud sources, find a per-source reference (f0, Mc, sky) with a cheap likelihood fit
    started from the flow's median, heterodyne the window against it, and learn offsets on a
    local prior. This is the network-side fix; §3 is the eval-side one.
  - A per-injection truncated or fine-tuned round is the cheap diagnostic of whether the gap is
    sample density or capacity (or the objective, per 022).
  - Use Fisher only as a preconditioner within one mode, never as the width.

## 3. Correct with the exact likelihood

- **Importance sampling is the field's exactness layer.** Median IS efficiency: DINGO-IS ~10%
  ([001](001-dax22-neural-importance-sampling-rapid.md) (A)), DINGO-BNS 31–49%
  ([003](003-dax24-real-time-gravitational-wave.md) (B)), BilbyFlow 7%
  ([007](007-pinchbeck26-bilbyflow.md) (C)), labrador ~1%
  ([005](005-roulet26-domain-optimized-machine-learning.md) (B)), ET 10% with evidences
  matching Bilby ([015](015-santoliquido25-fast-accurate-parameter-estimation.md) (B)),
  exoplanet FMPE 10–19% ([027](027-gebhard24-flow-matching-atmospheric-retrieval.md) (B)). For
  LISA MBHBs, Spadaro+26 get 50% at SNR 87, 12% at 500 and **0.05% (N_eff = 10) at 1000**
  ([022](022-spadaro26-accurate-efficient-simulation-based.md) (B)): IS works until the flow is
  many widths too wide, exactly where we need it.
- **IS efficiency is also a misspecification flag.** Gebhard+24 see ε drop to 1.5e-5 when σ =
  0.5 data are analysed as σ = 0.1
  ([027](027-gebhard24-flow-matching-atmospheric-retrieval.md) (B)).
- **Flow-seeded MCMC when IS fails.** FluxMC reports that using the flow as the MCMC proposal
  stays unbiased where PTMCMC is confidently wrong in φ_c and ψ, 10–70× faster
  ([065](065-liang26-fluxmc.md) (C); its baseline is weak and its reference starts at the
  truth). Sasli+26 report flow births in RJMCMC converge faster but data-informed births win in
  crowded data ([061](061-sasli26-learned-proposals-trans-dimensional.md) (C)). Diffusion-HMC
  uses a learned model inside HMC ([085](085-mudur24-diffusion-hmc.md) (B)).
- **Or feed the simulator back into the flow.** Holzschuh–Thuerey report that fine-tuning an FM
  posterior with a control term from ∇ log L works where 10× more simulations without feedback
  give nothing, with strong-lens χ² matching NUTS at 10 s vs 564 s
  ([008](008-holzschuh24-flow-matching-posterior-inference.md) (C)). Training on the joint
  score helps sample efficiency, not the floor
  ([056](056-zeghal22-neural-posterior-estimation-differentiable.md) (B),
  [030](030-zeghal24-simulation-based-inference-benchmark.md) (B)).
- **Variational alternatives.** Likelihood-based VI with flows fits PTA posteriors in tens of
  minutes ([016](016-vallisneri24-rapid-parameter-estimation-pulsar.md) (B)).
- **What we take.**
  - Above SNR ~100, polish the flow's draws with a short Fisher-preconditioned MALA on log L +
    log prior (a few Newton steps on the median first, then K ≈ 20–50 steps, acceptance ~0.57).
    Each draw keeps its mode, so the flow sets the mode weights and the likelihood sets the
    widths. That is seconds per injection on one GPU. The only GW precedent for flow-seeded
    MCMC is C (065), so this is ours to test (see the end).
  - Implement IS with exact log q (the ODE divergence on the product manifold) as the
    diagnostic: ε vs SNR is the single number that says how far the flow is from exact.
  - Simulator feedback (∇ log L into a zero-initialised control head) is the training-side
    version, to try once the eval-side correction shows how much is left. It rests on one C
    paper (008).

## 4. Validate width, not just calibration

- **Calibration passes for prior-like posteriors.** SBC ranks and p-p plots are uniform for any
  posterior that is "the prior" on average
  ([018](018-talts18-validating-bayesian-inference-algorithms.md) (A)); local coverage is the
  sharper statement ([019](019-zhao21-diagnostics-conditional-density-models.md) (B)). TARP
  with prior-drawn references looks perfect for p̂ = p(θ) too; it needs data-dependent
  references ([028](028-lemos23-sampling-based-accuracy-testing.md) (B)). In GW NPE: Du+23 pass
  p-p while 10–40× wider than nested sampling
  ([063](063-du23-advancing-space-based-gravitational.md) (B)). Three C papers report the same:
  Sun+26 pass with φ_c at the prior and show that p-p admits shifted or rescaled posteriors,
  adding CRPS ([117](117-sun26-robust-parameter-inference-taiji.md) (C)); Zhang+26 call a p =
  0.003 failure calibrated ([066](066-zhang26-pre-localization-massive-black.md) (C)); Luo+26's
  prior-only baseline is as well-calibrated as their (undertrained) FMPE, whose 90% intervals
  overlap the MCMC's by ~1% ([040](040-luo26-artifact-conditioned-interval-diagnostics.md)
  (C)).
- **Over-confidence is the usual worry elsewhere, under-confidence ours.** Most SBI posteriors
  are overconfident at small budgets, and ensembles fix it
  ([054](054-hermans21-trust-crisis-simulation-based.md) (B)). Conservative regularisers,
  conformal and interval rescaling ([036](036-cabezas25-cp4sbi.md) (B),
  [105](105-masserano22-simulator-based-inference-waldo.md) (B),
  [055](055-patel23-variational-inference-coverage-guarantees.md) (B),
  [076](076-crisostomi23-neural-posterior-estimation-guaranteed.md) (B),
  [040](040-luo26-artifact-conditioned-interval-diagnostics.md) (C)) guarantee coverage, not
  sharpness. They are the wrong lever for over-dispersion.
- **Compare against the likelihood reference directly.** Per-parameter width ratios and JS
  divergence against a likelihood sampler are what the GW papers that report precision use
  ([001](001-dax22-neural-importance-sampling-rapid.md) (A): JSD ≤ 2×10⁻³ nat counts as
  indistinguishable; [022](022-spadaro26-accurate-efficient-simulation-based.md) (B); ET uses a
  bootstrap threshold of ~0.001 nat,
  [015](015-santoliquido25-fast-accurate-parameter-estimation.md) (B);
  [020](020-delmond26-neural-posterior-estimation-galactic.md) (C),
  [013](013-islam22-factorized-parameter-estimation-real.md) (C): JS < 0.1 nat). C2ST
  saturates (never below 0.6 even when marginals match,
  [030](030-zeghal24-simulation-based-inference-benchmark.md) (B); near chance at 72-D,
  [059](059-arruda25-diffusion-models-simulation-based.md) (B)); its local version finds where
  the flow fails without MCMC per injection ([035](035-linhart23-l-c2st.md) (B),
  [074](074-bansal25-conformal-c2st.md) (C)).
- **Check the joint with posterior predictive tests.** JADE compares each posterior draw with a
  fresh simulation at its own θ and reports χ²_red = 1.0024 against the expected 1.0000 ±
  0.0049 ([017](017-remy26-joint-inference-weak-lensing.md) (C)). No A/B paper here uses this
  test.
- **NPE can match MCMC at moderate SNR.** DINGO's flow alone is within a mean 1-D JSD of
  2.2×10⁻³ nat of LALInference on GW150914
  ([001](001-dax22-neural-importance-sampling-rapid.md) (A)), implicit and explicit inference
  agree for weak lensing ([030](030-zeghal24-simulation-based-inference-benchmark.md) (B)), and
  Delmond+26 report matching MCMC at SNR 11 including the φ0/ψ structure
  ([020](020-delmond26-neural-posterior-estimation-galactic.md) (C)). A large gap is a method
  defect, not inevitable.
- **What we take.**
  - The headline metric is the width ratio to MCMC per parameter, stratified by SNR, f0 and
    slot rank, plus JS against the reference.
  - Calibration (circular SBC ranks for φ± and ψ mod π/2, TARP on each source's joint with
    data-dependent references) is a necessary check, never evidence of precision.
  - Add IS efficiency (A/B-backed), CRPS (117, C) and the PPC χ²_red of draws against the data
    (017, C): all three penalise a posterior that is too wide.

## 5. Several sources in one window

- **Label switching: order by frequency.** Buscicchio+19 remove the N! modes with an ordering
  map and recover 6 overlapping LISA GBs
  ([023](023-buscicchio19-label-switching-problem-bayesian.md) (B)); GBMCMC labels by f0
  ([025](025-littenberg20-global-analysis-gravitational-wave.md) (B)); Delmond+26 order two
  sources by f0 ([020](020-delmond26-neural-posterior-estimation-galactic.md) (C)); SlotFlow's
  slots self-order by frequency ([021](021-houba25-slotflow.md) (C)). Mixture models face the
  same problem in amortised form ([112](112-kucharsky25-amortized-bayesian-mixture-models.md)
  (B)).
- **Overlap may be where flows break (C evidence only).** Delmond+26 report their two-source
  flow close to MCMC at Δf0 = 0.5 µHz and broad and sometimes biased at 0.2 µHz
  ([020](020-delmond26-neural-posterior-estimation-galactic.md) (C)); SlotFlow reports widths
  growing 2–6× from K = 1 to 10 ([021](021-houba25-slotflow.md) (C)). Overlapping GBs are
  genuinely correlated in the exact posterior
  ([023](023-buscicchio19-label-switching-problem-bayesian.md) (B)), so some widening is
  correct. Fewer than 1% of loud GB pairs overlap above 50% at 1 yr
  ([025](025-littenberg20-global-analysis-gravitational-wave.md) (B)), so overlap is rare.
- **Window edges need a buffer.** GBMCMC pads each window by one source bandwidth and subtracts
  loud neighbours ([025](025-littenberg20-global-analysis-gravitational-wave.md) (B)). Hu+25
  report that iterative subtraction converges but early iterations are biased
  ([068](068-hu25-hierarchical-subtraction-neural-density.md) (C)); Houba+25 that separators
  degrade the downstream posterior unless the inference is retrained on their output
  ([115](115-houba25-deep-source-separation-overlapping.md) (C)). Reviews of the overlap
  problem: [062](062-niu24-extracting-overlapping-gravitational-wave.md) (B).
- **A variable source count is solved elsewhere.** StarNet tiles crowded fields with a
  per-tile count ([037](037-liu21-variational-inference-deblending-crowded.md) (B)); Evidence
  Networks give amortised model selection ([075](075-jeffrey23-evidence-networks.md) (B));
  t-FMPE reports generating up to 20 components with a transformer FM and a count posterior
  ([034](034-meulen26-fast-inference-astronomical-time.md) (C)), and SlotFlow predicting K with
  99.85% accuracy ([021](021-houba25-slotflow.md) (C)).
- **What we take.**
  - Keep slots ordered (loudest first or by f0) and match by frequency at eval. Report
    two-source overlap cases separately.
  - Do not compose posteriors across windows by multiplying them: neighbouring windows are not
    conditionally independent given one source.
  - A count head (t-FMPE / SlotFlow style) is the path off the fixed 4 slots, when needed.

## 6. The data input and the summary

- **Train the summary jointly, on the raw high-resolution data.** A lossy compressor biases the
  sky ([116](116-vilchez24-efficient-massive-black-hole.md) (B)), and NPE-trained summaries
  match HMC where regression ones do not (029, below). Mao+25 report end-to-end training
  beating a frozen autoencoder and recovering the amplitude a two-stage model loses
  ([113](113-mao25-robust-scalable-simulation-based.md) (C)); Zhang+26 a frozen pretrained
  embedding collapsing the sky modes ([066](066-zhang26-pre-localization-massive-black.md)
  (C)); SlotFlow its hybrid time + FFT encoder beating either alone by large factors
  ([021](021-houba25-slotflow.md) (C)).
- **An MSE regression head is not a sufficient statistic.** VMIM (NPE-style) summaries match
  HMC; MSE loses about a fifth of the figure of merit
  ([029](029-lanzieri24-optimal-neural-summarisation-full.md) (B)). Score compression at a
  fiducial point is exactly a matched-filter projection
  ([046](046-alsing18-massive-optimal-data-compression.md) (A)).
- **Dingo-T1's tokenizer and masking.** A GLU frequency tokenizer over the same 16-bin tokens,
  no RoPE, with data-based token masking (~25%) that beats random masking 26.9% vs 15.7% IS
  efficiency ([024](024-kofler25-flexible-gravitational-wave-parameter.md) (B)).
- **Physics in the positions.** Exploiting known symmetries in preprocessing makes MBHB p-p
  pass ([064](064-liang24-rapid-parameter-estimation-merging.md) (B)); GNPE standardises a pose
  with blurred proxies ([002](002-dax21-group-equivariant-neural-posterior.md) (B)), though one
  global phase pose fixes only one source. Dandapat+26 report positional encodings that carry
  the analytic GW phase helping a PTA flow
  ([042](042-dandapat26-transformers-physics-informed-encodings.md) (C)), but by their own
  account the gain largely vanishes with 8× more training data.
- **Phase-only data carry information.** Grunthal+26 report that a phase-only likelihood
  detects the PTA background, and that a joint 240-D flow fails p-p where a factorised one
  passes ([044](044-grunthal26-not-just-phase.md) (C)), a reminder that a smaller conditional
  is easier to get right.
- **What we take.**
  - Keep training the data embedding end-to-end with the flow. Treat the x̂₁ head as an
    auxiliary, never as the summary.
  - Try data-based token masking as a cheap regulariser.
  - The matched-filter phasors at the current estimate are the natural extra input: the
    "reference" of §1 and §2 inside the network. Its direct support is C only (042, 043).

## 7. Flow matching for SBI: what the comparisons say

- **FM matches or beats NSF/NPE on the same data.** For MBHBs FMPE is far better than an NSF of
  twice the size ([064](064-liang24-rapid-parameter-estimation-merging.md) (B)); exoplanet FMPE
  trains ~3× faster with higher IS efficiency
  ([027](027-gebhard24-flow-matching-atmospheric-retrieval.md) (B)); Mao+25 report FM
  calibrating where MAF does not ([113](113-mao25-robust-scalable-simulation-based.md) (C)). On
  the sbibm tasks the gap is smaller: FM is competitive but slightly behind VP-EDM on the
  smallest tasks ([059](059-arruda25-diffusion-models-simulation-based.md) (B)), and comparable
  to NSF in Holzschuh+24 ([008](008-holzschuh24-flow-matching-posterior-inference.md) (C)).
- **Generic SBI recipe findings.** In the diffusion-SBI review: power-law t-sampling toward the
  data helps; OT couplings that ignore the condition hurt; EMA and annealed Langevin do not
  help; a DiT ties a residual MLP at 3.5× the cost
  ([059](059-arruda25-diffusion-models-simulation-based.md) (B)).
- **JADE's lessons for a joint posterior (C, unablated).** JADE uses x-prediction with v = (x̂₁
  − x_t)/max(1 − t, 0.05) and warns that re-weighting per-component losses biases the joint
  ([017](017-remy26-joint-inference-weak-lensing.md) (C)). Neither is ablated there.
  Holzschuh+24 report x-prediction with 1/(1−t) weighting consistently worse than v-prediction
  ([008](008-holzschuh24-flow-matching-posterior-inference.md) (C)), and Arruda+25 note that
  switching the prediction target silently changes the effective loss weighting
  ([059](059-arruda25-diffusion-models-simulation-based.md) (B)). On the weighting warning,
  constant per-coordinate weights leave each coordinate's minimiser at its conditional mean
  velocity, so they cannot bias the population optimum; they only move where finite capacity
  goes. SlotFlow reports a phase weight of 2 cutting phase width 38%
  ([021](021-houba25-slotflow.md) (C)).
- **Bounded and discrete coordinates.** Pawsterior reports variational FM handling bounded
  supports and categorical structure where FMPE fails
  ([111](111-carrascopollo26-pawsterior.md) (C)); it needs convex supports, so not our circles
  and sphere. Compositional and all-in-one score models give tall-data and
  arbitrary-conditional inference
  ([108](108-geffner22-compositional-score-modeling-simulation.md) (B),
  [109](109-linhart24-diffusion-posterior-sampling-simulation.md) (B),
  [110](110-gloeckler24-all-one-simulation-based.md) (A)).
- **What we take.**
  - Stay with flow matching; no reason to go back to discrete flows.
  - Keep one unweighted flow loss across coordinates as the default. The case against
    reweighting is one C paper (017) and the maths above says it cannot bias the optimum, so
    this is a default, not a finding.

## 8. What we do not use

- **Composing window posteriors** with compositional or tall-data scores: windows overlap and
  are not conditionally independent given a source.
- **Conservative training** (balanced NRE/NPE, ensembles for coverage) and **interval
  rescaling**: they widen, and our failure is already too wide.
- **Symmetric-group alignment of slots** (OT over S₄ permutations): with ordered slots and N =
  4 there is nothing to gain.
- **Misspecification machinery** (robust NPE, sensitivity-aware ABI): relevant once confusion
  noise or a non-stationary PSD enters the simulator, not before. When it does, train on a
  superset of nuisances.
- **Cosmology summaries and field-level tricks**
  ([047](047-charnock18-automatic-physical-inference-information.md) (A) to
  [052](052-villaescusanavarro21-multifield-cosmology-artificial-intelligence.md) (B),
  [089](089-alsing19-fast-likelihood-free-cosmology.md) (A) to [093](093-hahn23-simbig.md)
  (B)): they address low-SNR summary choice, not narrow posteriors.
- **Adjacent LISA work, not summarised here** (a different question from ours): population SBI
  of DWDs (`srinivasan2025simulation`, `desanti2026inferring`); detection and separation with
  no posteriors (`zhao2023space`, `ma2026deep`; Tay+26, Cain+26, not in the bib); SGWB TMNRE
  (`alvey2024simulation`, `alvey2025leveraging`, `dimitriou2024fast`); label switching in MCMC
  (`stephens2000dealing`, `jasra2005markov`); the review `zhao2025dawning`. Talks with no paper
  yet (2026): Alvey on the global fit with SBI, Giarda on trans-dimensional SlotFlow, Bellotti
  and Korsakova on MBHB flow matching, Cuceu on SGWB separation with diffusion, Srinivasan on
  EMRIs with a transformer.

## 9. Order

1. **Exact φ± conditionals on saved draws** (§1). Training-free; settles the phase question.
2. **Likelihood polish of loud draws** (§3). Training-free; settles whether the loud gap is a
   deployment problem or a network one.
3. **The width-first scorecard** (§4): width ratios and JS vs MCMC, CRPS, circular SBC, IS ε vs
   SNR.
4. **Retrain without φ±** (§1), then **references and local priors for loud sources** (§2).
5. **Variable source count** (§5) when the 4 fixed slots stop being enough.

## Choices resting only on C evidence

Each with the cheapest test that would confirm it on our problem.

- **Likelihood polish of loud draws** (§3; flow-seeded MCMC is 065 only). Run the MALA polish
  on saved draws of B's loudest sources and compare widths with the narrowband MCMC; check
  that the draws keep their modes. Training-free.
- **Simulator-feedback control head** (§3; 008 only). After the polish, fine-tune a
  zero-initialised control head on XS for t ≥ 0.8 and compare loud-source width ratios with
  the frozen flow.
- **CRPS and the PPC χ²_red in the scorecard** (§4; 117 and 017 only). Compute both on an
  existing run's eval draws and check they rank runs the way the MCMC width ratio does.
- **Overlap as the flow's failure mode** (§5; 020 and 021 only). Cut the XS scorecard by
  separation from the nearest source and compare width ratios near and far.
- **Matched-filter phasors as an input** (§6; 042 and 043 only, and 042's gain fades with
  more data). The velocity probe under a global phase rotation on a short XS run with the
  phasors, against one without.
- **One unweighted flow loss** (§7; 017 only, with 021 against). Low priority, since the
  optimum is unbiased either way: one XS run with weight 2 on sky and f0, scored by width
  ratio.
- **Not asking the network for the JKS amplitudes** (§1; 020 only). Matters only if we try
  that parametrisation: one XS run with (α_L, α_R) as flow coordinates, scored by width
  ratio.
