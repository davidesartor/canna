# Diffusion / flow-matching takeaways for canna.lisa

What we take from the papers in this folder for the training objective and recipe: the path,
the clock, the target, the loss weights, the sampler and the correction of the samples. Network
design is in [../network-architecture/](../network-architecture/), the GW/cosmology inference practice in
[../gw-cosmology-inference/](../gw-cosmology-inference/), the optimizer in
[../optimizers-schedules/000-takeaways.md](../optimizers-schedules/000-takeaways.md).

The setting all of this is read against: a conditional Riemannian flow from a prior draw to the
posterior on a product of lines, an S² and two circles per source; posterior widths from about
the prior's down to 10⁻⁵ of it, varying per draw with SNR; an exact, cheap, differentiable
likelihood; and a velocity for the φ± circles that does not depend on the data.

## How much to trust each source

Every paper link carries its evidence tier (Semantic Scholar counts, 2026-10-19):

- **A, established:** ≥100 citations, or peer-reviewed with ≥30 citations/yr.
- **B, credible:** peer-reviewed at a recognised venue, or a preprint with ≥20 citations. A
  single result, not yet independently confirmed.
- **C, unvetted:** unreviewed preprint or workshop paper with <20 citations. A hypothesis.

A C claim is written as "X report ..." and never alone justifies a recipe choice; where an A or
B paper supports the same choice, it is cited first. Recent papers have had little time to be
cited, so C means unvetted, not wrong. arXiv links with no file here are untiered; read them as
C.

**Choices resting only on C evidence**, each with the cheapest test on our problem:

- **Likelihood residual or ∇log L as a network input** (§5;
  [017](017-gilo26-flowbender.md) (C),
  [Holzschuh–Thuerey](../gw-cosmology-inference/008-holzschuh24-flow-matching-posterior-inference.md) (C)).
  Test: one XS run with a zero-initialised residual input against one without, same steps;
  compare the φ± rotation response and per-source coverage.
- **Weight decay on z, not y, for the Muon group** (§9;
  [010](010-apte26-anytime-training-schedule-free.md) (C), while
  [007](007-defazio24-road-less-scheduled.md) (A) prefers y for SF-AdamW). Test: two short runs
  differing only in decay placement; compare held-out loss and ‖z‖ late in the run.
- **Muon's inner momentum kept inside schedule-free** (§9; [009](009-kim26-amuse.md) (C),
  [010](010-apte26-anytime-training-schedule-free.md) (C)). Test: one short run with μ = 0.
- **The β ramp, log-linear or AMUSE's** (§9; [008](008-defazio26-schedulefree.md) (C),
  [009](009-kim26-amuse.md) (C); 008 finds its own anneal ineffective at 100 TPP). Test: fixed
  β against the ramp at our run length.
- **r = 1 averaging only for long runs** (§9; [008](008-defazio26-schedulefree.md) (C)). Test:
  r = 0 against r = 1 on held-out loss at our run length.

Resting on a single B result, so also worth checking before leaning on them: sampling density
over loss weight ([012](012-hang24-improved-noise-schedule-diffusion.md) (B)), x̂₁ over
velocity prediction in a head-to-head ([058](058-stark23-harmonic-self-conditioned-flow.md)
(B)), and the cut-locus failure of torus FM
([026](026-lou23-scaling-riemannian-diffusion-models.md) (B), qualitative).

## 1. What the objective leaves free

- **The minimiser is fixed; the trade-offs are not.** Any Bregman loss, any positive weight
  over t, any time density with full support and any invertible linear preconditioner of the
  output give the same population optimum, E[target | x_t, t, y]
  ([023](023-lipman24-flow-matching-guide-code.md) (A), Generator Matching
  [arXiv:2410.20587](https://arxiv.org/abs/2410.20587) (untiered),
  [arXiv:2511.16599](https://arxiv.org/abs/2511.16599) (untiered),
  [085](085-kim24-simple-reflow.md) (B)). So per-coordinate, per-t, per-SNR or Fisher weights
  are all legal. They change which errors a finite network trades off and the gradient
  variance, never what a perfect network converges to.
- **Huber, pseudo-Huber and LPIPS are not in that class.** With an independent
  prior-to-posterior coupling the target is noisy, and a robust loss estimates something other
  than the conditional mean: it shrinks posterior widths. Those losses come from reflow, where
  the coupling is nearly deterministic ([067](067-lee24-improving-training-rectified-flows.md)
  (A), [078](078-song23-improved-techniques-training-consistency.md) (A)).
- **The regression loss bounds the KL only for an SDE sampler.** For the deterministic ODE
  there is no likelihood guarantee from the loss
  ([020](020-albergo23-stochastic-interpolants.md) (A),
  [035](035-song21-maximum-likelihood-training-score.md) (A),
  [100](100-benton23-error-bounds-flow-matching.md) (B)). FMPE's mass-covering bound needs
  regularity a network does not guarantee
  ([018](018-dax23-flow-matching-scalable-simulation.md) (A)). Calibration has to be measured,
  and a small SDE or corrector step at sampling time is a calibration lever with theory behind
  it.

## 2. The clock: where training draws go

- **Prefer moving the density to moving the weight, weakly.** At the optimum they are
  equivalent ([011](011-esser24-scaling-rectified-flow-transformers.md) (A) gives the exact
  map). Logit-normal sampling ranks first of 61 densities and weightings in
  [011](011-esser24-scaling-rectified-flow-transformers.md) (A). Hang+24 report that sampling
  more t where it is hard beats multiplying the loss there (Laplace sampling 7.98 vs weighting
  8.88 FID, [012](012-hang24-improved-noise-schedule-diffusion.md) (B), one DiT-B setting).
  SiD2 ([036](036-hoogeboom24-simpler-diffusion-sid2.md) (A)) finds the training density has
  little effect once the weighting is tuned, so the gap is gradient variance, not the answer.
  Change where t is drawn first; a late-s loss multiplier is the fallback, not ruled out.
- **Shift by the width ratio.** The optimal shift grows with how much finer the target is than
  the base: +2 log(base/res) in log-SNR ([024](024-hoogeboom23-simple-diffusion.md) (A),
  [037](037-chen23-importance-noise-scheduling-diffusion.md) (A)), SD3's √(m/n)
  ([011](011-esser24-scaling-rectified-flow-transformers.md) (A)). Our analogue is prior width
  / posterior width, which spans 10–10⁵ and changes per draw. That argues for a shift
  conditioned on the draw's SNR (or Fisher width), not one global `time_power`. FMPE's
  p(t) ∝ t^{1/(1+α)}, α = 1 on GW ([018](018-dax23-flow-matching-scalable-simulation.md) (A)),
  is the same family as our warp.
- **Hedge with a mixture.** Image densities (logit-normal, Laplace) concentrate mid-path and
  de-emphasise the clean end because their targets are perceptual. Ours needs both early t
  (mode finding, detection) and very late t (the loud widths). Keep the uniform part and add a
  component log-uniform in 1 − s (`late_time_mix`). SiD2
  ([036](036-hoogeboom24-simpler-diffusion-sid2.md) (A)) found a wide log-SNR training range
  crucial, which supports keeping the uniform part.
  [arXiv:2512.20685](https://arxiv.org/abs/2512.20685) (untiered) reports that an analytic
  shift toward the data end cut calibration error by about 30%; not replicated.
- **Make late times distinguishable.** Condition on log(1 − s), EDM's c_noise = ¼ log σ
  ([005](005-karras22-elucidating-design-space-diffusion.md) (A)), beside the sinusoidal t
  embedding; a plain embedding of t cannot tell 1 − s = 10⁻³ from 10⁻⁴. Keep it bounded at the
  end ([049](049-lu24-simplifying-stabilizing-scaling-continuous.md) (A) on c_noise blow-ups).
- **One clock for all coordinates is wrong when they contract at different rates.**
  High-frequency or small-variance components lose their signal first under a shared clock
  ([013](013-falck25-fourier-space-perspective-diffusion.md) (B)); per-coordinate schedules
  work when conditioned ([066](066-sahoo23-diffusion-models-learned-adaptive.md) (B)). f₀ and
  Mc contract to ~10⁻³ of the prior while the phases barely contract. The cheap version is to
  whiten each flow coordinate by its expected posterior width; the principled one is a
  per-coordinate clock conditioned on y.
- **Free variance reduction.** Stratified (low-discrepancy) t within the batch
  ([021](021-kingma21-variational-diffusion-models.md) (A)). A learned t proposal that
  minimises loss variance ([065](065-kim24-adaptive-non-uniform-timestep.md) (B),
  [039](039-nichol21-improved-denoising-diffusion-probabilistic.md) (A)) is optional, and must
  be judged on the loss at all t.

## 3. The target and its weights

- **Keep regression targets O(1).** EDM's preconditioning
  ([005](005-karras22-elucidating-design-space-diffusion.md) (A)) at the posterior's scale, not
  the prior's. With a prior base, the variance of the velocity target rises toward the data
  end, from σ_post² to σ_pr² (the opposite of the score picture,
  [041](041-xu23-stable-target-field-reduced.md) (B)), so the late end dominates gradient norms
  unless the output is scaled.
- **x̂₁-prediction on the curved factors, converted with a clip.** FrameFlow and FoldFlow
  predict the endpoint on SO(3) and convert it as v = log_{x_t}(x̂₁)/(1 − s)
  ([015](015-yim23-fast-protein-backbone-generation.md) (A),
  [016](016-bose23-se-3-stochastic-flow.md) (A)); FrameFlow clips the 1/(1 − s) factor at
  t = 0.9 because the loss exploded. The head-to-head evidence is thinner: HarmonicFlow found
  x₁-prediction beat velocity, 20.5% vs 11.9%
  ([058](058-stark23-harmonic-self-conditioned-flow.md) (B)). Li & He
  ([004](004-li25-back-basics.md) (C)) report x-prediction as the stable choice for a
  concentrated target, but their failure of v needs patch dimension above token width, and all
  parametrisations work when it is not. Woo+26 ([076](076-woo26-riemannian-meanflow.md) (C))
  report v-prediction degrading on spheres up to D = 2048. Gagneux+26
  ([019](019-gagneux26-training-flow-matching.md) (C)) report velocity winning for U-Nets and
  fine patches and the denoiser for MLPs and low intrinsic dimension. Their clips: 0.05 in
  [004](004-li25-back-basics.md) (C), 0.05–0.1 in [076](076-woo26-riemannian-meanflow.md) (C).
- **But a point x̂₁ is only approximately the marginal velocity.** The exact field is
  κ̇/(1 − κ)·E[log_{x_t} X₁], not the log of the mean; the gap is large when the posterior over
  X₁ is wide or multimodal on a circle ([023](023-lipman24-flow-matching-guide-code.md) (A)).
  Use the x̂₁ form for the narrow coordinates; keep a velocity loss as the training objective.
- **Weights should grow toward the clean end, then cap.** SiD2's sigmoid weighting increases
  monotonically toward clean inputs and beats ε-mse and EDM weightings
  ([036](036-hoogeboom24-simpler-diffusion-sid2.md) (A)); Min-SNR finds pure SNR weighting (all
  weight on the clean end) worst and the capped form lowest in every t bin
  ([038](038-hang23-efficient-diffusion-training-min.md) (A)). Gagneux+26
  ([019](019-gagneux26-training-flow-matching.md) (C)) report the same direction for FM on
  CIFAR-10, not replicated. Grow-then-cap is the A-backed form; the cap is the default, not an
  ablation.
- **Learned per-(t, coordinate-group) uncertainty weights** (EDM2,
  [006](006-karras24-analyzing-improving-training-dynamics.md) (A)): a small MLP of log(1 − s)
  gives a log-variance u_g per group, loss Σ e^{−u_g}·mse_g + u_g. Every region then
  contributes O(1), which is what makes late-t loud-source errors visible. It will also
  down-weight any coordinate the network cannot learn, so introduce it only once every
  coordinate is learnable.
- **Use the intrinsic metric, at x_t.** A Euclidean loss on SPD matrices gave 0% valid samples
  against 100% with the Riemannian norm
  ([001](001-chen23-flow-matching-general-geometries.md) (A)). Compute the loss in the tangent
  space at x_t; never evaluate log_map at an off-manifold network output, whose log scales with
  its radius. Project ambient (cos, sin) outputs first.

## 4. Circles and spheres

- **Angular factors come out under-sharpened.** FrameFlow, FoldFlow and FlowMM all boost the
  angular velocity late at inference ([015](015-yim23-fast-protein-backbone-generation.md) (A),
  [016](016-bose23-se-3-stochastic-flow.md) (A), [014](014-miller24-flowmm.md) (A)); applying
  it to the Euclidean factors hurt, and training with it removed the signal. None checks
  calibration. For us it is a diagnostic (does the φ± or sky spread shrink toward the reference
  when boosted?), not a fix.
- **Weight per factor, deliberately.** FlowMM puts λ = 300–1500 on the torus loss against 1 for
  the lattice ([014](014-miller24-flowmm.md) (A)); FrameDiff, Torsional Diffusion and DiffCSP
  divide each angular loss by its expected target norm
  ([047](047-yim23-se-3-diffusion-model.md) (A),
  [045](045-jing22-torsional-diffusion-molecular-conformer.md) (A),
  [046](046-jiao23-crystal-structure-prediction-joint.md) (A)).
- **Periodic in the noise and in the features.** DiffCSP loses 51.5% → 34.1% without the
  wrapped normal and → 29.2% without Fourier features
  ([046](046-jiao23-crystal-structure-prediction-joint.md) (A)). Angle convention or symmetry
  errors fail silently as "learned the prior"
  ([045](045-jing22-torsional-diffusion-molecular-conformer.md) (A)).
- **The cut locus hurts torus FM.** Geodesic targets are discontinuous at the antipode, and
  with a uniform base half the conditional targets sit near it. Use multi-harmonic
  sin/cos(kφ), k = 1..6, as input features and a non-uniform training base on the circle
  ([026](026-lou23-scaling-riemannian-diffusion-models.md) (B), qualitative only;
  FrameFlow's IGSO3 base in [015](015-yim23-fast-protein-backbone-generation.md) (A), reported
  without numbers).
- **Dedicated circular density heads:** circular splines are weak on concentrated targets;
  Möbius mixtures do better ([031](031-rezende20-normalizing-flows-tori-spheres.md) (A)).

## 5. A coordinate whose velocity ignores the data

The φ± symptom (the flow returns the prior on the phases) matches a known failure, and the
literature points to the representation, not the clock or the head.

- **Why it stalls.** The phase's location is arg⟨h(f₀, …), y⟩, a high-frequency function of the
  data with no degree-1 correlation under a uniform prior. That is a high information exponent
  / leap: SGD sits at the y-blind saddle (v_φ ≈ 0 on the circle) for a time that grows with
  the exponent ([028](028-benarous20-online-stochastic-gradient-descent.md) (A),
  [029](029-abbe23-sgd-learning-neural-networks.md) (A)). Periodic single-index targets can be
  exponentially hard for gradient methods
  ([060](060-shamir16-distribution-specific-hardness-learning.md) (A),
  [089](089-song21-cryptographic-hardness-learning-single.md) (B)); spectral bias learns high
  frequencies last ([059](059-rahaman18-spectral-bias-neural-networks.md) (A)). These are
  worst-case results: the mechanism transfers, the rate is not established.
- **Fix it in the input.** A lifting of the inputs, not depth, removes spectral bias
  ([027](027-tancik20-fourier-features-let-networks.md) (A)). Give the network the demodulated
  phase: Re/Im of the matched-filter inner product of the data with the template at the
  current estimate (x_t or x̂₁). Then the target is nearly linear in the input.
- **Or give it a stepping stone.** An auxiliary output that is low-degree in the data and
  shares the hard direction's support lowers the leap
  ([029](029-abbe23-sgd-learning-neural-networks.md) (A)): a (cos φ±, sin φ±) phasor head on
  each source token.
- **Feed back through the forward model (C evidence only).** FlowBender
  ([017](017-gilo26-flowbender.md) (C)) feeds the residual or ∇log L at x̂₁ into a
  zero-initialised head, with dropout 0.1, and measures no calibration. Holzschuh–Thuerey
  ([../gw-cosmology-inference/008](../gw-cosmology-inference/008-holzschuh24-flow-matching-posterior-inference.md) (C))
  report simulator feedback in SBI giving −19% χ² and about 35× faster sampling than MCMC. Both
  are unreplicated; the demodulated input above is the A-backed route to the same signal.
  Self-conditioning on x̂₁ ([057](057-chen22-analog-bits.md) (A)) has mixed results, and
  Holzschuh–Thuerey (C) report no gain on low-dimensional SBI. If tried, use HarmonicFlow's
  variant with a prior draw as the null input, which gained 14.3% → 20.5%
  ([058](058-stark23-harmonic-self-conditioned-flow.md) (B)).
- **Rao-Blackwellise the target.** Regress onto a self-normalised importance average of the
  conditional velocity over K candidate endpoints. Closed-form FM gives modest, steady gains
  with the true endpoint always among the candidates
  ([062](062-bertrand25-closed-form-flow-matching.md) (A)). STF
  ([041](041-xu23-stable-target-field-reduced.md) (B)) cuts variance and wall clock, but its
  data-marginal proposal is useless when p(x₁ | x_t) is far narrower than the prior (ESS → 1),
  which is our late-s regime. PAFM ([002](002-stoica26-posterior-augmented-flow-matching.md)
  (C)) reports about 4× lower gradient variance for 6.6% throughput, under a theorem that does
  not cover the candidates it actually uses. The image papers have to drop p(y | x); we have it
  exactly. For the phases, candidates on the circle weighted by the likelihood (other
  parameters held at the draw) give a target whose expectation already depends on y. This
  extension is untested.
- **What does not help.** Critical-window and symmetry-breaking theory
  ([090](090-raya23-spontaneous-symmetry-breaking-generative.md) (B),
  [091](091-biroli24-dynamical-regimes-diffusion-models.md) (A)) is about unconditional data
  breaking symmetry along t; here y breaks it at every t. Memorisation theory
  ([062](062-bertrand25-closed-form-flow-matching.md) (A),
  [094](094-kadkhodaie23-generalization-diffusion-models-arises.md) (A)) assumes a finite
  training set. More late-t draws cannot help a velocity that ignores y at all t. Mixture
  velocity heads ([093](093-chen25-gaussian-mixture-flow-matching.md) (B)) learn a
  better-calibrated uniform until the phase is readable; they matter afterwards, for the
  φ₀ ↔ φ₀ + π type degeneracies. Contrastive FM's optimum is a fixed affine distortion of the
  FM field, independent of y ([arXiv:2506.05350](https://arxiv.org/abs/2506.05350)
  (untiered)): it adds no incentive to use the data.
- **Diagnose first.** Measure ∂v_φ/∂y (or the response to a global phase rotation of the
  data), and fit linear probes for (cos φ, sin φ) on the hidden source tokens at intermediate
  s. A probe that finds the phase means a head/optimisation problem; none means a
  representation problem.

## 6. Base and coupling

- **The prior is a valid base** ([033](033-liu22-flow-straight-fast.md) (A),
  [034](034-albergo22-building-normalizing-flows-stochastic.md) (A)), and its exact density
  keeps endpoint consistency trivial.
- **A fixed, analytic, y-dependent base helps.** Data-dependent couplings improved in-painting
  FID 1.35 → 1.13 with the same model
  ([025](025-albergo23-stochastic-interpolants-data-dependent.md) (A)); PriorGrad's
  hand-computed N(μ(c), Σ(c)) halved iterations ([043](043-lee21-priorgrad.md) (A)). That a
  learned one is fragile is a C report: Kim+26
  ([044](044-kim26-better-source-better-flow.md) (C)) see NaNs without variance pinning and the
  gain vanish on multimodal conditions. If we try one, it is an analytic periodogram or Fisher
  Gaussian, at least as wide as the posterior (a point-mass base needs SDE sampling,
  [071](071-chen24-probabilistic-forecasting-stochastic-interpolants.md) (B)), and the network
  keeps seeing y.
- **Couplings buy steps, not accuracy.** Minibatch OT ≈ independent coupling at adequate NFE
  ([068](068-tong23-improving-generalizing-flow-based.md) (A)); equivariant OT gains nothing
  with four particles ([003](003-klein23-equivariant-flow-matching.md) (A)); conditional OT
  needs repeated y and collapses to the identity pairing when every y is unique
  ([042](042-kerrigan24-dynamic-al-conditional-optimal.md) (B),
  [069](069-chemseddine24-conditional-wasserstein-distances-applications.md) (B)). Reflow
  compounds marginal error ([033](033-liu22-flow-straight-fast.md) (A),
  [085](085-kim24-simple-reflow.md) (B)).
- **Standardise every coordinate.** Unit-variance inputs and targets are EDM's preconditioning
  ([005](005-karras22-elucidating-design-space-diffusion.md) (A); per-example input
  normalisation in [037](037-chen23-importance-noise-scheduling-diffusion.md) (A)). Zhai+25
  ([070](070-zhai25-conditional-flow-matching-bayesian.md) (C)) report a conditional flow that
  failed to learn even the prior without it.

## 7. Exact correction

The flow is a proposal; the likelihood is exact. Use both.

- **Exact log q.** At ~44 flow dimensions the exact divergence (full Jacobian trace) is
  affordable ([030](030-grathwohl18-ffjord.md) (A),
  [018](018-dax23-flow-matching-scalable-simulation.md) (A); log q costs about 13× a sample
  there). On our manifold it needs the divergence on the product of spheres, RK stages that
  stay on the manifold, and a density for the clipped coordinates.
- **Then importance-sample.** DINGO-IS
  ([001](../gw-cosmology-inference/001-dax22-neural-importance-sampling-rapid.md) (A)) is the template:
  weights q/(L·π), effective sample size as the diagnostic. A too-wide proposal is the safe
  failure for IS, but efficiency falls roughly as (σ_true/σ_flow)^d, so pair it with tempering.
- **SMC when IS fails.** Twisted SMC
  ([095](095-wu23-practical-asymptotically-exact-conditional.md) (A)), Feynman–Kac correctors
  ([096](096-skreta25-feynman-kac-correctors-diffusion.md) (A)), flow-annealed IS
  ([097](097-midgley22-flow-annealed-importance-sampling.md) (A)).
- **Avoid fine-tuning that narrows.** Endpoint-MSE "MLE" fine-tuning
  ([098](098-li25-fine-tuning-flow-matching.md) (C)) regresses a sampled endpoint onto its
  paired target, a conditional mean, so it narrows a stochastic posterior; 098 itself measures
  only success rate and FID. Consistency distillation for SBI shrinks posteriors too.
  DPS-style likelihood guidance is a tuned weight knob, not a posterior
  ([087](087-chung22-diffusion-posterior-sampling-general.md) (A)).
- **Measure calibration with coverage, not classifiers.** Per-parameter rank/coverage
  (circular for the phases), TARP on the joint, expected coverage error. C2ST is insensitive
  to the over-dispersion we care about.

## 8. What we do not use for posteriors

- **Guidance of any kind.** CFG, interval guidance and autoguidance sample a sharpened tilt
  p(x|y)^w·p(x)^{1−w}, not the posterior
  ([054](054-chidambaram24-what-does-guidance-do.md) (B),
  [055](055-bradley24-classifier-free-guidance-predictor.md) (B),
  [056](056-karras24-guiding-diffusion-model-bad.md) (A),
  [088](088-wu24-theoretical-insights-diffusion-guidance.md) (B)). Autoguidance is useful only
  as a diagnostic of where a weaker model differs.
- **Few-step models and distillation** ([048](048-geng24-consistency-models-made-easy.md)–
  [053](053-geng25-mean-flows-one-step.md), all A; [077](077-song23-consistency-models.md)–
  [084](084-shaul25-transition-matching.md), A and B) buy speed, not calibration. If sampling
  cost matters later: Riemannian versions exist ([076](076-woo26-riemannian-meanflow.md) (C));
  IMM is the only distribution-level loss
  ([083](083-zhou25-inductive-moment-matching.md) (A)); sCM's stabilisers (bounded c_noise,
  tangent normalisation, adaptive per-t loss normalisation,
  [049](049-lu24-simplifying-stabilizing-scaling-continuous.md) (A)) carry over to ordinary FM
  training.

## 9. Optimizer

The full recipe and its tests are in [../optimizers-schedules/000-takeaways.md](../optimizers-schedules/000-takeaways.md).
From this folder's papers ([007](007-defazio24-road-less-scheduled.md) (A);
[008](008-defazio26-schedulefree.md)–[010](010-apte26-anytime-training-schedule-free.md), all
C). Everything below except the averaging weights and the post-hoc EMA rests on the three C
papers.

- Decay on z is right for Muon ([010](010-apte26-anytime-training-schedule-free.md) (C):
  stable and best throughout; decay at y unstable). Schedule-Free itself
  ([007](007-defazio24-road-less-scheduled.md) (A)) finds decay at y better on ImageNet and
  NanoGPT for SF-AdamW, so this is a C claim about Muon against an A default for AdamW. The
  AdamW group (head, embeddings, PLE tables) gets none, which leaves it without a restoring
  force.
- Keep Muon's inner momentum inside schedule-free: [009](009-kim26-amuse.md) (C) and
  [010](010-apte26-anytime-training-schedule-free.md) (C) each report that removing it
  degrades.
- AMUSE's horizon-free interpolation schedule, 1 − β_t ∝ (T₀/t)^0.8 with the guard rail that
  t·(1 − β_t) never shrinks ([009](009-kim26-amuse.md) (C)), is the principled form of our β
  ramp, as reported. ScheduleFree+ ([008](008-defazio26-schedulefree.md) (C)) finds its own β
  anneal ineffective at 100 TPP.
- Averaging weights c_t = η_t²/Σ η² ([007](007-defazio24-road-less-scheduled.md) (A):
  "greatly improved"); r = 1 only for long runs ([008](008-defazio26-schedulefree.md) (C)).
- An external EMA on top of schedule-free is not equivalent to its x; post-hoc EMA
  ([006](006-karras24-analyzing-improving-training-dynamics.md) (A)) is the cheap averaging
  sweep if we ever cool down y.
- None of these papers tests flow matching, and clipping is contested between them:
  [007](007-defazio24-road-less-scheduled.md) (A) found disabling it crucial for schedule-free
  on NanoGPT, while the C papers clip. Ablate it.

## 10. Order

1. Diagnose the phase (§5): rotation response and linear probes. Then either a demodulated
   input feature or a phasor head, with the likelihood-weighted phase target as the
   alternative.
2. Exact log q and importance sampling on saved draws (§7). It decides how much accuracy the
   network still owes us.
3. The clock (§2): SNR-conditioned shift or late-time mixture, log(1 − s) conditioning,
   stratified t.
4. Weights (§3): per-group normalisation with a Min-SNR-style cap, then learned uncertainty
   weights once every coordinate is learnable.
5. Angular hygiene (§4): multi-harmonic features, intrinsic loss, a boost test as a diagnostic.
