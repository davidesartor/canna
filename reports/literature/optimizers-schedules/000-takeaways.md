# Optimizer takeaways for canna.lisa

What we take from the papers in this folder, what we leave out, and what we test next.
The code is `optimizer(config)` and `schedule_free` in
[optim.py](../../../src/canna/lisa/optim.py). Loss weighting, the t sampler and variance
reduction are in [../diffusion-flow-matching/000-takeaways.md](../diffusion-flow-matching/000-takeaways.md); hard-to-learn
coordinates (the phases) are in its §5 and in
[../gw-cosmology-inference/000-takeaways.md](../gw-cosmology-inference/000-takeaways.md) §1.

## How much to trust each source

Every paper link carries its evidence tier:
- **A, established:** ≥ 100 citations, or peer-reviewed with ≥ 30 citations/yr, or reproduced
  at scale by independent groups.
- **B, credible:** peer-reviewed at a recognised venue, or a preprint with ≥ 20 citations. A
  single result, not yet independently confirmed.
- **C, unvetted:** an unreviewed preprint (or workshop paper) with < 20 citations. A
  hypothesis.

A C claim is written "X report ...", and never alone justifies a recipe choice. Counts are
Semantic Scholar's on 2026-10-19. Recent papers have had little time to be cited, so C means
unvetted, not wrong. Bare arXiv links point outside this folder and are untiered: treat them as
C until checked. The choices that rest on C evidence alone, each with the cheapest test on our
problem, are listed at the end.

## What we use

Schedule-free around NorMuon on the matrices and AdamW on everything else. That is
SF-NorMuon, plus ScheduleFree+'s changes to the averaging. The wrapper, Muon and its update
scaling rest on A sources. Inner momentum, decay on z and all of the averaging changes rest on
C sources alone.

| Piece | Our setting | Taken from | Why |
|---|---|---|---|
| Wrapper | Train at y, evaluate at x | [Defazio+24](003-defazio24-schedule-free.md) (A) | No horizon fixed in advance: a run can be extended or stopped at any point. |
| Base on matrices | Muon (bf16 Newton–Schulz) on hidden matrices, AdamW on the rest | [Jordan+24](007-jordan24-muon.md) (A) | Muon beats tuned AdamW on LLMs ([Liu+25](008-liu25-moonlight.md) (A), [Wen+25](031-wen25-fantastic-optimizers.md) (A), though only 1.1× at 1.2B). On diffusion, [Chen+26](009-chen26-cmuon.md) (B) report chunked Muon over 2× faster than AdamW on DiTs; [Schaipp+25](004-schaipp25-diffusion-benchmark.md) (C) and [Li+26](010-li26-muon-dit-scaling.md) (C) report gains, [Bolatov+26](005-bolatov26-diffusion-benchmark.md) (C) a tie or win on only 2 of 4 tasks. |
| Update size | `consistent_rms=0.2`: every matrix's update has RMS 0.2·lr | [Liu+25](008-liu25-moonlight.md) (A) | Ties Muon's step to the lr, the same for every matrix shape. |
| Row normalisation | NorMuon's per-row second moment (β₂ 0.95), Frobenius norm kept | [Li+25](006-li25-normuon.md) (B), [Apte+26](001-apte26-sf-normuon.md) (C) | [Li+25](006-li25-normuon.md) (B) report 11% over Muon at 1.1B, on LLMs. Inside schedule-free, [Apte+26](001-apte26-sf-normuon.md) (C) report a small gain over SF-Muon (3.136 against 3.142) and about 12% faster. Neither is replicated. |
| Split fused matrices | q/k/v, the modulation's shift/scale/gate and SwiGLU's value/gate are `stacked_linear`, so each map is orthogonalised on its own | [Chen+26](009-chen26-cmuon.md) (B) | [Chen+26](009-chen26-cmuon.md) (B) report that fused DiT matrices make Muon plateau late. One result. |
| Inner momentum | Muon 0.8; Adam b1 0.9 | [Apte+26](001-apte26-sf-normuon.md) (C), [Defazio+26](002-defazio26-schedulefree-plus.md) (C) | C only, and against the original, which drops it ([Defazio+24](003-defazio24-schedule-free.md) (A)). [Apte+26](001-apte26-sf-normuon.md) (C) report loss 3.14 → 3.28 without it; [Defazio+26](002-defazio26-schedulefree-plus.md) (C) report that it moves the large-batch cliff back to AdamW's. |
| Weight decay | λ = 0.05 on z, as lr·λ·z, Muon matrices only | [Liu+25](008-liu25-moonlight.md) (A), [Apte+26](001-apte26-sf-normuon.md) (C) | That Muon needs decay is A: without it, weights grow and Muon's lead fades ([Liu+25](008-liu25-moonlight.md) (A)). That it must act on z rests on [Apte+26](001-apte26-sf-normuon.md) (C) alone: they report decay at y diverging on long runs, and prove decay on z keeps every iterate bounded. λ = 0.05 is their value. |
| Warm start | x follows z for 2× the lr warmup | [Defazio+26](002-defazio26-schedulefree-plus.md) (C) | C only. [Defazio+26](002-defazio26-schedulefree-plus.md) (C) report that warmup iterates otherwise stay in the average for the whole run. |
| Averaging weights | t^r·lr², r = 1 | [Defazio+26](002-defazio26-schedulefree-plus.md) (C) | C only. [Defazio+26](002-defazio26-schedulefree-plus.md) (C) report r = 1 better on long runs, and ours are long (240k–650k steps). [Karras+23](014-karras23-edm2.md) (A) find a much narrower average best for diffusion (see the open question). |
| Interpolation β | 0.9 → 0.965, log-linear in t | [Defazio+26](002-defazio26-schedulefree-plus.md) (C) | C only. [Defazio+26](002-defazio26-schedulefree-plus.md) (C) report that low β converges faster early and high β ends lower; [Kim+26](015-kim26-amuse.md) (C) argue the same direction. |
| Clip, warmup | global norm 1.0; linear lr warmup | [Defazio+24](003-defazio24-schedule-free.md) (A), [Apte+26](001-apte26-sf-normuon.md) (C) | Schedule-free needs a warmup (A); [Semenov+25](030-semenov25-llm-optimizer-benchmark.md) (B) find SF-AdamW wants a long one (~25%). The clip value rests on [Apte+26](001-apte26-sf-normuon.md) (C) alone. |

SF-NorMuon's own defaults differ from ours in three ways we keep:
- constant β = 0.9;
- r = 0 with no warm start;
- SF-AdamW without inner momentum on the 1-D parameters.

ScheduleFree+'s long-run results favour ours, but that is one C report on LLMs against another.

## What we leave out

- **ScheduleFree+'s Polyak step.**
  - It scales each step by f(y) − f\*, and the code sets f\* = 0.
  - A flow-matching loss has a nonzero floor whose scale depends on the problem. With f\* = 0,
    the step tracks the whole loss rather than the excess over the floor, and never shrinks
    near the optimum. Theory backs this: a floor estimate below the true minimum converges
    only to a neighbourhood whose size is proportional to the gap
    ([arXiv:2511.07767](https://arxiv.org/abs/2511.07767),
    [arXiv:2505.20219](https://arxiv.org/abs/2505.20219), both untiered).
  - It is defined only for Adam and tested only on LLMs.
- **AdamC decay (λ·lr²).** It exists to decouple decay from the Polyak step, which we don't
  use. With a fixed lr, decay on z (SF-NorMuon) does the job.
- **The 1/EMA‖g‖₁ step weighting.** It couples the step size to the gradient norm, and through
  that to the clip and the decay. The step can drift away from the lr we set.
- **AMUSE's low starting β (0.4–0.6).** [Kim+26](015-kim26-amuse.md) (C) is Muon-only and
  tested only on LLMs and ResNets. Our β ramp is the same idea starting from 0.9. It is a
  candidate only if the anneal tests below show the fast sequence oscillating.
- **EDM2's forced weight normalisation and post-hoc EMA.** [Karras+23](014-karras23-edm2.md)
  (A) are alternatives to decay on z and to schedule-free's x, not additions to them.
  HyperP ([Ren+26](020-ren26-hyperp.md) (C)) is the same idea for Muon. All three are
  candidates only if the weight norm drifts (see "Clip and weight norm" below).
- **Other base optimizers, for now.** SOAP ([Vyas+24](025-vyas24-soap.md) (A)), KL-Shampoo
  ([Lin+25](026-lin25-kl-shampoo.md) (C)), PSGD-Kron ([027](027-li-psgd-kron.md) (A)) and
  AdEMAMix ([Pagliardini+24](028-pagliardini24-ademamix.md) (B)) are real alternatives. Both
  diffusion benchmarks (C) report SOAP tying or beating Muon. But tuned LLM comparisons
  shrink every claimed speedup ([Wen+25](031-wen25-fantastic-optimizers.md) (A)), and none of
  them answers the anneal question below. PSGD-Kron is the cheapest to try in JAX
  (`psgd-jax`); SOAP and KL-Shampoo have no optax code.
- **Lion, Sophia, MARS.** They need large batches or long horizons break them
  ([Semenov+25](030-semenov25-llm-optimizer-benchmark.md) (B)).
  [Bolatov+26](005-bolatov26-diffusion-benchmark.md) (C) report Lion diverging on one
  diffusion task, and sign updates suit fine precision poorly.
- **Schedule-free's tuning-free argument.** Schedule-free won AlgoPerf's *self-tuning* track
  ([Kasimbeg+25](032-kasimbeg25-algoperf.md) (B), [Defazio+24](003-defazio24-schedule-free.md)
  (A)): its strength is robustness without tuning. We tune by hand, so we keep it for the
  anytime property, not for that.

## How we judge an optimizer change

**On the posteriors, never on held-out loss alone.** The evidence is two independent C
benchmarks, but the posteriors are what we want anyway, so adopting this costs nothing.
- [Schaipp+25](004-schaipp25-diffusion-benchmark.md) (C) and
  [Bolatov+26](005-bolatov26-diffusion-benchmark.md) (C) both report SF-AdamW reaching a
  competitive loss with worse samples.
- [Bolatov+26](005-bolatov26-diffusion-benchmark.md) (C) report the denoising loss varying by
  under 2% across optimizers while FID varies more than 2×, and rankings flipping between
  tasks.

So:
- score each run with the scorecard (f0 width, detection, calibration) and the MCMC ratios;
- evaluate at x (`eval_flow`), and also at y (`weights: training`) when the lr schedule
  changes;
- compare two optimizers only after each has annealed. Rankings flip mid-run because of the lr
  decay ([Wen+25](031-wen25-fantastic-optimizers.md) (A)).

## The open question: no anneal at steady state

None of these papers tests schedule-free with Muon on diffusion or flow matching. The two
diffusion benchmarks (C) report that what plain schedule-free lacks is an anneal; the cooldown
literature below (A, B) finds the same of averaging on LLMs.

**Decay on z fixes the step size for good.** From SF-NorMuon's steady-state lemma
([Apte+26](001-apte26-sf-normuon.md) (C)), with the update RMS at 0.2·lr (the same √(2ηλ)
angular step as rotational equilibrium, [Kosson+23](018-kosson23-rotational-equilibrium.md)
(B)):

| Quantity | Formula | Ours (Muon lr 1e-2, λ 0.05) | SF-NorMuon (lr 1e-2, λ 0.05) |
|---|---|---|---|
| RMS(z) | 0.2·√(lr / 2λ) | 0.063 | 0.063 |
| Angular step per step | √(2·lr·λ) | 0.032 | 0.032 |
| Steps to steady state | 1/(lr·λ) | 2k | 2k |

- The weights turn by about 3% per step until the end of the run, as with SF-NorMuon's
  defaults. Nothing anneals that.
- Averaging smooths z's noise in x, but the cooldown literature finds averaging does not fully
  replace an anneal. A lower final lr is what helps ([Hägele+24](011-hagele24-cooldown.md) (A),
  [Bergsma+25](012-bergsma25-straight-to-zero.md) (B)). On a Navier–Stokes denoiser,
  [Schaipp+25](004-schaipp25-diffusion-benchmark.md) (C) report a cooldown repairing part of
  schedule-free's sample gap.
- Power Lines says to tune lr and λ through the timescale 1/(lr·λ)
  ([Bergsma+25b](013-bergsma25b-power-lines.md) (A)). Ours is 2k steps, under 1% of a
  240k-step run.
- Our averaging horizon is wide. t^r weights are EDM2's power-function EMA with γ = r:
  - r = 1 gives σ_rel 0.24;
  - EDM2's FID optimum is 0.07–0.13, which corresponds to r ≈ 5–10.

  The mapping is only approximate: EDM2's EMA sits on top of a decaying lr, and schedule-free's
  x feeds back into the gradient point. Still, EDM2 (A) is the stronger source here than
  ScheduleFree+'s (C) "r > 1 gives no further gain" on LLMs.

**Why an anneal should matter for narrow posteriors.**
- Muon's step has a fixed size, whatever the gradient's size
  ([Jordan+24](007-jordan24-muon.md) (A)), so it does not shrink near the optimum. PolarGrad
  scales the update by the gradient's nuclear norm to restore that
  ([Lau+25](017-lau25-polargrad.md) (B)).
- In the river-valley picture, a high lr makes progress along the valley while bouncing across
  it, and only the decay removes the bounce ([Wen+24](033-wen24-river-valley.md) (A)). Fine
  precision lives on the valley walls.
- [Kim+26](015-kim26-amuse.md) (C) argue Muon makes this worse: it speeds up the flat
  directions and amplifies oscillation in the high-curvature ones. A narrow posterior is a
  high-curvature direction.
- Averaging speeds training but complements decay rather than replacing it
  ([Ajroldi+25](034-ajroldi25-weight-averaging.md) (B), in line with
  [Hägele+24](011-hagele24-cooldown.md) (A)).

**If the anneal matters,** the recipe follows the cooldown papers:
- decay the lr to ≤ 0.01×, not 0.1× ([Semenov+25](030-semenov25-llm-optimizer-benchmark.md)
  (B), [Bergsma+25](012-bergsma25-straight-to-zero.md) (B): two independent results);
- raise β₂ during the cooldown ([Dremov+25](035-dremov25-cooldown-dynamics.md) (B), one
  result);
- to extend a finished run, continue at a lower lr rather than restarting at the peak
  ([Schaipp+25b](036-schaipp25b-convex-schedules.md) (B), one result);
- [Morwani+25](029-morwani25-sf-ademamix.md) (C) argue that schedule-free's gains survive
  without its x/y/z bookkeeping: (simplified) AdEMAMix plus a normal cooldown is the same
  family.

## Hyperparameters to revisit

- **β₂ = 0.95 is a short-run setting.** Both AdamW and NorMuon's row moment use it.
  [Semenov+25](030-semenov25-llm-optimizer-benchmark.md) (B) find 0.999 beats 0.95 on long LLM
  runs, and ours are long. [Dremov+25](035-dremov25-cooldown-dynamics.md) (B) find a higher β₂
  helps in a cooldown, and [Hägele+24](011-hagele24-cooldown.md) (A) that schedule-free is
  sensitive to β₂.
- **Newton–Schulz coefficients.** Polar Express's per-iteration coefficients are closer to
  orthogonal at the same step count and are stabilised for bf16
  ([Amsel+25](016-amsel25-polar-express.md) (A)). It is a change to `ns_coeffs` only.
- **λ as a spectral ceiling.** Muon with decay solves a spectral-norm-constrained problem of
  radius ∝ 1/λ ([Chen+25b](024-chen25b-muon-spectral-constraint.md) (A)). λ must leave room
  for the large weights a sharp posterior needs.
- **Batch size.** The gradient noise scale grows as the loss falls
  ([McCandlish+18](039-mccandlish18-gradient-noise-scale.md) (A)). With unlimited
  simulations, a larger batch late in training is a substitute for lr decay. Scale Adam's lr
  by √batch when the batch changes ([arXiv:2205.10287](https://arxiv.org/abs/2205.10287),
  untiered).

## Clip and weight norm

- **The global clip at 1.0 can become a hidden lr schedule.** If the weights shrink behind
  normalisations, the gradient norm grows and the clip binds on every step. Per-unit adaptive
  clipping ([AGC, Brock+21](037-brock21-agc.md) (A); `optax.adaptive_grad_clip`) or spike-only
  clipping ([ZClip, Kumar+25](038-kumar25-zclip.md) (C)) avoid that. Log the pre-clip gradient
  norm and the weight RMS to tell.
- **Decay under a schedule.** [Defazio+25](019-defazio25-adamc.md) (B) show that with a
  decaying lr, a fixed-λ decay makes the effective lr of normalised layers drift, and the
  gradient norm rises late in training. [Karras+23](014-karras23-edm2.md) (A) see the same
  norm-driven drift of the effective lr. At our constant lr this cannot happen; in a cooldown
  (E3) it can, and decay ∝ lr² (AdamC) is the fix.
- **Weights that are not scale-invariant.** Rotational equilibrium holds only for weights
  followed by a normalisation ([Kosson+23](018-kosson23-rotational-equilibrium.md) (B)). The
  head's hidden layer (`x_unembed.linear1`) is not, so decay just shrinks its rows. Muon's own
  recipe keeps only the final output layer on AdamW ([Jordan+24](007-jordan24-muon.md) (A)),
  as we already do, so it does not settle this hidden layer. Options that stop the shrinking:
  - moving `linear1` off Muon to AdamW as well;
  - Muown's separate row magnitudes ([Lion+26](021-lion26-muown.md) (C));
  - cautious weight decay, which only decays where update and weight agree in sign
    ([Chen+25](023-chen25-cautious-weight-decay.md) (C));
  - Muon-SW's decay scaled to keep the weight RMS fixed
    ([Apte+26b](022-apte26b-scaled-weight-decay.md) (C)).

## Experiments

Three XS runs, each changing one thing against the XS recipe. Each is scored as above.

| Run | Overrides on `XS` | Change | Question |
|---|---|---|---|
| E1, SF-NorMuon's steps | `LISA_LEARNING_RATE=1e-2 LISA_ADAM_LEARNING_RATE=1e-2 LISA_NESTEROV=false LISA_MOMENTUM_DTYPE=float32` | AdamW lr 1e-3 → 1e-2 (Muon's is already 1e-2); plain fp32 momentum | Are we under-stepping? |
| E2, smaller angular step | `LISA_WEIGHT_DECAY=0.005` | λ 0.05 → 0.005: RMS(z) 0.2, angular step 0.010, timescale 20k steps | Does the steady-state step limit fine parameters? |
| E3, cooldown | `LISA_COOLDOWN_FRAC=0.2` | lr → 0 linearly over the last 20% | Does an anneal recover what averaging misses? |

- E2 and E3 test the anneal from two sides. E1 runs against them: it makes AdamW's step
  10× larger.
- E3 gives up schedule-free's anytime property, so it is a diagnostic, not a recipe.
- **E1's momentum settings reach Adam too.** `optax.contrib.muon` passes `nesterov` and
  `mu_dtype` to its AdamW group as well.
- **E3 keeps x's averaging weights at the uncooled lr.** With the cooled lr², the cooldown's
  iterates would get almost no weight in x, and x would ignore the anneal. Score y as well.
- **Launch:** `slurm/lisa-chain.sh XS 2 LISA_OUTPUT_DIR=outputs/e1 <overrides>` for each run,
  about 11.5 h on one A100. Each needs its own `LISA_OUTPUT_DIR`: the run dir is
  `lisa-<config>`.

**After E1–E3**, one change per XS run, in this order:

| Run | Change | Question |
|---|---|---|
| E4 | β₂ 0.95 → 0.999 (AdamW and NorMuon rows) | Is the second moment too noisy for long runs? |
| E5 | Polar Express `ns_coeffs` | Is the bf16 NS output far enough from orthogonal to matter? |
| E6 | `adaptive_grad_clip` in place of the global clip | Is the clip binding every step? |
| E7 | Head (`x_unembed.linear1`) on AdamW, or cautious decay on it | Does decay shrink the head rows? |
| E8 | Two EMAs of y (σ_rel 0.05, 0.10), post-hoc reconstructed | Is a narrower average than x's better? |
| E9 | PSGD-Kron or AdEMAMix as the base, compared after cooldown | Is Muon the bottleneck at all? |

Free checks on existing runs: average the last k saved y checkpoints and score them
([LAWA, arXiv:2209.14981](https://arxiv.org/abs/2209.14981), untiered); at a cooldown's start,
reset z ← x ([Switch EMA, arXiv:2402.09240](https://arxiv.org/abs/2402.09240), untiered).

## Choices resting only on C evidence

To verify, not to assume. Each with the cheapest test on our problem:
- **Decay on z, not y** ([Apte+26](001-apte26-sf-normuon.md) (C)). Free first look: on an
  existing run, RMS(z) should settle near 0.063 within about 2k steps if their steady-state
  lemma holds. Then one XS run with decay at y, logging ‖z‖ and ‖z − x‖: they predict both
  grow.
- **λ = 0.05** (the value from [Apte+26](001-apte26-sf-normuon.md) (C)). Power Lines (A)
  fixes only the timescale 1/(lr·λ), not a value for Muon. E2 tests it.
- **Inner momentum,** Muon 0.8 and Adam b1 0.9 ([Apte+26](001-apte26-sf-normuon.md) (C),
  [Defazio+26](002-defazio26-schedulefree-plus.md) (C)). One XS run with both at 0, the
  original's setting ([Defazio+24](003-defazio24-schedule-free.md) (A)). The cliff they report
  is at 2M tokens per batch, so the gain may not carry to our batch size.
- **Warm start** at 2× the lr warmup ([Defazio+26](002-defazio26-schedulefree-plus.md) (C)).
  One XS run without it, scored early and at the end.
- **r = 1 averaging** ([Defazio+26](002-defazio26-schedulefree-plus.md) (C)), against EDM2's
  (A) narrower optimum. Free: the LAWA check above, scored against x; then E8.
- **β ramp 0.9 → 0.965** ([Defazio+26](002-defazio26-schedulefree-plus.md) (C),
  [Kim+26](015-kim26-amuse.md) (C)). One XS run at SF-NorMuon's constant β = 0.9.
- **Global clip at 1.0** (the value from [Apte+26](001-apte26-sf-normuon.md) (C)). Free: log
  the pre-clip gradient norm. If it sits above 1.0 on most steps, the clip is acting as an lr
  schedule; then E6.

Lone-B choices, one unreplicated result each: NorMuon's row normalisation
([Li+25](006-li25-normuon.md) (B), LLMs only) and splitting fused matrices
([Chen+26](009-chen26-cmuon.md) (B), image DiTs only).
