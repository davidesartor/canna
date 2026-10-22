# Architecture takeaways for canna.lisa

What we take from the papers in this folder for building the network: where the compute goes,
precision, attention, how numbers and positions get in, conditioning, the output heads, and the
phase symmetry. The objective and training recipe are in [../diffusion-flow-matching/](../diffusion-flow-matching/), GW and
cosmology inference practice in [../gw-cosmology-inference/](../gw-cosmology-inference/), the optimizer in
[../optimizers-schedules/000-takeaways.md](../optimizers-schedules/000-takeaways.md).

The setting all of this is read against: an MMDiT with a 4-token source stream (11 flow
coordinates each, ordered loudest first) and a 256-token data stream (STFT patches of a
whitened rfft window), joint attention with RoPE on a shared frequency axis, one adaLN
conditioning vector from t and the window frequency, and an f32 head giving a velocity and an
x̂₁ estimate. Three facts drive the choices:

- **Compute is the data stream.** About 98% of FLOPs are spent on the 256 data tokens,
  recomputed at every ODE stage for the same observation.
- **Loud sources need sub-ulp precision.** For the loudest sources a 1σ shift in the data is a
  fraction of a bf16 ulp, and their posterior widths come out tens of times too wide.
- **The φ± phases are not learned.** Their velocity does not depend on the data.

## How much to trust each source

Every paper link carries its evidence tier, from the line under the paper's citation:

- **A, established:** ≥100 citations, or peer-reviewed with ≥30 citations/yr.
- **B, credible:** peer-reviewed at a recognised venue, or a preprint with ≥20 citations. A
  single result, not yet independently confirmed.
- **C, unvetted:** an unreviewed preprint or workshop paper with <20 citations. A hypothesis.

A C claim is written "X reports …" and never alone justifies a design choice: it earns a test
on our problem first. Counts are Semantic Scholar on 2026-10-19; JiT
([051](051-li25-back-basics.md) (C)) has no entry there, so its 4 OpenAlex citations are a lower
bound. Recent papers have had little time to be cited, so C means unvetted, not wrong. The
choices that still rest only on C evidence are listed in §12, each with the cheapest test that
would confirm it here.

## 1. Compute the condition once, sample many draws per condition

- **Evidence is broad and replicated: four papers here, two A and two B, plus AlphaFold 3.**
  AlphaFold 3 (48 diffusion samples per trunk pass), MAR (4 t per condition vector,
  [004](004-li24-autoregressive-image-generation-without.md) (A)), DDT's reusable encoder
  ([001](001-wang25-ddt.md) (A)), FoldFlow-2 ([072](072-huguet24-sequence-augmented-se-3.md) (B))
  and Dingo-T1 ([013](013-kofler25-flexible-gravitational-wave-parameter.md) (B)) all compute the
  expensive condition once and run a cheap per-step network on top.
- **A one-way mask makes the cache exact.** OminiControl2 and the deep-fusion study put the
  condition tokens in the joint sequence but never let them attend to the noisy tokens, so
  their per-layer K/V are fixed per observation ([002](002-tan25-ominicontrol2.md) (A),
  [003](003-tang25-exploring-deep-fusion-large.md) (B); also EasyControl,
  [022](022-zhang25-easycontrol.md) (A), abstract only). The quality cost is small *only if
  trained with the mask*: caching without mask training gives deblur FID 45.0 vs 16.4
  ([002](002-tan25-ominicontrol2.md) (A)).
- **The condition stream does not need t.** One ablation backs this: the deep-fusion study's
  t-free variant has the best FID ([003](003-tang25-exploring-deep-fusion-large.md) (B)).
  Lumina-Image 2.0 ([020](020-qin25-lumina-image-2-0.md) (A)) and Nucleus-Image
  ([021](021-akiti26-nucleus-image.md) (C)) keep it t-free without ablating the choice. FM is the
  objective most robust to dropping t from the whole network (FID 3.01 → 2.61 without it,
  [025](025-sun25-noise-conditioning-necessary-denoising.md) (A)), but that argument needs x_t to
  reveal t, which a prior-draw base need not: it supports a t-free data stream, not a t-free
  source stream (§5).
- **What we take: the masked MMDiT.**
  - Data queries attend only to data keys; source queries see everything.
  - The data stream is modulated by the window frequency alone, not by t.
  - Cache the per-layer data K/V (about 2M numbers per observation) and run only the 4 source
    tokens per ODE stage: roughly 60× cheaper per stage, about 40× faster sampling.
  - In training, draw K (x₀, t) pairs per observation. The data stream is shared, so K = 16
    costs about 1.25 forwards and cuts the x₀/t part of the gradient variance ~K×.
  - Train the data path end to end; do not freeze it. Mao et al. report end-to-end beats a
    frozen two-stage pipeline ([065](065-mao25-robust-scalable-simulation-based.md) (C); §12).
  - Fallback if fully one-way loses quality: keep the last one or two blocks joint.
  - This is cheaper to build than a separate encoder plus cross-attention decoder (DDT-style),
    since it reuses the existing block.
- **Token sparsity is the fallback only.** TREAD, SPRINT, MaskDiT and MDT work only with dense
  first and last blocks plus an unmasked fine-tune, and degrade sharply past 75–87.5% sparsity
  ([045](045-krause25-tread.md) (B), [027](027-park25-sprint.md) (B)); the two B papers disagree,
  as TREAD did not reproduce under SPRINT's protocol. Dingo-T1's data-based token masking is
  the GW analogue. Under condition-once they are moot; if it is rejected, SPRINT-style routing
  of the data tokens is next. Routing drops exactly the bins loud-source precision depends on.

## 2. Precision: f32 at the inputs and conditioning, bf16 in the core

- **The sensitive parts are likely the IO and the conditioning, not the trunk: practice, not
  ablation.** Aurora runs f32 encoder/decoder around a bf16 backbone
  ([035](035-bodnar24-foundation-model-earth-system.md) (B)); rCM traced an instability to its
  time embedding and fixed it in f32
  ([008](008-zheng25-large-scale-diffusion-distillation.md) (B)); neither ablates precision.
  GenCast gives its noise-level path its own Fourier features
  ([038](038-price23-gencast.md) (A)), with no precision choice recorded. For spectral
  scientific models the Fourier layers are the precision-sensitive part ([059](059-hayford24-speeding-up-reducing-memory.md) (B)). The one
  quantitative ablation is Qi et al., who report bf16 alone costs about a third of the accuracy
  in an FM surrogate of rarefied flow
  ([026](026-qi26-physics-guided-generative-surrogates.md) (C), 0 citations).
- **bf16 may break relative position in attention.** Wang et al. report that RoPE rotations
  done in f32 still lose precision once q·k is computed in bf16
  ([011](011-wang24-when-precision-meets-position.md) (C), long-context LLMs), and a bf16 RoPE
  frequency buffer has been seen to shift angles by up to 1.4 rad (diffusers issue #14935, a
  bug report). If so, sub-bin frequency information read through attention is quantized. The
  nearest A evidence is SD3's bf16 logit growth
  ([005](005-esser24-scaling-rectified-flow-transformers.md) (A)), a stability problem, not a
  position one.
- **What we take.**
  - Keep the modulation path (time/frequency embeddings, adaLN projections) in f32: about 1% of
    FLOPs.
  - Once the data stream is computed once per observation (§1), run it in f32, and compute the
    4 source-query attention rows in f32. Both become nearly free. The f32 data stream follows
    from the ulp fact above and [059](059-hayford24-speeding-up-reducing-memory.md) (B); the f32
    attention rows rest only on [011](011-wang24-when-precision-meets-position.md) (C) (§12).
  - Precision fixes are unlikely to close the loud-source gap alone. A Dingo port to LISA comes
    out too diffuse at high SNR, which the authors blame on the mass-covering objective
    ([Spadaro+26](../gw-cosmology-inference/022-spadaro26-accurate-efficient-simulation-based.md) (B)), and
    Delmond et al. report the same broadening for galactic binaries
    ([Delmond+26](../gw-cosmology-inference/020-delmond26-neural-posterior-estimation-galactic.md) (C)).
    Pair the network with a likelihood correction.

## 3. Let attention be sharp

- **Gain-free QK-norm caps the logit.** With RMS-normalized q and k and d_head = 64, logits are
  at most about √64 = 8, so at best about 92% of the attention mass can land on one of 260
  keys. SD3 and nGPT both keep a learnable QK scale
  ([005](005-esser24-scaling-rectified-flow-transformers.md) (A),
  [009](009-loshchilov24-ngpt.md) (A)), and QK-norm is shown to lower retrieval sharpness
  ([057](057-yang25-rope-nope-back-again.md) (A)).
- **What we take.** Measure attention entropy of the source queries first; if it sits near the
  cap, add a learnable f32 per-head gain (8 scalars per block, AdamW group).
- **Value residual** (a skip from the first block's values into later attention) is a cheap
  untested extra from a single-author diffusion speedrun, which reports 4.02 → 3.64
  ([043](043-bhanded25-speedrunning-imagenet-diffusion.md) (C); §12).

## 4. Numbers and positions

- **Piecewise-linear embeddings resolve a coordinate to one bin.** PLE plus a linear layer
  gives sub-bin resolution only through that layer's arithmetic; adding periodic features (PLR)
  helps on tabular data ([012](012-gorishniy22-embeddings-numerical-features-tabular.md) (A)),
  and Fourier features are what let MLPs fit high frequencies at all
  ([036](036-tancik20-fourier-features-let-networks.md) (A), learnable variant
  [060](060-li21-learnable-fourier-features-multi.md) (A)). Features that mix coordinates before
  the periodic map did worse than a plain MLP in
  [012](012-gorishniy22-embeddings-numerical-features-tabular.md) (A), so embed each coordinate
  on its own.
  **What we take:** multi-scale f32 Fourier features of each precise coordinate (f₀ above
  all) beside PLE, in f32 before the first bf16 cast.
- **RoPE on a continuous frequency axis.**
  - Keep the high RoPE frequencies: they are what builds sharp positional heads
    ([034](034-barbero24-round-round-we-go.md) (A)), and a source must find its exact data bin.
  - Never interpolate or rescale positions when the window size changes: plain interpolation
    destroys high-frequency resolution ([058](058-peng23-yarn.md) (A)), and FiTv2 reports it
    catastrophic for 2-D RoPE ([068](068-wang24-fitv2.md) (C)). The positions are physical bins.
  - Check analytically that the RoPE base keeps relative-position discrimination positive over
    the full window ([033](033-men24-base-rope-bounds-context.md) (B)).
  - Learned RoPE variants (RoPE-Mixed, STRING, LieRE, ComRoPE) buy at most about a point
    in-distribution ([056](056-heo24-rotary-position-embedding-vision.md) (A); also
    [067](067-yu25-comrope.md) (B), [070](070-liu25-rethinking-rope.md) (C)). Low priority;
    position jitter is moot when positions are already continuous.
  - RoPE leaves the values unrotated, so attention cannot form a phase-aligned sum. Rotating
    v by the key's angle and the output back by the query's costs no parameters; removing it
    costs 2.45 dB PSNR in GTA ([073](073-miyato24-gta.md) (A)). See
    [../complex-valued-networks/000-takeaways.md](../complex-valued-networks/000-takeaways.md) §3.
- **Do not tokenize numbers as one scaled embedding** (xVal): its precision is about 10⁻², far
  too coarse.

## 5. Conditioning

- **adaLN-zero is the baseline**
  ([007](007-peebles22-scalable-diffusion-models-transformers.md) (A)): it beats cross-attention
  and in-context conditioning, and its zero init makes a fresh network ignore the condition,
  so a test of "does conditioning reach the output" is vacuous at init.
- **Precise per-token information should modulate, not just add.** Three results, two A and
  one B, favour multiplicative per-token conditioning: per-pixel adaLN (PixelDiT, FID 3.50 →
  2.36, [006](006-yu25-pixeldit.md) (A)), DeCo's adaLN vs additive (31.35 vs 36.02,
  [052](052-ma25-deco.md) (B)), and FMPE's GLU vs concatenation of t (0.5 vs 3.6 mnat,
  [../diffusion-flow-matching/018](../diffusion-flow-matching/018-dax23-flow-matching-scalable-simulation.md) (A)).
  Dingo-T1 feeds its per-token frequency bounds through a GLU in the tokenizer
  ([013](013-kofler25-flexible-gravitational-wave-parameter.md) (B)).
  **What we take:** per-source adaLN, c_i = c + g(features_i), with g fed by f32 features.
  Start with a null control where g sees only the source's own coordinates, then feed physics
  features (a local likelihood step, matched-filter phasors).
- **Where t matters.** Yang et al. find the network's Lipschitz constant in t blows up at the
  data end ([032](032-yang23-lipschitz-singularities-diffusion-models.md) (B), one paper, in an
  ε-prediction setting), so the source stream needs t most there, with a late-resolving time
  embedding such as log(1 − t). The data stream can drop it (§1).
- **Dual-stream depth.** Amber-Image reports late dual-stream layers can be merged at no
  prompt-score cost but lose fine detail ([024](024-yang26-amber-image.md) (C), 1 citation);
  FLUX.1 Kontext runs most of its depth single-stream, unablated
  ([040](040-labs25-flux-1-kontext.md) (A)). Keep dual-stream throughout, because fine detail is
  the loud-source problem; SD3's MMDiT beats single-stream DiT
  ([005](005-esser24-scaling-rectified-flow-transformers.md) (A)), but the depth question rests
  on the C paper (§12).

## 6. The output side

- **Tie the two heads.** We already predict the velocity and x̂₁ but never tie them.
  Self-Consistent Flow reports a consistency term between them takes FID 11.14 → 9.85
  ([010](010-han26-self-consistent-flow.md) (B): one paper, not replicated, and inconsistent
  about its own x-loss weight). The case here rests on the algebra more than on that paper: on
  flat coordinates under a geodesic path the true minimizers satisfy x̂₁ = x_t + (1 − s)·v
  exactly, so the term does not move the optimum, and it gives the velocity head x̂₁'s
  regression signal near s → 1, where the FM loss is weak.
  **What we take:** a consistency term on flat coordinates only, with stop-gradient through
  the chart, weight ~0.03–0.1; never evaluate a log map at an off-manifold prediction.
  SimpleFold's x̂ auxiliary loss is precedent, and its 1/t weighting collapsed training, so no
  1/(1 − s) weight ([054](054-wang25-simplefold.md) (B)).
- **Self-conditioning on x̂₁: three A papers.** Feed the network its own previous x̂₁ (half
  the time, under stop-gradient, zeros otherwise); at sampling it is carried from the previous
  stage for free ([031](031-chen22-analog-bits.md) (A),
  [049](049-jabri22-scalable-adaptive-computation-iterative.md) (A),
  [050](050-gulrajani23-likelihood-based-diffusion-language.md) (A)). It also lets RoPE place
  sources at x̂₁'s f₀ rather than x_t's, which is bins off for most of the path. Under §1 the
  extra pass is a source-stream-only forward, about 2% of the cost instead of 15–25%.
- **Do not import image-scale output tricks.** The wide DiT^DH head hurts on low-dimensional
  outputs (7.13 → 11.70,
  [016](016-zheng25-diffusion-transformers-representation-autoencoders.md) (A)); JiT reports
  that x-prediction matters only for high-dimensional tokens ([051](051-li25-back-basics.md)
  (C)), so at 11 coordinates per source every parametrization should work, an extrapolation
  from its 48-d patches (§12); DiP's MLP head is worse than none ([053](053-chen25-dip.md) (A));
  pixel decoders repair high-dimensional v-prediction failures. A two-layer f32 head is enough.
- **Residual on a reference.** Predicting a residual on a deterministic mean
  (ArchesWeatherGen, [055](055-couairon24-archesweather-archesweathergen.md) (B), no result
  extracted: precedent, not evidence) is the architecture form of heterodyning, which
  DINGO-BNS applies to the data ([063](063-dax24-real-time-gravitational-wave.md) (B)): give the
  head an f32 reference and let the network predict the correction.

## 7. The phase symmetry

- **A known GW-NPE pathology, solved analytically so far.** DINGO's coalescence phase "infers
  the prior" and is marginalized on an analytic grid
  ([DINGO-IS](../gw-cosmology-inference/001-dax22-neural-importance-sampling-rapid.md) (A)); Dingo-T1 still
  handles it analytically at 160M parameters
  ([013](013-kofler25-flexible-gravitational-wave-parameter.md) (B)). Contrastive FM names the
  same failure ("the network predicts the mean across conditions") but its fix biases the
  minimizer.
- **Options, in order of certainty.**
  1. Exact phase conditionals outside the network (see
     [../gw-cosmology-inference/000-takeaways.md](../gw-cosmology-inference/000-takeaways.md) §1): guaranteed.
  2. **U(1) canonicalization.** Compute a smooth data phase Φ(y) (from the
     |z|²-softmax-weighted sum of whitened bins), rotate y by e^{−iΦ}, predict, rotate back.
     Canonicalization gives exact equivariance with any backbone if the canonicalizer is
     continuous ([014](014-kaba22-equivariance-learned-canonicalization-functions.md) (A)); that
     it also helps a generative model train rests on C papers
     ([039](039-sareen25-symmetry-aware-generative-modeling.md) (C); Zhou et al. argue it
     removes the global-phase variance from the target,
     [015](015-zhou26-rethinking-diffusion-models-symmetries.md) (C); §12). One global phase
     fixes only one source (the loudest), as with a single GNPE pose
     ([062](062-dax21-group-equivariant-neural-posterior.md) (B)); it is the cheapest
     diagnostic. Other LISA NPE exploits known symmetries in preprocessing the same way
     ([064](064-liang24-rapid-parameter-estimation-merging.md) (B)).
  3. **A global phasor token** generated jointly with the sources, REG-style (FID 33.0 → 15.2
     with a meaningful summary token vs 23.31 for a learnable one,
     [029](029-wu25-representation-entanglement-generation.md) (A)).
  4. **Mid-block phasor supervision**: predict the matched-filter phasors from block 3 of 8
     (REPA placement, [028](028-yu24-representation-alignment-generation.md) (A)), with
     per-token targets ([047](047-singh25-what-matters-representation-alignment.md) (A)) and
     decayed late (HASTE, [046](046-wang25-repa-works-until-it.md) (B);
     [048](048-liu25-efficient-generative-model-training.md) (C)).
- With no pretrained GW encoder, the physics (phasors, F-statistic) plays the encoder's role in
  any alignment loss.

## 8. Generic training-efficiency extras

- **Registers.** 4–8 extra tokens in the source stream from block 2 cost almost nothing and
  may act as global scratch space. Proteina reports they help
  ([061](061-geffner25-proteina.md) (A)); Starodubcev et al. report they help in pixel space
  when inserted after the first blocks but hurt from block 0 and in a latent space
  ([018](018-starodubcev26-registers-matter-pixel-space.md) (C)). Mixed evidence; the block-2
  placement rests on the C paper (§12).
- **LayerSync.** Encoder-free self-alignment of an early to a late block (36.19 → 30.00),
  tested also on 8-layer non-image models ([030](030-haghighi25-layersync.md) (B), one paper).
  It is parameter-free and reported robust to λ, so it is the cheapest generic convergence
  trick to try: source tokens, block 2 → 6, λ ≈ 0.2.
- **Alignment helps early, then fade it.** REPA's gains
  ([028](028-yu24-representation-alignment-generation.md) (A)) shrink with training in iREPA
  ([047](047-singh25-what-matters-representation-alignment.md) (A)) and reverse late in HASTE
  (XL 7.3 vs 5.5 if not stopped, [046](046-wang25-repa-works-until-it.md) (B)). ERW also decays
  its alignment, but its headline speed-up is inconsistent with its own table
  ([048](048-liu25-efficient-generative-model-training.md) (C)).
- **Per-group clocks.** La-Proteina gives coordinate groups different time densities, and
  FrameFlow uses an inference-only per-group schedule ([037](037-geffner25-la-proteina.md) (B),
  [071](071-yim23-fast-protein-backbone-generation.md) (A)). A faster clock for f₀ and sky can
  be tried at sampling first.
- **Spend FLOPs per data token, not parameters.** SiD2 finds compute per token beats more
  parameters ([019](019-hoogeboom24-simpler-diffusion-sid2.md) (B); DiT-Air agrees, no file
  here); patch 16 is already what Dingo-T1 uses, so do not shrink it.

## 9. Optimizer interaction

- Muon on DiTs is contested, but mostly by C papers: positive at scale
  ([042](042-li26-scaling-muon-diffusion-transformers.md) (C), 0 citations), catastrophic in the
  speedrun ([043](043-bhanded25-speedrunning-imagenet-diffusion.md) (C)), negative in Chimera
  ([044](044-ge26-chimera.md) (C), no numbers). The one B paper, CMuon, has Muon slightly ahead
  of AdamW at DiT-B (3.31 vs 3.49) and shows the per-matrix learning-rate rule alone swings
  FID 8.73 → 3.31 ([041](041-chen26-cmuon.md) (B)), which may explain the speedrun's result.
- Storing q/k/v and the adaLN shift/scale/gate as separate matrices is effectively CMuon (B,
  though worse than plain Muon at 512²). Keeping the time and frequency MLPs on AdamW rests on
  Chimera's instability warning ([044](044-ge26-chimera.md) (C)), whose instability came from a
  timestep-MLP width mismatch rather than from Muon (§12). Muon's update-RMS scaling is
  already roughly μP. Details in [../optimizers-schedules/000-takeaways.md](../optimizers-schedules/000-takeaways.md).

## 10. What we do not use

- MoE (DiT-MoE, DiffMoE, Race-DiT, EC-DiT), SANA linear attention, parameter sharing:
  parameters and attention FLOPs are not the bottleneck.
- Contrastive FM: biased minimizer, breaks calibration.
- The DiT^DH wide head, x/x₁-prediction, pixel decoders: built for high-dimensional outputs.
- REPA-E: no VAE. Dispersive loss: weak with 4 source tokens and small batches.
- xVal number tokens.
- Approximate inference caching (Learning-to-Cache, ToCa): exact caching under §1 supersedes
  it.

## 11. Order

Each step against a small-model baseline at equal steps or wall-clock; read out held-out loss,
per-source widths against MCMC, calibration, and a phase-rotation probe.

1. **Masked MMDiT with K draws per observation** (§1). Mask ablation at K = 1 (pass if within
   noise), then K = 8 at equal wall-clock; time the cached sampler. Enables 2, 5 and 7.
2. **f32 data stream and f32 source-query attention** (§2). Pre-check: bf16 vs f32
   source-query logits on a trained model.
3. **U(1) canonicalization wrapper** (§7). Pass if the rotation probe responds and the loudest
   source's φ± posterior narrows; if not, the phase stays outside the network.
4. **Learnable per-head QK gain** (§3), after an attention-entropy check.
5. **f32 modulation path, then plus log(1 − t)** (§2, §5). Pass if late-s held-out loss and
   loud-source widths improve.
6. **Velocity/x̂₁ consistency term on flat coordinates** (§6), weight 0.03 and 0.1; check
   calibration.
7. **Per-source adaLN fed by f32 features** (§5), null control first.
8. **Phasor token or mid-block phasor supervision** (§7).
9. **Registers** and **LayerSync** (§8): convergence only.
10. **Multi-scale f32 Fourier features beside PLE** (§4).
11. **Data-token masking or SPRINT routing** (§1): only if 1 is rejected.

## 12. Choices resting only on C evidence

Each needs its test before it counts as a result.

- **f32 source-query attention rows** (§2,
  [011](011-wang24-when-precision-meets-position.md) (C)). Test: on a trained model, compare
  bf16 and f32 source-query logits and each source's argmax data bin (the step-2 pre-check);
  keep f32 only if they differ beyond noise.
- **Data path trained end to end, never frozen** (§1,
  [065](065-mao25-robust-scalable-simulation-based.md) (C)). Already the default; test only if
  freezing is proposed: freeze the data stream at a mid-run checkpoint, compare held-out loss.
- **Value residual** (§3, [043](043-bhanded25-speedrunning-imagenet-diffusion.md) (C)). Test:
  add it to XS, compare held-out loss at equal steps.
- **Dual-stream weights to the last block** (§5, [024](024-yang26-amber-image.md) (C)). Test:
  share data and source weights in the last two blocks, compare loud-source widths against
  MCMC.
- **Registers from block 2, not block 0** (§8,
  [018](018-starodubcev26-registers-matter-pixel-space.md) (C); [061](061-geffner25-proteina.md)
  (A) backs registers, not their placement). Test: 4 registers from block 0 vs block 2 on XS,
  held-out loss.
- **Time and frequency MLPs on AdamW** (§9, [044](044-ge26-chimera.md) (C)). Test: a short run
  with them on Muon; look for loss spikes and a worse late-s held-out loss.
- **Canonicalization helps training, beyond equivariance** (§7,
  [015](015-zhou26-rethinking-diffusion-models-symmetries.md) (C),
  [039](039-sareen25-symmetry-aware-generative-modeling.md) (C)). Test: the phase-rotation probe
  and the loudest source's φ± width with and without the wrapper (step 3).
- **No x-prediction at 11 coordinates** (§6, [051](051-li25-back-basics.md) (C)). Test: one XS
  run with an x-prediction loss, held-out loss at equal steps.
