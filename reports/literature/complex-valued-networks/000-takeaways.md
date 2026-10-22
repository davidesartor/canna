# Complex-valued network takeaways for canna.lisa

What we take from the papers in this folder on complex-valued networks for signal processing:
when complex arithmetic beats a real network fed the same data, how to encode complex samples,
attention whose weights carry phase, phase equivariance, how networks learn a phase at all, and
what complex parameters cost in practice. The general network design is in
[../network-architecture/000-takeaways.md](../network-architecture/000-takeaways.md); the phase symmetry options there
(§7) are not repeated here.

The setting all of this is read against: a flow-matching posterior network whose data stream
embeds whitened complex STFT samples (per sample, asinh|z| and the saturating phasor
z/√(1+|z|²), each through a per-scalar piecewise-linear embedding, summed), real softmax
attention with RoPE on a shared frequency axis rotating q and k only, and source tokens that
must read a phase-aligned (matched-filter) sum of the data. Three open problems drive the
questions: the φ± phases are not learned, loud-source widths are far above the reference, and
the smallest network loses f₀ precision at mid SNR.

## How much to trust each source

Every paper link carries its evidence tier, from the line under the paper's citation:

- **A, established:** ≥100 citations, or peer-reviewed with ≥30 citations/yr.
- **B, credible:** peer-reviewed at a recognised venue, or a preprint with ≥20 citations. A
  single result, not yet independently confirmed.
- **C, unvetted:** an unreviewed preprint or workshop paper with <20 citations. A hypothesis.

A C claim is written "X reports …" and never alone justifies a design choice: it earns a test.

## 1. Complex arithmetic buys little over a real network fed (Re, Im)

- **Parameter-matched, the gap is small or zero.**
  - Speech enhancement: real twins of DCUNET, DCCRN, GCRN and LSTMs tie or beat the complex
    models at equal parameters, with 1/2–1/3 the multiply-adds, and the complex models lose
    more as they shrink ([040](040-wu23-rethinking-complex-speech.md) (B)). In DCUnet almost
    all of the gain comes from estimating phase at all (complex vs magnitude mask, +0.21–0.33
    PESQ); complex vs real arithmetic adds 0–0.06 ([043](043-choi19-deep-complex-unet.md) (A)).
    DCCRN's gain over a real RI baseline is about +0.1 PESQ, not parameter-matched
    ([044](044-hu20-dccrn.md) (A)).
  - MRI reconstruction: +0.3 dB (U-Net) to +1.9 dB (unrolled) PSNR, at matched parameters but
    with √2 more feature maps, so "complex" and "more maps" are not separated
    ([022](022-cole21-complex-cnn-mri-phase.md) (A)).
  - MusicNet transcription: +3.3 AP for a deep complex net on the spectrum; shallow nets tie,
    and vision is a wash ([023](023-trabelsi18-deep-complex-networks.md) (A)).
  - On real-valued data real nets win, and the imaginary weights just track the real ones
    ([026](026-monning18-cvnn-real-classification.md) (B)).
- **Where complex does win clearly, data or parameters are scarce.** Radar denoising reaches
  near-top results with 10% of the data where real needs ≥50%
  ([042](042-fuchs21-complex-radar-denoising.md) (B)); SAR and modulation classifiers match at
  a fraction of the size ([010](010-chakraborty19-surreal.md) (B)); synthetic classes that
  differ only in the Re–Im correlation give 0–2 points
  ([021](021-barrachina21-noncircular-cvnn-vs-rvnn.md) (B)). Hirose ties the advantage to
  signal coherence ([020](020-hirose12-coherence-generalization.md) (A)). We train on unlimited
  simulations, so the data-efficiency argument does not apply.
- **No complex paper addresses precision.** Nothing here bears on loud-source widths, and §6
  shows complex arithmetic makes the precision budget worse.

**What we take:** keep real arithmetic. What matters is that the network sees phase in a usable
form (§2) and can aggregate it coherently (§3).

## 2. Encode complex samples as magnitude and phasor, never as a raw angle

- **Cartesian beats a raw angle.** A real net fed |z| and the raw angle arg z overfit and
  never beat one fed (Re, Im), reported in one paragraph without numbers
  ([021](021-barrachina21-noncircular-cvnn-vs-rvnn.md) (B)). No paper here tests (|z|, cos φ,
  sin φ) against (Re, Im). Polar processing degrades at low SNR, where the phase is
  noise-dominated ([010](010-chakraborty19-surreal.md) (B)).
- **Mapping Re/Im beats magnitude, most at low SNR.** RI spectral mapping beats magnitude
  mapping by 3–6 STOI points, the margin largest at −5 dB
  ([046](046-tan20-gcrn-complex-spectral-mapping.md) (A)); phase-sensitive objectives beat
  magnitude-only ones ([052](052-erdogan15-phase-sensitive-mask.md) (A),
  [049](049-williamson16-complex-ratio-mask.md) (A)).
- **Both together beat either.** CMGAN feeds [|Y|^0.3, compressed Re, compressed Im]; dropping
  to magnitude only costs 0.18 PESQ, to RI only 0.06 PESQ and 1.9 dB SSNR
  ([045](045-cao22-cmgan-compressed-spectra.md) (A)).
- **An untested matched-filter argument.** The matched-filter statistic Re(Σ w̄ z) is linear
  in (Re z, Im z). With magnitude and phasor embedded separately, |z|·cos(θ − arg w) is a
  product of two scalars that later layers must form. No paper here measures this cost.
- Phase is learnable only when amplitude informs it: without the amplitude-to-phase link,
  PHASEN's phase output collapses to keeping the noisy phase
  ([048](048-yin20-phasen.md) (A)). Never put a ReLU on a signed RI output (−5.5 dB there).

**What we take:** keep what LisaFlow has: per sample asinh|z| and the gated phasor
z/√(1+|z|²), which is (cos φ, sin φ) above the noise and fades to 0 at it. The evidence against
polar inputs is against a raw angle (21) and an undefined phase at the noise floor (10); the
phasor avoids both, and CMGAN supports a magnitude beside the phase information (45).

## 3. Phase-carrying attention

- **RoPE is already complex attention.** Pairing head dimensions as complex numbers, RoPE's
  score is Re⟨q e^{iθm}, k e^{iθn}⟩ ([005](005-su21-rope-complex-rotation.md) (A)). With RoPE on
  a frequency axis, the score between a source at bin ν and a data token at bin b already
  carries the phase ramp e^{iθ(ν−b)}.
- **The value side has no phase.** Softmax weights are positive and v is not rotated, so a
  source reads a weighted average of data tokens and cannot form a phase-aligned sum of them.
  GTA rotates the values by the key's group element and maps the output back to the query's
  frame, giving A_ij·ρ_iρ_j⁻¹v_j; no new parameters. Removing the value rotation costs 2.45 dB
  PSNR (CLEVR-TR) and 0.81 dB (MSN-Hard); in DiT on ImageNet, GTA vs 2-D RoPE is FID 5.87 vs
  6.26, not isolating the value part ([../network-architecture/73](../network-architecture/073-miyato24-gta.md)
  (A)). RoVE restates it for 1-D language models and reports large length-extrapolation gains
  ([004](004-garciacastellanos26-rotary-value-embeddings.md) (C)). In complex terms, the
  effective weights become A_ij·e^{iθ(j−i)}: phase set by position, not content.
- **Content-dependent phase in the weights is weakly supported.** In a matched comparison of
  complex transformer blocks, the variant whose weights carry phase (APAtt) scored worst, and
  all variants sat within about 0.02 AP of the real transformer; the complex front end
  mattered far more than the attention form
  ([001](001-eilers23-complex-transformer-blocks.md) (B)); the complex transformer of
  [002](002-yang20-complex-transformer.md) (B) gains with more parameters. Huang et al. report
  that removing a holographic coherent sum costs 17 points on PolSAR, against a weak real
  baseline ([003](003-huang25-holographic-transformer.md) (C)). Hioki reports a phase-coherent
  sigmoid-gated attention beating softmax transformers, but the same author's real version of
  it ties or wins ([006](006-hioki26-phase-coherent-transformer.md) (C)): the gain is the gate,
  not complex numbers.

**What we take:** rotate values by the key's RoPE angle and un-rotate the output by the query's
in the source←data attention. It is cache-compatible (rotate the cached
values once) and parameter-free. The evidence it helps on a single position axis is one DiT
comparison.

## 4. Equivariance helps when the symmetry is exact and the target a coherent product of phases

- Gauge-equivariant flows sample U(1) lattice theory about 1500× more efficiently than HMC
  near criticality, while non-equivariant models "fail to learn"
  ([008](008-kanwar20-gauge-equivariant-flows.md) (A)). Across 2680 baseline CNNs, none fit 4×4
  Wilson loops (a product of many phases) beyond predicting the mean; the equivariant net fits
  and generalises to larger lattices ([009](009-favoni22-lattice-gauge-equivariant-cnn.md) (B)).
  Invariance to complex scaling beats an unconstrained complex CNN at about 25k parameters
  ([007](007-singhal22-codomain-symmetry.md) (B)).
- **But equivariance does not make a network learn a phase.** A uniform posterior over φ is
  itself U(1)-equivariant, so an equivariant network can still return the prior; it removes
  the symmetry from what must be learned but adds no gradient toward the phase. Exact
  equivariance is cheapest by canonicalization (see
  [../network-architecture/000-takeaways.md](../network-architecture/000-takeaways.md) §7), which works with any
  backbone.
- Equivariance can also cost: rotating-feature object binding deliberately broke rotation
  equivariance because the equivariant version did worse
  ([012](012-lowe23-rotating-features.md) (B)), while phase bias and gating were essential in its
  predecessor ([011](011-lowe22-complex-autoencoder.md) (B)).
- **Equivariant activations lose.** modReLU, the phase-equivariant activation, is the worst
  complex activation in MRI ([022](022-cole21-complex-cnn-mri-phase.md) (A)) and failed to
  converge in vision ([023](023-trabelsi18-deep-complex-networks.md) (A)); split CReLU, which
  breaks the symmetry, is best in both. Phase-preserving linear maps mattered more than
  phase-equivariant nonlinearities. Magnitude-only nonlinearities do give exact rotation
  equivariance with fewer parameters in steerable CNNs
  ([013](013-worrall17-harmonic-networks.md) (A)).

**What we take:** for φ±, canonicalize the data phase rather than build an equivariant complex
backbone; don't expect either to make the phase learnable on its own (§5).

## 5. Phases are learned last, or not at all, unless given their own pathway

- **GW posterior networks give up on the phase.** DINGO "often infers the prior" for the
  coalescence phase, because its conditional is tight and strongly correlated with the other
  parameters; the fix is to sample it analytically, not through the network
  ([060](060-dax23-dingo-is-synthetic-phase.md) (A)). A LISA galactic-binary NPE that flattens
  Re/Im into a real network reports broader high-frequency posteriors, blamed on the phase
  parameters ([064](064-delmond26-lisa-gb-npe.md) (C)). GNPE removes known symmetries by
  iterating a pose estimate ([065](065-dax22-gnpe-equivariant-npe.md) (B)).
- **Neural quantum states see the same failure and have three fixes.**
  - The phase gradient is drowned: early in training the amplitude term is about 100× the
    phase term, and two decoupled real networks (amplitude, phase) beat one holomorphic network
    ([061](061-bukov21-two-real-nets-vs-holomorphic.md) (B)).
  - Ordinary deep nets come out with "essentially random" phases; a phasor-sum phase layer
    trained phase-first fixes it, but an optimiser can still lock onto a simple wrong sign
    pattern that the energy alone does not reveal
    ([062](062-szabo20-phase-ansatz-sign-problem.md) (A)).
  - Phase training needed natural gradient where amplitudes did not
    ([066](066-torlai18-nqs-tomography-amplitude-phase.md) (A)).
- Phase only becomes learnable when amplitude information feeds it
  ([048](048-yin20-phasen.md) (A), §2).

**What we take:** the phase needs a pathway built for it. In our order: analytic phase
conditionals or canonicalization first (architecture §7); then, if the network must learn it, a
phase-specific readout (a matched-filter phasor sum at the source's f₀) and a check of the
phase gradient's size against the rest of the loss. Complex weights are not on that list: no
paper shows them learning a phase a real network fed (Re, Im) cannot.

## 6. What complex parameters cost

- **Precision.** XLA has no bf16 complex dtype (complex64 is the smallest), so a complex
  backbone cannot use the bf16 core. The 3M (Gauss) product saves up to 25% of the arithmetic
  but its imaginary part loses relative accuracy after near-cancellation
  ([075](075-higham92-3m-complex-matmul-stability.md) (B)).
- **Compute.** A complex matmul is 3–4 real ones; complex models need 2–3× the MACs at equal
  parameters ([040](040-wu23-rethinking-complex-speech.md) (B)).
- **Gradients.** For a real loss, JAX's `grad` w.r.t. a complex leaf returns the conjugate of
  the steepest-ascent direction (PyTorch returns the direction itself), so an optax update
  moves the imaginary part the wrong way unless the gradient is conjugated first
  ([037](037-kreutzdelgado09-cr-calculus.md) (A)). We checked it: for |w − (3+5i)|² at w = 1+2i,
  `jax.grad` gives −4+6j against the ascent direction −4−6j.
- **Optimizers.** Adam on complex leaves keeps separate Re and Im second moments in PyTorch,
  not |g|² ([037](037-kreutzdelgado09-cr-calculus.md) (A)). Newton–Schulz on the realified
  matrix [[A, −B], [B, A]] equals the realified complex iteration (we checked, max error
  9e-8) provided both start from the complex Frobenius norm, which is √2 below the realified
  one; NorMuon row norms then come in equal pairs.
- **Stability.** Complex nets were "far more sensitive to initialisation", often failing to
  train with |z|² or deep tanh ([026](026-monning18-cvnn-real-classification.md) (B));
  holomorphic activations are unbounded and need care
  ([061](061-bukov21-two-real-nets-vs-holomorphic.md) (B)); a complex speech model was less
  robust to the learning rate ([040](040-wu23-rethinking-complex-speech.md) (B)).

## 7. What we do not use

- A complex-valued backbone: no parameter-matched gain in signal processing beyond the data-
  and size-limited regimes (§1), at 2–3× compute, no bf16 and a conjugated gradient (§6).
- Phase-aware complex activations. modReLU is worst in MRI and vision (§4); the cardioid
  halves the error of a real net on clean MR fingerprinting but loses to it on noisy T2
  ([024](024-virtue17-cardioid-mr-fingerprinting.md) (A)); kernel activations win only on small
  tasks ([029](029-scardapane18-complex-kernel-activations.md) (A)). With a real backbone the
  question does not arise.
- Fourier-domain mixing layers (FNO, GFNet) as a replacement for attention: they are complex
  only in the FFT, with real weights elsewhere, and target dense fields
  ([070](070-li21-fourier-neural-operator.md) (A), [071](071-rao21-global-filter-networks.md) (A)).
- A raw angle arg z as an input, or a Cartesian (Re, Im) encoding in place of the phasor
  (§2).

## 8. Order

1. Canonicalized data phase for φ± (architecture §7), with the phase gradient's size logged
   (§5).
2. Value rotation in the source←data attention (§3).
3. A phase-specific readout if 1 leaves the phase at the prior (§5).

## 9. Choices resting only on C evidence

- That value rotation helps on a 1-D axis beyond the one DiT comparison: RoVE
  ([004](004-garciacastellanos26-rotary-value-embeddings.md) (C)) is the only 1-D test. The
  mechanism itself rests on GTA (A).
- That coherent (phase-weighted) attention sums beat real attention: Holographic
  ([003](003-huang25-holographic-transformer.md) (C)) and the phase-coherent transformer
  ([006](006-hioki26-phase-coherent-transformer.md) (C)), the latter tied by its own real
  version.
- That phase is the cause of broader high-frequency LISA posteriors in a real NPE
  ([064](064-delmond26-lisa-gb-npe.md) (C)).
