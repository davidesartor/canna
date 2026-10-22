# BilbyFlow: user-friendly neural posterior estimation for gravitational-wave astronomy

Pinchbeck, Thrane et al., 2026, arXiv · [arXiv:2609.00766](https://arxiv.org/abs/2609.00766)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.1 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

NPE inside Bilby. A conditional NSF (12 parameters) uses a ResNet-18 embedding of the whitened H1–L1 strain in both the time and frequency domains (512-D), plus an MLP embedding of the log-PSD (64-D). Every result goes through IS with a likelihood that is numerically marginalised over (φc, ψ, t_c).

## What they find

- **Training:** ~1e6 waveforms × 1e4 sky positions × 1e5 noise realisations, recombined at random. Training takes 96 h on one A100.
- **Efficiency on simulated injections (256 events):** 57% reach ε ≥ 1%. The median ε is 1.49% and the worst is 0.01%. Gaussian noise and off-source noise give closely similar efficiencies.
- **Efficiency on GWTC-3 (38 events):** median 7%, the worst 0.11%. 10⁴ posterior samples take 3 min to 35 h on 16 cores, with a median of ≈17 min. GW200216 took 30 min, against 4 h for nested sampling.
- **Rejection sampling** is ~10× less efficient than IS, so the authors argue downstream codes should accept weighted samples.
- **P–P tests:** reweighted P–P curves are good to ~3σ in Gaussian noise. They show tail over-confidence in off-source noise, which the authors attribute to Gaussian-noise misspecification. Raw-proposal P–P curves are good to ~2σ, with tail over-confidence.
- **Why ε is low:** no single parameter predicts low ε. There is only a mild trend to lower ε at low Mc and high SNR. The authors read this as "uniformly under-trained", with data diversity rather than compute as the bottleneck; longer training already shows over-training.

## Implementation details

- **Nuisance extrinsics out of the flow:** φc, ψ and t_c are randomised in training but are not flow outputs. This is an implicit marginalisation, offered as the alternative to GNPE. The IS numerator is then the marginal likelihood over (φc, ψ, t_c):
  - t_c is handled by FFT plus Bilby's jitter.
  - φc uses a multi-harmonic DFT from 5 phase-sampled waveforms (m = 1..4, with an anti-aliasing check), following DINGO's synthetic phase.
  - ψ and t_c use grids with local refinement.
- **Reshaped training priors corrected by IS:**
  - d_L is drawn log-uniform over [100, 5000] Mpc, padded to 5200, instead of ∝ d_L². This enriches loud and nearby sources, "where the posterior is most narrow and diverges most from the prior".
  - Mc is optionally drawn log-uniform, padded below the physical bound.
  - A prior-swap resampling stage (10× oversampling, ESS 30–80%) comes before the likelihood IS.
- **Distance curriculum:** stages with increasing d_L^max, so the loud, information-rich signals are learned first and the noise floor is introduced gradually. Each stage has its own lr and cosine schedule. The θ standardisation is computed once over the full prior and held fixed, to avoid shocks at stage boundaries.
- **Auxiliary regression head:** a 3-layer MLP of width 256 on the embedding predicts 14 summaries of the noiseless signal: per-detector SNR, peak phase, peak time, spectral centroid, Δt_HL, SNR ratio and phase difference.
  - λ_aux ≈ 0.5, annealed to 0 within each stage.
  - Its purpose is to stop the embedding collapsing onto a low-rank subspace while the NLL gradient is still weak.
- **Data-constrained sky coordinates:** sky is parametrised by (Δt_HL, φ_det) instead of (α, δ), with the analytic Jacobian handled in the weights.
- **Bounded parameters:** sigmoid/scaled-logit to unbounded coordinates. All coordinates are standardised from one representative batch.
- **Combinatorial augmentation:** independent banks of intrinsic waveforms, sky positions and noise draws are recombined at random.
- **PSD conditioning:** conditions on the estimated PSD, which gives slightly broader posteriors that are less efficient but unbiased.
- **Proposed future work:** flow matching, transformer or JEPA embeddings, per-event fine-tuning (as few as ~10 steps; IS keeps it exact) and CUDA waveforms.

## Caveats

- Restricted to Mc ≳ 10 M⊙, 4 s segments and two detectors.
- Efficiency on synthetic data is far below that on real events. The authors attribute this to a selection effect: inclination/q cuts close the gap, giving a 4–10× improvement.
- Raw-proposal tails are over-confident, which is the main IS risk.
- The 76% figure (events with ε > 1%) is the optimistic one.
