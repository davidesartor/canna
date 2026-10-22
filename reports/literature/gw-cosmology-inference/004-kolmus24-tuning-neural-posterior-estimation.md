# Tuning neural posterior estimation for gravitational wave inference

Kolmus, Janquart et al., 2024, arXiv · [arXiv:2403.02443](https://arxiv.org/abs/2403.02443)

**Evidence: C.** Not peer-reviewed; 9 citations in 2.6 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Two separate points.
1. **Effective prior.** NPE accuracy at θ tracks the "sample exposure", ξ(θ) = expected number of training draws whose simulations closely match the simulation at θ (cosine similarity/match). A uninformative prior gives very uneven exposure, so pick the training prior to make exposure uniform in data space.
2. **Fine-tuning.** Amortised NPE for long, information-rich signals gives posteriors that look right but are too broad. After amortised training, freeze the context c = M(x_obs) and fine-tune only the flow on that event. Each cycle samples from R, computes the true unnormalised posterior, and minimises −w² log R with w = p(x|θ)p(θ)/R(θ) (no gradient through w). Each simulated batch is reused ~10 times.

## What they find

- Toy problem (4 coupled oscillators, 10-D, 2 s at 128 Hz):
  - Efficiency vs mass follows exposure. A uniform mass prior starves small masses; power-law α = −1.5 or −3 flattens it.
  - Training 100 → 200 → 400 epochs (5000 steps × batch 1024 each) helps every prior, but uniform-prior small masses stay poor.
  - 20 s signals (10× less exposure), with a 10× wider context and 10× longer training: ε = 0.02% after 100 epochs and 0.07% after 1000, versus 20–30% for 2 s signals. That is despite ~5×10⁹ unique samples, and the posteriors are too wide.
  - Fine-tuning (10 cycles, batch 10240, 10 repeats, 5 s) lifts 0.1% → 3.5%.
- Appendix B, conditioning type for the coupling MLPs: concatenate 39.1%, bias 38.9%, **scale 47.5%**, sigmoid 41.2% after fine-tuning. Training losses are similar. Scale conditioning is best for fine-tuning.
- Identity-initialised, unconditioned coupling layers added before fine-tuning: 0/1/2/4/8 extra layers give 2.1/4.4/7.9/8.3/7.9%.
- GW (aligned-spin BBH, Mc 5–100, HLV design PSD):
  - Fine-tuning takes ≤10 min on an RTX 3090 and reaches ε = 51.2% on a multimodal low-mass event.
  - JAX RADFRIENDS nested sampling takes >3 days, ~400× slower.
  - 500 injections: 14 have ε < 5%. Eleven are single huge-weight outliers; three needed 10 extra cycles.

## Implementation details

- **Exposure-matched priors:** Mc ~ power law α = −3 on [5, 100] and q ~ α = −1.5 on [0.2, 1]. Motivation: the average match between Mc = 5.000 and 5.025 equals the match between 60 and 90. The test is to plot exposure ξ(θ) for candidate priors before training (1000 inverse-CDF points). The posterior is then reweighted back to the physical prior through IS.
- **SNR, not distance:** sample the optimal SNR ~ U[10, 30] and rescale the distance to match it. This guarantees that training signals are detectable.
- **Waveform reuse:** generate 64 waveforms per batch and copy each 16× with new sky, ψ, SNR and t_c (extrinsic parameters are cheap to project). Generate on the GPU with ripple, building fresh batches each step to prevent overfitting.
- **Reduced-order basis:** eigendecompose the per-detector covariance matrix over 5×10⁶ strains, which takes constant memory unlike SVD. Keep 768 eigenvectors (min match 0.95). This acts as denoising and speeds convergence.
- Train on a band-limited 20–256 Hz input, and fine-tune with the full 20–2048 Hz likelihood: 10 cycles × 10⁵ at the low band, then 10 cycles × 5×10⁴ at the full band.
- **Mode symmetrisation during fine-tuning:** each sample gets 3 copies shifted by π in φ_c and/or π/2 in ψ. All four get the averaged likelihood, so no mode in the ψ–φ plane is missed.
- Base distribution is a truncated standard normal, matching bounded parameters such as phase.
- Flow: Bernstein-polynomial couplings (degree 128), 4 coupling layers, shallow MLPs. Shallow but expressive, and faster to fine-tune than RQ-splines.
- Weight-normalised linear layers in the context ResNet "significantly improved convergence". Adam at lr 1e-2 for 90% of training, then 1e-3.
- Stop IS at 10⁵ samples or n_eff = 5000.
- Sky in Earth-fixed polar coordinates removes the GMST dependence.

## Caveats

- Needs log q, which is easy for discrete flows but expensive for FM (ODE trace) unless the flow is distilled.
- The w² loss has high variance when the proposal is poor. The 3 non-converged cases and the single-outlier cases show it.
- Only aligned-spin; one network; design PSD.
- The exposure measure is defined along one parameter at a time.
