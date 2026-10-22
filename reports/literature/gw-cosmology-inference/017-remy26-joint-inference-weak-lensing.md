# Joint inference of weak lensing convergence map and cosmology with diffusion models

Remy, Chang & Willett, 2026, arXiv preprint, MNRAS-format · [arXiv:2606.31988](https://arxiv.org/abs/2606.31988)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.3 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Inputs and outputs share one token sequence:
  - 16 tokens for the 6 cosmological parameters;
  - shear-conditioning patch tokens;
  - 256 convergence-field patch tokens (128² × 5 bins, patch size 8).
- The network is a pixel-space JiT/DiT. It is trained with a rectified-flow interpolant x_t = t x1 + (1 − t) x0 and Gaussian base.
- It uses **x-prediction**: the network outputs a denoised x̂1, converted to a velocity (x̂1 − x_t)/max(1 − t, 0.05), and the v-loss is applied to that velocity (Li & He 2026, "JiT").

## What they find

- Calibration:
  - Joint posterior MIRA score 0.635 ± 0.017 (optimal 2/3, worst 1/2): mildly overconfident.
  - Marginal θ posterior MIRA 0.659 ± 0.014, within the theoretical error.
  - TARP on 500 observations × 500 samples is consistent with the diagonal.
- The θ marginals agree with NUTS. NUTS costs 2.18e6 gradient evaluations, which saturate tree depth 6 at every step: about 10× more than the 100k training simulations.
- Field checks:
  - χ²_red of samples against the data is 1.0024, compared with an expected 1.0000 ± 0.0049.
  - The aggregate power-spectrum error is 6.7% and the aggregate 1-point PDF error 5.2%.
  - Per-sample consistency test: each κ^(i) is compared with a fresh simulation at its own sampled θ^(i). This tests the joint posterior, not just the marginals.
- Cost: 0.2 s per sample (256 Heun steps). Training takes about 48 h on one GH200 (100k simulations, 750 epochs, batch 128).

## Implementation details

- Transformer: depth 12, d = 768, 12 heads, RMSNorm, SwiGLU, qk-norm.
- RoPE on the field and conditioning tokens only. The **cosmology tokens are excluded from RoPE and act as a register-like prefix**.
- adaLN-Zero on time.
- Parameter tokens: **the 6 parameters are projected into 16 tokens** (in-context class-conditioning trick from MAR/JiT, which adds capacity). These tokens get learned position embeddings and are averaged at the output into a separate linear head.
- Bottleneck (low-rank, b = 128) patch embedding and predictor.
- Initialisation: Xavier; **zero-init of both the adaLN layers and the final predictor**, so the network initially outputs x̂1 = x_t.
- Time sampling: logit-normal with (μ, σ) = (−0.8, 0.8), biased toward mid-to-high noise. Found beneficial.
- **Single unweighted MSE across field and parameters.** The authors explicitly warn that reweighting the θ and κ velocity components would no longer target the true joint velocity and would bias the joint posterior.
- Data: observation noise is re-drawn on the fly every step; flips and 90° rotations as augmentation.
- Optimisation: AdamW (0.9, 0.95), no weight decay, gradient clipping 1.0. Warmup then cosine decay, quoted as 5e-3 → 1e-4 (wording ambiguous). bf16. **EMA 0.999 used for all results.**
- Sampling: Heun, 256 steps, linear grid.
- The conditioning tokens can use coarser patches because noise kills small scales. Their output tokens are discarded.

## Caveats

- Log-normal toy simulator, flat sky, one patch.
- Mild overconfidence on the joint posterior.
- Internal inconsistency: the text says 256 conditioning tokens with p_cond = 8, but eq. 12 lists 64 conditioning tokens (528 in total).
- No ablations of x-prediction, the number of θ tokens, or the time distribution. These choices are justified by citation to Li & He.
- Must be retrained whenever the mask or noise model changes.
