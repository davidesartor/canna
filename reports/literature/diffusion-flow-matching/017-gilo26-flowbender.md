# FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows

Gilo, Elflein, Sobol, Litany, 2026, arXiv cs.CV; no venue · [arXiv:2606.20404](https://arxiv.org/abs/2606.20404)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.3 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Two passes per step. (1) Look-ahead with feedback input s=0 gives v_LA, then x̂₁ = x_t + (1−t)v_LA. (2) The feedback s_t is computed from the forward operator ℋ(x̂₁) vs y. Options are the first-order ∇L(ℋ(x̂₁), y), taken w.r.t. x_t or w.r.t. x̂₁ (the "shortcut" that skips the denoiser Jacobian), a zero-order measurement-space residual ℛ(ℋ(x̂₁), y) (works for non-differentiable ℋ like JPEG), or both. (3) The refinement pass v_ref = v_θ(x_t, t, c, s_t) is trained with plain FM MSE. s_t is a stop-gradient constant, so there is no backprop through the look-ahead or ℋ. Feedback dropout uses p_un. The gradient enters as extra input channels (zero-initialised input-conv weights) and the residual through the ControlNet conditioning path.

## What they find

SD3.5-Large plus ControlNet, Unsplash 1024², 40 Euler steps. Super-resolution PSNR: standard FT 34.35, FT + alignment loss 35.21, **zero-order 39.25 (FID 3.93 → 3.36)**, first-order w.r.t. x̂₁ 44.07 (but FID 6.45), FlowChef inference-time guidance 43.02 with **FID 18.96**. Depth→RGB MAE 0.0848 → 0.0747 (first-order w.r.t. x_t), where guidance blows FID up to 223. JPEG restoration PSNR 26.29 → 28.86 (zero-order). Mesh texturing (TRELLIS-2, LoRA r=128): M-PSNR 21.91 → 26.39. p_un=0.1 is best. Mechanism probe: only about 20% of the correction energy is along the gradient direction (cosine 0.42), so the network learns a non-linear correction policy, not a guidance step.

## Implementation details

Prior-step shortcut: for t > t_thresh reuse the previous step's x̂₁, which takes N+2 NFE down to N+1. It dips somewhat but still beats standard FT. Normalise the residual magnitude. Feedback dropout 0.1. Training cost is about +35–60% GPU-hours (38 h standard FT → 51 h zero-order, 61 h first-order).

## Caveats

Point-estimate or fidelity metrics only, with no posterior-calibration evaluation. Image and mesh tasks. Adding CFG on top still sometimes helps, so the policy is not complete.
