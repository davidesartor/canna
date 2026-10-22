# GenSBI: Generative Methods for Simulation-Based Inference in JAX

Amerio, 2026, preprint · [arXiv:2605.27499](https://arxiv.org/abs/2605.27499)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

A JAX (Flax nnx + diffrax + NumPyro + Orbax) library implementing flow matching (OT/rectified affine path, ODE solve t=0 noise → t=1 data), score matching (VP/VE SDE) and EDM, decoupled from three transformer backbones: SimFormer (one token per scalar, value+ID+condition-mask embeddings), Flux1 (Flux.1-style: double-stream blocks with joint attention between a θ_t stream and a conditioning stream, then single-stream blocks over the concatenation; adaLN-Zero time modulation; learned/sinusoidal/RoPE 1D/2D ID embeddings, combined with values by sum or concat) and Flux1Joint (single-stream Flux blocks + SimFormer condition mask for joint/any-conditional estimation). Diagnostics: SBC, TARP (with Jeffreys CIs), LC2ST, marginal coverage. Exact log q via divergence ODE (exact autodiff trace O(d) or Hutchinson).

## What they find

SBIBM C2ST at 10^6 sims, Flux1Joint+FM+EMA: Gaussian Linear 0.507, Gaussian Mixture 0.501, Two Moons 0.504, SLCP 0.549, Bernoulli GLM 0.557. Isolating architecture: SLCP Flux1 (conditional) 0.689 → Flux1Joint (joint) 0.549. At 10^5 sims best GenSBI vs literature on SLCP: 0.534 vs SimFormer 0.566 vs sbi NPE 0.742; Two Moons 0.502 vs 0.505. FM/SM/EDM converge to similar C2ST as budget grows; EDM much worse at 10^4 sims with Flux1Joint (0.825 Gaussian Mixture, 0.871 SLCP). Training cost identical across methods (≈4.6 it/s Flux1, ≈11.5 it/s Flux1Joint on V100, batch 256); sampling: FM 100 Euler steps 6.9 s for 10^4 samples, EDM 18 Heun steps, SM ~1000 steps. TARP on the toy GW (2 masses from 2×8192 strain, 1D-CNN embedding trained end to end) and toy lensing tasks lies near the diagonal.

## Implementation details

AdamW, cosine LR with linear warmup (peak 1e-4 to 4e-4, min = peak/100, warmup 500–1000), EMA 0.999–0.9999, early stopping on val-loss ratio; they stress "transformer FM models benefit from long training even after loss looks converged", default 5–10×10^5 steps. Typical Flux1: 8–16 single-stream + 4–8 double-stream blocks, 4–6 heads, tiny value embeddings (8–20 dims), ID merge "concat" preferred for small models (summing positional and value signals into a low dim space interferes). Euler with 100–200 steps suffices for near-straight FM paths; recommend stochastic FM samplers (ZeroEnds g=α√(t(1−t)), NonSingular g=α√(1−t)) when samples are under-dispersed. SBC caveat stated explicitly: a prior-returning posterior passes SBC.

## Caveats

Library paper; benchmarks are dim(θ) ≤ 10, GW/lensing tasks are 2-parameter toys without reference posteriors (TARP only). Hyperparameters are actually re-tuned per task/budget (tables A.4), despite "near-uniform" claim. No ablation of RoPE vs other ID embeddings, no study of narrow posteriors, no manifold support (all Euclidean, Gaussian base).
