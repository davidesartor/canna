# Rapid parameter estimation for pulsar-timing-array datasets with variational inference and normalizing flows

Vallisneri et al., 2024 · [arXiv:2405.08857](https://arxiv.org/abs/2405.08857)

**Evidence: B.** Peer-reviewed (Physical Review Letters); 14 citations in 2.4 yr (Semantic
Scholar, 2026-10-19).

## What they do

Non-amortised, likelihood-based stochastic-gradient VI: train a neural spline flow q_φ(θ) for *one* dataset by minimising reverse KL(q‖p(θ|y)) with the reparametrisation trick, ∇_φ E_{x∼q0}[log q0(x) − log|∂f/∂x| − log p(f_φ(x)|y)]. Requires a differentiable, batchable likelihood: the Enterprise PTA likelihood was reimplemented in JAX (flowjax for the flow). Evidence via IS with the flow as proposal.

## What they find

Simulated NANOGrav-15yr-like data, 67 pulsars, ~675k residuals, 136 parameters (CURN and HD models). NSF with 16 flow steps, 8 bins, ~365k params. CURN: batch 3,584 base draws, 0.4 s/iteration on A100 in float64, converged at ~2,200 iterations (~15 min). HD: batch 256 (memory-bound), 1.2 s/iter, ~1,200 iterations (~25 min) plus several minutes of JIT compile. 1-D Hellinger distance vs NUTS (NumPyro, same JAX likelihood, 16,384 samples): ~0.01 for most parameters, 0.002 for the GW parameters — comparable to HMC-subsample-vs-subsample distance; the flow recovers tails for a "difficult" pulsar.

## Implementation details

(i) Double precision throughout. (ii) Fresh base draws every batch, so no overfitting. (iii) Suggested (not used) annealing: multiply log p by a factor ramping 0→1 over the first few hundred iterations to soften the target. (iv) Accuracy checks without a reference sampler: loss convergence, seed-independence, capacity sweeps (vary layer count/size), and IS reweighting where accurate q gives small weight variance/high ESS. (v) Proposes transfer learning (initialise HD flow from converged CURN flow).

## Caveats

Short letter; no quantitative capacity/convergence study (deferred). Reverse KL is mode-seeking — the paper's posteriors are unimodal power-law noise posteriors, so multimodality is not tested. Simulated data only. No ablation of annealing.
