# Hierarchical subtraction with neural density estimators as a general solution to overlapping gravitational wave signals

Hu, 2025; arXiv preprint · [arXiv:2507.05209](https://arxiv.org/abs/2507.05209)

**Evidence: C.** Not peer-reviewed; 4 citations in 1.3 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Instead of a joint NPE over both signals, iterate: draw parameters of signal A from a *single-signal* NPE (public DINGO O1 model, unretrained), subtract the reconstructed waveform from the data, run the same NPE on the residual for signal B, subtract B from the original data, repeat. A Fisher-matrix argument (systematic shift Δθ^i = (F⁻¹)^{ij}(∂_j h | δH) from the unsubtracted residual δH) shows the bias contracts geometrically per iteration, so the sys/stat ratio in M_c vanishes after ~5 iterations even at Δt_c = 5–10 ms with similar masses and sky positions.

## What they find

Ensemble of N_en = 200 subtraction draws per iteration, 80 posterior samples each (16 000 final). DINGO ≈ 6 s per draw on an A100, ~20 min per iteration. Convergence = JSD between successive ensembles < 1e-4 nat; typical 9–16 iterations, most ≤ 15, hard cases > 20; several hours on 1 CPU + 1 GPU vs > 20 000 CPU-h for joint bilby PE. Four Δt_c = 5 ms cases vs bilby/dynesty joint ("OS") and single-signal ("SS") runs: early iterations are biased (d_L underestimated, wrong sky, spins overestimated mimicking precession) and the converged result lands on DINGO-SS, bilby-OS or in between. P-P on 64 BBH pairs (masses 10–80 M⊙, d_L ≤ 2000 Mpc, Δt_c ∈ {5,10,30,50} ms, 16 each, network SNR > 12) passes KS.

## Implementation details

(1) Likelihood-based resampling of the ensemble with *tempered* weights L^{1/T}, T annealed exponentially from 10 to 1; enforce a monotone increase of the ensemble-mean log L (retry up to 100 times, then fall back to a Metropolis-like accept with min(1, L^k/L^{k−2})). (2) DINGO samples are phase-marginalised (14-D), so the phase needed for subtraction is recovered by maximising the likelihood on a 100-point phase grid — a cheap analytic/brute-force treatment of one extrinsic angle. (3) No retraining at all: the single-source estimator is reused for N sources.

## Caveats

Greedy, not a true Gibbs sampler (GNPE/DINGO gives no density, so no exact conditional); the 200-member ensemble breaks one-to-one cross-source correlations; only 2 signals, few injections; converged posteriors still inherit DINGO's own inaccuracy (bilby-SS ≠ DINGO-SS) and no importance sampling is done.
