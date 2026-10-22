# SlotFlow: Amortized Trans-Dimensional Inference with Slot-Based Normalizing Flows

Houba, Giarda, Speri, 2025, arXiv v1 28 Nov 2025, astro-ph.IM · [arXiv:2511.23228](https://arxiv.org/abs/2511.23228)

**Evidence: C.** Not peer-reviewed; 4 citations in 0.9 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Amortised trans-dimensional NPE for a superposition of K ≤ 10 sinusoids. A dual-stream encoder gives a global context g; a classifier gives p(K | d); K slot contexts c_k = [g, one-hot(k)] condition one **shared** conditional flow, so q(θ_1..K | d, K) = Π_k q(θ_k | c_k) (factorised across slots given g). Label switching is handled by **Hungarian matching**: during training the per-slot NLL matrix (cost −log q(θ_j | c_k)) is solved for the best assignment, then the matched NLL is minimised.
x(t) = Σ A_k cos(2π f_k t + φ_k) + N(0, σ²), T = 300 s; f ~ U[2.5, 3.0] Hz with Δf_min = 0.01 Hz (= 2/T, enforced); A ~ U[0.5, 1.5]; φ ~ U[0, 2π]; σ ~ U[0, 1.5]. Per-source SNR from below 0 dB to ~40–60 dB, typical ~8 dB. Per-slot θ_k = [A, cos φ, sin φ, f] (phase as a unit-circle embedding, f centred at 2.7). Sinusoids only, no LISA response.

## What they find

- Training: 8×10⁶ train / 2×10⁶ val / 10⁴ test; Adam 1e-4, batch 128, ReduceLROnPlateau; 187 epochs ≈ 190 h on 4 GH200 (~305 GPU-h per the authors' accounting). Loss = CE(K) + matched NLL (+ optional noise term); teacher forcing on K during training. Target weights w_φ = 2, w_f = 3.
- Cardinality accuracy 99.85% (15 errors, all at K = 9–10).
- Calibration: coverage bias < 3%, mildly conservative.
- vs RJMCMC (Eryn, K = 3, chains started at the truth via Fisher): W₂ amplitude 0.0069 (std ratio 1.5–2×), phase 0.0227 (~2×), frequency 0.0006 Hz with **std ratio 2–3×** (σ_f 0.001 vs 0.0003 Hz). RJMCMC also shows amplitude–phase correlations that SlotFlow smooths out.
- Degradation with K: σ_A 2.2% → 4%, σ_φ 0.03 → 0.14 rad, σ_f 0.09 → 0.55 mHz from K = 1 to 10.
- Inference 13.5 ms on CPU vs 3–6 h RJMCMC.
- **Ablations:** time-only encoder → σ_f 71,600 mHz vs 10.2 mHz for the hybrid, K accuracy 90.8% vs 99.4%; FFT-only is 60% worse on phase (0.887 vs 0.554 rad). Phase weight sweep w_φ ∈ {1,2,5,10,20}: best at 2 (38% lower phase σ than 1); w_φ = 20 breaks frequency (381 mHz). Flow depth: L = 2 underfits, best 8–10; L = 12 gave no frequency gain; reduced conv stride helped only marginally. Early multi-scale + local-zoom tests roughly halve the frequency variance.

## Implementation details

Two encoder streams — long (300 s at 10 Hz, FFT Re/Im) and short (10 s at 512 Hz, time domain), each conv 2→32→64→128 (k8 s2, last k5 s1) + sinusoidal PE + 4-head MHA, then mean/max/attention pooling → g (256). The K classifier is on the FFT stream. Flow: 8 coupling layers alternating RQ-spline (48 bins, hidden 768, tails ±2) with affine layers; pure-spline stacks were unstable.
Hungarian-matched NLL; one-hot slot ID on a shared flow; (cos φ, sin φ) for phases; per-target loss weights; hybrid FFT + time encoders; affine layers interleaved to stabilise splines.

## Caveats

Frequency posteriors 2–3× too broad, blamed on an **encoder pooling bottleneck** (global g cannot carry sub-bin frequency); factorised slots cannot represent anti-correlations between sources closer than ~1/T; slots spontaneously specialise by frequency band (assignment consistency 1.00 across draws), i.e. the matching collapses to a frequency ordering anyway.
toy sinusoids, white noise, no LISA response or Doppler; RJMCMC reference only at K = 3 and initialised at truth; no p-p per K beyond aggregate coverage.
