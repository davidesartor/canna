# Compact Binary Coalescence Gravitational Wave Signals Counting and Separation

Zhao et al., 2024; PRD 111, 104028 (2025) · [arXiv:2412.18259](https://arxiv.org/abs/2412.18259)

**Evidence: B.** Peer-reviewed (Physical Review D); 2 citations in 1.8 yr (Semantic Scholar,
2026-10-19).

## What they do

UnMixFormer: CNN encoder → dual-path transformer (intra-chunk / inter-chunk attention, DPRNN-style) whose feed-forward blocks are Fourier Analysis Networks (FAN: φ(x) = [cos(W_p x) ‖ sin(W_p x) ‖ σ(B + W x)]) → a counting head (classification over K ∈ {2..5}) that selects one of several decoder heads, each emitting K separated waveforms. Loss L = λ·L_SI-SNR + (1−λ)·L_CE with a *permutation-invariant* SI-SNR (best permutation of outputs vs truths).

## What they find

Single Cosmic-Explorer 40 km detector, 1 s at 16 384 Hz, 2–5 signals mixing BBH (SEOBNRv4), NSBH (IMRPhenomT), BNS (TaylorF2), aligned spins, per-signal SNR U(10,50), t_c U(0.25,0.75) s, whitened, Tukey α = 1/8. Only 100k training samples (25k per count), 20k test. Counting accuracy 99.89 %, AUC 1.00000/0.99999/0.99993/0.99991 for 2–5; all miscounts are ±1. Mean overlap of separated waveforms 0.9831, > 0.99 for SNR > 20; at SNR 10 some overlaps are 0 (overcounting); single-signal denoising overlap 0.9969. Generalises to precession, eccentricity e₀ ≤ 0.5 and higher modes (SEOBNRE) without retraining.

## Implementation details

Adam lr 1e-3, no wd, chunk 64, hidden 256, 8 heads, 2 layers, grad clip 5, batch 16, 100 epochs, 47 h on one RTX 4090. Count head chooses the decoder (variable-cardinality output without padding). PIT loss handles label switching at training time.

## Caveats

No ablation of FAN or dual-path design; no posteriors, no uncertainty; single detector; tiny dataset; outputs re-sorted by overlap for display.
