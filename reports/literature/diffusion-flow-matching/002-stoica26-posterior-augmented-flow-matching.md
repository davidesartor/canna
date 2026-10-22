# Posterior Augmented Flow Matching

Stoica et al., 2026, arXiv cs.CV · [arXiv:2605.00825](https://arxiv.org/abs/2605.00825)

**Evidence: C.** Not peer-reviewed; 1 citation in 0.5 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- **Rewritten objective.** The FM expectation over p(z,y) p_t(z_t|z) is rewritten as an expectation over p_t(z_t) p(y) p_t(z | z_t, y), with the same value (Thm 4.1). For a given intermediate point z_t, every target z compatible with (z_t, y) is a valid supervision signal.
- **SNIS regression target.** PAFM regresses f_θ(z_t | t, y) onto a self-normalised importance-sampling (SNIS) average over K candidate targets z^j, which always include the true z^i:
  target = Σ_j w_j v(z_t | z^j), with w_j ∝ p_t(z_t | z^j) p(y | z^j) [p(z^j)/q(z^j)].
  With a prior proposal (q = p) the weights reduce to path likelihood × condition likelihood. Here p_t(z_t|z^j) = N((1-t) z^j, t² I) and v = (z_t - z^j)/t.
- **What this amounts to.** A Rao-Blackwellised estimate of the marginal velocity E[v | z_t, y]. Thm 4.2 claims the gradient variance drops by a factor of the Kish ESS (≥ 1).

## What they find

ImageNet-256, 400k iterations, batch 256, with REPA, no CFG, 50 NFE.
- SiT-B/2, FM vs PAFM with nearest-neighbour candidates: FID 27.57 → 24.88 at K=16 and sFID 11.44 → 6.81. Other K: K=4 gives 25.47, K=8 gives 25.25, K=64 gives 25.45. K=16 is the optimum, which they read as a bias–variance trade-off.
- SiT-XL/2: FID 11.14 → 9.85.
- CC12M text-to-image with MMDiT: FID 10.37 → 9.45.
- Without REPA (SiT-B/2): 39.98 → 38.19 at K=64, a small gain.
- **Ablation on how candidates are chosen** (Tables 2–3):
  - Random-crop augmentations of the same image: FID 24.15 at crop scale 1.25 (the "3.4 FID" headline).
  - K=10 draws from the VAE moment: 25.15.
- Mini-batch gradient variance falls about 4x (0.80 → 0.22).
- Cost: throughput -6.6%, memory +0.4%.
- 2-D crescent toy (4-layer MLP, all 2000 points as candidates): the velocity-field MSE is lower and more stable than FM, spurious inter-mode samples largely disappear, and PAFM is more robust when N is only 100–1000 points.

## Implementation details

- Nearest-neighbour candidates come from FAISS, precomputed and restricted to the same class, so p(y|z^j) = 1.
- For text conditioning, they first shortlist the M=128 images closest to the caption by SigLIP2 embedding, then take the K=32 nearest latents. They drop p(y|z^j) as "approximately uniform".
- Path likelihood is computed unnormalised as exp(-||z_t - (1-t) z^j||² / 2t²).
- Too many candidates (large K, or crops that are too large) adds distant, noisy targets. Gains peak at a moderate K.

## Caveats

- The practical candidate sets (nearest neighbours, crops, VAE draws) are not prior draws, and the p/q proposal correction is dropped. So the "unbiased" theorem does not cover what they actually run. The crops and VAE draws are not even samples from p_data.
- The proof of Thm 4.2 treats the SNIS variance as Σ w_j² Var(g) with fixed weights. It ignores the bias of SNIS and the randomness of the weights.
- Image generation only. Gains are modest (about 1–3 FID). No posterior or calibration metrics.
- No study of how ESS behaves at small t, where the path likelihood is sharp.
- When candidates are other data points with a different condition y, the method is only as good as p(y|z^j), which must be known or approximated.
