# F46 (2026-10-16): three B runs on B-rope — template residual, Fisher loss, 3× longer

- Three full runs on top of B-rope to tell apart why F45's gaps remain (φ₀ never learned; loud
  sky, A, f₀ 16–106× too wide): weighting, extraction, or training length.

**Two readings:**
- **Weighting:** the loss does not see precision (F6), so the Mahalanobis loss (proposed by the
  user and the collaborator) should close the loud gap.
- **Extraction:** the angles live in the annual Doppler and antenna-pattern sidebands, 2 bins
  apart on our 2-year grid (2004.08464's fast/slow decomposition, a collaborator's point). For
  the worst sources (f₀ ≈ 0.1–0.2 mHz, modulation index 0.3–0.5 rad) they are differences of
  ~1/SNR between a few large adjacent bins inside one token. Time-frequency tokens cannot
  resolve the Doppler track there (N time slices give ~N/2-bin frequency resolution). The
  template residual (data minus the sources' signals at the current point) turns the remaining
  error into an O(1) signal (φ₀ as a phase rotation, the sky as a sideband mismatch).

All three keep everything in B-rope (F39 fixes, NorMuon, SNR order, reconstruction loss, RoPE,
16-bin tokens); no ablations, as the collaborator asked.

| config | added to B-rope | steps |
|---|---|---|
| `B-fd-template` | residual y − h(x_t) as 6 more data channels, in training and at every sampling step (`train.with_residual`, `LisaFlow.templates`) | 200k |
| `B-fd-fisher` | flow loss dᵀWd, W = (s² + (1−s)²) I + clip((1−s)² G, 100), G each source's Fisher information in flow coordinates at x_t from its own band (`train.fisher_metric`); normalised by the batch's mean trace; held-out curves stay Euclidean | 200k |
| `B-fd-rope-long` | nothing: 3× longer (warmup 12k); baseline for the two above and the training-length test | 600k |

**Details:**
- **The residual is an input:** x_t is part of the conditioning and the gradient stops there,
  so the learned posterior is still the true one. At early t the template is junk; the network,
  conditioned on t, can learn to ignore it.
- **The weight is unbiased:** W is the inverse variance of the target velocity under a Gaussian
  posterior of precision G + I along a straight path, depends on (x_t, s) only, never on the
  target, so the minimiser is still the conditional mean. The Fisher block is checked against
  the likelihood Hessian (`tests/lisa/test_templates_fisher.py`).
- **How they run:** `slurm/lisa-chain.sh <config> <links> --n_random 50` chains lisa.sbatch
  links (afterany, resuming), then `slurm/lisa-post.sbatch` (afterok): x and y scorecards on
  the 240 B sources, eval with MCMC on the corner pages, eval draws, MCMC tables. The B MCMC is
  cached on TREX (`eval-27e6d96219`).

**Readout** against B-rope (18) and each other: MCMC tables (φ₀; loud sky, A, f₀), scorecards,
loss.
- Template closes φ₀ and Fisher does not → extraction.
- Fisher closes the loud gap and B-long does not → weighting, not training length.
