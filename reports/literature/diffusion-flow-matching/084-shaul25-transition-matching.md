# Transition Matching: Scalable and Flexible Generative Modeling

Shaul et al., 2025, arXiv preprint · [arXiv:2506.23589](https://arxiv.org/abs/2506.23589)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 11 citations in 1.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

- TM is a discrete-time, continuous-state generative paradigm. Over T steps it learns a stochastic Markov kernel p(X_{t+1} | X_t) from an arbitrary supervising process, which may be non-continuous.
- Difference TM (DTM) uses the linear FM interpolant. Per token, it predicts the difference Y = X_T − X_0 by sampling it from a small flow head (about 40M params) conditioned on the backbone's per-token features (1.7B DiT), then steps X_{t+1} = X_t + Y/T.
- The expected DTM step is exactly the FM Euler step, so FM is the mean-field (deterministic) limit of DTM.
- ARTM and FHTM add causal, autoregressive-over-tokens versions that use the "independent linear process" (fresh noise per step).

## What they find

- Setup: T2I on 350M Shutterstock pairs, SDXL-VAE latents 32×32×4, a 1.7B DiT, 500K iterations at batch 2048. All baselines were re-implemented on the same backbone and data.
- PartiPrompts:
  | metric | DTM (T = 32) | FM | MAR |
  |---|---|---|---|
  | CLIPScore | 26.84 | 25.97 | 26.98 |
  | PickScore | 21.18 | 21.04 | — |
  | ImageReward | 0.532 | 0.233 | 0.325 |
  | UnifiedReward | 5.123 | 4.784 | — |
  | Aesthetic | 5.421 | 5.291 | — |
- GenEval: DTM 0.54, FM 0.47, MAR 0.52.
- Sampling: DTM takes 1.6 s against FM's 10.8 s, i.e. 16 against 128 backbone forwards plus cheap head steps (about 7× faster).
- Ablations:
  - The independent linear process is far better than the dependent one for ARTM and FHTM.
  - Larger DTM patches (2×8×4) help in the 1–4 step regime.
  - DTM quality saturates by about T = 16–32 backbone steps.

## Implementation details

- Put a small per-token flow head on top of the big backbone, so most of the compute sits in a few backbone calls while the head runs its own inner ODE.
- The head is trained with plain FM on Y given the backbone features.
- The backbone runs only T times. Inner head steps are cheap.

## Caveats

- Within each step, tokens are sampled independently given X_t, so cross-token correlations of the step must come from many outer steps.
- Images only, with in-house data. No likelihood and no calibration evaluation.
