# Fast Inference on Astronomical Time Series with Trans-Dimensional Flow Matching Posterior Estimation

van der Meulen et al., 2026, arXiv preprint, MNRAS-format · [arXiv:2607.21134](https://arxiv.org/abs/2607.21134)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.2 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Pulses are double-sided exponentials (t0, A, r, s). Draw N ~ U{1..N_max} and set the amplitudes of components n > N to zero.
- A transformer has one token per component, and tokens n > N are masked from attention (teacher forcing with N_true).
- Each token concatenates [CNN(y) embedding, MLP(θ_n,τ), Fourier(τ)] in a 2:1:1 dimension ratio, giving more width to the data.
- A separate CNN+MLP classifier gives p(N|y), trained jointly with an unweighted sum of MSE and cross-entropy (the two are of similar scale).
- Inference: sample N ~ p(N|y), then integrate the FM ODE with N active tokens.

## What they find

- **Network configurations:**
  | N_max | Blocks | Heads | d_t | Params | Steps | Train time | 1000 samples |
  |---|---|---|---|---|---|---|---|
  | 5 | 6 | 4 | — | 4.6M | 300k | 5.3 h | 2 s |
  | 10 | 12 | 8 | — | 12.2M | 300k | 10.3 h | 5 s |
  | 20 | 12 | 8 | 256 | 12.2M | 600k | about 29 h | 10 s |
- **Single pulse:** C2ST against MCMC is 54.8%. FM takes 40 s for 25k samples; MCMC takes 30 s.
- **Five pulses:** C2ST is 92%; the FM posterior is wider. MCMC took 3.15 h, with autocorrelation length about 671.
- **Magnetar burst** (SGR J1550-5418): p(N|y) peaks at about 14, like trans-dimensional nested sampling (magnetron). NS has a tail out to 27 that is cut off by N_max = 20. Timing is 10 s against 10 min for about 1000 samples, so about 60× faster.
- **Classifier on simulations:** p(N|y) averaged over 2000 light curves peaks at N_true for N ∈ {1, 3, 5, 8}, and spreads out as N grows, because more components overlap.

## Implementation details

- **Order components chronologically** (sorted t0, with order-statistic priors) to kill label switching and multimodality.
- **Priors and windowing:**
  - Log-uniform priors on amplitude and rise time.
  - Peak times limited to [0.2, 0.8] of the window so that no component sits mostly outside it.
  - Real data is padded with simulated noise and downsampled to match K = 1000.
- **Training data on the fly**, so overfitting is not a concern. Dropout is 0.
- **Optimisation and hyperparameters:**
  - Time prior p(τ) ∝ τ^{4/3}, which puts more weight near τ = 1, following Dax+23.
  - σ_min = 1e-4; 200 Euler steps.
  - Batch 1024, gradient-norm clip at 1.
  - Adam, with a 500-step linear warm-up to 5e-4, cosine decay to 1e-6, and the last 500 steps held constant.
- **Standardisation:**
  - Standardise θ₁ per coordinate, using statistics from 10k prior draws.
  - Scale y by the mean amplitude (150).
- **Multi-scale CNN encoder:** four Conv1d layers; adaptive average pooling to length 16 at each scale; 1×1 projection to 64 channels; concatenate to 4096; MLP down to d_y.
- **Inference speed-up:** set N_max to the largest N with nonzero p(N|y), because attention cost is quadratic in the number of tokens.

## Caveats

- **Posteriors are too wide** relative to MCMC, and get wider with N. The authors call this "conservative" but have not checked coverage.
- **The classifier and the flow are independent,** so the flow never revisits N.
- **Fixed input length and background:** the background rate is fixed during training. Input length is fixed at K = 1000. Components outside [0.2, 0.8] are invisible to the network.
- **Model mismatch shows up as extra components:** FRB20190124F gets far more components than the 3 visible ones.
- **Masked tokens still cost compute** in training.
- No SBC, TARP or coverage tests.
