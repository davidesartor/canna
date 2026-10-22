# The Road Less Scheduled

Defazio et al., 2024, NeurIPS 2024 oral; MLCommons AlgoPerf 2024 self-tuning winner · [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 213 citations in 2.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

- **Schedule-Free SGD:**
  - y_t = (1-β)z_t + βx_t
  - z_{t+1} = z_t - γ∇f(y_t)
  - x_{t+1} = (1-c_{t+1})x_t + c_{t+1}z_{t+1}, with c_{t+1} = 1/(t+1).
- β interpolates between Polyak-Ruppert averaging (β=0) and primal averaging (β=1). The x sequence implicitly simulates a linear-decay schedule that ends at the current step.
- **Theorem 2:** for arbitrary weights w_t (c_t = w_t/Σw) and arbitrary β_t ∈ [0, 1], E[F(x_T) - F*] ≤ Σ w_t⟨g_t, z_t - x*⟩/Σw_t. This is an online-to-batch conversion, and it is the licence for time-varying β_t and for power or learning-rate-weighted averages.
- **Weighting:** factorial-power weights x ∝ Σ t^{r̄} z_t (c = (r+1)/(t+r+1)) remain valid.

## What they find

- **Benchmarks:** 28 problems.
  - Beats cosine on CIFAR-10/100, SVHN, IWSLT14 and NanoGPT, and beats linear decay on fastMRI and DLRM.
  - Ties on ImageNet ResNet-50 and MAE fine-tuning.
  - On convex LibSVM problems it is faster than linear decay on 11/12 and more accurate on 6.
- **Polyak and primal averaging alone (App. H)** clearly underperform schedules. Polyak averaging diverges in test loss on NanoGPT.
- **β sensitivity:**
  - β = 0.9 works everywhere except NanoGPT, where the loss "started to increase rapidly"; β = 0.98 was stable there.
  - On ImageNet over 200 epochs, the best β (0.9) is the same at every duration.
- **Learning rate:** the optimal LR is larger than the scheduled baseline's (e.g. SF-SGD LR 10 vs cosine 0.2 on CIFAR-10). LR sensitivity is similar to the baselines.

## Implementation details

- **SF-AdamW (Alg. 1):**
  - γ_t = γ·sqrt(1-β2^t)·min(1, t/T_warmup), with bias correction folded into the LR;
  - z ← z - γ_t g/(sqrt(v)+ε) - γ_t λ **y_t**, i.e. weight decay at y;
  - c_{t+1} = γ_t²/Σ_{i≤t} γ_i² ("performance was greatly improved" by this warmup weighting);
  - no first-moment buffer: β1 is the interpolation β.
- **Decay placement:** y or z both possible. y "gives significantly better performance on some problems including ImageNet and NanoGPT".
- **Warmup** is still required (5%, or 2k steps on NanoGPT).
- **NanoGPT:** "Disabling gradient norm clipping is crucial for the Schedule-Free runs" (clip 0.0, SF lr 5e-3 vs baseline 5e-4, β 0.98, decay 0.05).
- **AlgoPerf entry:** lr 2.5e-3, 1-β1 = 0.1, β2 = 0.9955, wd 0.081, 2% warmup, "polynomial in c_t average 0.75" (r = 0.75).
- **BatchNorm:** statistics must be recomputed at x.
- **Memory:** same as the base optimizer (store z plus x or y).
- **Limitation stated by the authors:** LR and WD still need sweeping, and the optimal values differ from scheduled runs.

## Caveats

- Each run is short relative to modern LLM horizons; the long-horizon instabilities are reported later in 2605.19095 and 2605.23061.
- No Muon or spectral base optimizer.
- No diffusion or flow-matching tasks. The closest are the fastMRI U-Net and MAE.
- Decay-at-y is the default here, but the later papers show it fails for Muon over long horizons.
