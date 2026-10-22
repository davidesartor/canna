# Improving the Training of Rectified Flows

Lee, Lin, Fanti; CMU/MSR; arXiv v2 Oct 2024, NeurIPS 2024 · [arXiv:2405.20320](https://arxiv.org/abs/2405.20320)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 121 citations in 2.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

- Argument that one Reflow round suffices: an intersection at time t needs z″ = z′ + ((1−t)/t)(x′−x″), and such a z″ is an atypical Gaussian draw (wrong norm, high autocorrelation) that the 1-RF never maps onto the data manifold. So the optimal 2-RF is nearly straight, its loss lower bound ≈ 0, and the observed loss directly measures the reducible error per t.
- Loss L(θ,t) = (1/t²)E‖x−x_θ‖² = (1/t²)E‖x−E[x|x_t]‖² + L̄. Here x_t = (1−t)x + t z, with t=0 data. v-prediction equals x-prediction with ω(t) = 1/t².
- **U-shaped density:** p_t(u) ∝ e^{au} + e^{−au} on [0,1], with a = 4, chosen because the 2-RF loss is high at both ends (Fig 3).
- **Losses:**
  - Pseudo-Huber: m = √(‖Δ‖² + c²) − c, with c = 0.00054·d as extracted (iCT uses 0.00054√d; the HTML may have lost the root).
  - LPIPS-Huber: (1−t)·m_hub + LPIPS(x, x_t − t v_θ).
  - LPIPS-Huber-1/t: the LPIPS term is multiplied by 1/t.
- **Diffusion-to-RF conversion (Table 2):**
  - VE: t_VE = t/(1−t), s_VE = 1/(1−t).
  - VP: t_VP = (1/9.95)(−0.05 + √(0.0025 − 19.9 ln((1−t)/√((1−t)²+t²)))), s_VP = α(t_VP)/(1−t).

## What they find

| Config | CIFAR-10 | AFHQ-64 | FFHQ-64 |
|---|---|---|---|
| (A) vanilla 2-RF | 12.21 | – | – |
| (B) + EDM init, batch 128→512 | 7.14 / 3.61 | 12.39 / 4.16 | 8.84 / 4.79 |
| (C) + U-shaped p_t | 5.17 / 3.37 (−28%) | 9.03 / 3.61 (−27%) | 6.81 / 4.66 (−23%) |
| (D) + pseudo-Huber | 5.24 / 3.34 | 8.20 / 3.55 | 7.06 / 4.79 |
| (E) + LPIPS-Huber | 3.42 / 2.95 | 4.13 / 3.15 | 5.21 / 4.26 |
| (F) + LPIPS-Huber-1/t | 3.38 / 2.76 | 4.11 / 3.12 | 5.65 / 4.41 |
| (G) + real-data fine-tune | 3.07 / 2.40 | – | – |
- Huber alone is only better at small batch (128) or on AFHQ; it is a wash or worse on CIFAR and FFHQ at batch 512.
- ImageNet-64: 4.31 at 1 NFE, 3.64 at 2 NFE.
- Reflow costs 1828.6M forward passes vs 5734.4M for CD and 2867.2M for CT.
- There is no ablation over a.

## Implementation details

- Place time density where the *reducible* loss is high. Raw loss is a valid proxy only when the irreducible part ≈ 0.
- Initialise from a pre-trained model via the time/scale conversions.
- Use a large batch.
- Weight the perceptual loss by 1/t so its gradient does not vanish near the data end.

## Caveats

- Every result is for reflow on synthetic pairs, i.e. distillation-like.
- LPIPS is image-specific and adds about 15% per-iteration cost.
- Only the U-shape with a=4 is tested.
- A non-ℓ₂ premetric moves the optimum away from E[x|x_t] unless the coupling is deterministic.
