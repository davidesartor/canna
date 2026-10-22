# AMUSE: Anytime Muon with Stable Gradient Evaluation

Kim et al., 2026, arXiv preprint, KAIST/KRAFTON/SNU · [arXiv:2605.22432](https://arxiv.org/abs/2605.22432)

**Evidence: C.** Not peer-reviewed; 4 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- **SF-Muon (eq. 4):**
  - Y = (1-β)Z + βX
  - M = μM + (1-μ)∇L(Y)
  - Z ← (1-η_tλ)Z - η_t·s_NS·NS(M)
  - X ← (1-c)X + cZ
- **Diagnosis** (river-valley analysis; MNIST MLP plus a quadratic example):
  - orthogonalised updates are bulk-oriented, but the normalisation amplifies residual noise along the dominant (sharp) directions, so the iterate "bounces" off valley walls;
  - evaluating the gradient closer to x (larger β) shrinks the dominant component;
  - a small fixed β is better early and a large β better late.
- **AMUSE schedule:** β_t = β_init for t ≤ T0 (the warmup), then 1-β_t = (1-β_init)·[c_{t+1}(1-c_{T0+1})/(c_{T0+1}(1-c_{t+1}))]^ρ, which approximates (1-β_init)(T0/t)^ρ.
  - It needs no horizon and β → 1.
  - ρ = 0 gives fixed-β SF-Muon. ρ = 1 keeps the implicit averaging window of x over y constant: ω_{t+1} = c/((1-c)(1-β_t)+c).
  - ρ < 1 lets the window grow sub-linearly.
  - Design rule (C.3): t(1-β_t) must not shrink, or x's effective window collapses.

## What they find

- **Steps to reach tuned cosine Muon's final quality:** 1.51× fewer (720M Llama), 1.34× (ImageNet ResNet-50), 2.57× (ViT MAE fine-tuning).
- **LLMs (124M/720M/1.3B, seq 512, FineWeb):** lowest perplexity throughout. SF-AdamW struggles at large batch, AMUSE does not.
- **Images:** best on CIFAR-10/100, SVHN, ImageNet, ISIC U-Net and MAE.
- **124M sensitivity (Table 16), val ppl:**
  - β_init 0.4/0.5/0.6/0.7 → 28.81/28.70/28.61/28.73 (ρ = 0.8);
  - ρ 0.6/0.7/0.8/0.9 → 28.79/28.68/28.61/28.63;
  - every config beats cosine Muon (29.21).
- **ImageNet:** 77.03–77.24 vs cosine Muon 76.97. ρ = 0.9 falls to 76.89.
- **Ablations (F.6):**
  - fixed-β SF-Muon (best β = 0.95) stalls late;
  - capping β_t at 0.95 is worse, and the gap grows late (continuing to raise β matters);
  - **removing Muon momentum "substantially degrades" both AMUSE and SF-Muon**;
  - the β_t schedule alone on SF-AdamW hurts, but with Adam β1 = 0.9 it helps (the two only work together);
  - constant-LR Muon + EWA (0.99/0.999/0.9995) or + a 2k-step WSD decay improves Muon but never matches AMUSE;
  - post-hoc EWA or LR decay barely improves AMUSE itself.
- **Long runs (3.4×/4.5× Chinchilla, hyperparameters tuned at 16k):** AMUSE is best at almost every checkpoint.
- **Overhead:** 1.067× AdamW iteration time.

## Implementation details

- **Fixed:** μ = 0.95; β2 = 0.999 for non-Muon parameters; no LR decay.
- **Averaging:** c_{t+1} = η_t²/Σ η_i² with linear warmup η_t = η·min(1, t/T0).
- **Weight decay:** standard decoupled decay applied **to z** (C.4); decay at y is an unused option.
- **Selected settings:**
  - 124M: lr 0.01, warmup 6000, wd 0.05, clip 0.5, β_init 0.6, ρ 0.8, batch 256.
  - 720M: lr 0.01, warmup 2000, wd 0.1, clip 0.1, β_init 0.4, ρ 0.8, batch 1984.
  - 1.3B: same as 720M, batch 2048.
- **Non-matrix parameters** use SF-AdamW (LLM, MAE) or SF-SGD (images).
- **BatchNorm:** recompute statistics at x.

## Caveats

- Classification and LM only; no generative or diffusion training.
- Baseline hyperparameters are taken from a benchmark (Semenov et al.), while AMUSE's grid is small. The comparison is fair to the baselines but not exhaustive.
- Short sequences (512).
- The river-valley analysis is shown on small models.
