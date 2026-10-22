# On the Importance of Noise Scheduling for Diffusion Models

Ting Chen, 2023, arXiv tech report · [arXiv:2301.10972](https://arxiv.org/abs/2301.10972)

**Evidence: A.** Not peer-reviewed; 245 citations in 3.7 yr (Semantic Scholar, 2026-10-19).

## What they do

x_t=√γ(t)·b·x₀+√(1−γ(t))·ε with fixed γ(t)=1−t and an input scale b<1 shifts the log-SNR curve vertically without changing its shape, unlike cosine/sigmoid temperature schedules which mostly change it near t→1. Normalise x_t to unit variance per example before feeding the network (acts as the first layer; also used at sampling).

## What they find

Table 3 (scale fixed to 1): best schedule differs per resolution — 64²: 1−t 2.04; 128²: sigmoid(0,3,τ=0.7) 3.91; 256²: sigmoid(0,3,τ=0.5) 4.28 vs plain cosine 21.6 and 1−t 7.21. Table 4 (input scaling, γ=1−t): 64² best b=1 (2.04) to 0.9 (2.23); 128² best b=0.6 (3.50) vs b=1 4.51; 256² best b=0.4 (3.52) vs b=1 7.21 (≈2× FID). Scaling beats the best schedule-function result at 256² (4.28 → 3.52) and has one knob. Chosen b: 1.0 (64²), 0.6 (128²), 0.5 (256²), 0.2 (512²), 0.1 (768², 1024²). Results are noisy and non-monotonic in b (e.g. 128², 1−t: b=0.7 5.34, 0.6 3.50, 0.9 3.86).

## Implementation details

RIN backbone, LAMB, self-conditioning 0.9, EMA 0.9999, 1000-step DDPM sampling; at inference the standard cosine schedule works fine even if trained with 1−t (continuous t decouples train and sample schedules).

## Caveats

Single seed, ImageNet only, smaller/shorter models than RIN; FID tables noisy enough that differences <~1 FID are not reliable.
