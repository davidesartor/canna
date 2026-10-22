# Classifier-Free Guidance is a Predictor-Corrector

Bradley & Nakkiran, 2024, arXiv; venue not on page · [arXiv:2408.09000](https://arxiv.org/abs/2408.09000)

**Evidence: B.** Not peer-reviewed; 91 citations in 2.2 yr (Semantic Scholar, 2026-10-19).

## What they do

CFG does not sample the γ-powered distribution p(x|c)^γ p(x)^{1−γ}, and DDPM-CFG and DDIM-CFG sample different distributions from the same scores. In the SDE limit, CFG-DDPM equals a DDIM predictor step on the conditional plus a Langevin corrector step on the γ′-powered noisy distribution with **γ′ = 2γ−1** (predictor and corrector anneal along different paths). This is "PCG".

## What they find

- Closed form with p(x)=N(0,2) and p(x|c)=N(0,1). The γ-powered target is N(0, 2/(γ+1)). **CFG-DDIM gives N(0, 2^{1−γ})**, exponentially sharp in γ. CFG-DDPM gives N(0, (2−2^{2−2γ})/(2γ−1)) ≈ N(0, 2/(2γ−1)), about 2× the target variance at large γ. Both are sharper than the target for γ>1.
- Two-cluster mixture: both CFG variants are mean-shifted, and DDIM-CFG is also non-Gaussian.
- SDXL experiments are qualitative only. PCG with K=1 Langevin step and 100 steps ≈ CFG-DDPM with 200 steps, and more Langevin steps improve quality even at γ=1.

## Implementation details

Mainly an analysis tool. Langevin corrector steps are a separate knob from guidance, and since Langevin does not fully mix at K=1, changing K also changes the effective guidance.

## Caveats

The analysis covers only DDPM-CFG. The deterministic DDIM/ODE CFG (what flow-matching samplers use) has no clean SDE characterisation. Experiments are 1-D plus qualitative SDXL.
