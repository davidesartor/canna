# Optimizers for Diffusion Models: A Controlled Benchmark (Bolatov et al., 2026)

[arXiv:2609.23055](https://arxiv.org/abs/2609.23055), preprint.

**Evidence: C.** Not peer-reviewed; 0 citations in 0.1 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Seven optimizers, among them AdamW, Muon, SOAP, MARS-M, Lion and SF-AdamW.
- Four tasks (text8, LM1B, QM9, CelebA) across masked, uniform and Gaussian (DDPM)
  diffusion.
- Each optimizer is tuned per task, with 3 seeds.
- Score the denoising loss and, on CelebA, FID.

## What they find

- **SF-AdamW finishes last on all four tasks.**
  - It leads mid-run: on text8 between steps 7.5k and 25.5k, and on QM9 between 3.75k and
    21.25k. It falls behind once the scheduled optimizers decay their lr.
  - On CelebA it has the third-lowest loss but the worst FID: 15.3, against 6.9 for AdamW.
- **Loss barely separates optimizers, but samples do.** The denoising loss varies by 1.8%
  across optimizers, while FID spans 6.3–15.3.
- **Rankings do not transfer across tasks** (Kendall τ from −0.14 to 0.71).
- **Other optimizers:**
  - Muon, SOAP and MARS-M tie or beat AdamW on 2 of the 4 tasks. None has a significant gain
    on CelebA FID; MARS-M has the best mean.
  - Lion diverged on LM1B.
- The authors list ScheduleFree+ and SF-Muon as untested.

## Limits

- Small models and short runs.
- Only one task has a sample metric (FID).
