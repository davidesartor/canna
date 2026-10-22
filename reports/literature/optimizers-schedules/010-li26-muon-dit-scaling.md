# Scaling Muon for Diffusion Transformers (Li et al., 2026)

[arXiv:2608.20818](https://arxiv.org/abs/2608.20818), preprint.

**Evidence: C.** Not peer-reviewed; 0 citations in 0.2 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Train MMDiT-style diffusion transformers from 1.3B to 15B parameters with Muon and with
  AdamW.
- Propose **Periodic Row-wise Muon**: a full Newton–Schulz step every K steps, and a cheaper
  row-normalised step in between.

## What they find

- Muon beats AdamW at every scale, on both loss and generation quality. The best quality
  metric (FD-DINO) improves by 12.9–19.1%.
- Periodic Row-wise Muon cuts optimizer time by about 50% and keeps Muon's quality.

## Limits

- Text-to-image generation only.
- No schedule-free experiments.
