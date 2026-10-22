# Fantastic Pretraining Optimizers and Where to Find Them (Wen et al., 2025)

[arXiv:2509.02046](https://arxiv.org/abs/2509.02046), preprint.

**Evidence: A.** Not peer-reviewed; 101 citations in 1.1 yr (Semantic Scholar, 2026-10-19).

## What they do

- Tune ten optimizers carefully at 0.1–1.2B parameters.

## What they find

- Claimed speedups shrink against a well-tuned AdamW.
- Muon and SOAP are best, but their 1.4× at 0.1B falls to 1.1× at 1.2B.
- **Rankings flip mid-run because of the lr decay**, so optimizers must be compared after
  their cooldowns.

## Limits

- LLMs only.
