# ZClip: Adaptive Spike Mitigation for LLM Pre-Training (Kumar et al., 2025)

[arXiv:2504.02507](https://arxiv.org/abs/2504.02507), preprint.

**Evidence: C.** Not peer-reviewed; 11 citations in 1.5 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Track the gradient norm's running mean and variance and clip only when its z-score is
  anomalous.

## What they find

- Loss spikes are removed without clipping the steady state, unlike a fixed threshold that is
  either always or never active.
