# The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm (Amsel et al., 2025)

[arXiv:2505.16932](https://arxiv.org/abs/2505.16932), ICLR 2026.

**Evidence: A.** Peer-reviewed; 122 citations in 1.4 yr (Semantic Scholar, 2026-10-19).

## What they do

- Replace Muon's fixed quintic Newton–Schulz polynomial with per-iteration coefficients that
  are minimax-optimal for approximating the matrix sign function.
- Stabilise the iteration for bf16.

## What they find

- The output is closer to orthogonal for the same number of steps.
- Consistent validation-loss gains for GPT-2 trained with Muon.

## Limits

- LLMs only. A drop-in change of the NS coefficients, so it is cheap to test anywhere.
