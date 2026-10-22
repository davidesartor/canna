# Why Gradients Rapidly Increase Near the End of Training, AdamC (Defazio et al., 2025)

[arXiv:2506.02285](https://arxiv.org/abs/2506.02285), preprint.

**Evidence: B.** Not peer-reviewed; 22 citations in 1.4 yr (Semantic Scholar, 2026-10-19).

## What they do

- Explain the late-training rise of the gradient norm in layers followed by a normalisation.
- AdamC: scale the weight decay with lr/lr_max, i.e. decay ∝ γ².

## What they find

- Under a decaying lr, decay at a fixed λ makes the effective (angular) learning rate of
  normalised layers drift, which drives the gradient norm up at the end of training.
- AdamC removes the rise.

## Limits

- The mechanism needs an lr schedule. At a constant lr, decay ∝ γ² is decay ∝ γ with a
  rescaled λ.
