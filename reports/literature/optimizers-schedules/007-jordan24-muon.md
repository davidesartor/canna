# Muon: an optimizer for the hidden layers of neural networks (Jordan et al., 2024)

[Blog post](https://kellerjordan.github.io/posts/muon/), not on arXiv. The base of the Muon
family, and the source of `optax.contrib.muon`.

**Evidence: A.** Blog post, neither peer-reviewed nor indexed, but reproduced at scale by
independent groups (Moonlight, NorMuon and the Muon papers in this folder).

## What they do

- **Hidden 2-D weights:** take the momentum of the gradient, then approximately orthogonalise
  it with 5 quintic Newton–Schulz iterations. The update is the matrix's polar factor, with
  every singular value near 1. That makes it steepest descent in the spectral norm.
- **Everything else goes to AdamW:** embeddings, the output head, gains and biases.
- **Defaults:** momentum 0.95 with Nesterov; Newton–Schulz in bf16.

## What they find

- Speed records on small-scale LLM and CIFAR-10 training benchmarks, against tuned AdamW.
- **The step size is fixed by the lr**, whatever the size of the gradient. Newton–Schulz is
  cheap enough to run every step.

## Limits

- A blog post, mostly on speedrun-scale benchmarks.
- No weight decay and no update-size matching. Moonlight adds both for scale.
