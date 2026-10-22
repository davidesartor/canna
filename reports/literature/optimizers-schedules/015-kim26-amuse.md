# AMUSE: Anytime Muon with Stable Gradient Evaluation (Kim et al., 2026)

[arXiv:2605.22432](https://arxiv.org/abs/2605.22432), preprint.

**Evidence: C.** Not peer-reviewed; 4 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Analyse Muon in a river-valley loss landscape.
- Propose an anytime Muon in the schedule-free family. The gradient-evaluation point moves
  over training from the fast Muon sequence toward the averaged one. In schedule-free terms,
  that is a ramp of the interpolation β that starts low (0.4–0.6).

- They write SF-Muon with Muon's own momentum kept inside the schedule-free wrapper (Eq. 4).

## What they find

- A fixed interpolation β trades early speed against late stability. Moving it towards 1
  over time gets both.

- Muon speeds progress along the flat "river" but amplifies oscillation in high-curvature
  directions across it.
- Evaluating the gradient closer to the average damps that oscillation. They test it on LLMs
  and ResNets.

## Limits

- Muon only, with LLM and image-classification experiments. No diffusion experiments.
