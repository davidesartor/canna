# Understanding Hallucinations in Diffusion Models through Mode Interpolation

Aithal et al., 2024, NeurIPS 2024 · [arXiv:2406.09358](https://arxiv.org/abs/2406.09358)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 126 citations in 2.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

- Where data modes are disjoint, the true score has near-step-function jumps between them. The network learns a smooth approximation, so a non-zero fraction of samples land between the nearest modes ("mode interpolation" = hallucination).
- Hallucinated trajectories have high variance in the predicted x̂0 over the final reverse steps. That variance works as a detector.

## What they find

- 1-D mixture of 3 Gaussians (μ = 1, 2, 3; σ = 0.05; 50k training points; T = 1000; 100M samples drawn):
  - Interpolated mass is non-zero and decays with distance from the modes.
  - It shrinks with more training data and with larger mode separation: moving μ3 to 4 suppresses interpolation between μ2 and μ3.
- 2-D grid of 25 Gaussians (100k points): interpolation happens between nearest neighbours only. With the grid rotated into a diamond, there is none along the x-axis.
- Fig. 4: the learned score is a smoothed version of the true score, which has sharp jumps at small t.
- Simple Shapes (64×64, T = 1000): ~5% hallucinations, which grow exponentially under recursive self-training.
- Hands (ADM trained on 5k images at 128²): extra or missing fingers appear.
- Detector, variance of x̂0 over the last steps:
  - 1-D: removes 95% of hallucinations, keeps 98% of in-support samples.
  - 2-D: removes 96%, keeps 95%.
  - Shapes: removes 95%.
  - Hands: removes ~80%, keeps ~81%.
- Recursive training, 2-D grid: the modes merge into almost one by generation 20. Filtering with the detector beats random filtering, including on MNIST FID.

## Implementation details

use x̂0-trajectory variance over the last steps as a cheap out-of-support filter.

## Caveats

- Small DDPMs and unconditional toy problems.
- No fix for the smoothness itself is proposed; the paper only offers detection.
