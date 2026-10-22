# Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning

Chen, Zhang, Hinton, 2022; ICLR 2023 per the citing paper 2410.22573 · [arXiv:2208.04202](https://arxiv.org/abs/2208.04202)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 544
citations in 4.2 yr (Semantic Scholar, 2026-10-19).

## What they do

Discrete data is encoded as ⌈log₂K⌉ real-valued "analog bits" scaled to {−b, b} and modelled by continuous diffusion, then thresholded at 0. **Self-conditioning:** the denoiser f(x_t, x̃₀, t) also receives its own previous x₀ estimate, concatenated. In training, 50% of the time x̃₀ = 0. Otherwise x̃₀ = sg(f(x_t, 0, t)) from an extra forward pass with no backprop, which costs under 25% more training time. At sampling, the previous step's x̂₀ is fed in at no extra cost. Also proposed: **asymmetric time intervals**, where sampling calls f(x_t, t+ξ) with a small ξ>0.

## What they find

Self-conditioning on continuous ImageNet-64, DPM-Solver (App. E, Table 8): ε-prediction FID 6.10 → **4.24** at 20 NFE and 5.58 → 4.15 at 30 NFE. x₀-prediction 12.13 → **6.94** at 20 NFE and 11.05 → 6.43 at 30 NFE. Discrete CIFAR-10 FID 3.48 unconditional and 2.72 class-conditional, against a continuous DDPM at 3.17/2.95. Captioning with 10 steps: BLEU-4 34.5 vs autoregressive 33.9. Asymmetric time at 10 steps: BLEU-4 17.6 → 34.6.

## Implementation details

Zeroing on 50% of steps keeps the model usable with no self-conditioning input (needed at the first sampling step). The optimal ξ shrinks toward 0 as the step count grows. Appendix variants include a momentum-averaged x̂₀ and "self-guidance".

## Caveats

Gains are largest at few NFE and with x₀-prediction. Self-conditioning is not isolated for captioning. Image FID only.
