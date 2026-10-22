# Consistency Trajectory Model: Learning Probability Flow ODE Trajectory of Diffusion

Kim et al., 2023, ICLR 2024 · [arXiv:2310.02279](https://arxiv.org/abs/2310.02279)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 505
citations in 3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Learn the anytime-to-anytime PF-ODE jump G(x_t, t, s) for s <= t, parameterised Euler-style as (s/t) x_t + (1 - s/t) g_theta(x_t, t, s). Then G(x, t, t) = x for free, and g(x, t, t) is the denoiser (score), so one network gives both jumps and scores. Training ("soft consistency") matches the student jump t to s against a teacher ODE solve t to u followed by a stop-grad student jump u to s, with random u in [s, t). The loss is measured after transporting both predictions to time 0 with the stop-grad student. A DSM loss on g(x, t, t) and a GAN loss are added, each weighted VQGAN-style by gradient-norm ratios at the last layer.

## What they find

- Headline 1-NFE FID: CIFAR-10 1.98 unconditional / 1.73 conditional (2-NFE 1.87 / 1.63); IN64 1.92 (2-NFE 1.73, recall 0.57 vs EDM 0.67). CIFAR NLL 2.43 bits/dim. Without a teacher (CTM's own score as teacher), CIFAR 1-NFE 2.39 vs CT 8.70.
- Training cost: 30k iterations on IN64 (~5% of CM/EDM iterations). Retraining CM with the official code gave 10.53, not the reported 3.55.
- Ablations (figures only): soft consistency is better than local and comparable to global. DSM is needed for good large-NFE quality. GAN helps at all NFE. gamma-sampling with gamma = 0 (deterministic) improves monotonically with NFE, while gamma = 1 (CM multistep) degrades with NFE. Thm 1 bounds the error growth by overlapping jumps to 0: O(sqrt(T + t1 + ... + tN)).

## Implementation details

s enters as an extra positional embedding added to the t-embedding. Reuse the teacher's c_skip/c_out so g(x, t, t) starts at the teacher denoiser. Large EMA mu for the stop-grad target stabilises the GAN. GAN warm-up with lambda = 0 first. gamma-sampling: denoise to sqrt(1 - gamma^2) t_{n+1}, then renoise to t_{n+1}; sample variance scales as gamma^2.

## Caveats

- The headline numbers rely on a GAN discriminator (StyleGAN-XL features), with lower recall than EDM.
- The FID gains are partly feature-space adversarial.
- Costly targets: teacher ODE solve plus three student passes.
