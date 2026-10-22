# Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models

Lu & Song, 2024, ICLR 2025 · [arXiv:2410.11081](https://arxiv.org/abs/2410.11081)

**Evidence: A.** Peer-reviewed; 304 citations in 2.0 yr (Semantic Scholar, 2026-10-19).

## What they do

TrigFlow: x_t = cos(t) x0 + sin(t) z, z ~ N(0, sigma_d^2 I), t in [0, pi/2]; this is FM/v-prediction that also satisfies EDM preconditioning (c_skip = cos t, c_out = -sigma_d sin t, c_in = 1/sigma_d). The CM is the one-step first-order solve f = cos(t) x_t - sin(t) sigma_d F(x_t/sigma_d, t). They trace continuous-time CM instability to the sin(t) d_t F term (the time-embedding derivative), fix it architecturally, and recast the continuous CM gradient as an MSE to a stop-grad target, F_theta - F_theta^- - cos(t) df^-/dt, so EDM2-style adaptive weighting applies.

## What they find

2-step FID: CIFAR-10 sCT 2.06 (1-step 2.85); IN64 sCT 1.48 (1-step 2.04), sCD 1.66; IN512 sCD-XXL (1.5B) 1.88 (1-step 2.28) against a 1.73 teacher at 63x2 NFE. All three are within 10% of the teacher. sCD tracks the teacher's scaling with a constant FID ratio; sCT is better at small scale but higher-variance at large scale. Fig 5a: raw tangent vs clipped or normalised tangent shows normalisation/clipping gives "substantial improvements". Fig 5b: adaptive weighting improves on top of that. Fig 5c: discrete-time CMs improve with N up to ~1024, then degrade from precision; continuous beats all N. sCD reaches teacher-level 2-step quality in <20% of teacher compute (~20k finetune iterations). Versus VSD: VSD trades recall for precision and mode-collapses at high guidance, while sCD keeps teacher-like precision/recall.

## Implementation details

- Time and normalisation: c_noise(t) = t, not log(sigma_d tan t), which blows up as 1/cos t. Positional time embedding (Fourier scale ~0.02; large Fourier scale 16 is unstable). "Adaptive double normalisation": norm(x) * pnorm(s(t)) + pnorm(b(t)).
- Tangent: normalise as g/(||g|| + 0.1), or clip to [-1, 1]. Tangent warmup: multiply the sin(t) term by r ramping 0 to 1 over 10k iterations.
- Weighting: adaptive weighting via a learned w_phi(t) (EDM2 uncertainty trick), times a prior 1/(sigma_d tan t).
- Time proposal: lognormal on sigma_d tan t, P_mean -1.0 / -0.8, P_std 1.4–1.8. Keep P_mean small (near data): "training signal of CMs only come from the clean data".
- Numerics: compute the JVP with a cos(t) sin(t) prefactor folded into the tangent inputs to avoid FP16 overflow near t = 0 or pi/2. They wrote a custom flash-attention JVP.
- Optimiser and init: lr = teacher lr x 0.01, RAdam on CIFAR, init from the teacher EMA, dropout only at res <= 16, 2-step intermediate t = 1.1.

## Caveats

- Needs several architectural changes, so it is not drop-in for arbitrary diffusion architectures (their own limitation).
- Best results are distillation (sCD), which depends on teacher quality. CFG is incompatible with sCT.
- Image FID only. JVP plumbing is nontrivial at scale.
