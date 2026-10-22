# Align Your Flow: Scaling Continuous-Time Flow Map Distillation

Sabour et al., 2025, arXiv / NVIDIA; venue not stated in the text · [arXiv:2506.14603](https://arxiv.org/abs/2506.14603)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 96 citations in 1.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

The paper proves that consistency models degrade under multi-step sampling: for a Gaussian, the error grows with step count unless the model is exact. It distils a two-time flow map f(x_t, t, s) = x_t + (s − t)·F(x_t, t, s) with two continuous-time objectives. **AYF-EMD** (Eulerian) generalises consistency training, recovering it at s = 0 and FM as s → t. **AYF-LMD** (Lagrangian) is the alternative. Training borrows sCM's stabilisers (tangent normalisation, a regularised tangent warmup) and distils an *autoguided* teacher, with the guidance scale λ ~ U[1,3] fed in as a network input.

## What they find

- ImageNet 64, FID by step count:
  | model | 1 step | 2 steps | 4 steps | 8 steps |
  |---|---|---|---|---|
  | AYF | 2.98 | 1.25 | 1.15 | 1.12 |
  | AYF + adversarial fine-tune | 1.32 | 1.17 | 1.10 | 1.07 |
  | sCD (baseline) | 2.44 | 1.66 | — | — |
- ImageNet 512: AYF-S (280M) gets 3.32 / 1.87 / 1.70 at 1/2/4 steps, while sCD-XXL (1.5B) needs 2 steps for 1.88.
- Ablation (IN512, 1/2/4 steps):
  - LMD instead of EMD: 12.45 / 8.90 / 6.70. LMD over-smooths images, though it wins on 2D toys.
  - CFG teacher instead of autoguidance: 4.12 / 2.57 / 2.32.
  - sCD-S with the same teacher: 2.90 / 1.87 / 1.84, then *worsens* with more steps (4.62 at 64), which confirms the multi-step theorem.
  - A Shortcut reimplementation: 47.6 / 13.1 / 5.37.
- FLUX LoRA distillation: 10k steps, about 4 hours on 8 A100s.

## Implementation details

- Parameterisation: c_skip = 1, c_out = (s − t).
- Tangent normalisation: divide by ||g|| + c with c = 0.1. Adaptive weighting made no difference.
- Time conditioning: c_noise(t) = t.
- Tangent warmup: ramp r up to a clamp of r_max = 0.99. While r < 1 the loss carries an extra |t − s|·||F − v||² term, which acts as a straightness regulariser.
- Gap sampling: |t − s| ~ sigmoid(N(P_mean, P_std)), with (−0.8, 1.0) on IN512 and (−0.6, 1.6) on IN64. Train only on t > s.
- Sampling: "γ-sampling", which re-noises partially between steps.

## Caveats

- Distillation only; it needs a strong pretrained teacher.
- Adversarial fine-tuning adds GAN machinery.
- The EMD-beats-LMD result contradicts 2406.07507 and 2505.18825. The difference is probably the sCM stabilisers.
