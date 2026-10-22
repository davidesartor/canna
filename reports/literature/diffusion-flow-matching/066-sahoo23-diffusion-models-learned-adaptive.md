# Diffusion Models With Learned Adaptive Noise

MuLAN; Sahoo, Gokaslan, De Sa, Kuleshov, 2023; v3 Nov 2024, NeurIPS 2024 · [arXiv:2312.13236](https://arxiv.org/abs/2312.13236)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 64 citations in 2.8 yr
(Semantic Scholar, 2026-10-19).

## What they do

Treat the forward process as a variational posterior and learn it. VDM's claim that the ELBO is invariant to the schedule (only endpoints matter) holds only for scalar schedules; with a per-dimension ν(t) ∈ ℝ^d the diffusion loss −½∫(x₀−x_θ)²·∇_tν dt is a line integral of a non-conservative field and becomes path-dependent. Schedule: α_t²=sigmoid(−γ), σ_t²=sigmoid(γ), ν=exp(−γ), γ_min=−13.3, γ_max=5.0. **Polynomial schedule (Eq 62–63):** f_φ(c,t)=a²t⁵/5 + ab t⁴/2 + (b²+2ad)t³/3 + bd t² + d²t (the integral of (at²+bt+d)², so monotone with derivative that can vanish twice), γ_φ(c,t)=γ_min+(γ_max−γ_min)·f_φ(c,t)/f_φ(c,1); a,b,d ∈ ℝ^d from a 2-layer MLP (3072 hidden, swish) of z. Continuous-time loss ½E_t[(ε−ε_θ)ᵀdiag(∇_tγ(z,t))(ε−ε_θ)] plus KL(q(z|x₀)‖p(z)); z discrete k-hot (m=50, k=15) via sum-of-Gamma perturbed top-k.

## What they find

CIFAR-10 VLB bpd: VDM 2.65 at 10M steps; MuLAN 2.65 at 2M and 2.60 at 10M (FID-10k 23.91 → 18.54/17.62); ImageNet-32 3.72 (VDM, 2M) vs 3.72 at 1M and 3.71 at 2M. Wall clock: 2.6 vs 2.24 steps/s on 4 V100 but 30 vs 10 days to reach 2.65. NLL via flow ODE: CIFAR-10 2.55, ImageNet-32 3.67 (SOTA without augmentation; vs DiffEnc ≤2.62, NDM ≤2.70, DNF 3.04). v-param 2.59 (8M) vs ε 2.60 (10M). Ablations (CIFAR-10, batch 64, 2.5M steps, curves only): no aux latent → ≈VDM; scalar z-conditioned schedule → no gain (initially worse due to the extra KL); polynomial > monotone-NN > linear schedule. Swapping the learned schedule for scalar or linear in a trained model returns the bound to 2.65. Learned schedules showed no human-interpretable structure, even on synthetic frequency/mask datasets.

## Implementation details

pin the endpoints γ(0), γ(1) and learn only the interior shape (normalise by f(1)); degree-5 polynomial with derivative zero allowed at both ends is easier to optimise than a monotone MLP.

## Caveats

likelihood-focused (FID reported only for completeness, no EMA tuning); expensive (32-block UNets, 8–10M steps); gains in bpd are 0.05–0.1; the "SOTA" bpd numbers are for no-augmentation settings only.
