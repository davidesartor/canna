# Consistency Models

Song et al., 2023, ICML 2023 · [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 2348 citations
in 3.6 yr (Semantic Scholar, 2026-10-19).

## What they do

Learn the consistency function f(x_t,t) ↦ x_ε of the PF-ODE (EDM VE setting, σ(t)=t, T=80, ε=0.002). Boundary condition f(x,ε)=x enforced by skip parametrisation f_θ = c_skip(t) x + c_out(t) F_θ(x,t) with c_skip(ε)=1, c_out(ε)=0 (this is what rules out the trivial constant solution). Loss: λ(t_n) d(f_θ(x_{t_{n+1}},t_{n+1}), f_{θ⁻}(x̂_{t_n},t_n)) with θ⁻ an EMA "target network"; x̂_{t_n} is one ODE step with a teacher score (CD) or, for consistency training (CT), the same x plus the *same* noise z at the smaller t_n (unbiased score estimate −(x_t−x)/t², valid as N→∞ with Euler). Multistep sampling alternates f_θ and re-noising to τ_n.

## What they find

CIFAR-10 CD 1-step FID 3.55, 2-step 2.93 (PD: 8.34 / 5.58); ImageNet-64 CD 6.20 / 4.70 (PD 15.39 / 8.95). CT from scratch: CIFAR 8.70 / 5.83, ImageNet-64 13.0 / 11.1, LSUN Bedroom 16.0 / 7.85 — note CT recall is low (Bedroom 0.17 vs EDM 0.45; ImageNet 0.47 vs 0.67): one-step CT loses diversity. Ablations (Fig. 3, curves only): LPIPS ≫ ℓ1, ℓ2 for CD; Heun > Euler at equal N; N=18 best for CD, insensitive once N large; CT very sensitive to N (small N fast but biased, large N slow); adaptive N(k), μ(k) schedules clearly faster/better than constants. EMA+stopgrad target "greatly stabilises" vs θ⁻=θ.

## Implementation details

N(k)=⌈sqrt(k/K((s1+1)²−s0²)+s0²)−1⌉+1, μ(k)=exp(s0 log μ0 / N(k)), s0=2, s1=150–200, μ0=0.9–0.95; CIFAR lr 4e-4 bs 512; ImageNet/LSUN lr 8e-6/1e-5 bs 2048; λ≡1. Continuous-time variants (JVP) exist (App. B): stopgrad version beats non-stopgrad; discrete CD beats continuous CD (variance; higher-order solver); continuous CT beats discrete CT but *only stable when initialised from a pretrained EDM* (high-variance loss). Multistep sampling times chosen by greedy ternary search on FID.

## Caveats

Images only, FID-driven; heavy use of LPIPS (a perceptual metric with no analogue for parameter vectors); no calibration/coverage evaluation; one-step CT has visibly worse recall (mode coverage).
