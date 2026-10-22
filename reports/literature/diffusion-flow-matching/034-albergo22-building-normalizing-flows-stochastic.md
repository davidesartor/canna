# Building Normalizing Flows with Stochastic Interpolants

Albergo & Vanden-Eijnden, 2022, ICLR 2023 · [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 1147
citations in 4.1 yr (Semantic Scholar, 2026-10-19).

## What they do

Pick an interpolant I_t(x0, x1) with I_0 = x0, I_1 = x1, e.g. trig I = cos(πt/2)x0 + sin(πt/2)x1, with x0, x1 independent. Learn v by minimising G(v) = E[|v|² − 2∂_tI·v], whose unique minimiser is v = j/ρ (the conditional-expectation velocity). This is the same minimiser as the MSE E|v − ∂_tI|², but G has a known optimum value −E|v|².
- Prop. 1 holds for any time density ω(t) > 0, so weighting t does not move the minimiser.
- Max-min over the interpolant recovers Benamou–Brenier OT (Prop. 2). W2² ≤ e^{1+2K}H(v̂), K the Lipschitz constant of v̂ (Prop. 3).
- Gaussian base plus trig interpolant: score = −x − (2/π)tan(πt/2)v (Prop. 4), giving an SDE whose drift and diffusion are singular at t = 0, 1.

## What they find

- Tabular NLL (lower is better): POWER −0.57, GAS −12.35, HEPMASS 14.85, MINIBOONE 10.42, BSDS300 −156.22; FFJORD −0.46, −8.59, 14.92, 10.43, −157.40.
- CIFAR-10: NLL 2.99 bpd, FID 10.27. ImageNet-32: NLL 3.48, FID 8.49. No EMA or warmup tricks.
- About 400x cheaper per training step than MLE FFJORD on MiniBooNE.
- App. H: learning the interpolant (Fourier series, M = 7) and a Gaussian base (μ, Σ) shortens the path and improves likelihood on a checkerboard toy.

## Implementation details

- t ~ Beta(α, β) with (1.0, 0.5–1.0) on tabular and (1.0, 0.75) on images, "under the heuristic that the flow should be well trained near the target". Legitimate because any ω(t) keeps the minimiser.
- Diagnostic G̃ = G + E|v̂|², which must reach 0 at convergence (monitored in Fig. I.2). The MSE has no such absolute baseline.
- Tabular MLPs 4–5 layers, width 512–1024, lr 2e-3–3e-3; images U-Net, lr 1e-4–2e-4. Batch sizes for target, base and time are factorised separately.

## Caveats

- The score relation and SDE link need a Gaussian base. The finite-capacity effect of the Beta time density is asserted, not ablated.
- Image FIDs are far from the state of the art of the time.
