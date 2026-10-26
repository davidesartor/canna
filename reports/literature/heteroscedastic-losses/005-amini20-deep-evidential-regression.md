# Deep Evidential Regression

Amini, Schwarting, Soleimany & Rus, 2020, NeurIPS 2020 · [arXiv:1910.02600](https://arxiv.org/abs/1910.02600)

**Evidence: A.** Peer-reviewed (NeurIPS); 290 citations on the arXiv record, 6.9 yr since v1 (OpenAlex, 2026-10-25). Semantic Scholar rate-limited (429). The count excludes the NeurIPS record, so it is a lower bound.

## What they do

A regression net predicts the four parameters m = (γ, υ, α, β) of a Normal-Inverse-Gamma prior over the Gaussian (μ, σ²), in one forward pass.
- Prior: μ ~ N(γ, σ²/υ), σ² ~ Γ⁻¹(α, β) (Eq. 4). Heads: linear γ, softplus υ, α (+1) and β (Sec. 3.3).
- Prediction E[μ] = γ. Aleatoric E[σ²] = β/(α−1). Epistemic Var[μ] = β/(υ(α−1)) (Eq. 5).
- The marginal likelihood is a Student-t St(y; γ, β(1+υ)/(υα), 2α), giving L_NLL (Eq. 7-8) with Ω = 2β(1+υ): ½log(π/υ) − α log Ω + (α+½) log((y−γ)²υ + Ω) + log(Γ(α)/Γ(α+½)).
- Evidence regulariser (Eq. 9-10): L_R = |y−γ|·Φ with Φ = 2υ + α. Total loss L = L_NLL + λ L_R. λ = 0 gives over-confident estimates, and too large a λ inflates uncertainty (footnote 2).
- A KL-to-zero-evidence prior was considered and rejected: it is undefined at zero evidence and hypersensitive to ε (Sec. 3.3, S1.3).

## What they find

- Toy cubic (Fig. 3): y = x³ + ε, ε ~ N(0, 3), train on [-4, 4], test on [-6, 6]. Epistemic uncertainty grows outside the training range. With smaller λ the OOD epistemic uncertainty decays toward zero while the aleatoric one stays flat (Fig. S2).
- UCI (Table 1, 9 sets, 20 splits), evidential vs dropout vs 5-member ensemble. RMSE is competitive, not best: Boston 3.06 (Drop 2.97, Ens 3.28), Concrete 5.85 (5.23, 6.03), Energy 2.06 (1.66, 2.09), Power 4.23 (4.02, 4.11). NLL is best or tied on all: Protein 2.63 vs 2.89 and 2.83, Naval -5.73 vs -3.80 and -5.63, Yacht 1.03 vs 1.55 and 1.18, Boston 2.35 vs 2.46 and 2.41. The caption claims best NLL everywhere, but Energy (1.39 vs ensemble 1.38) and Power (2.81 vs dropout 2.80) are within error. Inference takes 0.85 to 1.18 ms against 2.9 to 3.8 ms.
- NYU Depth v2 (Table S1, Fig. 4C): evidential RMSE 0.024 ± 0.032 and NLL -1.128 ± 0.290, 7.8M parameters. A 5-member ensemble gets RMSE 0.023 ± 0.027 and NLL -1.077, with 39M parameters. Spatial dropout (N = 5) gets 0.031 and -1.227. Calibration error: dropout 0.126, ensemble 0.048, evidential 0.033.
- OOD on ApolloScape (Fig. 5) and FGSM adversarial inputs (Fig. 6): entropy CDFs shift up, "competitive across methods". No numeric AUC is given.

## Implementation details

- Toy: 3 hidden layers of 100 units, Adam 5e-3, 5000 iterations, batch 128, λ = 0.01.
- Depth: U-Net, 160×128 input, Adam 5e-5, 60k iterations, batch 32, λ = 0.1, spatial dropout 0.1, 3 runs. Output is four maps per pixel.
- UCI follows the Hernández-Lobato and Adams protocol. λ is not stated there.
- λ differs between experiments (0.01 vs 0.1) with no selection rule.

## Caveats

- Per-output (per pixel) parameters, so the method is per-component by construction. The α, β, υ heads are separate outputs of a shared trunk.
- No stop-gradients and no mean warm-up are used. Mean accuracy is not analysed. UCI RMSE is slightly worse than dropout on 4 of the 9 sets shown, and depth RMSE is similar to the ensemble. The only discussion of the mean-fit versus uncertainty trade-off is via λ.
- λ is the main limitation, as the authors say. Removing non-misleading evidence is only partly handled, and the variance prior is unexplored.
- Many results are qualitative (OOD, adversarial, aleatoric maps in Fig. S7).
- Meinert et al. (file 006) argue that the loss does not identify (γ, υ, α, β) and that the epistemic term is a convergence-speed proxy.
- The 1/(1+...) form of the NIG variance (β/(α−1)) diverges as α → 1. This is relevant when the trunk is bf16, and the softplus + 1 floor is the only guard.
