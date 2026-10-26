# Understanding Pathologies of Deep Heteroskedastic Regression

Wong-Toi, Boyd, Fortuin & Mandt, 2023 (v2 2024), arXiv preprint (venue not confirmed) · [arXiv:2306.16717](https://arxiv.org/abs/2306.16717)

**Evidence: C.** 0 citations in 3.3 yr on the arXiv record (OpenAlex, 2026-10-25; Semantic Scholar rate-limited). The arXiv page names no venue (v1 2023-06-29, v2 2024-02-13); the UAI 2024 venue was not confirmed from the page.

## What they do

Setting: Gaussian NLL with a mean net μ̂θ and a precision net Λ̂φ, both overparameterised, plus L2 penalties α‖θ‖² and β‖φ‖² (Eq. 5). The penalties are reparameterised as ρ (likelihood vs regularisation, ρ = 1 is MLE) and γ (how the regularisation is split between mean and precision nets) (Eq. 6).
- Two failure modes: a near-perfect mean fit with variance driven to zero, or a flat constant mean with the precision net absorbing all residuals as noise.
- A statistical field theory (Sec. 3) replaces the nets by free functions with a Dirichlet-energy penalty. The stationarity conditions (Eq. 9) are two coupled PDEs, with both regularisers scaled by 1/p(x). Proposition 1: no solution at ρ = 1, a non-unique solution at ρ = 0, and no valid solution if γ = 1 (no mean regularisation) with ρ in (0,1).
- Empirical phase diagram on the ρ–γ plane (Sec. 2, Table 1, Fig. 1): five regions, namely two underfit (flat mean), two overfit (mean or precision), and one calibrated region S. The boundaries are sharp for neural nets.
- Practical rule (Sec. 4.2, App. B.7, D.2): search only the diagonal ρ = 1 − γ, so the search is O(N) fits instead of O(N²). Pick the model at the logit-scale midpoint between the diagonal model with the lowest training μ-MSE and the one with the lowest training Λ^(−1/2)-MSE.

## What they find

Metrics are μ-MSE, Λ^(−1/2)-MSE (predicted std vs absolute residual) and ECE. No NLL or RMSE is reported. Six seeds.
- vs β-NLL (β = 0.5), Table 2 (test): Concrete μ-MSE 0.11 ± 0.02 vs 14.99 ± 28.75; Yacht 0.01 ± 0.01 vs 3.4×10² ± 1.9×10²; Housing 1.22 vs 8.5×10² ± 2.0×10⁴; Sine 0.80 vs 4.41 ± 6.90. Power is an exception: β-NLL has the lower μ-MSE (0.03 vs 0.04). On the 10M-point ClimSim solar-flux task: 0.29 vs 0.49 μ-MSE, 0.12 vs 0.29 Λ^(−1/2)-MSE.
- vs β-NLL and an MLE ensemble (Table 5): mixed. On Cubic and Curve β-NLL has the lower μ-MSE or comparable ECE (Cubic μ-MSE 0.234 vs 0.150). The MLE ensemble has the lowest ECE on Concrete (0.066 vs 0.247), Power and Yacht. The text itself says "competitive with or exceeds".
- The Solar Flux β-NLL and ensemble μ-MSE entries (0.4877, 1.0067) equal the Curve row, so Table 5 may contain a copy error.
- β-NLL at β = 0.5 can leave the mean badly fit in this paper's setup (Table 2: very large μ-MSE with large std on Concrete, Housing, Yacht). They run it with their own training scheme, with no per-dataset tuning of β.

## Implementation details

- Three hidden layers of 128 leaky-ReLU units (256 for ClimSim), softplus on variance and precision outputs.
- Two-stage training: the mean net alone first, then the mean and precision jointly.
- Adam with a triangular cyclic lr (1e-4 to 1e-2, cycles of 50k epochs), gradient clipping at 1000, 500k to 2.5M epochs on small data.
- Grid: 22 values of ρ and γ on a logit scale from 1e-10 to 1 − 1e-5. About 400 GPU hours on RTX 8000.
- Datasets: Concrete, Housing, Power, Yacht (Table 4), synthetic Sine, Cubic, Curve, and ClimSim (10.09M train, 124 inputs, scalar output).

## Caveats

- Regression with scalar output and L2 regularisation as the control knob. There is no β-NLL-style reweighting analysis and no learned log-variance task weight, so only the failure modes transfer. Both mean-collapse modes are exactly what a σ-weighted loss must avoid.
- The result that mean regularisation is necessary (no solution at γ = 1) is specific to overparameterised MLPs with L2. It does not say how a bf16 flow backbone with a σ head behaves.
- The mean-first stage (a warm-up from a mean-only fit) is used but not ablated.
- Theory is for a nonparametric field, and the authors say it cannot yet set hyperparameters for real nets. Needing many fits along the diagonal is a stated limitation.
- Small tabular sets, a scalar output, and a single β = 0.5 baseline. Faithful, Skafte and Stirn et al. are discussed but never compared.
- The abstract says "significantly improved"; the tables are mixed.
