# Flow Matching for Scalable Simulation-Based Inference

Dax, Wildberger, Buchholz et al., 2023, NeurIPS 2023 · [arXiv:2305.17161](https://arxiv.org/abs/2305.17161)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 155 citations in 3.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

NPE with a CNF trained by conditional flow matching: $\mathcal L = \mathbb E_{t\sim p(t),\theta_1\sim p(\theta),x\sim p(x|\theta_1),\theta_t}\|v_{t,x}(\theta_t)-u_t(\theta_t|\theta_1)\|^2$ with the Lipman OT Gaussian path ($\mu_t=t\theta_1$, $\sigma_t = 1-(1-\sigma_{min})t$), Gaussian base. Density via $q(\theta|x)=q_0\exp(-\int \mathrm{div}\,v)$. Two SBI-specific changes: (i) architecture is a pyramid *from x to v* (data feature extractor), with (t,θ) injected via **gated linear units in every residual block** (after a small residual (t,θ) embedding net); (ii) time prior $p_\alpha(t)\propto t^{1/(1+\alpha)}$ upweights t→1 (data end). Theorem: under strong regularity (bounded second derivatives of u, v; base score linear growth) $KL(p_1\|q_1)\le C\,\mathrm{MSE}^{1/2}$ → FM is "mass-covering" (forward-KL-like).

## What they find

SBI benchmark (10 tasks, N=10³–10⁵): C2ST comparable to NPE on most, better on SLCP at 10⁵; MMD clearly better than NPE; OT-FMPE beats a VP score-matching baseline (same net) on almost all configs. GW150914 (15-D BBH, x∈ℝ¹⁵⁷⁴⁴, 5·10⁶ sims, 400 epochs, bs 4096, A100): mean JSD **0.5 mnat vs NPE 3.6 mnat**, max <2 mnat (indistinguishability threshold), comparable to GNPE; trains in ~2 days (1.9·10⁸ params) vs NPE ~3 days (1.3·10⁸) → the "30% faster". IS evidences agree: log Z = −7667.969±0.005 (FMPE) vs −7667.958±0.006 (NPE). Ablations: GLU (t,θ)-conditioning vs concatenation on dim(x)=100 tasks (Fig 8): clear gain; delaying (t,θ) conditioning to the last (128-d) blocks hurts accuracy but lets the context embedding be cached (cheaper inference); time-prior reweighting α>0 beats uniform on Two Moons at all budgets (Fig 11). Same-size-as-NPE-embedding FMPE ≈ NPE accuracy with ~1/3 of parameters.

## Implementation details

sweep α ∈ {−0.25, −0.5, 0, 1, 4}; GW uses **α=1**, lr 5e-4, bs 4096; residual MLP 2048→16 diamond. Inference: dopri5 atol=rtol=1e-7, **248 network passes for sampling (26 s / 4096 batch), 350 passes + divergence for log-probs (352 s)** vs NPE 1.5 s — log-prob costs ~13× sampling, and memory limits batch size. GW IS used **phase marginalisation** (φ_c analytic in the likelihood) and φ_c, φ_JL, θ_JN were excluded from the JSD evaluation. Reference samples always within q's support (log q histograms, Fig 9), but can have small q density when the model is poor (Lotka–Volterra).

## Caveats

the mass-covering theorem needs regularity not ensured by nets; a single GW event (loud, GW150914); no PSD conditioning; no ablation of the time prior on GW; "mass-covering" ≠ conservative.
