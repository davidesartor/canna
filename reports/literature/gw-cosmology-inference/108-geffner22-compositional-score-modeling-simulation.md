# Compositional Score Modeling for Simulation-Based Inference

Geffner, Papamakarios, Mnih, 2022, ICML 2023 · [arXiv:2209.14249](https://arxiv.org/abs/2209.14249)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 63 citations in
4.1 yr (Semantic Scholar, 2026-10-19).

## What they do

For n i.i.d. observations, $p(\theta|x_{1:n})\propto p(\theta)^{1-n}\prod_j p(\theta|x_j)$. Define bridging densities that keep this factorisation, $p^f_t\propto (p(\theta)^{1-n})^{(T-t)/T}\prod_j p_t(\theta|x_j)$, so the score is the prior score (annealed) plus the sum of single-observation conditional scores from one network trained on single-(θ,x) pairs (F-NPSE). Because the naive sum of diffused scores is *not* the score of the diffused product, sampling uses annealed Langevin (L steps per noise level) rather than the plain reverse SDE; reference is N(0, I/n). PF-NPSE groups observations into subsets of size ≤ m (set-encoder network) to trade error accumulation for simulation cost; m=1 is F-NPSE, m=n_max is NPSE.

## What they find

4 SBI tasks (G-G 10-D, G-MoG 10-D, SIR, Lotka–Volterra), budgets 1e3–3e4 sims, up to 30 observations: F-/PF-NPSE best in most cells by squared MMD (and C2ST, App E.1); F-NPSE best for n ≤ 10, PF-NPSE (m=6) best for larger n; NRE+HMC degrades with n and misses a mode in the bimodal 2-D toy. Optimal m usually 3 or 6 (sweep m ∈ {1,3,6,12,18,30} at n=22). Conservative (gradient-of-scalar) score parameterisation makes no notable difference; sampler robust once L ≈ 5–10 Langevin steps per level.

## Implementation details

400 noise levels; L=5, δ_t = 0.3(1−α_t)/√α_t; ε-parameterised score; separate residual MLP embedders for θ and x, sinusoidal t embedding, LayerNorm throughout; constrained priors reparameterised to standard normal so $\nabla\log p(\theta)$ is defined everywhere; App D gives a hyperparameter-free O(T) Gaussian-transition sampler for score products (sometimes worse than Langevin).

## Caveats

error accumulates as n grows (n scores summed, each with error); requires the prior score; annealed Langevin cost O(LT); evaluated only on low-dimensional θ (≤10).
