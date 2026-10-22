# All-in-one simulation-based inference

Gloeckler, Deistler, Weilbach, Wood, Macke, 2024, ICML 2024 · [arXiv:2404.09636](https://arxiv.org/abs/2404.09636)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 91 citations in
2.5 yr (Semantic Scholar, 2026-10-19).

## What they do

Train one score-based diffusion model on the *joint* $\hat x=(\theta,x)$ with a transformer whose tokens are one per scalar variable: [learned variable-identifier embedding (or shared vector + random Fourier features of the index for function-valued variables), value, learned condition-state embedding]. A condition mask $M_C$ is resampled every batch; conditioned variables stay clean and the DSM loss is masked to latent variables, so a single network samples posterior, likelihood and arbitrary conditionals. An attention mask $M_E$ encodes the simulator's graphical model (directed or symmetrised). Interval/inequality constraints are imposed by guidance $\nabla\log\sigma(-s(t)c(\hat x_{0|t}))$ on the Tweedie-denoised estimate, plus self-recurrence (RePaint-style re-noising).

## What they find

SBIBM (Gaussian linear, SLCP, Two Moons, Gaussian mixture; 1k/10k/100k sims; C2ST, figures only): Simformer beats NPE (neural spline flow) on all tasks except Gaussian linear at 10k, even with dense attention; structured attention adds most on sparse-dependency tasks (factorised Gaussian linear; SLCP with 4 i.i.d. observations). "~10× fewer simulations than NPE" averaged across tasks. Training on posterior masks only is no better than all-conditionals, and on SLCP all-conditionals is better (learning the easy likelihood helps). Posterior-only Simformer ≈ NPSE with MLP. Reverse SDE: sharp transition, ≥50 evaluations are enough for best C2ST (except Two Moons/VPSDE). VESDE better on Two Moons, VPSDE slightly better on SLCP. Calibration: conservative like NPE. Arbitrary-conditional C2ST good on Tree, HMM, Two Moons, SLCP (100 MCMC-referenced conditionals each).

## Implementation details

Small transformer: token dim 50, 6 layers (8 for LV/SIR/HH), 4 heads × 10, FFN ×3; time = 128-d random Gaussian Fourier embedding; batch 1000, Adam. VESDE σ∈[1e-4, 15], VPSDE β∈[0.01, 10], t∈[1e-5, 1]; Euler–Maruyama with 500 steps. Condition mask sampled uniformly among {joint, posterior, likelihood, Bernoulli(0.3), Bernoulli(0.7)} — "slightly better than purely random". Directed attention masks must be updated algorithmically when conditioning changes (App. A1.1); attention masks alone do not guarantee conditional independence in multi-layer transformers (A1.2). For high-dimensional data (GW benchmark, 2×8192 samples) they collapse each detector stream into one token with a CNN embedding net and target only posterior/partial-posterior masks.

## Caveats

Low-dimensional benchmarks (θ ≤ 10); one-token-per-scalar does not scale to long data (they abandon it for GW). Log-probabilities need the probability-flow ODE. Results mostly in figures. Joint modelling is wasteful when data ≫ parameters (authors' own limitation).
