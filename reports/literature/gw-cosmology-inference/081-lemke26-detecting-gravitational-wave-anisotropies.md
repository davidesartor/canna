# Detecting Gravitational-Wave Anisotropies with Simulation-Based Inference

Lemke et al., 2026, arXiv preprint · [arXiv:2605.24082](https://arxiv.org/abs/2605.24082)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Train a BCE classifier on simulated per-frequency cross-correlation estimators ρ̂_ab, isotropic GWB (H0) vs one or more hotspots (H1). Under the Jeffrey–Wandelt evidence-network argument the logit converges to ln K. This replaces the Gaussian likelihood used in frequentist radiometer / max-SNR searches. The motivation is that ρ̂ follows a generalised χ² distribution, not a Gaussian (Fig. 1), and the covariance depends on the sky map being estimated.

## What they find

- 120 pulsars, NG15 amplitude log10 A = −14.76, local significance in the second frequency bin only.
- 3σ detection rate, 1 hotspot with 80% of the power: 21% (SBI) vs 11% (classical), about +90%.
- 2 hotspots at 40% each: 9% vs 3%, +200%.
- Ablation on data forced to be Gaussian with a fixed hotspot: SBI and the classical likelihood-ratio test are equal. The gain therefore comes entirely from non-Gaussianity. An MLP was used for this test because the GNN overfit the simpler data.
- Data scaling: performance saturates above about 1e5 training samples, using datasets from 1.6e4 to 5.76e6 (4.6M train, 0.6M validation, 0.6M test).
- A classifier trained only on the 1-hotspot/80% case generalises to 2–3 hotspots and 40% power with comparable ROC.
- Robustness to noise-parameter marginalisation and to a full timing-model fit (test_real, 2,000 samples): no significant loss.

## Implementation details

- Data as a fully connected graph:
  - Nodes are pulsar sky positions. Edges carry the whitened residual e_ab = (ρ̂_ab − HD_ab)/σ_ab, giving 14,280 directed edges.
  - Edge and node features are standardised over the whole training set.
- Network: a minimal GINE GNN.
  - Hidden size d_h = 16. Two GINE layers with BatchNorm and residual connections. An edge update MLP takes (h_a, h_b, e_ab).
  - Readout concatenates all node embeddings (7,680), then MLP 1024 → 256 → 1 with dropout 0.2.
- Optimisation: AdamW, lr 1e-3, wd 1e-3, at most 10 epochs, ReduceLROnPlateau (×0.5, patience 3), early stopping (patience 5), restore best validation checkpoint.
- Feed residuals from the known null model (minus HD) rather than raw estimators.
- Noise marginalisation at test time: take the median classifier probability over noise-posterior draws.
- Validation idea not yet run: sky scrambles of real data.

## Caveats

- Detection only, no map or parameter posterior.
- Single frequency bin, so the significances are local with no trials factor.
- Training assumes perfect noise knowledge and no timing refit.
- Simulation is CPU-heavy: about 35 CPU-h per 1e5 samples, not in JAX.
- The ln K calibration of the classifier is not checked (only ROC).
