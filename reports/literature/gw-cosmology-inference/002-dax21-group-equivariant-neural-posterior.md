# Group equivariant neural posterior estimation

Dax, Green et al., 2021, ICLR 2022 · [arXiv:2111.13139](https://arxiv.org/abs/2111.13139)

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 40
citations in 4.9 yr (Semantic Scholar, 2026-10-19).

## What they do

Suppose the posterior is (approximately) equivariant under a group G acting jointly on parameters and data, p(θ|x) = p(gθ|T_g x)|det J_g|. Define a blurred "pose proxy" ĝ = g^θ·ε with ε ~ κ, where κ is symmetric and narrow. Train q(θ'|T_{ĝ⁻¹}x) with θ' = ĝ⁻¹θ for exact equivariance, or q(θ|T_{ĝ⁻¹}x, ĝ) for approximate equivariance. The loss is the NPE loss with the extra expectation over ĝ ~ κ. At inference, Gibbs-sample the joint p(θ, ĝ|x): θ comes from the network given ĝ, and ĝ is θ's pose plus fresh blur. Initialise ĝ with a separate standard-NPE network q_init(pose|x). Without blur (κ = δ) the chain never moves. A wide κ converges in few iterations but aligns the data less tightly.

## What they find

- Toy damped oscillator (exact τ-translation, κ = N(0, 0.1 s²)): GNPE c2st ≈ NPE+CNN, both far better than plain NPE. The GNPE inputs have a visibly lower SVD effective dimension and cleaner first-layer filters. One Gibbs iteration suffices.
- GW (8 GWTC-1 events, IMRPhenomPv2, 15-D): plain NPE c2st up to 0.71. "Chained NPE" (infer the pose first, then condition on it) beats plain NPE but fails on GW170814 because its pose estimate is poor. An oracle pose from bilby makes chained NPE match MCMC, so the whole error is the pose. Fast GNPE (κ = U[±3 ms], 1 iteration, ~2 s) beats both baselines. Accurate GNPE (κ = U[±1 ms], 30 iterations, ~60 s) reaches c2st < 0.55 on all events, ~0.52 on average.
- All NPE hyperparameters were tuned on plain NPE and transferred unchanged to GNPE, so the advantage is likely understated.

## Implementation details

- Data: 8 s segments, 20–1024 Hz at Δf = 0.125 Hz. Per detector: Re/Im strain plus the PSD channel (10⁴⁶ S_n)^{-1/2}, 24,099 real inputs. **The first layer is a linear map per detector to 400 components, initialised with PCA of clean waveforms.** It is described as a key inductive bias that a CNN would preclude.
- The rest of the embedding is 24 residual FC blocks with batch norm and ELU, down to a 128-D feature vector. The proxy ĝ_rel is concatenated *after* the embedding.
- Flow: 30 rational-quadratic NSF coupling transforms, each with 5 two-layer residual blocks of width 512. That is 348 layers and 1.3–1.4×10⁸ parameters.
- Training: 5×10⁶ waveforms, 2% held out, batch 4096. 300 epochs at lr 3e-4 with a fixed PSD, then 150 epochs at 3e-5 with varying PSDs, conditioning the network on the PSD. Takes 16–18 days on one V100.
- Group: the per-detector arrival times split into a global shift (exact equivariance; translate θ) and relative shifts (approximate; condition on them). The data are always shifted by the full proxy. In the frequency domain the shift is just multiplication by e^{−2πifΔt}.
- The time prior (~20 ms) is much wider than the posterior. Removing that pose spread is what makes the residual problem learnable.
- Convergence: monitor the JS divergence between pose posteriors of successive iterations. Pick the κ width near the typical pose-posterior std. The distance prior is reweighted to cosmological afterwards. Separate networks are trained per distance range and observing run.

## Caveats

One problem family (BBH, ≤3 detectors). Gibbs sampling makes the density intractable, so IS needs a separate proxy-density flow (see 2210.05686). The pose has to be a group action that is cheap to apply to the data. Not tested with flow matching.
