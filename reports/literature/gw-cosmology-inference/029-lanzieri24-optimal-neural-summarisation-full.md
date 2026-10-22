# Optimal Neural Summarisation for Full-Field Weak Lensing Cosmological Implicit Inference

Lanzieri, Zeghal et al., 2024/25, A&A; v2 Feb 2025 · [arXiv:2407.10877](https://arxiv.org/abs/2407.10877)

**Evidence: B.** Peer-reviewed (Astronomy & Astrophysics); 20 citations in 2.3 yr (Semantic
Scholar, 2026-10-19).

## What they do

A controlled comparison of the loss used to train a neural compressor t = F(x) with dim t = dim θ, holding fixed the architecture (ResNet-18), the downstream NPE (RealNVP, 4 couplings, 2×128 SiLU) and the simulator. The simulator is a differentiable JAX log-normal 5-bin tomographic LSST-Y10 κ model, 256² pixels over 10×10 deg², wCDM with 6 parameters. Ground truth is full-field HMC (NUTS) on the same model. Theory: MSE learns the posterior mean, MAE the median, and GNLL also the mean (plus a covariance that is discarded). None of these is guaranteed sufficient. VMIM (−log q(θ|F(x)), with q a normalising flow trained jointly) maximises a lower bound on I(t, θ), so it is sufficient at optimum.

## What they find

FoM Ω_c–σ8: C_ℓ 1222, HMC 2520, VMIM 2526, MSE 2043, MAE 2316, GNLL 1882. Ω_c–w0: 100 / 198 / 190 / 152 / 162 / 178. σ8–w0: 77 / 176 / 171 / 139 / 149 / 153. VMIM beats MSE by about 1.24× and MAE by about 1.1–1.17× on every pair. w0: HMC −1.01, VMIM −1.00, MSE −1.13, MAE −1.16, GNLL −0.73 (GNLL is biased). Full field is about 2× the C_ℓ FoM. The companion C2ST between VMIM and HMC posteriors is 0.6, i.e. close but not identical.

## Implementation details

Two-stage pipeline. Train the compressor with VMIM, discard its flow, then train a fresh NPE on the frozen summaries ("hard to train a precise density estimator while the compressor still moves"). GNLL should equal MSE in theory but trains unstably in practice because of the log-det term, and was worse; the authors recommend plain MSE if only a mean is wanted. IMNN was not run: memory limits, since the large batches needed for the Fisher estimate did not fit.

## Caveats

One observed map, no averaging over realisations. FoM agreement is necessary but not sufficient for sufficiency. Simulation budget is not an axis (asymptotic regime). Log-normal model only, no systematics.
