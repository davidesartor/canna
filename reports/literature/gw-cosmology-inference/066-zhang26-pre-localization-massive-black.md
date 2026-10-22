# Pre-localization of Massive Black Hole Binaries in the Millihertz Band

X.-T. Zhang, J. Gair, C. Messenger, N. Korsakova, Y.-M. Hu, H.-Y. Chen, 2026, arXiv preprint · [arXiv:2604.24330](https://arxiv.org/abs/2604.24330)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.5 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Amortised NPE (embedding + NSF) for a single TianQin MBHB in a 5-day pre-merger segment ending 10–20 min before coalescence; posteriors in ~1 min on CPU (<10 s for 1e4 samples on GPU) for EM follow-up alerts.
IMRPhenomHM, non-spinning, A/E channels with input shape (2, 4320) (5 d at 0.01 Hz). Six inferred: log Mc, eta, lambda, beta, Delta t_c, D_L; iota, psi, phi_c are randomised in training and marginalised (never sampled). Priors: log10 Mc U[log10 2e5, 6], eta U[0.05, 0.25], z uniform in comoving volume on [0.1, 2], isotropic sky/iota, uniform psi/phi_c, Delta t_c = t_c - t_end U[600, 1200] s. Each target min-max scaled to (0,1) after transforms (log Mc, lambda mod 2pi, D_L^3).

## What they find

Reference injection SNR 162.9 (Mc 9e5, eta 0.2, z 2 / D_L ~16 Gpc). Against heterodyned PTMCMC (10 temps x 64 chains x 5e5 iters, ~4 h on 56 cores, reference template = injection, noise-free data): both find 8 sky modes; clustered A90 20.4 deg^2 (NSF) vs 5.3 deg^2 (MCMC), folded single-mode 3.34 vs 0.79 deg^2; 1-sigma widths NSF/MCMC: log Mc 0.0037/0.0006 (6x), eta 0.0082/0.0020 (4x), t_c 20.2/13.6 s, D_L 1.23/0.62 Gpc (2x). Cases II/III: log Mc 9x and 5x too wide, sky area 4.3x and 2.6x. JSD NSF vs MCMC: Mc 0.362, eta 0.274, sky ~0.01. A90 stable over 10 noise draws (20.2 +- 0.7 deg^2). p-p over 1000 events: Mc fails (p = 0.003), combined p = 0.010; authors still call it calibrated.

## Implementation details

Inception-like multi-kernel 1-D front end (kernels 7/14/28/56/128, adaptive pooling to 48) + ResNet-50 to a 128-d summary; glasflow CouplingNSF, 50 transforms, 256 hidden, n_bins 10. Stage-wise joint training (I: sky only; II: grow to 6 params; III: 1e7 train + 5e5 val, batch 2048, ~223 h on one A100-80GB; IV: retrain on fresh 1e7 at lr 1e-5). Adam 1e-4 cosine, <=200 epochs, patience 30.
curriculum over parameters (sky first, then add the rest); fresh-data retrain at low lr as a final stage; cluster sky samples on the embedded unit vector, not (lambda, beta); MAD / kNN outlier removal before computing HPD areas (i.e. raw NSF samples have outliers).

## Caveats

A two-step scheme (embedding pretrained by point-estimation regression, then frozen) took >10 days and collapsed degenerate sky modes onto a mean, so they switched to joint, stage-wise training. NSF systematically too broad in the best-measured intrinsic directions (Mc, eta), attributed to the embedding discarding fine high-frequency structure. Comparison is not like-for-like (MCMC on noise-free data, NSF on noisy).
- **Label switching / multiple sources:** single source; sky multimodality (8 TianQin-symmetric images) is handled by the flow and post hoc by DBSCAN clustering on the unit sphere plus an 8-fold symmetry "folding". Source confusion explicitly untreated; they suggest training on overlaps or a separation front end.
- Zero spin, single detector, 5-day window only, MCMC reference template at truth; "calibrated" despite p = 0.003 on Mc; widths compared on one to three injections.
