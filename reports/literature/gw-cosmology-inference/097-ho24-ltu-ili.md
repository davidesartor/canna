# LtU-ILI: An All-in-One Framework for Implicit Inference in Astrophysics and Cosmology

Ho et al., 2024, OJAp · [arXiv:2402.05137](https://arxiv.org/abs/2402.05137)

**Evidence: B.** Peer-reviewed (The Open Journal of Astrophysics); 38 citations in 2.7 yr
(Semantic Scholar, 2026-10-19).

## What they do

A software framework wrapping sbi, pydelfi and lampe behind one config interface: NPE/NLE/NRE (plus sequential variants), MDN/MAF/NSF NDEs, custom embedding nets (CNN, GNN, fishnets), deep ensembles, and a validation stage (C2ST against a reference, PIT/SBC rank histograms, P-P plots, TARP, log-prob on held-out data).

## What they find

SLCP benchmark (C2ST vs long-HMC reference, 10 runs): every engine beats rejection ABC; NLE > NPE > NRE on SLCP (they note this ordering is problem-specific, since SLCP's likelihood is simple); sequential rounds barely help at 10³ sims but give large gains at 10⁵. Toy 10-D sinusoid+quadratic: SNPE with 10×2000 sims matches 160k-sim HMC; ABC with the same budget is much broader. Cluster X-ray mass (3285 Magneticum mocks): ensemble of 2 MAF + 2 MDN NPE gives 0.0782 dex vs 0.0773 dex for the original MSE CNN; ensembling slightly inflates the error bars; widening the first conv layers reaches 0.071 dex. Quijote halo P_ℓ (ℓ=0,1,2; 1800 train sims): NLE ensemble of 6 MAFs with VI sampling, calibrated by both rank histograms and TARP; h is unconstrained, as expected physically. Quijote halo GNN (10k halos, 20 h⁻¹Mpc radius graph) is looser than the P(k) result.

## Implementation details

Deep ensembles (average 2–6 NDEs, weighted by validation loss) are the recommended default against single-NDE overconfidence (Hermans+22). Affine-map bounded priors to avoid mass outside the support. Early stopping on a validation split. Two-step compression (pydelfi) for simulation efficiency. Check calibration with both marginal (PIT) and joint (TARP) tests, and use C2ST against a reference posterior when one exists (the most discriminating test). In the multi-round GW example, validation loss rose early in later rounds, a sign of overfitting.

## Caveats

A software paper: the benchmarks are standard and the "science" examples are demonstrations, not ablations. No flow matching or diffusion; TMNRE is not included either. Discussion of misspecification is qualitative: more noise in the simulator, conservative losses, evidence networks, frequentist recalibration (Masserano+22).
