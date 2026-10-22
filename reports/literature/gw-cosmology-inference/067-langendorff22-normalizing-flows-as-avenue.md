# Normalizing flows as an avenue to study overlapping gravitational wave signals

Langendorff et al., 2022; PRL 130, 171402 (2023) · [arXiv:2211.15097](https://arxiv.org/abs/2211.15097)

**Evidence: B.** Peer-reviewed (Physical Review Letters); 27 citations in 3.9 yr (Semantic
Scholar, 2026-10-19).

## What they do

Conditional *continuous* normalizing flow (neural-ODE CNF, trained by maximum likelihood / forward KL, ODE solver MALI, trace instead of log-det) mapping a fixed (non-conditional) 30-D Gaussian to the joint posterior of two overlapping precessing BBHs (15 params each). Data compressed by an SVD basis (256 vectors) whose vectors are used as the *kernels of the first 1-D convolution* of a residual CNN, so the coalescence time is sampled like any other parameter (no Gibbs time-shift sampling as in DINGO). Angles embedded on a circle (phase, polarisation, spin azimuths) and on a sphere (sky).

## What they find

3-detector HLV design sensitivity, 8 s frames, f_low 20 Hz, IMRPhenomPv2; M_c ~ U(10,100), q ~ U(0.125,1), network SNR ~ Beta-shaped on [10,50] centred at 20 (distance fixed then rescaled to SNR); both t_c ~ U(t_ref ± 0.05 s) (the "high-bias" regime). P-P plots on 1000 overlapped events lie on the diagonal but with a visibly broader spread than single-signal DINGO. Against 15 joint-PE (bilby/dynesty, Janquart+ ref.32) injections: NF posteriors are *broader*, but the Bayesian joint PE was overconfident (injection outside 90% CI) in 4/15, the NF in none. Inference ~1 s vs O(20 days). Training ~12 days on one GTX 1080 (conclusions say "about one week").

## Implementation details

On-the-fly data generation, new noise per frame; static base (conditional base gave no benefit on toys); labels "A"/"B" with no explicit ordering in training — they mention time-ordering samples *a posteriori* as "a trick used to alleviate degeneracies" and attribute part of the P-P spread to bimodality from the A/B exchange symmetry when events are similar. No network sizes, no learning rates reported. Suggested importance sampling to fix broadness but did not do it.

## Caveats

4-page letter, no ablations (CNF vs discrete, SVD-CNN vs plain, circle embedding vs raw angles are all asserted, not tested); only 15 Bayesian comparisons; posteriors are visibly wider than the Bayesian ones, so "calibrated but imprecise". Label-switching handling is ad hoc.
