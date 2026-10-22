# Applications of machine learning in gravitational wave research with current interferometric detectors

Cuoco et al., 2025, Living Rev. Rel. 28, 2 · [arXiv:2412.15046](https://arxiv.org/abs/2412.15046)

**Evidence: B.** Peer-reviewed (Living Reviews in Relativity); 19 citations in 1.8 yr (Semantic
Scholar, 2026-10-19).

## What they do

an overview of ML across the LVK workflow:
- noise and glitches, denoising, waveform modelling;
- detection (CBC, burst, CW, stochastic);
- parameter estimation, populations, Kaggle challenges;
- next-generation detectors.

## What they find

- nessai needs 2× fewer likelihood calls than dynesty; i-nessai needs 2.68–13.3× fewer.
- VItamin (CVAE) reaches 1-D JSD O(10⁻²) nats against MCMC/NS. Their significance threshold, from Romero-Shaw+20, is 0.002 nats.
- DINGO reaches JSD O(10⁻³) nats.
- DINGO-IS needs O(10³) likelihood evaluations versus O(10⁶) for MCMC/NS, and adds only minutes.

## Implementation details

- **JSD on 1-D marginals is the community's standard comparison metric**, with 0.002 nats as the "statistically indistinguishable" bar.
- Self-supervised neural symmetry embeddings (Chatterjee+23) are mentioned for shrinking SBI models.
- Mao+24 (LISA) calibrate credible-interval coverage with autoencoders and ANNs.
- CW parameter estimation by ML is "still lacking" as of 2024. Only the SOAP + CVAE approach exists, and it works at high SNR only.

## Caveats

- Ground-based focus; LISA gets a single paragraph in §10.
- No overlapping-signal parameter estimation and no flow matching.
- Its KL formula (Eq. 3) contains a typo: log p / log q instead of log(p/q).
