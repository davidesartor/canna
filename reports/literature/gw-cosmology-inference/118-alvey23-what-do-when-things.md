# What to do when things get crowded? Scalable joint analysis of overlapping gravitational wave signals

Alvey, Bhardwaj et al., 2023; arXiv letter · [arXiv:2308.06318](https://arxiv.org/abs/2308.06318)

**Evidence: C.** Not peer-reviewed; 14 citations in 3.2 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Apply peregrine (TMNRE, swyft; see 2304.02035) unchanged to the joint 30-D posterior of two overlapping precessing BBHs. Learn only 1-D and 2-D *marginal* ratio estimators, truncating the prior over 7 sequential rounds to the region of non-negligible posterior mass (ε = 1e-5). Not amortised over data: one observation per analysis ("local amortisation" inside the truncated prior).

## What they find

HLV O4 PSD, 2048 Hz, 4 s, IMRPhenomXPHM, both signals network SNR 20 with similar M_c/q; Δt_c = 0.05 / 0.2 / 0.5 s (cases C1–C3). Joint 1-D marginals almost as narrow as the single-signal TMNRE baseline (slightly broader, e.g. detector-frame M_c); t_c of the second signal becomes bimodal because GW1's t_c is bimodal (cross-signal degeneracy). Chirp-mass widths shrink as Δt_c grows; the two M_c are nearly uncorrelated. Coverage on 2000 truncated-prior sims: on the diagonal for all 30 params. Cost: 7×10⁶ waveforms (10⁶/round × 7), only ~10× the single-signal count for twice the params, "≈15% of the waveform calls needed for a single signal with MCMC"; 2–3 h simulation/round on 18 cores, 8–12 h training/round on one A100, ~48 h total vs 23.8 days for dynesty joint PE.

## Implementation details

Label switching removed by the *prior*: t_c ~ U(−0.1, 0.1) s for GW1 and Δt_c ~ U(0, 1) s for GW2 (so time-ordered slots). Noise shuffling (re-pairing noise realisations with signals) against overfitting; batch 512; same embedding net as peregrine. Distance prior uniform in comoving volume, d_L 100–2500 Mpc.

## Caveats

Three hand-picked injections only (coverage tests use the truncated prior, i.e. local, not global calibration); not amortised (each new event needs ~48 h); marginal-only, so joint samples/corner plots across >2 params are not available; no comparison to an actual joint MCMC.
