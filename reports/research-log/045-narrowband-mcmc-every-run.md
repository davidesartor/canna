# F45 (2026-10-15): B-rope is within 1–2× of the narrowband MCMC at mid SNR; the flow never learns φ₀, and loud sources' sky stays 16–106× wide

- B-rope at SNR 17–78: f₀ 1.9–4.1× the MCMC, other parameters 1.0–2.1×, MCMC median inside the
  95% for 48 of 48 pairs.
- φ₀ is not learned: where the MCMC pins it, the flow is flat over [−π, π] (30–61×).
- Loud sources' sky, A and ι stay wide: up to 106× on B's loudest source.

**Request (the user):** benchmark every run against an MCMC of the corner pages' sources, each
source alone ("assuming independence") on its piece of the frequency stream; laptop for runs
15–18, cluster from now on.

**Method** (`canna.lisa.mcmc`, 6ad125a):
- **Sources:** gb-loudest, gb-median, gb-faintest and q0.10's four, picked as eval.py picks
  them, on the eval's noise realisation.
- **Likelihood:** the other three sources subtracted at their true values (the independence
  assumption); reads only the source's own `response_points` band, which moves with θ (16 bins
  on XS, 1024 on B). Exact for the whole window (`tests/lisa/test_mcmc.py`).
- **Sampler:** jexplore, 32 walkers × 6 temperatures, the training prior, angles on 2π boxes
  centred on the truth, walkers started on the Fisher ellipse. Burn-in 10k, 30k kept, thinned
  by 10, two seeds.
- **Validation:** on XS's loudest source (window 9, SNR 1366) it matches F27's joint
  32-parameter MCMC within 1–4% in every width (f₀ 0.000441 vs 0.000444 bins).
- **Convergence:** R-hat ≤ 1.0001, ESS ≥ 28k. Seeds agree within 0.3–2%, except each problem's
  faintest source (SNR 5.3, 7.3): broad non-Gaussian f₀, robust widths differ 20–27%.
- **Cost:** laptop GPU XS 2.0–3.1 min per chain, B 5.3–7.6 min. TREX CPU (8 cores): ~5
  ms/iteration XS, ~50 ms B; one array task per chain, ~5 min wall for XS, ~35 min for B.

**XS** (flow / MCMC robust width, x; XS-fixed / XS-rope):

| source | SNR | f₀ | A | sky (lon, sin lat) | ψ | sin ι | φ₀ |
|---|---|---|---|---|---|---|---|
| gb-loudest | 1366 | 52 / 28 | 4.7 / 4.7 | 16, 17 / 16, 17 | 10 / 10 | 5.0 / 4.7 | 20 / 20 |
| q0.10 s0 | 98 | 5.6 / 4.8 | 5.9 / 9.2 | 2.9, 2.3 / 3.1, 2.4 | 4.9 / 7.8 | 6.3 / 11 | **60 / 61** |
| gb-median | 62 | 6.4 / 7.6 | 2.3 / 3.1 | 9.7, 5.9 / 12, 6.0 | 1.1 / 0.8 | 2.5 / 2.8 | 2.2 / 2.1 |
| q0.10 s1 | 59 | 3.0 / 2.8 | 1.8 / 1.9 | 1.9, 2.1 / 2.0, 1.9 | 1.2 / 1.5 | 1.6 / 1.6 | **46 / 47** |
| q0.10 s2 | 34 | 4.4 / 3.3 | 2.0 / 2.3 | 1.6, 1.4 / 2.2, 1.6 | 1.7 / 1.9 | 1.7 / 2.1 | **30 / 31** |
| q0.10 s3 | 17.5 | 415 / 134 | 1.2 / 2.1 | 24, 24 / 24, 24 | 0.7 / 0.7 | 1.9 / 2.3 | 1.5 / 1.5 |
| gb-faintest | 5.3 | 3.6 / 4.6 | 8.1 / 6.1 | 1.7, 1.0 / 2.8, 1.4 | 1.4 / 1.3 | 1.5 / 1.4 | 1.8 / 1.9 |

MCMC median outside the flow's 95% for 2 of 56 pairs in each run (q0.10 s0's A and sin ι).
q0.10 s3 is slot confusion with its louder neighbours.

**B** (B-fixed / B-rope):

| source | SNR | f₀ | A | sky (lon, sin lat) | ψ | sin ι | φ₀ |
|---|---|---|---|---|---|---|---|
| gb-loudest | 1668 | 71177 / **56** | 2.6 / 2.8 | 49, 77 / **66, 106** | 1.9 / 2.3 | 2.3 / 2.6 | 4.3 / 4.2 |
| q0.10 s0 | 78 | 146 / **3.5** | 1.6 / 1.6 | 2.0, 1.7 / 2.1, 1.6 | 1.3 / 1.4 | 1.6 / 1.5 | **48 / 47** |
| q0.10 s1 | 68 | 304 / **4.1** | 1.0 / 1.0 | 2.1, 1.7 / 1.9, 1.7 | 1.7 / 1.6 | 1.0 / 1.0 | 4.7 / 4.7 |
| gb-median = q0.10 s2 | 61 | 289 / **3.6** | 1.1 / 1.1 | 1.5, 1.5 / 1.4, 1.5 | 1.1 / 1.2 | 1.1 / 1.1 | 1.8 / 1.8 |
| q0.10 s3 | 16.7 | 73 / **1.9** | 1.0 / 1.0 | 1.4, 1.4 / 1.3, 1.2 | 1.0 / 1.0 | 1.0 / 1.0 | 2.4 / 2.4 |
| gb-faintest | 7.3 | 4925 / 433 | 1.8 / 1.6 | 16, 4.8 / 5.4, 2.5 | 1.5 / 1.1 | 1.3 / 1.4 | 2.1 / 2.1 |

- **B-rope at SNR 17–78:** f₀ 1.9–4.1×, A 1.0–1.6×, sky 1.2–2.1×, ψ 1.0–1.6×, ι 1.0–1.5×; MCMC
  median inside the 95% for 48 of 48 pairs (B-fixed 47); draws > 1 bin off in f₀ 1–5% (B-fixed
  40–75%). This is F44's 70× against the true posterior; B-fixed was 73–304× in f₀ here.
- **The loudest source falls short:** f₀ 56×, sky 66–106×. RoPE fixed the frequency, not the
  sky at SNR 1668 (XS: 16–17×).

**The angles** (the user's and the collaborator's point):
1. **φ₀ is not learned.** Where the MCMC pins φ₀ (±0.04–0.07 rad: XS q0.10 s0–s2 and B q0.10
   s0, inclinations 16–27° from edge-on), the flow's φ₀ is flat over [−π, π]: 30–61× the MCMC
   width, even when ψ is right (XS q0.10 s1: ψ to ±0.03 rad, the MCMC's width). Where the true
   φ₀ posterior is broad (near face-on), the flow matches it (1.5–4.7×).
2. **Sky, A and ι at loud SNR:** 5–17× on XS, 3–106× on B, worst on B's loudest. Within ~2× on
   B at SNR ≤ 100.
3. **ψ** within 1–2× except the loudest sources (2.3–10×).
