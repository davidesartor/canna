# Interpretation of Rank Histograms for Verifying Ensemble Forecasts

Hamill, 2001, Mon. Weather Rev. 129(3), 550–560 · [DOI:10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2](https://doi.org/10.1175/1520-0493(2001)129%3C0550:IORHFV%3E2.0.CO;2)

**Evidence: A.** Peer-reviewed (Monthly Weather Review); 926 citations in 25.6 yr (Semantic
Scholar, 2026-10-22).

## What they do

A Notes and Correspondence piece on how *not* to read a rank histogram. With n sorted members, the verification falls into one of n + 1 bins. If the verification and the members are draws from the same distribution, E[P(V < x_i)] = i/(n + 1) (Eqs. 1–4), so every rank is equally likely and the histogram is flat. Ties (many members equal, as for zero precipitation) need special rules (Hamill & Colucci 1997, 1998).

The paper then builds synthetic counterexamples in which a flat histogram coexists with a bad ensemble, and ones in which a non-flat histogram has a cause other than the ensemble.

## What they find

- **Standard shapes** (Fig. 1; verification N(0, 1), 25 members from N(μ, σ), 10,000 tallies). Too little spread gives a U. Too much spread overpopulates the middle ranks. A positive or negative bias overpopulates one extreme.
- **Reliability only** (§2). The histogram measures reliability, not resolution (sharpness). Of two calibrated systems a user prefers the more specific one, so use it with ROC, Brier or ranked probability scores.
- **"A uniform rank histogram is a necessary but not sufficient criterion"** (§3a). Three constructions:
  - Fig. 2: an equal mix of three ensemble types, N(−0.5, 1), N(0.5, 1) and N(0, 1.3) (σ in the second argument), against N(0, 1) truth, 30,000 tallies. Two are biased and one is unbiased but overdispersed. The pooled histogram is "relatively uniform".
  - Fig. 3: members drawn non-randomly from N(0, 0.7) with acceptance I(X) ∝ N(0, 1)/N(0, 0.7), 10,000 tallies, meant as an analogy to singular-vector perturbations. The histogram is approximately uniform. (Our reading: this is rejection sampling, so the members are marginally N(0, 1) anyway.)
  - Fig. 4: N(0, 1) truth, ensembles drawn half the time from N(−1, 1) and half from N(1, 1), 20,000 tallies. The pooled histogram is a U "indistinguishable" from that of an underdispersed N(0, 0.69) ensemble. Conditional biases of opposite sign masquerade as a dispersion error.
- **"Reliability alone is not a good metric of forecast quality"** (§3a): it "apparently can be achieved even if" ensemble and verification are not drawn from the same distribution. Citing Gilmour & Smith (1997) and Smith (1999), reliability "may be illusory" unless a model trajectory can be found that shadows the real atmosphere within analysis uncertainty.
- **Remedy for cancellation.** If the ensemble may behave differently by regime (synoptic situation, region, season), build a histogram per regime; a shape that differs between regimes indicates conditional biases. Hamill & Colucci (1997, 1998) found very different shapes for small and large ensemble spread. Spread is a forecast-side statistic, so the stratum is known without the truth.
- **Flat margins, wrong joint** (§3b, Fig. 5; 25 members at two grid points with variance 1 and zero covariance, truth covariance 0.5, 10,000 tallies). The histograms of x₁ and x₂ are flat; the one of x₁ − x₂ is not. Our arithmetic: the members' difference has variance 2 and the truth's 1, so it is overdispersed (a ∩; the figure's shape is not stated in the text). The paper recommends checking derived quantities (wind from geopotential, vorticity) and the minimum spanning tree rank (Smith 1999).
- **Observation error** (§3c, Fig. 6; 25 members from N(0, 1), observation = N(0, 1) + N(0, s), s = 0.33, 0.67, 1.00, 10,000 tallies). For s ≪ 1 the histogram stays near uniform; as s grows the extreme ranks fill up, mimicking underdispersion. If observation errors are a significant fraction of the spread, rank the observation against members with simulated observation noise added (Anderson 1996).
- **Sampling noise and correlation** (§4). A χ² test of flatness is valid only for independent samples. Under the null, the difference D between the ranks of two samples has P(D) = (n + 1 − |D|)/(n + 1)² for |D| < n + 1 (Eq. 5), which gives the reference to measure rank correlation. In a quasigeostrophic channel model (129 × 65 grid, 125 km spacing), geopotential ranks are still correlated at a lag of 20 grid points (2500 km), while potential vorticity ranks are nearly uncorrelated by lag 6 (750 km). In time, ranks are slightly correlated at 1 day and effectively uncorrelated at 2–3 days for wind and PV. So many tallies over a domain are far fewer independent samples.

## Implementation details

- Bins: n members give n + 1 ranks. Pool tallies over cases and locations only after checking that they are not strongly correlated, or the histogram's apparent precision is false.
- Add observation error to members before ranking when the verification is noisy (§3c). For us the analogue is ranking a noisy truth: our injected parameters are exact, so this does not arise, but it would for real-data validation against a reference catalog.
- Check derived variables and differences between components, not only each component (§3b).
- Stratify by regime or by ensemble spread (§3a).

## Caveats

- All the demonstrations are synthetic one- or two-variable Gaussians, plus one QG model for the correlation section. No real forecast data.
- The paper proposes no test that detects the Fig. 2 or Fig. 4 cancellation from pooled data. Stratification only helps if the right strata are chosen, which needs knowing the regimes.
- The χ² discussion warns against misuse but proposes no corrected test. Later work (Elmore 2005; Jolliffe & Primo 2008) does.
- "Necessary but not sufficient" is the paper's claim for marginal reliability. It does not claim that a flat pooled histogram is uninformative; the U, ∩ and slope signatures of Fig. 1 still hold when the error is systematic across cases.
- For our flow, Fig. 2/4 is the q = prior problem in miniature: per-observation errors of opposite sign pool into a flat SBC histogram. Stratifying by a flow-side statistic (posterior width, SNR bin) is the paper's remedy transferred.
- Full text read (journal PDF, via text extraction).
