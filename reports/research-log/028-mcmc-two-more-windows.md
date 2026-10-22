# F28 (2026-10-11): MCMC on two more windows — the flow misses sources next to louder ones, which the MCMC localises

- 6 of 16 sources are missed by the flow and all 6 are localised by the MCMC; 5 of them sit
  within 2.6 bins of a 6–90× louder source.
- Loud f₀ ratios grow with SNR (fixed absolute floor); coverage 127/128.

**Setup.** F27's protocol on eval windows 2 (q0.30: SNR 9.1, 37.2, 111, 235) and 7 (q0.80: SNR
9.3, 9.5, 64, 819), chosen because each has a faint/moderate source within 0.2 bins of a loud
one. Two seeds each, 14.6–18.1 min per run (`outputs/mcmc/run_more.sh`); `compare_mcmc.py
--windows 2 7` vs XS-late-768-cool (`outputs/mcmc/compare_windows27.log`). The paper tables now
carry all four windows (q1.00, q0.60, q0.30, q0.80).

**MCMC converged:** seeds within 4.3% in every width, W1 ≤ 0.06 σ (SNR-9.3 source 0.22 σ),
R-hat 1.0000, ESS ≥ 1.2×10⁵.

**Fisher:** f₀ within 3% of MCMC at SNR ≥ 62 (all 8 such sources), within 16% below. Other
parameters off by up to 4.5 for the nearly face-on SNR-819 source (cos ι = 0.995), where the
linearisation spreads along degeneracies the posterior bounds.

**Flow vs MCMC, four windows, 16 sources:**
- **Coverage:** MCMC median in the flow's 95% for **127/128** pairs (exception: sky latitude of
  a missed SNR-9.3 source), 68% for 112/128.
- **Loud f₀ ratios:** 6.4 (SNR 111), 11 (235), 38 (567), 44 (819), 66 (1245), 84 (1366) — a
  fixed absolute floor against MCMC's 1/SNR. Other parameters 3–8.5× above SNR 500.
- **Six missed** (56–89% of draws > 1 bin off), all localised by MCMC (f₀ within 16% of
  Fisher). Five neighbour a louder source: 14.4 next to 1245, 27.1 next to 567, 37.2 next to
  235 (0.16 bins), 9.1 next to 111, 9.3 next to 819 (0.21 bins). The sixth (SNR 9.5) is 4.9
  bins from the other SNR-9 source.
- **A found SNR-64 source 1.1 bins from the SNR-819 one** is 6–11× too wide in every parameter,
  8% of draws off; isolated moderate sources stay at 1–2.6×.
- **Caveat:** "draws off the source" (here and F27's ~20%) are assignments by `match_sources`
  (which weighs amplitude and sky as well as f₀), not draws placed elsewhere. An unlocked slot
  can be handed a bright source's slot although almost every draw has a slot on it
  (`app:evaluation-matching`: 99.7% in q0.60). The paper now says so.

**Reading.** Detection fails where a faint source shares bins with a loud one that the flow
places only to its floor (38–84× MCMC), leaving a mismodelled residual comparable to the faint
source's SNR. Removing the floor (F30) should then help crowded detection; to check on R1's
final model against these four windows (no new MCMC needed).
