# F27 (2026-10-10): flow against MCMC on two XS windows — never biased, too wide when loud, misses one faint source per window

- Never biased: the MCMC median is inside the flow's 95% for all 64 source–parameter pairs
  (68%: 57/64).
- Loud sources are too wide: f₀ 38–84×, other parameters 4–8× the MCMC.
- Moderate found sources are near-optimal; one faint source per window is missed although the
  MCMC localises it.

**Setup** (protocol of `paper/appendix/mcmc.tex`).
- **Windows:** q1.00 (injection 9: SNR 14.4, 1245, 33, 1366; f = 0.10 mHz) and q0.60 (injection
  5, `fig:corner-median`: SNR 27.1, 26.3, 62, 567). Same noise realisation as eval.py, so the
  flow (`XS-late-768-cool`, 1024 eval draws) saw the same data.
- **Sampler** (`lisa_checks/mcmc_xs.py`, now
  [`joint_mcmc.py`](../mcmc/scripts/joint_mcmc.py)): jexplore, temperature swaps, DE or stretch moves; 64
  walkers × 6 temperatures (1–10), all four sources jointly (32 dims). Training prior (M_c
  density by quadrature, χ²/dof 0.72 vs 2×10⁶ draws), float64 likelihood, Fisher-draw starts.
  100k burn-in, 300k kept thinned by 100, two seeds, 14 min each on the laptop GPU.
- **Comparison** (`lisa_checks/compare_mcmc.py`, now
  [`compare_joint.py`](../posterior-eval/flow-vs-reference/scripts/compare_joint.py)): relabelled by `match_sources`; ψ, φ₀ folded
  by the likelihood's exact symmetries (ψ + π; ψ + π/2 with φ₀ + π), asserted.

**MCMC converged.** Seeds agree within 2% (window 5) and 7.5% (window 9), W1 ≤ 0.1 σ, except
the SNR-14 source's broad multimodal sky and φ₀ (W1 0.3–0.5 σ; sky 2× Fisher). At SNR ≥ 567
Fisher matches MCMC in f₀ to 2% and most others within 20% (A–ι and ψ–φ₀ pairs 0.8–1.4), so "×
ideal" is the right yardstick where the floor is quoted. Below SNR ~60 Fisher can be off 2×.

**Flow / MCMC width** (1.4826 MAD; "other" = A, sky, ψ, sin ι, φ₀; M_c prior-dominated in both,
0.95–1.06):

| window, source | SNR | f₀ | other, median [range] | flow draws > 1 bin off in f₀ |
|---|---|---|---|---|
| 9, 3 | 1366 | **84** | 5.6 [4.2–6.6] | 0.3% |
| 9, 1 | 1245 | **66** | 6.4 [3.6–8.4] | 0% |
| 5, 3 | 567 | **38** | 4.8 [4.1–5.8] | 0% |
| 5, 2 | 62 | 6.2 | 2.1 [0.5–6.8] | 18% |
| 9, 2 | 33 | 4.8 | 1.6 [1.0–2.1] | 20% |
| 5, 0 | 27.1 | **231** (missed) | 5.0 [2.8–54] | 87% |
| 5, 1 | 26.3 | 2.3 | 1.1 [1.0–1.2] | 0.3% |
| 9, 0 | 14.4 | 21 | 5.3 [2.0–9.5] | 58% |

- **Loud f₀** is F23/F26's floor (0.037 bins vs MCMC 0.0004–0.0013). The other parameters are
  4–8× too wide along the posterior's own degeneracies (`corner_window9_s3.pdf`: spread along
  A–ι). The precision limit is not only f₀'s, though f₀ costs most.
- **SNR 26–33 found sources:** 1.0–2.1× in every parameter but f₀ (2.3–4.8×).
- **Missed source (SNR 27.1):** MCMC localises it to 0.023 bins (×1.06 Fisher); the flow is
  231× wider. In ~20% of draws the brighter neighbour (SNR 33 or 62) is placed elsewhere: slots
  get confused when one is not locked on.
- **SNR-62 ψ, φ₀:** near-face-on degeneracy (sin ι ≈ 0.95), only a combination constrained; the
  0.5 ratio is not meaningful (W1 0.3 σ).
- **Cost:** 14 min laptop GPU per window and seed vs ~1 s per window for the flow on an A100.

**Consequences.**
- `tab:mcmc` and `fig:mcmc` can be filled from `outputs/mcmc/` (`compare_window{9,5}.npz`,
  `corner_window*_s*.pdf`).
- R1's position features target f₀ (and sky via token positions). If R1 fixes f₀ but not the
  4–8× elsewhere, the remaining limit is elsewhere.
