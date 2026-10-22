# F55 (2026-10-21): XS-stft-h and B-stft-h — the -ht runs without the template: is the phase reference needed?

**Why.** [F54](054-xs-b-stft-ht-phases-learned.md)'s -ht runs learned φ₀ and the ψ–φ₀ ridge
(B: the pinned φ₀ ± 2ψ within 2.1–3.4× of the MCMC at SNR 17–78). They combined three things,
each absent from one earlier failure:
- the STFT tokens (B-stft learned no φ₀, [F51](051-xs-stft-b-stft-results.md)/[F52](052-phi0-velocity-ignores-data.md));
- the helicity coordinates;
- the template residual (B-template, with the old coordinates and frequency-domain tokens,
  learned no φ₀, [F47](047-b-template-b-fisher-no-phi0.md)).

The template is also what F54 blames for the f₀ cost: 1.3–1.45× B-stft's width, and 213 sources
found against 217. The user asked for the run without it.

**The runs.**
- **`XS-stft-h`** (1.2M steps) and **`B-stft-h`** (200k steps): XS-stft-ht and B-stft-ht with
  `templates: true` removed.
- **The same otherwise:** STFT tokens, `polarisation: helicity`, RoPE, `decay_adamw: false`.
- **Read against:** 24/25 (with the template) and 23/22 (the old coordinates, no template).

**The collaborator's branch, for the record.**
- On 19 Oct `oneshot` held a refactor of `canna.lisa`. Its notable changes:
  - one-way attention with the data stream cached, and 16 draws per observation;
  - x₁ prediction with self-conditioning;
  - phase harmonics 2–6 and multi-scale f₀ Fourier features;
  - φ₀ referenced to mid-observation;
  - float32 by default;
  - the template and the Fisher loss dropped.
- On 20 Oct it was rewritten. It now sits on top of F54 and holds only reports: literature
  reviews (architecture, complex networks, diffusion, optimizer, GW/cosmology inference),
  benchmarks (simulator throughput, step time, checkpoint overhead), diagnostics built on
  `velocity_probe.py` and `phase_combinations.py`, and the research log.
- The refactored code those reports run (`LISA_CONFIG`, narrow/full bands) is on no pushed
  branch; the 19 Oct version survives locally as `c2ca068`.
- Its reviews agree with F52's reading of φ₀: a phase that can be read only against a coherent
  reference gives gradient descent no partial credit. Their first-ranked fixes are the exact
  phase conditionals (von Mises in φ±) and a demodulated (template-referenced) input.

**Predictions, written before the result:**
1. **Without the template, φ± go back to the prior:**
   - the phase-rotation response falls to ~0.001 (24/25: 0.53–0.64 at 0.3 rad);
   - the pinned φ± is ≥ 20× the MCMC at SNR 17–78.

   This is F52's reading: helicity removes the four (ψ, φ₀) copies but gives no reference.
2. **If instead φ± stay learned** (the pinned combination ≤ 6× the MCMC at mid SNR on B, and
   the rotation response ≥ 0.3), the template is not needed. It is then dropped, and f₀ should
   come back to B-stft's (per source ≤ 1.1×, ≥ 216 of 240 found).
3. **In either case,** f₀, A, sky and ι within 10% of B-stft (22) and XS-stft (23).
4. **Readout:** the velocity and rotation probes at B epoch ~50 and XS epoch ~500 (they settled
   F53 mid-run), then the phase combinations and width ratios
   ([flow-vs-reference](../posterior-eval/flow-vs-reference/)), the MCMC tables and `compare_ht.py`.
