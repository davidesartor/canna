# F61 (2026-10-25): a float32 backbone at eval halves f₀'s width, and only f₀'s, but the narrower f₀ is overconfident: half the floor is bf16 rounding, the rest a learned f₀ error

- **f₀ narrows, nothing else moves:** with `LISA_BF16=none` on the same checkpoints, injections,
  noise and base draws, f₀'s width is 0.58× (narrow-S) and 0.43× (full-B) the bf16 one, median
  per eval source; M_c, A, sky, ψ and ι are within 1%, and φ₀ (which follows f₀) is 0.97× and 0.51×.
  full-B's loud f₀ goes from 0.44 to 0.21 bins, 167× → 93× the MCMC above SNR 300.
- **But it is overconfident:** on the 60 paired scorecard injections (240 sources), full-B's in68
  falls from 0.87–0.95 to 0.49–0.63 above SNR 15 and in95 from 1.00 to 0.83–0.92; narrow-S's
  from 0.70–0.78 to 0.53–0.59 above SNR 40. The bf16 rounding was spreading the draws over an
  f₀ estimate that is itself off by about its float32 width.
- **So bf16 is half of [F60](060-sampler-grid-not-the-f0-floor.md)'s floor, not all:** the
  [precision-error](../architecture/precision-error/README.md) probe on the trained runs puts
  the bf16 backbone's |Δx̂₁| at a median 0.02–0.08 posterior widths but 280–810 on full-B's
  loudest sources, and the sampled draws move by up to 254 widths (narrow-S: up to 29 in
  `attention` mode, 0.012 median). The rest is the network's own f₀ error, learned in bf16.
- **Evaluating in float32 is not the fix:** it trades a conservative floor for overconfident
  draws. The fix belongs in training: predict x̂₁ = x_t + (1 − s)·head(h), so the rounding and
  the network's absolute f₀ error both shrink along the path, as on fml (velocity prediction);
  or F59's time_power 3 and learning rates.

**Runs.** TREX A100, at the checkpoints' commit 3e5bdd4: eval 15713491 / 15713494 and scorecard
(`LISA_N_RANDOM=50 LISA_N_VALIDATION=0`, the bf16 scorecard's first 50 random injections plus the
10 eval ones) 15713492 / 15713495, 3–5 min each; precision-error probe 15713493 / 15713496 on
GPU. The evals are `outputs/runs/<run>/eval/<epoch>-average-32-none/` (eval dirs now carry the
precision when it is not the default, `config.eval_tag`), with their `mcmc report` and
corner.html.

**f₀ width [bins] on the 40 eval sources, median per SNR band (flow / MCMC in brackets):**

| run | SNR | bf16 | float32 |
|---|---|---|---|
| narrow-S | 15–40 | 0.018 (1.1×) | 0.008 (0.4×) |
| narrow-S | 100–300 | 0.027 (9.0×) | 0.015 (3.9×) |
| narrow-S | ≥ 300 | 0.025 (31×) | 0.013 (15×) |
| full-B | 15–40 | 0.48 (17×) | 0.18 (7.4×) |
| full-B | 100–300 | 0.45 (70×) | 0.14 (29×) |
| full-B | ≥ 300 | 0.44 (167×) | 0.21 (93×) |
