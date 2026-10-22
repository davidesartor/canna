# F53 (2026-10-19): XS-stft-ht and B-stft-ht — the STFT runs with the helicity phases and the template residual

**The user's request.** Start from 22 (B-stft; XS-stft for XS) and change two things:
1. Infer (φ₀ + 2ψ, φ₀ − 2ψ) instead of (φ₀, ψ).
2. Give the template, as in B-template.

**The changes:**
- **`polarisation: helicity`** (F49). The orientation is sin ι on [−1, 1], with φ₊ and φ₋ each
  on a circle. The geometry, checked:
  - The map (ψ, φ₀) → (φ₊, φ₋) is 4 to 1, and the points that land together are exactly the
    likelihood's symmetric copies.
  - Face-on measures φ₋ only, face-off φ₊ only (jaxgb).
  - The prior, and so the base draws, is uniform on the product, with the two phases
    independent.
  - Geodesics stay on the manifold, and exp_map(log_map) returns the target
    (`tests/lisa/test_helicity.py`).
  - At face-on, φ₊ has no physical meaning and the flow keeps it at its prior, with no
    singularity in the chart. The old chart had φ₀ winding twice around ψ's pole there.
- **`templates: true`** (F46). The data minus the sources at x_t enter as a second group of
  STFT features beside the data's own (`network.stft_tokens`, one group per 6 channels). The
  sampler rebuilds them at every step. That is 24,576 input features per token on B (12,288 in
  B-stft).
- **Everything else is 22 / 23:** STFT tokens, RoPE, `decay_adamw: false`, 200k (B) and 1.2M
  (XS) steps.

**What to read.**
- **The φ₀ ± 2ψ widths against the MCMC** (`reports/diagnostics/draw-widths/compare_draws.py at 09abb90`):
  uniform in every run so far, 29–2248× the MCMC.
- **The phase-rotation probe** (the rotation test of
  [`velocity_probe.py`](../architecture/data-dependence/scripts/velocity_probe.py)): does the phase
  velocity now depend on the data?
- **f₀ against 22 / 23.** B-template's 0.33-bin f₀ floor (F47) is the risk to watch.

**Mid-run read** (19 Oct, ~09:00 CEST; XS-stft-ht at epoch 544 of 1200, B-stft-ht at epoch 79
of 200; probes run on the jobs' own GPUs):
- **The phases are learned, for the first time since F39:**

  | | φ velocity spread / target | R², s < 0.5 | head rows / median | phase error, head vs x_t (s < 0.5) |
  |---|---|---|---|---|
  | every run since F39 (final) | 0.003–0.005 | 0.000 | 0.06–0.14 | equal (1.1 rad) |
  | XS-stft-ht, epoch 544 | 0.57–0.60 | 0.37–0.48 | 0.43–0.52 | 0.26 vs 1.11 rad |
  | B-stft-ht, epoch 79 | 0.60–0.61 | 0.64–0.72 | 1.27–1.35 | 0.19 vs 1.12 rad |

- **They read the data's phase.** Rotating the data by 0.3 rad moves the φ± velocities by
  0.53–0.62 of their scale, and by π/2 by 1.0–1.4 (every run before: 0.001).
  - With the template, the rotation also leaves a phase mismatch in the residual, which moves
    the other coordinates too: by 0.05–0.45 at 0.3 rad on XS's sky and f₀. (The probe now
    rebuilds the residual from the rotated data.)
- **Resources:** B uses 27 of 80 GB of GPU memory at 97% utilisation and 2.1 GB of host memory
  of the 56 GB allocated; XS uses 1.0 GB of host memory. Both are well inside their
  allocations.
- **Next:** the final runs' φ₀ ± 2ψ widths against the MCMC.
