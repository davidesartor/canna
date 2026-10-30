# bf16 precision error: how far each `bf16` mode moves the flow from float32, in posterior widths (2026-10-24)

## Takeaways

- **Trained runs (2026-10-25, research log [F61](../../research-log/061-fp32-backbone-halves-f0-but-overconfident.md)):**
  `misty-meadow-narrow-S` and `lucky-falcon-full-B` on a TREX A100. In `backbone` mode full-B's
  |Δx̂₁| is a median 0.02–0.08 posterior widths but 280–810 on its loudest sources, and the
  sampled draws move by up to 254 widths; `attention` mode stays below 0.04 (median). A float32
  eval halves f₀'s width and only f₀'s. The smoke-run notes below are kept for the record.

- **Smoke data only.** The only checkpoint on Unity is dusty-otter-narrow-XS (2000 steps), measured
  on CPU, so these numbers show the mechanism and the tooling, not a trained flow's error. Rerun on
  narrow-B and full-B on a GPU (commands below) before quoting any of it.
- **The recording copies of `encode`/`decode` reproduce the unpatched forward pass** (max |Δ| 1.7e-6,
  float32), and the float64 floor sits at about 1e-7 relative token error and 2e-6 σ in x̂₁, so
  the float32 reference is clean.
- **Relative token error at the backbone output (CPU, smoke run, median):** `attention` 2e-3 at t = 0
  falling to 1e-4 by t = 0.99, and `backbone` 8e-3 falling to 4e-3, which is about the bf16 roundoff
  2^-8 = 4e-3. So bf16 backbone error does not accumulate past one rounding.
- **In posterior widths (median over sources and coordinates):** `backbone` |Δx̂₁| is 0.12 σ at
  t = 0 and 0.04 σ at t ≥ 0.99, and 0.14 σ of draw displacement after the 32-step sampler.
  `attention` is 0.03 σ at t = 0 and below 0.001 σ at t ≥ 0.99, and 0.06 σ after the sampler.
  - **The maximum matters more than the median:** the worst source and coordinate reaches
    100 to 300 σ in `backbone` mode (127 σ in `attention` mode). That is the high-SNR sources, whose
    Fisher width is tiny; relative error is constant but the width shrinks as 1/SNR. The page's SNR
    scatter shows which sources and coordinates these are. This is the number to check on a trained
    run.
  - Sample width ratios to float32 span 0.90 to 1.31 (`backbone`) and 0.89 to 1.19 (`attention`),
    against 0.97 to 1.03 for float64.
- **Relative velocity error grows toward late t** in `backbone` mode (0.005 at t = 0 to 0.18 at
  t = 0.9999), since the velocity divides x̂₁ − x by 1 − s (now 1 − t).
- **A 1σ move is smaller than the bf16 rounding of the decode-input token for some (source,
  coordinate)**: the ratio has median 4.9 and minimum 0.057.

## What the page shows

[precision-error.html](precision-error.html). The top bar picks a run and the platform it was
measured on (`<run>-cpu` or `<run>-gpu`). A banner flags smoke runs and CPU measurements.

- **Tiles:** the run, the device, and backbone mode's median x̂₁ error at t = 0.99 and after the
  sampler, both in posterior widths.
- **Where the error enters:** the relative token error ‖h_mode − h_f32‖ / ‖h_f32‖ of the source
  stream after every decode block (median and 90th percentile), at a chosen t. The bf16 roundoff
  2⁻⁸ is marked. A table gives the same error for the cached data stream (tokens, keys, values)
  per encode block.
- **x̂₁ error in widths:** |Δx̂₁| / σ per source and coordinate against the source's SNR, at a
  chosen t and coordinate. A table gives the relative velocity error per mode and t.
- **After the sampler:** the default sampler from the same base draws in every mode. It shows the
  draw displacement in widths against SNR, plus each mode's sample-width ratio to float32.
- **Token resolution:** how a 1σ move of each coordinate changes the float32 decode-input token,
  against the error from rounding that token to bf16.
- **What this means.**

Speed and memory are in [step-time-memory](../../benchmarks/step-time-memory/). This report is
accuracy only.

## Method

`scripts/precision_probe.py` runs on the run's own commit (via `slurm/run-src.sh`):

- **Modes.** The trained parameters (float32) are put into four network structures that differ only
  in their static dtypes, as `train.setup` builds them:
  - `none`: float32 backbone, no attention cast. This is the reference.
  - `attention`: float32 backbone, bf16 attention.
  - `backbone`: bf16 backbone and attention. This is the training default.
  - `f64`: a float64 backbone, the reference's own floor. Attention logits stay float32 through
    `preferred_element_type`. The mode is dropped if XLA rejects that.
  - Embeddings, modulation and unembedding are float32 in every mode, as in the code.
- **Inputs.** The 10 eval injections at the SNR deciles (`eval_injections(problem, seed)`), with
  observation, base draws and noise keys built exactly as `eval.py` builds them.
- **Recording.** `MMDiT.encode` and `MMDiT.decode` are swapped for copies that also return every
  block's activations: the data tokens, keys and values, and the source stream entering each block
  and leaving the last. A check against the unpatched forward pass is stored as `patch_check`.
- **States.** For each t in {0, 0.5, 0.9, 0.99, 0.999, 0.9999}:
  - 16 base draws are moved along the straight path to the truth, x_t = (1 − t) u + t x₁.
  - x̂₁ is the head's mean m, from one network call. data/ predates this: it was measured with a
    time power and self-conditioning on the float32 network's own first-call prediction.
- **Widths.** The posterior width per source and coordinate is 1.4826 × MAD of the Fisher forecast
  draws (`fisher_samples`, 1024 draws, the same draws `eval.py` uses), mapped to flow coordinates.
  - The mcmc reference report checks the Fisher forecast against MCMC. Above SNR 40, Fisher ÷ MCMC
    width lies in 0.8–1.25 for about 90% of source parameters. Below SNR 40 it spans 0.35–2.63.
  - So low-SNR widths here are good to a factor of about 2.
  - Fisher draws in Mc can be negative at low SNR, which gives NaN in flow coordinates. MAD ignores
    them.
- **End to end.** `sample_posterior` with the run's `ode_steps` (midpoint steps), from the same 64
  base draws in every mode. It reports the median displacement from the float32 draws in widths,
  and the ratio of the sample widths.
- **Token resolution.** At the truth (t = 0.99), each coordinate in turn is moved by 1σ, and
  the change in the float32 decode-input token is compared with that token's bf16 rounding error.
- **Platform.**
  - On CPU (`JAX_PLATFORMS=cpu`), the casts are exactly the modes', but XLA's CPU bf16 matmuls
    accumulate differently from GPU tensor cores, and attention never takes the cuDNN path.
  - On GPU the sbatch sources `slurm/gpu.sh` as for a float32 run (no GEMM autotuning, highest matmul
    precision), so the reference is true float32. The bf16 modes then run the bf16 kernels that
    training uses.
  - In training, a `backbone` run's float32 embedding matmuls use XLA's default (TF32) precision
    instead. That difference is not measured here.

## Rebuild

From the repo root, CPU:

    source slurm/unity.env; sbatch $CANNA_SBATCH_CPU reports/architecture/precision-error/scripts/probe.sbatch <run>

GPU, for the numbers that describe training's own arithmetic:

    source slurm/unity.env; sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/architecture/precision-error/scripts/probe.sbatch <run>

The job writes `data/<run>-<platform>.json` and rebuilds the page from every JSON in `data/`. To
rebuild only the page (stdlib only): `uv run --frozen python reports/architecture/precision-error/scripts/build_page.py`.

## Data

`data/<run>-<platform>.json`, one per run and platform.
- **Run and platform:** run, config, commit, epoch, `run_bf16`, device, platform, `matmul_precision`.
- **Setup:** `modes`, `reference`, `t_grid`, `patch_check`.
- **Per injection:** `snr` [10, S]; `width` [10, S, 8].
- **Per block** (median and p90 per block): `encode[mode][y|k|v]` [L, 2]; `decode[mode]` [t, L+1, 2].
- **Output error:** `x1_over_width[mode]` [10, t, S, 8]; `velocity[mode]` [10, t].
- **End to end:** `e2e_displacement[mode]`, `e2e_width_ratio[mode]` [10, S, 8].
- **Token resolution:** `token_ratio` [10, S, 8]; `token_rounding` [10, S].

Present: `dusty-otter-narrow-XS-cpu` (smoke, 2000 steps, commit 11fc8a4, CPU, job 66014389, 7 min).
