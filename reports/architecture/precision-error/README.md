# bf16 precision error: how far each `bf16` mode moves the flow from float32, in posterior widths (2026-11-02)

## Takeaways

Run `261101-royal-ferret-narrow-B` (epoch 1250, A100, job 66528662, 128 eval injections, 16 states
and 64 draws each, f0-sorted). The old `misty-meadow-narrow-S` and `lucky-falcon-full-B`
(research log [F61](../../research-log/061-fp32-backbone-halves-f0-but-overconfident.md)) and
the `dusty-otter-narrow-XS` smoke records predate f0 ordering and the frame model; they still
show on the page, and their JSONs are for you to delete. `dusty-island-full-B` was not probed.

- **bf16 error is small in posterior widths on narrow-B.** `backbone` mode (training's default):
  median |Δx₁|/width is 0.004 at t = 0 and 0.003 at t ≥ 0.99, and the 32-step sampler moves draws
  a median 0.008 widths. `attention` is 0.001 and 0.001, sampler 0.001. The old full-B
  and narrow-S medians were 10 to 100 times larger.
- **The maxima are not.** The worst source and coordinate reaches 22 widths (`backbone`, t = 0) and
  24 (t = 0.999), 7 in `attention`. After the sampler, `backbone` displaces a draw by up to 5
  widths and sample widths to float32 span 0.80 to 2.02 (`attention` 0.89 to 1.19).
- **Relative token error** at the backbone output: `attention` 2e-3 and `backbone` 6e-3 to 8e-3
  throughout, about the bf16 roundoff 2^-8 = 4e-3, so it does not accumulate past one rounding.
- **Relative velocity error grows toward late t** in `backbone` mode: 0.005 at t = 0 to 0.47 at
  t = 0.9999 (`attention` 0.002 to 0.034), since the velocity divides by 1 − t. The x₁ error does not grow
  with it.
- **A 1σ move is smaller than the bf16 rounding of the decode-input token for some (source,
  coordinate)**: the ratio has median 48 and minimum 0.16.
- **The recording copies of `encode`/`decode` reproduce the unpatched forward pass** (max |Δ|
  1.7e-6, float32). There is no float64 floor for this run: on the GPU the `f64` mode is
  dropped (cuBLAS has no float64 GEMM path here). The CPU smoke record has it (about 1e-7 relative
  token error), which is the old XS run.

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

`scripts/precision_probe.py` runs on the working tree's `src`:

- **Modes.** The trained parameters (float32) are put into four network structures that differ only
  in their static dtypes, as `train.setup` builds them:
  - `none`: float32 backbone, no attention cast. This is the reference.
  - `attention`: float32 backbone, bf16 attention.
  - `backbone`: bf16 backbone and attention. This is the training default.
  - `f64`: a float64 backbone, the reference's own floor. Attention logits stay float32 through
    `preferred_element_type`. The mode is dropped if XLA rejects that.
  - Embeddings, modulation and unembedding are float32 in every mode, as in the code.
- **Inputs.** The eval's injections (`eval_injections(problem, seed)`, f0-sorted), with
  observation, base draws (`fold_in(key, 1)`) and noise keys built exactly as `eval.py` builds them.
- **Recording.** `MMDiT.encode` and `MMDiT.decode` are swapped for copies that also return every
  block's activations: the data tokens, keys and values, and the source stream entering each block
  and leaving the last. A check against the unpatched forward pass is stored as `patch_check`.
- **States.** For each t in {0, 0.5, 0.9, 0.99, 0.999, 0.9999}:
  - 16 base draws are moved along the straight path to the truth, x_t = (1 − t) u + t x₁.
  - x̂₁ is the head's mean m, from one network call. data/ predates this: it was measured with a

- **Widths.** The posterior width per source and coordinate is 1.4826 × MAD of the Fisher forecast
  draws (`fisher_samples`, 1024 draws, the same draws `eval.py` uses), mapped to flow coordinates.
  - The mcmc reference report checks the Fisher forecast against MCMC. Above SNR 40, Fisher ÷ MCMC
    width lies in 0.8–1.25 for about 90% of source parameters. Below SNR 40 it spans 0.35–2.63.
  - So low-SNR widths here are good to a factor of about 2.
  - Fisher draws in Mc can be negative at low SNR, which gives NaN in flow coordinates. MAD ignores
    them.
- **End to end.** `sample_posterior(problem, ...)` with the run's `ode_steps` (midpoint steps), from the same 64
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

From the repo root, as a single bare sbatch (GPU, 15 min on an A100 for 128 injections; the
sbatch takes bf16 from the run's config and sources `slurm/gpu.sh` for the float32 reference):

    sbatch --export=ALL,PYTHON=$PWD/.venv/bin/python --partition=gpu-highvram,gpu-preempt-highvram --account=pi_mehuber_umass_edu --qos=normal --constraint=a100 --gres=gpu:1 reports/architecture/precision-error/scripts/probe.sbatch 261101-royal-ferret-narrow-B

The job writes `data/<run>-<platform>.json` and rebuilds the page from every JSON in `data/`. To
rebuild only the page (stdlib only): `python3 reports/architecture/precision-error/scripts/build_page.py`.

## Data

`data/<run>-<platform>.json`, one per run and platform.
- **Run and platform:** run, config, commit, epoch, `run_bf16`, device, platform, `matmul_precision`.
- **Setup:** `modes`, `reference`, `t_grid`, `patch_check`.
- **Per injection:** `snr` [128, S]; `width` [128, S, 8].
- **Per block** (median and p90 per block): `encode[mode][y|k|v]` [L, 2]; `decode[mode]` [t, L+1, 2].
- **Output error:** `x1_over_width[mode]` [128, t, S, 8]; `velocity[mode]` [128, t].
- **End to end:** `e2e_displacement[mode]`, `e2e_width_ratio[mode]` [128, S, 8].
- **Token resolution:** `token_ratio` [128, S, 8]; `token_rounding` [128, S].

Present: `261101-royal-ferret-narrow-B-gpu` (epoch 1250, commit 8fc3ed9, A100, job 66528662, 15 min, modes none/attention/backbone) and the old records `misty-meadow-narrow-S-gpu`, `lucky-falcon-full-B-gpu` and `dusty-otter-narrow-XS-cpu` (smoke, 2000 steps, commit 11fc8a4, CPU, job 66014389, 7 min).
