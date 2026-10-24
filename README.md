# CANNA

**Inverse problems with conditional flow matching**: given an observation `y`, sample the
posterior `p(x | y)` over the parameters that produced it. A network learns a velocity field
that carries a base sample to a posterior sample along a geodesic (a straight line where the
chart is flat). JAX + equinox.

## Layout

Each problem is a **self-contained package** with its own `configs/`: `point/` and
`sinusoid/` (toy problems) and `lisa/` (galactic binaries). There is no shared problem base
or trainer; a new problem is a new package, copied and edited. Only `networks/` (the `eqx`
blocks and the `MLP`/`MMDiT` backbones) is shared.

- **The problem** — an `eqx.Module` with the priors, the simulator and the preprocessing
  that turns an observation into a network input.
- **The training draw** (`train.py`) — `train_sample(problem, key)` draws `(p, o)`, forms
  `y`, walks from a base point to the whitened parameters and returns the point and the
  velocity to match.
- **The chart** (`physical_to_flow` / `flow_to_physical`) — the only bridge between physical
  parameters and flow coordinates, so the physics keeps its own units.
- **The geometry** (`sinusoid/geometries.py` only) — `log_map` / `exp_map` for parameters on
  a manifold: angles on a circle, sky on a sphere, interchangeable sources as a `Set`. It is
  meant to follow from the priors (uniform on a sphere ⟹ lives on one). `point` and `lisa`
  are flat and have none.

Around the code: `slurm/` (job scripts), `reports/` (literature, research log, benchmarks,
diagnostics; one folder each, with its scripts and data) and `paper/`.

## Usage

```bash
uv run python -m canna.<problem>.train --config <name>   # point/sinusoid: writes outputs/<problem>-<name>/
uv run python -m canna.<problem>.eval  --config <name>   # corner plots from that checkpoint
LISA_CONFIG=<band>-<size> uv run python -m canna.lisa.train   # a new lisa run
LISA_RUN=<run> uv run python -m canna.lisa.{train,eval,scorecard}   # resume or evaluate it
LISA_RUN=<run> uv run python -m canna.lisa.mcmc {run,status,report}
uv run python -m canna.lisa.plots <run dir, eval dir or .npz>   # figures from saved data
```

`point`/`sinusoid` configs are `XS` or `B`, and every CLI flag overrides them. `lisa` reads
`configs/band/<band>.yaml` and `configs/size/<size>.yaml` (default `full-B`), then any
`LISA_<FIELD>=value` env var (`LISA_TOTAL_STEPS=1000000`, `LISA_BF16=attention`,
`LISA_OUTPUT_DIR=…`) over a field of `canna.lisa.config.Config`.

A `lisa` run is named `<adjective>-<noun>-<config>` and addressed by `LISA_RUN`. Its
`config.yaml` is the source of truth: the resolved config and the commit it trained at.
Training refuses uncommitted changes to `src/`; later jobs may change only the eval fields
(`ode_steps`, `n_random`, `n_validation`, `weights`, `bf16`) and run on the run's commit
(`source slurm/run-src.sh` puts a worktree of it, under `outputs/src/`, on `PYTHONPATH`).
Jobs write only data; `canna.lisa.plots` draws the figures.

```
outputs/runs/<run>/
  config.yaml
  history.npz                                # losses, held-out losses, λ and error per time bin
  checkpoints/<epoch>/                       # plots: losses.pdf, lambda.pdf
  eval/<epoch>-<weights>-<ode_steps>/
    eval.yaml                                # the settings it was evaluated with
    draws.npz                                # eval
    scorecard.npz, validation.npz, validation_scores.npz   # scorecard
    mcmc_compare.txt, mcmc_compare.npz, mcmc.npz            # mcmc report
                                             # plots: corner.html, validation.pdf
```

To keep runs small, the draws in `draws.npz`, `validation.npz` and `mcmc.npz` are
stored as float16 offsets from the truth in units of their robust width (`canna.lisa.samples`,
~1e-3 of a width); read them with `samples.load`, which also reads plain files.

## The LISA problem (`canna.lisa`)

Galactic binaries in simulated LISA data: each frequency window holds four sources, and the
flow returns their joint posterior over frequency, chirp mass, amplitude, sky, inclination
and the two circular-mode phases φ₀ ± 2ψ. Why the code is the way it is lives in
[`reports/research-log/`](reports/research-log/000-takeaways.md).

| module | holds |
|---|---|
| `config.py` | `Config`, `load()` (defaults, yamls, `LISA_*`), `load_run()`, run names |
| `physics.py` | constants, the chirp-mass and galactic sky priors, `fdot` |
| `problem.py` | `LisaGB` (pure configuration), the frequency window, the noise PSD |
| `simulator.py` | sources, signal, noise, SNR, likelihood, whitening |
| `chart.py` | the 8 flat flow coordinates per source |
| `network.py` | `LisaFlow` and its STFT tokens |
| `optim.py` | schedule-free Muon/AdamW and the averaged weights |
| `train.py` | training draw, loss, step, orbax checkpoints, `main` |
| `sample.py` | the posterior sampler: midpoint steps for x-prediction, RK4 for velocity |
| `injections.py` | eval injections, Fisher draws, source matching |
| `plots.py` | every figure |
| `eval.py`, `scorecard.py`, `mcmc.py` | post-training entry points (`validate.py` is the scorecard's validation) |

**Bands and sizes.** A config is a band (how much spectrum a window covers, and the
training length) and a network size:

| band | window | f₀ reach | tokens (16 bins each) | steps |
|---|---|---|---|---|
| `narrow` | 128 bins | ≤ 1.3 mHz | 8 | 650k |
| `full` | 4096 bins | 0.1–12 mHz | 256 | 240k |

| size | width × blocks | heads | parameters |
|---|---|---|---|
| `XS` | 256 × 4 | 4 | 16M |
| `S` | 384 × 6 | 6 | 47M |
| `B` | 512 × 8 | 8 | 106M |
| `L` | 768 × 12 | 12 | 346M |
| `XL` | 1024 × 16 | 16 | 810M |

Every run so far used `B`. The research log calls the bands `XS` and `B`: its `XS-…` runs are
`narrow-B`, its `B-…` runs `full-B`.

**Recipe** (research log F32, F39–F44, F56).

- **Data tokens.** The whitened rfft window (A, E, T, real and imaginary), 16 bins per token,
  each token as its band's 16 time samples (a critically sampled STFT), fed as asinh|z| and
  the gated phase z / √(1 + |z|²) (F50, F51).
- **Sources.** Sorted loudest first, each slot with a learned embedding. 8 flat coordinates
  per source, `[f0, Mc, l, b, Re z+, Im z+, Re z−, Im z−]`: log f₀ over the window,
  standardised log M_c, the sky in galactic (l, b) divided by the galactic-mixture prior's
  stds, and the helicity amplitudes z± = A (1 ∓ sin ι)²/4 · e^{i(φ₀ ± 2ψ)} in units of SNR ≈ 1,
  radius warped by asinh. Phases are referenced to the middle of the observation. The
  amplitude prior is log-uniform in each source's SNR over `snr_range`.
- **Embeddings.** Values go through piecewise-linear encodings in float32; time, window
  frequency and positions go in as sinusoids and an MLP. Each source also carries
  |z| cos kφ, |z| sin kφ for k = 2–6 and Fourier features of its f₀ bin.
- **Attention.** One-way: the data stream attends only to itself and is conditioned on the
  window frequency alone, so its keys and values are cached per observation (shared over
  `draws: 16` in training, over every ODE step in sampling). The source stream is conditioned
  on t, log₁₀(1 − t) and the window, adds 4 attention-sink tokens, and has a learned per-head
  q gain. RoPE in frequency places each source at its f₀ and each data token at its centre
  bin (F44).
- **Objective.** Predicts the endpoint x₁, trained as the velocity (x₁ − x_t)/(1 − s) with
  1 − s floored at 1e-6, and a learned loss weight per (log₁₀(1 − s) bin, coordinate group).
  Self-conditioned on its stopped x₁ half the steps. t is stratified over the batch; half the
  draws have 1 − s log-uniform down to 1e-4.
- **Optimiser.** Schedule-free (decay 0.05 on z, averaging weights t·lr² after twice the
  warmup, β 0.9 → 0.965) around Muon (lr 1e-2, NorMuon rows, momentum 0.8) for the
  backbone, conditioning MLPs, output modulation and head's hidden layer, and AdamW (lr 1e-3,
  b2 0.95, no decay) for the rest. Checkpoints keep the optimiser state, since the averaged
  weights x that eval uses live there (`LISA_WEIGHTS=training` for y).
- **Precision.** The backbone runs in bf16 by default (`bf16: backbone`, embeddings and head
  in float32); `attention` keeps it float32 with bf16 attention, `none` is float32
  throughout. A float32 backbone on a GPU needs
  `XLA_FLAGS=--xla_gpu_autotune_level=0 JAX_DEFAULT_MATMUL_PRECISION=highest` (XLA's autotuner
  picks wrong kernels); the slurm scripts set this. Evaluate in the precision the run trained in.

Dropped variants (patch tokens, `psi_phi0` orientation, residual-template and Fisher-loss
inputs, AdamW decay) are at commit 641d9ed and before; the WDM-input runs at tag `wdm-r1`.

**Where it stands.** At mid SNR the B models are within 1–2× of the MCMC in every parameter
but φ₀. The loudest sources are still far off: φ₋ ~100× and sky ~20× the MCMC width (F56,
F58); see the takeaways for the rest.

**Evaluation.** `eval` draws the flow on ten eval injections, beside Fisher forecasts;
`plots` turns them into one page, `corner.html`: every injection's corners per source (its
draws matched onto the injected sources) or pooled, against the Fisher draws and the MCMC,
in physical coordinates (sky in ecliptic λ, β) or the flow's own, with the flow/reference
widths. `scorecard` scores every source of those ten plus `n_random` prior
injections (f₀ width against the ideal, detection, offset, 68%/95% coverage), then runs the
SBC/TARP validation over `n_validation` draws. To evaluate a run kept elsewhere, point
`LISA_OUTPUT_DIR` at the folder holding its `runs/<run>`.

**Reference MCMC.** `canna.lisa.mcmc` samples each eval source alone, the others held at
their true values, on its own band of the data. The chains depend only on the problem, so
they are cached per problem in `outputs/mcmc/eval-<id>/` and shared by every run. `mcmc report
[--eval <dir>]` writes the flow-over-MCMC widths (`mcmc_compare.txt`) and the chains
(`mcmc.npz`), which the corner pages then draw. Needs jexplore (Python 3.13, pinned).

## On a Slurm cluster

Submit from the repo root after sourcing the cluster's flags (`slurm/unity.env` or
`slurm/trex.env`, which set `$CANNA_SBATCH_GPU` and `$CANNA_SBATCH_CPU`). Each script does
one thing; chain them by hand (`--dependency`). Trailing `LISA_<FIELD>=value` pairs are
exported to the module.

```bash
source slurm/unity.env
sbatch $CANNA_SBATCH_GPU slurm/lisa.sbatch <config>                    # train; requeues itself near the time limit
sbatch $CANNA_SBATCH_GPU slurm/lisa.sbatch <config> LISA_RUN=<run>     # resume
sbatch $CANNA_SBATCH_GPU slurm/lisa-post.sbatch <run> scorecard        # or eval, or mcmc report
sbatch $CANNA_SBATCH_CPU --array=0-<n-1> slurm/lisa-mcmc.sbatch <config>  # reference MCMC; n from `mcmc status --count`
```

The scripts carry only resources (L and XL need `--mem=64G`); partition, account and QOS
come from the `.env`, so another cluster needs only its own. Unity holds a job after 5
requeues: 6 × 24 h, enough for XL's 4–5 days.
