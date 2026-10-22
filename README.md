# CANNA

A framework for solving **inverse problems with conditional flow matching**:
given an observation `y`, sample the posterior `p(x | y)` over the parameters
that produced it.

A network learns a velocity field that transports a base sample to a posterior
sample along a geodesic.

## Layout

Each problem is a **self-contained package** — `point/`, `sinusoid/`, `lisa/` —
owning its `problem.py`, `network.py`, `train.py`, `eval.py` and `configs/`.
There is no abstract problem base and no shared trainer: a new inverse problem
is a new package, copied and edited. Only `networks/` (the reusable `eqx` blocks
and the `MLP`/`MMDiT` backbones) is shared.

- **The problem** (`problem.py`) — an `eqx.Module` holding the priors to draw
  parameters from, the simulator that turns them into an observation, and the
  preprocessing that makes it a network input.
- **The training draw** (`train.py`) — `train_sample(problem, key)` draws
  `(p, o)`, forms the conditioning `y`, walks a geodesic from a base point to
  the whitened parameters, and returns the point and the velocity to match.
  `point/` keeps it inside its trainer script, so only `sinusoid`/`lisa`
  export it.
- **The geometry** (`geometries.py`, per package) — where the parameters live.
  Supplies `log_map` / `exp_map`, so the flow steps along the manifold rather
  than through the coordinates. Angles ride a circle (`Spherical(2)`), sky
  positions ride a sphere, and interchangeable sources form a `Set` whose
  posterior is permutation-invariant — no branch cuts, no poles, no label
  ambiguity. Flat parameters need no manifold at all, so `point/` has no
  geometry module.
- **The chart** (`physical_to_flow` / `flow_to_physical`) — how parameters are
  coordinatized for the flow, and back. It is the only bridge between the two,
  so the physics stays in the units it is written in and the flow only ever
  sees the manifold.

Stating the priors is most of the work of stating the geometry: a prior uniform
on a sphere is a parameter that lives on one, an angle drawn on a circle wraps,
and a block of interchangeable sources is a set. The geometry a problem needs is
largely **inferrable from the priors it puts on its parameters**, rather than
declared twice and kept in sync by hand.

## Usage

```bash
uv run python -m canna.<problem>.train --config <name>   # writes outputs/<problem>-<name>/
uv run python -m canna.<problem>.eval  --config <name>   # corner plots from that checkpoint
```

`<problem>` is `point`, `sinusoid` or `lisa`; `<name>` is a run `.yaml` in that
package's `configs/` (e.g. `XS-fd`, `B-fd` for `lisa`). Every CLI flag overrides the config.

## The LISA problem (`canna.lisa`)

Galactic binaries in simulated LISA data. Each window of the spectrum holds four
sources, and the flow returns their joint posterior over frequency, chirp mass,
amplitude, sky position, polarisation, inclination and phase. Why the code is the
way it is lives in [`docs/lisa-research-log.md`](docs/lisa-research-log.md): every
run, what it found, and the decision rules written before its result.

**Sizes.** The configs come in rungs that differ only in how much of the band a
window covers, and so in how many conditioning tokens the network sees:

| rung | window | f₀ reach | tokens (16 bins each) |
|---|---|---|---|
| `XS` | 64 bins | ≤ 1.3 mHz | 4 |
| `B` | 4096 bins | the whole 0.1–12 mHz | 256 |

**Training recipe** (research log F32, F39–F44). The network sees the whitened
rfft window, the real and imaginary parts of A, E and T as six channels, 16 bins per token.

- **Data tokens.** Either the 16 rfft bins as they are (re/im of A, E, T), or, with
  `stft: true`, the band's 16 time samples, 45.7 days apart (an orthonormal inverse DFT per
  token: a critically sampled STFT, same bands, same positions). Each sample is then fed as
  asinh|z| and the gated phase z / √(1 + |z|²). On B this narrows loud sources' sky, A and ι
  to 0.6–0.75× per source at no cost below SNR 100. On XS it costs ~20% in f₀ at mid SNR
  (F51).
- **Embeddings.** Every input scalar (each data value, each source coordinate, the window
  frequency) goes through a piecewise linear encoding and one linear map, in float32. Only
  the MMDiT backbone runs in bf16.
- **Sources.** Sorted loudest first by SNR, so each slot is a fixed label (no permutation
  matching), with a learned slot embedding. A reconstruction head predicts each slot's
  parameters (weight 0.1 beside the flow loss).
- **RoPE in frequency** (`rope: true`). Every attention rotates q and k by the position in
  bins: each source at its current f₀, each data token at its centre. The score of a source
  on a token then depends on f₀ − f_token. On B this took f₀ from a flat ~3-bin floor to
  0.035 bins (F44).
- **Optimiser.**
  - Muon (lr 1e-3, NorMuon rows, momentum 0.8) takes every matrix of the backbone.
  - AdamW (lr 3e-4, b2 0.95) takes the encodings, the token positions, the last head layer
    and anything 1-D.
  - Both run inside our own schedule-free wrapper: decay 0.05 on z, averaging weights t·lr²,
    averaging after twice the warmup, β 0.9 → 0.965.
  - `decay_adamw: false` keeps the decay off the AdamW parameters (the STFT runs).
- **Checkpoints and eval.** Checkpoints keep the optimiser state, because the evaluated
  weights x are schedule-free's average, which lives in it. Eval and the scorecard load x
  (`--weights training` for y).
- **Clock.** `time_power: 3`: a warped flow clock, `s = 1 − (1 − t)³`, that spends more of
  the training near the end of the path, where a loud source's posterior is resolved.

| config | what it is |
|---|---|
| `XS-fd`, `XS-fd-rope` | XS, 512 × 8, 1.2M steps, without and with RoPE |
| `B-fd`, `B-fd-rope` | B, 200k steps, without and with RoPE |
| `B-fd-rope-long` | B-rope at 600k steps. **The current B model** (F48) |
| `B-fd-template` | B-rope conditioned also on the data minus the sources at the flow's current point (F46; worse, F47) |
| `B-fd-fisher` | B-rope with the Fisher-weighted (Mahalanobis) flow loss (F46; no change, F47) |
| `XS-stft`, `B-stft` | XS-rope and B-rope with STFT data tokens and no AdamW decay (F50). **B-stft** equals B-long on loud sources at 200k steps (F51) |
| `XS-stft-ht`, `B-stft-ht` | the STFT runs inferring φ₀ ± 2ψ (`polarisation: helicity`) and given the template residual (`templates: true`, a second group of STFT features). The first models that learn φ₀ and the ψ–φ₀ correlation, at 1.3–1.45× B-stft's f₀ width (F54) |
| `XS-stft-h`, `B-stft-h` | the same without the template. **The current recipe:** φ₀ ± 2ψ within 1.8–2.3× of the MCMC on B at SNR 17–78, f₀ 1.03–1.25× B-stft's (F56) |
| `B-stft-hd` | B-stft-h plus each source's band of the data Doppler-demodulated at its current sky (`doppler: true`; Cornish & Larson 2002). Halves the typical sky offset, but not the large per-source ones (F58) |
| `B-fd-helicity` | B-long + Fisher loss + no AdamW decay + the orientation as sin ι and the circular-mode phases φ₀ ± 2ψ (`polarisation: helicity`; F49, not run) |

**Known limits (F45, F48, F51).**
- **φ₀ needs the helicity coordinates** (`polarisation: helicity`, flow coordinates φ₀ ± 2ψ).
  - **In the (ψ, φ₀) chart** (every run from F39 to F51) the flow's φ₀ ignores the data and
    keeps its prior draw (F52). The likelihood's symmetries make four copies of every
    posterior there, and the network's targets average to zero over them.
  - **With the helicity coordinates** φ₀ ± 2ψ is read from the data, with or without the
    template: within 1.8–2.3× of the MCMC on B at SNR 17–78 (F56).
  - **The loudest sources' phases stay wide:** 101× on B, 51× on XS.
  - **Loud sources' A, ι and sky are 1.2× wider** than without the new coordinates.
- **The loudest sources' sky** is ~20× wider than the MCMC on the B-stft runs (30–44× on B-long), and ~10× on XS-stft-h.
- **At mid SNR** the B models are within 1–2× of the MCMC in every parameter but φ₀.
- **Individually biased sky posteriors:** single sources' λ, β sit up to 2–3 MCMC widths (1–2 flow widths) off at mid SNR, and the loudest by its own width. The Doppler-demodulated band (`B-stft-hd`) trims the typical offset, not these (F58).

The WDM-input code and configs of the earlier runs (`XS-late*`, `XS-pos`, `B-late`,
`B-pos`, …) live at the tag `wdm-r1`. To rescore one of those checkpoints, check that tag out
in a worktree.

**Evaluating a run.**

```bash
uv run python -m canna.lisa.scorecard --config <name>   # outputs/lisa-<name>/scorecard.npz
uv run python -m canna.lisa.eval      --config <name>   # outputs/lisa-<name>/corner/*.pdf
```

A finished run is archived locally as `outputs/NN-lisa-<name>`, numbered in the order the runs
were made (research log §8 maps the numbers to runs). To re-evaluate one, point
`--output_dir` at a folder holding a `lisa-<name>` symlink to it.

**The reference MCMC.** `canna.lisa.mcmc` samples the corner pages' sources (gb-loudest,
gb-median, gb-faintest and the four of q0.10) one at a time, with the other sources held at
their true values, reading only each source's own band of the data. The chains depend only
on the problem, so they are cached per problem in `outputs/mcmc/eval-<id>/` and reused by
every run. The eval draws them on its corner pages in green, and
`python -m canna.lisa.mcmc report --config <name> --run <folder>` writes `mcmc_compare.txt`,
the flow's width over the MCMC's per source and parameter. They need jexplore, in a
Python 3.13 venv (`.venv313`; research log §7).

The scorecard scores every source of 210 injections: its frequency width
against the ideal one, whether it was found, the offset from the truth, and
whether the 68% and 95% intervals cover it. The eval draws corner plots, flow
against a Fisher forecast, into `corner/`:

- `q<quantile>.pdf`: one page per injection, at ten window-SNR quantiles, with every
  source's parameters side by side.
- `q<quantile>-pooled.pdf`: the same injection with its sources pooled. The
  `[draw, source, param]` samples become `[draw × source, param]`, so each marginal
  holds every source at once; f₀, for instance, shows one spike per binary.
- `gb-loudest.pdf`, `gb-median.pdf`, `gb-faintest.pdf`: a single binary, the loudest,
  median and faintest by SNR of all the sources in those ten injections. Its draws
  are first relabelled onto it, since the flow's source slots are interchangeable.

Both stop if the run has no checkpoint.
`--ode_steps`, `--n_random` and `--dtype float32` adjust them. Evaluate in the
precision the model trained in (bf16): fp32 gives the same widths.

**float32 on a GPU needs autotuning off.** XLA's matmul autotuner picks wrong
float32 kernels for this network, on the laptop GPU and on A100s alike. bf16 is
unaffected. The slurm scripts handle this; by hand, run fp32 on a GPU with
`XLA_FLAGS=--xla_gpu_autotune_level=0 JAX_DEFAULT_MATMUL_PRECISION=highest`.

**On a Slurm cluster.** Submit from the repo root. Flags after the config go to
the Python module:

```bash
slurm/lisa-chain.sh <config> <links> --n_random 50   # train in resuming 24 h links, then everything after
sbatch slurm/lisa-local.sbatch <config>              # a run that fits one slot, trained in the node's /tmp
sbatch slurm/lisa-post.sbatch <config>               # scorecards x and y, eval, eval draws, MCMC tables
sbatch slurm/lisa-scorecard-cpu.sbatch <config> --dtype float32 --n_random 40   # CPU cross-check
```

Both training scripts first call `slurm/mcmc-ensure.sh`, which submits the problem's
reference MCMC to CPU nodes (`slurm/lisa-mcmc.sbatch`) if it is not cached yet. A run
resumes from its latest checkpoint, so `lisa-chain.sh` links training jobs with
`afterany` and starts the post job with `afterok` on the last link. The scripts carry
TREX's partition and account names; adjust them for another cluster.
