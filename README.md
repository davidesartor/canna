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
package's `configs/` (`B`, `XS`). Every CLI flag overrides the config.

## The LISA problem (`canna.lisa`)

Galactic binaries in simulated LISA data. Each window of the spectrum holds four
sources, and the flow returns their joint posterior over frequency, chirp mass,
amplitude, sky position, polarisation, inclination and phase. Why the code is the
way it is lives in [`docs/lisa-research-log.md`](docs/lisa-research-log.md): every
run, what it found, and the decision rules written before its result.

**Sizes.** The configs come in rungs that differ only in how much of the band a
window covers, and so in how many conditioning tokens the network sees:

| rung | window | f₀ reach | tokens |
|---|---|---|---|
| `XS` | 64 bins | ≤ 1.3 mHz | 32 |
| `S` | 256 bins | ≤ 4.2 mHz | 128 |
| `B` | 4096 bins | the whole 0.1–12 mHz | 2048 |

**Training recipe.** The `*-late` configs carry what the XS runs settled on:

- `warmup_frac: 0.1`: the auxiliary heads train for the first 10% of the steps
  only, then pure flow matching.
- `time_power: 3`: a warped flow clock, `s = 1 − (1 − t)³`, that spends more of
  the training near the end of the path, where a loud source's posterior is
  resolved.
- `cooldown_steps`: the last steps decay the learning rate linearly to 0. Over
  the last 20% of a run this halved the frequency error floor and removed a
  run-specific bias, the largest gain so far.
- `save_opt_state: false` leaves the optimiser state out of checkpoints, for a
  tight disk quota. `require_checkpoint: true` makes a continuation config
  (`*-cool`) stop rather than train from scratch when its checkpoint is missing.

| config | what it is |
|---|---|
| `XS`, `S`, `B` | the original runs: a constant lr and a long auxiliary phase |
| `XS-late` | the recipe without the cooldown, 512 × 8, 1M steps |
| `XS-late-768` | the same with a 768 × 8 network |
| `XS-late-768-cool`, `XS-late-cool` | 200k-step cooldowns continuing those two from 1M steps |
| `S-late`, `S-late-768` | S with the full recipe, 1M steps of which the last 200k cool down |

**Evaluating a run.**

```bash
uv run python -m canna.lisa.scorecard --config <name>   # outputs/lisa-<name>/scorecard.npz
uv run python -m canna.lisa.eval      --config <name>   # outputs/lisa-<name>/corner/*.pdf
```

The scorecard scores every source of 210 injections: its frequency width
against the ideal one, whether it was found, the offset from the truth, and
whether the 68% and 95% intervals cover it. The eval draws corner plots of ten
injections against a Fisher forecast. Both stop if the run has no checkpoint.
`--ode_steps`, `--n_random` and `--dtype float32` adjust them. Evaluate in the
precision the model trained in (bf16): fp32 gives the same widths.

**float32 on a GPU needs autotuning off.** XLA's matmul autotuner picks wrong
float32 kernels for this network, on the laptop GPU and on A100s alike. bf16 is
unaffected. The slurm scripts handle this; by hand, run fp32 on a GPU with
`XLA_FLAGS=--xla_gpu_autotune_level=0 JAX_DEFAULT_MATMUL_PRECISION=highest`.

**On a Slurm cluster.** Submit from the repo root. Flags after the config go to
the Python module:

```bash
sbatch slurm/lisa.sbatch <config>                 # train
sbatch slurm/lisa-scorecard.sbatch <config>       # scorecard
sbatch slurm/lisa-eval.sbatch <config>            # corner plots
sbatch slurm/lisa-scorecard-cpu.sbatch <config> --dtype float32 --n_random 40   # CPU cross-check
```

A run resumes from its latest checkpoint, so a training job longer than the
partition's time limit is a chain of submissions:
`sbatch --dependency=afterany:<jobid> slurm/lisa.sbatch <config>`. The scripts
carry TREX's partition and account names; adjust them for another cluster.
