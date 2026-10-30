# Sampling cost: what one LISA posterior costs, flow against MCMC (2026-10-24)

> **Timed on the 10-22 network** (measured 2026-10-24, src at 54f11d7 or earlier, before the catalogue
> prior, unordered sources, the x1 head and the per-source frame). The scripts follow HEAD and were
> not rerun. The MCMC seconds are older chains and unaffected.

## Takeaways

- **A posterior from the flow costs a fraction of a second.**
  - 256 draws of one window take 0.22 s (narrow-B) and 0.25 s (full-B), warm, on an A100 80 GB.
  - The cost is linear in draws (1024: 0.86 and 0.98 s). It goes from 0.085 s at XS to 0.92 s at
    XL: 9–10× the time for 51× the parameters.
  - The first call compiles: 27–37 s.
- **MCMC costs hours.**
  - The joint MCMC reference on 2 CPU cores takes 3.4–5.7 h per window and seed.
  - Its median (4.1 h) is 59,000–66,000× the flow at size B. The hardware differs: read this as
    the cost of a result, not a like-for-like speedup.

## What the page shows

[sampling-cost.html](sampling-cost.html). The top bar picks a band, a size and the draws (256 or
1024); a slider sets the number of windows to sample.

**Tiles**: the flow's time per window for the selection, its first (compiling) call, and the
joint MCMC median with its ratio to the flow.

1. **How does the flow's cost grow with the network?** Seconds per window against the parameters,
   per band and draws.
2. **Flow or MCMC: how long until a window's posterior?** The flow per config against the MCMC
   joint MCMC reference with a min–max whisker.
   Below it, the time for N windows on each side.
3. **What this means**: the takeaways, short.

The state lives in the URL hash, e.g. `#cfg=full-B&draws=256&win=100`.

## Method

[`bench_sampling.py`](scripts/bench_sampling.py), [`sampling.sbatch`](scripts/sampling.sbatch).
- The data were timed on the 10-22 network's `sample_posterior`: bf16 backbone, 32 ODE steps,
  x-prediction with self-conditioning (65 evaluations per draw), chunk 256 (the `prediction` and
  `self_condition` fields of `data/a100.jsonl`).
- The script now times HEAD's `sample_posterior(problem, flow, z, y, f, ode_steps)` on a fresh
  `setup(cfg)`, z standard normal of shape (draws, S, 8): 32 midpoint steps, 64 calls per draw, no
  self-conditioning, default chunk 256. It no longer records those two fields.
- The first call compiles (`first_s`); `best_s` is the best of 3 warm windows.
- Job 66002149, A100-SXM4 80 GB.

[`collect_mcmc.py`](scripts/collect_mcmc.py) reads each joint MCMC chain's `seconds` from the
`canna.lisa.mcmc` cache (`outputs/mcmc/joint-<problem id>/inj{j}_seed{s}.npz`).

All ran on 2 CPU cores.

## How to rebuild

The page needs only the standard library and runs on a login node:

    uv run --frozen python reports/benchmarks/sampling-cost/scripts/build_page.py

It reads `data/`, inlines everything, and prints every number the takeaways quote.

To remeasure: the `scripts/sampling.sbatch` header; `scripts/collect_mcmc.py` after new chains.

## Data

- `data/a100.jsonl`: per config × draws, `first_s`, `seconds` and `best_s`.
- `data/mcmc.json`: the joint MCMC chains' seconds under `joint`, one record (inj, seed, cache,
  seconds) per chain. The current file predates the joint-chain cache and still holds the older
  `narrowband` and `convergence` keys and `window*` joint chains.

The paper reads these in `paper/scripts/fig_sampling-cost.py`, `numbers_flow-matching.py`,
`numbers_mcmc.py` and `numbers_training.py`.
