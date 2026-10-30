# Attention maps: where the source slots attend, per block and head (2026-10-24)

## Takeaways

- **No trained run has been measured yet.** The only record is the 2000-step smoke run
  `dusty-otter-narrow-XS` (0.3% of narrow's 650k steps). It checks the pipeline and the weight
  recomputation end to end, and is labelled as a smoke run on the page. Its maps say nothing
  about the architecture.
- **The recomputed weights match the network's to bf16 resolution.** On the smoke run (bf16
  backbone) the x stream implied by the recomputed weights matches every block's real one to
  0.5–1.1% of its largest entry, about one bf16 ulp there. The tapped flow (a Python loop over
  the blocks) and the plain flow (a scan) already differ by 0.7% in output, so the gap is the
  network's own bf16 rounding, not a wrong softmax. A float32 run would tighten this check.
- The report will answer, once narrow-B and full-B have trained:
  - **Does each slot attend to its own source's tokens?** The own-band share of each block's data
    attention against t, against the uniform baseline, and by SNR.
  - **How much mass goes to the sinks?** Per block, against t.
  - **How focused is each head?** Normalised entropy per block × head.
- To fill it, run the probe on each new run (see [How to rebuild](#how-to-rebuild)). The page
  then defaults to the longest-trained non-smoke run.

## What the page shows

[attention-maps.html](attention-maps.html). The top bar picks a run (smoke runs are marked), a
path time t, and a head or the mean over heads. A banner flags any run under 50k steps as a smoke
run.

1. **Where does a slot's attention go, block by block?** Stacked mass per block: self, other
   slots, sinks and data, at the selected t.
2. **Does each slot attend to its own source's tokens?** The share of the data attention on the
   tokens holding 95% of the slot's own source's power (or on the other sources' tokens) against
   t, one line per block. The uniform baseline is dashed.
3. **How much attention goes to the sinks, and when?** Sink mass against t per block.
4. **How focused is each head?** A block × head heatmap of normalised entropy, over the data
   tokens only or over all keys.
5. **Does a louder source get more of its slot's attention?** Own-band share by source SNR per
   block.
6. **What does one observation's attention look like?** Slot × data-token weights for one block,
   with each source's tokens and f0 token marked.
7. **What this means**: how to read each card. It lists which runs are smoke and which are
   trained.

The URL hash `#run=<tag>` selects a run. The velocity's actual sensitivity to the data is the
[data-dependence](../data-dependence/) report. Its error against the truth is
[reference comparison](../../posterior-eval/reference-comparison/).

## Method

[`scripts/attention_maps.py`](scripts/attention_maps.py) loads a run's checkpoint like
[the velocity probe](../data-dependence/scripts/velocity_probe.py).
It runs on the run's commit (`slurm/run-src.sh`). `src/` is not modified.

- **Tapping.** The weights are tapped in memory only:
  - The flow's `MMDiT` backbone is swapped (`eqx.tree_at`) for a `TappedMMDiT`. It holds the same
    blocks, and its decode is a Python loop over them instead of a scan.
  - Before each block's real `decode`, the probe recomputes that block's softmax with the block's
    own modules. Those are its modulation, `heads` (q/k/v projections, rms norm, qk gain, RoPE on
    the slots) and the sink stream. The logits are computed in float32, as in `MMDiTBlock.decode`.
  - From those weights it rebuilds the block's x update: the GTA rotation back, `out_proj_x` with
    its gate, and the MLP. It records the largest gap to what `decode` returned. It also records
    the gap between the tapped flow's output and the plain flow's. Both gaps are relative to the
  largest |entry|.
- **Observations.** 64 fresh training observations from `train_sample` (keys 400–403 × 16), at
  t ∈ {0.1, 0.5, 0.9, 0.99, 0.999, 0.9999}, one network call each (the data in `data/` was tapped
  on the sampler's second, self-conditioned pass of the earlier network).
- **Keys.** They are ordered [slots, sinks, data tokens]. Only the slot query rows are kept.
  - **Mass**: self, the other slots, sinks and data.
  - **Data share**: the data attention renormalised over the tokens. It is split into the slot's
    own band, the other sources' bands and the rest. A source's band is the tokens touching the
    fewest bins that hold 95% of its whitened power, as in data-dependence.
  - **Baseline**: each set's token fraction.
  - **Entropy**: over all keys / log(#keys), and over the data share / log(#tokens).
- Slots match sources by index: x_t comes from `train_sample`, whose paths end on the target's
  sources loudest first.

## How to rebuild

From the repo root on a login node, `source slurm/unity.env` first. On a new run (narrow-B or
full-B; GPU for B and up):

    sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/architecture/attention-maps/scripts/attention_maps.sbatch <run>

An XS or S run fits the CPU partition:

    sbatch $CANNA_SBATCH_CPU reports/architecture/attention-maps/scripts/attention_maps.sbatch <run>

The optional arguments are `[tag] [training]`, as for data-dependence. The job writes
`data/<tag>.json`, then rebuilds the page. To rebuild the page alone (stdlib only):

    uv run --frozen --no-sync python reports/architecture/attention-maps/scripts/build_page.py

## Data

`data/<tag>.json`, one per probed run. It holds:
- the run's identity: `run`, `config`, `commit`, `epoch`, `steps`, `total_steps`, `weights`,
  `bf16`, `device` and `seconds`;
- the shapes: `blocks`, `heads`, `slots`, `sinks`, `tokens` and `patch`;
- the grids: `t_grid`, `snr_edges`, `band_power` and `rings`;
- the checks, relative to the largest |entry|: `check_block` [block] and `check_output`;
- `baseline` [slot, own/other/rest];
- `per_t[t]`, with:
  - `mass` [block, head, slot, self/others/sinks/data];
  - `split` [block, head, slot, 3];
  - `split_snr[i]` [block, head];
  - `entropy` and `data_entropy` [block, head, slot];
  - `enrichment` [block, head, ring], data attention density by distance from the f0 token,
    over uniform;
  - `example` [block, slot, token], observation 0, head mean, in units of `example_unit`;
- `example`: observation 0's `snr`, `f0_token` and `own` mask.

| record | run | steps | job |
|---|---|---|---|
| `dusty-otter-narrow-XS.json` | smoke, commit 11fc8a4 | 2000 | 66014428 (66014232 had absolute checks) |
