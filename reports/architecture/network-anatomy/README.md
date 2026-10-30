# Network anatomy: what LisaFlow is made of and what a forward costs (2026-10-30)

## Takeaways

Measured with jobs 66324931 (narrow-XS) and 66324948 (the other nine) on the `canna.lisa` network of src commit 8310cb3: registers, per-source frame head, absolute RoPE, no `hat_embed`, no piecewise-linear `y_embed`. The data records name fa52925, an earlier id of the same network code.

- **The encode outweighs a decode more on the full band.**
  - One full-B `encode` (the data stream, once per observation) is 1.494 GFLOP against 0.386 GFLOP for one `decode`: 3.9 decodes (3.7 at XS, 4.0 at XL). Narrow-B is 0.534 against 0.383 GFLOP, 1.4 decodes.
  - In the training forward the encode is 19.5% of full-B (7.66 GFLOP per observation) and 8.0% of narrow-B (6.65 GFLOP).
- **Sampling cost per draw is small and the cache is cheap.**
  - A full-B draw costs 21.4 GFLOP over 64 network calls (32 midpoint steps × 2), against 4.8 GFLOP once per observation (1.49 encode + 3.34 time-conditioning shared by the draws).
  - The cache holds data and register keys and values: 0.56 MiB per observation at full-B and 0.19 MiB at narrow-B.
- **Parameters split four ways, and modulation is 37.5% of every block.**
  - At B the data, register, source and sink streams hold 23%, 26%, 25% and 23% of 132.8M parameters (the register share includes the frame head), so the 4 sink tokens carry as many weights as the 32 data tokens.
  - Modulation is 37.5% of every block's weights but 1.9% of a full-B data block's FLOPs. The SwiGLU MLP is 58% of the FLOPs and the attention core 2%.
- **The frame is nearly free in compute.** `frame_modulation` and `frame_unembed` are 0.11% of the full-B encode (0.31% narrow-B); the 4 registers add 4 tokens to 32.
- **`y_embed` is no longer a cost.** A linear map of each token's 144 STFT numbers, it is 0.3% of the full-B encode and 1.3% at XS.
- **Counts and measurement agree with the other reports.** Params are 16.6M, 55.9M, 132.8M, 449.3M, 1066.7M for XS to XL (the full band is 6-25k larger than the narrow); Muon takes 99.6% of XS and 100.0% of XL. XLA's count of a lone block is within 0.9% of the matmul walk in every config, whose whole-forward totals are the figures quoted. The last block carries 0.79% of B's parameters (1.05M: the data and sink queries and attention gates) that nothing reads.

## What the page shows

[network-anatomy.html](network-anatomy.html). The bar picks a band and a size (URL hash
`#cfg=full-B`); clicking a row of a chart or table selects it too.
Cards, each one question:

1. **How is the network wired?** An SVG diagram with the shapes of the selected config, registers and frame head included.
2. **What shape is each tensor?** The stages of one example's forward.
3. **Where do the parameters live?** Share by group across all ten configs.
4. **What is one block made of?** Params and FLOPs per stream and part, middle and last block,
   plus the embeddings and head.
5. **What does the module tree hold?** Every leaf with its shape and Muon or AdamW tag.
6. **How does the compute of a forward scale?** Encode and decode FLOPs for all ten configs.
7. **What does sampling and training cost?** Sampler per draw and per step, training forward and
   gradient.
8. **What this means.**

## Method

[`anatomy.py`](scripts/anatomy.py) builds each `LisaFlow` abstractly with `config.load`, so no
weights are allocated (CPU only).
- **Shapes and parameters:** `jax.eval_shape` and the eqx tree leaves, grouped by stream and kind
  from the leaf paths; Muon membership from `optim.is_muon`.
- **Part FLOPs:** XLA `cost_analysis` of each part (projections, modulation, MLP, per stream)
  lowered alone, in float32. The attention core is the block's total minus those parts.
- **Whole-forward FLOPs:** a jaxpr walk summing 2·m·n·k over every `dot_general`, with scan bodies
  times their length and `cond` branches by min or max. Used for encode, decode, `sample_posterior`
  (1 and 2 draws, so the per-draw and per-observation parts separate), and the training objective
  and its gradient. XLA's own count of the whole jitted function counts a scan body once, so it is
  recorded only as a cross-check.
- Counts are matmul FLOPs (2 per multiply-add); elementwise work is excluded from the walk.

## How to rebuild

The page needs only the standard library and runs on a login node:

    uv run --frozen python reports/architecture/network-anatomy/scripts/build_page.py

To remeasure (CPU, 4 cores, 16G, about 2 min for all ten configs; skips configs already in the
data file):

    sbatch $CANNA_SBATCH_CPU reports/architecture/network-anatomy/scripts/anatomy.sbatch [config ...]

## Data

- `data/anatomy.jsonl`: one line per config. Keys: the config's dimensions, `stages` (name, shape,
  dtype), `leaves` (path, shape, muon), `flops` (`embed`, `blocks` middle/last, `xla_whole`,
  `matmul` with encode, decode, `sampler_1`, `sampler_2`, training forward and gradient),
  `cache_bytes`, `commit`.
- Jobs 66324931 (narrow-XS) and 66324948 (the other nine configs), CPU, 4 cores, 16G, about 20 s and 100 s; src commit 8310cb3 (records: fa52925, an earlier id of the same network code).
