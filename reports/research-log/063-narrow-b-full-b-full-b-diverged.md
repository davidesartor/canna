# F63 (2026-10-31): narrow-B and full-B on a8e3477; full-B diverged at epoch 93, narrow-B still training

**Outcome.** full-B (`261030-stormy-cedar-full-B`) stopped at epoch 93 of 240 on a non-finite
loss. It had learned well until about epoch 45, then ran away for 50 epochs: the training frame
loss rose from −0.14 to 3.5, and the gradient norm went from ~20 to 540. Only the latest checkpoint
is kept, and that is epoch 92, already mid-runaway, so the run can't be resumed from a good state.
narrow-B (`261030-grand-cove-narrow-B`) is at epoch 506 of 650 with no sign of a runaway, and
its losses have been flat since about epoch 100.

**The runs.** Davide's handoff (2026-10-30): both jobs come from commit a8e3477 at the configs'
defaults: batch 1024 (64 observations × 16 draws), bf16 backbone, Muon lr 10⁻², AdamW 10⁻³,
weight decay 0.05, log interval 1000, and `visible_snr: 7.0` window centres. Each `config.yaml`
was checked against the handoff, and only `run` and `commit` differed. Both run on TREX, on
one A100 80 GB PCIe each.

| run | job | steps | warmup | s / 1000 steps | handoff estimate |
|---|---|---|---|---|---|
| narrow-B | 16264497 | 650 000 | 13 000 | 75 | 64 |
| full-B | 16264568 | 240 000 | 4 800 | 80 | 124 |

**full-B's runaway.** Losses are per-epoch medians. The catalog columns are the averaged weights
x on the fixed eval catalog.

| epoch | frame | flow | catalog frame | catalog flow | grad norm |
|---|---|---|---|---|---|
| 10 | 0.291 | −0.807 | 0.144 | −0.934 | 7.2 |
| 40 | −0.138 | −1.044 | −0.344 | −1.147 | 19.9 |
| 50 | −0.104 | −1.045 | −0.275 | −1.238 | 24.7 |
| 60 | 0.075 | −1.037 | −0.079 | −1.316 | 44.9 |
| 70 | 0.615 | −1.023 | 0.120 | −1.448 | 102 |
| 80 | 1.087 | −0.997 | 0.215 | −1.505 | 166 |
| 90 | 2.111 | −0.577 | 0.050 | 0.208 | 281 |
| 92 | 3.536 | 0.674 | 0.059 | 5.959 | 540 |

- **The frame head went first.** The training frame loss bottomed out at epoch 44 (−0.145) and
  catalog frame at epoch 36 (−0.355). After that both climbed, and the gradient norm grew roughly
  geometrically.
- **The flow held on for another 30 epochs.** Catalog flow kept improving to its best at epoch 77
  (−1.572), with the training flow loss flat at −1.04, even while the frame loss rose. From epoch
  88 the flow broke too. The schedule-free y weights spiked first: y frame reached 3.4, 6.7, 5.1
  and 10.1 at epochs 85, 87, 89 and 92. The NaN came at step 92 379.
- **Weights didn't grow.** Backbone w_rms fell steadily (0.093 at epoch 10, 0.033 at epoch 92),
  so the runaway isn't growing weights. That points to the frame head's loss or the optimiser
  rather than weight decay.

**narrow-B, at epoch 506.** Frame loss sits at 0.81–0.82 and flow at −0.80 from epoch 100 on.
Catalog flow went from −0.750 to −0.762 over epochs 100–506, with its best at epoch 441 (−0.766).
The gradient norm doubled in that span (5.8 to 12.7) as w_rms fell (0.027 to 0.021). That's the
same direction as full-B's lead-up, but about 20× slower.

**Not decided (next).** Rerunning full-B costs ~5.5 h at 80 s per 1000 steps. Before that it
needs a guard: keep the best catalog checkpoint as well as the latest, or stop on a rising frame
loss. It also needs a cause; suspects are the frame head's β-NLL variance and the Muon learning
rate at full band. narrow-B's eval and scorecard (`lisa-post.sbatch`) run once it finishes.
