# F63 (2026-10-31): narrow-B and full-B on a8e3477; full-B diverged at epoch 93: schedule-free's z ran away again (F36), and the frame's galactic-latitude width collapsed

**Outcome.** full-B (`261030-stormy-cedar-full-B`) stopped at epoch 93 of 240 on a non-finite
loss. Measured on its last checkpoint (epoch 92), the cause is [F36](036-schedule-free-z-runaway.md)'s
mechanism back at Muon lr 10⁻²:
- The schedule-free iterate z is 4–5× the averaged weights x in norm across every Muon matrix,
  with |z − x| ≈ |z|.
- The training weights y = βx + (1 − β)z (β = 0.946 by then) sit 5–27% of |x| away from x.
- In that perturbed y, the frame's width for galactic latitude b collapsed to e^{−9.5} prior
  widths on ordinary SNR 10–70 sources. The resulting k² of up to 10²⁵ ran the gradient up, the
  flow's targets z₁ = k with it, and the run hit a NaN.
- The averaged weights x stayed sane until the broken targets reached them in the last
  ~5 epochs.

narrow-B (`261030-grand-cove-narrow-B`, still training) has the same z blow-up (up to 10× x)
but no outliers, because its sources are faint: its frame stays near the prior.

**The runs.** Davide's handoff (2026-10-30): both jobs come from commit a8e3477 at the configs'
defaults: batch 1024 (64 observations × 16 draws), bf16 backbone, Muon lr 10⁻², AdamW 10⁻³,
weight decay 0.05 (on z, Muon matrices only), log interval 1000, and `visible_snr: 7.0`. Each
`config.yaml` matched the handoff except `run` and `commit`. Both ran on TREX, one A100 80 GB
PCIe each.

| run | job | steps | warmup | s / 1000 steps | handoff estimate |
|---|---|---|---|---|---|
| narrow-B | 16264497 | 650 000 | 13 000 | 75 | 64 |
| full-B | 16264568 | 240 000 | 4 800 | 80 | 124 |

**full-B's history.** Losses are per-epoch medians of the training steps, which are evaluated
at y. The catalog columns are x on the fixed eval catalog. The last column is each epoch's
worst single step.

| epoch | frame | flow | catalog frame | catalog flow | grad norm | max step frame NLL |
|---|---|---|---|---|---|---|
| 10 | 0.291 | −0.807 | 0.144 | −0.934 | 7.2 | 1.1 |
| 45 | −0.124 | −1.042 | −0.317 | −1.140 | 22.1 | 0.88 |
| 60 | 0.075 | −1.037 | −0.079 | −1.316 | 44.9 | 34 |
| 70 | 0.615 | −1.023 | 0.120 | −1.448 | 102 | 3.7 × 10⁵ |
| 80 | 1.087 | −0.997 | 0.215 | −1.505 | 166 | 6.6 × 10²¹ |
| 90 | 2.111 | −0.577 | 0.050 | 0.208 | 281 | 2.5 × 10⁵ |
| 92 | 3.536 | 0.674 | 0.059 | 5.959 | 540 | 1.1 × 10⁵ |

The signature is the training loss at y rising from epoch ~45 while x's catalog flow kept
improving to epoch 77 (−1.572): y left the basin that x sits in. The typical step stayed modest.
The damage came from rare steps whose k² was huge: the raw gradient reached 10¹⁸ against a clip
at 1, so a single sample set each such step's direction, and Muon's momentum (0.8) carried it on.

**Where the outliers come from.** 64 training batches (4096 observations) were drawn exactly as
in training and put through epoch 92's frame. Run with `frame_outliers.py` on TREX, ~1 min.

| weights | mean k², b | entries with k² > 100, b | max k², b | max k², the other 7 coordinates | median δs_b |
|---|---|---|---|---|---|
| y (training) | 6844 | 5.4% | 4.1 × 10⁷ | 2.6 × 10³ (l) | −0.62 (min −9.5) |
| x (averaged) | 0.14 | 0 | 194 | 517 | +1.05 |

- **The outliers are not loud sources.** In 80 000 full-B training sources the loudest has
  SNR 271. The worst cases are SNR 11–70 sources at 5–9 mHz, with |x₁,b| < 1, predicted at
  σ_b ≈ 10⁻⁴ prior widths. A typical one: x₁ = 0.69, μ = 0.001, log σ = −9.1, k² = 4 × 10⁷.
- **b is the weak spot** because the frame head's output layer is AdamW-routed, and
  schedule-free decays only the Muon matrices. Its z rows have no restoring force: the δs_b row
  of z has norm 15.7 against 7.0 in x, the largest of the 16 rows. cos(x, z) is 0.56–0.74 over
  the δs rows.
- **x's b is too wide** (median δs_b = +1.05, k² = 0.009), the averaged side of the same split.

**The z runaway, per parameter** (`param_drift.py`, ~20 s):

| run, epoch | Muon matrices, \|z\| / \|x\| | worst leaves | \|y − x\| / \|z − x\| (= 1 − β) | x's w_rms |
|---|---|---|---|---|
| full-B, 92 | 3.2–5.1 (backbone) | `y_mlp.linear2` 5.1, `c_mlp.linear2` 4.6 | 0.054 | 0.033 (0.093 at epoch 10) |
| narrow-B, 533 | 3–10 | `y_mlp.linear2` 10.0, `frame_modulation` 8.5, `f0_mlp.linear2` 7.6 | 0.039 | 0.021 (0.036 at epoch 50) |

Muon takes a fixed-size step (0.2 · lr RMS per element), and the decay on z pulls back by
lr · λ per step. Their balance, [F40](040-xs-sfplus-regressed.md)'s 0.2 · √(lr / 2λ), is 0.063
RMS at lr 10⁻² and λ = 0.05. The training log's z_x (the RMS of z − x) reads 0.060 on narrow-B
and 0.096 on full-B. F40's run had 0.013. That noise floor is 2–3× x's own RMS (0.02–0.03). x
averages the noise away and shrinks: x's w_rms falls steadily in both runs. So |z|/|x|, and with
it y's offset from x, grows through the run. The β anneal 0.9 → 0.965 slows y's offset but
doesn't bound it.

**narrow-B, at epoch 506.** Frame loss sits at 0.81–0.82 and flow at −0.80 from epoch 100 on.
Catalog flow went from −0.750 to −0.762 over epochs 100–506, with its best at epoch 441 (−0.766).
The largest single-step frame NLL stayed 1.1–1.2 through all 521 epochs. On its live epoch-534
checkpoint, k² > 100 occurs in 1.8 × 10⁻⁴ of entries for y and 1.6 × 10⁻⁴ for x, so y and x behave
the same. Its sources are faint (training SNR median 1.7, max 31), so its frame barely leaves the
prior and the y perturbation costs nothing visible. Whether the same noise is what holds its
losses flat since epoch ~100 is open.

**Not decided (next).** Rerunning full-B costs ~5.5 h. Candidate changes, to fold into one run:
- bound z: a lower Muon lr (the noise floor scales as √lr, so 10⁻³ gives 0.020), or decay on z
  for the AdamW leaves too, at least the frame head's output layer;
- cap the frame's width from below, e.g. a soft floor on δs, so that no single sample's k² can
  set a step;
- keep the best catalog checkpoint as well as the latest, since the only checkpoint here is
  mid-runaway and can't be resumed.

narrow-B's eval and scorecard are queued behind it (`lisa-post.sbatch`, `afterok`).
