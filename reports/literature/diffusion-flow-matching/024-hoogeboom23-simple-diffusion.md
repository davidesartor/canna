# simple diffusion: End-to-end diffusion for high resolution images

Hoogeboom, Heek, Salimans, 2023, ICML 2023 · [arXiv:2301.11093](https://arxiv.org/abs/2301.11093)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 489 citations in
3.7 yr (Semantic Scholar, 2026-10-19).

## What they do

Average-pooling s×s pixels multiplies SNR by s² (Eq 4), so a schedule designed at 64² adds too little noise at d². Define the schedule at a reference resolution and shift: SNR^{d×d}(t)=SNR^{64×64}(t)·(64/d)², i.e. log SNR(t) = −2 log tan(πt/2) + 2 log(64/d) (Eq 5). Interpolated schedule (Eq 6): log SNR = t·logSNR_shift256(t) + (1−t)·logSNR_shift32(t), so high-frequency detail (low noise end) uses a small shift and global structure (high noise end) uses a large one. Also: multiscale loss Σ_s (1/s)·L^{s×s} on downsampled ε, v-prediction, scaling only the 16×16 level, dropout only at low resolutions, DWT / patching as first layer.

## What they find

ImageNet FID50k (train/eval), Table 2: 128²: cosine 2.96/3.38 → shift-to-64 2.41/3.03 → shift-to-32 2.26/2.88. 256²: cosine 7.65/6.87 → shift128 5.05/4.74 → shift64 3.94/3.89 → shift32 3.76/3.71 (≈2× FID from the shift alone). Shift-64 vs shift-32 difference small; shift-64 recommended because faster early. Multiscale loss (Table 6): hurts at 256 (3.76→4.00), helps at 512 (4.85→4.30). Dropout (Table 3, 128²): no dropout 3.74/3.91 vs dropout from 64: 2.27/2.85; dropout at full res 3.19/3.85. Downsampling at 512² (Table 5): none 5.60 → DWT-2 4.85, Conv-4×4 5.04, and ~1.46× faster. Final: 2.26 (128²), 3.76 (256²), 4.30 (512²) FID without guidance (U-Net); U-ViT 2B 1.94/2.77/3.54.

## Implementation details

VP process, α²=sigmoid(logSNR), σ²=sigmoid(−logSNR). v-prediction "trains more reliably, especially at larger resolutions"; loss is unweighted ε-mse. Shifted schedules tolerate little guidance; interpolated schedule recommended when guidance is used. The noise-schedule shift moves the *training* density of log-SNR; the sampler uses the same schedule.

## Caveats

Images only; FID only; the shift rule is justified for spatially redundant data (pooling) — for data without such redundancy the argument does not apply as is.
