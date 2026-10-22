# Improved Techniques for Training Consistency Models

Song & Dhariwal, 2023, ICLR 2024 · [arXiv:2310.14189](https://arxiv.org/abs/2310.14189)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 515
citations in 3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Consistency *training* (no teacher) made competitive by six changes: (1) teacher EMA μ=0, i.e. θ⁻ = stopgrad(θ) — proved necessary: for a point-mass toy, with θ⁻≠θ the N→∞ CT loss is (1−σ_min/σ)²(θ−θ⁻)², independent of the data, so it carries no signal; with θ⁻=θ the 1/Δσ-scaled gradient tends to the MSE gradient toward the truth. (2) Pseudo-Huber metric d=√(‖x−y‖²+c²)−c, c=0.00054√d. (3) λ(σ_i)=1/(σ_{i+1}−σ_i). (4) Exponential discretisation curriculum N(k)=min(s0·2^{⌊k/K'⌋},s1)+1, s0=10, s1=1280. (5) Lognormal noise sampling P_mean=−1.1, P_std=2.0 (default Karras grid over-weights high noise). (6) Less sensitive time embedding (Fourier scale 16 → 0.02) and large dropout (0.3 CIFAR) with dropout RNG shared between student and teacher.

## What they find

CIFAR-10 uncond. iCT 1-step FID 2.83 (2-step 2.46); iCT-deep 2.51 / 2.24 (CT-LPIPS was 8.70/5.83; CD-LPIPS 3.55/2.93). ImageNet-64 cond. iCT-deep 3.25 / 2.77, recall 0.63 (CT-LPIPS recall 0.47) — the recall gap of original CT closes. Ablations are curves only (Fig. 1–4): removing teacher EMA "notably improves" FID for both ℓ2 and LPIPS; pseudo-Huber c=0.03 optimal on CIFAR; FID vs N follows a power law until saturation (hence cap 1281); exponential curriculum beats const/sqrt/linear/square/cosine; lognormal beats default noise schedule. High-sensitivity time embedding makes continuous-time CT *diverge* from random init; scale 0.02 fixes it.

## Implementation details

400k iters CIFAR; large batch (4096 for ImageNet per ECT's account of iCT cost, bs 4096 × 800k); student EMA kept for inference only. Pseudo-Huber reduces variance of Adam update norms (their Fig. 6b) — the robustness argument. Two-step sampling captures most of the gain; more steps diminishing.

## Caveats

Image FID only; per-trick numbers only in plots; extremely expensive (week on 8 GPUs for CIFAR per ECT); no distributional-calibration metric beyond precision/recall.
