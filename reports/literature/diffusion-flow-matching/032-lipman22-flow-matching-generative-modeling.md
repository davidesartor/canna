# Flow Matching for Generative Modeling

Lipman et al., 2022, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 7072
citations in 4.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Regress a network v_t(x) onto a target field u_t that generates a chosen probability path p_t. The marginal u_t is intractable, so the method regresses instead onto a per-sample conditional field u_t(x|x1) under a conditional path p_t(x|x1). The CFM loss E_{t,x1,x~p_t(.|x1)}‖v_t(x) − u_t(x|x1)‖² has the same gradient as the marginal FM loss (Thm 2). For Gaussian conditional paths N(μ_t(x1), σ_t(x1)²), the conditional field is u_t(x|x1) = σ'_t/σ_t (x − μ_t) + μ'_t (Thm 3). The "OT" path μ_t = t x1, σ_t = 1 − (1 − σ_min)t gives the regression target x1 − (1 − σ_min)x0, which is constant along each conditional trajectory. VP/VE diffusion paths are special cases of the same construction.

## What they find

The ablation uses the same U-Net, hyperparameters and epochs for every method (Table 1).
- CIFAR-10, NLL/FID/NFE: FM-OT 2.99 / 6.35 / 142; FM with the VP diffusion path 3.10 / 8.06 / 183; DDPM 3.12 / 7.48 / 274; score matching 3.16 / 19.94 / 242; ScoreFlow 3.09 / 20.78 / 428.
- ImageNet-64: FM-OT 3.31 / 14.45 / 138 vs DDPM 3.32 / 17.36 / 264.
- ImageNet-128: FID 20.9.
- Two effects can be separated. Changing the objective from score matching to FM on the same diffusion path improves FID and NFE. Changing the path from diffusion to OT improves them again.
- Fixed-step sampling: FM-OT needs about 60% of the NFE to reach the same ODE error. FM converges faster in training (ImageNet-128: 500k iterations at batch 1.5k vs 4.36M at batch 256 for ADM).
- 64→256 super-resolution: FID 3.4 vs 5.2 for SR3.

## Implementation details

- Training: Adam (0.9, 0.999), no weight decay. Polynomial-decay LR with warmup, peak 1e-4 to 5e-4.
- t is drawn uniformly. Diffusion baselines sample t in [0, 1 − 1e-5].
- Evaluation uses dopri5 at tolerance 1e-5.
- Precision: fp32 for CIFAR and ImageNet-32; 16-bit mixed precision at higher resolution.
- σ_min is "sufficiently small"; the HTML gives no value. It is commonly 1e-4 or 0.

## Caveats

- Only Gaussian base and Gaussian conditional paths in Euclidean space. The marginal field is not OT even though each conditional path is.
- CIFAR FIDs are weak because the architecture was not tuned for CIFAR.
- The paper does not study loss weighting or time densities.
