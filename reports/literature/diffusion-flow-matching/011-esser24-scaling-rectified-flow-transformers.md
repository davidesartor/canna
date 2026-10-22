# Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

Esser et al., 2024, ICML 2024 · [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 4994 citations
in 2.6 yr (Semantic Scholar, 2026-10-19).

## What they do

Rectified flow z_t=(1−t)x₀+tε trained with CFM, but t drawn from a non-uniform density π(t). Sampling t~π is equivalent to the uniform-t loss with weight w_t^π = t/(1−t)·π(t) (Eq 18, in their ε-weighted unified form w_t^RF = t/(1−t)). Three densities: (i) logit-normal π_ln(t;m,s)=1/(s√(2π))·1/(t(1−t))·exp(−(logit t − m)²/(2s²)), sampled as t=sigmoid(u), u~N(m,s); (ii) "mode with heavy tails" f_mode(u;s)=1−u−s(cos²(πu/2)−1+u), t=f_mode(u), u~U, monotone for −1≤s≤2/(π−2), s=0 is uniform, s>0 favours middle, s<0 favours endpoints (non-vanishing density at 0,1); (iii) CosMap t=1−1/(tan(πu/2)+1), density 2/(π−2πt+2πt²), which reproduces the cosine schedule's log-SNR. Resolution shift: toy argument with a constant image of n pixels, σ(t,n)=t/(1−t)·1/√n; equating uncertainty across resolutions gives t_m = α t_n /(1+(α−1)t_n), α=√(m/n) (Eq 23), equivalent to a log-SNR shift λ_{t_m}=λ_{t_n}−2 log α (Eq 24–25). Also MMDiT (separate weights per modality, joint attention) and QK-RMSNorm.

## What they find

61 formulations × 2 datasets (ImageNet-as-captions, CC12M), ranked by non-dominated sorting over CLIP/FID on 24 settings (Table 1). Average rank: rf/lognorm(0,1) 1.54 (best overall; 1.25 at 5 steps, 1.50 at 50 steps); rf/lognorm(1.0,0.6) 2.08; rf/lognorm(0.5,0.6) 2.71 but rank 8.5 at 5 steps and 1.0 at 50 steps (narrow mid-biased densities trade few-step quality for many-step quality); rf/mode(1.29) 2.75; eps/linear (LDM) 2.88; rf/cosmap 4.13; edm(0,0.6) 5.63; **plain rf with uniform t 5.67**; v/linear 6.83; edm(−1.2,1.2) (EDM defaults) 15.58. Table 2 (25 steps, FID ImageNet/CC12M): rf uniform 49.70/94.90 → rf/lognorm(0,1) 45.78/89.91; rf/lognorm(−0.5,1) 45.64/89.70; rf/mode(1.75) 44.39/94.06; rf/lognorm(1.0,0.6) 114.26/147.69 (best CLIP, terrible FID). So logit-normal vs uniform t is a ~8% FID gain at this scale, and over-concentrating the density is catastrophic for FID. Resolution shift: human preference rises steeply for α>1.5, little difference among larger α; α=3.0 used for training and sampling at 1024². QK-norm: without it bf16 mixed precision diverges at high res (attention-logit growth); fp32 fixes it at ~2× cost; RMSNorm on Q,K + AdamW ε=1e−15 makes bf16 stable. MMDiT beats DiT (concat), CrossDiT and UViT on CC12M val loss/CLIP/FID (Fig 4, no numbers in text); 3 weight sets vs 2 gives only a small gain.

## Implementation details

AdamW lr 1e−4, 1000 warmup, batch 1024, EMA 0.99 every 100 steps for the ablation. Validation loss evaluated *stratified* on 8 equally spaced t (losses differ wildly across t); for scaling plots, average over equidistant t levels excluding t=1. Model size parametrised by depth d: hidden 64d, heads d. Validation loss correlates strongly with GenEval/human preference. Larger models need fewer sampling steps. Depth-38 needed an lr drop at 300k steps to avoid divergence.

## Caveats

Only images (latent space, d=16 channels); ranking is a Pareto/rank aggregate, not a single metric; ablation models are small and short; the density choice is confounded with sampler step count; the shift α=√(m/n) comes from a constant-image toy model and is tuned empirically, not validated quantitatively on FID.
