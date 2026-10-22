# Improved Denoising Diffusion Probabilistic Models

Nichol & Dhariwal, 2021, ICML 2021 · [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 6011 citations
in 5.7 yr (Semantic Scholar, 2026-10-19).

## What they do

(1) Learn Σ_θ as a log-space interpolation, Σ=exp(v log β_t + (1−v) log β̃_t), trained by L_hybrid = L_simple + 0.001·L_vlb with stop-gradient on μ in the VLB term. (2) Cosine schedule ᾱ_t = f(t)/f(0), f(t)=cos²(((t/T)+s)/(1+s)·π/2), s=0.008 (chosen so √β₀ is just below the pixel bin 1/127.5), β clipped at 0.999. Motivation: with the linear schedule the last ~20% of reverse steps can be skipped with little FID loss, so that part of training is wasted. (3) Importance sampling for the pure VLB: p_t ∝ √E[L_t²], estimated from a history of the last 10 losses per t (uniform until every t has 10), loss reweighted by 1/p_t.

## What they find

ImageNet-64, 200k iterations (NLL bits/dim, FID): linear T=1k L_simple 3.99/32.5; T=4k 3.77/31.3; linear hybrid 3.66/32.2; **cosine simple 3.68/27.0**; cosine hybrid 3.62/28.0; cosine VLB (importance-sampled) 3.57/56.7. 1.5M iterations: hybrid 3.57/19.2, VLB 3.53/40.1. CIFAR-10 500k: linear simple 3.73/3.29 (T=1k); cosine hybrid 3.17/3.19; VLB 2.94/11.47. So the cosine schedule is worth ~5 FID and 0.1 bpd at 64², T 1k→4k is worth 0.22 bpd. Importance sampling lets L_vlb beat L_hybrid on NLL; it does **not** help L_hybrid. Learned sigmas keep FID near-optimal at 100 sampling steps (vs 4000 trained); DDIM is better below 50 steps. Class-conditional ImageNet-64 FID 2.92, recall 0.71 vs BigGAN-deep 4.06/0.59. Ensemble θ_vlb at the first/last 100 steps + θ_hybrid in the middle: FID 19.9, NLL 3.52.

## Implementation details

add schedule offset s so the first noise level is not below the data quantisation; linear schedule still better at 256²; adaptive GroupNorm(h)(w+1)+b conditioning; cosine-schedule models overfit faster (needed dropout 0.3 vs 0.1 on CIFAR).

## Caveats

discrete-time DDPM; likelihood-oriented; the importance sampler targets gradient variance of the VLB, not sample quality.
