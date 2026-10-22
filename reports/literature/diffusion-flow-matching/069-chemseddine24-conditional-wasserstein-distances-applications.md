# Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching

Chemseddine, Hagemann, Steidl, Wald, 2024; JMLR-style manuscript, v3 2025 · [arXiv:2403.18705](https://arxiv.org/abs/2403.18705)

**Evidence: B.** Peer-reviewed (Journal of machine learning research); 49 citations in 2.6 yr
(Semantic Scholar, 2026-10-19).

## What they do

W_{p,Y} = inf over plans whose (y1,y2) marginal is the diagonal; equals E_y[W_p^p(P_{X|y},P_{Z|y})], strictly larger than plain joint W. Geodesics have zero y-velocity. Practical relaxation: cost ‖x1−x2‖^p + β‖y1−y2‖^p (equivalent to scaling y by √β), β→∞ recovers the conditional plan; minibatch OT on joint (y,x) then FM loss ‖v(y,x_t)−(x−z)‖².

## What they find

5-D GMM inverse problem (Sinkhorn dist. to true posterior, 100 posteriors×10 runs): 10 Euler steps OT 0.0225 vs random 0.0247; 3 steps 0.0760 vs 0.1044. CIFAR-10 class-conditional FID: β=1 4.97, 3 4.94, 5 4.38, 20 4.33, 100 4.10, "diagonal" (within-class random) 4.92. Low β mixes classes (wrong conditioning); high β needed.

## Implementation details

OT batch 500, training batch 100 (CIFAR); POT; 140k-param MLP for GMM.

## Caveats

y in experiments is either a class label (many x per y) or low-dimensional; gains modest (≈0.8 FID; ~10% Sinkhorn distance at 10 steps, larger at 3 steps). Source-y construction not explicit.
