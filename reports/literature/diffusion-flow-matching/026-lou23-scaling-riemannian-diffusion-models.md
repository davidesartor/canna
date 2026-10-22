# Scaling Riemannian Diffusion Models

Lou, Xu & Ermon, 2023, NeurIPS 2023 · [arXiv:2310.20030](https://arxiv.org/abs/2310.20030)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 23 citations in 3.0 yr
(Semantic Scholar, 2026-10-19).

## What they do

Accurate heat kernels on symmetric spaces. The Laplacian is reduced to the maximal torus (1-D for spheres). At small t it uses Schwinger–DeWitt (Varadhan plus a curvature/volume correction) or a sum over paths (Lie groups; on tori it is exactly a wrapped Gaussian), switching to a truncated eigen-sum above a cutoff τ. This makes DSM accurate and cheap, with exact sampling via rejection or a torus ODE.

## What they find

- Earth NLL: DSM-accurate −4.69 / −0.27 / 0.44 / −1.51, vs DSM-inaccurate −1.28 / 0.13 / 0.73 / −0.60.
  - **So heat-kernel accuracy alone moves Volcano NLL by 3.4 nats.**
- CIFAR-100 OOD on the hypersphere: raw CIDER embedding FPR ~90% → 28.95 (SVHN) with the diffusion likelihood.
- Heat-kernel relative error: for S¹²⁷, Varadhan is still ~10% off when the eigen-expansion NaNs.

## Implementation details

- **Torus checkerboard comparison:** RFM "learns suboptimal distributions with noticeable artifacts" and "fails outright for more complex checkerboards". The authors attribute this to **non-smooth geodesic vector fields** (cut-locus discontinuity of the log map) that "break the smoothness assumption". Score matching learned them accurately. This is qualitative only.
- Torus MLP inputs are sin(kx), cos(kx) for k=1..6, 4 hidden layers, SiLU, Adam 1e-3, 100k steps, batch 100.
- The heat kernel is ~uniform by t≈5.
- Eigenfunctions of Sⁿ are numerically unstable.

## Caveats

The checkerboard FM failure has no numbers and uses a uniform base. Only one FM variant was tested.
