# On the Spectral Bias of Neural Networks

Rahaman et al., 2018, ICML 2019 · [arXiv:1806.08734](https://arxiv.org/abs/1806.08734)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 2785 citations
in 8.3 yr (Semantic Scholar, 2026-10-19).

## What they do

A ReLU net is continuous piecewise-linear, so its Fourier transform is a rational function $\tilde f(k)=\sum_n C_n(\theta,k)\,1_{H_n}(k)/k^{n+1}$: decay $k^{-d-1}$ in generic directions, as slow as $k^{-2}$ orthogonal to region faces. Numerators are bounded by (#linear regions)×(Lipschitz const), and the Lipschitz constant is bounded by $\prod\|W^{(k)}\|$, which grows only gradually under GD; hence high frequencies need large weights and arrive late. They also show $|d\tilde h(k)/dt| = O(k^{-\Delta})$ for the residual, and for a single-frequency target $|\partial_t\tilde h(k_0)|=O(k_0^{-1})$ (App. C.5).

## What they find

Exp. 1: 6×256 ReLU MLP, full-batch Adam, 80k steps, 200 points on [0,1], target $\sum_i A_i\sin(2\pi k_i z+\varphi_i)$, $k=5..50$: low frequencies converge first *even when high frequencies have 10× larger amplitude* ($A$ from 0.1 to 1). Exp. 2: after convergence, isotropic parameter perturbations destroy high-frequency components first (low-frequency ones are more robust). Exp. 3/4 (MNIST, binary): low-frequency target noise hurts validation, high-frequency noise is fit only late (early-stopping dip ~iteration 200). Exp. 5/6: the same latent signal ($k$ up to 200–400) is *easier* to fit when the data manifold embedding itself oscillates (flower curve with L petals): with $L=16$ high latent frequencies are learned, with $L=0$ (plain circle) they are not (Fig. 8/9). Mechanism (stationary phase): $l=J_\gamma(\bar z)k$, latent frequency $l$ can be produced by small input frequency $k$ if the embedding has high-frequency coordinate functions.

## Implementation details

Adam lr 3e-4, full-batch (no SGD, to isolate the effect); qualitative ablations: more depth/width/larger weight clip ⇒ more high-frequency power (App. A.5, δ-function target).

## Caveats

all quantitative experiments are 1-D or 2-D inputs (or MNIST via kernel eigenfunctions as a "frequency" proxy); no transformers, no conditional generative models; the theory is a bound, not a rate.
