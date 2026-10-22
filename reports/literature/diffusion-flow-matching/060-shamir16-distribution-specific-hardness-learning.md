# Distribution-Specific Hardness of Learning Neural Networks

Shamir, 2016, JMLR 2018 · [arXiv:1609.01037](https://arxiv.org/abs/1609.01037)

**Evidence: A.** Peer-reviewed (Journal of machine learning research); 121 citations in 10.1 yr
(Semantic Scholar, 2026-10-19).

## What they do

Two negative results. (i) For a fixed, "nice" target network $[\sum_i[\langle w_i^\star,x\rangle]_+]_{[0,1]}$, no *linearly-invariant* algorithm (e.g. GD after whitening/preconditioning; Adagrad/BN-like) learns it for all input distributions (reduction from intersections of halfspaces). (ii) For a fixed nice input distribution (any mixture of densities whose square root is Fourier-concentrated: Gaussians, GMMs, Schwartz densities) the target $x\mapsto\psi(\langle w^\star,x\rangle)$ with $\psi$ 1-periodic is not learnable by gradient methods: for random direction of $w^\star$ with $\|w^\star\|=2r$, $\mathrm{Var}_{w^\star}(\nabla F_{w^\star}(w)) \le c_2 G_w(e^{-c_3 d}+\sum_{n\ge1}\epsilon(nr))$, i.e. $\exp(-\Omega(\min\{d,r^2\}))$ for Gaussians — for *any* predictor family $f(w,x)$ (improper learning allowed).

## What they find

Thm 4: with an ε-approximate gradient oracle, $\varepsilon=\sqrt[3]{\text{bound}}$, the output after $p/\varepsilon$ iterations is independent of $w^\star$ w.p. $1-p$ — i.e. $\exp(\Omega(\min(d,r^2)))$ iterations needed. Worked example (cos target, cos hypothesis, 2-D Gaussian, $w^\star=(2,2)$): the loss has only three critical points (max at 0, minima at $\pm w^\star$) — *no spurious minima or saddles* — but it is flat everywhere except near the optimum (Fig. 2). The mechanism: in Fourier space the target is localised around integer multiples of $w^\star$, convolved with the narrow $\hat\varphi$; almost no L² mass of any fixed predictor overlaps it.

## Implementation details

none (theory). Notes that gradient normalisation cannot rescue it: the needed normalisation is huge and needs gradient precision beyond machine precision; with finite-precision/SGD noise the signal is "completely obliterated".

## Caveats

hardness requires the period to be short relative to the input spread along $w^\star$ (large $r=\|w^\star\|/2$) *and* the direction to be unknown in moderately high d; it is about a *single scalar periodic ridge function of an unknown linear projection*. A target that is periodic in a *known* or low-dimensional coordinate, or a phase that winds once around an origin (arg of a complex projection), is not covered.
