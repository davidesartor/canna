# On the Cryptographic Hardness of Learning Single Periodic Neurons

Song, Zadik, Bruna, 2021, NeurIPS 2021 · [arXiv:2106.10744](https://arxiv.org/abs/2106.10744)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 39 citations in 5.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

Weakly learning $f(x)=\cos(2\pi\gamma\langle w,x\rangle)$, $w\in S^{d-1}$, $x\sim N(0,I_d)$, from labels with adversarial noise $|\xi_i|\le\beta$. Three regimes: (a) constant small β: information-theoretically learnable with $O(d\log(\gamma/\beta))$ samples by an $\exp(d\log(\gamma/\beta))$-time covering algorithm; (b) $\beta\ge 1/\mathrm{poly}(d)$ and $\gamma\ge 2\sqrt d$: as hard as worst-case approximate SVP (via CLWE) for *any* poly-time algorithm, not just gradient/SQ; holds for any 1-periodic O(1)-Lipschitz ϕ; (c) $\beta\le\exp(-\tilde O(d^3))$: poly-time learnable from $d+1$ samples by LLL lattice reduction (integer-relation detection recovers the unknown periods, then the cosine is inverted and w solved linearly). Corollary: noiseless phase retrieval with optimal d+1 samples.

## What they find

frequency threshold $\gamma\ge2\sqrt d$; noise threshold inverse-polynomial; LLL time $O(d^6\log^3 M)$. Noise is *necessary* for hardness: the gradient/SQ lower bounds (Shamir'18, Song'17) hold even noiseless, yet a non-gradient algorithm succeeds noiselessly.

## Implementation details

none for NN training; the positive algorithm is lattice-based and numerically fragile (needs exponentially small noise).

## Caveats

worst-case-to-average-case complexity result; requires frequency growing with dimension and an unknown direction; says nothing about moderate γ or low-dimensional projections; no experiments.
