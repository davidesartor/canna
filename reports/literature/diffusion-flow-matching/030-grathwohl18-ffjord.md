# FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models

Grathwohl, Chen et al., 2018, ICLR 2019 · [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 1129
citations in 8.1 yr (Semantic Scholar, 2026-10-19).

## What they do

CNF trained by maximum likelihood with the instantaneous change of variables, $\log p(z(t_1)) = \log p(z(t_0)) - \int \mathrm{Tr}(\partial f/\partial z)\,dt$, integrated jointly with the state (augmented ODE). The O(D²) exact trace is replaced by Hutchinson's estimator $\epsilon^T (\partial f/\partial z)\epsilon$ (one VJP per eval), with $\epsilon$ held fixed across one solve so the dynamics stay deterministic and the estimator unbiased. Bottleneck trick: use cyclic trace to estimate an H×H trace when a hidden layer is narrower than D (lower variance). Training via the adjoint method.

## What they find

best reversible (one-pass) model on tabular NLL: POWER −0.46, GAS −8.59, HEPMASS 14.92, MINIBOONE 10.43, BSDS300 −157.40 nats (beats Glow/RealNVP; loses to autoregressive MAF-DDSF on most). MNIST 0.99 bits/dim (multiscale), CIFAR10 3.40 (Glow 3.35) with <2% of Glow's params. VAE negative ELBO: MNIST 82.82 vs Sylvester 83.32. Ablations: bottleneck trick speeds convergence with Gaussian ε but not with Rademacher ε; multiscale uses ~2× NFE of single-scale but reaches lower loss; NFE becomes independent of D (VAE latent sweep) — it depends on the complexity of the transport, not dimension.

## Implementation details

Hutchinson only during training; the **exact trace is used for reported test likelihoods** wherever feasible (variance of Hutchinson-based val LL <1e-4 on images). Time concatenated to the input of every layer (hypernetworks tried, not better). RK4(5) Dopri, tol 1e-5 (tabular atol 1e-8, rtol 1e-6); error in density normalisation tracks solver tolerance, becomes non-negligible above 1e-5 and training can diverge. NFE grows during training (70–100 on 2D toys) and can become prohibitive; small weight decay keeps dynamics non-stiff; spectral norm/weight decay reduce NFE but hurt NLL slightly. Conditioning a CNF on an encoder's output directly made ODEs "too difficult to integrate"; used low-rank data-dependent weight updates instead.

## Caveats

max-likelihood-trained CNF (not flow matching); 2018-scale; no study of density accuracy for very concentrated targets; solver tolerance and the trace variance interact with precision (all f32).
