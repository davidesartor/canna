# Preconditioned Stochastic Gradient Descent, PSGD / Kron (Li, 2015 onward)

[arXiv:1512.04202](https://arxiv.org/abs/1512.04202); the Kron variant ships in the
`psgd-jax` package. The arXiv ID was not re-checked.

**Evidence: A.** Peer-reviewed (IEEE Transactions on Neural Networks and Learning Systems); 132
citations in 10.8 yr (Semantic Scholar, 2026-10-19).

## What they do

- Fit a preconditioner online by a "whitening" criterion on gradient and parameter
  perturbations.
- Kron: a Kronecker-factored preconditioner, a drop-in Adam replacement with no β₂ and no ε.

## What they find

- Competitive with or better than Adam on a range of tasks, with few hyperparameters.

## Limits

- `psgd-jax` / `psgdax` are optax-compatible, so this is the cheapest Kronecker method to
  try in JAX.
