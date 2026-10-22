# The Complex Gradient Operator and the CR-Calculus

Kreutz-Delgado, 2009, UCSD ECE275 lecture supplement (arXiv only) ·
[arXiv:0906.4835](https://arxiv.org/abs/0906.4835)

**Evidence: A** by the citation rule. Lecture notes, not peer-reviewed; 513 citations in
17.3 yr (Semantic Scholar, 2026-10-19). It is the standard reference for the Wirtinger
convention that autodiff frameworks adopt.

## What they do

A self-contained treatment of calculus for **real-valued functions of complex vectors**,
f: ℂⁿ → ℝ. That is the case for every loss over complex parameters, and such an f is never
holomorphic unless it is constant. Using the Wirtinger (CR) calculus, z and z̄ are treated
as independent variables:

∂/∂z = ½(∂/∂x − i ∂/∂y),  ∂/∂z̄ = ½(∂/∂x + i ∂/∂y).

The notes derive, in this notation:

- first-order expansions;
- the complex gradient under a general metric;
- complex Hessians and second-order expansions;
- Newton and Gauss–Newton steps.

The aim is optimisation directly in complex coordinates, without splitting into
(Re, Im) by hand.

## What they find

- **Real-valued f.** The two Wirtinger derivatives are conjugates of each other,
  ∂f/∂z̄ = conj(∂f/∂z), so one of them carries all the information.
- **Steepest ascent.** Under the Euclidean metric it is along the conjugate cogradient
  (∂f/∂z̄)ᴴ, up to a convention-dependent factor of 2. Equivalently, it is the ordinary
  real gradient of f(x, y) repacked as x + iy.
- **Holomorphic maps** (∂g/∂z̄ = 0) inside the network, for example complex linear layers,
  compose with the plain complex chain rule. Non-holomorphic pieces such as |z|, arg z or
  split activations need both derivatives.
- **Second order.** The full complex Hessian needs both the ∂²/∂z∂zᴴ and ∂²/∂z∂zᵀ blocks.
  Dropping the second (a "holomorphic-only" Hessian) is wrong for a generic real loss.

## Implementation details

- **PyTorch** returns the *conjugate* Wirtinger derivative ∂L/∂z̄ for real L. Its docs say
  the negative of this "is precisely the direction of steepest descent", so existing
  optimisers work on complex parameters unchanged (docs 2.14, "Autograd for Complex
  Numbers").
- **JAX** `grad` of a real-valued f(z) returns the conjugate of that. The cookbook's example
  is grad(|z|²) at 3+4i = 6 − 8i, and it advises descending "in the direction of the
  conjugate of grad(f)(x)" (JAX Autodiff Cookbook, "Complex numbers and
  differentiation").
- Our reading of the two docs: a JAX update `p − lr·grad` on a complex leaf moves the
  imaginary part the wrong way unless the gradient is conjugated first.
- We did not verify how each Optax transform handles complex leaves. Optax does provide
  `optax.contrib.split_real_and_imaginary`, which runs an inner transform on (Re, Im) as
  two real arrays.

## Caveats

- These are teaching notes. The results are standard (Brandwood 1983, van den Bos 1994),
  not new.
- The framework conventions above come from official docs, not from the paper. They are
  cited here because this is the file covering optimisation of complex parameters.
- Adam-type optimisers are outside the notes' scope. PyTorch documents that optimisers step
  complex parameters as their `view_as_real` pair, so the second moment is per real or
  imaginary component and not |g|². Some codebases override this (see file 70).
