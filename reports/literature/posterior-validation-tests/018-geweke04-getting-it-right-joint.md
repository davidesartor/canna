# Getting It Right: Joint Distribution Tests of Posterior Simulators

Geweke, 2004, J. Am. Stat. Assoc. 99(467), 799–804 · [DOI:10.1198/016214504000001132](https://doi.org/10.1198/016214504000001132)

**Evidence: A.** Peer-reviewed (JASA); 115 citations in 22.1 yr (Semantic Scholar,
2026-10-22).

## What they do

A model is a proper prior p(θ) and a data density p(y | θ); together they give the joint p(θ, y) (Eq. 1). Improper priors are handled only as a limit of proper ones (§4). Pick test functions g(θ, y) with finite variance under the joint (Eq. 2) and estimate E[g] two ways.

- **Marginal-conditional simulator** (Eqs. 3–4). θ̃⁽ᵐ⁾ ~ p(θ), ỹ⁽ᵐ⁾ ~ p(y | θ̃⁽ᵐ⁾). Independent draws, so ḡ has an ordinary standard error σ̂/√M₁.
- **Successive-conditional simulator** (Eq. 5). Start from a prior draw and alternate one step of the posterior simulator, θ⁽ᵐ⁾ ~ q(θ | θ⁽ᵐ⁻¹⁾, y⁽ᵐ⁻¹⁾), with a fresh y⁽ᵐ⁾ ~ p(y | θ⁽ᵐ⁾). If q has p(θ | y) as its invariant distribution and the chain is ergodic, the pairs converge to the same joint.
- **Test** (Eq. 6): z = (ḡ − g̃) / (σ̂²/M₁ + τ̂²/M₂)^{1/2}, with τ̂² a time-series variance estimate for the dependent run (computed with the BACC software). Many g are tested at once with a Bonferroni bound.

An error can sit in the prior simulator (3), the data simulator (4) or the posterior simulator (5); (5) "is usually the most likely candidate" because it is the most complicated, but the test cannot say which. The §3 cases are errors introduced on purpose; §4 and §5 rerun archived code from two of the author's published papers.

## What they find

- **Two-component t mixture** (§3; ν = 5, θ = (μ₁, μ₂, σ₁², σ₂², p), six observations, latent labels s and scales ω, two Gibbs samplers MCMC1 and MCMC2, errors introduced in MCMC1; the 5 first and 15 second moments of θ as test functions; 2.5 × 10⁵ iterations of each simulator, under a minute). Table 1, rejections out of 20 at .05 / .01 / .005 / .001:
  - correct code: 0 / 0 / 0 / 0 for both samplers;
  - beta(1, 1) in the prior simulator where the sampler assumes beta(2, 2): 4 / 3 / 3 / 2;
  - data simulator in the successive-conditional run ignoring the sampler's ω and drawing fresh ones: 10 / 9 / 9 / 9;
  - 5 instead of 6 degrees of freedom in the conditional posterior of ω: 5 / 3 / 3 / 3;
  - conditional posterior variance of (μ₁, μ₂) set to 0: 11 / 10 / 10 / 9;
  - ω drawn several steps after s instead of right after it: 7 / 6 / 6 / 6.
  Here g depends on θ only, so "it is not necessary to generate y⁽ᵐ⁾" in the marginal-conditional run.
- **Unit-root model** (§4; Geweke 1994, made proper with .01/σ² ~ χ²(4); T = 10; 90 test functions on parameters only). Marginal-conditional: 10⁶ draws, about 30 s. Successive-conditional: 4 × 10⁶ iterations keeping every 400th, about 25 min. The Bonferroni critical value is Φ⁻¹[.001/(2·90)], |z| > 4.394, and 37 tests exceeded it. The bug: the conditional draw of (a₁..a₄) actually came from its marginal in the joint conditional of (a₁..a₄, σ), and the same routine did the same for (γ, δ). After the fix: 2 rejections at .05, none at .01. Table 3 (per capita real GNP, one of six series) shows the bug mattered for the published results: the Bayes factor for ρ = 1 was .445 instead of .0462 (ten times too high), for s = 17 .639 instead of .274, and the posterior mean of ρ about .90 instead of .85. Raising burn-in from 200 to 2,000 also changed some moments, but the coding error mattered an order of magnitude more.
- **Reduced-rank regression** (§5; Geweke 1996; L = 4, p = 3, q = 2, T = 6; 46 test functions, including the eigenvalues of Y′Y, i.e. functions of the data). Marginal-conditional 10⁶ draws in 6.5 min; successive-conditional every 40th of 4 × 10⁶ iterations, about an hour. Normalisation 1 passed (2 of 46 at .05, 0 at .01). Under normalisation 2 the successive-conditional run hit exponent overflow after several thousand iterations; a 10³-iteration run gave 10 rejections at .05, 6 at .005 and 2 at .001, with relative numerical efficiencies as low as .001. The cause: a precision τ = 0 coded for Ψ*, so the joint did not exist. Fixing that left 12 rejections at .05 and 8 at .001 (among them |z| = 20.32 and 16.50 for the eigenvalues of Y′Y), traced to a second error in the Φ and Σ conditionals. After both fixes: 2 at .05, 0 at .01.
- "Joint distribution tests indicate the existence of errors, but not their source" (§4); the failing test functions still point where to look first.
- §6: the tests need only the prior and data simulators added to existing code, so a reader or client can check a published sampler.

## Implementation details

- Eq. 5 takes q(θ | θ⁽ᵐ⁻¹⁾, y), a transition kernel. An i.i.d. posterior sampler is a valid kernel that ignores θ⁽ᵐ⁻¹⁾, so an amortised flow fits the scheme, but the M steps are sequential: each y depends on the previous θ.
- The paper's runs use tiny datasets (6 or 10 observations) and heavy thinning (every 40th or 400th). It does not discuss how mixing scales with data size; our inference is that a narrow posterior makes the chain move slowly, which an exact i.i.d. sampler avoids.
- g needs finite variance under the joint (Eq. 2), so heavy-tailed priors need bounded test functions.
- Choose g that involve y as well as θ (as in §5); with q = prior, θ⁽ᵐ⁾ is independent of y⁽ᵐ⁾ (our derivation), so every parameter-only g has the right mean while a g coupling θ and y does not.

## Caveats

- Cook, Gelman & Rubin (2006; [009](009-cook06-validation-software-posterior-quantiles.md)) report Geweke's method took about 4 min to converge (R̂ < 1.2) against about 1 min for theirs, and that a sampler returning the prior passes both. That holds for the parameter-only test functions they and two of Geweke's three examples use, not for Eq. 2 in general.
- Modrák et al. (2023; [001](001-modrak23-sbc-test-quantities-sensitivity.md)) show the data-averaged-posterior condition differs from SBC: each can pass while the other fails.
- Moment z-tests detect mean shifts in g only; Grosse's blog post on the method recommends P–P plots of the two runs as well.
- Bonferroni over many correlated g is conservative; power against a given error depends on picking a g that moves.
- Full text read (journal PDF, via text extraction).
