# A Kernel-Based Conditional Two-Sample Test Using Nearest Neighbors (with Applications to Calibration, Regression Curves, and Simulation-Based Inference)

Chatterjee, Niu & Bhattacharya, 2024, arXiv preprint · [arXiv:2407.16550](https://arxiv.org/abs/2407.16550)

**Evidence: C.** Unreviewed preprint (v2, 29 Aug 2024; no journal version found); 9 citations in
2.2 yr (Semantic Scholar, 2026-10-22). Its claims are the authors' alone: a hypothesis, not a result.

## What they do

The paper tests H₀: P_{X|Z} = P_{Y|Z} from triples (X_i, Y_i, Z_i). In simulation-based inference, Z is the observation, X the true parameter, and Y one draw from the approximate posterior given Z.

- **Measure.** The expected conditional MMD, ECMMD = E_Z ‖μ_{X|Z} − μ_{Y|Z}‖², the mean squared distance between the conditional kernel mean embeddings. With a characteristic kernel it is zero iff the conditionals are equal for almost every Z.
- **Estimator.** Build the directed K-nearest-neighbour graph of the Z_i. Average H((X_i, Y_i), (X_j, Y_j)) = K(X_i, X_j) + K(Y_i, Y_j) − K(X_i, Y_j) − K(X_j, Y_i) over the graph's edges. Cost O(Kn log n) in any dimension. The kernel is on X and Y only; Z enters only through the neighbour graph.
- **Asymptotic test.** η_n = √(nK) · ECMMD / σ̂ → N(0, 1) under H₀, with σ̂² computed from the squared H over the edges. Normality is proved for K = o(n^{1/44}). Consistency against fixed alternatives needs K = o(n / log n).
- **Model-based tests** (Section 5), for when the conditional law of Y is a model one can sample from:
  - **Finite-sample test** (Algorithm 1). Draw M fresh model samples per point, recompute η for each, and rank the observed η among them. It is valid at every n, and consistent if M > 1/α − 1.
  - **Derandomized test.** Average over M_n model draws per point. It is asymptotically normal as M_n → ∞.

## What they find

- **Classification calibration** (n = 100, K = 15 or 25, M_n = 20). Type-I error at 5%: ECMMD asymptotic 0.032–0.066, derandomized 0.040–0.054, SKCE 0.064–0.076. SKCE is slightly more powerful. ECMMD catches up as K grows and is much faster.
- **Regression calibration** (n = 50 or 75, K = 2 to 10). ECMMD has more power than SKCE, rising with K. At small perturbation ρ, ECMMD keeps non-trivial power while SKCE has almost none.
- **SBI benchmarks.** Test budget n = 1000 triples, K = 50, α = 0.05, one posterior draw per observation, 100 repeats on the same trained model. Gaussian kernel on θ with bandwidth the median of ‖θ_i − θ′_i‖. Rejection rates against training budget:
  - Two Moons (θ in 2-D, budgets 10² to 10⁵): The text says MDNs with 5 or 7 components emulate the posterior at moderate budget (1 and 3 components were also run). NSF with 5 transforms passes.
  - Simplified SLCP (θ in 5-D, budgets 10⁴ to 10⁵): MDNs with up to 25 components are still rejected. NSF passes at moderate budget.
  - These results are in figures only. I did not extract exact rejection rates.
- **Photometric redshift from images.** Z is a 20 × 20 GalSim galaxy image, used directly for the neighbour graph. The redshift is a mixture whose modes switch with the galaxy's rotation angle. The model is a 2-component convolutional MDN trained on 20,000 samples. Test n = 2500, K = 100. The true conditional has four modes for a range of angles set by a threshold θ, so the 2-component model is wrong there. Rejection rates are shown against θ and the mode offset μ (figure only).
- **Wind turbines** (real data, linear vs Gaussian kernel). The conditional-mean test does not reject in 2007–2008, while the distribution test rejects at K = 40 and 45 (p = 0.037 and 0.046). The difference is in higher moments.

## Implementation details

- **Kernel.** Gaussian, with bandwidth set by the median distance between true and emulated draws.
- **K.** 15–50 in their experiments, 100 for the image example. The theory only gives growth rates.
- **One draw per observation** suffices for the asymptotic test. The model-based tests use more draws (M_n = 20).
- **Neighbour metric on Z.** The image example builds the graph on the raw 20 × 20 images. I did not find the metric stated; Euclidean is my assumption.
- **Needs.** Samples only: no density, no likelihood, no trained classifier.
- **Code.** github.com/anirbanc96/ECMMD-CondTwoSamp.

## Caveats

- Unreviewed. The SBI and photo-z results are figure-only and use small problems.
- Power depends on the K nearest neighbours in Z-space sharing similar posteriors. For a long raw observation, Euclidean neighbours may not be informative, and the paper gives no guidance on learned embeddings of Z. This is my reading, not a claim of the paper.
- A posterior equal to the prior is not tested. In principle it is an alternative the test is consistent against, but power again needs informative neighbours. This is my reading, not a claim of the paper.
- Too-wide posteriors are not tested as a separate case in the SBI section.
- The asymptotic theory allows only very slowly growing K (o(n^{1/44})).
