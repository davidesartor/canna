# PQMass: Probabilistic Assessment of the Quality of Generative Models Using Probability Mass Estimation

Lemos, Sharief, Whitammer, Salhi, Stone, Perreault-Levasseur & Hezaveh, 2024, ICLR 2025 · [arXiv:2402.04355](https://arxiv.org/abs/2402.04355)

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 12
citations in 2.7 yr (Semantic Scholar, 2026-10-22).

## What they do

PQMass is a sample-only two-sample test. It needs no density, no likelihood and no trained network.

1. **References.** Draw n_R reference points, half from each sample set. Remove them from the sets, so the regions are independent of the test points.
2. **Partition.** The references define Voronoi cells. Each remaining point is assigned to its nearest reference.
3. **Test.** Count the points of each set in each cell. Pearson's χ² on the 2 × n_R table of counts, χ²_PQM, follows χ²(n_R − 1) under H₀. This gives a p-value.
4. **Repeat.** Re-draw the references several times and report the mean and spread of χ²_PQM. When samples are scarce and the same points are re-tessellated, the mean χ² is no longer χ²-distributed. A permutation test on the mean then gives an exact p-value (Appendix I).

They prove that two cells already distinguish p ≠ q with positive probability (Prop. 2.1). As n_R grows, the chance that the tessellation separates p and q goes to 1.

## What they find

- **Samplers on a 2-D Gaussian mixture** (10,000 samples each, n_R = 100, 20 re-tessellations): χ²_PQM for MCMC 1621 ± 82, FAB 678 ± 88, GGNS 109 ± 12. This ranking agrees with W₂, RBF MMD and JSD. On a 10-D Neal funnel: MCMC 855 ± 50, FAB 97 ± 14, GGNS 473 ± 18. That ranking agrees with MMD but not with W₂.
- **Mode dropping** in a mixture of 10 Gaussians in R^d (5,000 vs 5,000). PQMass stays stable as d grows, while W₂ and linear MMD get noisier.
- **Additive noise on ImageNet** (pixel space, no feature extractor). Mean χ²_PQM rises from 95.55 at noise variance 0 to 107.63 at 0.01, 181.00 at 0.05 and 352.49 at 0.10. FLD with DINOv2 features moves much less (9.29 → 11.93 over the same range).
- **Small sample, 2-D shift.** With 50 samples each from N(0, I) and N(0.5, I), PQMass confidently detects the difference. Two Schilling nearest-neighbour tests do not (p = 0.56 and 0.71).
- **Scale change in 100-D.** With 1000 samples each from a standard Gaussian, and one set scaled by 1.1, the permutation test gives p = 0.02. The null case gives p = 0.58.
- **Choice of metric.** In a GMM test (2¹² samples per set, 2¹⁴ repeats), correlation and cosine metrics miss a 1.08 rescaling because they ignore scale. They are the most sensitive to a small rotation. Euclidean distance catches both.
- **Ablation on CIFAR-10 generative models.** The ranking is stable for n_R = 50, 100, 200 and for an L1 metric. Inception-v3 features change some values.
- **Cost.** On 256 × 256 images, 1000 evaluations take under an hour on one A100 GPU, about 3 s per χ² evaluation.

## Implementation details

- **Defaults.** n_R = 100 reference points and the Euclidean (L2) metric. Repeat with new references, about 20 times in their tables.
- **References** are sampled from a uniform mixture of the two empirical sets, so the cells cover both.
- **Null.** χ²(n_R − 1) for one tessellation on fresh points. In high dimensions, re-tessellations of the same points are close to χ² under the null; in low dimensions they are not, and the permutation test is the fallback.
- **Cost.** Nearest-reference assignment only, linear in the number of points times n_R.
- **Code.** github.com/Ciela-Institute/PQM.

## Caveats

- Unconditional. It compares two sample sets, not a conditional q(θ|y). It could be run on joint samples (θ, y) from the simulator against (θ̃, y) with θ̃ ~ q(·|y). That use is my inference, not something the paper tests. Its power there would depend on the metric over the concatenated (θ, y) space, and a long y would dominate Euclidean distances. This is my reading, not a claim of the paper.
- Run per observation, with exact posterior samples as the other set, it tests a single posterior. That needs reference posterior samples, such as MCMC.
- The metric decides what PQMass can see, as the cosine example shows.
- Most experiments are generative-model evaluation (images, proteins, samplers), not posterior validation.
