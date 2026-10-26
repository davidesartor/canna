# Heteroscedastic loss takeaways for canna.lisa

What we take from the papers in this folder for learning a per-component scale σ next to the
flow's x̂₁ prediction, and for training it with a Gaussian NLL, β-NLL or a decoupled ("faithful")
loss. Loss weighting over t is in [../diffusion-flow-matching/000-takeaways.md](../diffusion-flow-matching/000-takeaways.md) §3.

The setting: x-prediction with self-conditioning, so that x̂₁ = anchor + σ⊙out, where the anchor is the
previous pass's x̂₁ (0 without self-conditioning). σ is one value per source slot × coordinate,
from the head. It does two jobs: it sets the output's resolution and it weights the loss.
Posterior widths, and hence σ, span 10⁴ or more across SNR. The time weighting is the learned
λ table (16 bins × 6 groups, a Kendall log-variance per bin), to be folded into the same NLL.

## How much to trust each source

The tiers are those of [the diffusion folder](../diffusion-flow-matching/000-takeaways.md#how-much-to-trust-each-source):

- **A, established:** ≥100 citations, or peer-reviewed with ≥30 citations/yr.
- **B, credible:** peer-reviewed, or a preprint with ≥20 citations.
- **C, unvetted:** a hypothesis.

Counts were taken on 2026-10-25. Semantic Scholar rate-limited most lookups, so several counts come
from OpenAlex's arXiv record and are lower bounds. The venue sets the tier where the count is
clearly short.

Scope is papers from 2020 onward. Four papers on this topic are already reviewed in the diffusion
folder and are linked, not duplicated:
- Nichol & Dhariwal 2021, learned Σ with the hybrid loss ([dfm/039](../diffusion-flow-matching/039-nichol21-improved-denoising-diffusion-probabilistic.md) (A))
- Kingma & Gao 2023, weightings over t ([dfm/022](../diffusion-flow-matching/022-kingma23-understanding-diffusion-objectives-as.md) (A))
- Karras+24 EDM2, learned u(σ) per noise level ([dfm/006](../diffusion-flow-matching/006-karras24-analyzing-improving-training-dynamics.md) (A))
- Chen+25 GMFlow, a predicted distribution over the velocity ([dfm/093](../diffusion-flow-matching/093-chen25-gaussian-mixture-flow-matching.md) (B))

Papers in this folder:
- β-NLL, [001](001-seitzer22-pitfalls-heteroscedastic-beta-nll.md) (A)
- Faithful, [002](002-stirn23-faithful-heteroscedastic-regression.md) (B)
- natural parameters, [003](003-immer23-bayesian-heteroscedastic-natural-parameters.md) (B)
- warm-up, [004](004-sluijterman23-optimal-training-mean-variance.md) (B)
- evidential regression, [005](005-amini20-deep-evidential-regression.md) (A) and its critique [006](006-meinert23-unreasonable-effectiveness-evidential-regression.md) (B)
- pathologies, [007](007-wongtoi24-pathologies-heteroskedastic-regression.md) (C)
- diffusion covariance with an imperfect mean, [008](008-bao22-optimal-covariance-imperfect-mean.md) (B)
- full-covariance NLL, [009](009-shukla24-tic-tac-covariance-heteroscedastic.md) (B)

None of them trains a flow with x-prediction and a learned σ, or covers a 10⁴ range of scales.

## 1. The pathology is real, and it is about the mean

- **Plain NLL fits the mean worse than MSE.** The mean's gradient is (μ − y)/σ². Badly fit points
  raise their σ, which lowers their gradient, so they stall.
  - [001](001-seitzer22-pitfalls-heteroscedastic-beta-nll.md) (A): on a sinusoid, no NLL run gets
    below RMSE 0.10 against an optimum of 0.01. Across UCI, NLL ties the best RMSE on 5 of 12
    datasets, against 12 for MSE.
  - [002](002-stirn23-faithful-heteroscedastic-regression.md) (B): naval RMSE is 0.53 with NLL,
    against 0.025 with faithful.
- **The loss's minimiser is unchanged.** With unlimited capacity, every weighting still gives
  E[x₁ | x_t, y]
  ([dfm/022](../diffusion-flow-matching/022-kingma23-understanding-diffusion-objectives-as.md) (A)).
  In practice, though, the weighting decides which points the network fits well and which it
  leaves stalled.

## 2. The fixes, by evidence

- **A mean-only warm-up removes most of it**
  ([004](004-sluijterman23-optimal-training-mean-variance.md) (B)).
  - On yacht, RMSE drops from 8.87 without warm-up to 1.00 with it.
  - After the warm-up, it barely matters whether the mean is then frozen or trained jointly.
  - Regularising the variance about 10× more than the mean also works without a warm-up.
  - For us, `init_from` an x-prediction run is this warm-up.
- **Decouple the mean from σ.**
  - [002](002-stirn23-faithful-heteroscedastic-regression.md) (B): the mean and trunk get the MSE
    gradient, and σ is trained on a stop-gradiented mean. The mean fit then equals the mean-only
    model's by construction. Win/tie counts on UCI (RMSE/ECE/LL) are 11/8/11, against 7/2/3 for
    β-NLL at 0.5.
  - [dfm/039](../diffusion-flow-matching/039-nichol21-improved-denoising-diffusion-probabilistic.md) (A):
    the same idea at scale in diffusion. L_simple trains the mean, and the stop-gradiented L_vlb
    trains Σ.
  - [008](008-bao22-optimal-covariance-imperfect-mean.md) (B): learns a diagonal variance per data
    point on a frozen, pretrained mean. The variance regresses the squared residual, which corrects
    for the mean's own error. FID at 10 steps improves from 34.3 to 24.1, for under 10% extra
    compute.
  - Against: [003](003-immer23-bayesian-heteroscedastic-natural-parameters.md) (B) finds that
    faithful's shared trunk fails when the variance depends on inputs the mean ignores. On
    FashionMNIST its RMSE is 51.8, against 18–24 for the others.
- **β-NLL.** [001](001-seitzer22-pitfalls-heteroscedastic-beta-nll.md) (A) multiplies the NLL by
  sg(σ^{2β}), and β = 0.5 is its recommended default.
  - It recovers most of the mean fit: energy RMSE is 1.12 at β = 0.5, 2.25 for NLL, 0.92 for MSE.
    It costs likelihood (NYU depth LL −7.50 at β = 0.5, against −4.52 for NLL).
  - It is the most widely used fix, but the evidence against it is consistent:
    - [007](007-wongtoi24-pathologies-heteroskedastic-regression.md) (C) finds β = 0.5 leaves the
      mean far off on 3 of 4 small UCI sets.
    - [009](009-shukla24-tic-tac-covariance-heteroscedastic.md) (B) notes it is not a valid
      likelihood, and measures pose-estimation likelihood of −4.2e3, against −92 for NLL.
- **Natural parameters**
  ([003](003-immer23-bayesian-heteroscedastic-natural-parameters.md) (B)). The head outputs
  (μ/σ², −1/(2σ²)), so the gradients are plain residuals and no stop-gradient or β is needed.
  - Much of the gain comes from its empirical-Bayes Laplace regularisation, so it is not a
    drop-in loss.
  - On UCI the gain is mostly empirical Bayes plus the posterior predictive: naive NLL with both
    scores best (5 wins, 6 ties of 8), natural NLL second (2, 5). Natural parameters mainly add
    stability, since naive NLL with EB but no posterior predictive collapses (boston LL −6.03).
- **Not for us.**
  - Evidential regression ([005](005-amini20-deep-evidential-regression.md) (A)): its NIG loss
    does not identify its parameters, and its uncertainties are heuristic
    ([006](006-meinert23-unreasonable-effectiveness-evidential-regression.md) (B)).
  - Full covariance ([009](009-shukla24-tic-tac-covariance-heteroscedastic.md) (B)): its
    covariance is built from the input Jacobian and Hessian, so the cost grows with output
    dimension, and a diagonal suffices for an x̂₁ head.

## 3. What changes under x̂₁ = anchor + σ⊙out

All the papers analyse β for a head that outputs μ directly. With our parametrisation the error
in units of σ is ẽ = (x̂₁ − x₁)/σ = out − (x₁ − anchor)/σ, which is O(1) for every source. The
per-sample gradient that reaches `out` is then:

| loss | gradient on out |
|---|---|
| NLL (β = 0) | 2ẽ |
| β-NLL | 2ẽ·σ^{2β} |
| MSE on x̂₁ (today's loss up to λ) | 2ẽ·σ² |

- **β = 0 is already the Fisher-like whitening we wanted.** Under this parametrisation, plain NLL
  gives a loud source and a quiet one the same gradient.
- **β = 0.5 tilts it back toward wide posteriors by σ.** Across our 10⁴ spread of widths, the
  loud sources get about 10⁴× less gradient than the quiet ones. That works against the goal of
  making loud f0 pay to get sharper.
- **Why the literature reaches for β > 0 doesn't apply here.** The stalled mean that motivates
  β > 0 is a cold-start problem, and our `init_from` warm start is the cure that 004 measured.
  The remaining risk is σ inflating to absorb errors. A clamp on log σ² and the faithful split
  (σ on a stop-gradiented error) both guard against that.
- **These are open questions, not results from any paper here.** Which β to use under the σ⊙out
  head, and whether to fold λ in as a per-bin log-variance (EDM2 style,
  [dfm/006](../diffusion-flow-matching/006-karras24-analyzing-improving-training-dynamics.md) (A)),
  are both untested.

## 4. Recipe implied

1. Warm start from an x-prediction run (`init_from`), with log σ² initialised to zero and
   clamped ([004](004-sluijterman23-optimal-training-mean-variance.md) (B)).
2. Use one NLL in σ-units, ẽ² + log σ², with the λ table as an extra per-bin log-variance.
   Start at β = 0, since the σ⊙out head already whitens (§3), not at the β = 0.5 default
   carried over from the μ-head literature.
3. If σ inflates or the mean regresses, switch to the faithful split: `out` trained on ẽ
   with σ held under a stop-gradient, and σ trained on a stop-gradiented error
   ([002](002-stirn23-faithful-heteroscedastic-regression.md) (B),
   [dfm/039](../diffusion-flow-matching/039-nichol21-improved-denoising-diffusion-probabilistic.md) (A),
   [008](008-bao22-optimal-covariance-imperfect-mean.md) (B)).
4. Check with the per-SNR width ratio to MCMC and in68, not the loss. 001 and 009 both show that
   likelihood and mean fit can move in opposite directions.
