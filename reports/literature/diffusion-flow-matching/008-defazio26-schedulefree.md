# ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models

Defazio, 2026, arXiv preprint, FAIR · [arXiv:2605.19095](https://arxiv.org/abs/2605.19095)

**Evidence: C.** Not peer-reviewed; 7 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

SF-AdamW fixes for LLM pretraining at scale (to 500B tokens).
- **(1) Inner momentum:** put Adam's β1 (0.75–0.9) back inside the base step. This removes the large-batch "cliff": without it SF-AdamW breaks at a 2M-token batch, AdamW at 4M. The mechanism is tolerance of a larger LR. It beats the alternative "C-refinement" c = (1-β)C/(t+1) with C = 50.
- **(2) Weight-decay and gradient-norm drift:** small decay (0.002) mostly compensates for growing gradient norms. The fix is to scale the step by 1/‖g‖₁ (an EMA at 0.9) rather than rely on decay. Averaging points on a sphere also shrinks ‖x‖ by about 2×, because the mean of points on a sphere lies inside it.
- **(3) Polyak step for Adam:** γ_t = (f(y) - f* + β⟨∇f(y), z - x⟩)/(sqrt(π/2)·‖∇f(y)‖₁), with 0.9 EMAs. This makes the method LR-free and beats a γ/L1 grid.
- **(4) AdamC decoupled decay:** -γ_t²λ, with λ of 5–50. It is required with Polyak steps.
- **(5) c-warmup:** c_t = 1 (x tracks z) for 2× the LR warmup; 800 steps for 400 warmup steps was best. It leaves a hump in the curve but a lower final loss.
- **(6) β anneal, log-linear in 1-β:** 1-β_t = (1-β_i)^{1-τ}(1-β_f)^τ with τ = t/T_anneal. Text uses 0.9→0.965; the 1000-TPP table uses 0.8→0.965. Low β is better early, high β late.
- **(7) r weighting:** w_t = t^r·max_{i≤t}γ_i² (t^r γ_max^p in the algorithm).

## What they find

- **1000 tokens/param, 120M model:** matches a 45% longer linear-decay run, i.e. about 31% less compute.
- **100 TPP:** beats WSD at all scales, and linear decay except at 2B.
- **20 TPP:** beats WSD, with a small gap to linear decay for larger models.
- **r = 1** is clearly better beyond ~30B tokens (200B-token run, 120M model), slightly worse below that. r > 1 does not help.
- **The β anneal is ineffective at 100 TPP** (those runs use fixed β = 0.9, r = 0).
- **Loss-curve fit:** a/sqrt(t+b) + c predicts the final loss from 10% of the run.
- **Fig 1 config:** 500M model, 500B tokens, 4M batch; decay 20, clip 1, r = 1. Baselines: linear decay lr 3.5e-3, WSD 2.5e-3, wd 0.05.

## Implementation details

- β2 = 0.98 in some runs, clip 1–10.
- The β anneal needs the horizon T, which partly undoes "schedule-free".
- r = 1 favours later iterates (a shorter effective window), which only pays off for long runs.
- With the Polyak step, f* must be estimated (EMAs at 0.9).

## Caveats

- LLM pretraining only, AdamW only (no Muon), a single author.
- Ablations are mostly single-seed curves.
- No diffusion or flow-matching tasks, and no test where the loss floor is noise-dominated, as it is for FM losses.
