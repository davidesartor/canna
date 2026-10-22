# ScheduleFree+ (Defazio, 2026)

[arXiv:2605.19095](https://arxiv.org/abs/2605.19095), FAIR tech report. Schedule-free fixes
for LLMs at scale, plus a learning-rate-free step.

**Evidence: C.** Not peer-reviewed; 7 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Llama-style LLMs from 120M to 2B parameters, AdamW-based only. Five changes to schedule-free:

1. **Inner momentum back in** (Adam β₁ 0.75–0.9). Plain SF hits a batch-size "cliff" at 2M
   tokens per batch, where AdamW's is at 4M. Inner momentum moves it to AdamW's.
2. **Inverse gradient-norm weighting** (Eq. 10). Steps are scaled by 1/EMA‖g‖₁ (EMA 0.9),
   which counters the gradient-norm drift that weight decay causes under SF. It is separate
   from the lr-free step: they also run it with a grid-searched lr ("ScheduleFree 1/L1",
   Fig. 8).
3. **A Polyak step** (Eq. 12, Algorithm 1). The step size is
   γ = (f(y) − f\* + β⟨∇f(y), z − x⟩) / (√(π/2)·‖∇f(y)‖₁), with EMAs of the noisy
   estimates. The reference code takes f\* = 0 (`max(0, f + ip_term) / grad_l1_ema`). That
   gives a learning-rate-free optimizer.
4. **AdamC decay, "fully decoupled"** (Eq. 14): z ← z − γ·update − γ²λz, with λ 5–50.
   They call it "necessary when using the Polyak step size". Their argument: with an
   adaptive γ, decay proportional to γ couples the lr to the gradient norm, and that feedback
   "WILL break" lr adaptation. With γ², the steady-state gradient/weight ratio does not
   depend on γ.
5. **Averaging tweaks:**
   - *Warm start:* no averaging (c = 1) for the first steps. Their heuristic is 2× the lr
     warmup.
   - *β annealed* log-linearly from 0.9 (or 0.8) to 0.965 over the run.
   - *Averaging weight* t^r·max γ², with r = 1. The default, uniform weighting after the
     warmup, "underweights later iterates", so x lags behind a late drop in the loss.
   - Evaluate at x, the average, not at y.

## What they find

- At 1000 tokens per parameter, 31% less compute than a tuned linear-decay schedule to reach
  the same loss. No final-loss advantage at 20–100 tokens per parameter.
- β: small values converge faster early, and 0.965 is better late. The ramp gets both.
- r = 1 is slightly worse below about 30B tokens and clearly better after. r > 1 gives no
  further gain.
- Warm start: a longer c-warmup leaves a "hump" when averaging begins, but a lower final loss.
- Larger "normal" weight decay hurts SF-AdamW unless it is AdamC-style.
- Polyak beats a tuned grid of fixed step sizes, mostly through a larger early lr.
- WSD loses to linear decay when both are tuned. Cosine down to 10% is suboptimal.
- SF loss curves are very predictable from a convex bound.

## Limits

- LLMs only, AdamW only. No Muon, vision or diffusion experiments.
- The Polyak step assumes the loss's floor is known; the code sets it to 0.
- A tech report: few seeds and no error bars.
