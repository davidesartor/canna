# Anytime Training with Schedule-Free Spectral Optimization (Apte et al., 2026)

[arXiv:2605.23061](https://arxiv.org/abs/2605.23061), preprint. SF-NorMuon: schedule-free
wrapped around NorMuon.

**Evidence: C.** Not peer-reviewed; 4 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- **Matrices:** NorMuon inside schedule-free.
  - Explicit momentum M ← μM + (1−μ)G, with μ = 0.8 (a plain EMA).
  - Polar factor (Newton–Schulz), scaled to update RMS 0.2·η (`κ = 0.2`).
  - Per-row second-moment normalisation (NorMuon).
- **Weight decay on z,** not y: z ← z − ηλz − η̂·P.
- **Everything else:** SF-AdamW.
- **Best settings, one configuration for all horizons:**

  | Setting | Value |
  |---|---|
  | lr (both groups) | 0.01 |
  | λ | 0.05, on z |
  | SF β, matrices | 0.9 (constant) |
  | SF β, 1-D params | 0.95, Adam β₂ 0.99 |
  | Warmup | 2000 steps |
  | Averaging weight | lr² from step 1 (r = 0, no warm start) |
  | Clip | 1.0 |

- **Tests:**
  - Llama-style 125M and 772M at 1–8× Chinchilla, and 125M to 48× Chinchilla;
  - ViT-B/16 on ImageNet-1K for 30k steps.

## What they find

- **LLMs:** matches or beats tuned AdamW at every horizon. It trails horizon-aware cosine
  NorMuon by about 0.03 nats.

  | 125M, 1× Chinchilla, 3 seeds | Validation loss |
  |---|---|
  | cosine NorMuon | 3.300 ± 0.006 |
  | SF-NorMuon | 3.318 ± 0.002 |
  | AdamW | 3.350 ± 0.003 |
  | SF-AdamW | 3.400 ± 0.003 |

- **Other anytime methods do worse.**
  - NorMuon-WSD: 3.310–3.317 at 1×, and below SF-NorMuon at 2×.
  - NorMuon at constant lr with a weight EMA: 3.346.
- **Decay must act on z.** In their words, "for SF methods, the validation loss at X
  diverges without weight decaying specifically at Z".
  - With no decay, SF-NorMuon "diverges faster" than SF-AdamW. Muon's fixed-size steps push z
    at a constant rate with no restoring force.
  - Why x follows: z runs at a constant lr and its loss "often diverges"; x averages every
    past z, so z "eventually contaminates" x.
  - The fix is decay inside the base dynamics, not an ℓ2 term in the loss. It bounds every
    iterate, and ‖z‖ converges to a steady state.
  - Decay at y is the best SF-AdamW setting up to about 30B tokens, then it diverges.
  - With decay at y, z and z − x grow together; a coupled-norm argument shows the
    iteration's leading eigenvalue is above 1 for every β.
  - With decay on z, every iterate stays bounded: ‖Z‖ ≤ κ√(mn)/λ.
- **Steady state.** At quasi-steady state, λρ = (κ/2)[−α + √(α² + 2ηλ)], where ρ is RMS(z)
  and α is the alignment of update and weight. It tracks measured RMS(z) at η = 0.01,
  λ = 0.05. For α ≈ 0, ρ ≈ κ√(η/2λ), and the per-step angular change is about √(2ηλ).
- **Ablations,** 125M at 8×:
  - no momentum (μ = 0): 3.14 → 3.28;
  - no row normalisation (SF-Muon): 3.142 against 3.136, and about 12% slower to the same
    loss.
- **ViT-B/16:** 73.84% top-1, against SF-AdamW 73.44% and AdamW 73.32%.
- **Convergence proof:** a stationarity guarantee for schedule-free spectral descent.

## Limits

- LLMs and one ViT. No diffusion or flow matching.
- It still trails a scheduled NorMuon in its own setting. The authors call fully matching it
  open.
