# Anytime Training with Schedule-Free Spectral Optimization

Apte et al., 2026, arXiv preprint, JPMorganChase · [arXiv:2605.23061](https://arxiv.org/abs/2605.23061)

**Evidence: C.** Not peer-reviewed; 4 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

SF-NorMuon (Alg. 1):
- Y = (1-β1)Z + β1X; G = ∇L(Y); M = μM + (1-μ)G.
- P = polar(M) by Newton-Schulz: 5 steps, (3.4445, -4.7750, 2.0315), bf16.
- v = β2 v + (1-β2)·mean_cols(P⊙P), a per-row second moment; P̂ = P/(sqrt(v)+ε).
- η̂_t = 0.2·η_t·sqrt(mn)/‖P̂‖_F (RMS matching).
- Z ← Z - η_tλZ - η̂_t P̂.
- c_{t+1} = η_t²/Σ η_i²; X ← (1-c)X + cZ.
- **Inconsistencies in the paper:**
  - the text says sqrt(max(m, n)), but the algorithm and code use sqrt(mn);
  - the algorithm writes ηλ, but the code uses the warmed-up lr: `z.mul_(1-lr*decay)`, applied before the update.
- **Theory:**
  - Õ(T^{-1/4}) stationarity, Õ(T^{-1/2}) deterministic.
  - Lemma 3.1: with decay at z, ‖Z‖_F ≤ ‖Z0‖(1-ηλ)^t + κ sqrt(mn)/λ, so all iterates are bounded.
  - Remark 2: with decay at y, the leading eigenvalue 1 + (ηλ/2)(sqrt(1+6β+β²) - (1-β)) > 1, so the bound diverges.
  - Lemma 3.2: steady state λρ = (κ/2)[-α + sqrt(α² + 2ηλ)], with ρ = ‖Z‖_F/sqrt(mn) and α the cosine between P̂ and Z, small and negative. It matches the measured runs (Fig 4).

## What they find

- **Fig 2, decay placement, 125M out to 48× Chinchilla:**
  - SF-AdamW: no decay diverges; decay at y is best to ~30B tokens and then diverges; decay at z is stable but worse early.
  - SF-NorMuon: no decay diverges *faster*; decay at y is unstable; **decay at z is stable and best throughout**.
- **Table 1, val loss at X, 125M, 1×/2×/4×/8× Chinchilla** (SF-NorMuon vs SF-AdamW vs AdamW-cosine vs NorMuon-cosine):
  - 3.318/3.229/3.170/3.136
  - vs 3.399/3.282/3.205/3.141
  - vs 3.354/3.231/3.170/3.124
  - vs 3.292/3.200/3.139/3.103
  - 35–36% fewer steps than SF-AdamW to reach SF-AdamW's final loss (11% at 8×).
- **772M (1×/2×/4×):** 2.777/2.729/2.698 vs SF-AdamW 2.852/2.780/2.726, i.e. 47–52% fewer steps. It stays 0.025–0.036 nats behind horizon-tuned cosine NorMuon.
- **Ablations at 8×:**
  - **μ = 0 (no Muon momentum): 3.14 → 3.28**;
  - no row normalisation: 3.136 → 3.142 (~12% slower).
- **Other baselines (1×):**
  - WSD NorMuon 3.310–3.317 (lr 4e-3/8e-3);
  - constant-lr NorMuon + EMA (ω 0.99/0.998): best 3.346, behind SF-NorMuon's 3.318.
- Seed std ≤ 0.006; overhead ~2% vs AdamW.
- **ViT-B/16 ImageNet, 30k steps, top-1 at 10k/20k/30k:** SF-NorMuon 72.22/73.46/73.84, SF-AdamW 64.05/70.87/73.44, AdamW 68.75/71.89/73.32. Decay at z is again best for NorMuon, decay at y for AdamW.

## Implementation details

- **125M settings:**
  - SF-NorMuon: η 0.01, λ 0.05 at z, β1 0.9, β2 0.95, μ 0.8, 2000 warmup;
  - 1-D parameters (embeddings, norms): SF-AdamW (0.95, 0.99) with decay at y;
  - clip 1.0, batch 512×1024 tokens.
- **Grid:** μ ∈ {0.8, 0.9, 0.95, 0.99}, λ ∈ {0, 0.05, 0.1}, location ∈ {y, z}.
- **ViT:** η 0.015, μ 0.9, λ 0.25 at z, warmup 4000.
- **Mechanism:** z runs at constant lr forever, its own validation loss "often diverges", and x inherits the divergence.
- **Steady state with α ≈ 0** (my rearrangement of Lemma 3.2): ρ ≈ κ·sqrt(η/(2λ)).
  - Their 125M run: 0.2·sqrt(0.01/0.1) ≈ 0.063.
  - At η 1e-3, λ 0.05, κ = 0.2: ≈ 0.02.

## Caveats

- LM and ViT classification only, one batch size, β fixed (no anneal).
- No diffusion or flow-matching experiments.
- The decay-at-y failure for NorMuon is shown at long horizons. At short horizons the difference is smaller.
