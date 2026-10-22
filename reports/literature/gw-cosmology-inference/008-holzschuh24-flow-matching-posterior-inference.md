# Flow Matching for Posterior Inference with Simulator Feedback

Holzschuh et al., 2024, arXiv preprint / cs.LG · [arXiv:2410.22573](https://arxiv.org/abs/2410.22573)

**Evidence: C.** Not peer-reviewed; 5 citations in 2.0 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Pretrain a conditional FM posterior v_φ(t, θ_t, x_o) and freeze it.
- Compute the 1-step estimate θ̂₁ = θ_t + (1 − t)·v.
- Run the simulator at θ̂₁ to get a control signal c:
  - gradient-based, for a differentiable simulator: c = ∇_θ̂₁ C(θ̂₁, x_o), plus the cost itself;
  - learned, for a non-differentiable simulator: an encoder of (S(θ̂₁), x_o), with gradients stopped at the simulator.
- A small control network v^C(t, v, c) outputs the corrected velocity, trained with the ordinary CFM loss. Because c is a deterministic function of θ_t, the FM theory still holds, unlike DPS-style likelihood guidance.

## What they find

- **SBI benchmark** (10⁵ simulations, about 300K params, C2ST): FM gets LV 0.93, SLCP 0.79, SIR 0.79, TM 0.58, comparable to CNF, NSF and FFJORD.
- **Lotka–Volterra training-set size:** going from 10⁵ to 10⁶ to 10⁷ simulations gives no C2ST gain without feedback. Fine-tuning with gradient feedback (about 9×10⁶ simulator calls) clearly beats all of them.
- **Lotka–Volterra controls:** gradient-based control is better than learned control, which is better than the zero-control ablation, which gives only a slight gain. So the improvement comes from the simulator, not from the extra parameters.
- **Strong lensing:** 23 parameters (17 free), 160×160 pixels, 250k training and 25k validation samples.
  | Method | Avg χ² | Time |
  |---|---|---|
  | NUTS | 1.83 | 564 s |
  | AIES | 1.74 | 672 s |
  | DPS | 9.98 | 427 s |
  | FM | 1.83 | 10 s |
  | FM + simulator feedback | 1.48 | 19 s |
- **SBC:** the plain-FM rank histograms are biased (for example x_center), and feedback improves them.

## Implementation details

- **Fine-tuning setup:**
  - Freeze the pretrained flow. The paper sees no loss from doing so, and the control network then needs no backprop through the big net.
  - The control network is about 10–11% of the parameters: 3×64 plus 3×32 residual blocks, GLU time conditioning, a 16-dimensional time embedding.
  - Fine-tune with batch size 16 for 24 h, after a 45 h pretrain on one A100.
- **Only apply control late:** use it only for t ≥ 0.8 and output the pretrained flow below that. The 1-step θ̂₁ is unreliable early on (App. B, Fig. 8, C2ST of θ̂₁ against t*).
- **Stochastic simulators:** draw a fresh noise seed z per sample during training, and keep z fixed along each trajectory at inference.
- **The χ² cost value is itself an input to the control network**, not only its gradient.
- **Negative findings on the SBI benchmark (Fig. 3):**
  - A prior-as-base distribution (independent coupling) is worse than a Gaussian base on LV and SIR; it only gives better outliers on SLCP.
  - Self-conditioning on θ̂₁ without a simulator gives no gain.
  - x-prediction with 1/(1−t) weighting is consistently worse than v-prediction.
- Euler integration is used at inference, and the step-size sweep is in App. B, Fig. 9.
- Hyperparameters per task follow Wildberger+23 (time-prior exponent α, batch size, learning rate, residual-block widths, in Table 3).

## Caveats

- Inference cost grows with simulator cost, since every ODE step at t ≥ 0.8 calls the simulator.
- With a non-differentiable (learned) control, the density via the change-of-variables formula is lost.
- The prior is baked into the network.
- Lensing is evaluated by average χ² and SBC only, with no C2ST against MCMC.
- The MCMC baselines and FM infer different parameter sets: 17 tied parameters for MCMC against 23 for FM.
- DPS is badly tuned (Table 7).
- Single lensing setup.
