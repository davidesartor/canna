# F36 (2026-10-13): why schedule-free failed on XS-fd — its fast sequence z ran away, and only the training point y stayed good

- z drifted to ~2× x's size and out of the basin; only a thin neighbourhood of y is a good
  network.
- Cause: Muon's fixed-size steps, amplified ~10× in z by the interpolation, with no restoring
  decay.

Diagnosis of XS-fd's final state (`lisa_checks/sf_anatomy.py at 05393cd`,
`lisa_checks/sf_line.py at 05393cd`, on 7adfd3d; 8 × 256 common samples).

**The three weight sets.**
- Over the backbone, |z|/|x| = 1.92, cos(x, z) = 0.80. MLP input weights: x RMS 0.46–0.49, z
  0.88–1.01, initial 0.0255.
- If |z| grows ∝ t, the uniform average sits at half its final norm: x is mostly a radially
  shrunk z, not a denoised one.
- z moved ~1.0 RMS per element in 1.2M steps; a random walk of Muon's 2×10⁻⁴ steps would move
  0.22, so z drifts persistently.

**Loss along the line from x to z:**

| point | flow loss |
|---|---|
| x (α = 0) | 0.467 |
| α = 0.05 | 0.459 |
| **y (α = 0.111)** | **0.4575** |
| α = 0.2 | 0.574 |
| α = 0.3 | 85 |
| α = 0.5 | 1.9×10⁹ |
| z (α = 1) | 2.7×10²¹ |

- x sits on the steep side of y's neighbourhood.
- Not a per-layer scale lag: x rescaled per layer to y's norms scores 0.519, y rescaled to x's
  0.522.

**Mechanism.**
1. The gradient is taken at y = 0.9x + 0.1z and the base optimizer steps z. Late on x hardly
   moves (c_t ≈ 10⁻⁶), so for y to follow the gradient z must travel ~10× as far.
2. Muon's step has fixed size (RMS 0.2·lr per element): Newton–Schulz normalises the magnitude
   away, so z never slows near convergence, unlike SGD.
3. With no restoring force (decay 1e-9 per step, F34), z drifts without bound and leaves the
   basin.
4. Schedule-free assumes x–z lies in one basin, as in a convex problem. Here it crosses a
   cliff, so x cannot beat y: x trails y by 0.012 (F33), with F10's late drop leaving x on
   pre-drop iterates on top.

**Consequences.**
- Weight decay ~0.05 is necessary (F35: ~40% slower growth) but may not be sufficient: in F35 z
  drifted from x at the same rate either way.
- The interpolation multiplies unbounded steps by 10 in z. Whether dropping Muon's momentum
  helps (F35's C) is untested.
- Robust option: keep Muon and evaluate on y's trajectory (an EMA of y, or a short cooldown of
  y), which never leaves the region where gradients are taken.
