"""The learned loss weights λ [time bin, coordinate group] of a run: over training, at init, and what they weight at the end."""

import dataclasses
import json
import math
import sys
from pathlib import Path

import jax
import numpy as np

from canna.lisa import config as run_config
from canna.lisa.network import LOSS_GROUPS, LOSS_TIME_BINS, LOSS_TIME_DECADES
from canna.lisa.sample import ODE_STEPS
from canna.lisa.train import load_trained, setup

REPORT = Path(__file__).parents[1]


def bin_density(cfg: run_config.Config, edges: np.ndarray) -> np.ndarray:
    """Share of training draws per log10(1 - s) bin: uniform t mixed with log-uniform 1 - s on [floor, 1]."""
    lo, hi = edges[:-1], edges[1:]
    uniform = 10.0 ** (hi / cfg.time_power) - 10.0 ** (lo / cfg.time_power)
    # the code clips every 1 - s below the first edge into the first bin
    uniform[0] += 10.0 ** (edges[0] / cfg.time_power)
    floor = math.log10(cfg.late_time_floor)
    late = np.clip(np.minimum(hi, 0) - np.maximum(lo, floor), 0, None) / -floor
    return (1 - cfg.late_time_mix) * uniform + cfg.late_time_mix * late


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT / "data" / "run.json"
    cfg = run_config.load_run()
    history = np.load(cfg.run_dir / "history.npz")
    edges = np.asarray(history["time_edges"], np.float64)
    assert edges.size == LOSS_TIME_BINS + 1 and edges[0] == -LOSS_TIME_DECADES

    # λ at init (the run's own key), and at the last checkpoint for the averaged and training weights
    init = np.asarray(setup(cfg)[2].flow.loss_log_var, np.float64)
    _, averaged, epoch = load_trained(dataclasses.replace(cfg, weights="average"))
    _, training, _ = load_trained(dataclasses.replace(cfg, weights="training"))
    final_x = np.asarray(averaged.loss_log_var, np.float64)
    final_y = np.asarray(training.loss_log_var, np.float64)

    sizes = np.array([g.stop - g.start for g in LOSS_GROUPS])
    elements = int(sizes.sum())
    density = bin_density(cfg, edges)
    log_var = np.asarray(history["log_var"], np.float64)
    bin_error = np.asarray(history["bin_error"], np.float64)
    steps = np.asarray(history["steps"], np.int64)
    gap = np.abs(log_var[epoch - 1] - final_y).max()
    print(f"[loss-weights] history λ at epoch {epoch} vs the checkpoint's training weights: max |Δ| {gap:.2g}")
    assert np.isfinite(bin_error).all(), "an empty bin in the diagnostic batch"

    # per-element weight on the squared velocity error, relative to the plain flow loss's mean over all coordinates
    weight = lambda lam: np.exp(-lam) * elements / (len(sizes) * sizes)
    # share of the objective's error term per (bin, group) under the learned weights, and under the plain loss
    err = bin_error[epoch - 1]
    learned_share = density[:, None] * err * np.exp(-final_y)
    plain_share = density[:, None] * err * sizes / elements

    # AdamW moves a parameter by at most about its lr per step: the warmup ramp, then the peak
    reach = np.where(
        steps < cfg.warmup_steps,
        cfg.adam_learning_rate * steps**2 / (2 * cfg.warmup_steps),
        cfg.adam_learning_rate * (steps - cfg.warmup_steps / 2),
    )

    # the share of training draws later than the sampler's last query: midpoint x-steps at t = i/N and (i + 1/2)/N
    last_query = 1 - (2 * cfg.ode_steps - 1) / (2 * cfg.ode_steps)
    last_decade = cfg.time_power * math.log10(last_query)
    floor = math.log10(cfg.late_time_floor)
    unqueried = (1 - cfg.late_time_mix) * last_query + cfg.late_time_mix * max(
        0.0, (last_decade - floor) / -floor
    )

    record = dict(
        run=cfg.run,
        config=cfg.name,
        commit=cfg.commit,
        epoch=int(epoch),
        device=jax.devices()[0].device_kind,
        total_steps=cfg.total_steps,
        warmup_steps=cfg.warmup_steps,
        adam_learning_rate=cfg.adam_learning_rate,
        late_time_mix=cfg.late_time_mix,
        late_time_floor=cfg.late_time_floor,
        time_power=cfg.time_power,
        ode_steps=cfg.ode_steps,
        sampler_default_steps=ODE_STEPS,
        groups=[str(g) for g in history["groups"]],
        group_sizes=sizes.tolist(),
        time_edges=edges.tolist(),
        density=density.tolist(),
        sampler_min_decade=last_decade,
        unqueried_share=unqueried,
        steps=steps.tolist(),
        log_var=log_var.round(5).tolist(),
        bin_error=np.log(bin_error).round(5).tolist(),
        init=init.round(5).tolist(),
        final_average=final_x.round(5).tolist(),
        final_training=final_y.round(5).tolist(),
        adam_reach=reach.round(5).tolist(),
        moved=np.abs(log_var - init).max(axis=(1, 2)).round(5).tolist(),
        weight_init=weight(init).tolist(),
        weight_final=weight(final_x).tolist(),
        learned_share=(learned_share / np.nansum(learned_share)).tolist(),
        plain_share=(plain_share / np.nansum(plain_share)).tolist(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, allow_nan=False))
    print(
        f"[loss-weights] {cfg.run} epoch {epoch}: max |λ - init| {record['moved'][-1]:.3g}, AdamW reach {reach[-1]:.3g}"
    )
    print(
        f"[loss-weights] {unqueried:.1%} of training draws sit later than the {cfg.ode_steps}-step sampler's last query (log10(1-s) = {last_decade:.2f})"
    )
    print(f"[loss-weights] wrote {out}")


if __name__ == "__main__":
    main()
