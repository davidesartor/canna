"""Parameter count and checkpoint size across the model-config sweep.

Builds each MMDiT in the sweep and reads back its trainable parameter count and
serialized checkpoint size -- no throughput, so this one is cheap. Shapes depend
only on x_dim / y_channels, so it is problem-agnostic apart from N_SOURCES.
"""

from canna import lisa

from _bench import CONFIGS, build_model, checkpoint_bytes, param_count
from _helpers import problem_name, save_text


def test_model_size():
    header = (
        f"{'config':>8} {'hidden':>7} {'blocks':>7} {'heads':>6} "
        f"{'params':>14} {'ckpt (MB)':>10}"
    )
    lines = [f"{problem_name()}  n_sources={lisa.N_SOURCES}", header]
    for cfg in CONFIGS:
        model = build_model(cfg)
        n_params = param_count(model)
        ckpt_mb = checkpoint_bytes(model) / 1e6
        assert n_params > 0
        lines.append(
            f"{cfg.name:>8} {cfg.hidden_dim:>7} {cfg.num_blocks:>7} "
            f"{cfg.num_heads:>6} {n_params:>14,} {ckpt_mb:>10.1f}"
        )
    save_text("model_size", "\n".join(lines))
