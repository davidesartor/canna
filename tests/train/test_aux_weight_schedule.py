"""Cosine anneal of the auxiliary heads: 1 at step 0, 0 after warmup_frac * total_steps."""

import math

from canna.train import aux_weight_schedule


def test_starts_at_one():
    assert aux_weight_schedule(0, 1000, 0.5) == 1.0


def test_reaches_zero_at_warmup_end():
    total_steps, warmup_frac = 1000, 0.5
    warmup_steps = int(total_steps * warmup_frac)
    assert abs(aux_weight_schedule(warmup_steps, total_steps, warmup_frac)) < 1e-6


def test_plateaus_after_warmup():
    total_steps, warmup_frac = 1000, 0.5
    warmup_steps = int(total_steps * warmup_frac)
    assert abs(aux_weight_schedule(warmup_steps + 1, total_steps, warmup_frac)) < 1e-6
    assert abs(aux_weight_schedule(total_steps, total_steps, warmup_frac)) < 1e-6


def test_monotone_non_increasing_within_warmup():
    total_steps, warmup_frac = 1000, 0.5
    warmup_steps = int(total_steps * warmup_frac)
    values = [
        aux_weight_schedule(s, total_steps, warmup_frac)
        for s in range(0, warmup_steps + 1, 10)
    ]
    assert all(a >= b - 1e-6 for a, b in zip(values, values[1:]))


def test_no_rebound_past_warmup():
    """a naive cos(pi * step / warmup_steps) without clamping rebounds to 1 at step=2*warmup_steps"""
    total_steps, warmup_frac = 1000, 0.25
    warmup_steps = int(total_steps * warmup_frac)
    assert abs(aux_weight_schedule(2 * warmup_steps, total_steps, warmup_frac)) < 1e-6


def test_matches_half_cosine_reference_at_warmup_midpoint():
    """pins the exact shape to 0.5*(1+cos(pi*progress)), not just the endpoints"""
    total_steps, warmup_frac = 1000, 1.0
    warmup_steps = total_steps * warmup_frac
    mid = int(warmup_steps / 2)
    expected = 0.5 * (1 + math.cos(math.pi * mid / warmup_steps))
    assert abs(aux_weight_schedule(mid, total_steps, warmup_frac) - expected) < 1e-3


def test_full_warmup_frac_stays_positive_at_midpoint():
    total_steps, warmup_frac = 1000, 1.0
    assert aux_weight_schedule(total_steps // 2, total_steps, warmup_frac) > 0.0


def test_zero_warmup_frac_is_immediately_annealed():
    """warmup_frac=0 means no warmup at all, so the weight is already 0 at step 0"""
    assert abs(aux_weight_schedule(0, 1000, 0.0)) < 1e-6


def test_total_steps_zero_does_not_blow_up():
    """zero-length run must not nan/inf/raise from a warmup_steps=0 division"""
    result = aux_weight_schedule(0, 0, 0.5)
    assert math.isfinite(result)


def test_returns_plain_float():
    """declared return type is float, not a traced jax Array -- callers may use it in python control flow"""
    result = aux_weight_schedule(5, 100, 0.5)
    assert isinstance(result, float)
