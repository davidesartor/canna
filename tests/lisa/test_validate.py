"""The validation pass: ranks, pre-ranks, γ and scores tell a calibrated toy posterior from a prior and a too-wide one."""

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from canna.lisa import config
from canna.lisa.problem import LisaGB, sample_f
from canna.lisa.simulator import clean_signal, sample_observation, sample_physical
from canna.lisa.validate import (
    analyse,
    binomial_cdf,
    energy_score,
    gamma_test,
    quantities,
    report,
)

N, M, S = 600, 63, 2
NOISE = 0.5
ALPHA_FLOOR = 1e-3


def toy_samples(kind: str, seed: int = 0) -> dict:
    """theta ~ N(0, 1), y = theta + N(0, NOISE^2) per coordinate, draws from the exact posterior, the prior, or 2x too wide."""
    rng = np.random.default_rng(seed)
    truth = rng.standard_normal((N, 1, S, 8))
    y = truth + NOISE * rng.standard_normal(truth.shape)
    mean, std = y / (1 + NOISE**2), np.sqrt(NOISE**2 / (1 + NOISE**2))
    noise = rng.standard_normal((N, M, S, 8))
    draws = dict(exact=mean + std * noise, prior=noise, wide=mean + 2 * std * noise)[
        kind
    ]
    x = np.concat([truth, draws], axis=1)
    return dict(
        x=x,
        f0_bins=x[..., 0],
        # a data-dependent quantity: closeness of each parameter set to the data
        overlap=-np.linalg.norm(x - y, axis=-1),
        log_likelihood=-np.sum((x - y) ** 2, axis=(-2, -1)),
        source_snr=np.full((N, M + 1, S), 20.0),
        snr=np.full((N, M + 1), 30.0),
        tarp_ref=rng.standard_normal((N, S, 8)),
    )


def p_values(rows: dict) -> dict:
    k = rows["draws"] + 1
    ps = {
        f"{q} {s}": gamma_test(r[:, s], k)[0]
        for q, r in rows["rank"].items()
        for s in range(S)
    }
    return ps | {
        f"joint {q}": gamma_test(r, k)[0] for q, r in rows["joint_rank"].items()
    }


def test_binomial_cdf_matches_the_exact_sum():
    n, p = 30, np.array([0.1, 0.5, 0.83])
    exact = [
        [
            sum(math.comb(n, j) * q**j * (1 - q) ** (n - j) for j in range(k + 1))
            for k in range(n + 1)
        ]
        for q in p
    ]
    assert np.allclose(binomial_cdf(n, p), exact, atol=1e-12)


def test_uniform_ranks_sit_inside_their_band():
    k = M + 1
    ranks = np.random.default_rng(3).integers(0, k, 800)
    p, (lo, hi) = gamma_test(ranks, k)
    ecdf = (ranks[:, None] < np.arange(1, k)).mean(0) - np.arange(1, k) / k
    assert p > ALPHA_FLOOR and np.all(lo <= ecdf) and np.all(ecdf <= hi)
    assert np.all(lo < 0) and np.all(hi > 0)


def test_the_exact_posterior_passes_everything():
    rows = analyse(toy_samples("exact"))
    assert min(p_values(rows).values()) > ALPHA_FLOOR


def test_the_prior_passes_the_marginals_and_fails_the_data_dependent_quantities():
    rows = analyse(toy_samples("prior"))
    ps = p_values(rows)
    assert all(ps[f"f0 {s}"] > ALPHA_FLOOR for s in range(S))
    assert all(ps[f"overlap {s}"] < ALPHA_FLOOR for s in range(S))
    assert ps["joint log_likelihood"] < ALPHA_FLOOR
    # the truth is closer to its data than any prior draw: its rank piles up at the top
    assert rows["rank"]["overlap"].mean() / M > 0.9


def test_a_too_wide_posterior_ranks_central_and_scores_worse():
    wide, exact = analyse(toy_samples("wide")), analyse(toy_samples("exact"))
    middle = lambda r: np.mean((r >= (M + 1) / 4) & (r < 3 * (M + 1) / 4))
    assert middle(wide["rank"]["f0"]) > 0.6
    assert wide["rank"]["band_depth"].mean() / M > 0.6
    assert min(p_values(wide).values()) < ALPHA_FLOOR
    assert wide["energy"].mean() > exact["energy"].mean()
    assert np.all(wide["crps"].mean(axis=(0, 1)) > exact["crps"].mean(axis=(0, 1)))


def test_energy_score_in_one_dimension_is_the_gaussian_crps():
    rng = np.random.default_rng(1)
    x = np.concat([[0.0], rng.standard_normal(4000)])[None, :, None]
    crps = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
    assert abs(energy_score(x)[0] - crps) < 0.01


def test_the_report_prints_every_stratum(capsys):
    report(analyse(toy_samples("exact")), "toy")
    out = capsys.readouterr().out
    assert "15-40" in out and "all" in out and "band_depth" in out


@pytest.fixture(scope="module")
def narrow():
    problem = LisaGB(**config.load("narrow-B").problem)
    f = sample_f(problem, jr.key(0))
    truth = sample_physical(problem, jr.key(1), f)
    return problem, f, truth, sample_observation(problem, jr.key(2), truth, f)


def test_quantities_do_not_depend_on_the_input_source_order(narrow):
    problem, f, truth, o = narrow
    theta = jnp.stack([truth, truth[::-1], truth[jnp.array([2, 0, 3, 1])]])
    q = quantities(problem, theta, o, f)
    for v in q.values():
        assert np.allclose(v[0], v[1:], rtol=1e-10, atol=1e-10)
    assert np.all(np.diff(np.asarray(q["source_snr"]), axis=-1) <= 0)


def test_the_overlaps_add_up_to_the_snr_on_noiseless_data(narrow):
    problem, f, truth, _ = narrow
    q = quantities(problem, truth[None], clean_signal(problem, truth, f), f)
    # <h|h_s> / |h_s| summed with weights |h_s| is <h|h> = snr^2
    total = np.sum(np.asarray(q["overlap"] * q["source_snr"]))
    assert np.isclose(total, float(q["snr"][0]) ** 2, rtol=1e-8)


def test_a_whole_validation_runs_end_to_end(small_flow, tmp_path, capsys):
    """The eval job's path on the tiny test network: draws, then ranks, report, scores and pages."""
    from canna.lisa.plots import eval_pages
    from canna.lisa.samples import pack_validation
    from canna.lisa.validate import summarise, validate_injection

    problem, flow, _, _, _ = small_flow
    rows = [
        validate_injection(problem, flow, jr.key(i), 1.0, n_draws=8, ode_steps=2)
        for i in range(12)
    ]
    samples = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}
    assert samples["x"].shape == (12, 9, problem.n_sources, 8)
    assert all(np.all(np.isfinite(samples[k])) for k in ("x", "source_snr", "overlap"))
    scores = summarise(samples, 0, "test")
    assert {
        "rank_f0",
        "rank_tarp",
        "joint_rank_log_likelihood",
        "joint_rank_tarp",
    } <= set(scores)
    np.savez(tmp_path / "scorecard.npz", epoch=1, ode_steps=2, **pack_validation(samples), **scores)
    assert eval_pages(tmp_path) == [tmp_path / "validation.html"]
    assert (
        "slot 0" in capsys.readouterr().out and (tmp_path / "validation.html").exists()
    )
