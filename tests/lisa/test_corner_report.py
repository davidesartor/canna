"""The corner report: the ecliptic sky, and a synthetic eval round-tripped through the page's float16 draws."""

import base64
import json
import re

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from canna.lisa import config
from canna.lisa.chart import physical_to_flow
from canna.lisa.plots import OBLIQUITY, corner_report, equatorial_to_ecliptic
from canna.lisa.problem import LisaGB, sample_f
from canna.lisa.simulator import sample_physical


def test_the_ecliptic_is_the_equator_tilted_by_the_obliquity():
    lam, beta = equatorial_to_ecliptic(np.array([0.0, 1.5 * np.pi]), np.array([0.0, np.pi / 2 - OBLIQUITY]))
    assert np.allclose([lam[0], beta[0]], 0.0, atol=1e-12)  # the vernal equinox
    assert np.isclose(beta[1], np.pi / 2)  # the ecliptic north pole


def test_a_synthetic_eval_round_trips_through_the_page(tmp_path):
    name = "narrow-B"
    problem_args = config.load(name).problem
    problem = LisaGB(**problem_args)
    run_dir, eval_dir = tmp_path / "run", tmp_path / "run" / "eval" / "1-average-32"
    eval_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"name": name, "commit": "abc1234", "seed": 0, "problem": problem_args}))
    (eval_dir / "eval.yaml").write_text(yaml.safe_dump({"ode_steps": 32, "weights": "average"}))

    n_injections, n_draws, rng = 2, 64, np.random.default_rng(0)
    f = np.array([float(sample_f(problem, jr.key(j))) for j in range(n_injections)])
    truth = np.stack([np.asarray(sample_physical(problem, jr.key(10 + j), f[j])) for j in range(n_injections)])
    # draws a hair off the truth in every coordinate, f0 by ~0.01 bins, fdot and A by 1e-3 of themselves
    spread = np.array([0.01 / problem.t_obs, 0.0, 0.0, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
    spread = spread + np.array([0, 1e-3, 1e-3, 0, 0, 0, 0, 0]) * np.abs(truth[:, None])
    draws = truth[:, None] + spread * rng.standard_normal((n_injections, n_draws, *truth.shape[1:]))
    np.savez(
        eval_dir / "draws.npz",
        draws=draws,
        matched=draws,
        fisher=draws[:, ::-1],
        truth=truth,
        f=f,
        snr=np.full(n_injections, 50.0),
        source_snr=np.tile(np.linspace(30.0, 10.0, problem.n_sources), (n_injections, 1)),
        grid=np.stack(np.divmod(np.arange(n_injections), n_injections), axis=1),
        epoch=1,
    )

    page = corner_report(eval_dir).read_text()
    index = json.loads(re.search(r"const DATA = (.*?);</script>", page).group(1))
    raw = np.frombuffer(base64.b64decode(re.search(r'id="blob">(.*?)</script>', page).group(1)), "<f2")
    assert len(index["injections"]) == n_injections and "mcmc" not in index["meta"]["draws"]
    source = index["injections"][1]["sources"][2]
    flow_truth = np.asarray(physical_to_flow(problem, jnp.asarray(truth[1]), f[1]))[2]
    assert np.allclose(source["truth"]["flow"], flow_truth, rtol=0, atol=1e-12)
    for view in ("physical", "flow"):
        offset, n = source["sets"]["flow"][view]
        back = np.asarray(source["truth"][view]) + raw[offset : offset + n * 8].reshape(n, 8) * source["scale"][view]
        if view == "flow":
            exact = np.asarray(physical_to_flow(problem, jnp.asarray(draws[1]), f[1]))[:, 2]
        else:
            exact = draws[1][:, 2, 0] * problem.t_obs - index["injections"][1]["start_bin"]
            back = back[:, 0]
        # float16 keeps 1e-3 of a flow width; the draws are all within a few widths of the truth
        width = np.asarray(source["scale"][view]) if view == "flow" else source["scale"][view][0]
        assert np.all(np.abs(back - exact) <= 5e-3 * width)
