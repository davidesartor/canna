"""Every shipped run config must still splat into its package's problem and network."""

import argparse
import importlib
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
import pytest
import yaml

PACKAGES = ("point", "sinusoid", "lisa")


class PointSample(NamedTuple):
    xt: jnp.ndarray
    dx: jnp.ndarray
    t: jnp.ndarray
    y: jnp.ndarray


def configs():
    for package in PACKAGES:
        root = Path(importlib.import_module(f"canna.{package}").__file__).parent
        for path in sorted((root / "configs").glob("*.yaml")):
            yield package, path


CONFIGS = list(configs())
IDS = [f"{package}-{path.stem}" for package, path in CONFIGS]


def args(path: Path) -> argparse.Namespace:
    with open(path) as f:
        config = yaml.safe_load(f)
    # dtype and muon are argparse-only defaults, never set by a run config
    config.setdefault("dtype", "float32")
    config.setdefault("muon", True)
    return argparse.Namespace(**config)


def test_every_package_ships_at_least_one_config():
    assert {package for package, _ in CONFIGS} == set(PACKAGES)


@pytest.mark.parametrize("package,path", CONFIGS, ids=IDS)
def test_config_holds_only_a_problem_and_a_network_dict_plus_scalars(package, path):
    with open(path) as f:
        config = yaml.safe_load(f)
    assert isinstance(config["problem"], dict)
    assert isinstance(config["network"], dict)
    # flat schema: no class: key, no init_args: nesting, no cross-package composition
    assert not any(key in config for key in ("class", "init_args"))
    scalars = {k: v for k, v in config.items() if k not in ("problem", "network")}
    assert all(isinstance(v, (int, float, str, bool)) for v in scalars.values())


def build(package: str, path: Path):
    """(problem, flow, sample), the way that package's train.py builds them."""
    config = args(path)
    train = importlib.import_module(f"canna.{package}.train")

    # point inlined its trainer into __main__, so neither TrainState nor train_sample
    # is importable: rebuild the sample the same way train.py's __main__ does
    if package == "point":
        key_sample, key_network = jr.split(jr.key(config.seed))
        problem = train.NoisyPoint(**config.problem)
        p = problem.sample_physical(key_sample)
        x = problem.physical_to_flow(p)
        sample = PointSample(
            xt=x,
            dx=x,
            t=jnp.asarray(0.5),
            y=problem.preprocess(problem.sample_observation(key_sample, p)),
        )
        flow = train.PointFlow(
            **config.network,
            x_shape=sample.xt.shape,
            y_shape=sample.y.shape,
            dtype=jnp.dtype(config.dtype),
            param_dtype=jnp.float32,
            key=key_network,
        )
        return problem, flow, sample

    state = train.TrainState.from_config(config)
    return state.problem, state.flow, train.train_sample(state.problem, state.key)


@pytest.mark.parametrize("package,path", CONFIGS, ids=IDS)
def test_config_builds_the_problem_and_the_network(package, path):
    problem, flow, _ = build(package, path)
    assert problem is not None and flow is not None


@pytest.mark.parametrize("package,path", CONFIGS, ids=IDS)
def test_config_problem_and_network_agree_on_the_conditioning_shape(package, path):
    # the network is shaped from one problem sample, so a mismatch shows up as a
    # shape error the first time the flow is actually called
    _, flow, sample = build(package, path)
    outs = flow(*(sample.xt, sample.t, sample.y, *sample[6:]))

    # point has no reconstruction heads, so its flow returns the velocity alone
    if package == "point":
        assert outs.shape == sample.dx.shape
    else:
        dx, x_target, y_target = outs
        assert dx.shape == sample.dx.shape
        assert x_target.shape == sample.x_target.shape
        assert y_target.shape == sample.y_target.shape
