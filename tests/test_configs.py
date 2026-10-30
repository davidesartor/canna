"""Every shipped run config must still splat into its package's problem and network."""

import argparse
import importlib
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
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
        # lisa is a namespace package whose configs are a band and a size, composed at load
        if package == "lisa":
            config_dir = importlib.import_module("canna.lisa.config").CONFIG_ROOT
            paths = sorted(config_dir.glob("band/*.yaml")) + sorted(
                config_dir.glob("size/*.yaml")
            )
        else:
            module = importlib.import_module(f"canna.{package}")
            paths = sorted((Path(module.__file__).parent / "configs").glob("*.yaml"))
        for path in paths:
            yield package, path


CONFIGS = list(configs())
IDS = [f"{package}-{path.stem}" for package, path in CONFIGS]


def args(package: str, path: Path):
    # lisa's run configs only set what differs from its Config defaults: each band at B, each size on narrow
    if package == "lisa":
        lisa_config = importlib.import_module("canna.lisa.config")
        name = f"{path.stem}-B" if path.parent.name == "band" else f"narrow-{path.stem}"
        return lisa_config.load(name, root=path.parent.parent, bf16="none")
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
    # a lisa band holds the problem (and maybe network flags), a lisa size only the network
    required, allowed = {
        "band": (("problem",), ("problem", "network")),
        "size": (("network",), ("network",)),
    }.get(path.parent.name, (("problem", "network"),) * 2)
    assert all(isinstance(config[key], dict) for key in required)
    assert all(isinstance(config[key], dict) for key in allowed if key in config)
    assert not ({"problem", "network"} - set(allowed)) & set(config)
    # flat schema: no class: key, no init_args: nesting, no cross-package composition
    assert not any(key in config for key in ("class", "init_args"))
    scalars = {k: v for k, v in config.items() if k not in ("problem", "network")}
    assert all(isinstance(v, (int, float, str, bool)) for v in scalars.values())


def build(package: str, path: Path):
    """(problem, flow, sample), the way that package's train.py builds them."""
    config = args(package, path)
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

    # lisa's flow and sample stay abstract (XL's flow alone takes GBs); its problem needs concrete values to build
    if package == "lisa":
        problem = train.LisaGB(**config.problem)
        flow = eqx.filter_eval_shape(
            lambda: train.LisaFlow(
                problem,
                **config.network,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                key=jr.key(config.seed),
            )
        )
        sample = eqx.filter_eval_shape(train.train_sample, problem, jr.key(config.seed))
        return problem, flow, sample

    state = train.TrainState.from_config(config)
    sample = train.train_sample(state.problem, state.key)
    return state.problem, state.flow, sample


@pytest.mark.parametrize("package,path", CONFIGS, ids=IDS)
def test_config_builds_the_problem_and_the_network(package, path):
    problem, flow, _ = build(package, path)
    assert problem is not None and flow is not None


@pytest.mark.parametrize("package,path", CONFIGS, ids=IDS)
def test_config_problem_and_network_agree_on_the_conditioning_shape(package, path):
    # the network is shaped from one problem sample, so a mismatch shows up as a
    # shape error the first time the flow is actually called
    _, flow, sample = build(package, path)

    # point has no reconstruction head and lisa's one output is z and log σ² per coordinate: both return one array
    if package == "point":
        assert flow(sample.xt, sample.t, sample.y).shape == sample.dx.shape
    elif package == "lisa":
        out = eqx.filter_eval_shape(
            lambda net, s: net(s.xt[0], s.t[0], s.y, s.f), flow, sample
        )
        assert sample.x_target.shape == sample.dx.shape[1:]
        assert out.shape == (*sample.x_target.shape[:-1], 2 * sample.x_target.shape[-1])
    else:
        outs = flow(*(sample.xt, sample.t, sample.y, *sample[6:]))
        dx, x_target, y_target = outs
        assert dx.shape == sample.dx.shape
        assert x_target.shape == sample.x_target.shape
        assert y_target.shape == sample.y_target.shape
