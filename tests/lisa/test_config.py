"""A run's config.yaml is its source of truth: it round-trips, holds its training fields, and names its eval dirs."""

import dataclasses

import pytest
import yaml

from canna.lisa import config


def saved_run(tmp_path, **overrides):
    cfg = dataclasses.replace(
        config.load("narrow-B", output_dir=tmp_path, **overrides),
        run=config.run_name("narrow-B", tmp_path),
        commit="0" * 40,
    )
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    config.save(cfg, cfg.run_dir / "config.yaml")
    return cfg


def test_a_run_name_is_fresh_ends_in_its_config_and_is_reserved(tmp_path):
    runs = {config.run_name("narrow-B", tmp_path) for _ in range(20)}
    assert len(runs) == 20
    assert all(run.endswith("-narrow-B") and run.count("-") == 4 for run in runs)
    assert all((tmp_path / "runs" / run).is_dir() for run in runs)


def test_env_vars_read_scientific_ints_and_skip_empty_values(monkeypatch):
    monkeypatch.setenv("LISA_TOTAL_STEPS", "1e6")
    monkeypatch.setenv("LISA_RUN", "")
    cfg = config.load("narrow-B")
    assert cfg.total_steps == 1_000_000 and cfg.run == ""
    monkeypatch.setenv("LISA_TOTAL_STEPS", "1.5e-3")
    with pytest.raises(AssertionError, match="integer"):
        config.load("narrow-B")


def test_config_yaml_round_trips(tmp_path):
    cfg = saved_run(tmp_path, learning_rate=3e-3)
    assert config.load_run(cfg.run, output_dir=tmp_path) == cfg


def test_only_eval_fields_may_change(tmp_path):
    cfg = saved_run(tmp_path)
    changed = config.load_run(
        cfg.run, output_dir=tmp_path, ode_steps=8, weights="training"
    )
    assert (changed.ode_steps, changed.weights) == (8, "training")
    with pytest.raises(AssertionError, match="learning_rate"):
        config.load_run(cfg.run, output_dir=tmp_path, learning_rate=1.0)
    with pytest.raises(AssertionError, match="micro_batches"):
        config.load_run(cfg.run, output_dir=tmp_path, micro_batches=2)
    resumed = config.load_run(
        cfg.run, config.RESUME_FIELDS, output_dir=tmp_path, micro_batches=2
    )
    assert resumed.micro_batches == 2
    extended = config.load_run(
        cfg.run, config.RESUME_FIELDS, output_dir=tmp_path, total_steps=2 * cfg.total_steps
    )
    assert extended.total_steps == 2 * cfg.total_steps


def test_an_eval_dir_records_its_settings_and_refuses_a_clash(tmp_path):
    cfg = saved_run(tmp_path)
    path = config.eval_dir(cfg, 7, backend="gpu")
    assert path == cfg.run_dir / "eval" / f"7-average-{cfg.ode_steps}"
    config.eval_dir(cfg, 7, n_random=3)
    record = yaml.safe_load((path / "eval.yaml").read_text())
    assert record["backend"] == "gpu" and record["n_random"] == 3
    with pytest.raises(AssertionError, match="backend"):
        config.eval_dir(cfg, 7, backend="cpu")
