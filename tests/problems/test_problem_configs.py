from pathlib import Path

import pytest
import yaml

import canna
from canna import problems
from canna.problems.lisa import LisaGB

CONFIG_ROOT = Path(canna.__file__).parent / "configs" / "problem"
PROBLEM_CONFIGS = sorted(CONFIG_ROOT.glob("*.yaml"))


@pytest.mark.parametrize("config", PROBLEM_CONFIGS, ids=lambda p: p.stem)
def test_problem_config_inits(config):
    """every shipped problem config must instantiate (e.g. no DC-crossing band)"""
    spec = yaml.safe_load(config.read_text())
    getattr(problems, spec["class"])(**spec.get("init_args", {}))


def test_too_short_tobs_fails_at_init():
    """a t_obs so short the lowest-f0 band straddles DC must be rejected at init"""
    with pytest.raises(AssertionError):
        LisaGB(t_obs=7 * 24 * 60 * 60)
