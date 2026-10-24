"""The run config: dataclass defaults, then LISA_CONFIG=<band>-<size>'s two .yamls, then LISA_<FIELD> env vars."""

import dataclasses
import os
import random
import sys
from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).parent / "configs"
CHOICES = dict(
    momentum_dtype=("bfloat16", "float32"),
    weights=("average", "training"),
    bf16=("none", "attention", "backbone"),
)
# what a finished run's config.yaml lets eval, scorecard and validation change; a resume, also the chunking
EVAL_FIELDS = ("ode_steps", "n_random", "n_validation", "weights", "bf16")
RESUME_FIELDS = ("micro_batches", "total_steps", "ode_steps", "n_random", "n_validation", "weights")
LOCATION_FIELDS = ("output_dir", "run")
ADJECTIVES = """
amber ancient bold brisk calm clever crimson dusty eager faint fierce gentle golden hidden
hollow humble icy jolly keen lively lucky mellow misty noble odd pale proud quiet rapid rusty
silent sleepy steady stormy swift tender velvet wild witty young
""".split()
NOUNS = """
badger beacon breeze canyon comet coral crane delta ember falcon fern fjord glacier harbor
heron island lantern lynx maple meadow meteor nebula orbit otter pebble pulsar quasar raven
reef river spruce summit thistle tide tundra valley walrus willow wren zephyr
""".split()


@dataclasses.dataclass(frozen=True)
class Config:
    """Everything a run reads: the problem and network kwargs and the training/eval settings."""

    name: str = "full-B"
    problem: dict = dataclasses.field(default_factory=dict)
    network: dict = dataclasses.field(default_factory=dict)
    output_dir: Path = Path("outputs")
    # the run's directory name under output_dir/runs, and the commit it trained at
    run: str = ""
    commit: str = ""
    seed: int = 0
    # optimizer: Muon's lr and decay (on its z, per unit lr), AdamW's lr, schedule-free's beta ramp
    batch_size: int = 4096
    # the batch's gradient accumulated over this many equal chunks, to fit a smaller GPU
    micro_batches: int = 1
    learning_rate: float = 1e-2
    adam_learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 10_000
    cooldown_frac: float = 0.0
    muon_momentum: float = 0.8
    nesterov: bool = True
    momentum_dtype: str = "bfloat16"
    adam_b1: float = 0.9
    sf_beta: float = 0.9
    sf_beta_end: float = 0.965
    # objective: batch_size (x0, t) samples, `draws` per observation, s = 1 - (1 - t)^time_power,
    # a late_time_mix share log-uniform in 1 - s
    draws: int = 16
    total_steps: int = 500_000
    log_interval: int = 1000
    time_power: float = 1.0
    late_time_mix: float = 0.5
    late_time_floor: float = 1e-4
    # eval, scorecard and validation
    ode_steps: int = 32
    n_random: int = 200
    n_validation: int = 1000
    weights: str = "average"
    # what runs in bfloat16: nothing, the attention scores and weighted sums, or the whole backbone
    bf16: str = "backbone"

    def __post_init__(self):
        for field, choices in CHOICES.items():
            assert getattr(self, field) in choices, f"{field} must be one of {choices}"
        assert self.batch_size % self.draws == 0, "draws must divide batch_size"
        assert (
            self.batch_size // self.draws % self.micro_batches == 0
        ), "micro_batches must divide batch_size // draws, the observations"

    @property
    def run_dir(self) -> Path:
        assert self.run, "no run: set LISA_RUN=<name>"
        return self.output_dir / "runs" / self.run


def coerce(default, value):
    """A yaml/env/kwarg value as the type of the field's default."""
    if isinstance(default, bool):
        assert isinstance(value, bool), f"{value!r} is not a bool"
        return value
    if isinstance(default, dict):
        assert isinstance(value, dict), f"{value!r} is not a mapping"
        return value
    # yaml reads 1e6 as a string
    if isinstance(default, int) and not isinstance(value, int):
        number = float(value)
        assert number.is_integer(), f"{value!r} is not an integer"
        return int(number)
    return type(default)(value)


def env_layer() -> dict:
    """The LISA_<FIELD> env vars, yaml-parsed, by field name; an empty one counts as unset."""
    return {
        var.removeprefix("LISA_").lower(): yaml.safe_load(text)
        for var, text in os.environ.items()
        if var.startswith("LISA_")
        and var not in ("LISA_CONFIG", "LISA_CONFIG_ROOT")
        and text.strip()
    }


def merge(layers: list[dict], values: dict | None = None) -> dict:
    """Fold config layers into one field -> value dict, each value coerced to its field's type; dicts merge."""
    defaults = Config()
    fields = {f.name for f in dataclasses.fields(Config)} - {"name"}
    values = dict(values or {})
    for layer in layers:
        unknown = set(layer) - fields
        assert not unknown, f"unknown config fields {sorted(unknown)}"
        for field, value in layer.items():
            value = coerce(getattr(defaults, field), value)
            if isinstance(value, dict):
                value = {**values.get(field, {}), **value}
            values[field] = value
    return values


def load(
    name: str | None = None, root: Path | str | None = None, **overrides
) -> Config:
    """Read band/<band>.yaml and size/<size>.yaml over the defaults, then LISA_<FIELD> env vars, then `overrides`; dicts merge."""
    name = name or os.environ.get("LISA_CONFIG", "full-B")
    root = Path(root or os.environ.get("LISA_CONFIG_ROOT", CONFIG_ROOT))

    assert name.count("-") == 1, f"config {name!r} is not <band>-<size>, e.g. narrow-B"
    band, size = name.split("-")
    layers = []
    for path in (root / "band" / f"{band}.yaml", root / "size" / f"{size}.yaml"):
        with open(path) as file:
            layers.append(yaml.safe_load(file) or {})
    return Config(name=name, **merge(layers + [env_layer(), overrides]))


def load_run(
    run: str | None = None, allowed: tuple[str, ...] = EVAL_FIELDS, **overrides
) -> Config:
    """A run's config.yaml; env vars and `overrides` may change only `allowed` fields (and where the run lives)."""
    layer = merge([env_layer(), overrides])
    if run:
        layer["run"] = run
    output_dir = layer.get("output_dir", Config.output_dir)
    assert layer.get("run"), "no run: set LISA_RUN=<name>"
    with open(Path(output_dir) / "runs" / layer["run"] / "config.yaml") as file:
        stored = yaml.safe_load(file)
    name = stored.pop("name")
    stored = merge([stored])

    fixed = sorted(
        field
        for field, value in merge([layer], stored).items()
        if field not in allowed + LOCATION_FIELDS and value != stored.get(field)
    )
    assert (
        not fixed
    ), f"{fixed} are fixed by the run's config.yaml; only {allowed} may change"
    location = dict(output_dir=Path(output_dir), run=layer["run"])
    return Config(name=name, **(merge([layer], stored) | location))


def save(config: Config, path: Path) -> None:
    """Write the resolved config as yaml, the run's source of truth."""
    values = dataclasses.asdict(config)
    values["output_dir"] = str(values["output_dir"])
    with open(path, "w") as file:
        yaml.safe_dump(values, file, sort_keys=False)


def run_name(name: str, output_dir: Path | str = Config.output_dir) -> str:
    """A fresh <adjective>-<noun>-<config> run name, its dir under output_dir/runs created so no other run takes it."""
    (Path(output_dir) / "runs").mkdir(parents=True, exist_ok=True)
    while True:
        run = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}-{name}"
        try:
            (Path(output_dir) / "runs" / run).mkdir()
            return run
        except FileExistsError:
            continue


def eval_tag(config: Config, epoch: int | str) -> str:
    """<epoch>-<weights>-<ode_steps>, then -<bf16> when the eval's precision is not the default backbone one."""
    tag = f"{epoch}-{config.weights}-{config.ode_steps}"
    return tag if config.bf16 == Config.bf16 else f"{tag}-{config.bf16}"


def eval_dir(config: Config, epoch: int, **stage: object) -> Path:
    """<run>/eval/<eval_tag>/, its eval.yaml recording the settings and `stage`'s."""
    path = config.run_dir / "eval" / eval_tag(config, epoch)
    path.mkdir(parents=True, exist_ok=True)
    settings = dict(
        epoch=int(epoch),
        weights=config.weights,
        ode_steps=config.ode_steps,
        bf16=config.bf16,
    )
    settings |= stage
    record = path / "eval.yaml"
    if record.exists():
        with open(record) as file:
            stored = yaml.safe_load(file)
        clash = sorted(k for k in settings if k in stored and stored[k] != settings[k])
        assert not clash, f"{path} was evaluated with other {clash}: {stored}"
        settings = stored | settings
    with open(record, "w") as file:
        yaml.safe_dump(settings, file, sort_keys=False)
    return path


if __name__ == "__main__":
    print(run_name(load(sys.argv[1]).name, load(sys.argv[1]).output_dir))
