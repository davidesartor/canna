"""Numbers of appendix/amortisation.tex: training time of the cooldown forks, their fit, and the MCMC windows to break even."""

import json
import math

import yaml

import style

DATA = style.PAPER / "data/runs/source-count.json"
COST = style.REPO / "reports/benchmarks/sampling-cost/data/a100.jsonl"
BANDS = style.REPO / "src/canna/lisa/configs/band"
INTERIOR_RESPONSES = 3
CONFIGS = [
    ("narrow", 2),
    ("narrow", 4),
    ("narrow", 8),
    ("full", 4),
    ("full", 8),
    ("full", 16),
]
FORKS = (4, 8, 16, 32)
WORD = {"narrow": "Narrow", "full": "Full"}

runs = {r["run"]: r for r in json.loads(DATA.read_text())["runs"]}
flow_s = {
    r["config"].split("-")[0]: r["best_s"]
    for r in map(json.loads, COST.read_text().splitlines())
    if r["config"] in ("narrow-B", "full-B") and r["draws"] == 256
}


def hours(run: dict) -> float:
    return (
        run["epochs_planned"]
        * run["log_interval"]
        / 1000
        * run["smoke_epoch_s_per_1000"]
        / 3600
    )


def family(band: str, n: int) -> list[dict]:
    """The cooldown forks and the 64-epoch run (full S=4: its Student-t continuation), shortest first."""
    names = [f"261103-n{n}-{band}-B-e{k}" for k in FORKS]
    names.append(
        "261103-n4-full-B-t44" if (band, n) == ("full", 4) else f"261103-n{n}-{band}-B"
    )
    return [runs[m] for m in names if m in runs and runs[m]["evals"]]


def f0_ratio(run: dict) -> float:
    return next(x for x in run["evals"][-1]["vs_mcmc"] if x["band"] == "15-40")[
        "width_ratio"
    ]["f0 [bins]"]


def windows_per_pass(band: str) -> int:
    problem = yaml.safe_load((BANDS / f"{band}.yaml").read_text())["problem"]
    bins = (problem["f0_range"][1] - problem["f0_range"][0]) * problem["t_obs"]
    return math.ceil(bins / (INTERIOR_RESPONSES * problem["response_points"]))


def span(values: list[float], fmt: str = ".2f") -> str:
    lo, hi = min(values), max(values)
    return (
        f"{lo:{fmt}}" if f"{lo:{fmt}}" == f"{hi:{fmt}}" else f"{lo:{fmt}}--{hi:{fmt}}"
    )


numbers: dict[str, str] = {}
for band, word in WORD.items():
    families = [family(b, n) for b, n in CONFIGS if b == band]
    shortest = [fam[0] for fam in families]
    longest = [fam[-1] for fam in families]
    train_h = max(hours(r) for r in longest)
    passes = windows_per_pass(band)
    numbers[f"Amortisation{word}ShortEpochs"] = str(shortest[0]["epochs_planned"])
    numbers[f"Amortisation{word}ShortHours"] = span([hours(r) for r in shortest], ".1f")
    numbers[f"Amortisation{word}LongHours"] = f"{train_h:.0f}"
    numbers[f"Amortisation{word}ShortRatio"] = span(
        [f0_ratio(r) for r in shortest], ".1f"
    )
    numbers[f"Amortisation{word}LongRatio"] = span(
        [f0_ratio(r) for r in longest], ".1f"
    )
    numbers[f"Amortisation{word}FlowSeconds"] = f"{flow_s[band]:.2f}"
    numbers[f"Amortisation{word}Passes"] = f"{passes}"
    # MCMC time per window at which one pass over the band pays for the longest training
    numbers[f"Amortisation{word}PassMinutes"] = (
        f"{(3600 * train_h / passes + flow_s[band]) / 60:.1f}"
    )
    # windows to break even if a chain takes an hour, or a day
    numbers[f"Amortisation{word}HourWindows"] = (
        f"{3600 * train_h / (3600 - flow_s[band]):.0f}"
    )
    numbers[f"Amortisation{word}DayWindows"] = (
        f"{3600 * train_h / (86400 - flow_s[band]):.1f}"
    )
numbers["AmortisationInteriorResponses"] = str(INTERIOR_RESPONSES)
numbers["AmortisationDraws"] = "256"
style.save_numbers("amortisation", numbers)
