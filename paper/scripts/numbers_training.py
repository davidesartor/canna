"""Numbers of appendix/training.tex: the optimiser and schedule from canna.lisa, the compute from reports/benchmarks."""

import importlib.util
import inspect
import json
import math
import re
from statistics import StatisticsError

import style

import optax

from canna.lisa import config
from canna.lisa.optim import (
    ADAM_B2,
    CLIP_NORM,
    MUON_FIELDS,
    MUON_RMS,
    WARM_START_WARMUPS,
    normuon_rows,
    optimizer,
    schedule_free,
)
from canna.lisa.problem import LisaGB

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
SIZES = ("XS", "S", "B", "L", "XL")
BANDS = ("narrow", "full")
BENCH = style.REPO / "reports/benchmarks"
MISSING = "--"


def power(x: float) -> str:
    exponent = round(math.log10(x))
    assert math.isclose(x, 10.0**exponent), x
    return f"1e{exponent}"


def jsonl(path) -> list[dict]:
    """The records of a benchmark file, none if it is not there yet."""
    if not path.is_file():
        print(f"missing {path}")
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


configs = {(b, s): config.load(f"{b}-{s}") for b in BANDS for s in SIZES}
narrow, full = configs["narrow", "B"], configs["full", "B"]

# the optimiser and batch fields are shared by every band and size, so the text can quote one value
for field in (
    "batch_size",
    "micro_batches",
    "draws",
    "learning_rate",
    "adam_learning_rate",
    "weight_decay",
    "cooldown_frac",
    "muon_momentum",
    "nesterov",
    "momentum_dtype",
    "adam_b1",
    "sf_beta",
    "sf_beta_end",
    "log_interval",
    "ode_steps",
    "weights",
    "bf16",
):
    values = {getattr(c, field) for c in configs.values()}
    assert len(values) == 1, (field, values)
for band in BANDS:
    assert (
        len(
            {
                (configs[band, s].warmup_steps, configs[band, s].total_steps)
                for s in SIZES
            }
        )
        == 1
    ), band
assert full.cooldown_frac == 0.0 and full.nesterov and full.momentum_dtype == "bfloat16"
assert full.bf16 == "backbone" and full.weights == "average" and full.micro_batches == 1
assert (
    full.total_steps % full.log_interval == 0
    and narrow.total_steps % narrow.log_interval == 0
)
warmup_share = {c.warmup_steps / c.total_steps for c in (narrow, full)}
assert len(warmup_share) == 1, warmup_share
normuon_beta2 = inspect.signature(normuon_rows).parameters["beta2"].default
average_power = inspect.signature(schedule_free).parameters["r"].default
assert not re.search(r"\br=", inspect.getsource(optimizer))
newton_schulz_steps = (
    inspect.signature(optax.contrib.muon).parameters["ns_steps"].default
)
assert "ns_steps" not in inspect.getsource(optimizer)
adam_eps = float(
    re.search(
        r"optax\.adamw\(.*?eps=([0-9.e+-]+)", inspect.getsource(optimizer), re.S
    ).group(1)
)

# the training job's time limit and the requeue signal's lead, from slurm/lisa.sbatch
sbatch = (style.REPO / "slurm/lisa.sbatch").read_text()
hours, minutes, seconds = map(
    int, re.search(r"--time=(\d+):(\d+):(\d+)", sbatch).groups()
)
assert minutes == seconds == 0
requeue_s = int(re.search(r"--signal=B:USR1@(\d+)", sbatch).group(1))

# Muon's share of the parameters, per band and size (step-time-memory's params.jsonl)
muon_share = {
    r["config"]: r["param_optimizer"]["Muon"] / r["params"]
    for r in jsonl(BENCH / "step-time-memory/data/params.jsonl")
}
share = lambda name: f"{100 * muon_share[name]:.0f}" if name in muon_share else MISSING

numbers = {
    "TrainingClipNorm": f"{CLIP_NORM:g}",
    "TrainingMuonRate": power(full.learning_rate),
    "TrainingAdamRate": power(full.adam_learning_rate),
    "TrainingMuonRms": f"{MUON_RMS:g}",
    "TrainingMomentum": f"{full.muon_momentum:g}",
    "TrainingNorMuonBeta": f"{normuon_beta2:g}",
    "TrainingBetaOne": f"{full.adam_b1:g}",
    "TrainingBetaTwo": f"{ADAM_B2:g}",
    "TrainingWeightDecay": f"{full.weight_decay:g}",
    "TrainingAveragePower": f"{average_power:g}",
    "TrainingNewtonSchulzSteps": f"{newton_schulz_steps}",
    "TrainingNewtonSchulzWord": WORDS[newton_schulz_steps],
    "TrainingAdamEps": power(adam_eps),
    "TrainingBetaStart": f"{full.sf_beta:g}",
    "TrainingBetaEnd": f"{full.sf_beta_end:g}",
    "TrainingWarmStartWord": WORDS[WARM_START_WARMUPS],
    "TrainingMuonFieldCount": WORDS[len(MUON_FIELDS)],
    "TrainingMuonShareXS": share("full-XS"),
    "TrainingMuonShareB": share("full-B"),
    "TrainingMuonShareXL": share("full-XL"),
    "TrainingBatch": str(full.batch_size),
    "TrainingObservations": f"{full.batch_size // full.draws}",
    "TrainingDraws": f"{full.draws}",
    "TrainingEpochSteps": str(full.log_interval),
    "TrainingWarmupPercent": f"{100 * warmup_share.pop():g}",
    "TrainingJobHours": f"{hours}",
    "TrainingRequeueMinutes": (
        WORDS[requeue_s // 60] if requeue_s % 60 == 0 else f"{requeue_s}\\,s"
    ),
}
for band, cfg in (("Narrow", narrow), ("Full", full)):
    numbers[f"TrainingWarmup{band}"] = str(cfg.warmup_steps)
    numbers[f"TrainingWarmStart{band}"] = str(WARM_START_WARMUPS * cfg.warmup_steps)
    numbers[f"TrainingSteps{band}"] = str(cfg.total_steps)
    numbers[f"TrainingEpochs{band}"] = str(cfg.total_steps // cfg.log_interval)

# the B runs' step on an A100 with the bf16 backbone at the default batch (step-time-memory's step
# check), unsplit if it fits, else the fewest micro-batches that ran
steps = {}
for r in jsonl(BENCH / "step-time-memory/data/a100-bf16backbone-b256.jsonl"):
    if r.get("step_s") is None or r["batch"] * r["draws"] != full.batch_size:
        continue
    if (
        r["config"] not in steps
        or r["micro_batches"] < steps[r["config"]]["micro_batches"]
    ):
        steps[r["config"]] = r
for band in BANDS:
    name, key = f"{band}-B", f"TrainingCompute{band.capitalize()}B"
    step = steps.get(name)
    numbers[f"{key}DrawPercent"] = (
        f"{100 * step['draw_s'] / step['step_s']:.0f}"
        if step and step.get("draw_s")
        else MISSING
    )
    numbers[f"{key}Hours"] = (
        f"{step['step_s'] * configs[band, 'B'].total_steps / 3600:.0f}"
        if step
        else MISSING
    )

# checkpoint size, the save's block on the trainer and a restart's cost (training-overhead's model)
spec = importlib.util.spec_from_file_location(
    "chunk_model", BENCH / "training-overhead/scripts/overhead.py"
)
chunk_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunk_model)
overhead = {}
latest = {}
for path in sorted((BENCH / "training-overhead/data").glob("*.jsonl")):
    for r in jsonl(path):
        key = (r["config"], chunk_model.series_of(r))
        if key not in latest or int(r["job"]) > int(latest[key]["job"]):
            latest[key] = r
# full-B's 80 GB record shared its GPU with another process, so its restart is from the 40 GB card alone
overhead_series = {
    "narrow-B": "A100 80 GB · bf16 backbone",
    "full-B": "A100 40 GB · bf16 backbone",
}
for (name, series), r in latest.items():
    if overhead_series.get(name) != series or r.get("oom"):
        continue
    try:
        overhead[name] = chunk_model.costs(r)
    except (KeyError, IndexError, TypeError, StatisticsError) as error:
        print(f"overhead record {name} incomplete: {error!r}")
for band in BANDS:
    c, key = overhead.get(f"{band}-B"), f"TrainingCheckpoint{band.capitalize()}B"
    numbers[f"{key}Gib"] = f"{c['gib']:.2f}" if c else MISSING
    numbers[f"{key}Block"] = f"{c['block']:.2f}" if c else MISSING
    numbers[f"{key}RestartMinutes"] = f"{c['R'] / 60:.0f}" if c else MISSING
    default = chunk_model.chunk(c, full.log_interval) if c else None
    best = (
        min(
            (L for L in chunk_model.NICE if L <= c["total"]),
            key=lambda L: chunk_model.chunk(c, L)["lost"],
        )
        if c
        else None
    )
    numbers[f"{key}LostPercent"] = f"{100 * default['lost']:.1f}" if c else MISSING
    numbers[f"{key}BestInterval"] = str(best) if c else MISSING
    numbers[f"{key}BestLostPercent"] = (
        f"{100 * chunk_model.chunk(c, best)['lost']:.1f}" if c else MISSING
    )
assert chunk_model.DEFAULT_INTERVAL == full.log_interval
numbers["TrainingCheckpointChunkHours"] = f"{chunk_model.CHUNK_S // 3600}"

# bf16 over float32 at the precision sweep's batch, the same network on both sides (step-time-memory's sweep)
SWEEP_BATCH = 128
sweep = {}
for chip in ("a100", "l40s"):
    for precision, suffix in (
        ("none", ""),
        ("attention", "-bf16attention"),
        ("backbone", "-bf16backbone"),
    ):
        for band in BANDS:
            for r in jsonl(
                BENCH / f"step-time-memory/data/{chip}{suffix}-{band}.jsonl"
            ):
                if (
                    r["batch"] == SWEEP_BATCH
                    and r["micro_batches"] == 1
                    and r.get("step_s")
                ):
                    sweep[chip, precision, r["config"]] = r["step_s"]
speedup = {
    (chip, precision): [
        sweep[chip, "none", name] / step
        for (c, p, name), step in sweep.items()
        if (c, p) == (chip, precision) and (chip, "none", name) in sweep
    ]
    for chip in ("a100", "l40s")
    for precision in ("attention", "backbone")
}
numbers["TrainingSweepBatch"] = f"{SWEEP_BATCH}"
for chip, label in (("a100", "A"), ("l40s", "L")):
    numbers[f"TrainingBackboneSpeedup{label}Min"] = (
        f"{min(speedup[chip, 'backbone']):.0f}"
    )
    numbers[f"TrainingBackboneSpeedup{label}Max"] = (
        f"{max(speedup[chip, 'backbone']):.0f}"
    )
attention_gains = speedup["a100", "attention"] + speedup["l40s", "attention"]
numbers["TrainingAttentionGainPercent"] = f"{100 * (max(attention_gains) - 1):.0f}"

# the full band's batch draw: its planned memory, and its time on the CPU (batch-draw's simulator bench)
training_sources = inspect.signature(LisaGB).parameters["n_sources"].default
numbers["TrainingSources"] = WORDS[training_sources]
numbers["TrainingDrawPlanFullGib"] = f"{steps['full-B']['draw_plan_gib']:.1f}"
cpu_bench = json.loads((BENCH / "batch-draw/data/cpu.json").read_text())
cpu_draw_ms = {
    r["band"]: r["ms"]
    for r in cpu_bench["records"]
    if r["kind"] == "train_sample"
    and r["sources"] == training_sources
    and r["n"] == full.batch_size // full.draws
}
numbers["TrainingDrawCpuCores"] = WORDS[
    int(re.search(r"(\d+) cores", cpu_bench["device"]).group(1))
]
numbers["TrainingDrawCpuFullSeconds"] = f"{cpu_draw_ms['full-B'] / 1e3:.1f}"
numbers["TrainingDrawCpuOverStep"] = (
    f"{cpu_draw_ms['full-B'] / 1e3 / steps['full-B']['step_s']:.0f}"
)

for k, v in numbers.items():
    if MISSING in v:
        print(f"not measured yet: {k}")
style.save_numbers("training", numbers)
