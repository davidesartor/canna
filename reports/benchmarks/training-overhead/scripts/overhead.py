"""The 24 h chunk model over bench_overhead.py's records; build_page.py and the paper's scripts import it."""

import json
from statistics import median

CHUNK_S = 24 * 3600
MARGIN_S = 300
DEFAULT_INTERVAL = 1000
NICE = [m * 10**k for k in range(7) for m in (1, 2, 5)]
CURVE = [round(10 ** (1 + k / 40)) for k in range(161)]


def costs(r: dict) -> dict:
    """The measured parts of one config: step, restart and every per-epoch cost."""
    repo = r["checkpoint"][0]
    short, last = r["loop"][0], r["loop"][-1]
    step = r["step_s"]
    block = median(last["save_s"][1:])
    alone = median(repo["write_s"]) - median(repo["save_s"])
    # the short loop's later saves wait out the previous write: its length while the trainer runs
    waits = [s - block for s in short["save_s"][1:]]
    gaps = [
        h + e + d
        for h, e, d in zip(
            short["history_s"], short["epoch_s"][1:], short["diagnostics_s"][1:]
        )
    ]
    in_loop = (
        median(w + g for w, g in zip(waits, gaps)) if median(waits) > 0.05 else alone
    )
    contention = (
        max(0.0, median(last["epoch_s"][1:]) - last["epoch_s"][0])
        if len(r["loop"]) > 1
        else 0.0
    )
    restart = dict(
        uv=r["uv_s"],
        imports=r["import_s"] + r["git_s"] + r["setup_s"],
        restore=repo["restore_s"],
        compiles=r["diagnostics_compile_s"]
        + r["epoch_compile_s"]
        + max(0.0, r["diagnostics_first_s"] - r["diagnostics_s"])
        + max(0.0, r["epoch_first_s"] - r["steps"] * step),
    )
    return dict(
        step=step,
        restart=restart,
        R=sum(restart.values()),
        block=block,
        save_alone=median(repo["save_s"]),
        background=in_loop,
        background_alone=alone,
        contention=contention,
        diagnostics=r["diagnostics_s"],
        history={int(k): v for k, v in r["history_s"].items()},
        total=r["total_steps"],
        gib=sum(repo["file_bytes"].values()) / 2**30,
        long_epoch=last["epoch_s"][0] if len(r["loop"]) > 1 else None,
    )


def chunk(c: dict, interval: int) -> dict:
    """How one chunk splits at this log interval: margin, restart, per-epoch costs, killed work."""
    (l0, h0), (l1, h1) = sorted(c["history"].items())
    e0, e1 = c["total"] / l0, c["total"] / l1
    history = max(0.0, h1 + (h0 - h1) * (c["total"] / interval - e1) / (e0 - e1))
    train = interval * c["step"]
    gap = history + train + c["contention"] + c["diagnostics"]
    wait = max(0.0, c["background"] - gap)
    per_epoch = c["diagnostics"] + history + c["block"] + c["contention"] + wait
    epoch = train + per_epoch
    n = (CHUNK_S - MARGIN_S - c["R"]) / epoch - 0.5
    base = dict(per_epoch=per_epoch, wait=wait, history=history, n=max(n, 0.0))
    if n <= 0:
        return base | dict(
            lost=1.0,
            margin=MARGIN_S / CHUNK_S,
            restart=c["R"] / CHUNK_S,
            epochs=0.0,
            killed=0.0,
        )
    # a kill inside the last background write loses that epoch too
    unsaved = min(c["background"], epoch) / epoch * train
    parts = dict(
        margin=MARGIN_S,
        restart=c["R"],
        epochs=n * per_epoch,
        killed=epoch / 2 + unsaved,
    )
    out = {k: v / CHUNK_S for k, v in parts.items()}
    return base | out | dict(lost=sum(out.values()))


def series_of(r: dict) -> str:
    """Card and bf16 mode, e.g. 'A100 80 GB · bf16 backbone'; the two A100s are kept apart."""
    parts = r["chip"].removeprefix("NVIDIA ").split("-")
    chip = parts[0]
    if chip == "A100":
        chip += " " + parts[-1].replace("GB", " GB")
    return f"{chip} · bf16 {r['bf16']}"


def latest(paths) -> dict:
    """One record per (config, series): a config rerun in its own process (a later job) supersedes the shared-process one."""
    out = {}
    for path in paths:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["config"], series_of(r))
            if key not in out or int(r["job"]) > int(out[key]["job"]):
                out[key] = r
    return out
