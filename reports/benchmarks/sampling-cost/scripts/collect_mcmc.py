"""The wall time each joint reference MCMC chain saved with itself, per injection and seed, to data/mcmc.json."""

import json
import os
import re
from pathlib import Path

import numpy as np

from canna.lisa import config as run_config
from canna.lisa.mcmc import cache_dir
from canna.lisa.problem import LisaGB

FOLDER = Path(__file__).parents[1]


def chains(directory: Path) -> list[dict]:
    """One record per cached chain: its file's fields (inj, seed) and its seconds."""
    records = []
    for path in sorted(directory.glob("inj*_seed*.npz")):
        fields = {k: int(v) for k, v in re.findall(r"([a-z]+)(\d+)", path.stem)}
        with np.load(path) as chain:
            records.append(
                fields | dict(cache=path.parent.name, seconds=float(chain["seconds"]))
            )
    return records


if __name__ == "__main__":
    # as canna.lisa.mcmc picks it: a run's config when LISA_RUN names one, else the bare config
    config = run_config.load_run() if os.environ.get("LISA_RUN") else run_config.load()
    data = dict(joint=chains(cache_dir(LisaGB(**config.problem), config.seed)))
    (FOLDER / "data/mcmc.json").write_text(json.dumps(data, indent=1))
    for kind, records in data.items():
        minutes = [r["seconds"] / 60 for r in records]
        print(
            f"{kind}: {len(records)} chains, {min(minutes):.1f}-{max(minutes):.1f} min"
        )
