"""The wall time each reference MCMC chain saved with itself, narrowband per source, joint and convergence trace per window, to data/mcmc.json."""

import json
import re
import sys
from pathlib import Path

import numpy as np

OUTPUTS = Path("outputs/mcmc")
FOLDER = Path(__file__).parents[1]


def chains(pattern: str) -> list[dict]:
    """One record per cached chain: its file's fields (inj, src, window, seed) and its seconds."""
    records = []
    for path in sorted(OUTPUTS.glob(pattern)):
        fields = {k: int(v) for k, v in re.findall(r"([a-z]+)(\d+)", path.stem)}
        with np.load(path) as chain:
            records.append(
                fields | dict(cache=path.parent.name, seconds=float(chain["seconds"]))
            )
    return records


if __name__ == "__main__":
    problem = sys.argv[1] if len(sys.argv) > 1 else "ad8a87a575"
    data = dict(
        narrowband=chains(f"eval-{problem}/inj*_src*_seed*.npz"),
        joint=chains(f"joint-{problem}/window*_seed*.npz"),
        convergence=chains(f"convergence/traces/joint-{problem}/window*_seed*.npz"),
    )
    (FOLDER / "data/mcmc.json").write_text(json.dumps(data, indent=1))
    for kind, records in data.items():
        minutes = [r["seconds"] / 60 for r in records]
        print(
            f"{kind}: {len(records)} chains, {min(minutes):.1f}-{max(minutes):.1f} min"
        )
