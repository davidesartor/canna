"""Health of the cached joint reference chains, to data/joint-<problem id>/chains.json."""

import json
import sys
from pathlib import Path

import numpy as np

from canna.lisa import config as run_config
from canna.lisa.mcmc import MCMC_ROOT, cache_dir
from canna.lisa.problem import LisaGB

REPORT = Path(__file__).parents[1]


def main() -> int:
    config = run_config.load()
    problem = LisaGB(**config.problem)

    joint = []
    for path in sorted(
        cache_dir(problem, config.seed, MCMC_ROOT).glob("inj*_seed*.npz")
    ):
        z = np.load(path)
        joint.append(
            dict(
                injection=int(z["injection"]),
                seed=int(z["chain_seed"]),
                window_snr=float(z["window_snr"]),
                rhat_max=float(z["rhat"].max()),
                ess_min=float(z["ess"].min()),
                tau_max=float(z["tau"].max()),
                seconds=float(z["seconds"]),
            )
        )

    out = cache_dir(problem, config.seed, REPORT / "data") / "chains.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(joint=joint), indent=1) + "\n")
    injections = len({c["injection"] for c in joint})
    print(f"{len(joint)} joint chains over {injections} injections; saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
