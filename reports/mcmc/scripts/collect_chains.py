"""Health and widths of the cached narrowband and joint chains, to data/joint-<problem id>/chains.json."""

import json
import re
import sys
from pathlib import Path

import numpy as np

from joint_mcmc import chain_dir

from canna.lisa import config as run_config
from canna.lisa.mcmc import MCMC_ROOT, cache_dir, fold
from canna.lisa.problem import LisaGB

REPORT = Path(__file__).parents[1]


def width(a: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=0)), axis=0)


def health(z) -> dict:
    return dict(
        rhat_max=float(z["rhat"].max()),
        ess_min=float(z["ess"].min()),
        tau_max=float(z["tau"].max()),
        seconds=float(z["seconds"]),
    )


def main() -> int:
    config = run_config.load()
    problem = LisaGB(**config.problem)

    # narrowband: per chain, and the widths over every seed of a source
    narrowband, folded = [], {}
    for path in sorted(cache_dir(problem, config.seed).glob("inj*_src*_seed*.npz")):
        j, s, k = (int(v) for v in re.findall(r"\d+", path.stem))
        z = np.load(path)
        draws = fold(z["physical"], z["truth"], problem.t_obs)
        folded.setdefault((j, s), []).append(draws)
        narrowband.append(
            dict(inj=j, src=s, seed=k, snr=float(z["snr"]), draws=len(draws))
            | health(z)
            | dict(width=width(draws).tolist())
        )
    pooled = [
        dict(inj=j, src=s, width=width(np.concatenate(parts)).tolist())
        for (j, s), parts in folded.items()
    ]

    joint = []
    for path in sorted(
        chain_dir(problem, config.seed, MCMC_ROOT).glob("window*_seed*.npz")
    ):
        j, k = (int(v) for v in re.findall(r"\d+", path.stem))
        joint.append(dict(window=j, seed=k) | health(np.load(path)))

    out = chain_dir(problem, config.seed, REPORT / "data") / "chains.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(narrowband=narrowband, narrowband_pooled=pooled, joint=joint), indent=1
        )
        + "\n"
    )
    print(
        f"{len(narrowband)} narrowband chains over {len(pooled)} sources, {len(joint)} joint chains; saved {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
