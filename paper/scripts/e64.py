"""The models the paper reports (2026-11-18, user): the 64-epoch main run of every band and source count, "e64".

The cooldown forks (e4 to e32) stay only in the fork comparison of app:results. Full S=4's main run diverged at epoch 51,
so its e64 is the Student-t continuation from snapshot 44 (261103-n4-full-B-t44, TREX 18085060, eval 18162618); until
that eval is fetched into outputs/runs, every full S=4 entry is left blank (eval_dir returns None).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "outputs/runs"
CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]
MAIN_RUN = {("full", 4): "261103-n4-full-B-t44"}


def run_name(band: str, n: int) -> str:
    return MAIN_RUN.get((band, n), f"261103-n{n}-{band}-B")


def eval_dir(band: str, n: int) -> Path | None:
    """The run's last eval (outputs/runs/<run>/eval/<eval>/), None if it has none yet."""
    evals = sorted((RUNS / run_name(band, n) / "eval").glob("*/draws.npz"))
    return evals[-1].parent if evals else None


def reference_for(truth):
    """The packed joint-MCMC reference (reports/mcmc/data/eval-*.npz) whose injections are `truth`: matched by value,
    since the problem id that canna.lisa.mcmc.load_reference hashes changed after the runs' commit."""
    import numpy as np

    from canna.lisa.samples import load

    for path in sorted((REPO / "reports/mcmc/data").glob("eval-*.npz")):
        with np.load(path) as z:
            t = z["truth"] if "truth" in z else None
        if t is None:
            t = load(path)["truth"]
        if t.shape == truth.shape and np.allclose(t, truth):
            return load(path)
    raise LookupError("no packed reference matches these injections")
