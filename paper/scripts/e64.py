"""The models the paper reports, one per band and source count (2026-11-19, user: "pick the best runs, provided that they
are all close to the end"). Candidates: the 64-epoch main run and its fork cooled from epoch 32 (40 epochs), the two that
trained to near their end and agree within the run-to-run spread; the pick is the lower eval-catalogue NLL (the paper's
selection rule before 2026-11-18). Full S=4's main run diverged at epoch 51, so its candidate is the Student-t
continuation from snapshot 44 (261103-n4-full-B-t44), whose logged NLL does not compare (Gaussian k^2 of a Student-t
frame); it is narrower than the e32 fork on every parameter and is taken. Each model is read at the eval it has
(best checkpoint, or last for t44); no new evals. (2026-11-18 to 19: the 64-epoch main run everywhere, "e64".)
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "outputs/runs"
CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]
SELECTED = {
    ("narrow", 2): "261103-n2-narrow-B-e32",  # eval-catalogue NLL -1.379 vs main -1.362
    ("narrow", 4): "261103-n4-narrow-B",  # -1.135 vs e32 -1.092
    ("narrow", 8): "261103-n8-narrow-B-e32",  # -0.974 vs main -0.917
    ("full", 4): "261103-n4-full-B-t44",  # NLL not comparable; narrower than e32 (-1.200) on every parameter
    ("full", 8): "261103-n8-full-B",  # -1.105 vs e32 -1.083
    ("full", 16): "261103-n16-full-B-e32",  # -0.887 vs main -0.874
}
MAIN_RUN = {("full", 4): "261103-n4-full-B-t44"}  # the 64-epoch run of each problem (fork table "main" column)


def run_name(band: str, n: int) -> str:
    """The reported model of the problem."""
    return SELECTED[band, n]


def main_name(band: str, n: int) -> str:
    """The 64-epoch run of the problem (full S=4: the Student-t continuation)."""
    return MAIN_RUN.get((band, n), f"261103-n{n}-{band}-B")


def model_label(band: str, n: int) -> str:
    """Table label of the reported model and, in brackets, the epoch of its eval: "e32 (40)" for the fork cooled from
    epoch 32, "main (62)" for the 64-epoch run, "t44 (64)" for full S=4's continuation from epoch 44."""
    run = run_name(band, n)
    ev = eval_dir(band, n)
    epoch = ev.name.split("-")[0] if ev is not None else "?"
    kind = "t44" if run.endswith("-t44") else f"e{run.rsplit('-e', 1)[1]}" if "-B-e" in run else "main"
    return f"{kind} ({epoch})"


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
