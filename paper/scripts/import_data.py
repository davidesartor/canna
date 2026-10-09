"""Bring new run outputs into paper/data, in the small form the figure scripts read.

Maintainer tool: it reads the local, gitignored run outputs, which collaborators do not
have, and writes the compact files that are committed under paper/data.

    # a slurm log -> data/logs/<job>.csv (epoch lines only: no host, user or account info)
    python import_data.py logs path/to/canna-lisa-13882121.out [more.out ...]

    # eval draws -> data/eval/: only the injections the figures use, in float64
    python import_data.py eval ../../outputs/lisa-XS-late-768-cool/eval_draws.npz \
        ../../outputs/lisa-XS/eval-tools/fisher_eval_injections.npz --injections 1 5 9

Scorecards are already small: copy outputs/scorecards/<run>.npz to data/scorecards/.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
EPOCH = re.compile(
    r"^\[epoch (\d+)/(\d+)\] flow=([\d.eE+-]+) x=([\d.eE+-]+) y=([\d.eE+-]+)"
    r" aux_weight=([\d.]+)(?: lr_scale=([\d.]+))?(?: time=(\d+)s)?"
)
COLUMNS = ["epoch", "total", "flow", "x", "y", "aux_weight", "lr_scale", "seconds"]


def import_log(path: Path) -> Path:
    """Keep the per-epoch lines of a canna training log, nothing else."""
    job = re.search(r"(\d{6,})", path.name).group(1)
    rows = []
    with open(path) as f:
        for line in f:
            m = EPOCH.match(line)
            if m:
                e, total, flow, x, y, aux, lr, t = m.groups()
                rows.append([e, total, flow, x, y, aux, lr or "1", t or ""])
    out = DATA / "logs" / f"{job}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"{path.name}: {len(rows)} epochs -> {out.relative_to(DATA.parent)}")
    return out


def import_eval(draws_path: Path, fisher_path: Path, injections: list[int]) -> None:
    """The flow and Fisher draws of the chosen eval injections, plus every truth."""
    with np.load(draws_path) as d, np.load(fisher_path) as f:
        assert np.allclose(d["truth"], f["truth"]), "draws and Fisher are not the same injections"
        idx = np.asarray(injections)
        np.savez_compressed(
            DATA / "eval" / "final_eval_draws.npz",
            injections=idx, draws=d["draws"][idx], truth=d["truth"], snr=d["snr"],
            epoch=d["epoch"],
        )
        np.savez_compressed(
            DATA / "eval" / "fisher_eval_draws.npz",
            injections=idx, fisher=f["fisher"][idx], truth=f["truth"],
        )
    print(f"eval injections {injections} -> data/eval/final_eval_draws.npz, fisher_eval_draws.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    p_logs = sub.add_parser("logs")
    p_logs.add_argument("files", nargs="+", type=Path)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("draws", type=Path)
    p_eval.add_argument("fisher", type=Path)
    p_eval.add_argument("--injections", type=int, nargs="+", default=[1, 5, 9])
    args = parser.parse_args()
    if args.what == "logs":
        for path in args.files:
            import_log(path)
    else:
        (DATA / "eval").mkdir(parents=True, exist_ok=True)
        import_eval(args.draws, args.fisher, args.injections)
