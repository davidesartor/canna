"""Redraw corner pages from the samples saved beside them, with no network, Fisher or MCMC run.

    python -m canna.lisa.replot outputs/18-lisa-B-rope/corner [more dirs or .npz files]

Every page eval.py draws (corner_page) leaves <page>.npz next to <page>.pdf; this rewrites
each .pdf from its .npz with the current styling.
"""

import sys
from pathlib import Path

import matplotlib


def main(argv=None) -> int:
    matplotlib.use("Agg")
    from .eval import replot_page

    targets = [Path(a) for a in (sys.argv[1:] if argv is None else argv)]
    files = [p for t in targets for p in (sorted(t.glob("*.npz")) if t.is_dir() else [t])]
    if not files:
        print("no saved page samples found", file=sys.stderr)
        return 1
    for npz in files:
        print(replot_page(npz), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
