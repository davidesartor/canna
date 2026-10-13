"""The schedule-free Muon comparison (research log F35): curves and scorecards of each variant.

Reads outputs/compare/<tag>/ (scorecard.npz, scorecard_training.npz) and the variant's slurm
log, whose epoch lines carry the optimizer diagnostics (train.py: heldout = x, heldout_y = y,
w_rms = backbone weight RMS, z_x = RMS of schedule-free's z - x).

    .venv/bin/python lisa_checks/compare_variants.py outputs/compare A B C D
"""

import re
import sys
from pathlib import Path

import numpy as np

root, tags = Path(sys.argv[1]), sys.argv[2:]
EPOCH = re.compile(
    r"\[epoch (\d+)/\d+\] flow=([\d.]+) x=([\d.]+) heldout=([\d.]+) heldout_y=([\d.]+)"
    r" grad_norm=([\d.]+) w_rms=([\d.]+) z_x=([\d.]+)"
)
COLS = ["epoch", "flow", "x", "held_x", "held_y", "grad_norm", "w_rms", "z_x"]


def curve(tag):
    log = next((root / tag).glob("*.out"))
    rows = [tuple(map(float, m.groups())) for m in EPOCH.finditer(log.read_text())]
    return {c: np.array(v) for c, v in zip(COLS, zip(*rows))}


def floor(card):
    loud = (card["snr"] >= 100) & (card["width"] < 1)
    return np.median(card["width"][loud])


print(f"{'variant':8s} {'epoch':>5s} {'flow(y)':>8s} {'held x':>7s} {'held y':>7s} {'x-y':>7s}"
      f" {'w_rms':>6s} {'z-x':>6s} {'gnorm':>6s}")
for tag in tags:
    c = curve(tag)
    for e in (10, 25, 50, 75, 100, 125, 150):
        i = np.searchsorted(c["epoch"], e)
        if i >= len(c["epoch"]):
            continue
        sl = slice(max(i - 4, 0), i + 1)  # five-epoch means
        print(f"{tag:8s} {int(c['epoch'][i]):5d} {c['flow'][sl].mean():8.4f} {c['held_x'][sl].mean():7.4f}"
              f" {c['held_y'][sl].mean():7.4f} {(c['held_x'][sl] - c['held_y'][sl]).mean():+7.4f}"
              f" {c['w_rms'][i]:6.3f} {c['z_x'][i]:6.3f} {c['grad_norm'][sl].mean():6.3f}")
    print()

print(f"{'variant':8s} {'weights':9s} {'floor':>7s} {'found':>6s} {'in68':>5s} {'in95':>5s} {'rank':>5s}"
      f" | loud: {'in68':>5s} {'rank':>5s}")
for tag in tags:
    for name, f in (("x", "scorecard.npz"), ("y", "scorecard_training.npz")):
        path = root / tag / f
        if not path.exists():
            continue
        d = dict(np.load(path))
        loud = d["snr"] >= 100
        print(f"{tag:8s} {name:9s} {floor(d):7.4f} {np.mean(d['width'] < 1):6.3f} {d['in68'].mean():5.2f}"
              f" {d['in95'].mean():5.2f} {d['rank'].mean():5.2f} | {d['in68'][loud].mean():11.2f}"
              f" {d['rank'][loud].mean():5.2f}")
