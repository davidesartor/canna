"""Wall times of the joint chains run on TREX CPU nodes (2 cores each) from their Slurm logs
(outputs/analysis-261113/logs, local only) into data/runs/mcmc-cost.json: the long chains of narrow S=4 (job 17673807, 16 x
400 000 iterations, Fisher start) and narrow S=8 (17673808, 8 x 400 000, prior start for the faint sources), and, while
it runs, the rate of full S=16's prior-started chains (18033012, 200 000 iterations) from their progress lines. A
reference chain is the same 400 000 iterations (burn-in plus kept), so every chain's hours are scaled to 400 000
iterations (full S=16's completed 200 000-iteration chains doubled). numpy only."""

import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
LOGS = PAPER.parent / "outputs/analysis-261113/logs"
JOBS = {"narrow-4": ("long-n4", 17673807, 400_000), "narrow-8": ("prior-n8", 17673808, 400_000), "full-16": ("prior-f16", 18033012, 200_000)}

out = {}
for cfg, (name, job, iterations) in JOBS.items():
    hours, rate_hours = [], []
    for path in sorted(LOGS.glob(f"{name}-{job}_*.out")):
        text = path.read_text(errors="replace")
        m = re.search(r"^Elapsed time\s*:\s*(\d+)-?(\d*):?(\d+):(\d+)", text, re.M)
        if m and "COMPLETED" in text:
            parts = [int(x) for x in m.groups() if x != ""]
            h, mi, s = parts[-3:]
            days = parts[0] if len(parts) == 4 else 0
            hours.append((24 * days + h + mi / 60 + s / 3600) * 400_000 / iterations)
            continue
        # still running: extrapolate the last progress line "[i/total] seconds s" to 400 000 iterations
        prog = re.findall(r"\[(\d+)/(\d+)\] (\d+) s", text)
        if prog:
            i, total, sec = map(int, prog[-1])
            rate_hours.append(sec / i * 400_000 / 3600)
    out[cfg] = dict(job=job, iterations=iterations, cores=2, hours=hours, extrapolated_400k_hours=rate_hours)
    print(cfg, f"{len(hours)} completed {min(hours, default=float('nan')):.1f}-{max(hours, default=float('nan')):.1f} h,"
          f" {len(rate_hours)} running, 400k extrapolated {min(rate_hours, default=float('nan')):.0f}-{max(rate_hours, default=float('nan')):.0f} h")
(PAPER / "data/runs/mcmc-cost.json").write_text(json.dumps(out, indent=1))
