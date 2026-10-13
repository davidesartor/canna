"""Tabulate compare_mcmc.py logs side by side: per-source width ratios and coverage.

    python3 lisa_checks/mcmc_table.py R1=_r1 "fd x=_fd" "fd y=_fd_y" "sf+ y=_sfplus_y"
"""
import re
import statistics as st
import sys


def parse(path):
    out, cur, win = {}, None, None
    for line in open(path):
        m = re.match(r"window (\d+): window SNR", line)
        if m:
            win = int(m.group(1))
        m = re.match(r" source (\d+), SNR ([\d.]+)", line)
        if m:
            cur = (win, float(m.group(2)))  # by SNR: source indices changed with the ordering
            out[cur] = {}
        for key in ("flow / MCMC width", "MCMC median in flow 68", "MCMC median in flow 95"):
            if cur and line.strip().startswith(key):
                vals = line.strip()[len(key):].split()
                out[cur][key] = [v == "True" if v in ("True", "False") else float(v) for v in vals]
        m = re.search(r"flow draws > 1 bin from the MCMC median in f0: ([\d.]+)", line)
        if m and cur:
            out[cur]["off"] = float(m.group(1))
    return out


runs = dict(a.split("=") for a in sys.argv[1:])
d = {n: parse(f"outputs/mcmc/compare{t}.log") for n, t in runs.items()}
names = list(d)
cols = [(0, "f0"), (2, "A"), (3, "lon"), (4, "sin lat"), (6, "sin iota")]
print("win      SNR | " + " | ".join(f"{c}: " + " ".join(f"{n:>6s}" for n in names) for _, c in cols))
last = names[-1]
for k in sorted(d[last], key=lambda k: -k[1]):
    w = {n: d[n][k]["flow / MCMC width"] for n in names}
    print(f"{k[0]:3d}  {k[1]:7.1f} | " + " | ".join(
        " " * (len(c) + 2) + " ".join(f"{w[n][i]:6.1f}" for n in names) for i, c in cols))
print()
for n, dd in d.items():
    c95 = sum(sum(v["MCMC median in flow 95"]) for v in dd.values())
    c68 = sum(sum(v["MCMC median in flow 68"]) for v in dd.values())
    loud = [v["flow / MCMC width"] for k, v in dd.items() if k[1] >= 500]
    mid = [v["flow / MCMC width"] for k, v in dd.items() if 50 <= k[1] < 500]
    found = [v["flow / MCMC width"] for k, v in dd.items() if v["off"] <= 0.5 and k[1] < 50]
    print(f"{n:6s}: in95 {c95}/128 in68 {c68}/128 | loud (SNR >= 500) f0 median {st.median(l[0] for l in loud):5.1f},"
          f" A/sky/orient median {st.median(x for l in loud for x in (l[2], l[3], l[4], l[6])):4.1f}"
          f" | SNR 50-500 f0 {st.median(l[0] for l in mid):4.1f}"
          f" | missed {sum(v['off'] > 0.5 for v in dd.values())}/16")
