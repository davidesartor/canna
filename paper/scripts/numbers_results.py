"""Numbers and table rows of appendix/results.tex: the source-count runs, from data/runs/source-count.json.

That file is written by import_runs.py (local, like the rest of paper/data/). Widths are flow over joint MCMC,
1.4826 MAD of the truth-relabelled, (psi, phi0)-folded draws, median over the sources of an SNR band; found is
an f0 width below one bin; coverage is the share of true flow coordinates inside the central 68% and 95%
intervals, pooled over the eight coordinates and every source of the band (outputs/runs/261103-*.py).
"""

import json
import math

import numpy as np

import style

DATA = style.PAPER / "data/runs/source-count.json"
runs = json.loads(DATA.read_text())["runs"]
CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]
BANDS = ("7-15", "15-40", ">=40")
BAND_TEX = {"7-15": "7--15", "15-40": "15--40", ">=40": r"$\ge40$"}
WORDS = {2: "two", 4: "four", 8: "eight", 16: "sixteen"}
FORKS = (4, 8, 16, 32)

by_name = {r["run"]: r for r in runs}


def main(band: str, n: int) -> dict:
    return by_name[f"261103-n{n}-{band}-B"]


def fork(band: str, n: int, epoch: int) -> dict:
    return by_name[f"261103-n{n}-{band}-B-e{epoch}"]


def family(band: str, n: int) -> list[dict]:
    """The main run and its cooldown forks that have an eval."""
    out = [main(band, n)] + [fork(band, n, e) for e in FORKS if f"261103-n{n}-{band}-B-e{e}" in by_name]
    return [r for r in out if r["evals"]]


def selected(band: str, n: int) -> dict:
    """The model with the lowest eval-catalogue NLL at its evaluated epoch, among the main run and its forks."""
    return min(family(band, n), key=lambda r: r["evals"][-1]["catalog_at_eval"])


def label(run: dict) -> str:
    return "main" if run["fork_epoch"] is None else f"fork e{run['fork_epoch']}"


def row(ev: dict, band: str, table: str) -> dict | None:
    return next((x for x in ev[table] if x["band"] == band), None)


def scores_band(ev: dict, band: str) -> dict | None:
    """The scorecard rows use 40-100 and >=100 above 40: pool them like the MCMC rows' >=40."""
    if band != ">=40":
        return row(ev, band, "scores")
    rows = [x for x in ev["scores"] if x["band"] in ("40-100", ">=100", "100-inf")]
    n = sum(x["n"] for x in rows)
    if not n:
        return None
    pool = lambda k: sum(x[k] * x["n"] for x in rows) / n
    return {
        "n": n,
        "found": pool("found"),
        "in68": [sum(x["in68"][c] * x["n"] for x in rows) / n for c in range(8)],
        "in95": [sum(x["in95"][c] * x["n"] for x in rows) / n for c in range(8)],
    }


def coverage(ev: dict, key: str) -> float:
    rows = ev["scores"]
    return sum(np.mean(x[key]) * x["n"] for x in rows) / sum(x["n"] for x in rows)


def f2(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "--"
    return f"{x:.1f}" if x >= 9.95 else f"{x:.2f}"


def hours(run: dict) -> float:
    return run["epochs_planned"] * run["log_interval"] / 1000 * run["smoke_epoch_s_per_1000"] / 3600


numbers: dict[str, str] = {}

# ---- the runs -------------------------------------------------------------------------------------------
run_rows = []
for band, n in CONFIGS:
    r = main(band, n)
    planned, done = r["epochs_planned"], r["epochs_done"]
    finite = [c for c in r["catalog"] if c is not None]
    best = min(finite)
    best_epoch = r["catalog"].index(best) + 1
    if done == planned:
        outcome = "finished"
    elif not r["evals"] and done < planned:
        outcome = r"\textcolor{red}{training}"
    else:
        outcome = f"diverged, epoch {done + 1}"
    best_cell = rf"\num{{{best:.3f}}} ({best_epoch})" if r["evals"] else "--"
    run_rows.append(
        rf"{band} & {n} & \num{{{r['total_steps'] / 1e6:.2f}}} & {planned} & \num{{{r['warmup_steps']}}}"
        rf" & {hours(r):.0f} & {outcome} & {best_cell} \\"
    )
numbers["ResultsRunsRows"] = "\n".join(run_rows)

ref = main("narrow", 2)
numbers["ResultsNarrowCounts"] = "2, 4 and 8"
numbers["ResultsFullCounts"] = "4, 8 and 16"
assert [n for b, n in CONFIGS if b == "narrow"] == [2, 4, 8] and [n for b, n in CONFIGS if b == "full"] == [4, 8, 16]
numbers["ResultsEpochs"] = str(ref["epochs_planned"])
numbers["ResultsSnapshotEvery"] = "4"
numbers["ResultsForkEpochs"] = ", ".join(map(str, FORKS))
numbers["ResultsCooldownPercent"] = f"{100 * ref['cooldown_frac']:.0f}"
numbers["ResultsNarrowLowMhz"] = f"{ref['f0_range_hz'][0] * 1e3:g}"
numbers["ResultsNarrowHighMhz"] = f"{ref['f0_range_hz'][1] * 1e3:g}"
numbers["ResultsFullLowMhz"] = f"{main('full', 4)['f0_range_hz'][0] * 1e3:g}"
numbers["ResultsFullHighMhz"] = f"{main('full', 4)['f0_range_hz'][1] * 1e3:g}"
numbers["ResultsCommit"] = ref["commit"][:7]
full4 = main("full", 4)
numbers["ResultsFullFourDivergedEpoch"] = str(full4["epochs_done"] + 1)
numbers["ResultsFullFourBestEpoch"] = str(full4["evals"][-1]["epoch"])
def hour_range(band: str) -> str:
    h = sorted({round(hours(main(band, n))) for b, n in CONFIGS if b == band})
    return str(h[0]) if len(h) == 1 else f"{h[0]}--{h[-1]}"


# measured on TREX: summed elapsed time of each main run's Slurm segments and its self-requeues, from the
# sacct blocks of outputs/trex-logs/source-count-261103/canna-lisa-1658035[3-8].out (2026-11-11)
WALL = {("narrow", 2): (33.1, 1), ("narrow", 4): (34.2, 1), ("narrow", 8): (31.9, 1),
        ("full", 4): (52.9, 2), ("full", 8): (66.0, 2), ("full", 16): (63.8, 2)}
numbers["ResultsWallNarrow"] = f"{min(WALL[b, n][0] for b, n in CONFIGS if b == 'narrow'):.0f}--{max(WALL[b, n][0] for b, n in CONFIGS if b == 'narrow'):.0f}"
numbers["ResultsWallFull"] = f"{min(WALL['full', n][0] for n in (8, 16)):.0f}--{max(WALL['full', n][0] for n in (8, 16)):.0f}"
numbers["ResultsWallFullFour"] = f"{WALL['full', 4][0]:.0f}"
numbers["ResultsRequeuesNarrow"] = "one"
numbers["ResultsRequeuesFull"] = "two"
assert {WALL[k][1] for k in WALL if k[0] == "narrow"} == {1} and {WALL[k][1] for k in WALL if k[0] == "full"} == {2}
numbers["ResultsNarrowHours"] = hour_range("narrow")
numbers["ResultsFullHours"] = hour_range("full")

# ---- widths against the joint MCMC, per configuration ----------------------------------------------------
COLS = ("f0 [bins]", "log fdot", "log A", "lon", "lat", "sin iota")
width_blocks = []
for band, n in CONFIGS:
    width_rows = []
    width_blocks.append(width_rows)
    if not family(band, n):
        width_rows.append(
            rf"{band}-{n} & \multicolumn{{11}}{{l}}{{\textcolor{{red}}{{TODO(run): training until about 11 November}}}} \\"
        )
        continue
    r = selected(band, n)
    ev = r["evals"][-1]
    first = True
    for b in BANDS:
        vm, sc = row(ev, b, "vs_mcmc"), scores_band(ev, b)
        if vm is None:
            continue
        lead = rf"{band}-{n}, {label(r)}" if first else ""
        first = False
        cells = " & ".join(f2(vm["width_ratio"][c]) for c in COLS)
        found = f"{sc['found']:.2f}" if sc else "--"
        c68 = f"{np.mean(sc['in68']):.2f}" if sc else "--"
        c95 = f"{np.mean(sc['in95']):.2f}" if sc else "--"
        width_rows.append(rf"{lead} & {BAND_TEX[b]} & {vm['n']} & {cells} & {found} & {c68} & {c95} \\")
numbers["ResultsWidthRows"] = "\n\\addlinespace\n".join("\n".join(rows) for rows in width_blocks)

# MCMC health over the six references in use
evs = [selected(b, n)["evals"][-1] for b, n in CONFIGS if family(b, n)]
numbers["ResultsMcmcRhatMax"] = f"{max(e['mcmc_max_rhat'] for e in evs):.3f}"
numbers["ResultsMcmcRhatNarrowMax"] = f"{max(selected('narrow', n)['evals'][-1]['mcmc_max_rhat'] for n in (2, 4, 8)):.3f}"
numbers["ResultsMcmcEssMin"] = f"{min(e['mcmc_min_ess'] for e in evs):.0f}"
numbers["ResultsMcmcFzeroCoverLo"] = f"{min(e['mcmc_f0_in68'] for e in evs):.2f}"
numbers["ResultsMcmcFzeroCoverHi"] = f"{max(e['mcmc_f0_in68'] for e in evs):.2f}"

# pooled calibration of every eval, main and forks
all_evals = [e for r in runs for e in r["evals"]]
numbers["ResultsEvals"] = str(len(all_evals))
numbers["ResultsCoverSixtyEightLo"] = f"{min(coverage(e, 'in68') for e in all_evals):.2f}"
numbers["ResultsCoverSixtyEightHi"] = f"{max(coverage(e, 'in68') for e in all_evals):.2f}"
numbers["ResultsCoverNinetyFiveLo"] = f"{min(coverage(e, 'in95') for e in all_evals):.2f}"
numbers["ResultsCoverNinetyFiveHi"] = f"{max(coverage(e, 'in95') for e in all_evals):.2f}"


def width(band: str, n: int, b: str, col: str = "f0 [bins]") -> float:
    return row(selected(band, n)["evals"][-1], b, "vs_mcmc")["width_ratio"][col]


def found(band: str, n: int, b: str) -> float:
    return scores_band(selected(band, n)["evals"][-1], b)["found"]


def off(band: str, n: int, key: str = "off_peak_snr15") -> float:
    return selected(band, n)["evals"][-1][key]


# narrow-2: the flow matches the MCMC
n2 = [width("narrow", 2, b, c) for b in ("7-15", "15-40") for c in COLS]
numbers["ResultsNarrowTwoWidthLo"] = f"{min(n2):.2f}"
numbers["ResultsNarrowTwoWidthHi"] = f"{max(n2):.2f}"
numbers["ResultsNarrowTwoFzeroMid"] = f"{width('narrow', 2, '15-40'):.2f}"
numbers["ResultsNarrowTwoFoundLow"] = f"{100 * found('narrow', 2, '7-15'):.0f}"
numbers["ResultsNarrowTwoOff"] = f"{100 * off('narrow', 2):.0f}"
numbers["ResultsNarrowTwoMcmcOff"] = f"{100 * off('narrow', 2, 'mcmc_off_peak_snr15'):.0f}"
numbers["ResultsNarrowTwoSkyMid"] = f"{width('narrow', 2, '15-40', 'lon'):.2f}"
# narrow-4
n4 = [width("narrow", 4, b, "f0 [bins]") for b in ("7-15", "15-40")]
numbers["ResultsNarrowFourFzeroLo"] = f"{min(n4):.2f}"
numbers["ResultsNarrowFourFzeroHi"] = f"{max(n4):.2f}"
numbers["ResultsNarrowFourSkyMid"] = f"{width('narrow', 4, '15-40', 'lon'):.2f}"
numbers["ResultsNarrowFourFoundLow"] = f"{100 * found('narrow', 4, '7-15'):.0f}"
numbers["ResultsNarrowFourOff"] = f"{100 * off('narrow', 4):.0f}"
numbers["ResultsNarrowFourMcmcOff"] = f"{100 * off('narrow', 4, 'mcmc_off_peak_snr15'):.0f}"
# narrow-8
numbers["ResultsNarrowEightFzeroLow"] = f"{width('narrow', 8, '7-15'):.1f}"
numbers["ResultsNarrowEightFzeroMid"] = f"{width('narrow', 8, '15-40'):.1f}"
numbers["ResultsNarrowEightFoundLow"] = f"{100 * found('narrow', 8, '7-15'):.0f}"
numbers["ResultsNarrowEightOff"] = f"{100 * off('narrow', 8):.0f}"
numbers["ResultsNarrowEightMcmcOff"] = f"{100 * off('narrow', 8, 'mcmc_off_peak_snr15'):.0f}"
# full band
for n, word in ((4, "Four"), (8, "Eight"), (16, "Sixteen")):
    numbers[f"ResultsFull{word}FzeroMid"] = f"{width('full', n, '15-40'):.1f}"
    numbers[f"ResultsFull{word}FzeroLoud"] = f"{width('full', n, '>=40'):.0f}"
    numbers[f"ResultsFull{word}FdotLoud"] = f"{width('full', n, '>=40', 'log fdot'):.0f}"
    numbers[f"ResultsFull{word}SkyLoud"] = f"{width('full', n, '>=40', 'lon'):.0f}"
    numbers[f"ResultsFull{word}AmpLoud"] = f"{width('full', n, '>=40', 'log A'):.1f}"
    numbers[f"ResultsFull{word}FoundLoud"] = f"{100 * found('full', n, '>=40'):.0f}"
    numbers[f"ResultsFull{word}FoundMid"] = f"{100 * found('full', n, '15-40'):.0f}"
    numbers[f"ResultsFull{word}Off"] = f"{100 * off('full', n):.0f}"
    numbers[f"ResultsFull{word}McmcOff"] = f"{100 * off('full', n, 'mcmc_off_peak_snr15'):.0f}"
    numbers[f"ResultsFull{word}Model"] = label(selected("full", n))
for n, word in ((2, "Two"), (4, "Four"), (8, "Eight")):
    numbers[f"ResultsNarrow{word}Model"] = label(selected("narrow", n))
# the faint sources: flow and MCMC agree below SNR 7 everywhere
FAINT = [("narrow", 2), ("narrow", 4), ("full", 4), ("full", 8)]  # narrow-8 and full-16 on their own  # narrow-8 is reported on its own
faint = [width(b, n, "0-7", c) for b, n in FAINT for c in COLS]
numbers["ResultsFaintWidthLo"] = f"{min(faint):.2f}"
numbers["ResultsFaintWidthHi"] = f"{max(faint):.2f}"
numbers["ResultsNarrowEightFaintFzero"] = f"{row(selected('narrow', 8)['evals'][-1], '0-7', 'vs_mcmc')['width_ratio']['f0 [bins]']:.1f}"
faint16 = row(selected("full", 16)["evals"][-1], "0-7", "vs_mcmc")["width_ratio"]
numbers["ResultsFullSixteenFaintFzero"] = f"{faint16['f0 [bins]']:.1f}"
numbers["ResultsFullSixteenFaintSkyLo"] = f"{min(faint16['lon'], faint16['lat']):.2f}"
numbers["ResultsFullSixteenFaintSkyHi"] = f"{max(faint16['lon'], faint16['lat']):.2f}"
numbers["ResultsFullSixteenMcmcRhat"] = f"{selected('full', 16)['evals'][-1]['mcmc_max_rhat']:.3f}"

# ---- cooldown forks ----------------------------------------------------------------------------------------
fork_rows = []
for band, n in CONFIGS:
    fam = {r["fork_epoch"]: r for r in family(band, n)}
    if not fam:
        continue
    cells = []
    for e in (*FORKS, None):
        r = fam.get(e)
        if r is None:
            cells.append("--")
            continue
        ev = r["evals"][-1]
        cells.append(rf"\num{{{ev['catalog_at_eval']:.3f}}} / {f2(row(ev, '15-40', 'vs_mcmc')['width_ratio']['f0 [bins]'])}")
    fork_rows.append(rf"{band}-{n} & " + " & ".join(cells) + r" \\")
numbers["ResultsForkRows"] = "\n".join(fork_rows)
cat = lambda band, n, e: (fork(band, n, e) if e else main(band, n))["evals"][-1]["catalog_at_eval"]
numbers["ResultsNarrowTwoForkGain"] = f"{cat('narrow', 2, 4) - cat('narrow', 2, 32):.2f}"
numbers["ResultsFullFourForkGain"] = f"{cat('full', 4, 4) - cat('full', 4, 32):.2f}"
loud_f0 = lambda r: row(r["evals"][-1], ">=40", "vs_mcmc")["width_ratio"]["f0 [bins]"]
numbers["ResultsFullFourMainLoud"] = f"{loud_f0(main('full', 4)):.0f}"
numbers["ResultsFullFourForkLoud"] = f"{loud_f0(fork('full', 4, 32)):.1f}"


# ---- per source (import_runs.py "sources"): closest-pair separation, where the unfound loud sources sit ------
def sources(band: str, n: int) -> dict:
    s = selected(band, n)["evals"][-1]["sources"]
    return {
        "snr": np.array([x["snr"] for x in s]),
        "sep": np.array([np.inf if x["sep_bins"] is None else x["sep_bins"] for x in s]),
        "f0_mhz": np.array([1e3 * x["f0_hz"] for x in s]),
        "width": np.array([x["flow_f0_width_bins"] for x in s]),
        "f0_ratio": np.array([x["ratio"][0] for x in s]),
    }


# narrow band, SNR >= 7: f0 width ratio by distance (bins) to the window's nearest other injected source
SEP_EDGES = [(0, 3), (3, 10), (10, np.inf)]
for n, word in ((4, "Four"), (8, "Eight")):
    s = sources("narrow", n)
    for (lo, hi), tag in zip(SEP_EDGES, ("Close", "Mid", "Far")):
        m = (s["snr"] >= 7) & (s["sep"] >= lo) & (s["sep"] < hi)
        numbers[f"ResultsNarrow{word}Sep{tag}"] = f"{np.median(s['f0_ratio'][m]):.1f}"
        numbers[f"ResultsNarrow{word}Sep{tag}N"] = str(int(m.sum()))
numbers["ResultsSepCloseBins"], numbers["ResultsSepFarBins"] = "3", "10"

# full band, SNR >= 40: frequency of the unfound sources (flow f0 width >= 1 bin)
FULL_SPLIT_MHZ = 10
numbers["ResultsFullSplitMhz"] = str(FULL_SPLIT_MHZ)
for n, word in ((4, "Four"), (8, "Eight"), (16, "Sixteen")):
    s = sources("full", n)
    loud = s["snr"] >= 40
    high, unfound = s["f0_mhz"] >= FULL_SPLIT_MHZ, s["width"] >= 1
    numbers[f"ResultsFull{word}LoudHighShare"] = f"{100 * high[loud].mean():.0f}"
    numbers[f"ResultsFull{word}UnfoundHighShare"] = f"{100 * high[loud & unfound].mean():.0f}"
    numbers[f"ResultsFull{word}FoundHigh"] = f"{100 * (~unfound)[loud & high].mean():.0f}"
    numbers[f"ResultsFull{word}FoundLowBand"] = f"{100 * (~unfound)[loud & ~high].mean():.0f}"
    numbers[f"ResultsFull{word}FzeroHigh"] = f"{np.median(s['f0_ratio'][loud & high]):.0f}"
    numbers[f"ResultsFull{word}FzeroLowBand"] = f"{np.median(s['f0_ratio'][loud & ~high]):.1f}"

# ---- the joint chains alone: posterior width of log10 fdot over the prior's conditional width ---------------
chains = json.loads(DATA.read_text())["chains"]
for band in ("narrow", "full"):
    snr = np.concatenate([c["snr"] for c in chains if c["band"] == band])
    ratio = np.concatenate([c["log10_fdot_width_over_prior"] for c in chains if c["band"] == band])
    for (lo, hi), tag in zip(((0, 7), (7, 15), (15, 40), (40, np.inf)), ("Faint", "Low", "Mid", "Loud")):
        m = (snr >= lo) & (snr < hi)
        numbers[f"Results{band.title()}FdotPrior{tag}"] = f"{np.median(ratio[m]):.2f}"
        numbers[f"Results{band.title()}FdotPrior{tag}N"] = str(int(m.sum()))

# ---- the validation pass: rank tests on prior injections, which never see the truth (scorecard.npz) -------------
P = lambda p: "--" if p is None else (f"{p:.3f}" if p >= 0.001 else "$<$0.001")
VALIDATION_TESTS = [("coords_min", None), ("tarp", None), ("joint_tarp", None), ("overlap", None),
                    ("own_snr", None), ("joint_log_likelihood", None)]
validation_rows = []
for band, n in CONFIGS:
    if not family(band, n):
        continue
    r = selected(band, n)
    ev = r["evals"][-1]
    v = ev.get("validation")
    if not v:
        continue
    for stratum, tex in (("all", "all"), ("15-40", "15--40"), ("40-100", "40--100")):
        cell = v[stratum]
        ps = [cell["coords_min"]] + [cell[t]["p"] if cell[t] else None for t, _ in VALIDATION_TESTS[1:]]
        ll = cell["joint_log_likelihood"]
        lead = rf"{band}-{n}, {label(r)}" if stratum == "all" else ""
        validation_rows.append(
            rf"{lead} & {tex} & {cell['n_sources']} & " + " & ".join(P(p) for p in ps)
            + (rf" & {ll['bias']:+.2f} & {ll['middle']:.2f}" if ll else r" & -- & --") + r" \\"
        )
    validation_rows.append(r"\addlinespace")
numbers["ResultsValidationRows"] = "\n".join(validation_rows[:-1])
vals = [selected(b, n)["evals"][-1]["validation"] for b, n in CONFIGS if family(b, n) and selected(b, n)["evals"][-1].get("validation")]
numbers["ResultsValidationInjections"] = str(selected("narrow", 2)["evals"][-1]["validation_injections"])
numbers["ResultsValidationDraws"] = "127"

style.save_numbers("results", numbers)
