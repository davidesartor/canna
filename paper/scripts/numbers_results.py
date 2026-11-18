"""Numbers and table rows of appendix/results.tex: the source-count runs, from data/runs/source-count.json.

That file is written by import_runs.py (local, like the rest of paper/data/). Widths are flow over joint MCMC,
1.4826 MAD of the truth-relabelled, (psi, phi0)-folded draws, median over the sources of an SNR band; found is
an f0 width below one bin; coverage is the share of true flow coordinates inside the central 68% and 95%
intervals, pooled over the eight coordinates and every source of the band (outputs/runs/261103-*.py).
"""

import json
import re
import math

import numpy as np

import e64
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


def selected(band: str, n: int) -> dict | None:
    """The reported model (e64.py): the 64-epoch main run, full S=4's Student-t continuation; None until it is scored.
    (Until 2026-11-18: the lowest eval-catalogue NLL among the main run and its forks.)"""
    r = by_name.get(e64.run_name(band, n))
    return r if r and r["evals"] else None


def label(run: dict | None) -> str:
    return "--" if run is None else "main" if run["fork_epoch"] is None else f"fork e{run['fork_epoch']}"


NAN = float("nan")
# a full S=4 cell until its e64 (Student-t continuation) eval is fetched and scored
PENDING = r"\textcolor{red}{TODO(run): 261103-n4-full-B-t44, eval pending on TREX}"


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
    r = selected(band, n)
    if r is None:
        width_rows.append(rf"{band}-{n} & \multicolumn{{11}}{{l}}{{{PENDING}}} \\")
        continue
    ev = r["evals"][-1]
    first = True
    for b in BANDS:
        vm, sc = row(ev, b, "vs_mcmc"), scores_band(ev, b)
        if vm is None:
            continue
        lead = rf"{band}-{n}" if first else ""
        first = False
        cells = " & ".join(f2(vm["width_ratio"][c]) for c in COLS)
        found = f"{sc['found']:.2f}" if sc else "--"
        c68 = f"{np.mean(sc['in68']):.2f}" if sc else "--"
        c95 = f"{np.mean(sc['in95']):.2f}" if sc else "--"
        width_rows.append(rf"{lead} & {BAND_TEX[b]} & {vm['n']} & {cells} & {found} & {c68} & {c95} \\")
numbers["ResultsWidthRows"] = "\n\\addlinespace\n".join("\n".join(rows) for rows in width_blocks)

# MCMC health over the six references in use
evs = [selected(b, n)["evals"][-1] for b, n in CONFIGS if selected(b, n)]
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
    r = selected(band, n)
    return NAN if r is None else row(r["evals"][-1], b, "vs_mcmc")["width_ratio"][col]


def found(band: str, n: int, b: str) -> float:
    r = selected(band, n)
    return NAN if r is None else scores_band(r["evals"][-1], b)["found"]


def off(band: str, n: int, key: str = "off_peak_snr15") -> float:
    r = selected(band, n)
    return NAN if r is None else r["evals"][-1][key]


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
FAINT = [k for k in [("narrow", 2), ("narrow", 4), ("full", 4), ("full", 8)] if selected(*k)]  # narrow-8 and full-16 on their own  # narrow-8 is reported on its own
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
    r = selected(band, n)
    s = [] if r is None else r["evals"][-1]["sources"]
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
MOCK_DATA = style.PAPER / "data/runs/mock-validation.json"  # 261103-validation.py on the Gaussian mock, CONFIGS order
mock = dict(zip(CONFIGS, json.loads(MOCK_DATA.read_text()))) if MOCK_DATA.is_file() else {}


def validation_row(lead: str, tex: str, cell: dict) -> str:
    ps = [cell["coords_min"]] + [cell[t]["p"] if cell[t] else None for t, _ in VALIDATION_TESTS[1:]]
    ov, ll = cell["overlap"], cell["joint_log_likelihood"]
    return (rf"{lead} & {tex} & {cell['n_sources']} & " + " & ".join(P(p) for p in ps)
            + (rf" & {ov['bias']:+.2f}" if ov else " & --")
            + (rf" & {ll['bias']:+.2f} & {ll['middle']:.2f}" if ll else r" & -- & --") + r" \\")


for band, n in CONFIGS:
    r = selected(band, n)
    if r is None:
        validation_rows += [rf"{band}-{n} & \multicolumn{{10}}{{l}}{{{PENDING}}} \\", r"\addlinespace"]
        continue
    v = r["evals"][-1].get("validation")
    if not v:
        continue
    for stratum, tex in (("all", "all"), ("15-40", "15--40"), ("40-100", "40--100")):
        validation_rows.append(validation_row(rf"{band}-{n}" if stratum == "all" else "", tex, v[stratum]))
        if stratum == "all" and (band, n) in mock:
            validation_rows.append(validation_row(r"\quad Gaussian mock", "all", mock[band, n]["strata"]["all"]))
    validation_rows.append(r"\addlinespace")
numbers["ResultsValidationRows"] = "\n".join(validation_rows[:-1])
vals = [selected(b, n)["evals"][-1]["validation"] for b, n in CONFIGS if selected(b, n) and selected(b, n)["evals"][-1].get("validation")]
numbers["ResultsValidationInjections"] = str(selected("narrow", 2)["evals"][-1]["validation_injections"])
numbers["ResultsValidationDraws"] = "127"
if mock:
    off = lambda c, k: c["strata"]["all"][k]["bias"]
    flows = {k: selected(*k)["evals"][-1]["validation"]["all"] for k in mock if selected(*k)}
    numbers["ResultsMockLikeOffsetLo"] = f"{min(off(m, 'joint_log_likelihood') for m in mock.values()):+.2f}"
    numbers["ResultsMockLikeOffsetHi"] = f"{max(off(m, 'joint_log_likelihood') for m in mock.values()):+.2f}"
    numbers["ResultsMockOverlapOffsetLo"] = f"{min(off(m, 'overlap') for m in mock.values()):+.2f}"
    numbers["ResultsMockOverlapOffsetHi"] = f"{max(off(m, 'overlap') for m in mock.values()):+.2f}"
    numbers["ResultsFlowOverlapOffsetLo"] = f"{min(f['overlap']['bias'] for f in flows.values()):+.2f}"
    numbers["ResultsFlowOverlapOffsetHi"] = f"{max(f['overlap']['bias'] for f in flows.values()):+.2f}"
    excess = [flows[k]["joint_log_likelihood"]["bias"] - off(mock[k], "joint_log_likelihood") for k in flows]
    numbers["ResultsLikeExcessLo"] = f"{min(excess):+.2f}"
    numbers["ResultsLikeExcessHi"] = f"{max(excess):+.2f}"

# ---- the overlap rank per flow-coordinate group (import_overlap.py; TREX 2026-11-18) -------------------------------
OVERLAP_DATA = style.PAPER / "data/runs/overlap-groups.json"
if OVERLAP_DATA.is_file():
    og = json.loads(OVERLAP_DATA.read_text())
    signed = lambda v: f"{v:+.2f}".replace("-0.00", "+0.00")
    rng = lambda vals: (signed(min(vals)), signed(max(vals)))
    g = lambda lab, grp, st="all": [c[lab]["groups"][grp][st] for c in og.values()]
    d = lambda lab, grp, st="all": [c[lab]["groups"][grp][st] - c[lab]["groups"]["drawn"][st] for c in og.values()]
    single = ("frequency", "sky", "phases")
    pairs = {
        # the flow's offset with one of frequency, sky or phases set to the truth, and its rise over the drawn offset
        "ResultsOverlapFixFlow": [v for grp in single for v in g("flow", grp)],
        "ResultsOverlapFixFlowRise": [v for grp in single for v in d("flow", grp)],
        # the mock's change with frequency or sky set to the truth, and with the phases
        "ResultsOverlapFixMockFreqSky": [v for grp in ("frequency", "sky") for v in d("mock", grp)],
        "ResultsOverlapFixMockPhases": d("mock", "phases"),
        # frequency and phases set together, the flow
        "ResultsOverlapFreqPhasesFlow": g("flow", "frequency+phases"),
        # the moduli |z+-| set to the truth, change of the offset below snr_hat 7
        "ResultsOverlapModuliFaintFlow": d("flow", "moduli", "snr<7"),
        "ResultsOverlapModuliFaintMock": d("mock", "moduli", "snr<7"),
        # the flow's own rank offsets of |z+-| and of the own SNR, every stratum
        "ResultsOverlapAbsZFlow": [c["flow"]["ranks"][k][st] for c in og.values() for k in ("abs_zp", "abs_zm") for st in ("snr<7", "snr>=7")],
        "ResultsOverlapOwnSnrFlow": [c["flow"]["ranks"]["own_snr"][st] for c in og.values() for st in ("snr<7", "snr>=7")],
    }
    for name, vals in pairs.items():
        numbers[f"{name}Lo"], numbers[f"{name}Hi"] = rng(vals)

# ---- faint sources against the prior (import_faint.py): below SNR 3 the posterior is the prior ----------------------
FAINT_DATA = style.PAPER / "data/runs/faint-prior.json"
if FAINT_DATA.is_file():
    fp = json.loads(FAINT_DATA.read_text())
    numbers["ResultsInvisibleSnr"] = f"{fp['invisible_snr']:g}"
    by = {(c["band"], c["n_sources"]): c for c in fp["configs"]}
    for (band, n), word in (((b, k), f"{b.title()}{WORDS[k].title()}") for b, k in CONFIGS):
        c = by.get((band, n), {"flow_over_prior": NAN, "mcmc_over_prior": NAN})
        numbers[f"Results{word}FaintFlowPrior"] = f"{c['flow_over_prior']:.2f}"
        numbers[f"Results{word}FaintMcmcPrior"] = f"{c['mcmc_over_prior']:.2f}"
    others = [by[k] for k in by if k not in (("narrow", 8), ("full", 16))]
    numbers["ResultsFaintFlowPriorLo"] = f"{min(min(c['flow_over_prior'], c['mcmc_over_prior']) for c in others):.2f}"
    numbers["ResultsFaintFlowPriorHi"] = f"{max(max(c['flow_over_prior'], c['mcmc_over_prior']) for c in others):.2f}"

# ---- frame or flow (import_frame.py): f0 widths of the data-only frame and of the flow, against the chains ------------
FRAME_DATA = style.PAPER / "data/runs/frame-vs-flow.json"
if FRAME_DATA.is_file():
    fr = json.loads(FRAME_DATA.read_text())
    for tag, word in (("f4", "Four"), ("f8", "Eight"), ("f16", "Sixteen")):
        for group, gword in (("ge40_lo", "Low"), ("ge40_hi", "High")):
            # e64 not yet run through 261103-frame-vs-flow.py (import_frame.py skips other models): left as ??
            g = fr.get(tag, {}).get(group, {"frame_over_mcmc": NAN, "flow_over_frame": NAN})
            numbers[f"ResultsFull{word}Frame{gword}"] = f"{g['frame_over_mcmc']:.0f}"
            numbers[f"ResultsFull{word}Contract{gword}"] = f"{1 / g['flow_over_frame']:.0f}"
    for tag, word in (("n4", "Four"), ("f8", "Eight")):
        if f"{tag}-fp32" in fr:
            band = "Narrow" if tag[0] == "n" else "Full"
            for group, gword in (("15-40", "Mid"), ("ge40_lo", "Loud")):
                if group in fr[tag] and group in fr[f"{tag}-fp32"]:
                    numbers[f"Results{band}{word}FpBf{gword}"] = f"{fr[tag][group]['flow_over_mcmc']:.2f}"
                    numbers[f"Results{band}{word}FpThirtyTwo{gword}"] = f"{fr[tag + '-fp32'][group]['flow_over_mcmc']:.2f}"
                    numbers[f"Results{band}{word}FpBfCover{gword}"] = f"{fr[tag][group]['flow_within_1']:.2f}"
                    numbers[f"Results{band}{word}FpThirtyTwoCover{gword}"] = f"{fr[tag + '-fp32'][group]['flow_within_1']:.2f}"

# ---- sampler steps and robustness (import_probes.py; laptop GPU, 2026-11-13) ------------------------------------------
PROBE_DATA = style.PAPER / "data/runs/probes.json"
if PROBE_DATA.is_file():
    pr = json.loads(PROBE_DATA.read_text())
    STEP_WORDS = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour"}
    for cfg, band in (("n4", "Narrow"), ("f8", "Full")):
        if cfg not in pr["sampler"]:
            continue
        sp = pr["sampler"][cfg]
        numbers[f"ResultsSampler{band}Max"] = str(sp["max_steps"])
        for prow in sp["rows"]:
            if prow["steps"] in STEP_WORDS and prow["width_over_max_loud"] is not None:
                numbers[f"ResultsSampler{band}{STEP_WORDS[prow['steps']]}"] = f"{prow['width_over_max_loud']:.2f}"
    if pr.get("grids"):
        # midpoint on the power-2 and log grids from GRID_FROM steps up, both bands: the range of the width ratio
        GRID_FROM = 16
        mid = [prow["width_over_uniform_loud"] for g in pr["grids"].values() for prow in g["rows"]
               if prow["rule"] == "midpoint" and prow["steps"] >= GRID_FROM]
        numbers["ResultsSamplerGridFrom"] = str(GRID_FROM)
        numbers["ResultsSamplerGridLo"] = f"{min(mid):.2f}"
        numbers["ResultsSamplerGridHi"] = f"{max(mid):.2f}"
        for cfg, band in (("n4", "Narrow"), ("f8", "Full")):
            for prow in pr["grids"].get(cfg, {}).get("rows", []):
                if prow["rule"] == "euler" and prow["schedule"] == "uniform" and prow["steps"] in (32, 128):
                    word = "ThirtyTwo" if prow["steps"] == 32 else "Max"
                    numbers[f"ResultsSamplerEuler{band}{word}"] = f"{prow['width_over_uniform_loud']:.2f}"
                if prow["rule"] == "euler" and prow["schedule"] == "log" and prow["steps"] == 128:
                    numbers[f"ResultsSamplerEulerLog{band}Max"] = f"{prow['width_over_uniform_loud']:.2f}"
    if "n4" in pr["robustness"]:
        rb = pr["robustness"]["n4"]
        numbers["ResultsRobustInjections"] = str(rb["injections"] if not isinstance(rb["injections"], list) else len(rb["injections"]))
        numbers["ResultsRobustDraws"] = str(rb["draws"])
        c, b = rb["cases"]["noise ×2"], rb["cases"]["baseline"]
        numbers["ResultsRobustNoiseFaintWidth"] = f"{c['faint']['width_over_baseline']:.2f}"
        numbers["ResultsRobustNoiseFaintOffset"] = f"{c['faint']['offset_over_width']:.1f}"
        numbers["ResultsRobustNoiseFaintOffsetBase"] = f"{b['faint']['offset_over_width']:.1f}"
        numbers["ResultsRobustNoiseFaintWithin"] = f"{100 * c['faint']['within_one']:.0f}"
        numbers["ResultsRobustBaseFaintWithin"] = f"{100 * b['faint']['within_one']:.0f}"
        numbers["ResultsRobustNoiseLoudWidth"] = f"{c['loud']['width_over_baseline']:.2f}"
        numbers["ResultsRobustNoiseLoudWithin"] = f"{100 * c['loud']['within_one']:.0f}"
        numbers["ResultsRobustBaseLoudWithin"] = f"{100 * b['loud']['within_one']:.0f}"
        numbers["ResultsRobustNoiseSlotSnr"] = f"{c['claimed_slot_snr']:.1f}"
        numbers["ResultsRobustBaseSlotSnr"] = f"{b['claimed_slot_snr']:.1f}"
        numbers["ResultsRobustMoreFaintFound"] = f"{100 * rb['cases']['one source more']['faint']['found']:.0f}"
        numbers["ResultsRobustEmptySlotSnr"] = f"{rb['cases']['empty window']['claimed_slot_snr']:.1f}"
        others = [rb["cases"][k]["loud"]["within_one"] for k in ("noise ×0.5", "f0 past the edge", "fdot above the prior", "SNR above the prior")]
        numbers["ResultsRobustOtherWithinLo"] = f"{100 * min(others):.0f}"
        numbers["ResultsRobustOtherWithinHi"] = f"{100 * max(others):.0f}"
    if "f8" in pr["robustness"]:
        rf = pr["robustness"]["f8"]["cases"]
        c, b = rf["noise ×2"], rf["baseline"]
        numbers["ResultsRobustFullNoiseLoudWidth"] = f"{c['loud']['width_over_baseline']:.2f}"
        numbers["ResultsRobustFullNoiseLoudOffset"] = f"{c['loud']['offset_over_width']:.1f}"
        numbers["ResultsRobustFullNoiseLoudOffsetBase"] = f"{b['loud']['offset_over_width']:.1f}"
        numbers["ResultsRobustFullNoiseLoudWithin"] = f"{100 * c['loud']['within_one']:.0f}"
        numbers["ResultsRobustFullBaseLoudWithin"] = f"{100 * b['loud']['within_one']:.0f}"
        numbers["ResultsRobustFullNoiseSlotSnr"] = f"{c['claimed_slot_snr']:.1f}"
        numbers["ResultsRobustFullBaseSlotSnr"] = f"{b['claimed_slot_snr']:.1f}"
        others = [rf[k]["loud"]["within_one"] for k in ("noise ×0.5", "f0 past the edge", "fdot above the prior", "SNR above the prior")]
        numbers["ResultsRobustFullOtherWithinLo"] = f"{100 * min(others):.0f}"
        numbers["ResultsRobustFullOtherWithinHi"] = f"{100 * max(others):.0f}"

# ---- Jensen-Shannon divergence of the flow's marginals from the chains' (import_js.py, eq. 30 of Delmond+26) -----
JS_DATA = style.PAPER / "data/runs/js-divergence-main.json"  # e64 (import_js.py main); was js-divergence.json
if JS_DATA.is_file():
    jd = json.loads(JS_DATA.read_text())
    numbers["ResultsJsBins"] = str(jd["bins"])
    numbers["ResultsJsDraws"] = str(jd["models"][0]["null"]["n_flow"])
    numbers["ResultsJsFloorMedian"] = f"{np.median([m['null']['median'] for m in jd['models']]):.3f}"
    numbers["ResultsJsFloorHigh"] = f"{max(m['null']['p95'] for m in jd['models']):.3f}"
    js3 = lambda x: f"{x:.3f}" if x < 0.1 else f"{x:.2f}"
    JS_BANDS = (("7-15", 7, 15), ("15-40", 15, 40), (">=40", 40, np.inf))
    js_blocks, js_by = [], {}
    jd_by = {(m["band"], m["n_sources"]): m for m in jd["models"]}
    for band, n in CONFIGS:
        m = jd_by.get((band, n))
        if m is None:
            js_blocks.append([rf"{band}-{n} & \multicolumn{{11}}{{l}}{{{PENDING}}} \\"])
            continue
        r = selected(band, n)
        assert m["run"] == r["run"] and m["eval"] == r["evals"][-1]["eval"], (m["run"], r["run"])
        snr = np.array([s["snr"] for s in m["sources"]])
        js = np.array([s["js"] for s in m["sources"]])  # [sources, 8 COLUMNS]
        js_by[band, n] = (snr, js)
        rows, first = [], True
        for b, lo, hi in JS_BANDS:
            k = (snr >= lo) & (snr < hi)
            if not k.any():
                continue
            lead = rf"{band}-{n}" if first else ""
            first = False
            cells = " & ".join(js3(np.median(js[k, c])) for c in range(8))
            rows.append(rf"{lead} & {BAND_TEX[b]} & {int(k.sum())} & {cells} & {js3(np.median(js[k].mean(1)))} \\")
        js_blocks.append(rows)
    numbers["ResultsJsRows"] = "\n\\addlinespace\n".join("\n".join(rows) for rows in js_blocks)
    # below SNR 3 both are the prior, except where the chains sit at their Fisher start (narrow-8, full-16)
    EMPTY = (np.zeros(0), np.zeros((0, 8)))
    jsb = lambda key: js_by.get(key, EMPTY)
    faint = lambda key: jsb(key)[1][jsb(key)[0] < 3]
    good = [k for k in CONFIGS if k not in (("narrow", 8), ("full", 16)) and k in js_by]
    faint_all = [np.median(faint(k)[:, c]) for k in good for c in range(1, 8)]  # all but f0
    numbers["ResultsJsFaintLo"] = js3(min(faint_all))
    numbers["ResultsJsFaintHi"] = js3(max(faint_all))
    numbers["ResultsJsFaintFullFzero"] = js3(max(np.median(faint(("full", n))[:, 0]) for n in (4, 8) if ("full", n) in js_by))
    numbers["ResultsJsFaintNarrowTwoFzero"] = js3(np.median(faint(("narrow", 2))[:, 0]))
    numbers["ResultsJsFaintNarrowFourFzero"] = js3(np.median(faint(("narrow", 4))[:, 0]))
    numbers["ResultsJsFaintNarrowEightFzero"] = js3(np.median(faint(("narrow", 8))[:, 0]))
    numbers["ResultsJsFaintFullSixteenFzero"] = js3(np.median(faint(("full", 16))[:, 0]))
    # narrow S=2 at SNR 7-40: every coordinate's median
    snr2, js2 = jsb(("narrow", 2))
    k = (snr2 >= 7) & (snr2 < 40)
    med2 = [np.median(js2[k, c]) for c in range(8)]
    numbers["ResultsJsNarrowTwoLo"], numbers["ResultsJsNarrowTwoHi"] = js3(min(med2)), js3(max(med2))
    numbers["ResultsJsNarrowTwoMean"] = js3(np.median(js2[k].mean(1)))
    for n in (4, 8):
        s, j = jsb(("narrow", n))
        k = (s >= 7) & (s < 40)
        numbers[f"ResultsJsNarrow{WORDS[n].title()}Mean"] = js3(np.median(j[k].mean(1)))
        numbers[f"ResultsJsNarrow{WORDS[n].title()}Fzero"] = js3(np.median(j[k, 0]))
    for n in (4, 8, 16):
        s, j = jsb(("full", n))
        loud = s >= 40
        numbers[f"ResultsJsFull{WORDS[n].title()}LoudFzero"] = js3(np.median(j[loud, 0]))
        numbers[f"ResultsJsFull{WORDS[n].title()}LoudFdot"] = js3(np.median(j[loud, 1]))
        numbers[f"ResultsJsFull{WORDS[n].title()}LoudSky"] = js3(np.median(j[loud, 3]))
        numbers[f"ResultsJsFull{WORDS[n].title()}LoudMean"] = js3(np.median(j[loud].mean(1)))
        k = (s >= 7) & (s < 15)
        numbers[f"ResultsJsFull{WORDS[n].title()}LowMean"] = js3(np.median(j[k].mean(1)))
    # the scale: eq. 30 between two Gaussians, by width ratio and by offset in widths (quadrature)
    x = np.linspace(-400, 400, 800001)
    gauss = lambda mu, sd: np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    def js_gauss(p, q):
        mix = 0.5 * (p + q)
        kl = lambda a: np.trapezoid(np.where(a > 1e-300, a * np.log2(np.maximum(a, 1e-300) / np.maximum(mix, 1e-300)), 0.0), x)
        return 0.5 * kl(p) + 0.5 * kl(q)

    for tag, sd in (("OneHalf", 1.5), ("Two", 2.0), ("Three", 3.0), ("Ten", 10.0)):
        numbers[f"ResultsJsGaussWidth{tag}"] = js3(js_gauss(gauss(0, 1), gauss(0, sd)))
    for tag, mu in (("Half", 0.5), ("One", 1.0), ("Two", 2.0)):
        numbers[f"ResultsJsGaussOffset{tag}"] = js3(js_gauss(gauss(0, 1), gauss(mu, 1)))

# ---- Section 4 summary table: the 64-epoch main run (e64) of every problem (import_js.py main, import_sbc.py) -----
JS_MAIN = style.PAPER / "data/runs/js-divergence-main.json"
SBC_MAIN = style.PAPER / "data/runs/sbc-main.json"
if JS_MAIN.is_file() and SBC_MAIN.is_file():
    jm = {(m["band"], m["n_sources"]): m for m in json.loads(JS_MAIN.read_text())["models"]}
    sm = json.loads(SBC_MAIN.read_text())
    pfmt = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    rows = []
    for band, n in sorted(CONFIGS, key=lambda c: (c[0] != "narrow", c[1])):
        key = f"{band}-{n}"
        if (band, n) not in jm or key not in sm:
            # not landed yet (full S=4: the Student-t continuation 261103-n4-full-B-t44)
            rows.append(rf"n{n}-{band} & {n} & & & & & \\")
            continue
        m, s = jm[band, n], sm[key]
        snr = np.array([x["snr"] for x in m["sources"]])
        js = np.array([x["js"] for x in m["sources"]]).mean(1)  # mean over the eight coordinates
        low, high = js[(snr >= 7) & (snr < 15)].mean(), js[snr >= 15].mean()
        windows = f"{len(np.unique([x['injection'] for x in m['sources']]))} / {s['flow']['windows']}"
        rows.append(rf"n{n}-{band} & {n} & {windows} & {low:.2f} & {high:.2f} & {s['flow']['in68']:.3f} & {pfmt(s['flow']['min_p'])} \\")
    numbers["ResultsSummaryRows"] = "\n".join(rows)
    numbers["ResultsSummaryFloor"] = f"{np.median([m['null']['median'] for m in jm.values()]):.2f}"

# anything computed from a model that has not landed (full S=4's e64) prints as nan: mark it instead
numbers = {k: (r"\textcolor{red}{??}" if re.search(r"\bnan\b", v) else v) for k, v in numbers.items()}

style.save_numbers("results", numbers)
