"""The numbers in appendix/mcmc.tex: the joint reference sampler's defaults and the MCMC reports' data."""

import ast
import inspect
import json
import math
import re

import numpy as np

import mcmc_reference
import style
from canna.lisa import config
from canna.lisa import mcmc
from canna.lisa.injections import N_INJECTIONS
from canna.lisa.problem import LisaGB

REPORTS = style.REPO / "reports"
JOINT = REPORTS / "mcmc"
CONVERGENCE = REPORTS / "mcmc/data/convergence-261113"  # updated 2026-11-16: narrow-4 long chains, catalogue prior
FISHER_BAND = (0.8, 1.25)
# updated 2026-11-18: Fisher against the six packed references; "high" from SNR 40, "faint" below 5, and the full band's
# SNR 5-15 bins where the forecast is wider than the chains
HIGH_SNR, MID_SNR, FAINT_SNR = 40.0, 15.0, 5.0
# the window drawn in fig_mcmc-posterior.py
POSTERIOR_BAND, POSTERIOR_INJECTION = ("narrow", 4), 101
COST = style.PAPER / "data/runs/mcmc-cost.json"
WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)
# by draws.json column: the existing chains sampled log Mc where the current sampler has log fdot
TEX_NAMES = {
    "f0 [bins]": r"$f_0$",
    "log Mc": r"$\ln\Mc$",
    "log fdot": r"$\ln\fdot$",
    "log A": r"$\ln\Amp$",
    "lon": r"$\alpha$",
    "lat": r"$\delta$",
    "psi": r"$\psi$",
    "sin iota": r"$\cos\iota$",
    "phi0": r"$\phi_0$",
}


def argparse_defaults(path) -> dict:
    """Literal defaults of every add_argument call in a script, by option name."""
    tree = ast.parse((style.REPO / path).read_text())
    return {
        node.args[0].value.lstrip("-"): ast.literal_eval(kw.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_argument"
        for kw in node.keywords
        if kw.arg == "default" and isinstance(kw.value, (ast.Constant, ast.List))
    }


def thousands(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "\\,")


def tex_param(name: str) -> str:
    """Convergence-trace name "s<k> <column>" -> "<tex symbol> of source <k + 1>"."""
    slot, column = name.split(" ", 1)
    return f"{TEX_NAMES[column]} of source {int(slot[1:]) + 1}"


cfg = config.load("narrow-B")
problem = LisaGB(**cfg.problem)
full = LisaGB(**config.load("full-B").problem)
run = argparse_defaults("src/canna/lisa/mcmc.py")
cores = int(
    re.search(
        r"--cpus-per-task=(\d+)", (style.REPO / "slurm/lisa-mcmc.sbatch").read_text()
    )[1]
)
n_injections = N_INJECTIONS
print(f"sampler {run}; cores {cores}; {n_injections} eval injections")

# updated 2026-11-18: the six packed references in use (mcmc_reference.py): one seed per injection, every injection with
# chains; widths of folded draws, the drift column in fdot
refs = {key: mcmc_reference.reference(*key) for key in mcmc_reference.PROBLEMS}
seeds = sorted({int(s) for d in refs.values() for s in d["n_seeds"]})
assert len(seeds) == 1, seeds
records = [dict(r, band=band, n=n) for (band, n) in refs for r in mcmc_reference.sources(band, n)]
snr_all = np.array([r["snr"] for r in records])
ratio_all = np.array([r["ratio"] for r in records])
others = [c for c in range(mcmc.N_PARAMS) if c != 1]  # every width but the drift's
in_band = lambda r: (r >= FISHER_BAND[0]) & (r <= FISHER_BAND[1])
high = snr_all >= HIGH_SNR
mid = (snr_all >= MID_SNR) & ~high
faint = snr_all < FAINT_SNR
is_full = np.array([r["band"] == "full" for r in records])
full_mid = is_full & (snr_all >= FAINT_SNR) & (snr_all < MID_SNR)
print(f"{len(records)} reference sources; high {high.sum()} in band {in_band(ratio_all[high][:, others]).mean():.2f}, "
      f"mid {mid.sum()} {in_band(ratio_all[mid][:, others]).mean():.2f}; faint f0 median {np.median(ratio_all[faint, 0]):.2f}; "
      f"full SNR 5-15 f0 median {np.median(ratio_all[full_mid, 0]):.2f}")
# per band, the f0 ratio by source count in the full band's SNR 5-15 bins: the range of the per-count medians
full_mid_f0 = [np.median(ratio_all[full_mid & np.array([r["n"] == n for r in records]), 0]) for n in (4, 8, 16)]
# the full band's mid-SNR drift: the forecast's fdot width over the chains'
full_mid_drift = np.median(ratio_all[full_mid, 1])
rhat_max = max(float(np.nanmax(d["rhat"])) for d in refs.values())
ess_min = min(float(np.nanmin(d["ess"])) for d in refs.values())
# convergence study of the long chains
summary = json.loads((CONVERGENCE / "summary.json").read_text())
curves = np.load(CONVERGENCE / "curves.npz")
crit, conv_chain = summary["criterion"], summary["chain"]
long_runs = summary["windows"]
paper = {j: w["settings"]["paper"] for j, w in long_runs.items()}
first_run = next(iter(long_runs.values()))


def segment_excess(j: str, start: int) -> tuple[float, str]:
    """Worst burn-test segment from `start` on, in units of its tolerance, and its parameter."""
    seg = curves[f"w{j}_burn_window"]
    ratio = np.maximum(
        curves[f"w{j}_burn_window_median"] / crit["median_tol"],
        curves[f"w{j}_burn_window_width"] / crit["width_tol"],
    )
    ratio = ratio[seg >= start]
    i, p = np.unravel_index(ratio.argmax(), ratio.shape)
    return float(ratio[i, p]), str(curves[f"w{j}_names"][p])


first_excess = {j: segment_excess(j, 0) for j in long_runs}
post_burn = {j: segment_excess(j, paper[j]["burn"]) for j in long_runs}
passing = [j for j, p in paper.items() if p["passes"]]
# failing runs from the narrowest miss to the worst
failing = sorted(
    (j for j, p in paper.items() if not p["passes"]), key=lambda j: post_burn[j][0]
)
print(
    f"convergence: passes {passing}, fails {failing}; first segment {first_excess}; after burn {post_burn}"
)
taus = [p["tau_max"] for p in paper.values()]
# mcmc.tex quotes the mildest and worst failure: "--" when every long chain passes
failing_numbers = dict.fromkeys(
    (
        "McmcConvMildAfter",
        "McmcConvMildParam",
        "McmcConvWorstFirst",
        "McmcConvWorstAfter",
        "McmcConvWorstParam",
        "McmcConvWorstSnr",
    ),
    "--",
)
if failing:
    mild, worst = failing[0], failing[-1]
    worst_slot = int(post_burn[worst][1].split(" ", 1)[0][1:])
    failing_numbers = {
        "McmcConvMildAfter": f"{post_burn[mild][0]:.2f}",
        "McmcConvMildParam": tex_param(post_burn[mild][1]),
        "McmcConvWorstFirst": f"{first_excess[worst][0]:.1f}",
        "McmcConvWorstAfter": f"{post_burn[worst][0]:.1f}",
        "McmcConvWorstParam": tex_param(post_burn[worst][1]),
        "McmcConvWorstSnr": f"{long_runs[worst]['snr'][worst_slot]:.0f}",
    }


# the window of the posterior figure: its two faint sources spread their f0 draws far from the truth
post = refs[POSTERIOR_BAND]
post_snr = post["source_snr"][POSTERIOR_INJECTION]
faintest = int(np.argmin(post_snr))
faint_offsets = mcmc_reference.folded(post["mcmc"][POSTERIOR_INJECTION, :, faintest], post["truth"][POSTERIOR_INJECTION, faintest])
far = np.abs(faint_offsets[:, 0]) > 1
spread = np.percentile(np.abs(faint_offsets[:, 0]), 90)
print(f"posterior window {POSTERIOR_INJECTION}: SNR {np.round(post_snr, 1)}, faintest {far.mean():.2f} of f0 draws > 1 bin off, 90% within {spread:.0f} bins")
# cost: the joint chains run on TREX (import_mcmc_cost.py)
cost = json.loads(COST.read_text())
# updated 2026-11-16: label-free reading of the long chains (reports/mcmc/scripts/label_free_widths.py, run on TREX).
# The traces are f0-sorted per draw, so a faint source crossing a loud one mixes their ranks; a loud source (SNR >= 7)
# takes the slot nearest its true f0 instead, and "found" is the share of draws with any slot within found_bins of
# the true f0. Narrow S=4: the Fisher-started convergence chains; narrow S=8: chains with every SNR < 7 source started
# from the prior (windows 66, 81, 51, 57).
def labelfree(name: str) -> dict:
    r = json.loads((CONVERGENCE / f"labelfree-{name}.json").read_text())
    groups = {}
    for rec in r["injections"]:
        snr = np.array(rec["snr"])
        for g, sel in (("faint", snr < r["faint_snr"]), ("mid", (snr >= r["faint_snr"]) & (snr < r["loud_snr"])),
                       ("loud", snr >= r["loud_snr"])):
            if not sel.any():
                continue
            d = groups.setdefault(g, {k: [] for k in ("q1", "q4", "ref", "flow", "prior", "fq1", "fq4", "fref", "fflow", "fprior")})
            q, fq = np.array(rec["chain_quarters"]), np.array(rec["found_chain_quarters"])
            for k, v in (("q1", q[0]), ("q4", q[3]), ("ref", rec["reference"]), ("flow", rec["flow"]), ("prior", rec["prior"])):
                d[k] += list(np.asarray(v)[sel])
            # found shares per window (median over the window's sources of the group)
            for k, v in (("fq1", fq[0]), ("fq4", fq[3]), ("fref", rec["found_reference"]), ("fflow", rec["found_flow"]),
                         ("fprior", rec["found_prior"])):
                d[k].append(float(np.median(np.asarray(v)[sel])))
    groups = {g: {k: np.asarray(v) for k, v in d.items()} for g, d in groups.items()}
    return dict(found_bins=r["found_bins"], **groups)


lf4, lf8 = labelfree("long-narrow-4"), labelfree("prior-narrow-8")
# updated 2026-11-18: full S=16's prior-started chains (TREX 18033012, windows 70, 77, 31, 42, two seeds, 200 000
# iterations; 31 and 42 hold only SNR < 3 sources), read by the same script (18033013)
lf16 = labelfree("prior-full-16")
med = lambda a: float(np.median(a))
rng = lambda a, f="{:.2f}": (f.format(np.min(a)), f.format(np.max(a)))
print(f"label-free narrow-4: loud chain/ref {med(lf4['loud']['q4'] / lf4['loud']['ref']):.2f}, faint chain/prior "
      f"{med(lf4['faint']['q1'] / lf4['faint']['prior']):.2f} -> {med(lf4['faint']['q4'] / lf4['faint']['prior']):.2f}, "
      f"mid found {lf4['mid']['fq1']} -> {lf4['mid']['fq4']}")
print(f"label-free narrow-8: loud chain/ref {med(lf8['loud']['q4'] / lf8['loud']['ref']):.2f}, mid found "
      f"{lf8['mid']['fq1']} -> {lf8['mid']['fq4']}, ref {lf8['mid']['fref']}, flow {lf8['mid']['fflow']}")
labelfree_numbers = {
    "McmcLabelFreeFoundBins": f"{lf4['found_bins']:g}",
    # narrow S=4, Fisher start
    "McmcConvLoudChainRef": f"{med(lf4['loud']['q4'] / lf4['loud']['ref']):.2f}",
    "McmcConvLoudDriftLo": rng(lf4["loud"]["q4"] / lf4["loud"]["q1"])[0],
    "McmcConvLoudDriftHi": rng(lf4["loud"]["q4"] / lf4["loud"]["q1"])[1],
    "McmcConvFaintPriorFirst": f"{med(lf4['faint']['q1'] / lf4['faint']['prior']):.2f}",
    "McmcConvFaintPriorLast": f"{med(lf4['faint']['q4'] / lf4['faint']['prior']):.2f}",
    "McmcConvFaintRefPrior": f"{med(lf4['faint']['ref'] / lf4['faint']['prior']):.2f}",
    "McmcConvFaintFlowPrior": f"{med(lf4['faint']['flow'] / lf4['faint']['prior']):.2f}",
    "McmcConvMidFoundFirst": ", ".join(f"{x:.2f}" for x in lf4["mid"]["fq1"]),
    "McmcConvMidFoundLast": ", ".join(f"{x:.2f}" for x in lf4["mid"]["fq4"]),
    "McmcConvMidFoundRef": ", ".join(f"{x:.2f}" for x in lf4["mid"]["fref"]),
    "McmcConvMidFoundFlow": ", ".join(f"{x:.2f}" for x in lf4["mid"]["fflow"]),
    # narrow S=8, prior start for every SNR < 7 source
    "McmcPriorStartLoudChainRef": f"{med(lf8['loud']['q4'] / lf8['loud']['ref']):.2f}",
    "McmcPriorStartLoudFlowChain": f"{med(lf8['loud']['flow'] / lf8['loud']['q4']):.2f}",
    "McmcPriorStartFaintChainPrior": f"{med(lf8['faint']['q4'] / lf8['faint']['prior']):.2f}",
    "McmcPriorStartFaintRefPrior": f"{med(lf8['faint']['ref'] / lf8['faint']['prior']):.2f}",
    "McmcPriorStartFaintFlowPrior": f"{med(lf8['faint']['flow'] / lf8['faint']['prior']):.2f}",
    "McmcPriorStartMidFoundFirstLo": rng(lf8["mid"]["fq1"])[0],
    "McmcPriorStartMidFoundFirstHi": rng(lf8["mid"]["fq1"])[1],
    "McmcPriorStartMidFoundLastLo": rng(lf8["mid"]["fq4"])[0],
    "McmcPriorStartMidFoundLastHi": rng(lf8["mid"]["fq4"])[1],
    "McmcPriorStartMidFoundRefLo": rng(lf8["mid"]["fref"])[0],
    "McmcPriorStartMidFoundRefHi": rng(lf8["mid"]["fref"])[1],
    "McmcPriorStartMidFoundFlowLo": rng(lf8["mid"]["fflow"])[0],
    "McmcPriorStartMidFoundFlowHi": rng(lf8["mid"]["fflow"])[1],
    "McmcPriorStartChance": f"{med(lf8['mid']['fprior']):.2f}",
    # full S=16, prior start for every SNR < 7 source
    "McmcPriorStartSixteenLoudChainRef": f"{med(lf16['loud']['q4'] / lf16['loud']['ref']):.1f}",
    "McmcPriorStartSixteenLoudChainRefLo": rng(lf16["loud"]["q4"] / lf16["loud"]["ref"], "{:.1f}")[0],
    "McmcPriorStartSixteenLoudChainRefHi": rng(lf16["loud"]["q4"] / lf16["loud"]["ref"], "{:.1f}")[1],
    "McmcPriorStartSixteenLoudGrowing": WORDS[int(np.sum(lf16["loud"]["q4"] > lf16["loud"]["q1"]))],
    "McmcPriorStartSixteenLoudN": WORDS[len(lf16["loud"]["q4"])],
    "McmcPriorStartSixteenLoudFlowRefLo": rng(lf16["loud"]["flow"] / lf16["loud"]["ref"])[0],
    "McmcPriorStartSixteenLoudFlowRefHi": rng(lf16["loud"]["flow"] / lf16["loud"]["ref"])[1],
    "McmcPriorStartSixteenFaintChainPrior": f"{med(lf16['faint']['q4'] / lf16['faint']['prior']):.2f}",
    "McmcPriorStartSixteenFaintRefPrior": f"{med(lf16['faint']['ref'] / lf16['faint']['prior']):.2f}",
    "McmcPriorStartSixteenFaintFlowPrior": f"{med(lf16['faint']['flow'] / lf16['faint']['prior']):.2f}",
    "McmcPriorStartSixteenMidFoundLastLo": rng(lf16["mid"]["fq4"])[0],
    "McmcPriorStartSixteenMidFoundLastHi": rng(lf16["mid"]["fq4"])[1],
    "McmcPriorStartSixteenMidFoundRefLo": rng(lf16["mid"]["fref"])[0],
    "McmcPriorStartSixteenMidFoundRefHi": rng(lf16["mid"]["fref"])[1],
    "McmcPriorStartSixteenMidFoundFlowLo": rng(lf16["mid"]["fflow"])[0],
    "McmcPriorStartSixteenMidFoundFlowHi": rng(lf16["mid"]["fflow"])[1],
}

style.save_numbers(
    "mcmc",
    {
        # sampler settings
        "McmcWalkers": str(run["nwalker"]),
        "McmcTemps": WORDS[len(run["temps"])],
        "McmcTempMax": f"{max(run['temps']):g}",
        "McmcBurn": thousands(run["burn"]),
        "McmcKeep": thousands(run["keep"]),
        "McmcThin": str(run["thin"]),
        # the references in use ran one seed per injection (the sampler's default lists two)
        "McmcSeeds": WORDS[seeds[0]],
        "McmcInjections": str(n_injections),
        "McmcChains": str(n_injections * seeds[0]),
        "McmcReferences": WORDS[len(refs)],
        "McmcChainsAll": thousands(n_injections * seeds[0] * len(refs)),
        "McmcKeepDraws": thousands(mcmc.N_KEEP),
        "McmcSlots": WORDS[problem.n_sources],
        "McmcResponsePoints": str(problem.response_points),
        "McmcResponsePointsFull": thousands(full.response_points),
        "McmcJointCores": WORDS[cores],
        "McmcJointDim": str(problem.n_sources * mcmc.N_PARAMS),
        "McmcBoxSigmas": f"{mcmc.BOX_SIGMAS:g}",
        "McmcFdotSigmas": f"{mcmc.FDOT_SIGMAS:g}",
        "McmcFdotCutMass": f"{0.5 * math.erfc(mcmc.FDOT_SIGMAS / math.sqrt(2)):.1g}",
        # the references' own diagnostics
        "McmcRhatGroups": WORDS[
            inspect.signature(mcmc.gelman_rubin).parameters["groups"].default
        ],
        "McmcRefRhatMax": f"{rhat_max:.3f}",
        "McmcRefEssMin": thousands(ess_min),
        # convergence study
        "McmcConvRuns": " and ".join(sorted(long_runs, key=int)),
        "McmcConvIterations": thousands(conv_chain["iterations"]),
        "McmcConvSeeds": WORDS[len(first_run["seeds"])],
        "McmcConvThin": str(conv_chain["thin"]),
        "McmcConvStep": thousands(first_run["step"]),
        "McmcConvBurnTaus": str(crit["burn_taus"]),
        "McmcConvEssCrit": thousands(crit["ess_min"]),
        "McmcConvRhatCrit": f"{crit['rhat_max']:g}",
        "McmcConvMedianTol": f"{crit['median_tol']:g}",
        "McmcConvWidthTol": f"{100 * crit['width_tol']:.0f}",
        "McmcConvTauLo": f"{min(taus):.0f}",
        "McmcConvTauHi": f"{max(taus):.0f}",
        "McmcConvEssLo": thousands(min(p["ess_min"] for p in paper.values())),
        "McmcConvRhatHi": f"{max(p['rhat_max'] for p in paper.values()):.4f}",
        "McmcConvDriftMedianHi": f"{max(p['median_drift_max'] for p in paper.values()):.3f}",
        "McmcConvDriftWidthHi": f"{max(p['width_drift_max'] for p in paper.values()):.3f}",
        "McmcConvPassing": " and ".join(passing),
        "McmcConvPassingBurns": " and ".join(
            thousands(long_runs[j]["burn"]) for j in passing
        ),
        "McmcConvFailing": " and ".join(failing),
        **failing_numbers,
        # the joint posterior figure
        "McmcPosteriorInjection": str(POSTERIOR_INJECTION),
        "McmcPosteriorWindowSnr": f"{post['snr'][POSTERIOR_INJECTION]:.0f}",
        "McmcPosteriorFaintSnr": f"{post_snr[faintest]:.1f}",
        "McmcPosteriorFarPercent": f"{100 * far.mean():.0f}",
        "McmcPosteriorSpreadBins": f"{spread:.0f}",
        # Fisher against the joint MCMC, the six references pooled
        "McmcFisherLo": f"{FISHER_BAND[0]:g}",
        "McmcFisherHi": f"{FISHER_BAND[1]:g}",
        "McmcFisherSources": thousands(len(records)),
        "McmcFisherHighSnr": f"{HIGH_SNR:.0f}",
        "McmcFisherHighCount": str(int(high.sum())),
        "McmcFisherHighInside": f"{100 * in_band(ratio_all[high][:, others]).mean():.0f}",
        "McmcFisherHighMedianLo": f"{np.median(ratio_all[high][:, others], axis=0).min():.2f}",
        "McmcFisherHighMedianHi": f"{np.median(ratio_all[high][:, others], axis=0).max():.2f}",
        "McmcFisherMidSnr": f"{MID_SNR:.0f}",
        "McmcFisherMidCount": str(int(mid.sum())),
        "McmcFisherMidInside": f"{100 * in_band(ratio_all[mid][:, others]).mean():.0f}",
        "McmcFisherFaintSnr": f"{FAINT_SNR:.0f}",
        "McmcFisherFaintFzero": f"{np.median(ratio_all[faint, 0]):.2f}",
        "McmcFisherFullMidLo": f"{min(full_mid_f0):.1f}",
        "McmcFisherFullMidHi": f"{max(full_mid_f0):.1f}",
        "McmcFisherFullMidDrift": f"{full_mid_drift:.1f}",
        # cost: 400 000-iteration joint chains on TREX (import_mcmc_cost.py); the narrow-band range quoted in the main text
        "McmcCostJointLo": f"{min(cost['narrow-4']['hours'] + cost['narrow-8']['hours']):.1f}",
        "McmcCostJointHi": f"{max(cost['narrow-4']['hours'] + cost['narrow-8']['hours']):.1f}",
        # cost: 400 000-iteration joint chains on TREX (import_mcmc_cost.py)
        "McmcCostNarrowFourLo": f"{min(cost['narrow-4']['hours']):.1f}",
        "McmcCostNarrowFourHi": f"{max(cost['narrow-4']['hours']):.1f}",
        "McmcCostNarrowEightLo": f"{min(cost['narrow-8']['hours']):.1f}",
        "McmcCostNarrowEightHi": f"{max(cost['narrow-8']['hours']):.1f}",
        "McmcCostFullSixteenLo": f"{min(cost['full-16']['hours'] or cost['full-16']['extrapolated_400k_hours']):.0f}",
        "McmcCostFullSixteenHi": f"{max(cost['full-16']['hours'] or cost['full-16']['extrapolated_400k_hours']):.0f}",
        **labelfree_numbers,
    },
)
