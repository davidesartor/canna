"""The numbers in appendix/mcmc.tex: the joint reference sampler's defaults and the MCMC reports' data."""

import ast
import inspect
import json
import math
import re

import numpy as np

import style
from canna.lisa import config
from canna.lisa import mcmc
from canna.lisa.injections import N_INJECTIONS
from canna.lisa.problem import LisaGB

REPORTS = style.REPO / "reports"
JOINT = REPORTS / "mcmc"
CONVERGENCE = REPORTS / "mcmc/data/convergence-261113"  # updated 2026-11-16: narrow-4 long chains, catalogue prior
FISHER_BAND = (0.8, 1.25)
HIGH_SNR, LOW_SNR = 100.0, 40.0
SEED_TOL = 0.03
# the injection drawn in fig_mcmc-posterior.py
POSTERIOR_INJECTION = 2
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
# the current problem's chains: older problems' folders may sit beside them
joint_dir = mcmc.cache_dir(problem, cfg.seed, JOINT / "data")
pid = joint_dir.name.removeprefix("joint-")
run = argparse_defaults("src/canna/lisa/mcmc.py")
cores = int(
    re.search(
        r"--cpus-per-task=(\d+)", (style.REPO / "slurm/lisa-mcmc.sbatch").read_text()
    )[1]
)
n_injections = N_INJECTIONS
print(f"problem {pid}; sampler {run}; cores {cores}; {n_injections} eval injections")

# the joint chains' widths against Fisher, per source of every injection with chains; widths in units of draws.json's scale
draws = json.loads((JOINT / "data/draws.json").read_text())
columns = draws["columns"]
assert len(columns) == mcmc.N_PARAMS
injections = {w["j"]: w for w in draws["windows"] if w["joint_width"]}
records = []
for j, w in injections.items():
    for s, snr_s in enumerate(w["snr"]):
        seeds = np.array([seed[s] for seed in w["joint_width"]])
        joint_rel = seeds.mean(axis=0)
        records.append(
            dict(
                injection=j,
                source=s,
                snr=snr_s,
                joint=joint_rel * np.array(w["scale"][s]),
                fisher=np.array(w["fisher_width"][s]) / joint_rel,
                seed=seeds[1] / seeds[0] if len(seeds) == 2 else None,
            )
        )
for r in records:
    print(
        f"inj {r['injection']} s{r['source']} SNR {r['snr']:7.1f}  fisher/joint {np.round(r['fisher'], 2)}"
    )

collected = json.loads((joint_dir / "chains.json").read_text())
joint_runs = collected["joint"]
rhat_j = max(d["rhat_max"] for d in joint_runs)
ess_j = min(d["ess_min"] for d in joint_runs)
tau_j = max(d["tau_max"] for d in joint_runs)
seed_dev_j = np.abs(np.array([r["seed"] for r in records if r["seed"] is not None]) - 1)
print(
    f"{len(joint_runs)} joint chains: R-hat max {rhat_j:.5f}, ESS min {ess_j:.0f}, tau max {tau_j:.0f}, seed width dev {seed_dev_j.max():.3f}, {np.mean(seed_dev_j <= SEED_TOL):.3f} within {SEED_TOL}"
)

# Fisher against the joint MCMC
high = [r for r in records if r["snr"] >= HIGH_SNR]
low = [r for r in records if r["snr"] < LOW_SNR]
high_no_drift = np.array([np.delete(r["fisher"], 1) for r in high])
inside = (high_no_drift >= FISHER_BAND[0]) & (high_no_drift <= FISHER_BAND[1])
outliers = [
    (r["snr"], columns[c], r["fisher"][c], r["injection"], r["source"])
    for r in high
    for c in range(8)
    if c != 1 and not FISHER_BAND[0] <= r["fisher"][c] <= FISHER_BAND[1]
]
low_no_drift = np.array([np.delete(r["fisher"], 1) for r in low])
fisher_drift = np.array([r["fisher"][1] for r in records])
joint_drift = np.array([r["joint"][1] for r in records])
print(
    f"fisher: {len(high)} sources >= {HIGH_SNR}, {inside.sum()}/{inside.size} inside, outliers {outliers}"
)
print(
    f"fisher below {LOW_SNR}: {len(low)} sources, {low_no_drift.min():.2f}-{low_no_drift.max():.2f}; {columns[1]} {fisher_drift.min():.2f}-{fisher_drift.max():.2f}"
)
print(f"{columns[1]} width: MCMC {joint_drift.min():.3f}-{joint_drift.max():.3f}")

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

# cost: the joint chains' own wall times
joint_hours = np.array([c["seconds"] for c in joint_runs]) / 3600
print(f"cost: joint {joint_hours.min():.1f}-{joint_hours.max():.1f} h")

# the injection of the posterior figure: its faintest source puts part of its draws in another f0 mode
posterior = injections[POSTERIOR_INJECTION]
faint = int(np.argmin(posterior["snr"]))
faint_offsets = np.concatenate(
    [np.array(seed[faint]).T for seed in posterior["draws"]["joint"]]
) * (np.array(posterior["scale"][faint]) / draws["quant"])
far = np.abs(faint_offsets[:, 0]) > 1
print(
    f"posterior injection {POSTERIOR_INJECTION}: faint source {faint}, {far.mean():.3f} of f0 draws > 1 bin off"
)

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
        "McmcSeeds": WORDS[len(run["seeds"])],
        "McmcInjections": str(n_injections),
        "McmcChains": str(n_injections * len(run["seeds"])),
        "McmcKeepDraws": thousands(mcmc.N_KEEP),
        "McmcSlots": WORDS[problem.n_sources],
        "McmcResponsePoints": str(problem.response_points),
        "McmcResponsePointsFull": thousands(full.response_points),
        "McmcJointCores": WORDS[cores],
        "McmcJointDim": str(problem.n_sources * mcmc.N_PARAMS),
        "McmcBoxSigmas": f"{mcmc.BOX_SIGMAS:g}",
        "McmcFdotSigmas": f"{mcmc.FDOT_SIGMAS:g}",
        "McmcFdotCutMass": f"{0.5 * math.erfc(mcmc.FDOT_SIGMAS / math.sqrt(2)):.1g}",
        # the joint chains on disk
        "McmcDrift": TEX_NAMES[columns[1]],
        "McmcJointWindows": (
            WORDS[len(injections)]
            if len(injections) < len(WORDS)
            else str(len(injections))
        ),
        "McmcJointChains": str(len(joint_runs)),
        "McmcJointWindowList": ", ".join(
            str(j) for j in sorted(injections, reverse=True)
        ),
        # convergence of the joint chains
        "McmcRhatGroups": WORDS[
            inspect.signature(mcmc.gelman_rubin).parameters["groups"].default
        ],
        "McmcJointRhatMax": f"{rhat_j:.4f}",
        "McmcJointEssMin": thousands(ess_j),
        "McmcJointTauMax": f"{tau_j:.0f}",
        "McmcJointSeedWidthTypical": f"{100 * SEED_TOL:.0f}",
        "McmcJointSeedShare": f"{100 * np.mean(seed_dev_j <= SEED_TOL):.0f}",
        "McmcJointSeedWidthDev": f"{100 * seed_dev_j.max():.1f}",
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
        "McmcPosteriorWindowSnr": f"{posterior['window_snr']:.0f}",
        "McmcPosteriorFaintSnr": f"{posterior['snr'][faint]:.1f}",
        "McmcPosteriorFarPercent": f"{100 * far.mean():.0f}",
        "McmcPosteriorFarBins": f"{np.median(faint_offsets[far, 0]):.1f}",
        "McmcPosteriorDecShift": f"{np.median(faint_offsets[:, 4]):.1f}",
        # Fisher against the joint MCMC
        "McmcFisherLo": f"{FISHER_BAND[0]:g}",
        "McmcFisherHi": f"{FISHER_BAND[1]:g}",
        "McmcFisherHighSnr": f"{HIGH_SNR:.0f}",
        "McmcFisherHighCount": str(len(high)),
        "McmcFisherInside": str(int(inside.sum())),
        "McmcFisherTotal": str(inside.size),
        "McmcFisherOutliers": "; ".join(
            f"{TEX_NAMES[c]} of source {s + 1} (\\acs{{snr}} {snr_:.0f}) of injection {j} ({v:.2f})"
            for snr_, c, v, j, s in sorted(outliers, key=lambda o: o[2])
        ),
        "McmcFisherLowSnr": f"{LOW_SNR:.0f}",
        "McmcFisherLowCount": str(len(low)),
        "McmcFisherLowLo": f"{low_no_drift.min():.2f}",
        "McmcFisherLowHi": f"{low_no_drift.max():.2f}",
        "McmcFisherDriftLo": f"{fisher_drift.min():.2f}",
        "McmcFisherDriftHi": f"{fisher_drift.max():.2f}",
        "McmcDriftWidthLo": f"{joint_drift.min():.2f}",
        "McmcDriftWidthHi": f"{joint_drift.max():.2f}",
        # cost
        "McmcCostJointLo": f"{joint_hours.min():.1f}",
        "McmcCostJointHi": f"{joint_hours.max():.1f}",
        **labelfree_numbers,
    },
)
