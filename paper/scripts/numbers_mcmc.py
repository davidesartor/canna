"""The numbers in appendix/mcmc.tex: sampler defaults, the chain cache and the MCMC reports' data."""

import ast
import inspect
import json
import re

import jax
import jax.random as jr
import numpy as np

import style
from canna.lisa import config
from canna.lisa import mcmc
from canna.lisa.injections import N_QUANTILES
from canna.lisa.physics import sample_chirp_mass
from canna.lisa.problem import LisaGB

REPORTS = style.REPO / "reports"
JOINT = REPORTS / "mcmc"
COMPARE = REPORTS / "posterior-eval/flow-vs-reference/data"
CONVERGENCE = REPORTS / "mcmc/data/convergence"
COST = REPORTS / "benchmarks/sampling-cost/data"
PRIOR = REPORTS / "prior/data/prior.json"
FISHER_BAND = (0.8, 1.25)
HIGH_SNR, LOW_SNR = 100.0, 40.0
NARROW_TOL = 0.05
NARROW_BAND = (0.9, 1.1)
# the joint window drawn in fig_mcmc-posterior.py
POSTERIOR_WINDOW = 2
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
TEX_NAMES = [
    r"$f_0$",
    r"$\ln\Mc$",
    r"$\ln\Amp$",
    r"$\alpha$",
    r"$\delta$",
    r"$\psi$",
    r"$\cos\iota$",
    r"$\phi_0$",
]


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


def width(a: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=0)), axis=0)


def tex_param(name: str) -> str:
    """Convergence-trace name "s<k> <column>" -> "<tex symbol> of source <k + 1>"."""
    slot, column = name.split(" ", 1)
    return f"{TEX_NAMES[mcmc.COLUMNS.index(column)]} of source {int(slot[1:]) + 1}"


cfg = config.load("narrow-B")
problem = LisaGB(**cfg.problem)
full = LisaGB(**config.load("full-B").problem)
pid = mcmc.problem_id(problem, cfg.seed)
narrow = argparse_defaults("src/canna/lisa/mcmc.py")
joint = argparse_defaults(JOINT / "scripts/joint_mcmc.py")
cores = int(
    re.search(
        r"--cpus-per-task=(\d+)", (style.REPO / "slurm/lisa-mcmc.sbatch").read_text()
    )[1]
)
joint_cores = int(
    re.search(
        r"--cpus-per-task=(\d+)", (JOINT / "scripts/joint_mcmc.sbatch").read_text()
    )[1]
)
print(f"problem {pid}; narrowband {narrow}; joint {joint}; cores {cores}/{joint_cores}")

# narrowband chains: convergence per chain, seed-to-seed widths
collected = json.loads((JOINT / "data" / f"joint-{pid}" / "chains.json").read_text())
chains = collected["narrowband"]
nb = {}
for c in chains:
    nb.setdefault((c["inj"], c["src"]), {})[c["seed"]] = c
nb_pooled = {
    (c["inj"], c["src"]): np.array(c["width"]) for c in collected["narrowband_pooled"]
}
rhat_nb = max(c["rhat_max"] for c in chains)
ess_nb = min(c["ess_min"] for c in chains)
tau_nb = max(c["tau_max"] for c in chains)
seed_dev_nb = max(
    np.abs(np.array(v[1]["width"]) / np.array(v[0]["width"]) - 1).max()
    for v in nb.values()
    if len(v) == 2
)
nb_snr = np.array([v[0]["snr"] for v in nb.values()])
print(
    f"{len(chains)} narrowband chains, {len(nb)} sources, SNR {nb_snr.min():.1f}-{nb_snr.max():.1f}"
)
print(
    f"narrowband R-hat max {rhat_nb:.5f}, ESS min {ess_nb:.0f}, tau max {tau_nb:.0f}, seed width dev {seed_dev_nb:.3f}"
)

# joint chains against narrowband and Fisher, per source of the four windows
compares = {
    int(re.search(r"window(\d+)", p.name)[1]): np.load(p)
    for p in sorted((COMPARE / f"joint-{pid}").glob("compare_window*.npz"))
}
rows = list(next(iter(compares.values()))["rows"])
i_mcmc, i_fisher, i_seed = (
    rows.index("mcmc width"),
    rows.index("fisher/mcmc width"),
    rows.index("seed1/seed0 width"),
)
records = []
for j, c in compares.items():
    for s in range(len(c["snr"])):
        nb_width = nb_pooled[(j, s)]
        records.append(
            dict(
                window=j,
                source=s,
                snr=float(c["snr"][s]),
                window_snr=float(c["window_snr"]),
                joint=c[f"s{s}"][i_mcmc],
                fisher=c[f"s{s}"][i_fisher],
                seed=c[f"s{s}"][i_seed],
                narrow=nb_width / c[f"s{s}"][i_mcmc],
            )
        )
for r in records:
    print(
        f"w{r['window']} s{r['source']} SNR {r['snr']:7.1f}  fisher/joint {np.round(r['fisher'], 2)}  narrow/joint {np.round(r['narrow'], 3)}"
    )

joint_runs = collected["joint"]
rhat_j = max(d["rhat_max"] for d in joint_runs)
ess_j = min(d["ess_min"] for d in joint_runs)
tau_j = max(d["tau_max"] for d in joint_runs)
seed_dev_j = max(np.abs(r["seed"] - 1).max() for r in records)
print(
    f"{len(joint_runs)} joint chains: R-hat max {rhat_j:.5f}, ESS min {ess_j:.0f}, tau max {tau_j:.0f}, seed width dev {seed_dev_j:.3f}"
)

narrow_all = np.array([r["narrow"] for r in records])
narrow_no_mc = np.delete(narrow_all, 1, axis=1)
narrow_snr = np.array([r["snr"] for r in records])
near = np.abs(narrow_no_mc - 1) <= NARROW_TOL
within = (narrow_no_mc >= NARROW_BAND[0]) & (narrow_no_mc <= NARROW_BAND[1])
# the sources with a ratio outside NARROW_BAND, each by its worst column, worst first
worst = [
    max((c for c in range(8) if c != 1), key=lambda c: abs(np.log(r["narrow"][c])))
    for r in records
]
wide = sorted(
    ((r, c) for r, c, ok in zip(records, worst, within.all(axis=1)) if not ok),
    key=lambda rc: -abs(np.log(rc[0]["narrow"][rc[1]])),
)
print(
    f"narrow/joint: all {narrow_all.min():.3f}-{narrow_all.max():.3f}, no log Mc {narrow_no_mc.min():.3f}-{narrow_no_mc.max():.3f}, median {np.median(narrow_no_mc):.4f}, {within.mean():.3f} in {NARROW_BAND}"
)
print(
    "narrow/joint outside band: "
    + "; ".join(
        f"w{r['window']} s{r['source']} SNR {r['snr']:.1f} {mcmc.COLUMNS[c]} {r['narrow'][c]:.3f}"
        for r, c in wide
    )
)
next_wide, next_column = wide[1]

# Fisher against the joint MCMC
high = [r for r in records if r["snr"] >= HIGH_SNR]
low = [r for r in records if r["snr"] < LOW_SNR]
high_no_mc = np.array([np.delete(r["fisher"], 1) for r in high])
inside = (high_no_mc >= FISHER_BAND[0]) & (high_no_mc <= FISHER_BAND[1])
outliers = [
    (r["snr"], mcmc.COLUMNS[c], r["fisher"][c], r["window"], r["source"])
    for r in high
    for c in range(8)
    if c != 1 and not FISHER_BAND[0] <= r["fisher"][c] <= FISHER_BAND[1]
]
low_no_mc = np.array([np.delete(r["fisher"], 1) for r in low])
fisher_mc = np.array([r["fisher"][1] for r in records])
joint_mc = np.array([r["joint"][1] for r in records])
print(
    f"fisher: {len(high)} sources >= {HIGH_SNR}, {inside.sum()}/{inside.size} inside, outliers {outliers}"
)
print(
    f"fisher below {LOW_SNR}: {len(low)} sources, {low_no_mc.min():.2f}-{low_no_mc.max():.2f}; log Mc {fisher_mc.min():.2f}-{fisher_mc.max():.2f}"
)

# Fisher against the narrowband MCMC of every window, widths in units of the pooled narrowband width
draws = json.loads((JOINT / "data/draws.json").read_text())
nb_fisher = [
    (snr, np.delete(np.array(w["fisher_width"][s]), 1))
    for w in draws["windows"]
    for s, snr in enumerate(w["snr"])
    if snr >= LOW_SNR
]
nb_fisher_all = np.concatenate([r for _, r in nb_fisher])
nb_inside = (nb_fisher_all >= FISHER_BAND[0]) & (nb_fisher_all <= FISHER_BAND[1])
# sources whose forecast misses some width by more than a factor of two
nb_miss = [
    (snr, np.maximum(r, 1 / r).max())
    for snr, r in nb_fisher
    if np.maximum(r, 1 / r).max() > 2
]
nb_miss_snr = [snr for snr, _ in nb_miss]
print(
    f"fisher/narrow above {LOW_SNR}: {len(nb_fisher)} sources, {nb_inside.mean():.3f} inside; misses > 2x {nb_miss}"
)

# the chirp-mass prior's own robust width, against the MCMC's
log_mc_prior = np.log(
    np.asarray(
        jax.vmap(lambda k: sample_chirp_mass(k, problem.mass_range))(
            jr.split(jr.key(0), 200_000)
        )
    )
).ravel()
prior_mc_width = float(width(log_mc_prior))
print(
    f"log Mc width: MCMC {joint_mc.min():.3f}-{joint_mc.max():.3f}, prior {prior_mc_width:.3f}"
)

# convergence study of the joint sampler
summary = json.loads((CONVERGENCE / "summary.json").read_text())
curves = np.load(CONVERGENCE / "curves.npz")
crit, conv_chain = summary["criterion"], summary["chain"]
windows = summary["windows"]
paper = {j: w["settings"]["paper"] for j, w in windows.items()}
passing = [int(j) for j, p in paper.items() if p["passes"]]
failing = [int(j) for j, p in paper.items() if not p["passes"]]


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


first_excess = {j: segment_excess(j, 0) for j in windows}
post_burn = {j: segment_excess(j, paper[j]["burn"]) for j in windows}
print(
    f"convergence: passes {passing}, fails {failing}; first segment {first_excess}; after burn {post_burn}"
)
taus = [w["settings"]["paper"]["tau_max"] for w in windows.values()]

# cost: the chains' own wall times
cost = json.loads((COST / "mcmc.json").read_text())
nb_seconds = np.array(
    [c["seconds"] for c in cost["narrowband"] if c["cache"] == f"eval-{pid}"]
)
per_window = {}
for c in cost["narrowband"]:
    if c["cache"] == f"eval-{pid}":
        per_window[(c["inj"], c["seed"])] = (
            per_window.get((c["inj"], c["seed"]), 0.0) + c["seconds"]
        )
window_minutes = np.array(list(per_window.values())) / 60
joint_hours = np.array([c["seconds"] for c in cost["joint"]]) / 3600
print(
    f"cost: narrowband {nb_seconds.min() / 60:.1f}-{nb_seconds.max() / 60:.1f} min, window {window_minutes.min():.0f}-{window_minutes.max():.0f} min, joint {joint_hours.min():.1f}-{joint_hours.max():.1f} h"
)

mc_check = json.loads(PRIOR.read_text())["bands"]["narrow"]["chirp_mass_check"]
p9, p7 = windows["9"], windows["7"]
w2 = windows["2"]

# the window of the posterior figure: its faintest source puts part of its draws in another f0 mode
posterior = compares[POSTERIOR_WINDOW]
faint = int(np.argmin(posterior["snr"]))
faint_offsets = posterior[f"s{faint}_mcmc_offsets"].astype(np.float64)
far = np.abs(faint_offsets[:, 0]) > 1
print(
    f"posterior window {POSTERIOR_WINDOW}: faint source {faint}, {far.mean():.3f} of f0 draws > 1 bin off"
)

style.save_numbers(
    "mcmc",
    {
        # sampler settings
        "McmcWalkers": str(narrow["nwalker"]),
        "McmcTemps": WORDS[len(narrow["temps"])],
        "McmcTempMax": f"{max(narrow['temps']):g}",
        "McmcBurn": thousands(narrow["burn"]),
        "McmcKeep": thousands(narrow["keep"]),
        "McmcThin": str(narrow["thin"]),
        "McmcSeeds": WORDS[len(narrow["seeds"])],
        "McmcChains": str(len(chains)),
        "McmcSources": str(len(nb)),
        "McmcWindows": WORDS[N_QUANTILES],
        "McmcSlots": WORDS[problem.n_sources],
        "McmcSnrMin": f"{problem.snr_range[0]:g}",
        "McmcSnrMax": thousands(problem.snr_range[1]),
        "McmcResponsePoints": str(problem.response_points),
        "McmcResponsePointsFull": thousands(full.response_points),
        "McmcCores": WORDS[cores],
        "McmcJointCores": WORDS[joint_cores],
        "McmcMcCheckDraws": thousands(mc_check["draws"]),
        "McmcMcCheckBins": str(mc_check["bins"]),
        "McmcMcCheckChi": f"{mc_check['chi2_per_dof']:.2f}",
        # joint sampler settings
        "McmcJointWalkers": str(joint["nwalker"]),
        "McmcJointBurn": thousands(joint["burn"]),
        "McmcJointKeep": thousands(joint["keep"]),
        "McmcJointThin": str(joint["thin"]),
        "McmcJointDim": str(problem.n_sources * mcmc.N_PARAMS),
        "McmcJointWindows": WORDS[len(compares)],
        "McmcJointChains": str(len(joint_runs)),
        "McmcJointSources": str(len(records)),
        "McmcJointWindowList": ", ".join(
            str(j) for j in sorted(compares, reverse=True)
        ),
        "McmcJointSnrLo": f"{min(r['window_snr'] for r in records):.0f}",
        "McmcJointSnrHi": thousands(max(r["window_snr"] for r in records)),
        "McmcJointSrcSnrLo": f"{min(r['snr'] for r in records):.1f}",
        "McmcJointSrcSnrHi": f"{max(r['snr'] for r in records):.0f}",
        # convergence of the narrowband and joint chains
        "McmcRhatGroups": WORDS[
            inspect.signature(mcmc.gelman_rubin).parameters["groups"].default
        ],
        "McmcRhatMax": f"{rhat_nb:.4f}",
        "McmcEssMin": thousands(ess_nb),
        "McmcTauMax": f"{tau_nb:.0f}",
        "McmcSeedWidthDev": f"{100 * seed_dev_nb:.1f}",
        "McmcJointRhatMax": f"{rhat_j:.4f}",
        "McmcJointEssMin": thousands(ess_j),
        "McmcJointTauMax": f"{tau_j:.0f}",
        "McmcJointSeedWidthDev": f"{100 * seed_dev_j:.1f}",
        "McmcNarrowJointLo": f"{narrow_no_mc.min():.2f}",
        "McmcNarrowJointHi": f"{narrow_no_mc.max():.2f}",
        "McmcNarrowJointMcLo": f"{narrow_all[:, 1].min():.2f}",
        "McmcNarrowJointMcHi": f"{narrow_all[:, 1].max():.2f}",
        "McmcNarrowJointTol": f"{100 * NARROW_TOL:.0f}",
        "McmcNarrowJointNear": str(int(near.sum())),
        "McmcNarrowJointCount": str(near.size),
        "McmcNarrowJointBelow": str(int((narrow_no_mc < 1 - NARROW_TOL).sum())),
        "McmcNarrowJointAbove": str(int((narrow_no_mc > 1 + NARROW_TOL).sum())),
        "McmcNarrowJointMedian": f"{np.median(narrow_no_mc):.3f}",
        "McmcNarrowJointBandLo": f"{NARROW_BAND[0]:g}",
        "McmcNarrowJointBandHi": f"{NARROW_BAND[1]:g}",
        "McmcNarrowJointWithin": f"{100 * within.mean():.0f}",
        "McmcNarrowJointWideSources": WORDS[len(wide)],
        "McmcNarrowJointNextParam": TEX_NAMES[next_column],
        "McmcNarrowJointNextSnr": f"{next_wide['snr']:.0f}",
        "McmcNarrowJointNextWindow": str(next_wide["window"]),
        "McmcNarrowJointNextRatio": f"{next_wide['narrow'][next_column]:.2f}",
        "McmcNarrowJointLoSnr": f"{narrow_snr[np.unravel_index(narrow_no_mc.argmin(), narrow_no_mc.shape)[0]]:.1f}",
        # convergence study
        "McmcConvIterations": thousands(conv_chain["iterations"]),
        "McmcConvSeeds": WORDS[len(p9["seeds"])],
        "McmcConvThin": str(conv_chain["thin"]),
        "McmcConvStep": thousands(p9["step"]),
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
        "McmcConvPassing": " and ".join(str(j) for j in passing),
        "McmcConvFailing": " and ".join(str(j) for j in failing),
        "McmcConvBurnNine": thousands(p9["burn"]),
        "McmcConvBurnSeven": thousands(p7["burn"]),
        "McmcConvWindowTwoFirst": f"{first_excess['2'][0]:.1f}",
        "McmcConvWindowTwoAfter": f"{post_burn['2'][0]:.1f}",
        "McmcConvWindowFiveAfter": f"{post_burn['5'][0]:.2f}",
        "McmcConvWindowFiveParam": tex_param(post_burn["5"][1]),
        "McmcConvWindowTwoSnr": f"{w2['snr'][0]:.0f}",
        # the joint posterior figure
        "McmcPosteriorWindow": str(POSTERIOR_WINDOW),
        "McmcPosteriorWindowSnr": f"{float(posterior['window_snr']):.0f}",
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
            f"{TEX_NAMES[mcmc.COLUMNS.index(c)]} of source {s + 1} (\\acs{{snr}} {snr_:.0f}) of window {w} ({v:.2f})"
            for snr_, c, v, w, s in sorted(outliers, key=lambda o: o[2])
        ),
        "McmcFisherLowSnr": f"{LOW_SNR:.0f}",
        "McmcFisherLowCount": str(len(low)),
        "McmcFisherLowLo": f"{low_no_mc.min():.2f}",
        "McmcFisherLowHi": f"{low_no_mc.max():.2f}",
        "McmcFisherNarrowSources": str(len(nb_fisher)),
        "McmcFisherNarrowInside": f"{100 * nb_inside.mean():.0f}",
        "McmcFisherNarrowMisses": WORDS[len(nb_miss)],
        "McmcFisherNarrowMissSnrLo": f"{min(nb_miss_snr):.0f}",
        "McmcFisherNarrowMissSnrHi": f"{max(nb_miss_snr):.0f}",
        "McmcFisherNarrowMissMax": f"{max(m for _, m in nb_miss):.1f}",
        "McmcFisherMcLo": f"{fisher_mc.min():.2f}",
        "McmcFisherMcHi": f"{fisher_mc.max():.2f}",
        "McmcMcWidthLo": f"{joint_mc.min():.2f}",
        "McmcMcWidthHi": f"{joint_mc.max():.2f}",
        "McmcMcPriorWidth": f"{prior_mc_width:.2f}",
        # cost
        "McmcCostSourceLo": f"{nb_seconds.min() / 60:.1f}",
        "McmcCostSourceHi": f"{nb_seconds.max() / 60:.1f}",
        "McmcCostWindowLo": f"{window_minutes.min():.0f}",
        "McmcCostWindowHi": f"{window_minutes.max():.0f}",
        "McmcCostJointLo": f"{joint_hours.min():.1f}",
        "McmcCostJointHi": f"{joint_hours.max():.1f}",
    },
)
