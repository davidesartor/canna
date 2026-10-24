"""mcmc.html from data/: joint against narrowband chains, long-chain burn-in, and the Fisher forecast."""

import datetime
import json
from pathlib import Path

import numpy as np

REPORT = Path(__file__).parents[1]
DATA = REPORT / "data"
AGREE = (0.8, 1.25)
FAINT_SNR = 40


def sig(a, digits: int = 4):
    """Nested lists of `digits` significant figures, NaN as null."""
    a = np.asarray(a, np.float64)
    flat = [None if not np.isfinite(v) else float(f"{v:.{digits}g}") for v in a.ravel()]
    return np.array(flat, dtype=object).reshape(a.shape).tolist()


def curves(path: Path) -> dict:
    z = np.load(path)
    out = {}
    for key in z.files:
        j, name = key[1:].split("_", 1)
        value = z[key]
        out.setdefault(j, {})[name] = (
            value.tolist() if value.dtype.kind in "iU" else sig(value)
        )
    return out


def summarise_agreement(draws: dict, chains: dict) -> dict:
    """Chain health, seed agreement and joint over narrowband widths."""
    joint_chains = chains["joint"]
    narrow_chains = chains["narrowband"]
    out = dict(
        joint_rhat_max=max(c["rhat_max"] for c in joint_chains),
        joint_ess_min=min(c["ess_min"] for c in joint_chains),
        joint_tau_max=max(c["tau_max"] for c in joint_chains),
        narrow_rhat_max=max(c["rhat_max"] for c in narrow_chains),
        narrow_ess_min=min(c["ess_min"] for c in narrow_chains),
        narrow_tau_max=max(c["tau_max"] for c in narrow_chains),
        narrow_chains=len(narrow_chains),
        joint_chain_count=len(joint_chains),
    )

    # seed agreement and the joint-over-narrowband width, per source-parameter
    joint_seed, narrow_seed, joint_over_narrow, rows = [], [], [], []
    for w in draws["windows"]:
        nw = np.array(w["narrow_width"])
        narrow_seed.append(np.abs(nw[1] / nw[0] - 1).ravel())
        if w["joint_width"]:
            jw = np.array(w["joint_width"])
            joint_seed.append(np.abs(jw[1] / jw[0] - 1).ravel())
            mean = jw.mean(0)
            joint_over_narrow.append(mean.ravel())
            for s, snr in enumerate(w["snr"]):
                rows.append((w["j"], s, snr, mean[s]))
    joint_seed, narrow_seed = np.concatenate(joint_seed), np.concatenate(narrow_seed)
    ratio = np.concatenate(joint_over_narrow)
    out |= dict(
        joint_seed_within3=float((joint_seed <= 0.03).mean()),
        joint_seed_max=float(joint_seed.max()),
        narrow_seed_within3=float((narrow_seed <= 0.03).mean()),
        narrow_seed_max=float(narrow_seed.max()),
        joint_over_narrow_count=int(ratio.size),
        joint_over_narrow_median=float(np.median(ratio)),
        joint_over_narrow_min=float(ratio.min()),
        joint_over_narrow_max=float(ratio.max()),
        joint_over_narrow_within=float(((ratio >= 0.9) & (ratio <= 1.1)).mean()),
        joint_over_narrow_wide=[
            dict(
                window=j,
                source=s,
                snr=snr,
                parameter=draws["columns"][p],
                ratio=float(r[p]),
            )
            for j, s, snr, r in rows
            for p in range(8)
            if r[p] > 1.1 or r[p] < 0.9
        ],
    )
    return out


def summarise_convergence(summary: dict, z) -> dict:
    """Burn-in and keep checks of the long chains."""
    windows = summary["windows"]
    paper = {j: w["settings"]["paper"] for j, w in windows.items()}

    # the slowest series at the paper's burn-in, per window
    slowest = {}
    for j, w in windows.items():
        at = list(z[f"w{j}_burn_burn"]).index(paper[j]["burn"])
        slowest[j] = str(z[f"w{j}_names"][int(np.nanargmax(z[f"w{j}_burn_tau"][at]))])

    taus = [s["tau_max"] for w in windows.values() for s in w["settings"].values()]
    taus += [w["tau_max"] for w in windows.values() if "tau_max" in w]
    return dict(
        paper_passes=[j for j, p in paper.items() if p["passes"]],
        paper_ess_min=min(p["ess_min"] for p in paper.values()),
        paper_rhat_minus_one_max=max(p["rhat_max"] for p in paper.values()) - 1,
        paper_median_drift_max=max(p["median_drift_max"] for p in paper.values()),
        burn={j: w["burn"] for j, w in windows.items()},
        fml_passes=[j for j, w in windows.items() if w["settings"]["fml"]["passes"]],
        recommended=summary["recommended"],
        tau_max_range=[min(taus), max(taus)],
        trace_thin=summary["chain"]["thin"],
        slowest_parameter=slowest,
    )


def summarise_fisher(windows: list, diagonal: dict) -> dict:
    """Fisher widths over the MCMC's, and where its Hessian leans on the prior."""
    fisher_narrow, fisher_joint = [], []
    for w in windows:
        fw = np.array(w["fisher_width"])
        for s, snr in enumerate(w["snr"]):
            fisher_narrow.append((snr, fw[s]))
        if w["joint_width"]:
            mean = np.array(w["joint_width"]).mean(0)
            for s, snr in enumerate(w["snr"]):
                fisher_joint.append((snr, fw[s] / mean[s]))

    # Fisher over MCMC, loud and faint, without the prior-dominated log Mc
    out = {}
    keep = [p for p in range(8) if p != 1]
    for name, pairs in (("narrow", fisher_narrow), ("joint", fisher_joint)):
        loud = np.concatenate([r[keep] for snr, r in pairs if snr >= FAINT_SNR])
        faint = np.concatenate([r[keep] for snr, r in pairs if snr < FAINT_SNR])
        mc = np.array([r[1] for snr, r in pairs])
        out |= {
            f"fisher_{name}_loud_sources": sum(snr >= FAINT_SNR for snr, _ in pairs),
            f"fisher_{name}_loud_agree": float(
                ((loud >= AGREE[0]) & (loud <= AGREE[1])).mean()
            ),
            f"fisher_{name}_loud_range": [float(loud.min()), float(loud.max())],
            f"fisher_{name}_faint_range": [float(faint.min()), float(faint.max())],
            f"fisher_{name}_mc_range": [float(mc.min()), float(mc.max())],
        }

    # Fisher f0 over the ideal sqrt(3)/(pi rho), narrow band
    f0_ratio = np.concatenate(
        [
            np.array(w["f0_width_bins"], float) / np.array(w["ideal_width_bins"], float)
            for w in diagonal["bands"]["narrow"]["injections"]
        ]
    )
    out |= dict(
        fisher_f0_over_ideal_median=float(np.median(f0_ratio)),
        fisher_f0_over_ideal_max=float(f0_ratio.max()),
    )

    # non-positive averaged Hessian diagonals and floored eigenvalues, both bands
    injections = [i for band in diagonal["bands"].values() for i in band["injections"]]
    hessian = np.array([i["log10_hessian_diag"] for i in injections], float)
    nonpositive = ~np.isfinite(hessian.reshape(len(injections), 4, 8))
    out |= dict(
        sources=4 * len(injections),
        injections=len(injections),
        nonpositive_sources=int(nonpositive.any(-1).sum()),
        nonpositive_by_parameter=dict(
            zip(diagonal["params"], nonpositive.sum((0, 1)).tolist())
        ),
        floored_injections=sum(i["floored_eigenvalues"] > 0 for i in injections),
    )
    return out


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCMC reference</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --ref: #52514e;
  --s3: #1b9a6a; --s4: #8a55d6;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --ref: #c3c2b7;
    --s3: #2fb383; --s4: #a07ce8;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --ref: #c3c2b7;
  --s3: #2fb383; --s4: #a07ce8;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
a { color: var(--s1); }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 24px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; }
.layout { display: grid; grid-template-columns: 290px minmax(0, 1fr); gap: 20px; align-items: start; }
.side { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto; }
.content { min-width: 0; }
.content > .card + .card { margin-top: 16px; }
@media (max-width: 900px) {
  .layout { grid-template-columns: minmax(0, 1fr); }
  .side { position: static; max-height: none; }
}
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: auto; display: inline-block; flex: none; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; margin: 0 0 16px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 12px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
button.plain { font: inherit; font-size: 13px; padding: 5px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--ink-2); cursor: pointer; }
.flowrow { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
.stack { margin-bottom: 14px; }
.stack > .k { display: block; color: var(--ink-2); font-size: 13px; margin-bottom: 6px; }
.seg.grid { display: grid; width: 100%; grid-template-columns: repeat(var(--cols), minmax(0, 1fr)); gap: 1px; background: var(--border); }
.seg.grid button { background: var(--surface); padding: 4px 6px; }
.seg.grid button + button { border-left: 0; }
.seg.grid button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.seg .sn { display: block; font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
.seg button.joint .sn { color: var(--s1); font-weight: 600; }
.seg button[aria-pressed="true"] .sn { color: inherit; opacity: 0.8; }
.banner { background: #c62828; color: #fff; padding: 8px 16px; font: 13px/1.4 system-ui, sans-serif; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 500; }
tr.pick { cursor: pointer; }
tr.sel td { background: var(--grid); }
#p-corner { flex: 1 1 460px; min-width: 0; }
#p-pair { flex: 1 1 300px; min-width: 0; }
#p-corner canvas { display: block; cursor: crosshair; }
#p-trace { flex: 1 1 480px; min-width: 0; }
#p-acf { flex: 0 1 260px; min-width: 0; }
ul.means { padding-left: 20px; margin: 8px 0 0; max-width: 80ch; }
ul.means li { margin-bottom: 8px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
</style>
</head>
<body>
<div id="banner" class="banner" hidden></div>
<script>
window.onerror = (message, source, line) => {
  const b = document.getElementById("banner");
  b.hidden = false;
  b.textContent = `This page hit an error: ${message} (line ${line}). Some cards may be blank.`;
};
</script>
<main>
<h1>Can the MCMC reference be trusted?</h1>
<p class="sub" id="sub"></p>
<div class="layout">
<aside class="card side">
  <div class="stack"><span class="k">Window (window SNR; blue: joint and long chains)</span><div class="seg grid" id="window" style="--cols: 5"></div></div>
  <div class="stack"><span class="k">Source (SNR, loudest first)</span><div class="seg grid" id="source" style="--cols: 4"></div></div>
  <div class="stack"><span class="k">Parameter</span><div class="seg grid" id="param" style="--cols: 4"></div></div>
  <div class="stack"><span class="k">Paired with (corner)</span><div class="seg grid" id="pair" style="--cols: 4"></div></div>
  <p class="note">The parameter drives every card. Clicking a corner cell, a column, a point or a table row changes these pickers.</p>
</aside>
<div class="content">
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <h2>Do the joint and narrowband chains agree?</h2>
    <p class="note">Every parameter pair of the selected source, as offsets from the truth in its narrowband width (1.4826 × MAD of both narrowband seeds pooled; the cross is the truth). Dots are 400 draws per chain, dashed ellipses the Fisher forecast's 1σ and 2σ. Click a cell to enlarge it on the right.</p>
    <div class="controls"><span class="seg" id="sets"></span></div>
    <div class="legend" id="legend-a"></div>
    <div class="flowrow"><div id="p-corner"></div><div id="p-pair"></div></div>
    <p class="note" id="pair-note"></p>
    <p class="note">Below, each chain's width over the pooled narrowband width per parameter: the band is 0.8–1.25; seed against seed is the chains' own noise, joint against narrowband what freeing the other sources does. Under each column, the prior's width in the same unit.</p>
    <div class="legend" id="legend-b"></div>
    <div id="p-widths"></div>
  </div>
  <div class="card">
    <h2>Are the long chains past burn-in?</h2>
    <p class="note" id="note-d"></p>
    <div id="conv-body">
      <div class="controls"><span class="seg" id="series"></span><span class="seg" id="view"></span></div>
      <div class="legend" id="legend-d"></div>
      <div class="flowrow"><div id="p-trace"></div><div id="p-acf"></div></div>
      <p class="note" id="note-e"></p>
      <div class="controls"><span class="seg" id="conv"></span><button type="button" class="plain" id="worst"></button></div>
      <div id="p-conv"></div>
    </div>
    <div class="tablewrap" id="t-conv"></div>
  </div>
  <div class="card">
    <h2>How far does the Fisher forecast follow the MCMC?</h2>
    <p class="note" id="note-c"></p>
    <div class="controls"><span class="seg" id="against"></span></div>
    <div class="legend" id="legend-c"></div>
    <div id="p-fisher"></div>
    <p class="note" id="note-h"></p>
    <div class="controls"><span class="seg" id="band"></span></div>
    <div class="legend" id="legend-h"></div>
    <div id="p-hessian"></div>
    <div class="tablewrap" id="t-hessian"></div>
  </div>
  <div class="card">
    <h2>What this means</h2>
    <ul class="means" id="means"></ul>
  </div>
</div>
</div>
</main>
<script>const DATA = __DATA__;</script>
<script>
const NAMES = ["f₀", "log M<sub>c</sub>", "log A", "lon", "lat", "ψ", "sin ι", "φ₀"];
const PLAIN = ["f₀", "log Mc", "log A", "lon", "lat", "ψ", "sin ι", "φ₀"];
const UNITS = [" bins", "", "", " rad", " rad", " rad", "", " rad"];
const Q = DATA.draws.quant, QT = DATA.traces.quant, WINDOWS = DATA.draws.windows, JOINT = DATA.draws.joint_windows;
const LONG = Object.keys(DATA.convergence.windows).map(Number);
const SETS = [
  {key: "j0", text: "joint seed 0", colour: "--s1"},
  {key: "j1", text: "joint seed 1", colour: "--s3"},
  {key: "n", text: "narrowband", colour: "--s2"},
  {key: "f", text: "Fisher", colour: "--ref"},
];
const SEED_COLOURS = ["--s1", "--s2", "--s3", "--s4"];
const CHOICES = {series: ["param", "logl"], view: ["band", "walker"], conv: ["median", "width", "ess", "rhat"], against: ["joint", "narrow"], band: ["narrow", "full"]};
const S = {window: 9, source: 0, param: 3, pair: 4, sets: ["j0", "j1", "n", "f"], series: "param", view: "band", conv: "median", against: "joint", band: "narrow"};
const marks = {};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const fmt = (v, d = 2) => v === null || v === undefined || !isFinite(v) ? "–" : Number(v).toFixed(d);
const big = v => Math.round(v).toLocaleString("en");
const priorText = v => v >= 1e4 ? `${Math.round(v / 1e3)}k` : v >= 10 ? big(v) : (+v.toPrecision(2)).toString();
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function rgba(name, alpha) {
  const h = css(name).replace("#", ""), n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}
const W_ = () => WINDOWS[S.window];
const T_ = () => DATA.traces.windows[String(S.window)];
const hasJoint = j => JOINT.includes(j);
const series = () => S.series === "logl" ? 32 : S.source * 8 + S.param;
const seriesName = k => k === 32 ? "log L" : `source ${Math.floor(k / 8)} ${PLAIN[k % 8]}`;
const jointMean = (w, s, p) => w.joint_width.reduce((a, jw) => a + jw[s][p], 0) / w.joint_width.length;

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1)), int = (k, n) => { const v = parseInt(h.get(k), 10); if (Number.isInteger(v) && v >= 0 && v < n) S[k] = v; };
  int("window", WINDOWS.length); int("source", 4); int("param", 8); int("pair", 8);
  if (h.has("sets")) S.sets = h.get("sets").split(".").filter(k => SETS.some(s => s.key === k));
  for (const [k, allowed] of Object.entries(CHOICES)) if (allowed.includes(h.get(k))) S[k] = h.get(k);
  if (["light", "dark"].includes(h.get("theme"))) document.documentElement.dataset.theme = h.get("theme");
}
function writeHash() {
  const p = {window: S.window, source: S.source, param: S.param, pair: S.pair, sets: S.sets.join("."), series: S.series, view: S.view, conv: S.conv, against: S.against, band: S.band};
  if (document.documentElement.dataset.theme) p.theme = document.documentElement.dataset.theme;
  history.replaceState(null, "", "#" + new URLSearchParams(p).toString());
}
readHash();

function seg(id, items, key, after) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    S[key] = typeof S[key] === "number" ? Number(b.dataset.v) : b.dataset.v;
    if (after) after();
    marks[key]();
    schedule();
  }));
  marks[key]();
}
function pick(changes) {  // set state from a click inside a panel and re-mark the pickers
  Object.assign(S, changes);
  if ("window" in changes) sourceSeg();
  Object.values(marks).forEach(m => m());
  schedule();
}
function sourceSeg() {
  seg("source", W_().snr.map((snr, s) => [s, `s${s}<span class="sn">${snr < 100 ? snr.toFixed(1) : Math.round(snr)}</span>`]), "source");
}
seg("window", WINDOWS.map((w, j) => [j, `${j}<span class="sn">${Math.round(w.window_snr)}</span>`]), "window", sourceSeg);
document.querySelectorAll("#window button").forEach(b => { if (hasJoint(Number(b.dataset.v))) b.classList.add("joint"); });
sourceSeg();
seg("param", NAMES.map((n, p) => [p, n]), "param");
seg("pair", NAMES.map((n, p) => [p, n]), "pair");
seg("series", [["param", "parameter"], ["logl", "log L"]], "series");
seg("view", [["band", "16–84% of walkers"], ["walker", "one walker"]], "view");
seg("conv", [["median", "median drift"], ["width", "width drift"], ["ess", "ESS"], ["rhat", "R̂ − 1"]], "conv");
seg("against", [["joint", "against joint"], ["narrow", "against narrowband"]], "against");
seg("band", [["narrow", "narrow band"], ["full", "full band"]], "band");
(() => {
  const host = document.getElementById("sets");
  host.innerHTML = SETS.map(s => `<button type="button" data-v="${s.key}"><span class="sw" style="background: var(${s.colour})"></span> ${s.text}</button>`).join("");
  marks.sets = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(S.sets.includes(b.dataset.v))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    const k = b.dataset.v;
    S.sets = S.sets.includes(k) ? S.sets.filter(v => v !== k) : SETS.map(s => s.key).filter(v => v === k || S.sets.includes(v));
    marks.sets();
    schedule();
  }));
  marks.sets();
})();
const legendSwatch = (colour, text, dashed) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(${colour})" stroke-width="2"${dashed ? ' stroke-dasharray="4 3"' : ""}/></svg>${text}</span>`;
const legendDot = (colour, text, hollow) => `<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" ${hollow ? `fill="none" stroke="var(${colour})" stroke-width="1.5"` : `fill="var(${colour})"`}/></svg>${text}</span>`;

const M = {l: 46, r: 14};
function ticks(lo, hi, count) {
  const raw = Math.max((hi - lo) / Math.max(count, 1), 1e-12), e = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(m => m * e).find(s => s >= raw);
  const out = [];
  for (let i = Math.ceil(lo / step); i * step <= hi; i++) out.push(i * step);
  return out;
}
const logTicks = (lo, hi) => [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 4, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 1e4, 1e5, 1e6, 1e7].filter(v => v >= lo * 0.999 && v <= hi * 1.001);
const frame = (W, H) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img">`;
const hline = (W, y, colour, dash, width = 1) => `<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" stroke="var(${colour})" stroke-width="${width}"${dash ? ' stroke-dasharray="4 3"' : ""}/>`;
const vline = (x, t, bottom, colour, dash, width = 1) => `<line x1="${x}" x2="${x}" y1="${t}" y2="${bottom}" stroke="var(${colour})" stroke-width="${width}"${dash ? ' stroke-dasharray="4 3"' : ""}/>`;
const poly = (pts, colour, width = 1.25, extra = "") => pts.length ? `<path d="${pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("")}" fill="none" stroke="var(${colour})" stroke-width="${width}" stroke-linejoin="round" ${extra}/>` : "";
function xAxis(W, y, X, values, label, format = v => +v.toPrecision(3)) {
  let s = `<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" stroke="var(--axis)"/>`;
  values.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${y}" y2="${y + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${y + 16}" text-anchor="middle">${format(v)}</text>`);
  return s + `<text x="${(M.l + W - M.r) / 2}" y="${y + 31}" text-anchor="middle">${label}</text>`;
}
function yGrid(W, Y, values, format = v => +v.toPrecision(3)) {
  return values.map(v => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${format(v)}</text>`).join("");
}
const yLabel = (H, t, b, text) => `<text class="lbl" transform="translate(11 ${(t + H - b) / 2}) rotate(-90)" text-anchor="middle">${text}</text>`;

// draws of one set for the selected source: 8 columns of offsets in narrowband widths, or null
function drawsOf(key, w, s) {
  const raw = key === "j0" ? w.draws.joint[0]?.[s] : key === "j1" ? w.draws.joint[1]?.[s] : key === "n" ? w.draws.narrow[s] : w.draws.fisher[s];
  return raw ? raw.map(col => col.map(v => v / Q)) : null;
}
function quantile(sorted, q) { return sorted[clamp(Math.round(q * (sorted.length - 1)), 0, sorted.length - 1)]; }
function ellipse(cov, mean, i, j, k, X, Y) {  // the k-sigma contour of the (i, j) marginal, x = j, y = i
  const a = cov[j][j], b = cov[i][j], c = cov[i][i];
  const half = (a + c) / 2, root = Math.sqrt(((a - c) / 2) ** 2 + b * b), l1 = Math.max(half + root, 0), l2 = Math.max(half - root, 0);
  const th = 0.5 * Math.atan2(2 * b, a - c), pts = [];
  for (let n = 0; n <= 64; n++) {
    const t = 2 * Math.PI * n / 64, u = k * Math.sqrt(l1) * Math.cos(t), v = k * Math.sqrt(l2) * Math.sin(t);
    pts.push([X(mean[j] + u * Math.cos(th) - v * Math.sin(th)), Y(mean[i] + u * Math.sin(th) + v * Math.cos(th))]);
  }
  return pts;
}
function context() {
  const w = W_(), s = S.source, visible = SETS.filter(set => S.sets.includes(set.key));
  const draws = {};
  visible.forEach(set => { const d = drawsOf(set.key, w, s); if (d) draws[set.key] = d; });
  const cov = w.fisher_cov[s], mean = w.fisher_mean[s];
  // per parameter: the union of each visible chain's 0.5-99.5% range, the truth and the Fisher's ±2.5σ
  const range = [];
  for (let p = 0; p < 8; p++) {
    let lo = 0, hi = 0;
    for (const [key, d] of Object.entries(draws)) {
      if (key === "f") continue;
      const sorted = [...d[p]].sort((a, b) => a - b);
      lo = Math.min(lo, quantile(sorted, 0.005)); hi = Math.max(hi, quantile(sorted, 0.995));
    }
    if (S.sets.includes("f") || !Object.keys(draws).some(k => k !== "f")) {
      const sd = Math.sqrt(cov[p][p]);
      lo = Math.min(lo, mean[p] - 2.5 * sd); hi = Math.max(hi, mean[p] + 2.5 * sd);
    }
    if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
    const pad = 0.06 * (hi - lo);
    range.push([lo - pad, hi + pad]);
  }
  return {w, s, draws, cov, mean, range, visible};
}

function histogram(values, lo, hi, bins) {
  const h = new Array(bins).fill(0);
  values.forEach(v => { const i = Math.floor((v - lo) / (hi - lo) * bins); if (i >= 0 && i < bins) h[i] += 1; });
  const total = values.length * (hi - lo) / bins;
  return h.map(v => v / total);
}
const gauss = (x, m, sd) => Math.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

function cornerPanel(c) {
  const host = document.getElementById("p-corner");
  const W = Math.max(280, Math.min(host.clientWidth || 600, 620)), L = 44, top = 4, cell = (W - L - 4) / 8, H = 8 * cell + top + 30;
  const dpr = window.devicePixelRatio || 1;
  let cv = host.querySelector("canvas");
  if (!cv) {
    cv = document.createElement("canvas");
    host.appendChild(cv);
    cv.addEventListener("click", ev => {
      const r = cv.getBoundingClientRect(), g = cv._geo;
      const j = Math.floor((ev.clientX - r.left - g.L) / g.cell), i = Math.floor((ev.clientY - r.top - g.top) / g.cell);
      if (j >= 0 && j < 8 && i >= j && i < 8) pick({param: j, pair: i});
    });
  }
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  cv.style.width = `${W}px`; cv.style.height = `${H}px`;
  cv._geo = {L, top, cell};
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const pad = 3, inner = cell - 2 * pad, selI = Math.max(S.param, S.pair), selJ = Math.min(S.param, S.pair);
  const px = (p, v, x0) => x0 + pad + (v - c.range[p][0]) / (c.range[p][1] - c.range[p][0]) * inner;
  const py = (p, v, y0) => y0 + pad + (c.range[p][1] - v) / (c.range[p][1] - c.range[p][0]) * inner;
  g.font = "11px system-ui, sans-serif";
  for (let i = 0; i < 8; i++) for (let j = 0; j <= i; j++) {
    const x0 = L + j * cell, y0 = top + i * cell;
    g.strokeStyle = css("--grid"); g.lineWidth = 1;
    g.strokeRect(x0 + 0.5, y0 + 0.5, cell - 1, cell - 1);
    g.save();
    g.beginPath(); g.rect(x0 + 1, y0 + 1, cell - 2, cell - 2); g.clip();
    if (i === j) {
      // marginal: one histogram per chain and the Fisher Gaussian, sharing a height scale
      const [lo, hi] = c.range[j], bins = 24, curves = [];
      for (const [key, d] of Object.entries(c.draws)) if (key !== "f") curves.push([key, histogram(d[j], lo, hi, bins)]);
      const sd = Math.sqrt(c.cov[j][j]), peak = Math.max(S.sets.includes("f") ? gauss(0, 0, sd) : 0, ...curves.map(([, h]) => Math.max(...h)));
      const hy = v => y0 + cell - pad - v / peak * (inner * 0.92);
      if (lo < 0 && hi > 0) { g.strokeStyle = css("--axis"); g.beginPath(); g.moveTo(px(j, 0, x0), y0); g.lineTo(px(j, 0, x0), y0 + cell); g.stroke(); }
      for (const [key, h] of curves) {
        g.strokeStyle = css(SETS.find(s => s.key === key).colour); g.lineWidth = 1.25; g.beginPath();
        h.forEach((v, b) => { const xa = px(j, lo + b * (hi - lo) / bins, x0), xb = px(j, lo + (b + 1) * (hi - lo) / bins, x0); b ? g.lineTo(xa, hy(v)) : g.moveTo(xa, hy(v)); g.lineTo(xb, hy(v)); });
        g.stroke();
      }
      if (S.sets.includes("f")) {
        g.strokeStyle = css("--ref"); g.setLineDash([4, 3]); g.beginPath();
        for (let n = 0; n <= 60; n++) { const v = lo + n / 60 * (hi - lo), xx = px(j, v, x0), yy = hy(gauss(v, c.mean[j], sd)); n ? g.lineTo(xx, yy) : g.moveTo(xx, yy); }
        g.stroke(); g.setLineDash([]);
      }
    } else {
      g.strokeStyle = css("--axis"); g.lineWidth = 1; g.beginPath();
      g.moveTo(px(j, 0, x0), y0); g.lineTo(px(j, 0, x0), y0 + cell); g.moveTo(x0, py(i, 0, y0)); g.lineTo(x0 + cell, py(i, 0, y0)); g.stroke();
      for (const [key, d] of Object.entries(c.draws)) {
        if (key === "f") continue;
        g.fillStyle = rgba(SETS.find(s => s.key === key).colour, 0.35);
        for (let n = 0; n < d[j].length; n++) g.fillRect(px(j, d[j][n], x0) - 0.8, py(i, d[i][n], y0) - 0.8, 1.6, 1.6);
      }
      if (S.sets.includes("f")) {
        g.strokeStyle = css("--ref"); g.setLineDash([3, 2]); g.lineWidth = 1;
        for (const k of [1, 2]) {
          const pts = ellipse(c.cov, c.mean, i, j, k, v => px(j, v, x0), v => py(i, v, y0));
          g.beginPath(); pts.forEach(([xx, yy], n) => n ? g.lineTo(xx, yy) : g.moveTo(xx, yy)); g.stroke();
        }
        g.setLineDash([]);
      }
    }
    g.restore();
    if (i === selI && j === selJ) { g.strokeStyle = css("--ink"); g.lineWidth = 2; g.strokeRect(x0 + 1, y0 + 1, cell - 2, cell - 2); }
  }
  g.fillStyle = css("--ink-2"); g.textAlign = "right"; g.textBaseline = "middle";
  for (let i = 1; i < 8; i++) g.fillText(PLAIN[i], L - 5, top + (i + 0.5) * cell);
  g.textAlign = "center"; g.textBaseline = "top";
  for (let j = 0; j < 8; j++) g.fillText(PLAIN[j], L + (j + 0.5) * cell, top + 8 * cell + 6);
}

function pairPanel(c) {
  const host = document.getElementById("p-pair"), W = Math.max(280, host.clientWidth || 340), t = 10, b = 40, H = Math.min(W, 380) + b;
  const px_ = S.param, py_ = S.pair, [x0, x1] = c.range[px_], X = v => M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r);
  let s = frame(W, H);
  if (px_ === py_) {
    // one marginal, larger
    const bins = 40, curves = [];
    for (const [key, d] of Object.entries(c.draws)) if (key !== "f") curves.push([key, histogram(d[px_], x0, x1, bins)]);
    const sd = Math.sqrt(c.cov[px_][px_]), peak = 1.08 * Math.max(S.sets.includes("f") ? gauss(0, 0, sd) : 0, ...curves.map(([, h]) => Math.max(...h)));
    const Y = v => t + (peak - v) / peak * (H - t - b);
    s += yGrid(W, Y, ticks(0, peak, 4));
    if (x0 < 0 && x1 > 0) s += vline(X(0), t, H - b, "--axis");
    for (const [key, h] of curves) {
      const pts = [];
      h.forEach((v, n) => { pts.push([X(x0 + n * (x1 - x0) / bins), Y(v)], [X(x0 + (n + 1) * (x1 - x0) / bins), Y(v)]); });
      s += poly(pts, SETS.find(z => z.key === key).colour, 1.5);
    }
    if (S.sets.includes("f")) {
      const pts = [];
      for (let n = 0; n <= 120; n++) { const v = x0 + n / 120 * (x1 - x0); pts.push([X(v), Y(gauss(v, c.mean[px_], sd))]); }
      s += poly(pts, "--ref", 1.25, 'stroke-dasharray="4 3"');
    }
    s += yLabel(H, t, b, "density");
  } else {
    const [y0, y1] = c.range[py_], Y = v => t + (y1 - v) / (y1 - y0) * (H - t - b);
    s += yGrid(W, Y, ticks(y0, y1, 5));
    if (x0 < 0 && x1 > 0) s += vline(X(0), t, H - b, "--axis");
    if (y0 < 0 && y1 > 0) s += hline(W, Y(0), "--axis");
    for (const [key, d] of Object.entries(c.draws)) {
      if (key === "f") continue;
      const colour = SETS.find(z => z.key === key).colour;
      for (let n = 0; n < d[px_].length; n++) {
        const xx = X(d[px_][n]), yy = Y(d[py_][n]);
        if (xx >= M.l && xx <= W - M.r && yy >= t && yy <= H - b) s += `<circle cx="${xx.toFixed(1)}" cy="${yy.toFixed(1)}" r="1.7" fill="var(${colour})" opacity="0.45"/>`;
      }
    }
    if (S.sets.includes("f")) for (const k of [1, 2]) s += poly(ellipse(c.cov, c.mean, py_, px_, k, X, Y), "--ref", 1.25, 'stroke-dasharray="4 3"');
    s += yLabel(H, t, b, `${PLAIN[py_]} offset / narrowband width`);
  }
  s += xAxis(W, H - b, X, ticks(x0, x1, Math.max(3, Math.floor((W - M.l - M.r) / 70))), `${PLAIN[px_]} offset / narrowband width`);
  host.innerHTML = s + "</svg>";
  const sc = c.w.scale[c.s], unit = p => `${PLAIN[p]}: ${+sc[p].toPrecision(3)}${UNITS[p]}`;
  document.getElementById("pair-note").innerHTML = `One narrowband width of this source: ${unit(px_)}${px_ !== py_ ? `; ${unit(py_)}` : ""}. Window ${S.window}, f = ${(c.w.f * 1e3).toPrecision(5)} mHz, source ${c.s} at SNR ${fmt(c.w.snr[c.s], 1)}.`;
}

function widthPanel(c) {
  const host = document.getElementById("p-widths"), W = Math.max(280, host.clientWidth || 600), t = 10, b = 50, H = 250;
  const lo = 0.25, hi = 4, Y = v => t + (Math.log(hi) - Math.log(clamp(v, lo, hi))) / (Math.log(hi) - Math.log(lo)) * (H - t - b);
  const cw = (W - M.l - M.r) / 8, cx = p => M.l + (p + 0.5) * cw, w = c.w, s = c.s;
  let out = frame(W, H);
  out += `<rect x="${M.l + S.param * cw}" width="${cw}" y="${t}" height="${H - t - b}" fill="var(--ref)" opacity="0.07"/>`;
  out += `<rect x="${M.l}" width="${W - M.l - M.r}" y="${Y(1.25)}" height="${Y(0.8) - Y(1.25)}" fill="var(--grid)" opacity="0.7"/>`;
  out += yGrid(W, Y, [0.25, 0.5, 1, 2, 4]) + hline(W, Y(1), "--axis");
  const marker = (x, v, colour, shape) => {
    const y = Y(v), clipped = v < lo || v > hi, title = `<title>${fmt(v, 3)}</title>`;
    if (clipped) { const d = v > hi ? -1 : 1; return `<path d="M${x - 4},${y + d * 1} L${x + 4},${y + d * 1} L${x},${y - d * 6} Z" fill="var(${colour})">${title}</path>`; }
    if (shape === "diamond") return `<path d="M${x},${y - 5} L${x + 5},${y} L${x},${y + 5} L${x - 5},${y} Z" fill="var(${colour})">${title}</path>`;
    if (shape === "hollow") return `<circle cx="${x}" cy="${y}" r="3.5" fill="none" stroke="var(${colour})" stroke-width="1.5">${title}</circle>`;
    return `<circle cx="${x}" cy="${y}" r="4" fill="var(${colour})">${title}</circle>`;
  };
  for (let p = 0; p < 8; p++) {
    const x = cx(p), dx = Math.min(cw / 6, 14);
    w.joint_width.forEach((jw, k) => out += marker(x - (2 - k) * dx, jw[s][p], k ? "--s3" : "--s1"));
    w.narrow_width.forEach((nw, k) => out += marker(x + k * dx * 0.6, nw[s][p], "--s2", k ? "hollow" : "dot"));
    out += marker(x + 2 * dx, w.fisher_width[s][p], "--ref", "diamond");
    out += `<text class="lbl" x="${x}" y="${H - b + 16}" text-anchor="middle">${PLAIN[p]}</text>`;
    if (cw >= 64) out += `<text x="${x}" y="${H - b + 31}" text-anchor="middle">prior ×${priorText(w.prior_width[s][p])}</text>`;
    out += `<rect class="hit" data-p="${p}" x="${M.l + p * cw}" width="${cw}" y="${t}" height="${H - t}" fill="transparent" style="cursor: pointer"/>`;
  }
  out += yLabel(H, t, b, "width / narrowband width");
  host.innerHTML = out + "</svg>";
  host.querySelectorAll(".hit").forEach(r => r.addEventListener("click", () => pick({param: Number(r.dataset.p)})));
}

function tracePanel() {
  const T = T_(), k = series(), acfHost = document.getElementById("p-acf");
  const host = document.getElementById("p-trace"), W = Math.max(280, host.clientWidth || 500), t = 10, b = 40, H = 280;
  const X = v => M.l + v / T.iterations * (W - M.l - M.r), q = v => v / QT;
  // y range from what is drawn, capped so the start's excursion does not flatten the rest
  let lo = -1.5, hi = 1.5;
  T.chains.forEach(ch => (S.view === "band" ? ch.band[k].flat() : ch.walker[k]).forEach(v => { lo = Math.min(lo, q(v)); hi = Math.max(hi, q(v)); }));
  lo = Math.max(lo, -12); hi = Math.min(hi, 12);
  const Y = v => t + (hi - clamp(v, lo, hi)) / (hi - lo) * (H - t - b);
  let s = frame(W, H);
  const burn = DATA.traces.burn, found = DATA.convergence.windows[String(S.window)].burn;
  s += `<rect x="${X(0)}" width="${X(burn) - X(0)}" y="${t}" height="${H - t - b}" fill="var(--ref)" opacity="0.07"/><text x="${X(0) + 4}" y="${t + 10}">paper burn-in</text>`;
  s += yGrid(W, Y, ticks(lo, hi, 5));
  T.chains.forEach((ch, n) => {
    const colour = SEED_COLOURS[n % 4];
    if (S.view === "band") {
      const [q16, q50, q84] = ch.band[k], xs = T.iteration_band;
      const area = xs.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(q(q84[i])).toFixed(1)}`).join("") + xs.map((x, i) => `L${X(xs[xs.length - 1 - i]).toFixed(1)},${Y(q(q16[xs.length - 1 - i])).toFixed(1)}`).join("") + "Z";
      s += `<path d="${area}" fill="var(${colour})" opacity="0.13"/>`;
      s += poly(xs.map((x, i) => [X(x), Y(q(q50[i]))]), colour, 1.25);
    } else {
      s += poly(T.iteration_walker.map((x, i) => [X(x), Y(q(ch.walker[k][i]))]), colour, 1);
    }
  });
  if (found !== null) s += vline(X(found), t, H - b, "--ink", true) + `<text class="lbl" x="${X(found) + 4}" y="${H - b - 6}">burn test passes from ${big(found)}</text>`;
  s += xAxis(W, H - b, X, ticks(0, T.iterations, Math.max(3, Math.floor((W - M.l - M.r) / 90))), "iteration", v => v >= 1000 ? `${v / 1000}k` : v);
  s += yLabel(H, t, b, "offset / kept width");
  host.innerHTML = s + "</svg>";
  document.getElementById("legend-d").innerHTML = T.chains.map((_, n) => legendSwatch(SEED_COLOURS[n % 4], `seed ${T.seeds[n]}`)).join("");

  // autocorrelation of the kept draws per seed, with the integrated time from the analysis
  const C = DATA.convergence.curves[String(S.window)], at = C.burn_burn.indexOf(burn), tau = at >= 0 ? C.burn_tau[at][k] : null;
  const AW = Math.max(220, acfHost.clientWidth || 260), AH = 280, lags = DATA.traces.lags, A = {l: 40};
  const AX = v => A.l + v / ((lags - 1) * T.thin) * (AW - A.l - M.r), AY = v => t + (1 - clamp(v, -0.2, 1)) / 1.2 * (AH - t - b);
  let a = `<svg viewBox="0 0 ${AW} ${AH}" width="${AW}" role="img">`;
  [0, 0.5, 1].forEach(v => a += `<line x1="${A.l}" x2="${AW - M.r}" y1="${AY(v)}" y2="${AY(v)}" stroke="var(--grid)"/><text x="${A.l - 6}" y="${AY(v) + 4}" text-anchor="end">${v}</text>`);
  T.chains.forEach((ch, n) => a += poly(ch.acf[k].map((v, i) => [AX(i * T.thin), AY(v)]), SEED_COLOURS[n % 4], 1.25));
  if (tau) a += `<line x1="${AX(tau)}" x2="${AX(tau)}" y1="${t}" y2="${AH - b}" stroke="var(--ink)" stroke-dasharray="4 3"/><text class="lbl" x="${AX(tau) + 4}" y="${t + 10}">τ = ${fmt(tau, 0)} it</text>`;
  a += `<line x1="${A.l}" x2="${AW - M.r}" y1="${AH - b}" y2="${AH - b}" stroke="var(--axis)"/>`;
  ticks(0, (lags - 1) * T.thin, 3).forEach(v => a += `<text x="${AX(v)}" y="${AH - b + 16}" text-anchor="middle">${v}</text>`);
  a += `<text x="${(A.l + AW - M.r) / 2}" y="${AH - b + 31}" text-anchor="middle">lag [iterations]</text>`;
  acfHost.innerHTML = a + "</svg>";
}

function convPanel() {
  const C = DATA.convergence.curves[String(S.window)], worst = document.getElementById("worst");
  const host = document.getElementById("p-conv"), W = Math.max(280, host.clientWidth || 600), t = 10, b = 40, H = 280, k = series();
  const sw = DATA.convergence.windows[String(S.window)], crit = DATA.convergence.criterion, paper = sw.settings.paper;
  const burnUsed = sw.burn !== null ? sw.burn : C.burn_burn[C.burn_burn.length - 1];
  let xs, ys, X, Y, tol, label, xlabel, yTicks, yFormat = v => +v.toPrecision(3);
  if (S.conv === "median" || S.conv === "width") {
    const key = S.conv === "median" ? "burn_window_median" : "burn_window_width";
    xs = C.burn_window.map(v => v + sw.step / 2); ys = C[key];
    tol = S.conv === "median" ? crit.median_tol : crit.width_tol;
    let top = 0.3; ys.forEach(r => r.forEach(v => { if (v !== null) top = Math.max(top, v); })); top = Math.min(top * 1.05, 1);
    X = v => M.l + v / DATA.convergence.chain.iterations * (W - M.l - M.r);
    Y = v => t + (top - clamp(v, 0, top)) / top * (H - t - b);
    yTicks = ticks(0, top, 5);
    label = S.conv === "median" ? "worst seed's median shift / reference half-width" : "worst seed's |width ratio − 1|";
    xlabel = `start of each ${big(sw.step)}-iteration segment`;
  } else {
    xs = C.keep_keep; ys = S.conv === "ess" ? C.keep_ess : C.keep_rhat.map(r => r.map(v => v === null ? null : Math.max(Math.abs(v - 1), 1e-6)));
    tol = S.conv === "ess" ? crit.ess_min : crit.rhat_max - 1;
    let lo = Infinity, hi = -Infinity; ys.forEach(r => r.forEach(v => { if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
    lo = Math.min(lo, tol) / 1.5; hi = Math.max(hi, tol) * 1.5;
    const span = [Math.max(...xs, paper.keep) * 1.05];
    X = v => M.l + v / span[0] * (W - M.l - M.r);
    Y = v => t + Math.log(hi / clamp(v, lo, hi)) / Math.log(hi / lo) * (H - t - b);
    yTicks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7].filter(v => v >= lo && v <= hi);
    yFormat = v => v >= 1e3 ? `${v / 1e3}k` : v < 1e-2 ? `1e${Math.round(Math.log10(v))}` : v;
    label = S.conv === "ess" ? "ESS (all walkers, one seed)" : "|R̂ − 1|";
    xlabel = `kept iterations after a burn-in of ${big(burnUsed)}`;
  }
  let s = frame(W, H) + yGrid(W, Y, yTicks, yFormat);
  if (S.conv === "median" || S.conv === "width") {
    s += vline(X(paper.burn), t, H - b, "--ref", true) + `<text x="${X(paper.burn) + 4}" y="${t + 10}">paper burn-in</text>`;
    if (sw.burn !== null) s += vline(X(sw.burn), t, H - b, "--ink", true) + `<text class="lbl" x="${X(sw.burn) + 4}" y="${t + 24}">burn test passes from ${big(sw.burn)}</text>`;
  } else {
    s += vline(X(paper.keep), t, H - b, "--ref", true) + `<text x="${X(paper.keep) - 4}" y="${t + 10}" text-anchor="end">paper keep</text>`;
  }
  s += hline(W, Y(tol), "--s2", true, 1.5);
  for (let n = 0; n < 33; n++) if (n !== k) s += poly(xs.map((x, i) => [X(x), Y(ys[i][n])]).filter((_, i) => ys[i][n] !== null), "--muted", 1, 'opacity="0.35"');
  s += poly(xs.map((x, i) => [X(x), Y(ys[i][k])]).filter((_, i) => ys[i][k] !== null), "--s1", 2.25);
  xs.forEach((x, i) => { if (ys[i][k] !== null) s += `<circle cx="${X(x)}" cy="${Y(ys[i][k])}" r="2.5" fill="var(--s1)"/>`; });
  s += xAxis(W, H - b, X, ticks(0, S.conv === "median" || S.conv === "width" ? DATA.convergence.chain.iterations : Math.max(...xs, paper.keep), 6), xlabel, v => v >= 1000 ? `${v / 1000}k` : v);
  s += yLabel(H, t, b, label);
  host.innerHTML = s + "</svg>";

  // the series that drifts most after the paper's burn-in
  let best = {v: -1};
  C.burn_window.forEach((start, i) => { if (start >= paper.burn) C.burn_window_median[i].forEach((v, n) => { if (v !== null && v > best.v) best = {v, n, start}; }); });
  worst.textContent = `Worst median drift after ${big(paper.burn)}: ${seriesName(best.n)}, ${fmt(best.v, 2)} at ${big(best.start)}`;
  worst.onclick = () => pick(best.n === 32 ? {series: "logl", conv: "median"} : {series: "param", source: Math.floor(best.n / 8), param: best.n % 8, conv: "median"});
  const tolText = {median: `the median of each ${big(sw.step)}-iteration segment, per seed, against the other seeds' second halves pooled, in their half 68% widths; the burn test asks every later segment of all 33 series to stay under ${crit.median_tol}`, width: `each segment's 68% width against the same reference; the burn test asks for ${Math.round(crit.width_tol * 100)}%`, ess: `the effective sample size of the kept draws per seed against the ${big(crit.ess_min)} required`, rhat: `split R̂ across the four seeds against ${crit.rhat_max}`};
  document.getElementById("note-e").innerHTML = `Below, the analysis of these chains: ${tolText[S.conv]}. Grey lines are all 32 parameters and log L; the blue one is ${seriesName(k)}.${S.conv === "ess" || S.conv === "rhat" ? ` The kept-length curves start after ${sw.burn !== null ? "the burn-in the test found" : "half the chain, since no burn-in passes the test"}.` : ""}`;
}

function convTable() {
  const rows = LONG.map(String).sort((a, b) => DATA.convergence.windows[b].window_snr - DATA.convergence.windows[a].window_snr);
  let s = `<table><thead><tr><th>window</th><th>SNR</th><th>burn test</th><th>τ max [it]</th><th>ESS min</th><th>max |R̂ − 1|</th><th>median / width drift</th><th>paper 100k/300k</th><th>fml 20k/60k</th></tr></thead><tbody>`;
  rows.forEach(j => {
    const w = DATA.convergence.windows[j], p = w.settings.paper, f = w.settings.fml;
    s += `<tr class="pick${Number(j) === S.window ? " sel" : ""}" data-j="${j}"><td>${j}</td><td>${Math.round(w.window_snr)}</td><td>${w.burn === null ? "none passes" : `from ${big(w.burn)}`}</td><td>${fmt(p.tau_max, 0)}</td><td>${big(p.ess_min)}</td><td>${(p.rhat_max - 1).toExponential(1)}</td><td>${fmt(p.median_drift_max, 3)} / ${fmt(p.width_drift_max, 3)}</td><td>${p.passes ? "passes" : "fails"}</td><td>${f.passes ? "passes" : "fails"}</td></tr>`;
  });
  const host = document.getElementById("t-conv");
  host.innerHTML = s + `</tbody></table><p class="note">Paper columns: burn 100 000, keep 300 000, thin 100, per seed. A setting passes when the kept draws pass (ESS, R̂, drift) and a burn-in passes the segment test. Click a row to pick its window.</p>`;
  host.querySelectorAll("tr.pick").forEach(r => r.addEventListener("click", () => pick({window: Number(r.dataset.j)})));
}

const lateDrift = j => { const C = DATA.convergence.curves[j], cv = DATA.convergence.windows[j]; return Math.max(...C.burn_window.flatMap((start, i) => start >= cv.settings.paper.burn ? C.burn_window_median[i].filter(v => v !== null) : [])); };

function fisherPanel() {
  const host = document.getElementById("p-fisher"), W = Math.max(280, host.clientWidth || 600), t = 10, b = 40, H = 290, p = S.param;
  const sx = [5, 2500], sy = [0.2, 5];
  const X = v => M.l + Math.log(v / sx[0]) / Math.log(sx[1] / sx[0]) * (W - M.l - M.r);
  const Y = v => t + Math.log(sy[1] / clamp(v, sy[0], sy[1])) / Math.log(sy[1] / sy[0]) * (H - t - b);
  let s = frame(W, H);
  s += `<rect x="${M.l}" width="${W - M.l - M.r}" y="${Y(DATA.agree[1])}" height="${Y(DATA.agree[0]) - Y(DATA.agree[1])}" fill="var(--grid)" opacity="0.7"/>`;
  s += yGrid(W, Y, [0.2, 0.5, 1, 2, 5]) + hline(W, Y(1), "--axis");
  s += vline(X(DATA.faint_snr), t, H - b, "--ref", true) + `<text x="${X(DATA.faint_snr) + 4}" y="${t + 10}">SNR ${DATA.faint_snr}</text>`;
  const points = [];
  WINDOWS.forEach((w, j) => {
    if (S.against === "joint" && !w.joint_width.length) return;
    w.snr.forEach((snr, src) => {
      const ref = S.against === "joint" ? jointMean(w, src, p) : 1;
      const ratio = w.fisher_width[src][p] / ref;
      const ideal = p === 0 ? DATA.fisher.narrow.injections[j].ideal_width_bins[src] / (w.scale[src][0] * ref) : null;
      points.push({j, src, snr, ratio, ideal});
    });
  });
  points.forEach(q => { if (q.ideal !== null) s += `<circle cx="${X(q.snr)}" cy="${Y(q.ideal)}" r="3.5" fill="none" stroke="var(--muted)" stroke-width="1.25"/>`; });
  points.forEach(q => {
    const sel = q.j === S.window && q.src === S.source;
    s += `<circle class="hit" data-j="${q.j}" data-s="${q.src}" cx="${X(q.snr)}" cy="${Y(q.ratio)}" r="${sel ? 6 : 4}" fill="var(--s1)" ${sel ? 'stroke="var(--ink)" stroke-width="2"' : 'opacity="0.85"'} style="cursor: pointer"><title>window ${q.j}, source ${q.src}, SNR ${fmt(q.snr, 1)}: ${fmt(q.ratio, 3)}</title></circle>`;
  });
  s += xAxis(W, H - b, X, W < 560 ? [10, 100, 1000] : logTicks(sx[0], sx[1]), "source SNR") + yLabel(H, t, b, `Fisher ÷ MCMC width, ${PLAIN[p]}`);
  host.innerHTML = s + "</svg>";
  host.querySelectorAll(".hit").forEach(r => r.addEventListener("click", () => pick({window: Number(r.dataset.j), source: Number(r.dataset.s)})));
  const inBand = points.filter(q => q.snr >= DATA.faint_snr && q.ratio >= DATA.agree[0] && q.ratio <= DATA.agree[1]).length, loud = points.filter(q => q.snr >= DATA.faint_snr).length;
  document.getElementById("note-c").innerHTML = `The Fisher forecast (<code>injections.fisher_samples</code>: the Hessian of −log L at the truth averaged over ${DATA.fisher.draws} noise draws, plus the prior precision, eigenvalues floored at the prior) starts the joint chains. Here its width of ${NAMES[p]} over the ${S.against === "joint" ? `joint chain's (mean of two seeds; ${JOINT.length * 4} sources)` : `pooled narrowband chain's (all ${WINDOWS.length * 4} sources)`}, against the source's SNR: ${inBand} of the ${loud} sources above SNR ${DATA.faint_snr} land in 0.8–1.25.${p === 1 ? " log M<sub>c</sub> is prior-dominated in the narrow band, so this is the Fisher's Gaussian stand-in for the prior." : ""}${p === 0 ? " Hollow grey: the ideal √3/(πρ) bins over the same MCMC width." : ""} Click a point to pick its window and source.`;
  document.getElementById("legend-c").innerHTML = legendDot("--s1", "Fisher ÷ MCMC") + (p === 0 ? legendDot("--muted", "ideal √3/(πρ) ÷ MCMC", true) : "") + `<span><svg width="22" height="10"><rect width="22" height="10" fill="var(--grid)"/></svg>0.8–1.25</span>`;
}

function hessianPanel() {
  const B = DATA.fisher[S.band], host = document.getElementById("p-hessian"), W = Math.max(280, host.clientWidth || 600), t = 10, b = 50, H = 290;
  const lo = -4, hi = 12, Y = v => t + (hi - clamp(v, lo, hi)) / (hi - lo) * (H - t - b - 16);
  const cw = (W - M.l - M.r) / 8, cx = p => M.l + (p + 0.5) * cw, floorY = H - b - 4;
  let s = frame(W, H);
  s += `<rect x="${M.l + S.param * cw}" width="${cw}" y="${t}" height="${H - t - b}" fill="var(--ref)" opacity="0.07"/>`;
  s += yGrid(W, Y, [-4, 0, 4, 8, 12]) + hline(W, Y(0), "--axis");
  s += `<text x="${M.l - 6}" y="${floorY + 4}" text-anchor="end">≤ 0</text>`;
  B.injections.forEach((inj, j) => inj.source_snr.forEach((snr, src) => {
    const sel = S.band === "narrow" && j === S.window && src === S.source;
    for (let p = 0; p < 8; p++) {
      const k = src * 8 + p, h = inj.log10_hessian_diag[k], x = cx(p) + (j - 4.5) / 10 * cw * 0.7, title = `<title>injection ${j}, source ${src}, SNR ${fmt(snr, 1)}: ${h === null ? "non-positive Hessian" : `${fmt(h - B.log10_prior_diag[k], 1)} decades`}</title>`;
      if (h === null) { s += `<path d="M${x - 4},${floorY - 4} L${x + 4},${floorY + 4} M${x - 4},${floorY + 4} L${x + 4},${floorY - 4}" stroke="var(--s2)" stroke-width="1.75">${title}</path>`; continue; }
      s += `<circle cx="${x}" cy="${Y(h - B.log10_prior_diag[k])}" r="${sel ? 5 : 2.5}" fill="var(--s1)" ${sel ? 'stroke="var(--ink)" stroke-width="2"' : 'opacity="0.6"'}>${title}</circle>`;
    }
  }));
  for (let p = 0; p < 8; p++) {
    s += `<text class="lbl" x="${cx(p)}" y="${H - b + 16}" text-anchor="middle">${PLAIN[p]}</text>`;
    s += `<rect class="hit" data-p="${p}" x="${M.l + p * cw}" width="${cw}" y="${t}" height="${H - t}" fill="transparent" style="cursor: pointer"/>`;
  }
  s += yLabel(H, t, b, "log₁₀ (Hessian ÷ prior precision)");
  host.innerHTML = s + "</svg>";
  host.querySelectorAll(".hit").forEach(r => r.addEventListener("click", () => pick({param: Number(r.dataset.p)})));
  document.getElementById("note-h").innerHTML = `Where does it lean on the prior? Per eval injection of the ${S.band} band (seed ${B.seed}) and source, each parameter's averaged Hessian diagonal over the prior's precision, in decades: below 0 the forecast is mostly prior. Crosses mark a non-positive averaged diagonal, which the added prior precision repairs. Within a column the injections run left to right by window.`;
  document.getElementById("legend-h").innerHTML = legendDot("--s1", "one source") + `<span><svg width="12" height="12"><path d="M2,2 L10,10 M2,10 L10,2" stroke="var(--s2)" stroke-width="1.75"/></svg>non-positive</span>`;
  let tbl = `<table><thead><tr><th>injection</th><th>window SNR</th><th>source SNRs</th><th>non-positive diagonals</th><th>floored eigenvalues</th><th>precision span [decades]</th></tr></thead><tbody>`;
  B.injections.forEach((inj, j) => {
    const bad = inj.log10_hessian_diag.map((h, k) => h === null ? `s${Math.floor(k / 8)} ${PLAIN[k % 8]}` : null).filter(Boolean);
    tbl += `<tr class="pick${S.band === "narrow" && j === S.window ? " sel" : ""}" data-j="${j}"><td>${j}</td><td>${Math.round(inj.window_snr)}</td><td>${inj.source_snr.map(v => fmt(v, 0)).join(", ")}</td><td>${bad.join(", ") || "–"}</td><td>${inj.floored_eigenvalues}</td><td>${fmt(inj.precision_decades, 1)}</td></tr>`;
  });
  document.getElementById("t-hessian").innerHTML = tbl + `</tbody></table><p class="note">Floored eigenvalues: how many of the 32 eigenvalues of the precision <code>fisher_samples</code> raises to the prior's. The precision span is the decades between its largest and smallest eigenvalue.${S.band === "narrow" ? " Click a row to pick its window." : " The full-band injections have no MCMC, so rows do not pick a window."}</p>`;
  if (S.band === "narrow") document.querySelectorAll("#t-hessian tr.pick").forEach(r => r.addEventListener("click", () => pick({window: Number(r.dataset.j)})));
}

function tiles() {
  const tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  const w = W_(), s = S.source, p = S.param, joint = w.joint_width.length > 0, long = LONG.includes(S.window);
  const chains = joint ? DATA.chains.joint.filter(r => r.window === S.window) : DATA.chains.narrowband.filter(r => r.inj === S.window && r.src === s);
  const rhat = Math.max(...chains.map(r => r.rhat_max)), ess = Math.min(...chains.map(r => r.ess_min));
  const against = joint && S.against === "joint", ref = against ? jointMean(w, s, p) : 1;
  const cv = long ? DATA.convergence.windows[String(S.window)] : null;
  document.getElementById("tiles").innerHTML = [
    tile(`Window ${S.window}`, `SNR ${Math.round(w.window_snr)}`, `f = ${(w.f * 1e3).toPrecision(5)} mHz; ${joint ? "joint and long chains" : "narrowband chains only"}`),
    tile(`Source ${s}`, `SNR ${fmt(w.snr[s], 1)}`, `slot ${s} of 4, loudest first`),
    tile("Chain health", `R̂ ${fmt(rhat, 4)}`, `ESS ≥ ${big(ess)}; ${joint ? "this window's joint chains" : "this source's narrowband chains"}`),
    tile(`Joint ÷ narrowband, ${NAMES[p]}`, joint ? fmt(jointMean(w, s, p), 2) : "–", joint ? "freeing the other three sources" : "no joint chain in this window"),
    tile(`Fisher ÷ MCMC, ${NAMES[p]}`, fmt(w.fisher_width[s][p] / ref, 2), `against the ${against ? "joint" : "narrowband"} width; 0.8–1.25 agrees`),
    tile("Burn test", !long ? "–" : cv.burn === null ? "none" : big(cv.burn), !long ? "no long chains in this window" : `paper 100k/300k ${cv.settings.paper.passes ? "passes" : "fails"}; fml 20k/60k ${cv.settings.fml.passes ? "passes" : "fails"}`),
  ].join("");
}

function texts() {
  const D = DATA.draws, sm = DATA.summary, cv = DATA.convergence.windows;
  document.getElementById("sub").innerHTML = `The flow is scored against MCMC posteriors of the narrow-B eval set (<code>${D.config}</code>, problem <code>${D.problem_id}</code>): ${WINDOWS.length} windows of four sources each. Every source has a narrowband chain (that source free, the other three subtracted at their truth; ${sm.narrow_chains} chains, two seeds each); windows ${JOINT.join(", ")} also have a joint chain (all four free, two seeds) and four ${big(DATA.convergence.chain.iterations)}-iteration long chains that test the burn-in. The Fisher forecast starts the joint chains. Built ${DATA.built} from <code>data/</code>.`;
  // joint-over-narrowband exceptions, one entry per source with its widest parameter
  const bySource = {};
  sm.joint_over_narrow_wide.forEach(r => { const key = `${r.window}.${r.source}`; if (!bySource[key] || Math.abs(Math.log(r.ratio)) > Math.abs(Math.log(bySource[key].ratio))) bySource[key] = r; });
  const exceptions = Object.values(bySource).sort((a, b) => b.ratio - a.ratio).map(r => `window ${r.window} source ${r.source} (SNR ${fmt(r.snr, 0)}: ${PLAIN[D.columns.indexOf(r.parameter)]} ${fmt(r.ratio, 2)}×)`);
  const pass = sm.paper_passes, fail = LONG.map(String).filter(j => !pass.includes(j));
  // Fisher misses above the faint cut: ratio outside 0.5-2 in any parameter but log Mc
  const misses = [];
  WINDOWS.forEach((w, j) => w.snr.forEach((snr, s) => { if (snr >= DATA.faint_snr && w.fisher_width[s].some((v, p) => p !== 1 && (v > 2 || v < 0.5))) misses.push({j, s, snr, params: w.fisher_width[s].map((v, p) => p !== 1 && (v > 2 || v < 0.5) ? PLAIN[p] : null).filter(Boolean)}); }));
  const missParams = [...new Set(misses.flatMap(m => m.params))];
  const bad = Object.entries(sm.nonpositive_by_parameter).filter(([, n]) => n > 0).map(([name, n]) => `${n} in ${name}`);
  const items = [
    `<b>The chains are converged and seed-stable.</b> The ${sm.joint_chain_count} joint chains have R̂ ≤ ${fmt(sm.joint_rhat_max, 5)} and ESS ≥ ${big(sm.joint_ess_min)}; the ${sm.narrow_chains} narrowband chains R̂ ≤ ${fmt(sm.narrow_rhat_max, 5)} and ESS ≥ ${big(sm.narrow_ess_min)}. The two seeds' widths agree within 3% for ${fmt(sm.joint_seed_within3 * 100, 0)}% of joint and ${fmt(sm.narrow_seed_within3 * 100, 1)}% of narrowband source-parameters (worst ${fmt(sm.joint_seed_max * 100, 0)}% and ${fmt(sm.narrow_seed_max * 100, 1)}%).`,
    `<b>Joint and narrowband agree unless the sources interact.</b> The joint width over the narrowband width has median ${fmt(sm.joint_over_narrow_median, 3)}, and ${fmt(sm.joint_over_narrow_within * 100, 0)}% of the ${sm.joint_over_narrow_count} joint source-parameters lie within 0.9–1.1. The ${exceptions.length} sources outside, by their widest parameter: ${exceptions.join("; ")}. Almost all are wider under the joint chain, as freeing the neighbours should make them.`,
    `<b>The paper's 100k burn / 300k keep is long enough, but windows ${fail.join(" and ")} never pass the strict burn test.</b> Over its kept draws every long window has ESS ≥ ${big(sm.paper_ess_min)}, |R̂ − 1| ≤ ${sm.paper_rhat_minus_one_max.toExponential(1)} and median drift ≤ ${fmt(sm.paper_median_drift_max, 3)}. Windows ${pass.join(" and ")} settle by ${pass.map(j => big(cv[j].burn)).join(" and ")} iterations; in windows ${fail.join(" and ")} segments after 100 000 still drift by up to ${fail.map(j => fmt(lateDrift(j), 2)).join(" and ")} half-widths against the 0.1 tolerance, slow wandering the pooled 300 000 iterations average out. fml's 20 000 / 60 000 / 20 passes in window${sm.fml_passes.length === 1 ? "" : "s"} ${sm.fml_passes.join(" and ")} only. τ<sub>max</sub> is ${fmt(sm.tau_max_range[0], 0)}–${fmt(sm.tau_max_range[1], 0)} iterations, within 2× the trace's thinning of ${sm.trace_thin}, slowest in ${[...new Set(Object.values(sm.slowest_parameter))].join(", ")}: an upper bound, so the ESS values are lower bounds.`,
    `<b>The Fisher forecast is a good guide for loud sources only.</b> Above SNR ${DATA.faint_snr}, ${fmt(sm.fisher_narrow_loud_agree * 100, 0)}% of its widths (without log M<sub>c</sub>) are within 0.8–1.25 of the narrowband chain's over ${sm.fisher_narrow_loud_sources} sources, and ${fmt(sm.fisher_joint_loud_agree * 100, 0)}% of the joint chain's over ${sm.fisher_joint_loud_sources}. It is off by more than 2× for ${misses.length} of them (SNR ${fmt(Math.min(...misses.map(m => m.snr)), 0)}–${fmt(Math.max(...misses.map(m => m.snr)), 0)}) in ${missParams.join(", ")}, along the A–ι and ψ–φ₀ degeneracies (up to ${fmt(sm.fisher_narrow_loud_range[1], 1)}×). Below SNR ${DATA.faint_snr} the ratio spans ${fmt(sm.fisher_narrow_faint_range[0])}–${fmt(sm.fisher_narrow_faint_range[1])}, and for log M<sub>c</sub> ${fmt(sm.fisher_narrow_mc_range[0])}–${fmt(sm.fisher_narrow_mc_range[1])}. Its f₀ width is a median ${fmt(sm.fisher_f0_over_ideal_median)}× the ideal √3/(πρ) (at most ${fmt(sm.fisher_f0_over_ideal_max)}×). The averaged Hessian has a non-positive diagonal in ${sm.nonpositive_sources} of ${sm.sources} sources (${bad.join(", ")}), and the eigenvalue floor binds in ${sm.floored_injections} of ${sm.injections} injections.`,
    `<b>So the reference can be trusted as the flow's score:</b> the joint chain where it exists, at the paper's setting, with seed differences of a few percent in width and up to ${fmt(Math.max(...LONG.map(String).map(lateDrift)), 2)} half-widths in median within its own noise. The Fisher forecast is a start point for the chains, not a reference.`,
  ];
  document.getElementById("means").innerHTML = items.map(v => `<li>${v}</li>`).join("");
}

function render() {
  const c = context(), long = LONG.includes(S.window);
  tiles(); legends(); cornerPanel(c); pairPanel(c); widthPanel(c);
  document.getElementById("conv-body").hidden = !long;
  document.getElementById("note-d").innerHTML = long
    ? `The four long chains of window ${S.window} (64 walkers each, ${big(DATA.convergence.chain.iterations)} iterations from iteration 0, stored every ${T_().thin}), for ${seriesName(series())}: per seed the 16–84% band and median of the walkers, or walker 0 alone, as offsets from the pooled kept median in kept widths (clipped at ±12). The shaded start is the paper's burn-in. Right: the autocorrelation of the kept draws, with the integrated time τ from the analysis.`
    : `Window ${S.window} has no long chains: pick a blue window or a row below.`;
  if (long) { tracePanel(); convPanel(); }
  convTable(); fisherPanel(); hessianPanel(); texts();
  writeHash();
}
function legends() {
  const shown = SETS.filter(set => S.sets.includes(set.key));
  document.getElementById("legend-a").innerHTML = shown.map(set => set.key === "f" ? legendSwatch(set.colour, "Fisher 1σ, 2σ", true) : legendDot(set.colour, set.text)).join("") || "<span>Nothing selected.</span>";
  document.getElementById("legend-b").innerHTML = legendDot("--s1", "joint seed 0") + legendDot("--s3", "joint seed 1") + legendDot("--s2", "narrowband seed 0") + legendDot("--s2", "narrowband seed 1", true) + `<span><svg width="12" height="12"><path d="M6,1 L11,6 L6,11 L1,6 Z" fill="var(--ref)"/></svg>Fisher</span>`;
}

let pending = false;
function schedule() {  // one render per frame however fast the pickers change
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}
addEventListener("resize", schedule);
addEventListener("hashchange", () => { readHash(); sourceSeg(); Object.values(marks).forEach(m => m()); schedule(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
new MutationObserver(schedule).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
render();
</script>
</body>
</html>
"""


def main() -> int:
    draws = json.loads((DATA / "draws.json").read_text())
    chains = json.loads(next(DATA.glob("joint-*/chains.json")).read_text())
    convergence = DATA / "convergence"
    summary = json.loads((convergence / "summary.json").read_text())
    traces = json.loads((convergence / "traces.json").read_text())
    diagonal = json.loads((DATA / "fisher-diagonal.json").read_text())
    numbers = (
        summarise_agreement(draws, chains)
        | summarise_convergence(summary, np.load(convergence / "curves.npz"))
        | summarise_fisher(draws["windows"], diagonal)
    )
    keys = (
        "window_snr",
        "source_snr",
        "log10_hessian_diag",
        "f0_width_bins",
        "ideal_width_bins",
        "precision_decades",
        "floored_eigenvalues",
    )
    fisher = dict(
        draws=diagonal["fisher_draws"],
        **{
            name: dict(
                seed=band["seed"],
                log10_prior_diag=sig(band["log10_prior_diag"]),
                injections=[
                    {k: w[k] if k == "floored_eigenvalues" else sig(w[k]) for k in keys}
                    for w in band["injections"]
                ],
            )
            for name, band in diagonal["bands"].items()
        },
    )
    page = dict(
        built=datetime.date.today().isoformat(),
        draws=draws,
        chains=chains,
        convergence=dict(
            criterion=summary["criterion"],
            chain=summary["chain"],
            windows=summary["windows"],
            curves=curves(convergence / "curves.npz"),
        ),
        traces=traces,
        fisher=fisher,
        agree=AGREE,
        faint_snr=FAINT_SNR,
        summary=numbers,
    )
    html = PAGE.replace("__DATA__", json.dumps(page, separators=(",", ":")))
    (REPORT / "mcmc.html").write_text(html)
    print(json.dumps(numbers, indent=1))
    print(f"mcmc.html: {len(html) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
