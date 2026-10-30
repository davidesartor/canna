"""Write fisher-scorecard.html from data/*.json: the f0 scorecard (width over the Fisher bound, pulls, coverage, ranks) and flow-over-Fisher widths, per setting.

Stdlib only; run from the repo root:
    uv run --frozen python reports/posterior-eval/fisher-scorecard/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]


def settings() -> list[dict]:
    """Every data/<setting>.json, runs first, then mocks by band and mock."""
    out = [json.loads(p.read_text()) for p in sorted((REPORT / "data").glob("*.json"))]
    # records from before the (f0, fdot, A) chart hold 10 eval injections (40 sources): no run eval of the current layout yet
    out = [d for d in out if d["n_eval"] == 256]
    return sorted(out, key=lambda d: (d["group"] != "run", d["setting"]))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fisher scorecard</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --warn: #fab219; --warn-ink: #0b0b0b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 24px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 12px 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
a { color: var(--s1); }
.sub { color: var(--ink-2); margin: 0 0 16px; max-width: 78ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 88ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.card + .card { margin-top: 16px; }
.top { position: sticky; top: 0; z-index: 10; background: var(--page); padding: 8px 0; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
.bar { display: flex; flex-wrap: wrap; gap: 6px 20px; align-items: center; margin: 4px 0; }
.bar .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.sib { font-size: 13px; color: var(--ink-2); display: flex; flex-wrap: wrap; gap: 2px 12px; margin-top: 6px; }
.sib b { color: var(--ink); font-weight: 600; }
.banner { border: 1px solid var(--border); border-left: 4px solid var(--warn); background: var(--surface); border-radius: 8px; padding: 10px 14px; margin: 0 0 16px; font-size: 14px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 4px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
table { border-collapse: collapse; font-size: 13px; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
.scroll { overflow-x: auto; }
tr.hl td { background: color-mix(in srgb, var(--warn) 18%, transparent); }
.yes { color: var(--s2); font-weight: 600; }
.no { color: var(--muted); }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 84ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<div class="top">
  <div class="bar"><span class="k">Setting</span><span class="seg" id="group"></span>
    <span id="runbar"><span class="seg" id="run"></span></span>
    <span id="mockbar"><span class="k">Band</span><span class="seg" id="band"></span> <span class="k">Mock</span><span class="seg" id="mock"></span></span></div>
  <div class="sib"><b>Posterior eval:</b>
    <a href="../test-power/test-power.html">test-power</a><a href="../sampler-convergence/sampler-convergence.html">sampler-convergence</a><a href="../robustness/robustness.html">robustness</a><a href="../mock-flows/mock-flows.html">mock-flows</a><a href="../reference-comparison/reference-comparison.html">reference-comparison</a><b>fisher-scorecard</b><a href="../calibration/calibration.html">calibration</a><a href="../data-space-checks/data-space-checks.html">data-space-checks</a><a href="../proper-scores/proper-scores.html">proper-scores</a></div>
</div>
<h1>Is the flow as sharp as the Fisher bound, and are its widths honest?</h1>
<p class="sub">The eval's scorecard scores each source's f0 draws against the truth: width over the monochromatic Fisher bound, the offset in widths, and how often the truth lies inside the central 68% and 95%. Where the eval kept its draws, the width of every parameter is also compared with the repo's Fisher forecast.</p>
<div class="banner" id="banner"></div>
<div class="tiles" id="tiles"></div>

<div class="card">
  <h2>How wide is f0 against the Fisher bound?</h2>
  <p class="note">One dot per source: the f0 width (1.4826 × MAD of its matched draws) over <code>ideal</code> = √3 / (π·SNR) bins, the monochromatic bound, against the source's SNR. Dashed line: 1.0. Dotted line: the fisher mock's median ratio (narrow band), so a flow at 1.0 is slightly sharper than a calibrated Gaussian. Horizontal ticks are the median per SNR band.</p>
  <div class="legend" id="lg"></div>
  <div id="p-width"></div>
</div>

<div class="card">
  <h2>Are the offsets consistent with the widths?</h2>
  <p class="note">Left: pull = (median of the matched f0 draws − truth) / width, against SNR; a calibrated Gaussian has 68% of dots within ±1 and median |pull| 0.67. Right: all pulls as a histogram, against N(0, 1). Pulls use <code>match_sources</code>, which relabels each draw toward the truth, so they err on the optimistic side.</p>
  <div class="two"><div id="p-pull"></div><div id="p-hist"></div></div>
</div>

<div class="card">
  <h2>Is the truth inside the credible intervals?</h2>
  <p class="note">Left: share of sources whose truth lies in the central 68% and 95% of their f0 draws, per SNR band, with 95% Wilson whiskers (sources in one window are treated as independent) and the nominal levels dashed. Right: the histogram of the truth's rank among the draws (flat when calibrated; U is too narrow, hump too wide), with the ±2σ binomial band.</p>
  <div class="bar"><span class="k">Injections</span><span class="seg" id="subset"></span></div>
  <div class="two"><div id="p-cov"></div><div id="p-rank"></div></div>
</div>

<div class="card" id="card-fisher">
  <h2>How wide is each parameter against the Fisher forecast?</h2>
  <p class="note" id="n-fisher"></p>
  <div id="fisher-body">
    <div class="bar"><span class="k">Parameter</span><span class="seg" id="param"></span></div>
    <div id="p-fisher"></div>
    <div class="scroll"><table id="t-fisher"></table></div>
  </div>
</div>

<div class="card">
  <h2>Does the scorecard see each mock's defect?</h2>
  <p class="note">The mocks are a fixed check, not findings. Each row is one mock of the chosen band. Columns: median f0 width over <code>ideal</code> and in68 for sources below SNR 40 and at or above 100 (all injections), the f0 flow-over-Fisher median over the 64 eval injections, and whether each view flags the defect: <b>width</b> when the f0 flow/Fisher ratio is outside 0.8–1.25, <b>coverage</b> when overall in68 is outside 0.58–0.78.</p>
  <div class="bar"><span class="k">Band</span><span class="seg" id="mband"></span></div>
  <div class="scroll"><table id="t-mock"></table></div>
</div>

<div class="card means">
  <h2>Caveats and how to read this</h2>
  <ul id="caveats"></ul>
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const BY = Object.fromEntries(D.map(d => [d.setting, d]));
const RUNS = D.filter(d => d.group === "run"), MOCKS = D.filter(d => d.group === "mock");
const BANDS = ["narrow", "full"].filter(b => MOCKS.some(d => d.band === b)), MOCKNAMES = ["fisher", "wide", "narrow", "biased", "prior", "phases-prior", "loud-floor"].filter(m => MOCKS.some(d => d.mock === m));
const PARAMS = ["f0", "fdot", "A", "lon", "lat", "psi", "iota", "phi0"];
const EDGES = [[0, 15], [15, 40], [40, 100], [100, Infinity]];
const S = {group: RUNS.length ? "run" : "mock", run: RUNS.length ? RUNS[0].setting : "", band: BANDS[0], mock: "fisher", subset: "all", param: "f0", mband: BANDS[0]};
const NS = "http://www.w3.org/2000/svg";
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const ok = v => v !== null && v !== undefined && Number.isFinite(v);
const fmt = v => !ok(v) ? "–" : Math.abs(v) >= 1e4 ? v.toExponential(1) : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.1 ? v.toFixed(2) : v.toExponential(1);
const med = v => { const s = v.filter(ok).sort((a, b) => a - b); if (!s.length) return NaN; const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const cur = () => S.group === "run" ? BY[S.run] : BY[`mock-${S.band}-${S.mock}`];

function rowsOf(d) {
  const r = d.rows, n = r.snr.length, out = [];
  for (let i = 0; i < n; i++) out.push({snr: r.snr[i], width: r.width[i], ideal: r.ideal[i], offset: r.offset[i], rank: r.rank[i], in68: !!r.in68[i], in95: !!r.in95[i], eval: !!r.is_eval[i], inj: r.injection[i]});
  return out;
}
const subsetRows = rows => S.subset === "all" ? rows : rows.filter(r => (S.subset === "eval") === r.eval);

function svgEl(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function niceTicks(lo, hi, n = 5) {
  const step0 = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(step0))), f = step0 / p;
  const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p, out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9 * step; t += step) out.push(+t.toPrecision(10));
  return out;
}
// a chart frame with linear or log axes; returns scale functions and the svg
function frame(host, {w = 480, h = 300, xlog, ylog, xdom, ydom, xlabel, ylabel}) {
  const m = {l: 52, r: 12, t: 10, b: 38}, svg = svgEl("svg", {viewBox: `0 0 ${w} ${h}`, role: "img"});
  const tx = xlog ? Math.log10 : v => v, ty = ylog ? Math.log10 : v => v;
  const x = v => m.l + (tx(v) - tx(xdom[0])) / (tx(xdom[1]) - tx(xdom[0])) * (w - m.l - m.r);
  const y = v => h - m.b - (ty(v) - ty(ydom[0])) / (ty(ydom[1]) - ty(ydom[0])) * (h - m.t - m.b);
  const ticks = (dom, log) => {
    if (!log) return niceTicks(dom[0], dom[1]);
    const out = []; for (let e = Math.ceil(Math.log10(dom[0]) - 1e-9); e <= Math.floor(Math.log10(dom[1]) + 1e-9); e++) out.push(Math.pow(10, e));
    return out;
  };
  for (const t of ticks(ydom, ylog)) {
    svg.append(svgEl("line", {x1: m.l, x2: w - m.r, y1: y(t), y2: y(t), stroke: "var(--grid)"}));
    svg.append(svgEl("text", {x: m.l - 6, y: y(t) + 4, "text-anchor": "end"}, fmt(t)));
  }
  for (const t of ticks(xdom, xlog)) {
    svg.append(svgEl("line", {x1: x(t), x2: x(t), y1: m.t, y2: h - m.b, stroke: "var(--grid)"}));
    svg.append(svgEl("text", {x: x(t), y: h - m.b + 14, "text-anchor": "middle"}, fmt(t)));
  }
  svg.append(svgEl("line", {x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: "var(--axis)"}));
  svg.append(svgEl("text", {x: (m.l + w - m.r) / 2, y: h - 4, "text-anchor": "middle", class: "lbl"}, xlabel));
  svg.append(svgEl("text", {x: 12, y: (m.t + h - m.b) / 2, "text-anchor": "middle", class: "lbl", transform: `rotate(-90 12 ${(m.t + h - m.b) / 2})`}, ylabel));
  host.replaceChildren(svg);
  const clipId = "c" + Math.random().toString(36).slice(2);
  const defs = svgEl("defs"); const cp = svgEl("clipPath", {id: clipId});
  cp.append(svgEl("rect", {x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b})); defs.append(cp); svg.append(defs);
  const g = svgEl("g", {"clip-path": `url(#${clipId})`}); svg.append(g);
  return {svg, g, x, y, m, w, h};
}
const clampY = (f, ydom, ylog, v) => Math.min(Math.max(v, ydom[0] * (ylog ? 1.0001 : 1)), ydom[1]);
function dot(f, xv, yv, rows, color, filled, title) {
  const c = svgEl("circle", {cx: f.x(xv), cy: f.y(yv), r: filled ? 3.6 : 2.6, fill: filled ? color : "none", "fill-opacity": 0.85, stroke: color, "stroke-width": filled ? 1 : 1.2, "stroke-opacity": 0.8});
  c.append(svgEl("title", {}, title)); return c;
}
function hline(f, v, dash, label, color = "var(--ref)") {
  f.g.append(svgEl("line", {x1: f.m.l, x2: f.w - f.m.r, y1: f.y(v), y2: f.y(v), stroke: color, "stroke-dasharray": dash, "stroke-width": 1}));
  if (label) f.svg.append(svgEl("text", {x: f.w - f.m.r - 2, y: f.y(v) - 3, "text-anchor": "end", class: "lbl"}, label));
}
function legend(items) {
  document.getElementById("lg").innerHTML = items.map(([c, filled, t]) => `<span><svg width="12" height="12" viewBox="0 0 12 12" style="width:12px"><circle cx="6" cy="6" r="4" fill="${filled ? c : "none"}" stroke="${c}" stroke-width="1.5"/></svg>${t}</span>`).join("");
}
function wilson(k, n) {
  const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [c - h, c + h];
}
const logDom = (vals, pad = 1.15) => { const v = vals.filter(x => ok(x) && x > 0); return [Math.min(...v) / pad, Math.max(...v) * pad]; };
const bandOf = s => EDGES.findIndex(([lo, hi]) => s >= lo && s < hi);
function bandTicks(f, rows, yfun, color) {
  EDGES.forEach(([lo, hi], i) => {
    const m = rows.filter(r => bandOf(r.snr) === i); if (!m.length) return;
    const xs = m.map(r => r.snr), a = Math.max(lo, Math.min(...xs)), b = Math.min(hi, Math.max(...xs) * 1.0001);
    const v = med(m.map(yfun)); if (!ok(v) || v <= 0) return;
    f.g.append(svgEl("line", {x1: f.x(a), x2: f.x(b), y1: f.y(v), y2: f.y(v), stroke: color, "stroke-width": 3}));
  });
}

function seg(id, items, key, after) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, t]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${String(S[key]) === String(v)}">${t}</button>`).join("");
  host.onclick = e => { const b = e.target.closest("button"); if (b) { S[key] = b.dataset.v; writeHash(); render(); } };
}
function writeHash() { history.replaceState(null, "", "#" + new URLSearchParams({g: S.group, run: S.run, band: S.band, mock: S.mock, subset: S.subset, param: S.param, mband: S.mband})); }
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1)), set = (k, key, ok2) => { const v = h.get(k); if (v && ok2(v)) S[key] = v; };
  set("g", "group", v => v === "run" ? RUNS.length : MOCKS.length); set("run", "run", v => BY[v]); set("band", "band", v => BANDS.includes(v));
  set("mock", "mock", v => MOCKNAMES.includes(v)); set("subset", "subset", v => ["all", "eval", "random"].includes(v)); set("param", "param", v => PARAMS.includes(v)); set("mband", "mband", v => BANDS.includes(v));
}

function controls() {
  const groups = [];
  if (RUNS.length) groups.push(["run", "Runs"]); if (MOCKS.length) groups.push(["mock", "Mock flows"]);
  seg("group", groups, "group");
  document.getElementById("runbar").style.display = S.group === "run" ? "" : "none";
  document.getElementById("mockbar").style.display = S.group === "mock" ? "" : "none";
  seg("run", RUNS.map(d => [d.setting, `${esc(d.run)} · epoch ${d.epoch} · ${d.ode_steps} steps`]), "run");
  seg("band", BANDS.map(b => [b, b]), "band");
  seg("mock", MOCKNAMES.map(m => [m, m]), "mock");
  seg("subset", [["all", "all"], ["eval", "64 eval"], ["random", "random"]], "subset");
  seg("param", PARAMS.map(p => [p, p]), "param");
  seg("mband", BANDS.map(b => [b, b]), "mband");
}

function render() {
  controls();
  const d = cur(); if (!d) return;
  const rows = rowsOf(d), sub = subsetRows(rows);
  const banner = document.getElementById("banner");
  banner.style.display = "";
  banner.innerHTML = d.group === "run"
    ? `<b>Smoke run.</b> ${esc(d.run)} is a 2,000-step narrow-XS pipeline test stopped at epoch ${d.epoch}: its widths are near the prior's and say nothing about a trained flow. Real findings need the narrow-B and full-B runs.`
    : `<b>Mock flow.</b> A linearised-Gaussian stand-in for a posterior (<code>${esc(d.mock)}</code>, ${esc(d.band)} band), run through the repo's own eval. It is a fixed check that each view sees what it should, not a result about any flow.`;
  const found = rows.filter(r => r.width < 1).length / rows.length;
  const tile = (k, v, s) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${s}</div></div>`;
  document.getElementById("tiles").innerHTML =
    tile("Sources scored", rows.length, `${d.n_eval} eval rows + ${d.n_rows - d.n_eval} random`) +
    tile("Median f0 width / ideal", fmt(med(rows.map(r => r.width / r.ideal))), "1.0 = monochromatic bound") +
    tile("Overall in68 / in95", `${(100 * rows.filter(r => r.in68).length / rows.length).toFixed(0)}% / ${(100 * rows.filter(r => r.in95).length / rows.length).toFixed(0)}%`, "nominal 68% / 95%") +
    tile("Found (width < 1 bin)", `${(100 * found).toFixed(0)}%`, d.fisher ? "draws kept: Fisher card on" : "no draws.npz: Fisher card off");

  // width over ideal
  legend([["var(--s2)", true, "eval injections"], ["var(--s1)", false, "random injections"]]);
  const ok1 = rows.filter(r => r.width > 0 && r.ideal > 0);
  const ratios = ok1.map(r => r.width / r.ideal), refRow = BY["mock-narrow-fisher"];
  const xdom = logDom(rows.map(r => r.snr), 1.1), ydom = logDom([...ratios, 0.9, 1.3], 1.3);
  let f = frame(document.getElementById("p-width"), {xlog: true, ylog: true, xdom, ydom, xlabel: "source SNR", ylabel: "f0 width / ideal"});
  hline(f, 1, "5 4", "ideal 1.0"); if (refRow) hline(f, med(rowsOf(refRow).map(r => r.width / r.ideal)), "1 3", "fisher mock");
  for (const r of ok1.filter(r => !r.eval)) f.g.append(dot(f, r.snr, r.width / r.ideal, 0, "var(--s1)", false, `inj ${r.inj}, SNR ${fmt(r.snr)}: width ${fmt(r.width)} bins, ×ideal ${fmt(r.width / r.ideal)}`));
  for (const r of ok1.filter(r => r.eval)) f.g.append(dot(f, r.snr, r.width / r.ideal, 0, "var(--s2)", true, `eval inj ${r.inj}, SNR ${fmt(r.snr)}: width ${fmt(r.width)} bins, ×ideal ${fmt(r.width / r.ideal)}`));
  bandTicks(f, ok1, r => r.width / r.ideal, "var(--ink)");

  // pulls
  const pr = rows.filter(r => r.width > 0).map(r => ({...r, z: r.offset / r.width}));
  const zlim = Math.max(3, Math.min(8, Math.max(...pr.map(r => Math.abs(r.z))) * 1.05));
  f = frame(document.getElementById("p-pull"), {xlog: true, xdom, ydom: [-zlim, zlim], xlabel: "source SNR", ylabel: "pull = offset / width"});
  hline(f, 0, "", ""); hline(f, 1, "5 4", "±1"); hline(f, -1, "5 4", "");
  for (const r of pr.filter(r => !r.eval)) f.g.append(dot(f, r.snr, Math.max(-zlim, Math.min(zlim, r.z)), 0, "var(--s1)", false, `inj ${r.inj}, SNR ${fmt(r.snr)}: pull ${r.z.toFixed(2)}`));
  for (const r of pr.filter(r => r.eval)) f.g.append(dot(f, r.snr, Math.max(-zlim, Math.min(zlim, r.z)), 0, "var(--s2)", true, `eval inj ${r.inj}, SNR ${fmt(r.snr)}: pull ${r.z.toFixed(2)}`));
  const nb = 24, lo = -4, hi = 4, cnt = new Array(nb).fill(0);
  for (const r of pr) { const k = Math.floor((Math.max(lo, Math.min(hi - 1e-9, r.z)) - lo) / (hi - lo) * nb); cnt[k]++; }
  const dens = cnt.map(c => c / pr.length / ((hi - lo) / nb));
  f = frame(document.getElementById("p-hist"), {xdom: [lo, hi], ydom: [0, Math.max(0.5, ...dens) * 1.1], xlabel: "pull (clipped at ±4)", ylabel: "density"});
  dens.forEach((v, k) => f.g.append(svgEl("rect", {x: f.x(lo + k * (hi - lo) / nb) + 0.5, width: Math.max(1, f.x(lo + (k + 1) * (hi - lo) / nb) - f.x(lo + k * (hi - lo) / nb) - 1), y: f.y(v), height: f.y(0) - f.y(v), fill: "var(--s1)", "fill-opacity": 0.7})));
  let path = ""; for (let i = 0; i <= 80; i++) { const z = lo + (hi - lo) * i / 80; path += `${i ? "L" : "M"}${f.x(z)},${f.y(Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI))}`; }
  f.g.append(svgEl("path", {d: path, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5}));

  // coverage and ranks
  const nbands = EDGES.map((e, i) => sub.filter(r => bandOf(r.snr) === i));
  const used = nbands.map((m, i) => [m, i]).filter(([m]) => m.length);
  f = frame(document.getElementById("p-cov"), {xdom: [0, used.length], ydom: [0, 1.05], xlabel: "source SNR band (n)", ylabel: "coverage"});
  hline(f, 0.68, "5 4", ""); hline(f, 0.95, "5 4", "");
  used.forEach(([m, i], j) => {
    for (const [k, key, col] of [[0, "in68", "var(--s1)"], [1, "in95", "var(--s3)"]]) {
      const c = m.filter(r => r[key]).length, p = c / m.length, [wl, wh] = wilson(c, m.length), x0 = j + 0.12 + k * 0.38;
      const bar = svgEl("rect", {x: f.x(x0), width: f.x(j + 0.46) - f.x(j + 0.08), y: f.y(p), height: f.y(0) - f.y(p), fill: col, "fill-opacity": 0.8});
      bar.append(svgEl("title", {}, `${key} ${(100 * p).toFixed(0)}% (${c}/${m.length}), Wilson ${(100 * wl).toFixed(0)}–${(100 * wh).toFixed(0)}%`));
      f.g.append(bar);
      const xm = f.x(x0) + (f.x(j + 0.46) - f.x(j + 0.08)) / 2;
      f.g.append(svgEl("line", {x1: xm, x2: xm, y1: f.y(wl), y2: f.y(wh), stroke: "var(--ink)", "stroke-width": 1.2}));
    }
    const [lo2, hi2] = EDGES[i];
    f.svg.append(svgEl("text", {x: f.x(j + 0.5), y: f.h - f.m.b + 14, "text-anchor": "middle"}, `${lo2}–${isFinite(hi2) ? hi2 : "∞"} (${m.length})`));
  });
  f.svg.append(svgEl("text", {x: f.m.l + 4, y: f.m.t + 10, class: "lbl"}, "blue in68, green in95"));
  const nr = 10, rc = new Array(nr).fill(0); for (const r of sub) rc[Math.min(nr - 1, Math.floor(r.rank * nr))]++;
  const ex = sub.length / nr, sd = Math.sqrt(sub.length * 0.1 * 0.9);
  f = frame(document.getElementById("p-rank"), {xdom: [0, 1], ydom: [0, Math.max(...rc, ex + 2 * sd) * 1.1], xlabel: `rank of the truth among the draws (${sub.length} sources)`, ylabel: "count"});
  f.g.append(svgEl("rect", {x: f.m.l, width: f.w - f.m.l - f.m.r, y: f.y(ex + 2 * sd), height: f.y(ex - 2 * sd) - f.y(ex + 2 * sd), fill: "var(--grid)", "fill-opacity": 0.7}));
  hline(f, ex, "5 4", "");
  rc.forEach((c, k) => { const b = svgEl("rect", {x: f.x(k / nr) + 1, width: f.x(1 / nr) - f.x(0) - 2, y: f.y(c), height: f.y(0) - f.y(c), fill: "var(--s1)", "fill-opacity": 0.8}); b.append(svgEl("title", {}, `rank ${(k / nr).toFixed(1)}–${((k + 1) / nr).toFixed(1)}: ${c}`)); f.g.append(b); });

  fisherCard(d);
  mockTable();
  caveats(d);
}

function fisherCard(d) {
  const note = document.getElementById("n-fisher"), body = document.getElementById("fisher-body");
  if (!d.fisher) {
    note.textContent = "This setting's eval dir has no draws.npz, so there is no flow-over-Fisher comparison. Re-running the eval with draws kept (or a later run that has them) fills this card.";
    body.style.display = "none"; return;
  }
  body.style.display = "";
  note.innerHTML = "Per source and parameter on the 64 eval-grid injections: the flow's width (1.4826 × MAD of its <code>matched</code> draws, angles wrapped around the truth) over the width of the repo's <code>fisher_samples</code> forecast, against the source's SNR. 1.0 means as sharp as the Fisher forecast; below 1 is sharper than it; a floor shows as the ratio rising with SNR. <code>fisher</code> is centred on the truth and is a width yardstick only, not a calibrated reference.";
  const k = PARAMS.indexOf(S.param), pts = [];
  d.fisher.snr.forEach((row, i) => row.forEach((s, j) => pts.push({snr: s, ratio: d.fisher.ratio[i][j][k], i, j})));
  const f = frame(document.getElementById("p-fisher"), {xlog: true, ylog: true, xdom: logDom(pts.map(p => p.snr), 1.1), ydom: logDom([...pts.map(p => p.ratio), 0.8, 1.25], 1.3), xlabel: "source SNR", ylabel: `${S.param} width / Fisher width`});
  hline(f, 1, "5 4", "Fisher 1.0");
  const colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--ink-2)"];
  for (const p of pts) f.g.append(dot(f, p.snr, p.ratio, 0, colors[p.j], true, `eval inj ${p.i}, source ${p.j}, SNR ${fmt(p.snr)}: ${fmt(p.ratio)}× Fisher`));
  f.svg.append(svgEl("text", {x: f.m.l + 4, y: f.m.t + 10, class: "lbl"}, "colour = source slot (loudest first)"));
  const t = document.getElementById("t-fisher");
  t.innerHTML = "<tr><th>median flow / Fisher</th>" + PARAMS.map(p => `<th>${p}</th>`).join("") + "</tr>" +
    [["SNR &lt; 100", "low"], ["SNR ≥ 100", "loud"], ["all", "all"]].map(([l, key]) => `<tr><td>${l}</td>${PARAMS.map(p => `<td>${fmt(d.fisher.median_ratio[p][key])}</td>`).join("")}</tr>`).join("");
}

function overall(d, key) { const r = d.rows[key]; return r.reduce((a, b) => a + b, 0) / r.length; }
function bandRow(d, lo, hi) {
  const rows = rowsOf(d).filter(r => r.snr >= lo && r.snr < hi && r.width > 0);
  return {ratio: med(rows.map(r => r.width / r.ideal)), in68: rows.filter(r => r.in68).length / (rows.length || NaN)};
}
function mockTable() {
  const t = document.getElementById("t-mock"), rows = MOCKNAMES.map(m => BY[`mock-${S.mband}-${m}`]).filter(Boolean);
  const head = "<tr><th>mock</th><th>width/ideal SNR&lt;40</th><th>width/ideal ≥100</th><th>in68 SNR&lt;40</th><th>in68 ≥100</th><th>overall in68</th><th>f0 flow/Fisher</th><th>width view flags</th><th>coverage view flags</th></tr>";
  const body = rows.map(d => {
    const lowB = bandRow(d, 0, 40), hiB = bandRow(d, 100, Infinity), o68 = overall(d, "in68");
    const fr = d.fisher ? d.fisher.median_ratio.f0.all : NaN, w = ok(fr) && (fr < 0.8 || fr > 1.25), c = o68 < 0.58 || o68 > 0.78;
    return `<tr${d.mock === "loud-floor" ? ' class="hl"' : ""}><td>${d.mock}</td><td>${fmt(lowB.ratio)}</td><td>${fmt(hiB.ratio)}</td><td>${fmt(lowB.in68)}</td><td>${fmt(hiB.in68)}</td><td>${fmt(o68)}</td><td>${fmt(fr)}</td><td class="${w ? "yes" : "no"}">${w ? "yes" : "no"}</td><td class="${c ? "yes" : "no"}">${c ? "yes" : "no"}</td></tr>`;
  }).join("");
  t.innerHTML = head + body;
}
function caveats(d) {
  const fm = BY["mock-narrow-fisher"], pm = BY["mock-narrow-prior"], lm = BY["mock-narrow-loud-floor"], bm = BY["mock-narrow-biased"];
  const li = [];
  if (fm) li.push(`<b>The ideal is the monochromatic bound.</b> The narrow fisher mock, which carries the repo's full Fisher forecast plus the prior, sits at ${fmt(med(rowsOf(fm).map(r => r.width / r.ideal)))}× it over all sources${rowsOf(fm).some(r => r.snr >= 100) ? ` (SNR ≥ 100: ${fmt(bandRow(fm, 100, Infinity).ratio)}×)` : " (no source reaches SNR 100, or even 15: the mocks ran with visible_snr = 0, so they are read at faint SNR only)"}. At faint SNR the width is the prior's, not the bound's, so a ratio away from 1.0 is not a defect there. Full band: ${BY["mock-full-fisher"] ? fmt(med(rowsOf(BY["mock-full-fisher"]).map(r => r.width / r.ideal))) + "× over all sources" : "n/a"}.`);
  if (pm) li.push(`<b><code>rank</code> and <code>offset</code> are optimistic.</b> <code>match_sources</code> relabels each draw toward the truth. The prior mock ignores the data yet scores overall in68 ${fmt(overall(pm, "in68"))} and in95 ${fmt(overall(pm, "in95"))}; only its width (${fmt(med(rowsOf(pm).map(r => r.width / r.ideal)))}× ideal) shows it. Coverage that looks nominal on a wide posterior proves nothing.`);
  li.push("<b><code>fisher</code> is centred on the truth.</b> It is a width yardstick, never a calibrated reference: a flow cannot know the truth, and the fisher mock (centred on the data's MAP) is the calibrated stand-in.");
  if (lm) li.push(rowsOf(lm).some(r => r.snr >= 100) ? `<b>Coverage hides a width floor.</b> Loud-floor reads conservative (overall in68 ${fmt(overall(lm, "in68"))}), but its width over ideal reaches ${fmt(bandRow(lm, 100, Infinity).ratio)}× at SNR ≥ 100 and its f0 flow/Fisher ${fmt(lm.fisher.median_ratio.f0.loud)}×. A real flow with such a floor would pass a coverage-only reading.` : "<b>Loud-floor is not exercised.</b> Its width cap starts at source SNR 18.4 and no eval source reaches it, so it equals the fisher mock here; rerun the mocks once loud sources exist.");
  if (bm) li.push(`<b>A shift shows in pulls and coverage, not in width.</b> The biased mock keeps its width at ${fmt(med(rowsOf(bm).map(r => r.width / r.ideal)))}× ideal and drops overall in68 to ${fmt(overall(bm, "in68"))}.`);
  li.push("<b>Sources are not independent.</b> Four sources share a window, and the 64 eval injections are 8 windows × 8 SNR ranks, a stratified set: whiskers and the rank band assume independent sources.");
  document.getElementById("caveats").innerHTML = li.map(x => `<li>${x}</li>`).join("");
}

addEventListener("hashchange", () => { readHash(); render(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
readHash();
render();
</script>
</body>
</html>
"""


def main() -> None:
    data = settings()
    page = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = REPORT / "fisher-scorecard.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB, {len(data)} settings)")


if __name__ == "__main__":
    main()
