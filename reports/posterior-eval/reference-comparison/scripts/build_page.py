"""reference-comparison.html from data/<setting>.json (reference_probe.py): the flow's draws against the MCMC reference.

Stdlib only; run from the repo root:
    uv run --frozen python reports/posterior-eval/reference-comparison/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]
DATA = REPORT / "data"

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reference comparison</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --ref: #52514e;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --ref: #c3c2b7;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
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
.controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; margin: 0 0 16px; }
.controls .k, .row .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 12px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
button.plain { font: inherit; font-size: 13px; padding: 5px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--ink-2); cursor: pointer; }
.flowrow { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
.legend svg { width: auto; flex: none; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
.side .seg { flex-wrap: wrap; }
.side .stack { display: block; margin-bottom: 14px; }
.side .stack .k { display: block; color: var(--ink-2); font-size: 13px; margin-bottom: 4px; }
.side .d { color: var(--muted); font-size: 12px; margin-top: 4px; }
.seg.wrap { flex-wrap: wrap; }
.bar { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: center; margin-bottom: 16px; }
.bar .d { color: var(--muted); font-size: 13px; max-width: 80ch; }
.bar .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.nav { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; margin: 0 0 12px; }
.nav a { color: var(--ink-2); }
.nav a.here { color: var(--ink); font-weight: 600; text-decoration: none; }
.banner { border-left: 4px solid var(--s2); margin-bottom: 16px; }
.banner b { font-weight: 600; }
.empty { color: var(--muted); font-size: 13px; padding: 24px 0; text-align: center; }
.half { flex: 1 1 360px; min-width: 0; }
.tablewrap { overflow-x: auto; margin-top: 12px; }
table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 3px 8px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--grid); }
th { color: var(--ink-2); font-weight: 500; }
td:first-child, th:first-child { text-align: left; color: var(--ink-2); }
td.off { color: var(--s2); }
.click { cursor: pointer; }
#tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--page); font-size: 12px; padding: 4px 8px; border-radius: 6px; display: none; z-index: 10; max-width: 320px; }
#err { display: none; background: #c62828; color: #fff; padding: 8px 16px; font: 13px ui-monospace, monospace; }
ul.means { margin: 8px 0 0; padding-left: 20px; }
ul.means li { margin-bottom: 8px; max-width: 90ch; }
</style>
</head>
<body>
<div id="err"></div>
<div id="tip"></div>
<main>
<h1>Where does the flow differ from the MCMC reference?</h1>
<nav class="nav" id="nav"></nav>
<p class="sub">Posterior draws set against the MCMC reference: which parameters come out too wide, whether that grows with SNR, whether the truth still lands where the flow says, and what the flow's posterior looks like next to the joint MCMC of a whole window. Pick a setting (a run, or a mock with a known width) in the bar; every card follows it.</p>
<div class="card bar"><div><span class="k">Setting</span><span class="seg wrap" id="setting"></span></div><div class="d" id="setting-d"></div></div>
<div class="card banner" id="banner"></div>
<div class="layout">
<aside class="card side">
  <div class="stack"><span class="k">Reference</span><span class="seg" id="ref"></span><div class="d" id="ref-d"></div></div>
  <div class="stack"><span class="k">Parameter</span><span class="seg" id="param"></span><div class="d">Pinned φ± is whichever of φ₊ = φ₀ + 2ψ, φ₋ = φ₀ − 2ψ the reference measures tighter.</div></div>
  <div class="stack"><span class="k">Selected source</span><div id="src-d" class="d"></div><button type="button" class="plain" id="clear">Clear</button></div>
</aside>
<div class="content">
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <h2>Which parameters is the flow too wide in?</h2>
    <p class="note">Each dot is one source: the flow's draw width over the reference's, on a log axis; the gray line at 1 is the reference and the dark tick each row's median. Read the pinned φ± row before φ₀: φ₀ inherits ψ's spread, so its marginal hides how wide the combination the data measure is. Click a row to pick the parameter, a dot to pick the source.</p>
    <div id="p-strip"></div>
  </div>
  <div class="card">
    <h2>Does the gap grow with SNR?</h2>
    <p class="note" id="snr-note"></p>
    <div class="legend" id="snr-legend"></div>
    <div id="p-snr"></div>
  </div>
  <div class="card">
    <h2>Does the truth fall where the flow says?</h2>
    <p class="note">Left: where each eval source's truth ranks among the flow's draws, for the selected parameter (pinned φ± is φ₋ for sin ι &gt; 0, φ₊ otherwise; the phases are cut opposite the draws' circular mean). A calibrated flow gives a flat histogram inside the gray 95% band. Right: the fraction of sources whose truth falls in the flow's central 68% and 95%, per parameter, against the nominal level and its 95% binomial band. This needs no MCMC, so it covers every band.</p>
    <div class="legend"><span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--s1)"/></svg>central 68%</span><span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--s2)"/></svg>central 95%</span><span><svg width="22" height="10"><rect x="1" y="1" width="20" height="8" fill="var(--grid)"/></svg>calibrated, 95% band</span></div>
    <div class="flowrow"><div class="half" id="p-rank"></div><div class="half" id="p-cover"></div></div>
  </div>
  <div class="card">
    <h2>What does the flow's posterior look like next to the joint MCMC?</h2>
    <p class="note" id="pair-note">Offsets from the truth of the flow's draws (blue) and the joint MCMC's (gray), which samples every source of the window together; the Fisher Gaussian (orange) is optional. The marginals are each scaled to their own peak. The table holds the full comparison for the source.</p>
    <div class="controls">
      <div><span class="k">Window</span><span class="seg" id="window"></span></div>
      <div><span class="k">Source</span><span class="seg" id="source"></span></div>
    </div>
    <div class="controls">
      <div><span class="k">x</span><span class="seg wrap" id="x"></span></div>
      <div><span class="k">y</span><span class="seg wrap" id="y"></span></div>
    </div>
    <div class="controls">
      <div><span class="k">Zoom</span><span class="seg" id="zoom"></span></div>
      <div><span class="k">Fisher</span><span class="seg" id="fisher"></span></div>
    </div>
    <div class="legend" id="pair-legend"></div>
    <div id="p-pair"></div>
    <div class="tablewrap" id="t-pair"></div>
  </div>
  <div class="card">
    <h2>What this means</h2>
    <ul class="means" id="means"></ul>
  </div>
</div>
</div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
window.onerror = (msg, src, line) => {
  const e = document.getElementById("err");
  e.style.display = "block";
  e.textContent = `Page error: ${msg} (line ${line})`;
};
const D = JSON.parse(document.getElementById("data").textContent);
const P = D.parameters;
const SHORT = ["f₀", "M_c", "A", "RA", "Dec", "ψ", "sin ι", "φ₀", "φ₊", "φ₋"];
const PIN = -1;
const PHI0 = 7, PHI_PLUS = 8, PHI_MINUS = 9;
const S = {setting: D.settings.length ? D.settings[0].setting : "", ref: "source", param: PIN, src: "", window: -1, source: 0,
           x: PHI_PLUS, y: PHI_MINUS, zoom: "mcmc", fisher: 0};
const NAV = ["test-power", "sampler-convergence", "robustness", "mock-flows", "reference-comparison", "fisher-scorecard", "calibration", "data-space-checks", "proper-scores"];
const NAV_HTML = NAV.map(n => n === "reference-comparison" ? `<a class="here">${n}</a>` : `<a href="../${n}/${n}.html">${n}</a>`).join("");

// page state from the URL hash: #setting=...&param=...
for (const part of location.hash.slice(1).split("&")) {
  const [k, v] = part.split("=").map(decodeURIComponent);
  if (k in S && v !== undefined) S[k] = typeof S[k] === "number" ? Number(v) : v;
}
const writeHash = () => history.replaceState(null, "", "#" + Object.entries(S).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&"));

const $ = id => document.getElementById(id);
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN; };
const quantile = (a, q) => { const s = [...a].sort((x, y) => x - y); if (!s.length) return NaN; const i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo); };
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const fmtX = v => !isFinite(v) ? "–" : v >= 100 ? `${Math.round(v)}×` : v >= 10 ? `${v.toFixed(0)}×` : v >= 1 ? `${v.toFixed(1)}×` : `${v.toFixed(2)}×`;
const fmt = (v, d = 3) => v === null || v === undefined || !isFinite(v) ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : Number(v.toPrecision(d)).toString();
const pname = p => p === PIN ? "pinned φ±" : P[p];
const tip = text => `data-tip="${String(text).replace(/"/g, "&quot;")}"`;

const R = () => D.settings.find(r => r.setting === S.setting) || D.settings[0];
const hasJoint = r => r.joint.length > 0;
const pinnedOf = ref => ref[PHI_PLUS] <= ref[PHI_MINUS] ? PHI_PLUS : PHI_MINUS;

// flow/reference width ratios per source for the chosen reference
function comparisons(r) {
  if (S.ref === "joint") {
    const out = [];
    r.joint.forEach(w => w.sources.forEach(s => out.push({key: `${w.window}_${s.s}`, j: w.window, s: s.s, snr: s.snr,
      ratio: s.rows["flow/mcmc width"], ref: s.rows["mcmc width"]})));
    return out;
  }
  if (!r.widths) return [];
  return r.widths.sources.filter(s => s.mcmc).map(s => ({key: `${s.injection}_${s.source}`, j: s.injection, s: s.source, snr: s.snr,
    ratio: s.flow.map((v, i) => v / s.mcmc[i]), ref: s.mcmc}));
}
const ratioOf = (c, p) => p === PIN ? c.ratio[pinnedOf(c.ref)] : c.ratio[p];

function logTicks(lo, hi) {
  const out = [], e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi));
  const mults = e1 - e0 <= 2 ? [1, 2, 5] : [1];
  for (let e = e0; e <= e1; e++) mults.forEach(m => { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); });
  return out;
}
function ticks(lo, hi, count) {
  const raw = Math.max((hi - lo) / Math.max(count, 1), 1e-12), e = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(m => m * e).find(s => s >= raw);
  const out = [];
  for (let i = Math.ceil(lo / step); i * step <= hi + 1e-12; i++) out.push(i * step);
  return out;
}
const tickLabel = v => v >= 1000 ? `${v / 1000}k` : String(Number(v.toPrecision(3)));
const logScale = (lo, hi, a, b) => v => a + (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (b - a);
const linScale = (lo, hi, a, b) => v => a + (v - lo) / (hi - lo) * (b - a);
function logRange(values, pad = 1.4) {
  const v = values.filter(x => isFinite(x) && x > 0);
  if (!v.length) return [0.1, 10];
  return [Math.min(...v, 1) / pad, Math.max(...v, 1) * pad];
}
function xAxis(X, values, y, x0, x1, label, log) {
  let s = `<line x1="${x0}" x2="${x1}" y1="${y}" y2="${y}" stroke="var(--axis)"/>`;
  values.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${y}" y2="${y + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${y + 16}" text-anchor="middle">${log ? tickLabel(v) : fmt(v, 3)}</text>`);
  return s + `<text x="${(x0 + x1) / 2}" y="${y + 31}" text-anchor="middle">${label}</text>`;
}
function yAxis(Y, values, x, y0, y1, label, log) {
  let s = `<line x1="${x}" x2="${x}" y1="${y0}" y2="${y1}" stroke="var(--axis)"/>`;
  values.forEach(v => s += `<line x1="${x - 4}" x2="${x}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--axis)"/><text x="${x - 7}" y="${Y(v) + 4}" text-anchor="end">${log ? tickLabel(v) : fmt(v, 3)}</text>`);
  return s + `<text transform="translate(${x - 44},${(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">${label}</text>`;
}
const gridX = (X, values, y0, y1) => values.map(v => `<line x1="${X(v)}" x2="${X(v)}" y1="${y0}" y2="${y1}" stroke="var(--grid)"/>`).join("");
const gridY = (Y, values, x0, x1) => values.map(v => `<line x1="${x0}" x2="${x1}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/>`).join("");
const svg = (W, H, body) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${body}</svg>`;
const empty = text => `<div class="empty">${text}</div>`;
const dot = (fill, r = 4) => `<svg width="12" height="12"><circle cx="6" cy="6" r="${r}" fill="${fill}"/></svg>`;
const ring = stroke => `<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="${stroke}" stroke-width="1.5"/></svg>`;
const line = (stroke, dash = "") => `<svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${stroke}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ""}/></svg>`;
const jitter = key => { let h = 0; for (const c of key) h = (h * 31 + c.charCodeAt(0)) % 997; return (h % 9 - 4) * 1.4; };

const marks = {};
function seg(id, items, key, after) {
  const host = $(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    S[key] = typeof S[key] === "number" ? Number(b.dataset.v) : b.dataset.v;
    if (after) after();
    schedule();
  }));
}

function selectSource(key) {
  S.src = key;
  const [j, s] = key.split("_").map(Number);
  if (R().joint.some(w => w.window === j)) { S.window = j; S.source = s; }
}

// a run's sampler, training length and smoke label
function runFacts(r) {
  return {evalName: r.eval, epoch: r.epoch, steps: r.total_steps, smoke: r.smoke, ode: r.ode_steps, mock: r.kind === "mock"};
}

const MOCKS = {
  "mock-fisher": "the linearised Gaussian (Fisher + prior) posterior around the data's MAP; width ratio near 1 where the posterior is Gaussian.",
  "mock-wide": "fisher with every width x3; the ratio should read about 3.",
  "mock-narrow": "fisher with every width x0.5; the ratio should read about 0.5.",
  "mock-biased": "fisher with f0 shifted +1 sigma; widths right, f0 off-centre.",
  "mock-prior": "an identity flow: prior draws that ignore the data.",
  "mock-phases-prior": "fisher with the z+- phases redrawn from the prior.",
  "mock-loud-floor": "widths floored at the loud-source f0 floor: the ratio grows with SNR.",
};
function renderSide() {
  $("nav").innerHTML = NAV_HTML;
  seg("setting", D.settings.map(r => [r.setting, r.label]), "setting", () => { S.src = ""; S.window = -1; S.source = 0; });
  const r = R(), f = runFacts(r);
  $("setting-d").innerHTML = f.mock ? `Mock: ${MOCKS[r.setting] || ""} Its draws are a stand-in for a flow, not a trained network.` : `<code>${r.run}</code>, ${f.steps.toLocaleString("en-US")} training steps${f.smoke ? ", a smoke run" : ""}.`;
  const nSource = r.widths ? r.widths.sources.filter(s => s.mcmc).length : 0;
  const nJoint = r.joint.reduce((n, w) => n + w.sources.length, 0);
  seg("ref", [["source", `all injections (${nSource})`], ["joint", `pair-card injections (${nJoint})`]], "ref");
  $("ref-d").textContent = S.ref === "joint"
    ? "The joint MCMC of the four injections in the pair card below, every source of an injection sampled together."
    : "Every injection with cached joint MCMC chains; each source's widths are marginals of its injection's joint chains, relabelled onto the truth.";
  seg("param", [[PIN, "pinned φ±"], ...P.map((p, i) => [i, SHORT[i]])], "param");
  $("src-d").textContent = S.src ? (() => { const [j, s] = S.src.split("_"); const w = r.widths && r.widths.sources.find(x => `${x.injection}_${x.source}` === S.src); return `injection ${j}, slot ${s}` + (w ? `, SNR ${fmt(w.snr, 3)}, sin ι ${fmt(w.sin_iota, 2)}` : ""); })() : "none: click a dot";
  Object.values(marks).forEach(m => m());
}

function renderBanner() {
  const r = R(), f = runFacts(r), b = $("banner");
  let html = "";
  if (f.smoke) html += `<b>Smoke run.</b> <code>${r.run}</code> is a ${f.steps.toLocaleString("en-US")}-step narrow-XS pipeline test stopped at epoch ${f.epoch}: its numbers show the comparison working, not how good the method is. Real findings need narrow-B and full-B. `;
  if (f.mock) html += `<b>Mock.</b> These draws are not a trained flow: they test that this comparison reads the width the mock was built with. `;
  if (f.ode) html += `<b>Sampler.</b> The draws come from eval <code>${f.evalName}</code>: ${f.ode} ODE steps on a uniform grid. ${f.ode <= 32 ? "This sampler is known to collapse narrow posteriors late in the path (see sampler-convergence), so a flow width near or below the reference's may be the sampler, not the network; collapse only narrows, so it cannot explain a flow that is too wide." : ""}`;
  b.innerHTML = html;
  b.style.display = html ? "block" : "none";
}

function renderTiles() {
  const r = R(), f = runFacts(r), comps = comparisons(r);
  const tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  const out = [tile("Setting", f.mock ? "mock" : r.config, f.mock ? r.setting : `epoch ${f.epoch}${f.smoke ? " · smoke run" : ""}`)];
  out.push(tile("Sampler", f.ode ? `${f.ode} steps` : "none", f.mock ? "a mock: no ODE" : "uniform grid"));
  const refName = "joint MCMC";
  if (comps.length) {
    const pin = comps.map(c => ratioOf(c, PIN)), phi0 = comps.map(c => c.ratio[PHI0]);
    out.push(tile("Pinned φ± width", fmtX(median(pin)), `flow / ${refName}, median; ${fmtX(Math.min(...pin))}–${fmtX(Math.max(...pin))} over ${comps.length}`));
    out.push(tile("φ₀ width", fmtX(median(phi0)), `flow / ${refName}, median; hides the pinned gap`));
  }
  if (r.widths) {
    const ranks = r.widths.sources.flatMap(s => s.rank.slice(0, 8));
    const inside = q => ranks.filter(v => Math.abs(v - 0.5) <= q / 2).length / ranks.length;
    out.push(tile("Truth coverage", `${Math.round(100 * inside(0.68))}% · ${Math.round(100 * inside(0.95))}%`, `in the central 68% · 95%, ${r.widths.sources.length} sources × 8 parameters`));
  }
  $("tiles").innerHTML = out.join("");
}

// card 1: the flow/reference ratio per parameter, one dot per source
function renderStrip() {
  const host = $("p-strip"), r = R(), comps = comparisons(r);
  if (!comps.length) { host.innerHTML = empty(noReference(r)); return; }
  const rows = [PIN, ...P.map((_, i) => i)];
  const all = rows.flatMap(p => comps.map(c => ratioOf(c, p)));
  const [lo, hi] = logRange(all);
  const W = Math.max(320, host.clientWidth), ML = 92, MR = 16, RH = 26, top = 6, H = top + rows.length * RH + 40;
  const X = logScale(lo, hi, ML, W - MR), tv = logTicks(lo, hi);
  let s = gridX(X, tv, top, top + rows.length * RH);
  s += `<line x1="${X(1)}" x2="${X(1)}" y1="${top}" y2="${top + rows.length * RH}" stroke="var(--ref)" stroke-width="1.5"/>`;
  rows.forEach((p, i) => {
    const y = top + i * RH + RH / 2, sel = p === S.param;
    s += `<rect class="click" data-param="${p}" x="0" y="${top + i * RH}" width="${W}" height="${RH}" fill="${sel ? "var(--grid)" : "transparent"}" opacity="0.6" ${tip(`Pick ${pname(p)}`)}/>`;
    s += `<text class="lbl click" data-param="${p}" x="${ML - 10}" y="${y + 4}" text-anchor="end" style="${sel ? "font-weight:600" : ""}">${p === PIN ? "pinned φ±" : P[p]}</text>`;
    comps.forEach(c => {
      const v = ratioOf(c, p); if (!(v > 0)) return;
      const on = c.key === S.src;
      s += `<circle class="click" data-src="${c.key}" cx="${X(v)}" cy="${y + jitter(c.key)}" r="${on ? 5.5 : 3.5}" fill="var(--s1)" fill-opacity="${on ? 1 : 0.55}" stroke="${on ? "var(--ink)" : "none"}" stroke-width="1.5" ${tip(`injection ${c.j} slot ${c.s}, SNR ${fmt(c.snr, 3)}: ${pname(p)} ${fmtX(v)}`)}/>`;
    });
    const m = median(comps.map(c => ratioOf(c, p)).filter(v => v > 0));
    s += `<line x1="${X(m)}" x2="${X(m)}" y1="${y - 9}" y2="${y + 9}" stroke="var(--ink)" stroke-width="2.5" ${tip(`median ${fmtX(m)}`)}/>`;
  });
  s += xAxis(X, tv, top + rows.length * RH, ML, W - MR, "flow width / joint MCMC width", true);
  host.innerHTML = svg(W, H, s);
}

function noReference(r) {
  if (S.ref === "joint") return "No joint MCMC is cached for these injections at this run's problem and seed: see How to rebuild in the README.";
  return "No joint MCMC is cached for this run's problem and seed: only the truth ranks below apply.";
}

// card 2: ratio against SNR for the selected parameter
function renderSnr() {
  const host = $("p-snr"), r = R(), comps = comparisons(r), p = S.param;
  $("snr-note").textContent = `${pname(p)}: flow width over the joint MCMC's against the source's SNR, both on log axes; the gray line at 1 is the reference. A flat cloud means a constant relative error; a rising one means the reference narrows as 1/SNR while the flow does not keep up.` + (p === PIN ? " The orange rings are φ₀ for the same sources." : "");
  $("snr-legend").innerHTML = `<span>${dot("var(--s1)")}${pname(p)}</span>` + (p === PIN ? `<span>${ring("var(--s2)")}φ₀</span>` : "") + `<span>${line("var(--ref)")}reference</span>`;
  if (!comps.length) { host.innerHTML = empty(noReference(r)); return; }
  const ys = comps.map(c => ratioOf(c, p)).concat(p === PIN ? comps.map(c => c.ratio[PHI0]) : []);
  const [ylo, yhi] = logRange(ys), xs = comps.map(c => c.snr), xlo = Math.min(...xs) / 1.3, xhi = Math.max(...xs) * 1.3;
  const W = Math.max(320, host.clientWidth), ML = 60, MR = 16, top = 8, ph = 260, H = top + ph + 40;
  const X = logScale(xlo, xhi, ML, W - MR), Y = logScale(ylo, yhi, top + ph, top), xt = logTicks(xlo, xhi), yt = logTicks(ylo, yhi);
  let s = gridX(X, xt, top, top + ph) + gridY(Y, yt, ML, W - MR);
  s += `<line x1="${ML}" x2="${W - MR}" y1="${Y(1)}" y2="${Y(1)}" stroke="var(--ref)" stroke-width="1.5"/>`;
  if (p === PIN) comps.forEach(c => { const v = c.ratio[PHI0]; if (v > 0) s += `<circle cx="${X(c.snr)}" cy="${Y(v)}" r="4" fill="none" stroke="var(--s2)" stroke-width="1.5" ${tip(`injection ${c.j} slot ${c.s}: φ₀ ${fmtX(v)}`)}/>`; });
  comps.forEach(c => {
    const v = ratioOf(c, p); if (!(v > 0)) return;
    const on = c.key === S.src;
    s += `<circle class="click" data-src="${c.key}" cx="${X(c.snr)}" cy="${Y(v)}" r="${on ? 6 : 4}" fill="var(--s1)" fill-opacity="${on ? 1 : 0.7}" stroke="${on ? "var(--ink)" : "var(--surface)"}" stroke-width="${on ? 1.5 : 1}" ${tip(`injection ${c.j} slot ${c.s}, SNR ${fmt(c.snr, 3)}: ${pname(p)} ${fmtX(v)}${p === PIN ? ` (${P[pinnedOf(c.ref)]})` : ""}`)}/>`;
  });
  s += xAxis(X, xt, top + ph, ML, W - MR, "source SNR", true) + yAxis(Y, yt, ML, top, top + ph, "flow / MCMC width", true);
  host.innerHTML = svg(W, H, s);
}

// card 3: truth ranks and coverage, every eval source
function rankOf(src, p) { return p === PIN ? src.rank[src.sin_iota > 0 ? PHI_MINUS : PHI_PLUS] : src.rank[p]; }
function renderCalibration() {
  const r = R(), hr = $("p-rank"), hc = $("p-cover");
  if (!r.widths) { hr.innerHTML = empty(noReference(r)); hc.innerHTML = ""; return; }
  const srcs = r.widths.sources, N = srcs.length, p = S.param, bins = 10;
  const counts = new Array(bins).fill(0);
  srcs.forEach(s => counts[Math.min(bins - 1, Math.floor(rankOf(s, p) * bins))]++);
  const exp = N / bins, sd = Math.sqrt(N * (1 / bins) * (1 - 1 / bins));
  let W = Math.max(280, hr.clientWidth), ML = 44, MR = 10, top = 8, ph = 220, H = top + ph + 40;
  const cmax = Math.max(...counts, exp + 2 * sd) * 1.1;
  let X = linScale(0, 1, ML, W - MR), Y = linScale(0, cmax, top + ph, top);
  let s = gridY(Y, ticks(0, cmax, 4), ML, W - MR);
  s += `<rect x="${ML}" y="${Y(exp + 1.96 * sd)}" width="${W - MR - ML}" height="${Y(Math.max(0, exp - 1.96 * sd)) - Y(exp + 1.96 * sd)}" fill="var(--grid)"/>`;
  s += `<line x1="${ML}" x2="${W - MR}" y1="${Y(exp)}" y2="${Y(exp)}" stroke="var(--ref)" stroke-dasharray="4 3"/>`;
  counts.forEach((c, i) => {
    const x0 = X(i / bins) + 1, x1 = X((i + 1) / bins) - 1;
    s += `<rect x="${x0}" y="${Y(c)}" width="${x1 - x0}" height="${Y(0) - Y(c)}" rx="2" fill="var(--s1)" ${tip(`rank ${i / bins}–${(i + 1) / bins}: ${c} sources`)}/>`;
  });
  const sel = srcs.find(x => `${x.injection}_${x.source}` === S.src);
  if (sel) { const xr = X(rankOf(sel, p)); s += `<path d="M${xr},${top + ph + 2} l-5,9 h10 z" fill="var(--ink)" ${tip(`selected source: rank ${fmt(rankOf(sel, p), 2)}`)}/>`; }
  s += xAxis(X, [0, 0.25, 0.5, 0.75, 1], top + ph, ML, W - MR, `rank of the truth among the flow's draws, ${pname(p)}`, false) + yAxis(Y, ticks(0, cmax, 4), ML, top, top + ph, "sources", false);
  hr.innerHTML = svg(W, H, s);

  // coverage per parameter at 68% and 95%
  const rows = [PIN, ...P.map((_, i) => i)], RH = 22;
  W = Math.max(280, hc.clientWidth); ML = 84; H = top + rows.length * RH + 40;
  X = linScale(0, 1, ML, W - MR);
  s = gridX(X, [0, 0.25, 0.5, 0.75, 1], top, top + rows.length * RH);
  [[0.68, "var(--s1)"], [0.95, "var(--s2)"]].forEach(([q]) => {
    const half = 1.96 * Math.sqrt(q * (1 - q) / N);
    s += `<rect x="${X(Math.max(0, q - half))}" y="${top}" width="${X(Math.min(1, q + half)) - X(Math.max(0, q - half))}" height="${rows.length * RH}" fill="var(--grid)" opacity="0.8"/>`;
    s += `<line x1="${X(q)}" x2="${X(q)}" y1="${top}" y2="${top + rows.length * RH}" stroke="var(--ref)" stroke-dasharray="4 3"/>`;
  });
  rows.forEach((pp, i) => {
    const y = top + i * RH + RH / 2, on = pp === p;
    s += `<rect class="click" data-param="${pp}" x="0" y="${top + i * RH}" width="${W}" height="${RH}" fill="${on ? "var(--grid)" : "transparent"}" opacity="0.5"/>`;
    s += `<text class="lbl click" data-param="${pp}" x="${ML - 8}" y="${y + 4}" text-anchor="end" style="${on ? "font-weight:600" : ""}">${pname(pp)}</text>`;
    [[0.68, "var(--s1)"], [0.95, "var(--s2)"]].forEach(([q, c]) => {
      const v = srcs.filter(x => Math.abs(rankOf(x, pp) - 0.5) <= q / 2).length / N;
      s += `<circle class="click" data-param="${pp}" cx="${X(v)}" cy="${y}" r="5" fill="${c}" stroke="var(--surface)" stroke-width="1.5" ${tip(`${pname(pp)}: truth in the central ${q * 100}% for ${Math.round(v * N)} of ${N}`)}/>`;
    });
  });
  s += xAxis(X, [0, 0.25, 0.5, 0.75, 1], top + rows.length * RH, ML, W - MR, "fraction of sources with the truth inside", false);
  hc.innerHTML = svg(W, H, s);
}

// card 4: flow, joint MCMC and Fisher draws of one source in one window
function renderPair() {
  const r = R(), host = $("p-pair"), table = $("t-pair");
  if (!hasJoint(r)) {
    ["window", "source"].forEach(id => $(id).innerHTML = "");
    host.innerHTML = empty("No joint MCMC is cached for these injections at this run's problem and seed: see How to rebuild in the README.");
    table.innerHTML = ""; $("pair-legend").innerHTML = ""; return;
  }
  if (!r.joint.some(w => w.window === S.window)) S.window = r.joint[0].window;
  const w = r.joint.find(x => x.window === S.window);
  if (!w.sources.some(x => x.s === S.source)) S.source = w.sources[0].s;
  const src = w.sources.find(x => x.s === S.source);
  seg("window", r.joint.map(x => [x.window, `${x.window} · SNR ${Math.round(x.window_snr)}`]), "window", () => { S.src = `${S.window}_${S.source}`; });
  seg("source", w.sources.map(x => [x.s, `${x.s} · ${fmt(x.snr, 3)}`]), "source", () => { S.src = `${S.window}_${S.source}`; });
  seg("x", P.map((_, i) => [i, SHORT[i]]), "x");
  seg("y", P.map((_, i) => [i, SHORT[i]]), "y");
  seg("zoom", [["mcmc", "MCMC"], ["all", "all draws"]], "zoom");
  seg("fisher", [[0, "off"], [1, "on"]], "fisher");
  Object.values(marks).forEach(m => m());
  $("pair-legend").innerHTML = `<span>${dot("var(--s1)", 3)}flow</span><span>${dot("var(--ref)", 3)}joint MCMC</span>` + (S.fisher ? `<span>${dot("var(--s2)", 3)}Fisher</span>` : "") + `<span>${line("var(--muted)", "3 3")}truth</span>`;

  const sets = [["mcmc", "var(--ref)", 0.35], ...(S.fisher ? [["fisher", "var(--s2)", 0.35]] : []), ["flow", "var(--s1)", 0.55]];
  const range = k => {
    const pool = S.zoom === "mcmc" ? src.draws.mcmc.map(d => d[k]) : sets.flatMap(([n]) => src.draws[n].map(d => d[k]));
    let lo = quantile(pool, 0.005), hi = quantile(pool, 0.995);
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    const pad = (hi - lo || Math.abs(lo) || 1) * 0.15;
    return [lo - pad, hi + pad];
  };
  const [xlo, xhi] = range(S.x), [ylo, yhi] = range(S.y);
  const Wt = Math.max(320, host.clientWidth), ML = 64, MG = 56, top = MG + 6;
  const pw = Math.min(Wt - ML - MG - 16, 620), ph = Math.round(pw * 0.7), H = top + ph + 42;
  const X = linScale(xlo, xhi, ML, ML + pw), Y = linScale(ylo, yhi, top + ph, top);
  const xt = ticks(xlo, xhi, Math.floor(pw / 90)), yt = ticks(ylo, yhi, 5);
  let s = `<defs><clipPath id="clip"><rect x="${ML}" y="${top}" width="${pw}" height="${ph}"/></clipPath></defs>`;
  s += gridX(X, xt, top, top + ph) + gridY(Y, yt, ML, ML + pw);
  s += `<line x1="${X(0)}" x2="${X(0)}" y1="${top - MG}" y2="${top + ph}" stroke="var(--muted)" stroke-dasharray="3 3"/><line x1="${ML}" x2="${ML + pw + MG}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--muted)" stroke-dasharray="3 3"/>`;
  s += `<g clip-path="url(#clip)">`;
  sets.forEach(([n, c, o]) => src.draws[n].forEach(d => s += `<circle cx="${X(d[S.x]).toFixed(1)}" cy="${Y(d[S.y]).toFixed(1)}" r="1.8" fill="${c}" fill-opacity="${o}"/>`));
  s += `</g>`;
  // marginals, each set scaled to its own peak
  const nb = 40;
  const hist = (vals, lo, hi) => { const h = new Array(nb).fill(0); vals.forEach(v => { const i = Math.floor((v - lo) / (hi - lo) * nb); if (i >= 0 && i < nb) h[i]++; }); const m = Math.max(...h, 1); return h.map(v => v / m); };
  sets.forEach(([n, c]) => {
    // a set with almost no draws in view has no meaningful marginal here
    const inView = src.draws[n].filter(d => d[S.x] >= xlo && d[S.x] <= xhi && d[S.y] >= ylo && d[S.y] <= yhi).length;
    if (inView < 0.05 * src.draws[n].length) return;
    const hx = hist(src.draws[n].map(d => d[S.x]), xlo, xhi), hy = hist(src.draws[n].map(d => d[S.y]), ylo, yhi);
    const px = hx.map((v, i) => `${i ? "L" : "M"}${(X(xlo + (i + 0.5) / nb * (xhi - xlo))).toFixed(1)},${(top - 4 - v * (MG - 10)).toFixed(1)}`).join("");
    const py = hy.map((v, i) => `${i ? "L" : "M"}${(ML + pw + 4 + v * (MG - 10)).toFixed(1)},${(Y(ylo + (i + 0.5) / nb * (yhi - ylo))).toFixed(1)}`).join("");
    const fill = n === "mcmc" ? `fill="${c}" fill-opacity="0.18"` : `fill="none"`;
    s += `<path d="${px}${n === "mcmc" ? `L${X(xhi)},${top - 4}L${X(xlo)},${top - 4}Z` : ""}" ${fill} stroke="${c}" stroke-width="1.5"/>`;
    s += `<path d="${py}${n === "mcmc" ? `L${ML + pw + 4},${Y(yhi)}L${ML + pw + 4},${Y(ylo)}Z` : ""}" ${fill} stroke="${c}" stroke-width="1.5"/>`;
  });
  s += xAxis(X, xt, top + ph, ML, ML + pw, `${P[S.x]} − truth`, false) + yAxis(Y, yt, ML, top, top + ph, `${P[S.y]} − truth`, false);
  const outside = src.draws.flow.filter(d => d[S.x] < xlo || d[S.x] > xhi || d[S.y] < ylo || d[S.y] > yhi).length / src.draws.flow.length;
  if (outside > 0.005) $("pair-legend").innerHTML += `<span>${Math.round(100 * outside)}% of the flow's draws lie outside this view</span>`;
  host.innerHTML = svg(Wt, H, s);

  // the comparison table of this source
  const rowNames = [["mcmc width", "MCMC width"], ["flow/mcmc width", "flow / MCMC width"], ["flow W1/sd", "flow W1 / MCMC width"], ["fisher/mcmc width", "Fisher / MCMC width"], ["seed1/seed0 width", "MCMC seed 1 / seed 0 width"]];
  let t = `<table><tr><th>window ${w.window}, slot ${src.s}, SNR ${fmt(src.snr, 3)}</th>${P.map((p, i) => `<th>${SHORT[i]}</th>`).join("")}</tr>`;
  rowNames.forEach(([k, label]) => t += `<tr><td>${label}</td>${src.rows[k].map(v => `<td>${fmt(v, 3)}</td>`).join("")}</tr>`);
  [["in68", "MCMC median in the flow's 68%"], ["in95", "MCMC median in the flow's 95%"]].forEach(([k, label]) =>
    t += `<tr><td>${label}</td>${P.map((_, i) => i < src[k].length ? `<td class="${src[k][i] ? "" : "off"}">${src[k][i] ? "in" : "out"}</td>` : "<td>–</td>").join("")}</tr>`);
  t += `</table><p class="note">Widths are 1.4826 MAD (circular for φ±), on the folded parameters; W1 is the 1-Wasserstein distance between the marginals over the MCMC width. ${Math.round(100 * src.far)}% of the flow's draws sit more than a frequency bin from the MCMC's median f₀. Setting <code>${r.setting}</code>, eval <code>${r.eval}</code>.</p>`;
  table.innerHTML = t;
}

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// what this means, from the selected run's numbers
function renderMeans() {
  const r = R(), f = runFacts(r), out = [];
  const src = r.widths ? r.widths.sources.filter(s => s.mcmc) : [];
  if (f.smoke) out.push(`<b>This is a smoke run</b> (${f.steps.toLocaleString("en-US")} steps, epoch ${f.epoch}): the gaps show the comparison working end to end, not how good the method is. Real findings need narrow-B and full-B; they drop in by rerunning the job in the README.`);
  if (f.mock) out.push(`<b>This is a mock</b>: the draws are the linearised Gaussian posterior with a known defect, so the width ratio above should read what it was built with (about 1 for fisher where the posterior is Gaussian, 3 for wide, 0.5 for narrow). It checks that the comparison measures what it claims.`);
  if (src.length) {
    const pin = src.map(s => s.flow[pinnedOf(s.mcmc)] / s.mcmc[pinnedOf(s.mcmc)]);
    const phi0 = src.map(s => s.flow[PHI0] / s.mcmc[PHI0]);
    const per = P.map((_, i) => median(src.map(s => s.flow[i] / s.mcmc[i])));
    const worst = per.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])[0];
    out.push(`<b>Where.</b> Against the joint MCMC the flow is widest in ${P[worst[1]]} (median ${fmtX(worst[0])}); the pinned phase combination sits at a median ${fmtX(median(pin))} while φ₀ alone reads ${fmtX(median(phi0))}, so judging the phases by φ₀ would understate the gap.`);
  } else if (r.widths) {
    out.push(`<b>No MCMC for this problem.</b> Only the truth ranks speak to this run: a flow that is too wide shows up as coverage above nominal and a hump in the middle of the rank histogram, too narrow as coverage below nominal and a U shape.`);
  }
  if (f.ode) out.push(`<b>The sampler.</b> These draws used ${f.ode} ODE steps on a uniform grid. Late-time collapse only narrows a posterior, so a flow far wider than the reference is the network's doing; once a trained run comes within a factor of a few of the reference, re-evaluate it with more steps before reading the width as the flow's.`);
  out.push(`<b>What would change this.</b> A trained narrow-band run moves every card here; a run whose problem has no cached chains has truth ranks only, so its widths can only be judged by calibration.`);
  $("means").innerHTML = out.map(x => `<li>${x}</li>`).join("");
}

function render() {
  if (!D.settings.length) { document.querySelector(".layout").innerHTML = empty("No records in data/ yet."); return; }
  renderSide(); renderBanner(); renderTiles(); renderStrip(); renderSnr(); renderCalibration(); renderPair(); renderMeans();
  Object.values(marks).forEach(m => m());
  writeHash();
}

// clicks on marks: data-param picks the parameter, data-src the source
document.addEventListener("click", ev => {
  const el = ev.target.closest("[data-param], [data-src]");
  if (!el) return;
  if (el.dataset.src) selectSource(el.dataset.src); else S.param = Number(el.dataset.param);
  schedule();
});
$("clear").addEventListener("click", () => { S.src = ""; schedule(); });
const tipEl = $("tip");
document.addEventListener("mousemove", ev => {
  const el = ev.target.closest && ev.target.closest("[data-tip]");
  if (!el) { tipEl.style.display = "none"; return; }
  tipEl.textContent = el.dataset.tip;
  tipEl.style.display = "block";
  tipEl.style.left = `${Math.min(ev.clientX + 12, innerWidth - tipEl.offsetWidth - 8)}px`;
  tipEl.style.top = `${ev.clientY + 14}px`;
});

let pending = false;
function schedule() {  // one render per frame
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}
addEventListener("resize", schedule);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
if (S.src) selectSource(S.src);
render();
</script>
</body>
</html>
"""


def main() -> None:
    settings = [json.loads(path.read_text()) for path in sorted(DATA.glob("*.json"))]
    for r in settings:
        r["joint"].sort(key=lambda w: -w["window_snr"])
    # real runs first, then the mocks
    settings.sort(key=lambda r: (r["kind"] == "mock", r["smoke"], r["setting"]))
    data = json.dumps(dict(parameters=settings[0]["parameters"], settings=settings), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out = REPORT / "reference-comparison.html"
    out.write_text(PAGE.replace("__DATA__", data))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): " + ", ".join(r["setting"] for r in settings))


if __name__ == "__main__":
    main()
