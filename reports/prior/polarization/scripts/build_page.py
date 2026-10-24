"""Write polarization.html from data/polarization.json: LisaGB's prior on the orientation (sin iota, psi, phi0) and the helicity chart's z+- built from it.

Standard library only; run from the repo root:
    uv run --frozen python reports/prior/polarization/scripts/build_page.py
"""

import json
import math
from datetime import date
from pathlib import Path

REPORT = Path(__file__).parents[1]
BANDS = ("narrow", "full")
TAU = 2 * math.pi
TINY = 1e-30
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior: polarization</title>
<script>
window.onerror = (msg, src, line) => {
  const b = document.getElementById("err");
  if (b) { b.hidden = false; b.textContent = `This page hit an error and may show stale views: ${msg} (line ${line}).`; }
};
</script>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100; --bar: #c3c2b7;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --bar: #52514e;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main > .card + .card { margin-top: 16px; }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.nav { font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.nav a { color: var(--c1); text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.sub { color: var(--ink-2); margin: 0 0 24px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 100ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 0 0 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 22px; flex: none; }
.row { display: flex; align-items: center; justify-content: flex-start; gap: 8px; margin-bottom: 10px; }
.row .k { color: var(--ink-2); font-size: 13px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 12px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.card > .seg { margin: 0 0 12px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
svg text.lab { fill: var(--ink-2); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; white-space: nowrap; z-index: 2; }
.tip .row { color: var(--ink-2); font-variant-numeric: tabular-nums; margin: 0; gap: 6px; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
td.on, th.on { background: color-mix(in srgb, var(--c1) 10%, transparent); }
.meaning ul { margin: 8px 0 0; padding-left: 20px; }
.meaning li { margin: 0 0 8px; max-width: 80ch; }
#err { position: sticky; top: 0; z-index: 10; background: #c0392b; color: #fff; padding: 10px 16px; font-size: 14px; }
.exgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; position: relative; }
.sliders { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px 16px; margin: 8px 0; }
.sliders label { font-size: 13px; color: var(--ink-2); display: block; }
.sliders output { float: right; font-variant-numeric: tabular-nums; }
input[type=range] { width: 100%; }
button.plain { font: inherit; padding: 4px 12px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--ink); cursor: pointer; }
.cell { cursor: pointer; }
.cell:hover rect { stroke: var(--ink-2); }
.note { color: var(--ink-2); font-size: 13px; margin: 6px 0 0; }
</style>
</head>
<body>
<div id="err" role="alert" hidden></div>
<main>
<p class="nav">The LisaGB prior by parameter group: <a data-page="../f0-fdot-amplitude/f0-fdot-amplitude.html" href="../f0-fdot-amplitude/f0-fdot-amplitude.html">f₀, ḟ, A</a> · <b>polarization</b> · <a data-page="../sky/sky.html" href="../sky/sky.html">sky</a> · <a data-page="../response-span/response-span.html" href="../response-span/response-span.html">response span</a> · <a data-page="../source-count/source-count.html" href="../source-count/source-count.html">source count</a></p>
<h1>The LisaGB prior: polarization</h1>
<p class="sub">The three angles (sin ι, ψ, φ₀), the helicity chart z± they become, and the flow's coordinates, over __WINDOWS__ drawn windows of four sources per band: what the prior draws, how jaxgb turns it into a signal, where the chart is singular, and which of its four equivalent branches comes back. __SUB__</p>
<div class="row"><span class="k">Band</span><span class="seg" id="band"></span></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>Which angles does the prior draw?</h2>
  <span class="seg" id="angle"></span>
  <p class="note" id="n-angle"></p>
  <div class="legend" id="lg-angle"></div>
  <div id="p-angle"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How does jaxgb turn them into h₊ and h×?</h2>
  <div class="note" id="n-form"></div>
  <div class="tablewrap" id="p-checks"></div>
</div>
<div class="card">
  <h2>What do the ellipses look like?</h2>
  <p class="note" id="n-gallery"></p>
  <div id="p-gallery"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Explorer: one source in every coordinate system</h2>
  <span class="seg" id="coord"></span>
  <p class="note" id="n-ex"></p>
  <div class="sliders" id="sliders"></div>
  <div class="row"><span class="k">Ω</span><input id="om" type="range" min="0" max="6.2832" step="0.005" aria-label="Omega"><button type="button" id="play" class="plain">play</button></div>
  <div class="exgrid" id="p-ex"></div>
  <div class="tablewrap" id="p-readout"></div>
</div>
<div class="card">
  <h2>What polarization does the prior give?</h2>
  <span class="seg" id="pol"></span>
  <p class="note" id="n-pol"></p>
  <div class="legend" id="lg-pol"></div>
  <div id="p-pol"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How do the coordinate systems compare?</h2>
  <span class="seg" id="sys"></span>
  <p class="note" id="n-sys"></p>
  <div class="exgrid" id="p-sys"></div>
  <div class="tablewrap" id="p-syscorr"></div>
  <div id="p-sysscatter"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What scale do the flow coordinates have?</h2>
  <span class="seg" id="flow"></span>
  <p class="note" id="n-flow"></p>
  <div class="legend" id="lg-flow"></div>
  <div id="p-flow"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Where are the angles singular?</h2>
  <span class="seg" id="sing"></span>
  <p class="note" id="n-sing"></p>
  <div class="legend" id="lg-sing"></div>
  <div id="p-sing"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Which branch does <code>flow_to_physical</code> return?</h2>
  <span class="seg" id="branch"></span>
  <p class="note" id="n-branch"></p>
  <div class="legend" id="lg-branch"></div>
  <div id="p-branch"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How exact is the round trip?</h2>
  <p class="note" id="n-trip"></p>
  <div class="tablewrap" id="p-trip"></div>
  <div class="tablewrap" id="p-poles"></div>
</div>
<div class="card">
  <h2>Why reference the phase at the middle?</h2>
  <span class="seg" id="phase"></span>
  <p class="note" id="n-phase"></p>
  <div class="legend" id="lg-phase"></div>
  <div id="p-phase"></div><div class="tip"></div>
</div>
<div class="card meaning">
  <h2>What this means</h2>
  <ul id="meaning"></ul>
</div>
</main>
<script>
const PG = __PAGE__;
const BANDS_ORDER = ["narrow", "full"];
const PI = Math.PI, TAU = 2 * Math.PI, TINY = 1e-30;
const COL = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"];
const VIEWS = {
  angle: [["sin", "sin ι"], ["iota", "ι"], ["psi", "ψ"], ["phi0", "φ₀"]],
  pol: [["axis", "axis ratio"], ["vi", "V/I"], ["ratio", "|z₊|/|z₋|"], ["mod", "|z±| and A"], ["arg", "arg z±"]],
  sys: [["physical", "physical"], ["polar", "z± polar"], ["cartesian", "z± Cartesian"]],
  flow: [["hist", "marginals"], ["std", "std of all 8"], ["corr", "8 × 8 correlations"]],
  sing: [["cond", "condition number"], ["corr", "ψ–φ₀ correlation"], ["table", "near the poles"]],
  branch: [["variants", "the four branches"], ["psi", "returned ψ"], ["phi0", "returned φ₀"], ["pole", "at a pole"]],
  phase: [["cond", "conditional"], ["marg", "marginal"], ["sigma", "σ(φ)"]],
  coord: [["phys", "physical"], ["cart", "z± Cartesian"], ["polar", "z± polar"]],
};
const CHOICES = {band: ["narrow", "full"], ...Object.fromEntries(Object.entries(VIEWS).map(([k, v]) => [k, v.map(x => x[0])]))};
const S = {band: "full", angle: "sin", pol: "axis", sys: "physical", flow: "hist", sing: "cond", branch: "variants", phase: "cond", coord: "phys", ex: [1, 0.5, 0.4, 1.0], om: 0};
const B = () => PG.bands[S.band];
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sum = a => a.reduce((x, y) => x + y, 0);
const sup = e => String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
const exp10 = (v, n = 3) => sig(10 ** v, n);

const marks = {};
function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; marks[key](); schedule(); }));
  marks[key]();
}
seg("band", [["narrow", "narrow-B"], ["full", "full-B"]], "band");
for (const [k, v] of Object.entries(VIEWS)) seg(k, v, k);

// state in the address bar: #band=full&angle=sin&…&ex=1,0.5,0.4,1&om=0; ex is (log10 A/unit, sin ι, ψ, φ₀)
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  for (const [k, vs] of Object.entries(CHOICES)) if (vs.includes(q.get(k))) S[k] = q.get(k);
  const ex = (q.get("ex") ?? "").split(",").map(Number);
  if (ex.length === 4 && ex.every(Number.isFinite)) S.ex = ex;
  if (q.has("om") && Number.isFinite(Number(q.get("om")))) S.om = mod(Number(q.get("om")), TAU);
  Object.values(marks).forEach(m => m());
}
function writeHash() {
  const state = {...S, ex: S.ex.map(v => +v.toFixed(4)).join(","), om: +S.om.toFixed(3)};
  history.replaceState(null, "", "#" + new URLSearchParams(Object.entries(state).filter(([, v]) => v)).toString());
  document.querySelectorAll(".nav a").forEach(a => a.href = `${a.dataset.page}#band=${S.band}`);
}
addEventListener("hashchange", () => { readHash(); schedule(); });

function tip(host, ev, html) {
  const t = host.nextElementSibling, card = host.closest(".card").getBoundingClientRect();
  t.innerHTML = html; t.style.display = "block";
  t.style.left = Math.max(4, Math.min(ev.clientX - card.left + 14, card.width - t.offsetWidth - 8)) + "px";
  t.style.top = (ev.clientY - card.top + 14) + "px";
}
const untip = host => host.nextElementSibling.style.display = "none";
const trow = (color, text) => `<div class="row">${color ? `<i class="dot" style="background:${color};width:8px;height:8px"></i>` : ""}${text}</div>`;
const legendItem = (color, text, dash) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${color}" stroke-width="2"${dash ? ` stroke-dasharray="${dash}"` : ""}/></svg>${text}</span>`;
const legendDot = (color, text) => `<span><i class="dot" style="background:${color}"></i>${text}</span>`;
const legendBox = (color, text) => `<span><i class="dot" style="background:${color};border-radius:2px"></i>${text}</span>`;
const html = (id, text) => document.getElementById(id).innerHTML = text;

// generic plots: layers in data units (log10 where an axis is log)
function niceTicks(lo, hi, n = 4) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag;
  const step = (r >= 5 ? 10 : r >= 2 ? 5 : r >= 1 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(+v.toPrecision(12));
  return out;
}
function logTicks(lo, hi) {
  const out = [], few = hi - lo < 2.5;
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) for (const m of few ? [1, 2, 5] : [1]) {
    const v = e + Math.log10(m); if (v >= lo - 1e-9 && v <= hi + 1e-9) out.push(v);
  }
  return out.length >= 2 ? out : niceTicks(lo, hi);
}
const logLabel = v => { const x = 10 ** v, e = Math.floor(v + 1e-9), m = Math.round(x / 10 ** e); return x >= 1e-2 && x < 1e4 ? String(+x.toPrecision(2)) : `${m > 1 ? m + "×" : ""}10${sup(e)}`; };
const pad = ([lo, hi], f = 0.04) => [lo - f * (hi - lo), hi + f * (hi - lo)];
const centres = e => e.slice(1).map((v, i) => (v + e[i]) / 2);
const nearest = (xs, v) => xs.reduce((best, x, i) => Math.abs(x - v) < Math.abs(xs[best] - v) ? i : best, 0);
const extent = vs => [Math.min(...vs), Math.max(...vs)];
function chart(host, o) {
  const W = Math.max(300, o.W ?? host.clientWidth), H = o.H ?? 280, m = {l: 54, r: 16, t: 18, b: 38};
  const [x0, x1] = o.x, [y0, y1] = o.y;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const pts = arr => arr.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label}">`;
  (o.ylog ? logTicks(y0, y1) : niceTicks(y0, y1, o.yn ?? 4)).forEach(t => s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${o.ylog ? logLabel(t) : sig(t, 3)}</text>`);
  (o.xlog ? logTicks(x0, x1) : niceTicks(x0, x1, o.xn ?? 5)).forEach(t => s += `<line x1="${X(t)}" x2="${X(t)}" y1="${H - m.b}" y2="${H - m.b + 4}" stroke="var(--axis)"/><text x="${X(t)}" y="${H - m.b + 17}" text-anchor="middle">${o.xlog ? logLabel(t) : sig(t, 3)}</text>`);
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 3}" text-anchor="middle">${o.xlabel}</text><text x="${m.l}" y="${m.t - 6}">${o.ylabel}</text>`;
  for (const L of o.layers) {
    if (L.band) s += `<rect x="${X(L.band[0]).toFixed(1)}" y="${m.t}" width="${(X(L.band[1]) - X(L.band[0])).toFixed(1)}" height="${H - m.t - m.b}" fill="${L.color}" opacity="0.45"/>`;
    if (L.bars) L.bars.v.forEach((v, i) => { if (v > 0) { const a = X(L.bars.e[i]) + 1, b = X(L.bars.e[i + 1]) - 1; s += `<rect x="${a.toFixed(1)}" y="${Y(v).toFixed(1)}" width="${Math.max(b - a, 0.5).toFixed(1)}" height="${(Y(Math.max(y0, 0)) - Y(v)).toFixed(1)}" fill="${L.color}" rx="1"/>`; } });
    if (L.dots) L.dots.forEach(([x, y]) => s += `<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="${L.r ?? 2}" fill="${L.color}" fill-opacity="${L.opacity ?? 0.5}"/>`);
    if (L.line) s += `<polyline points="${pts(L.line)}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 2}" stroke-linejoin="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.marks) L.line.forEach(([x, y]) => s += `<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="3.5" fill="${L.color}" stroke="var(--surface)" stroke-width="1.5"/>`);
  }
  s += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" visibility="hidden"/><circle class="ring" r="6" fill="none" stroke="var(--ink)" stroke-width="2" visibility="hidden"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  host.insertAdjacentHTML("beforeend", s + "</svg>");
  const svg = host.lastElementChild, cross = svg.querySelector(".cross"), ring = svg.querySelector(".ring"), hit = svg.querySelector(".hit");
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W, py = (ev.clientY - r.top) / r.height * H;
    const out = o.hover({x: x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0), px, py, X, Y});
    cross.setAttribute("visibility", out?.cross !== undefined ? "visible" : "hidden");
    ring.setAttribute("visibility", out?.ring ? "visible" : "hidden");
    if (!out) return untip(host);
    if (out.cross !== undefined) { cross.setAttribute("x1", X(out.cross)); cross.setAttribute("x2", X(out.cross)); }
    if (out.ring) { ring.setAttribute("cx", X(out.ring[0])); ring.setAttribute("cy", Y(out.ring[1])); }
    tip(host, ev, out.html);
  });
  hit.addEventListener("pointerleave", () => { untip(host); cross.setAttribute("visibility", "hidden"); ring.setAttribute("visibility", "hidden"); });
}
function nearestPoint(ps, h) {
  let best = null, bd = 14 ** 2;
  ps.forEach((p, i) => { const d = (h.X(p[0]) - h.px) ** 2 + (h.Y(p[1]) - h.py) ** 2; if (d < bd) { bd = d; best = [p, i]; } });
  return best;
}
// a histogram's density per unit x, and its quantile by linear interpolation inside the bin
const density = h => { const w = h.edges[1] - h.edges[0], n = sum(h.counts) + (h.outside ?? 0); return {w, n, v: h.counts.map(c => c / n / w), ref: h.reference ? h.reference.map(c => c / n / w) : null}; };
function histQuantile(h, q) {
  const n = sum(h.counts), target = q * n;
  let acc = 0;
  for (let i = 0; i < h.counts.length; i++) { if (acc + h.counts[i] >= target) return h.edges[i] + (target - acc) / h.counts[i] * (h.edges[i + 1] - h.edges[i]); acc += h.counts[i]; }
  return h.edges[h.edges.length - 1];
}
// one parameter's histogram with the closed-form prior where there is one
function histChart(host, h, o) {
  const dd = density(h), e = h.edges, x = centres(e);
  const top = Math.max(...dd.v, ...(dd.ref ?? [0]), ...(o.extra ?? []).filter(L => L.line).flatMap(L => L.line.map(p => p[1])));
  const layers = [...(o.under ?? []), {bars: {e, v: dd.v}, color: "var(--bar)"}];
  if (dd.ref) layers.push({line: x.map((v, i) => [v, dd.ref[i]]), color: "var(--ink-2)", width: 1.5, dash: "4 3"});
  (o.extra ?? []).forEach(L => layers.push(L));
  chart(host, {x: pad(o.x ?? [e[0], e[e.length - 1]], 0.02), y: [0, top * 1.15], xlog: o.xlog, xlabel: o.xlabel, ylabel: o.ylabel, label: o.label, layers,
    hover: ({x: v}) => { const i = nearest(x, v); return {cross: x[i], html: `<b>${o.xlabel}: ${o.fmt(e[i])} – ${o.fmt(e[i + 1])}</b>` + trow("var(--bar)", `${h.counts[i].toLocaleString()} drawn${dd.ref ? `, prior ${sig(h.reference[i], 4)}` : ""}`) + (o.tipExtra ? o.tipExtra(i) : "")}; }});
}
// a binned conditional statistic as [x, y] points, empty bins dropped
const binLine = (bn, key, f = v => v, g = v => v) => centres(bn.edges).map((x, i) => [g(x), bn[key][i]]).filter(p => p[1] !== null && p[1] > -Infinity).map(([x, y]) => [x, f(y)]);

// the chart, ported from chart.py (physical_to_flow :20, flow_to_physical :49); amplitudes in amplitude_unit
const mod = (v, m) => ((v % m) + m) % m;
const wrap = v => mod(v + PI, TAU) - PI;
function toZ(a, s, psi, phi0) {
  const rp = Math.asinh(a * (1 - s) ** 2 / 4), rm = Math.asinh(a * (1 + s) ** 2 / 4), fp = phi0 + 2 * psi, fm = phi0 - 2 * psi;
  return {rp, rm, fp, fm, z: [rp * Math.cos(fp), rp * Math.sin(fp), rm * Math.cos(fm), rm * Math.sin(fm)]};
}
function toPhys(z) {
  const op = Math.sqrt(Math.sinh(Math.hypot(z[0], z[1]))), om = Math.sqrt(Math.sinh(Math.hypot(z[2], z[3]))), sum = op + om;
  const fp = Math.atan2(z[1], z[0]), fm = Math.atan2(z[3], z[2]);
  return {A: sum * sum, s: clamp((om - op) / Math.max(sum, TINY), -(1 - 1e-6), 1 - 1e-6), psi: mod((fp - fm) / 4, TAU), phi0: mod((fp + fm) / 2, TAU)};
}
const unwarp = (r, f) => [Math.sinh(r) * Math.cos(f), Math.sinh(r) * Math.sin(f)];
// zW = −2[conj(z−) e^{iΩ} + z₊ e^{−iΩ}] and the two terms, in amplitude units
function zTerms(zz, om) {
  const [pr, pi] = unwarp(zz.rp, zz.fp), [mr, mi] = unwarp(zz.rm, zz.fm), c = Math.cos(om), n = Math.sin(om);
  return [[-2 * (pr * c + pi * n), -2 * (pi * c - pr * n)], [-2 * (mr * c + mi * n), -2 * (mr * n - mi * c)]];
}
const zW = (zz, om) => { const [a, b] = zTerms(zz, om); return [a[0] + b[0], a[1] + b[1]]; };
// jaxgb's own arithmetic: aplus = −a(1 + s²), across = −2as, w = Re(plus e^{iθ}) + i Re(cross e^{iθ})
function jaxW(a, s, psi, phi0, om) {
  const ap = -a * (1 + s * s), ac = -2 * a * s, c2 = Math.cos(2 * psi), s2 = Math.sin(2 * psi), th = om - phi0, c = Math.cos(th), n = Math.sin(th);
  return [ap * c2 * c - ac * s2 * n, ap * s2 * c + ac * c2 * n];
}
// the ellipse at A = 1, φ₀ = 0, from the closed form −e^{2iψ}[(1 + s²)cosθ + 2is sinθ]
const ellipse = (s, psi, n = 96) => Array.from({length: n + 1}, (_, i) => { const th = TAU * i / n, c2 = Math.cos(2 * psi), s2 = Math.sin(2 * psi), p = (1 + s * s) * Math.cos(th), q = 2 * s * Math.sin(th); return [-(p * c2 - q * s2), -(p * s2 + q * c2)]; });
const handed = s => Math.abs(s) < 1e-9 ? "linear" : s > 0 ? "counter-clockwise (z₋ dominates)" : "clockwise (z₊ dominates)";
const epsilon = s => 2 * Math.abs(s) / (1 + s * s);
const stokesV = s => 4 * s * (1 + s * s) / (1 + 6 * s * s + s ** 4);

// the page's own check of the port against the drawn test vectors
function portCheck() {
  const out = {z: 0, A: 0, s: 0, psi: 0, phi0: 0, ell: 0};
  for (const r of B().test_vectors.rows) {
    const zz = toZ(r.A, r.sin_iota, r.psi, r.phi0), back = toPhys(r.z);
    r.z.forEach((v, i) => out.z = Math.max(out.z, Math.abs(zz.z[i] - v)));
    out.A = Math.max(out.A, Math.abs(back.A / r.back.A - 1));
    out.s = Math.max(out.s, Math.abs(back.s - r.back.sin_iota));
    out.psi = Math.max(out.psi, Math.abs(wrap(back.psi - r.back.psi)));
    out.phi0 = Math.max(out.phi0, Math.abs(wrap(back.phi0 - r.back.phi0)));
    for (let k = 0; k < 8; k++) {
      const om = TAU * k / 8, a = zW(zz, om), b = jaxW(r.A, r.sin_iota, r.psi, r.phi0, om);
      out.ell = Math.max(out.ell, Math.hypot(a[0] - b[0], a[1] - b[1]) / r.A);
    }
  }
  return out;
}
function portBanner(pc) {
  const bad = [];
  if (Math.max(pc.z, pc.A, pc.s, pc.psi, pc.phi0) > 1e-9) bad.push(`the JS port misses the test vectors (max ${sig(Math.max(pc.z, pc.A, pc.s, pc.psi, pc.phi0), 2)} > 1e-9)`);
  if (pc.ell > 1e-12) bad.push(`zW and jaxgb's ellipse disagree (${sig(pc.ell, 2)} > 1e-12)`);
  const e = document.getElementById("err");
  if (bad.length && e) { e.hidden = false; e.textContent = "Check failed: " + bad.join("; ") + "."; }
}

const table = (head, rows, on = -1) => `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
const chi = h => h.chi2 === undefined ? "" : ` χ²/dof ${h.chi2.toFixed(0)}/${h.dof}.`;
const hline = (h, k = 1) => { const d = density(h); return centres(h.edges).map((x, i) => [x, d.v[i] * k]); };
const fix2 = v => v.toFixed(2);
const mean = a => sum(a) / a.length;

function tiles() {
  const b = B(), zs = b.flow.map(f => f.std), fs = b.flow_std.slice(0, 4), d = b.degeneracy, f64 = b.round_trip.float64, f32 = b.round_trip.float32, ph = b.phase;
  const conds = b.singularity.rows, fin = k => conds.map(r => r[k]).filter(Number.isFinite), gl = conds.map(r => Math.abs(r.sin_iota)), edge = Math.max(...gl);
  document.getElementById("tiles").innerHTML = [
    ["z± std", `${sig(Math.min(...zs), 3)}–${sig(Math.max(...zs), 3)}`, `against ${sig(Math.min(...fs), 2)}–${sig(Math.max(...fs), 2)} for f₀, ḟ, l and b`],
    ["four equivalent (ψ, φ₀)", `≤ ${sig(Math.max(...d.variants.map(v => v.signal)), 2)}`, `signal mismatch; a wrong step gives ${sig(d.wrong_step_median, 3)} (median)`],
    ["round trip, float64 / float32", `${sig(f64.signal_max, 2)} / ${sig(f32.signal_max, 2)}`, "largest signal mismatch over 256 sources"],
    ["corr(f₀, φ), start → middle", `${ph.start.f0_phi_conditional_median.toFixed(2)} → ${ph.middle.f0_phi_conditional_median.toFixed(2)}`, "conditional Fisher correlation, median"],
    ["condition number, angles / z", `${sig(Math.max(...fin("cond_angles")), 2)} / ${sig(Math.max(...fin("cond_z")), 3)}`, `worst over the sin ι grid, to |sin ι| = ${edge}`],
    ["largest |Re z±|, |Im z±|", sig(Math.max(...b.flow.map(f => Math.max(Math.abs(f.hist.edges[0]), Math.abs(f.hist.edges.at(-1))))), 3), "warped, over the band's sources"],
  ].map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}

function angleCard() {
  const host = document.getElementById("p-angle"), b = B(), o = b.physical.orientation;
  host.innerHTML = "";
  const v = S.angle;
  if (v === "sin") {
    histChart(host, o["sin iota"], {xlabel: "sin ι", ylabel: "density", label: "sin iota prior", fmt: fix2});
    html("lg-angle", legendBox("var(--bar)", `${b.sources.toLocaleString()} sources`) + legendItem("var(--ink-2)", "uniform on [−1, 1]", "4 3"));
    html("n-angle", `<code>simulator.py:78-82</code> draws ι = arcsin U(−1, 1), and :20 hands <code>jaxgb</code> π/2 − ι (a colatitude), so sin ι takes the usual cos ι's place and is uniform.${chi(o["sin iota"])} Face-on (+1) shows only φ₋ = φ₀ − 2ψ, face-off (−1) only φ₊; edge-on (0) shows both.`);
  } else if (v === "iota") {
    histChart(host, o["iota [rad]"], {xlabel: "ι [rad]", ylabel: "density", label: "iota prior", fmt: fix2});
    html("lg-angle", legendBox("var(--bar)", `${b.sources.toLocaleString()} sources`) + legendItem("var(--ink-2)", "cos ι / 2", "4 3"));
    html("n-angle", `ι itself has density cos ι / 2 on (−π/2, π/2): few sources are face-on or face-off in ι, though every sin ι is equally likely.${chi(o["iota [rad]"])}`);
  } else if (v === "psi") {
    histChart(host, o["psi [rad]"], {xlabel: "ψ [rad]", ylabel: "density", label: "psi prior", fmt: fix2});
    html("lg-angle", legendBox("var(--bar)", `${b.sources.toLocaleString()} sources`) + legendItem("var(--ink-2)", "uniform on [0, 2π)", "4 3"));
    html("n-angle", `ψ is uniform on [0, 2π) (<code>simulator.py:80</code>) and enters only through φ± = φ₀ ± 2ψ, which is why the flow predicts z± rather than ψ and φ₀.${chi(o["psi [rad]"])}`);
  } else {
    histChart(host, o["phi0 [rad]"], {xlabel: "φ₀ [rad]", ylabel: "density", label: "phi0 prior", fmt: fix2});
    html("lg-angle", legendBox("var(--bar)", `${b.sources.toLocaleString()} sources`) + legendItem("var(--ink-2)", "uniform on [0, 2π)", "4 3"));
    html("n-angle", `φ₀ is uniform on [0, 2π) (<code>simulator.py:82</code>).${chi(o["phi0 [rad]"])}`);
  }
}

function formCard() {
  const b = B(), pc = portCheck(), s = b.singularity;
  html("n-form", `<p><code>jaxgb.py:296-298</code> takes cos ι (= s = sin ι here), cos 2ψ and sin 2ψ; :301-302 set <b>A₊ = −A(1 + s²)</b> and <b>A× = −2As</b>; :303 mixes them with ψ, and the phase is φ₀ − Ω. Writing θ = Ω − φ₀, the (h₊, h×) pair of one source is</p>
  <p><code>w(Ω) = −A e^{2iψ} [(1 + s²) cos θ + 2is sin θ]</code>,</p>
  <p>an ellipse with semi-axes A(1 + s²) and 2A|s| whose major axis lies at 2ψ mod π. The chart's z± = A(1 ∓ s)²/4 · e^{i(φ₀ ± 2ψ)} are its two circular components: w = −2[conj(z₋) e^{iΩ} + z₊ e^{−iΩ}], so z₋ dominates for s &gt; 0 (counter-clockwise). The ellipticity ε = minor/major = 2|s|/(1 + s²), V/I = 4s(1 + s²)/(1 + 6s² + s⁴), and √|z₊| + √|z₋| = √A. The flow predicts asinh|z±| · e^{i arg z±} and <code>flow_to_physical</code> inverts it with A = (√|z₊|' + √|z₋|')² and s = (√|z₋|' − √|z₊|')/√A, where ' marks sinh of the radius.</p>`);
  const rows = [
    ["ellipse identity, jaxgb against zW (max, A = 1)", sig(b.test_vectors.ellipse_identity, 2), "drawn on the 12 test vectors"],
    ["axis ratio against 2|s|/(1 + s²)", sig(b.axis_ratio_check, 2), "over every drawn source"],
    ["split: z₊ and z₋ reproduce w", sig(b.split_check, 2), "over every drawn source"],
    ["z-Fisher variation over s", sig(s.z_fisher_variation, 2), `inner |s| ≤ 0.9: ${sig(s.z_fisher_variation_inner, 2)}`],
    ["linearity in the sum of components", sig(s.linearity_sum, 2), "the signal is linear in z₊, z₋"],
    ["linearity in the scale", sig(s.linearity_scale, 2), ""],
    ["Python port of toZ/toPhys against the test vectors", "see the builder's output", "printed at build time"],
    ["JS port against the test vectors", sig(Math.max(pc.z, pc.A, pc.s, pc.psi, pc.phi0), 2), `z ${sig(pc.z, 2)}, A ${sig(pc.A, 2)}, s ${sig(pc.s, 2)}, ψ ${sig(pc.psi, 2)}, φ₀ ${sig(pc.phi0, 2)} (tolerance 1e-9)`],
    ["JS zW against jaxgb's arithmetic", sig(pc.ell, 2), "tolerance 1e-12"],
  ];
  html("p-checks", table(["check", "value", "note"], rows));
  portBanner(pc);
}

function gallery() {
  const host = document.getElementById("p-gallery"), S_ = [1, 0.5, 0, -0.5, -1], P_ = [0, PI / 8, PI / 4, 3 * PI / 8];
  const w = 170, h = 150, K = 62 / 2.1, W = w * P_.length, H = h * S_.length + 8;
  let g = "";
  S_.forEach((s, i) => P_.forEach((psi, j) => {
    const cx = w * j + w / 2, cy = h * i + h / 2 + 4, pts = ellipse(s, psi).map(([x, y]) => `${(cx + x * K).toFixed(1)},${(cy - y * K).toFixed(1)}`).join(" ");
    const major = 2 * psi;
    g += `<g class="cell" data-s="${s}" data-p="${psi}"><rect x="${w * j + 4}" y="${h * i + 6}" width="${w - 8}" height="${h - 4}" rx="6" fill="var(--page)" stroke="var(--grid)"/>`
      + `<line x1="${cx - 60 * Math.cos(major)}" y1="${cy + 60 * Math.sin(major)}" x2="${cx + 60 * Math.cos(major)}" y2="${cy - 60 * Math.sin(major)}" stroke="var(--axis)" stroke-dasharray="3 3"/>`
      + `<polyline points="${pts}" fill="none" stroke="var(--c1)" stroke-width="2"/>`
      + `<text x="${w * j + 10}" y="${h * i + 22}">s = ${s}, ψ = ${["0", "π/8", "π/4", "3π/8"][j]}</text></g>`;
  }));
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="polarization ellipses, s against psi">${g}</svg>`;
  host.querySelectorAll(".cell").forEach(c => {
    const s = +c.dataset.s, p = +c.dataset.p;
    c.addEventListener("click", () => { S.ex = [S.ex[0], s, p, S.ex[3]]; schedule(); });
    c.addEventListener("pointermove", ev => tip(host, ev, `<b>s = ${s}, ψ = ${p.toFixed(3)}</b>` + trow("var(--c1)", `ε = ${epsilon(s).toFixed(3)}, V/I = ${stokesV(s).toFixed(3)}`) + trow(null, `${handed(s)}; click to load it in the explorer`)));
    c.addEventListener("pointerleave", () => untip(host));
  });
  html("n-gallery", `The (h₊, h×)/A curve over one period for s = sin ι from face-on (+1, a circle) to edge-on (0, a line) to face-off (−1, a circle the other way), and ψ turning the major axis (dashed, at 2ψ) by 2ψ. The handedness flips with the sign of s; the size is A(1 + s²) by 2A|s|. Click a cell to load its s and ψ into the explorer.`);
}

const COORDS = {
  phys: [["log₁₀ A / unit", -1, 3.5, 0.01], ["sin ι", -1, 1, 0.002], ["ψ", 0, TAU, 0.005], ["φ₀", 0, TAU, 0.005]],
  cart: [["Re z₊", -9, 9, 0.01], ["Im z₊", -9, 9, 0.01], ["Re z₋", -9, 9, 0.01], ["Im z₋", -9, 9, 0.01]],
  polar: [["asinh |z₊|", 0, 9, 0.01], ["arg z₊", -PI, PI, 0.005], ["asinh |z₋|", 0, 9, 0.01], ["arg z₋", -PI, PI, 0.005]],
};
let slidersFor = "";
const coordValues = (coord, ex) => {
  const zz = toZ(10 ** ex[0], ex[1], ex[2], ex[3]);
  return coord === "phys" ? [ex[0], ex[1], mod(ex[2], TAU), mod(ex[3], TAU)] : coord === "cart" ? zz.z : [zz.rp, wrap(zz.fp), zz.rm, wrap(zz.fm)];
};
function fromSliders() {
  const v = [...document.querySelectorAll("#sliders input")].map(i => +i.value);
  if (S.coord === "phys") return v;
  const z = S.coord === "cart" ? v : [v[0] * Math.cos(v[1]), v[0] * Math.sin(v[1]), v[2] * Math.cos(v[3]), v[2] * Math.sin(v[3])], p = toPhys(z);
  return [Math.log10(Math.max(p.A, 1e-3)), p.s, p.psi, p.phi0];
}
function buildSliders() {
  slidersFor = S.coord;
  document.getElementById("sliders").innerHTML = COORDS[S.coord].map(([name, lo, hi, st], i) => `<label>${name} <output></output><input type="range" min="${lo}" max="${hi}" step="${st}" aria-label="${name}"></label>`).join("");
  document.querySelectorAll("#sliders input").forEach(el => el.addEventListener("input", () => { S.ex = fromSliders(); explorer(); }));
}
let playing = false;
function tick() { if (!playing) return; S.om = mod(S.om + 0.03, TAU); explorer(); requestAnimationFrame(tick); }

function explorer() {
  if (slidersFor !== S.coord) buildSliders();
  const [la, s0, psi, phi0] = S.ex, s = clamp(s0, -1, 1), A = 10 ** la, zz = toZ(A, s, psi, phi0), ret = toPhys(zz.z), om = S.om;
  const vals = coordValues(S.coord, S.ex);
  document.querySelectorAll("#sliders input").forEach((el, i) => { el.value = vals[i]; if (el.previousElementSibling) el.previousElementSibling.textContent = vals[i].toFixed(3); });
  document.getElementById("om").value = om;
  // ellipse panel: (h₊, h×)/A from zW, jaxgb's points as rings, the two phasors
  const K = 135 / 2.1, X = v => 150 + v * K, Y = v => 150 - v * K, P = p => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`;
  const curve = Array.from({length: 121}, (_, i) => zW(zz, TAU * i / 120).map(v => v / A));
  const rings = Array.from({length: 24}, (_, i) => jaxW(1, s, psi, phi0, TAU * i / 24));
  const [t1, t2] = zTerms(zz, om).map(t => t.map(v => v / A)), w = [t1[0] + t2[0], t1[1] + t2[1]], maj = 2 * psi;
  let e = `<svg viewBox="0 0 300 300" role="img" aria-label="polarization ellipse of the source"><rect x="${X(-2.1)}" y="${Y(2.1)}" width="${X(2.1) - X(-2.1)}" height="${Y(-2.1) - Y(2.1)}" fill="none" stroke="var(--grid)"/>`
    + `<line x1="${X(-2.1)}" x2="${X(2.1)}" y1="150" y2="150" stroke="var(--grid)"/><line y1="${Y(-2.1)}" y2="${Y(2.1)}" x1="150" x2="150" stroke="var(--grid)"/>`
    + `<line x1="${X(-2.1 * Math.cos(maj))}" y1="${Y(-2.1 * Math.sin(maj))}" x2="${X(2.1 * Math.cos(maj))}" y2="${Y(2.1 * Math.sin(maj))}" stroke="var(--axis)" stroke-dasharray="3 3"/>`
    + `<polyline points="${curve.map(P).join(" ")}" fill="none" stroke="var(--ink)" stroke-width="2"/>`
    + rings.map(p => `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="3.2" fill="none" stroke="var(--ink-2)"/>`).join("")
    + `<polyline points="${P([0, 0])} ${P(t1)}" fill="none" stroke="var(--c1)" stroke-width="2"/><polyline points="${P(t1)} ${P(w)}" fill="none" stroke="var(--c2)" stroke-width="2"/>`
    + `<circle cx="${X(w[0])}" cy="${Y(w[1])}" r="5" fill="var(--ink)"/><text x="6" y="14">(h₊, h×)/A</text><text x="294" y="294" text-anchor="end">±2.1</text></svg>`;
  // z± planes: the prior's cloud (middle reference), this source
  const sc = B().scatter, KZ = 135 / 9, ZX = v => 150 + v * KZ, ZY = v => 150 - v * KZ;
  const plane = (r, a, cur, name, color) => `<svg viewBox="0 0 300 300" role="img" aria-label="${name} plane"><line x1="15" x2="285" y1="150" y2="150" stroke="var(--grid)"/><line y1="15" y2="285" x1="150" x2="150" stroke="var(--grid)"/>`
    + r.map((v, i) => `<circle cx="${ZX(v * Math.cos(a[i])).toFixed(1)}" cy="${ZY(v * Math.sin(a[i])).toFixed(1)}" r="1.5" fill="var(--muted)" fill-opacity="0.35"/>`).join("")
    + `<line x1="150" y1="150" x2="${ZX(cur[0])}" y2="${ZY(cur[1])}" stroke="${color}" stroke-width="1.5"/><circle cx="${ZX(cur[0])}" cy="${ZY(cur[1])}" r="5" fill="${color}"/><text x="6" y="14">${name}, ±9</text></svg>`;
  html("p-ex", e + plane(sc.rp, sc.ap, zz.z.slice(0, 2), "z₊", "var(--c1)") + plane(sc.rm, sc.am, zz.z.slice(2), "z₋", "var(--c2)")
    + `<svg viewBox="0 0 640 200" role="img" aria-label="h plus and h cross over omega" style="grid-column:1/-1"><line x1="40" x2="620" y1="100" y2="100" stroke="var(--grid)"/>`
    + [0, PI, TAU].map(v => `<line x1="${40 + v / TAU * 580}" x2="${40 + v / TAU * 580}" y1="170" y2="176" stroke="var(--axis)"/><text x="${40 + v / TAU * 580}" y="190" text-anchor="middle">Ω = ${["0", "π", "2π"][[0, PI, TAU].indexOf(v)]}</text>`).join("")
    + [0, 1].map(k => `<polyline points="${Array.from({length: 121}, (_, i) => `${40 + i / 120 * 580},${100 - zW(zz, TAU * i / 120)[k] / A * 40}`).join(" ")}" fill="none" stroke="${COL[k]}" stroke-width="2"/>`).join("")
    + `<line x1="${40 + om / TAU * 580}" x2="${40 + om / TAU * 580}" y1="10" y2="170" stroke="var(--ink)" stroke-dasharray="3 3"/><text x="44" y="14">h₊/A (blue), h×/A (orange)</text></svg>`);
  html("n-ex", `Load a source in any of the three coordinate systems; the others follow. Ω is the orbital phase: the black dot is zW at Ω, built from the two phasors (blue −2z₊e<sup>−iΩ</sup>, orange −2 conj(z₋)e<sup>iΩ</sup>) and the rings are jaxgb's own h₊, h× at 24 values of Ω, which sit on the curve. The z panels show the prior's z± (grey, ${sc.rp.length.toLocaleString()} sources, phase referenced at the window <i>middle</i>, which only rotates phases) The explorer itself uses the <i>start</i> reference, like the test vectors.`);
  const k = [0, 1, 2, 3].map(j => [mod(psi + j * PI / 2, TAU), mod(phi0 + (j % 2) * PI, TAU)]);
  const best = k.map(([a, c]) => Math.abs(wrap(a - ret.psi)) + Math.abs(wrap(c - ret.phi0))).reduce((i, v, j, arr) => v < arr[i] ? j : i, 0);
  const rows = [
    ["physical", `log₁₀ A/unit ${la.toFixed(3)}, sin ι ${s.toFixed(4)}, ψ ${mod(psi, TAU).toFixed(4)}, φ₀ ${mod(phi0, TAU).toFixed(4)}`],
    ["z± Cartesian", `Re z₊ ${zz.z[0].toFixed(3)}, Im z₊ ${zz.z[1].toFixed(3)}, Re z₋ ${zz.z[2].toFixed(3)}, Im z₋ ${zz.z[3].toFixed(3)}`],
    ["z± polar", `asinh|z₊| ${zz.rp.toFixed(3)} ∠ ${wrap(zz.fp).toFixed(3)}, asinh|z₋| ${zz.rm.toFixed(3)} ∠ ${wrap(zz.fm).toFixed(3)}`],
    ["the four equivalent (ψ, φ₀)", k.map(([a, c], j) => `(${a.toFixed(3)}, ${c.toFixed(3)})${j === best ? " <b>← returned by flow_to_physical</b>" : ""}`).join("; ")],
    ["shape", `ε = ${epsilon(s).toFixed(4)}, V/I = ${stokesV(s).toFixed(4)}, major axis 2ψ mod π = ${mod(2 * psi, PI).toFixed(4)}, ${handed(s)}`],
  ];
  html("p-readout", table(["", ""], rows) + (Math.abs(s) > 0.99 ? `<p class="note">Near a pole (|sin ι| &gt; 0.99) one of z± is almost zero: only ${s > 0 ? "φ₋ = φ₀ − 2ψ" : "φ₊ = φ₀ + 2ψ"} survives, so ψ and φ₀ are not separately determined.</p>` : ""));
}

function polCard() {
  const host = document.getElementById("p-pol"), b = B(), h = b.helicity, v = S.pol;
  host.innerHTML = "";
  const src = `${b.sources.toLocaleString()} sources`;
  if (v === "axis") {
    histChart(host, h["axis ratio"], {xlabel: "ε = minor / major", ylabel: "density", label: "axis ratio", fmt: v => v.toFixed(3)});
    html("lg-pol", legendBox("var(--bar)", src) + legendItem("var(--ink-2)", "from sin ι uniform", "4 3"));
    html("n-pol", `ε = 2|s|/(1 + s²) with s uniform on [−1, 1]; the dashed curve is the exact density of that map. Median ε = ${sig(histQuantile(h["axis ratio"], 0.5), 3)}.${chi(h["axis ratio"])}`);
  } else if (v === "vi") {
    histChart(host, h["V/I"], {xlabel: "V/I", ylabel: "density", label: "circular polarization fraction", fmt: v => v.toFixed(3)});
    html("lg-pol", legendBox("var(--bar)", src) + legendItem("var(--ink-2)", "from sin ι uniform", "4 3"));
    html("n-pol", `V/I = 4s(1 + s²)/(1 + 6s² + s⁴) runs from −1 (face-off) to +1 (face-on) and is odd in s; its density (dashed) is the uniform s pushed through that map.${chi(h["V/I"])}`);
  } else if (v === "ratio") {
    histChart(host, h["log10 |z+|/|z-|"], {xlabel: "log₁₀ |z₊|/|z₋|", ylabel: "density", label: "helicity ratio", fmt: v => v.toFixed(2)});
    html("lg-pol", legendBox("var(--bar)", src) + legendItem("var(--ink-2)", "(ln 10 / 8) sech²(q ln 10 / 4)", "4 3"));
    html("n-pol", `The log ratio q = log₁₀|z₊|/|z₋| = 2 log₁₀((1 − s)/(1 + s)) has the density (ln 10 / 8) sech²(q ln 10 / 4) (dashed) and is odd in s: negative where z₋ dominates (s &gt; 0). ${pct(share(h["log10 |z+|/|z-|"], 0), 1)} of sources have z₋ larger.${chi(h["log10 |z+|/|z-|"])}`);
  } else if (v === "mod") {
    const hs = [h["log10 |z+|"], h["log10 |z-|"], h["log10 A/unit"]], lines = hs.map(x => hline(x)), top = Math.max(...lines.flat().map(p => p[1])), cs = [COL[0], COL[1], "var(--ink)"];
    chart(host, {x: pad([hs[0].edges[0], hs[0].edges.at(-1)], 0.02), y: [0, top * 1.12], xlog: true, xlabel: "amplitude [amplitude_unit]", ylabel: "density per dex", label: "z modulus and amplitude",
      layers: lines.map((l, i) => ({line: l, color: cs[i], width: 2})),
      hover: ({x}) => { const i = nearest(lines[0].map(p => p[0]), x); return {cross: lines[0][i][0], html: `<b>${exp10(lines[0][i][0])} units</b>` + ["|z₊|", "|z₋|", "A"].map((n, k) => trow(cs[k], `${n}: ${sig(lines[k][nearest(lines[k].map(p => p[0]), lines[0][i][0])][1], 3)}`)).join("")}; }});
    html("lg-pol", legendItem(COL[0], "|z₊|") + legendItem(COL[1], "|z₋|") + legendItem("var(--ink)", "A"));
    html("n-pol", `The unwarped moduli |z±| = A(1 ∓ s)²/4 in units of <code>amplitude_unit</code>, against A itself. |z₊| and |z₋| are mirror images, with the long tail toward small values where (1 ∓ s)² → 0, and neither exceeds A: √|z₊| + √|z₋| = √A. A's own prior is on the <a href="../f0-fdot-amplitude/f0-fdot-amplitude.html">f₀, ḟ, A page</a>.`);
  } else {
    const a = h["arg z+ [rad]"], c = h["arg z- [rad]"];
    histChart(host, a, {xlabel: "arg z± [rad]", ylabel: "density", label: "arg of z plus and z minus", fmt: v => v.toFixed(2), extra: [{line: hline(c), color: COL[1], width: 1.5}]});
    html("lg-pol", legendBox("var(--bar)", "arg z₊") + legendItem(COL[1], "arg z₋") + legendItem("var(--ink-2)", "uniform", "4 3"));
    html("n-pol", `φ± = φ₀ ± 2ψ, each uniform because φ₀ and ψ are. They are not independent: z₊ and z₋ share φ₀ and ψ, which is the four-fold ambiguity of card 9.${chi(a)}${chi(c).replace("χ²", " arg z₋ χ²")}`);
  }
}
const share = (h, edge) => { let n = 0; h.counts.forEach((c, i) => { if (h.edges[i + 1] <= edge) n += c; }); return n / sum(h.counts); };

function sysCard() {
  const b = B(), sy = b.systems[S.sys], sc = b.scatter, grid = document.getElementById("p-sys");
  grid.innerHTML = sy.hists.map((_, i) => `<div class="h"></div><div class="tip"></div>`).join("");
  const hosts = [...grid.querySelectorAll(".h")];
  sy.hists.forEach((h, i) => { if (hosts[i]) histChart(hosts[i], h, {x: [h.edges[0], h.edges.at(-1)], xlabel: sy.names[i], ylabel: "density", label: sy.names[i], fmt: fix2, W: 300, H: 200}); });
  html("p-syscorr", table(["corr", ...sy.names], sy.names.map((n, i) => [n, ...sy.corr[i].map(r => `<span style="opacity:${0.35 + 0.65 * Math.min(1, Math.abs(r) * 3)}">${r.toFixed(3)}</span>`)])));
  const host = document.getElementById("p-sysscatter");
  host.innerHTML = "";
  const pairs = {
    physical: [sc.sin_iota, sc.log_amp, "sin ι", "log₁₀ A / unit", [-1, 1]],
    polar: [sc.rp, sc.rm, "asinh |z₊|", "asinh |z₋|", [sy.hists[0].edges[0], sy.hists[0].edges.at(-1)]],
    cartesian: [sc.rp.map((r, i) => r * Math.cos(sc.ap[i])), sc.rp.map((r, i) => r * Math.sin(sc.ap[i])), "Re z₊", "Im z₊", [sy.hists[0].edges[0], sy.hists[0].edges.at(-1)]],
  }[S.sys];
  const pts = pairs[0].map((x, i) => [x, pairs[1][i]]);
  chart(host, {x: pad(pairs[4], 0.02), y: pad(extent(pts.map(p => p[1])), 0.04), xlabel: pairs[2], ylabel: pairs[3], label: "scatter in the chosen coordinates", H: 300, layers: [{dots: pts, color: "var(--muted)", opacity: 0.35}],
    hover: h => { const hit = nearestPoint(pts, h); if (!hit) return null; const [[x, y]] = hit; return {ring: [x, y], html: trow("var(--muted)", `${pairs[2]} ${x.toFixed(3)}, ${pairs[3]} ${y.toFixed(3)}`)}; }});
  const note = {
    physical: `log₁₀ A, sin ι, ψ and φ₀, as drawn. Off-diagonal |r| is at most all below ${sig(Math.max(...sy.corr.flatMap((r, i) => r.filter((_, j) => i !== j).map(Math.abs))), 2)}.`,
    polar: `The two warped moduli asinh|z±| and the two phases. The corr table gives their linear correlations.`,
    cartesian: `What the flow actually predicts: Re and Im of z± after the warp. Each pair is a ring whose radius depends on sin ι and whose angle is uniform.`,
  }[S.sys];
  html("n-sys", `${note} The scatter shows ${pts.length.toLocaleString()} sources.`);
}

function flowCard() {
  const host = document.getElementById("p-flow"), b = B(), v = S.flow, names = ["f₀", "ḟ", "l", "b", "Re z₊", "Im z₊", "Re z₋", "Im z₋"];
  host.innerHTML = "";
  if (v === "hist") {
    const lines = b.flow.map(f => hline(f.hist)), top = Math.max(...lines.flat().map(p => p[1])), xr = extent(b.flow.flatMap(f => [f.hist.edges[0], f.hist.edges.at(-1)]));
    chart(host, {x: xr, y: [0, top * 1.12], xlabel: "flow coordinate (warped)", ylabel: "density", label: "flow coordinates of z plus and z minus",
      layers: lines.map((l, i) => ({line: l, color: COL[i], width: 1.8})),
      hover: ({x}) => { const i = nearest(lines[0].map(p => p[0]), x); return {cross: lines[0][i][0], html: `<b>${sig(lines[0][i][0], 3)}</b>` + b.flow.map((f, k) => trow(COL[k], `${f.name}: ${sig(lines[k][nearest(lines[k].map(p => p[0]), lines[0][i][0])][1], 3)}`)).join("")}; }});
    html("lg-flow", b.flow.map((f, i) => legendItem(COL[i], `${f.name}, std ${f.std.toFixed(3)}`)).join(""));
    html("n-flow", `The four z± coordinates have std ${sig(Math.min(...b.flow.map(f => f.std)), 3)}–${sig(Math.max(...b.flow.map(f => f.std)), 3)}, ${sig(mean(b.flow.map(f => f.std)) / mean(b.flow_std.slice(0, 4)), 2)}× the mean of f₀, ḟ, l and b (${sig(Math.min(...b.flow_std.slice(0, 4)), 2)}–${sig(Math.max(...b.flow_std.slice(0, 4)), 2)}). The radius asinh|z±| has rms ${b.radius.map(r => sig(r.rms, 4)).join(" and ")}.`);
  } else if (v === "std") {
    const mx = Math.max(...b.flow_std);
    html("p-flow", table(["coordinate", "std", ""], b.flow_std.map((s, i) => [names[i], s.toFixed(4), `<span style="display:inline-block;height:10px;width:${(s / mx * 200).toFixed(0)}px;background:${i > 3 ? "var(--c2)" : "var(--c1)"};border-radius:2px"></span>`])));
    html("lg-flow", "");
    html("n-flow", `The eight flat coordinates the flow predicts. f₀, ḟ, l and b have std ${b.flow_std.slice(0, 4).map(s => s.toFixed(2)).join(", ")}; the four z± have ${b.flow_std.slice(4).map(s => s.toFixed(3)).join(", ")}. The z± slots are therefore ${sig(mean(b.flow_std.slice(4)) / mean(b.flow_std.slice(0, 4)), 2)}× wider than the rest on average.`);
  } else {
    const c = b.corr;
    html("p-flow", table(["corr", ...names], c.map((r, i) => [names[i], ...r.map((x, j) => `<span style="opacity:${i === j ? 0.45 : 0.35 + 0.65 * Math.min(1, Math.abs(x) * 5)}">${x.toFixed(3)}</span>`)])));
    html("lg-flow", "");
    const off = c.flatMap((r, i) => r.map((x, j) => i === j ? 0 : Math.abs(x)));
    html("n-flow", `Pearson r of the eight flow coordinates over ${b.sources.toLocaleString()} sources. The largest off-diagonal |r| is ${sig(Math.max(...off), 2)}: no pair is correlated at a level the flow would have to model, so the structure left to learn is in the dependence, not the second moments.`);
  }
}

function singCard() {
  const host = document.getElementById("p-sing"), b = B(), v = S.sing, seen = new Set(), rows = [];
  host.innerHTML = "";
  b.singularity.rows.forEach(r => { const k = r.sin_iota.toFixed(6); if (!seen.has(k)) { seen.add(k); rows.push(r); } });
  rows.sort((p, q) => p.sin_iota - q.sin_iota);
  const src = b.singularity.source;
  if (v === "cond") {
    const keys = [["cond_angles", "angles (ln A, s, ψ, φ₀)", COL[1]], ["cond_z", "z (unwarped)", COL[0]], ["cond_flow", "flow (warped)", COL[2]]];
    const lines = keys.map(([k]) => rows.filter(r => Number.isFinite(r[k]) && r[k] > 0).map(r => [r.sin_iota, Math.log10(r[k])])), all = lines.flat().map(p => p[1]);
    chart(host, {x: [-1, 1], y: pad(extent(all), 0.05), ylog: true, xlabel: "sin ι", ylabel: "condition number of the unit-diagonal Fisher matrix", label: "condition number against sin iota",
      layers: lines.map((l, i) => ({line: l, color: keys[i][2], marks: true})),
      hover: ({x}) => { const i = nearest(rows.map(r => r.sin_iota), x), r = rows[i]; return {cross: r.sin_iota, html: `<b>sin ι = ${r.sin_iota}</b>` + keys.map(([k, n, c]) => trow(c, `${n}: ${sig(r[k], 3)}`)).join("")}; }});
    html("lg-sing", keys.map(([, n, c]) => legendItem(c, n)).join(""));
    const mx = Math.max(...rows.map(r => r.cond_angles).filter(Number.isFinite)), fz = rows.map(r => r.cond_z).filter(Number.isFinite), ff = rows.map(r => r.cond_flow).filter(Number.isFinite);
    html("n-sing", `The Fisher matrix of one source (f₀ = ${sig(src.f0 * 1e3, 3)} mHz, ḟ = ${sig(src.fdot, 3)} Hz/s) in each parametrisation, unit-diagonal so only its shape counts. In (ln A, s, ψ, φ₀) it is ${sig((rows.find(r => r.sin_iota === 0) ?? {}).cond_angles, 3)} at s = 0 and blows up to ${sig(mx, 2)} toward the poles, where ψ and φ₀ degenerate to φ₀ ∓ 2ψ. In z (the unwarped components) it stays at ${sig(Math.min(...fz), 4)}–${sig(Math.max(...fz), 4)}, flat to ${sig(b.singularity.z_fisher_variation, 2)}; in the warped flow coordinates at ${sig(Math.min(...ff), 2)}–${sig(Math.max(...ff), 2)}. This is why the flow predicts z±.`);
  } else if (v === "corr") {
    const lines = [["psi_phi_conditional", "conditional (the other parameters held)", COL[0]], ["psi_phi_marginal", "marginal (the others free)", COL[1]]].map(([k, n, c]) => ({k, n, c, l: rows.filter(r => Number.isFinite(r[k])).map(r => [r.sin_iota, r[k]])}));
    chart(host, {x: [-1, 1], y: [-1.05, 1.05], xlabel: "sin ι", ylabel: "corr(ψ, φ₀)", label: "psi phi0 correlation against sin iota",
      layers: lines.map(x => ({line: x.l, color: x.c, marks: true})),
      hover: ({x}) => { const i = nearest(rows.map(r => r.sin_iota), x), r = rows[i]; return {cross: r.sin_iota, html: `<b>sin ι = ${r.sin_iota}</b>` + lines.map(l => trow(l.c, `${l.n}: ${Number.isFinite(r[l.k]) ? r[l.k].toFixed(4) : "singular in float64"}`)).join("")}; }});
    html("lg-sing", lines.map(l => legendItem(l.c, l.n)).join(""));
    html("n-sing", `Near the poles ψ and φ₀ enter only as φ₀ ∓ 2ψ, so their correlation goes to ∓1: the pair cannot be told apart, though their combination is exact. Where the marginal curve is missing the matrix is singular in float64.`);
  } else {
    const near = rows.filter(r => Math.abs(r.sin_iota) >= 0.9);
    html("p-sing", table(["sin ι", "cond (angles)", "cond (z)", "cond (flow)", "corr(ψ, φ₀) conditional", "marginal"], near.map(r => [r.sin_iota, sig(r.cond_angles, 3), sig(r.cond_z, 4), sig(r.cond_flow, 3), r.psi_phi_conditional.toFixed(4), Number.isFinite(r.psi_phi_marginal) ? r.psi_phi_marginal.toFixed(4) : "singular in float64"])));
    html("lg-sing", "");
    html("n-sing", `The grid from |sin ι| = 0.9 to ${Math.max(...near.map(r => Math.abs(r.sin_iota)))}. The angle parametrisation's conditioning is ${sig(Math.min(...near.map(r => r.cond_angles).filter(Number.isFinite)), 2)} at the start of it; the flow's stays within ${sig(Math.min(...near.map(r => r.cond_flow)), 2)}–${sig(Math.max(...near.map(r => r.cond_flow)), 2)}.`);
  }
}

function branchCard() {
  const host = document.getElementById("p-branch"), b = B(), d = b.degeneracy, v = S.branch;
  host.innerHTML = "";
  if (v === "variants") {
    html("p-branch", table(["step", "ψ + …", "φ₀ + …", "signal mismatch", "z difference"], d.variants.map((x, k) => [`k = ${k}`, `${k}π/2`, `${k % 2}π`, sig(x.signal, 3), sig(x.z, 3)]).concat([["a wrong step", "π/4", "0", `${sig(d.wrong_step_median, 4)} (median)`, "–"]])));
    html("lg-branch", "");
    html("n-branch", `(ψ + kπ/2, φ₀ + (k mod 2)π) keeps φ₊ = φ₀ + 2ψ and φ₋ = φ₀ − 2ψ modulo 2π, so all four give one signal (mismatch ≤ ${sig(Math.max(...d.variants.map(x => x.signal)), 2)} over ${d.sources} sources) while a step that keeps neither gives ${sig(d.wrong_step_median, 3)}. <code>flow_to_physical</code> returns ψ in (−π/2, π/2) up to 2π: of ${d.branch_counts.reduce((p, q) => p + q, 0).toLocaleString()} sources it lands on branch 0, 1, 2, 3 for ${d.branch_counts.map(c => c.toLocaleString()).join(", ")}, so the four are about equally likely; the φ₀ offset of the branch is ${sig(d.branch_phi0_offset, 2)}.`);
  } else if (v === "psi") {
    histChart(host, d.returned_psi, {xlabel: "returned ψ [rad]", ylabel: "density", label: "returned psi", fmt: fix2});
    html("lg-branch", legendBox("var(--bar)", `${b.sources.toLocaleString()} sources`) + legendItem("var(--ink-2)", "(2π − 4|c|)/π², c = ψ centred", "4 3"));
    html("n-branch", `ψ comes back as (φ₊ − φ₋)/4 with each φ in (−π, π], so only c in (−π/2, π/2) is returned (centred ψ spans ${d.returned_psi_centred.map(x => x.toFixed(3)).join(" to ")}), with density (2π − 4|c|)/π²: a triangle, not the uniform ψ the prior drew.${chi(d.returned_psi)}`);
  } else if (v === "phi0") {
    const h = d.returned_phi0, x = centres(h.edges);
    histChart(host, h, {xlabel: "returned φ₀ [rad]", ylabel: "density", label: "returned phi0", fmt: fix2, extra: [{line: x.map(t => [t, (PI - Math.abs(wrap(t))) / PI ** 2]), color: COL[1], width: 1.5, dash: "2 3"}]});
    html("lg-branch", legendBox("var(--bar)", `${b.sources.toLocaleString()} sources`) + legendItem("var(--ink-2)", "uniform (phase at the middle)", "4 3") + legendItem(COL[1], "(π − |c|)/π² (phase at the start)", "2 3"));
    html("n-branch", `φ₀ comes back as (φ₊ + φ₋)/2. With the phase referenced at the window middle (<code>problem.phase_reference</code> = ${b.problem.phase_reference}) it is uniform; referenced at the start it would be the triangle (π − |c|)/π², whereas the start reference adds an f₀-dependent shift of ${sig(b.phase.shift.min, 3)}–${sig(b.phase.shift.max, 3)} rad.${chi(h)}`);
  } else {
    const p = d.pole_phase;
    html("p-branch", table(["pole", "kept: shift φ∓ by 0", "lost: move φ± by 2 × " + d.pole_shift], Object.keys(p).map(k => [`sin ι = ${k}`, sig(p[k].kept, 3), sig(p[k].lost, 3)])));
    html("lg-branch", "");
    html("n-branch", `At sin ι = ±1 one of z± vanishes and only φ∓ survives (φ₋ = φ₀ − 2ψ at +1, φ₊ at −1). Moving ψ by ${d.pole_shift} and φ₀ by 2 × ${d.pole_shift} (the sign making φ∓ unchanged) leaves the signal at ${sig(Math.max(...Object.values(p).map(x => x.kept)), 2)}; moving the other way, which changes the surviving phase by 4 × ${d.pole_shift}, costs ${sig(p["+1"].lost, 4)} = 2 sin ${(2 * d.pole_shift).toFixed(1)}. Only the surviving phase is determined by the signal.`);
  }
}

function tripCard() {
  const b = B(), r = b.round_trip, f = r.float64, g = r.float32, names = [["f0", "f₀ (relative)"], ["fdot", "ḟ (relative)"], ["A", "A (relative)"], ["sky", "sky (unit vector)"], ["sin_iota", "sin ι"], ["z", "z (flow coordinates)"], ["phase_minus", "φ₋ = φ₀ − 2ψ"], ["phase_plus", "φ₊ = φ₀ + 2ψ"], ["signal_median", "signal mismatch, median"], ["signal_max", "signal mismatch, max"]];
  html("p-trip", table(["round trip", "float64", "float32 flow output"], names.map(([k, n]) => [n, sig(f[k], 2), sig(g[k], 2)])));
  const poles = r.poles;
  html("p-poles", table(["sin ι", "gap", "float64 sin ι", "float64 signal", "float32 sin ι", "float32 signal"], poles.map(p => [p.sin_iota, p.gap, sig(p.float64.sin_iota, 2), sig(p.float64.signal, 2), sig(p.float32.sin_iota, 2), sig(p.float32.signal, 2)])));
  html("n-trip", `physical → flow → physical over ${b.sources.toLocaleString()} sources (signal mismatch over ${f.signal_sources}). The angles come back only up to the four-fold choice, so the test is on φ± and on the signal itself. In float64 the worst signal mismatch is ${sig(f.signal_max, 2)} (median ${sig(f.signal_median, 2)}). The float32 row feeds the flow's float32-rounded output into the float64 chart, not a full float32 chart, and reaches ${sig(g.signal_max, 2)} (median ${sig(g.signal_median, 2)}). The second table pushes sources to |sin ι| = 1 − gap: sin ι is recovered to ${sig(Math.max(...poles.filter(p => p.gap >= 1e-6).map(p => p.float64.sin_iota)), 2)} (float64) for gap ≥ 10⁻⁶ and to ${sig(Math.max(...poles.map(p => p.float64.sin_iota)), 2)} at worst, the clip at ±(1 − 10⁻⁶); the float32 signal error is ${sig(Math.min(...poles.map(p => p.float32.signal)), 2)}–${sig(Math.max(...poles.map(p => p.float32.signal)), 2)}.`);
}

function phaseCard() {
  const host = document.getElementById("p-phase"), b = B(), ph = b.phase, v = S.phase;
  host.innerHTML = "";
  const sh = ph.shift;
  if (v === "cond" || v === "marg") {
    const k = v === "cond" ? "f0_phi_conditional" : "f0_phi_marginal", med = v === "cond" ? "f0_phi_conditional_median" : "f0_phi_marginal_median";
    const ls = [["start", COL[1]], ["middle", COL[0]]].map(([n, c]) => ({n, c, l: hline(ph[n][k])})), top = Math.max(...ls.flatMap(x => x.l.map(p => p[1])));
    chart(host, {x: [-1, 1], y: [0, top * 1.12], xlabel: `corr(f₀, φ), ${v === "cond" ? "conditional" : "marginal"}`, ylabel: "density", label: "correlation of f0 and phi",
      layers: ls.map(x => ({line: x.l, color: x.c, width: 2})),
      hover: ({x}) => { const i = nearest(ls[0].l.map(p => p[0]), x); return {cross: ls[0].l[i][0], html: `<b>r ≈ ${ls[0].l[i][0].toFixed(2)}</b>` + ls.map(l => trow(l.c, `${l.n}: ${sig(l.l[i][1], 3)}`)).join("")}; }});
    html("lg-phase", ls.map(x => legendItem(x.c, `phase at the ${x.n}`)).join(""));
    html("n-phase", `The Fisher correlation of f₀ with the phase φ over ${ph.sources} sources of the band (${v === "cond" ? "the other parameters held" : "all others marginalised"}). Referenced at the window start the median is ${ph.start[med].toFixed(3)}; at the middle it is ${ph.middle[med].toFixed(3)}. ${v === "cond" ? `ḟ with φ goes ${ph.start.fdot_phi_conditional_median.toFixed(3)} → ${ph.middle.fdot_phi_conditional_median.toFixed(3)}.` : `ḟ with φ goes ${ph.start.fdot_phi_marginal_median.toFixed(3)} → ${ph.middle.fdot_phi_marginal_median.toFixed(3)}.`} Moving the reference from the middle to the start shifts the phase by ${sig(sh.min, 3)}–${sig(sh.max, 3)} rad (chirp part ≤ ${sig(sh.chirp_part_max, 3)}), so φ and f₀ are almost the same parameter. Referencing t at the middle removes most of that shift.`);
  } else {
    const pts = ph.middle.phi_std.map((m, i) => [Math.log10(m), Math.log10(ph.start.phi_std[i])]), lo = Math.min(...pts.flat()), hi = Math.max(...pts.flat());
    chart(host, {x: pad([lo, hi], 0.04), y: pad([lo, hi], 0.04), xlog: true, ylog: true, xlabel: "σ(φ), middle reference [rad]", ylabel: "σ(φ), start reference [rad]", label: "phase uncertainty, start against middle", H: 320,
      layers: [{line: [[lo, lo], [hi, hi]], color: "var(--ink-2)", width: 1, dash: "4 3"}, {dots: pts, color: "var(--c1)", opacity: 0.5}],
      hover: h => { const hit = nearestPoint(pts, h); if (!hit) return null; const [[x, y]] = hit; return {ring: [x, y], html: trow("var(--c1)", `middle ${sig(10 ** x, 3)}, start ${sig(10 ** y, 3)}`)}; }});
    html("lg-phase", legendDot("var(--c1)", `${pts.length} sources`) + legendItem("var(--ink-2)", "equal", "4 3"));
    html("n-phase", `The Fisher standard deviation of φ in each reference, per source. It is ${sig(ph.phi_std_ratio_median, 3)}× larger at the start (median ${sig(ph.start.phi_std_median, 3)} rad against ${sig(ph.middle.phi_std_median, 3)}): the start phase carries f₀'s uncertainty over half the window. In float32 the start's shift (up to ${sig(sh.max, 3)} rad) would be rounded by a median ${sig(sh.float32_error_median, 2)} rad (max ${sig(sh.float32_error_max, 2)}), another reason to keep the reference at the middle.`);
  }
}

function meaning() {
  const n = PG.bands.narrow, f = PG.bands.full, fs = f.flow.map(x => x.std), d = f.degeneracy, f64 = f.round_trip.float64, f32 = f.round_trip.float32, ph = f.phase, rows = f.singularity.rows;
  const cmax = Math.max(...rows.map(r => r.cond_angles).filter(Number.isFinite));
  html("meaning", [
    `<b>The prior is uniform in orientation.</b> sin ι, ψ and φ₀ are uniform, and the χ²/dof of each histogram is in the notes of card 1 and card 5. κ, the SNR of one amplitude unit averaged over ι, sets the SNR-7 line on the <a href="../f0-fdot-amplitude/f0-fdot-amplitude.html">f₀, ḟ, A page</a>.`,
    `<b>The flow predicts z±, not the angles.</b> z± = A(1 ∓ s)²/4 · e^{i(φ₀ ± 2ψ)} are the two circular components of the polarization ellipse. Their warped Re/Im have std ${sig(Math.min(...fs), 3)}–${sig(Math.max(...fs), 3)} (narrow ${sig(Math.min(...n.flow.map(x => x.std)), 3)}–${sig(Math.max(...n.flow.map(x => x.std)), 3)}) against ≈ 1 for the other coordinates.`,
    `<b>The angles are singular where the chart is not.</b> The Fisher condition number is ${sig(cmax, 2)} in (ln A, s, ψ, φ₀) toward the poles, ${sig(Math.max(...rows.map(r => r.cond_z).filter(Number.isFinite)), 3)} in z and ≤ ${sig(Math.max(...rows.map(r => r.cond_flow).filter(Number.isFinite)), 2)} in the flow's coordinates.`,
    `<b>Four angle choices are one signal.</b> (ψ + kπ/2, φ₀ + (k mod 2)π) agree to ${sig(Math.max(...d.variants.map(v => v.signal)), 2)}, against ${sig(d.wrong_step_median, 3)} for a wrong step, so evaluation must compare signals or z±, never ψ and φ₀ one by one. At a pole only one phase survives.`,
    `<b>The round trip is exact to the float.</b> Signal mismatch ${sig(f64.signal_max, 2)} in float64 and ${sig(f32.signal_max, 2)} from a float32 flow output; sin ι is recovered to ${sig(Math.max(...f.round_trip.poles.filter(p => p.gap >= 1e-6).map(p => p.float64.sin_iota)), 2)} for gap ≥ 10⁻⁶.`,
    `<b>The phase is referenced at the window middle.</b> That cuts corr(f₀, φ) from ${ph.start.f0_phi_conditional_median.toFixed(2)} to ${ph.middle.f0_phi_conditional_median.toFixed(2)} (narrow-B ${n.phase.middle.f0_phi_conditional_median.toFixed(2)}) and makes the returned φ₀ uniform; the start reference's phase also needs shifts of up to ${sig(ph.shift.max, 3)} rad, which float32 rounds by up to ${sig(ph.shift.float32_error_max, 2)} rad.`,
  ].map(t => `<li>${t}</li>`).join(""));
}

function render() { tiles(); angleCard(); formCard(); gallery(); explorer(); polCard(); sysCard(); flowCard(); singCard(); branchCard(); tripCard(); phaseCard(); meaning(); writeHash(); }
let pending = 0;
const schedule = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; render(); }); };
addEventListener("resize", schedule);
document.getElementById("om").addEventListener("input", ev => { S.om = +ev.target.value; explorer(); });
document.getElementById("play").addEventListener("click", ev => { playing = !playing; ev.target.textContent = playing ? "pause" : "play"; if (playing) tick(); else writeHash(); });
readHash();
render();
</script>
</body>
</html>
"""
def rounded(tree, digits: int = 5):
    """Every float in a JSON tree to `digits` significant figures; NaN becomes null."""
    return json.loads(
        json.dumps(tree),
        parse_float=lambda s: float(f"{float(s):.{digits}g}"),
        parse_constant=lambda s: None,
    )


def hist_quantile(h: dict, q: float) -> float:
    """Quantile of a histogram, interpolated linearly inside the bin."""
    target, acc = q * sum(h["counts"]), 0
    for i, c in enumerate(h["counts"]):
        if acc + c >= target:
            return h["edges"][i] + (target - acc) / c * (
                h["edges"][i + 1] - h["edges"][i]
            )
        acc += c
    return h["edges"][-1]


def wrap(v: float) -> float:
    """Angle in (-pi, pi]."""
    return (v + math.pi) % TAU - math.pi


def to_z(a: float, s: float, psi: float, phi0: float) -> list[float]:
    """Port of chart.physical_to_flow for one source."""
    rp, rm = math.asinh(a * (1 - s) ** 2 / 4), math.asinh(a * (1 + s) ** 2 / 4)
    fp, fm = phi0 + 2 * psi, phi0 - 2 * psi
    return [rp * math.cos(fp), rp * math.sin(fp), rm * math.cos(fm), rm * math.sin(fm)]


def to_phys(z: list[float]) -> dict:
    """Port of chart.flow_to_physical for one source."""
    op, om = (math.sqrt(math.sinh(math.hypot(z[i], z[i + 1]))) for i in (0, 2))
    fp, fm = math.atan2(z[1], z[0]), math.atan2(z[3], z[2])
    s = min(max((om - op) / max(op + om, TINY), -(1 - 1e-6)), 1 - 1e-6)
    return dict(A=(op + om) ** 2, s=s, psi=((fp - fm) / 4) % TAU, phi0=((fp + fm) / 2) % TAU)


def port_errors(rows: list[dict]) -> dict:
    """Max error of the Python port against the drawn test vectors."""
    err = dict(z=0.0, A=0.0, s=0.0, psi=0.0, phi0=0.0)
    for r in rows:
        z, back = to_z(r["A"], r["sin_iota"], r["psi"], r["phi0"]), to_phys(r["z"])
        err["z"] = max(err["z"], *(abs(a - b) for a, b in zip(z, r["z"])))
        err["A"] = max(err["A"], abs(back["A"] / r["back"]["A"] - 1))
        err["s"] = max(err["s"], abs(back["s"] - r["back"]["sin_iota"]))
        err["psi"] = max(err["psi"], abs(wrap(back["psi"] - r["back"]["psi"])))
        err["phi0"] = max(err["phi0"], abs(wrap(back["phi0"] - r["back"]["phi0"])))
    return err


def chi2(h: dict, reference: list[float]) -> tuple[float, int]:
    """Chi-squared and dof over bins whose reference count is at least 5."""
    pairs = [(c, r) for c, r in zip(h["counts"], reference) if r >= 5]
    return sum((c - r) ** 2 / r for c, r in pairs), len(pairs) - 1


def returned_references(h: dict, psi: bool) -> list[float]:
    """Expected counts of the returned psi (triangle on the centred branch) or phi0 (uniform)."""
    n, width = sum(h["counts"]) + h["outside"], h["edges"][1] - h["edges"][0]
    if not psi:
        return [n * width / TAU] * len(h["counts"])
    mids = [(a + b) / 2 for a, b in zip(h["edges"], h["edges"][1:])]
    centred = [abs(wrap(m)) for m in mids]
    return [
        n * width * (TAU - 4 * c) / math.pi**2 if c < math.pi / 2 else 0.0
        for c in centred
    ]


def with_chi2(tree):
    """Add chi2 and dof to every histogram that carries a reference."""
    if isinstance(tree, dict):
        if "counts" in tree and "reference" in tree:
            tree["chi2"], tree["dof"] = chi2(tree, tree["reference"])
        for v in tree.values():
            with_chi2(v)
    elif isinstance(tree, list):
        for v in tree:
            with_chi2(v)
    return tree


def page_data(drawn: dict) -> dict:
    """What the page plots, per band."""
    bands = {}
    for band in BANDS:
        b = {k: v for k, v in drawn["bands"][band].items() if k != "seconds"}
        deg = b["degeneracy"]
        deg["returned_psi"]["reference"] = returned_references(deg["returned_psi"], True)
        deg["returned_phi0"]["reference"] = returned_references(deg["returned_phi0"], False)
        bands[band] = with_chi2(b)
    data = rounded(dict(windows=drawn["windows"], bands=bands))
    for band in BANDS:
        data["bands"][band]["test_vectors"] = drawn["bands"][band]["test_vectors"]
    return data


def referenced(tree, path=""):
    """Yield (path, hist) for every histogram with a reference."""
    if isinstance(tree, dict):
        if "counts" in tree and "reference" in tree:
            yield path, tree
        for k, v in tree.items():
            yield from referenced(v, f"{path}/{k}")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from referenced(v, f"{path}[{i}]")


def report(data: dict) -> None:
    """Print every number the README quotes."""
    for band, b in data["bands"].items():
        print(f"== {band}: {b['windows']} windows, {b['sources']} sources, phase reference {b['problem']['phase_reference']}")
        print("  chi2/dof: " + "; ".join(f"{p.split('/', 1)[1]} {h['chi2']:.0f}/{h['dof']}" for p, h in referenced(b)))
        tv = b["test_vectors"]
        e = port_errors(tv["rows"])
        print(f"  python port vs test vectors (max, angles wrapped): " + ", ".join(f"{k} {v:.2e}" for k, v in e.items()) + f"; ellipse identity {tv['ellipse_identity']:.2e}")
        zs = [f["std"] for f in b["flow"]]
        fs = b["flow_std"]
        print(f"  z std {min(zs):.3f}-{max(zs):.3f} ({[round(z, 3) for z in zs]}); f0, fdot, l, b std {[round(f, 3) for f in fs[:4]]}; z/other {sum(zs) / 4 / (sum(fs[:4]) / 4):.2f}")
        print(f"  max |z coord| {max(max(abs(f['hist']['edges'][0]), abs(f['hist']['edges'][-1])) for f in b['flow']):.3f}; radius rms {[r['rms'] for r in b['radius']]}")
        off = max(abs(x) for i, r in enumerate(b["corr"]) for j, x in enumerate(r) if i != j)
        print(f"  flow corr max off-diagonal {off:.4f}")
        h = b["helicity"]
        sin = b["physical"]["orientation"]["sin iota"]
        pos = sum(c for c, hi in zip(sin["counts"], sin["edges"][1:]) if hi <= 0) / sum(sin["counts"])
        print(f"  axis ratio median {hist_quantile(h['axis ratio'], 0.5):.3f}, check {b['axis_ratio_check']:.2e}; share sin iota<0 {pos:.3f}; split {b['split_check']:.2e}")
        q = h["log10 |z+|/|z-|"]
        print(f"  share log10|z+|/|z-| < 0 (z- larger) {sum(c for c, hi in zip(q['counts'], q['edges'][1:]) if hi <= 0) / sum(q['counts']):.3f}")
        d, r64, r32, ph, sg = b["degeneracy"], b["round_trip"]["float64"], b["round_trip"]["float32"], b["phase"], b["singularity"]
        print(f"  four equivalent: signal <= {max(v['signal'] for v in d['variants']):.2e}, z <= {max(v['z'] for v in d['variants']):.2e}, wrong step median {d['wrong_step_median']:.4f}; branches {d['branch_counts']}; phi0 offset {d['branch_phi0_offset']:.2e}")
        print(f"  pole: kept {[v['kept'] for v in d['pole_phase'].values()]}, lost {[v['lost'] for v in d['pole_phase'].values()]} (2 sin {2 * d['pole_shift']:.1f} = {2 * math.sin(2 * d['pole_shift']):.4f})")
        print(f"  round trip signal_max f64 {r64['signal_max']:.2e} f32 {r32['signal_max']:.2e}; median {r64['signal_median']:.2e} / {r32['signal_median']:.2e}; poles sin_iota err f64 {[p['float64']['sin_iota'] for p in b['round_trip']['poles']]}")
        print(f"  corr(f0, phi) cond start {ph['start']['f0_phi_conditional_median']:.3f} -> middle {ph['middle']['f0_phi_conditional_median']:.3f}; marginal {ph['start']['f0_phi_marginal_median']:.3f} -> {ph['middle']['f0_phi_marginal_median']:.3f}; sigma ratio {ph['phi_std_ratio_median']:.3f}; shift {ph['shift']['min']:.3g}-{ph['shift']['max']:.3g}, f32 err max {ph['shift']['float32_error_max']:.3g}")
        rows = sg["rows"]
        fin = lambda k: [r[k] for r in rows if r[k] is not None and r[k] == r[k]]
        print(f"  cond angles max {max(fin('cond_angles')):.2e}, at 0 {next(r['cond_angles'] for r in rows if r['sin_iota'] == 0):.3f}; z {min(fin('cond_z')):.3f}-{max(fin('cond_z')):.3f}; flow {min(fin('cond_flow')):.2f}-{max(fin('cond_flow')):.2f}; grid to {max(abs(r['sin_iota']) for r in rows)}; z Fisher variation {sg['z_fisher_variation']:.2e}")


def main() -> None:
    drawn = json.loads((REPORT / "data" / "polarization.json").read_text())
    data = page_data(drawn)
    page = (
        PAGE.replace("__PAGE__", json.dumps(data, separators=(",", ":")))
        .replace("__WINDOWS__", f"{drawn['windows']:,}")
        .replace(
            "__SUB__",
            f"Built {date.today().isoformat()} from <code>data/polarization.json</code>.",
        )
    )
    out = REPORT / "polarization.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")
    report(data)


if __name__ == "__main__":
    main()
