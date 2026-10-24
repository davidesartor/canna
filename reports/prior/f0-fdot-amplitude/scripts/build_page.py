"""Write f0-fdot-amplitude.html from data/: the (f0, fdot, A) prior as a corner, who in it the data see, and what the flow is given.

Run from the repo root (numpy only, the login node is fine):
    uv run --frozen python reports/prior/f0-fdot-amplitude/scripts/build_page.py
"""

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from einops import rearrange

FOLDER = Path(__file__).parents[1]
DATA = FOLDER / "data"
BANDS = ("narrow", "full")
VISIBLE_SNR = 7.0
# the corner's 2-D cells per axis, its 1-D bins, and the visible sources it shows at most
CELLS = 64
BINS = 80
VISIBLE_POINTS = 2000
CENTRE_BINS = 24
CATALOGUE_DRAWS = 1_000_000
# the flow's z per chart coordinate, shown over ±Z_RANGE
Z_NAMES = ("z f₀", "z ḟ", "z l", "z b", "z Re z₊", "z Im z₊", "z Re z₋", "z Im z₋")
Z_RANGE = 5.0
# the contours: the 2-D masses of 1σ and 2σ, as in the eval corners
MASSES = (0.393, 0.865)
SMOOTH = np.array([0.054, 0.242, 0.399, 0.242, 0.054])
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior: f₀, ḟ, A</title>
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
  --ramp: #86b6ef,#6da7ec,#5598e7,#3987e5,#2a78d6,#256abf,#1c5cab,#184f95,#104281,#0d366b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --bar: #52514e;
    --ramp: #184f95,#1c5cab,#256abf,#2a78d6,#3987e5,#5598e7,#6da7ec,#86b6ef,#9ec5f4,#cde2fb;
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
.sub { color: var(--ink-2); margin: 0 0 24px; max-width: 80ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 90ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 0 0 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 22px; flex: none; }
.row { display: flex; align-items: center; justify-content: flex-start; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.sticky { position: sticky; top: 0; z-index: 3; background: var(--page); padding: 10px 0; border-bottom: 1px solid var(--border); }
input[type=range] { touch-action: none; }
.row .k { color: var(--ink-2); font-size: 13px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; vertical-align: middle; }
.seg button { font: inherit; font-size: 13px; color: var(--ink-2); background: transparent; border: 0; padding: 4px 10px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
.row input[type=range] { width: min(240px, 50vw); accent-color: var(--c3); margin: 0; }
#k-label { font-size: 13px; font-variant-numeric: tabular-nums; min-width: 14ch; }
tr[data-i] { cursor: pointer; }
tr[data-i]:hover td { background: var(--grid); }
tr.on td { color: var(--ink); font-weight: 600; }
.duo { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 8px; }
.corner { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 4px 12px; }
.trio { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.quad { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 900px) { .quad { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 700px) { .corner, .trio, .duo { grid-template-columns: minmax(0, 1fr); } .corner .blank { display: none; } }
.fit { grid-column: 2 / 4; align-self: start; font-size: 13px; color: var(--ink-2); }
@media (max-width: 700px) { .fit { grid-column: auto; } }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg text.lab { fill: var(--ink-2); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; white-space: nowrap; z-index: 2; }
.tip .row { color: var(--ink-2); font-variant-numeric: tabular-nums; margin: 0; gap: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; margin: 4px 0 8px; }
th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
.meaning ul { margin: 8px 0 0; padding-left: 20px; }
.meaning li { margin: 0 0 8px; max-width: 80ch; }
#err { position: sticky; top: 0; z-index: 10; background: #c0392b; color: #fff; padding: 10px 16px; font-size: 14px; }
</style>
</head>
<body>
<div id="err" role="alert" hidden></div>
<main>
<p class="nav">The LisaGB prior by parameter group: <b>f₀, ḟ, A</b> · <a data-page="../polarization/polarization.html" href="../polarization/polarization.html">polarization</a> · <a data-page="../sky/sky.html" href="../sky/sky.html">sky</a> · <a data-page="../response-span/response-span.html" href="../response-span/response-span.html">response span</a></p>
<h1>The LisaGB prior: f₀, ḟ and A</h1>
<p class="sub">__SUB__</p>
<div class="row sticky"><span class="k">Band</span><span class="seg" id="band"></span>
  <span class="k" style="margin-left:12px">Cut</span><input type="range" id="k" min="0" step="1" aria-label="SNR cut k"><span id="k-label"></span></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>The joint prior, as training draws it</h2>
  <p class="note">__N_CORNER__</p>
  <div class="legend" id="lg-corner"></div>
  <div class="corner">
    <div><div id="k00"></div><div class="tip"></div></div>
    <div class="fit">__FIT__</div>
    <div><div id="k01"></div><div class="tip"></div></div>
    <div><div id="k11"></div><div class="tip"></div></div>
    <div class="blank"></div>
    <div><div id="k02"></div><div class="tip"></div></div>
    <div><div id="k12"></div><div class="tip"></div></div>
    <div><div id="k22"></div><div class="tip"></div></div>
  </div>
</div>
<div class="card">
  <h2>Which windows hold a source the data see?</h2>
  <p class="note">__N_VISIBLE__</p>
  <div class="legend" id="lg-visible"></div>
  <div id="p-centre"></div><div class="tip"></div>
  <div id="p-visible"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What the flow is given</h2>
  <p class="note">__N_FLOW__</p>
  <div class="legend" id="lg-flow"></div>
  <div class="quad" id="p-z"></div>
</div>
<div class="card">
  <h2>What the cut does</h2>
  <p class="note">__N_CUT__</p>
  <div class="legend" id="lg-cut"></div>
  <div class="trio">
    <div><div id="p-snr"></div><div class="tip"></div></div>
    <div><div id="p-iota"></div><div class="tip"></div></div>
    <div><div id="p-b"></div><div class="tip"></div></div>
  </div>
  <div class="duo">
    <div><div id="p-mass"></div><div class="tip"></div></div>
    <div><div id="p-ess"></div><div class="tip"></div></div>
  </div>
  <div id="cut-table"></div>
</div>
<div class="card meaning">
  <h2>What this means</h2>
  <ul>__MEANING__</ul>
</div>
</main>
<script>
const PG = __PAGE__;
const S = {band: "full", k: String(PG.training_k)};
const OPTIONS = {band: [["narrow", "narrow-B"], ["full", "full-B"]]};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => v > 0 && v < 10 ** -(d + 2) ? `${(100 * v).toExponential(1)}%` : `${(100 * v).toFixed(d)}%`;
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const nearest = (xs, v) => xs.reduce((best, x, i) => Math.abs(x - v) < Math.abs(xs[best] - v) ? i : best, 0);
const mhz = logf => sig(10 ** logf * 1e3, 3);
const html = (id, text) => document.getElementById(id).innerHTML = text;
const NAMES = ["f₀ [mHz]", "log₁₀ ḟ [Hz/s]", "log₁₀ A"];
const SHORT = ["f₀", "ḟ", "A"];

function tip(host, ev, text) {
  const t = host.nextElementSibling, card = host.closest(".card").getBoundingClientRect();
  t.innerHTML = text; t.style.display = "block";
  t.style.left = Math.max(4, Math.min(ev.clientX - card.left + 14, card.width - t.offsetWidth - 8)) + "px";
  t.style.top = (ev.clientY - card.top + 14) + "px";
}
const untip = host => host.nextElementSibling.style.display = "none";
const trow = (color, text) => `<div class="row">${color ? `<i class="dot" style="background:${color};width:8px;height:8px"></i>` : ""}${text}</div>`;
const legendItem = (color, text, dash) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${color}" stroke-width="2"${dash ? ` stroke-dasharray="${dash}"` : ""}/></svg>${text}</span>`;
const legendBox = (color, text) => `<span><i class="dot" style="background:${color};border-radius:2px"></i>${text}</span>`;
const legendDot = (color, text) => `<span><i class="dot" style="background:${color}"></i>${text}</span>`;

function niceTicks(lo, hi, n = 4) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag;
  const step = (r >= 5 ? 10 : r >= 2 ? 5 : r >= 1 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(+v.toPrecision(12));
  return out;
}
// f0 axes are log10 Hz, ticked in mHz
const fTicks = ([lo, hi]) => [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2].filter(f => Math.log10(f) >= lo - 1e-9 && Math.log10(f) <= hi + 1e-9).map(f => [Math.log10(f), sig(f * 1e3, 2)]);
const axisTicks = (c, range) => c === 0 ? fTicks(range) : niceTicks(range[0], range[1], 4).map(t => [t, sig(t, 3)]);
// generic plot: layers in data units; ticks as [position, label] pairs
function chart(host, o) {
  const W = Math.max(240, host.clientWidth), H = o.H ?? 280, m = {l: o.ml ?? 54, r: 16, t: 18, b: o.xlabel ? 38 : 8};
  const [x0, x1] = o.x, [y0, y1] = o.y;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const cx = v => clamp(X(v), m.l, W - m.r), cy = v => clamp(Y(v), m.t, H - m.b);
  const pts = arr => arr.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label}">`;
  (o.yticks ?? niceTicks(y0, y1, 4).map(t => [t, sig(t, 3)])).forEach(([t, l]) => s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${l}</text>`);
  (o.xticks ?? niceTicks(x0, x1, 5).map(t => [t, sig(t, 3)])).forEach(([t, l]) => s += `<line x1="${X(t)}" x2="${X(t)}" y1="${H - m.b}" y2="${H - m.b + 4}" stroke="var(--axis)"/>${o.xlabel ? `<text x="${X(t)}" y="${H - m.b + 17}" text-anchor="middle">${l}</text>` : ""}`);
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--axis)"/>`;
  s += `${o.xlabel ? `<text x="${(m.l + W - m.r) / 2}" y="${H - 3}" text-anchor="middle">${o.xlabel}</text>` : ""}<text x="${m.l}" y="${m.t - 6}">${o.ylabel}</text>`;
  s += `<clipPath id="clip-${host.id}"><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/></clipPath><g clip-path="url(#clip-${host.id})">`;
  for (const L of o.layers) {
    if (L.image) s += `<image href="${L.image.url}" x="${X(L.image.x[0])}" y="${Y(L.image.y[1])}" width="${X(L.image.x[1]) - X(L.image.x[0])}" height="${Y(L.image.y[0]) - Y(L.image.y[1])}" preserveAspectRatio="none"/>`;
    if (L.span) s += `<rect x="${cx(L.span[0])}" width="${Math.max(1.5, cx(L.span[1]) - cx(L.span[0]))}" y="${m.t}" height="${H - m.t - m.b}" fill="${L.color}" opacity="${L.opacity ?? 0.1}"/>`;
    if (L.area && L.area.length) s += `<polygon points="${pts([[L.area[0][0], L.base ?? y0], ...L.area, [L.area.at(-1)[0], L.base ?? y0]])}" fill="${L.color}" opacity="${L.opacity ?? 1}"/>`;
    if (L.bars) L.bars.forEach(([a, b, v]) => { const base = Y(L.base ?? clamp(0, y0, y1)); s += `<rect x="${X(a).toFixed(1)}" y="${Math.min(Y(v), base).toFixed(1)}" width="${Math.max(X(b) - X(a), 0.5).toFixed(1)}" height="${Math.abs(base - Y(v)).toFixed(1)}" fill="${L.color}" rx="2"/>`; });
    if (L.line && L.line.length) s += `<polyline points="${pts(L.line)}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 2}" stroke-linejoin="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.segs && L.segs.length) s += `<path d="${L.segs.map(([a, b]) => `M${X(a[0]).toFixed(1)},${Y(a[1]).toFixed(1)}L${X(b[0]).toFixed(1)},${Y(b[1]).toFixed(1)}`).join("")}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 1.4}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.pts) L.pts.forEach(p => s += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="${L.r ?? 2}" fill="${L.color}" opacity="${L.opacity ?? 0.7}"/>`);
    if (L.hline !== undefined) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(L.hline)}" y2="${Y(L.hline)}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="${L.width ?? 1.2}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.vline !== undefined) s += `<line x1="${X(L.vline)}" x2="${X(L.vline)}" y1="${m.t}" y2="${H - m.b}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="${L.width ?? 1.2}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
  }
  s += "</g>";
  // labels outside the clip, so text at an edge stays readable
  for (const L of o.layers) {
    if (L.vline !== undefined && L.text) s += `<text class="lab" x="${X(L.vline) + (L.tdx ?? 0)}" y="${m.t + 12 + (L.dy ?? 0)}" text-anchor="${L.anchor ?? "middle"}">${L.text}</text>`;
    if (L.hline !== undefined && L.text) s += `<text class="lab" x="${L.left ? m.l + 4 : W - m.r - 4}" y="${cy(L.hline) - 4}" text-anchor="${L.left ? "start" : "end"}">${L.text}</text>`;
  }
  s += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" visibility="hidden"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  host.insertAdjacentHTML("beforeend", s + "</svg>");
  const svg = host.lastElementChild, cross = svg.querySelector(".cross"), hit = svg.querySelector(".hit");
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W, py = (ev.clientY - r.top) / r.height * H;
    const out = o.hover({x: x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0), y: y0 + (H - m.b - py) / (H - m.t - m.b) * (y1 - y0)});
    cross.setAttribute("visibility", out?.cross !== undefined ? "visible" : "hidden");
    if (!out) return untip(host);
    if (out.cross !== undefined) { cross.setAttribute("x1", X(out.cross)); cross.setAttribute("x2", X(out.cross)); }
    tip(host, ev, out.html);
  });
  hit.addEventListener("pointerleave", () => { untip(host); cross.setAttribute("visibility", "hidden"); });
}

// a colour on the sequential ramp at t in [0, 1]
function rampAt(t) {
  const ramp = css("--ramp").split(",").map(h => [1, 3, 5].map(j => parseInt(h.trim().slice(j, j + 2), 16)));
  const u = clamp(t, 0, 1) * (ramp.length - 1), i = Math.min(Math.floor(u), ramp.length - 2), f = u - i;
  return `rgb(${ramp[i].map((v, n) => Math.round(v + f * (ramp[i + 1][n] - v))).join(",")})`;
}
// marching squares over the cell centres (corner.html's, on an n × n grid): line segments of one level, in cell units
function contour(grid, n, level) {
  const segs = [], at = (i, j) => grid[j * n + i];
  const lerp = (a, b) => (level - a) / (b - a);
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    const code = (a >= level) | (b >= level) << 1 | (c >= level) << 2 | (d >= level) << 3;
    if (code === 0 || code === 15) continue;
    const e = [[i + lerp(a, b), j], [i + 1, j + lerp(b, c)], [i + lerp(d, c), j + 1], [i, j + lerp(a, d)]];
    const pairs = {1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[3, 0], [1, 2]], 6: [[0, 2]], 7: [[3, 2]],
      8: [[2, 3]], 9: [[0, 2]], 10: [[0, 1], [2, 3]], 11: [[1, 2]], 12: [[1, 3]], 13: [[0, 1]], 14: [[0, 3]]}[code];
    for (const [p, q] of pairs) segs.push([e[p], e[q]]);
  }
  return segs;
}
// corner.html's 2-D cell fill: one colour, more opaque where denser
const images = {};
function densityImage(grid, n, colour, key) {
  if (images[key + colour]) return images[key + colour];
  const rgb = [1, 3, 5].map(j => parseInt(colour.slice(j, j + 2), 16));
  const cv = document.createElement("canvas"); cv.width = n; cv.height = n;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(n, n);
  for (let r = 0; r < n; r++) for (let i = 0; i < n; i++) {
    const v = grid[(n - 1 - r) * n + i], k = 4 * (r * n + i);
    for (let c = 0; c < 3; c++) img.data[k + c] = rgb[c];
    img.data[k + 3] = v > 0.02 ? Math.round(255 * 0.45 * v) : 0;
  }
  ctx.putImageData(img, 0, 0);
  return images[key + colour] = cv.toDataURL();
}

const B = () => PG.bands[S.band];
const K = () => Math.max(0, PG.ks.indexOf(+S.k));
// the current cut's draws of the band
const D = () => B().cuts[K()];
const kNow = () => PG.ks[K()];
const SUP = {"-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹"};
const decade = t => t === 0 ? "1" : `10${String(t).split("").map(c => SUP[c]).join("")}`;
const snrTicks = ([lo, hi]) => [-3, -2, -1, 0, 1, 2, 3, 4].filter(t => t >= lo && t <= hi).map(t => [t, sig(10 ** t, 1)]);
const cellH = host => Math.round(0.86 * Math.max(240, host.clientWidth));
// the source's physical values, for the tooltips
const physical = (c, v) => c === 0 ? `${SHORT[0]} ${mhz(v)} mHz` : c === 1 ? `${SHORT[1]} ${sig(10 ** v, 3)} Hz/s` : `${SHORT[2]} ${sig(10 ** v, 3)}`;

function diagonal(c) {
  const b = B(), host = document.getElementById(`k${c}${c}`), M = {...b.marginal[c], ...D().marginal[c]}; host.innerHTML = "";
  const line = arr => M.x.map((x, i) => [x, arr[i]]);
  const top = 1.12 * Math.max(...M.drawn, ...M.catalogue, ...(M.visible ?? [0]));
  const layers = [{area: line(M.drawn), color: "var(--c1)", opacity: 0.15, base: 0}, {line: line(M.drawn), color: "var(--c1)", width: 1.6},
    {line: line(M.catalogue), color: "var(--ink-2)", width: 1.3, dash: "5 3"}];
  if (M.visible) layers.push({line: line(M.visible), color: "var(--c2)", width: 1.6});
  chart(host, {x: b.range[c], y: [0, top], H: cellH(host), xlabel: NAMES[c], ylabel: "density", label: `${SHORT[c]} marginal`, yticks: [], ml: 44,
    xticks: axisTicks(c, b.range[c]), layers,
    hover: ({x}) => { const i = nearest(M.x, x); return {cross: M.x[i], html: `<b>${physical(c, M.x[i])}</b>` + trow("var(--c1)", `drawn ${sig(M.drawn[i])}`)
      + trow("var(--ink-2)", `catalogue ${sig(M.catalogue[i])}`) + (M.visible ? trow("var(--c2)", `SNR ≥ ${PG.visible_snr} ${sig(M.visible[i])}`) : "")}; }});
}

function pair(cx, cy) {
  const b = B(), host = document.getElementById(`k${cx}${cy}`), P = {...b.pairs[`${cx}${cy}`], ...D().pairs[`${cx}${cy}`]}, n = PG.cells; host.innerHTML = "";
  const [rx, ry] = [b.range[cx], b.range[cy]], dx = (rx[1] - rx[0]) / n, dy = (ry[1] - ry[0]) / n;
  const at = ([a, c]) => [rx[0] + (a + 0.5) * dx, ry[0] + (c + 0.5) * dy];
  const layers = [{image: {url: densityImage(P.drawn, n, css("--c1"), `${S.band}${K()}${cx}${cy}`), x: rx, y: ry}}];
  for (const level of P.drawn_levels) layers.push({segs: contour(P.drawn, n, level).map(([p, q]) => [at(p), at(q)]), color: "var(--c1)", width: 1.4});
  for (const level of P.catalogue_levels) layers.push({segs: contour(P.catalogue, n, level).map(([p, q]) => [at(p), at(q)]), color: "var(--ink-2)", width: 1.2, dash: "5 3"});
  const C = b.curves, curve = arr => C.log_f0.map((x, i) => [x, arr[i]]);
  if (cx === 0 && cy === 1) layers.push({line: curve(C.fdot_mean), color: "var(--ink)", width: 1.2, dash: "2 3"});
  if (cx === 0 && cy === 2) layers.push({line: curve(C.amp_mean), color: "var(--ink)", width: 1.2, dash: "2 3"}, {line: curve(C.amp_snr7), color: "var(--c2)", width: 1.8});
  if (cx === 0 && cy === 2 && kNow()) layers.push({line: C.log_f0.map((x, i) => [x, C.amp_snr7[i] + Math.log10(kNow() / PG.visible_snr)]), color: "var(--c3)", width: 1.8});
  const V = D().visible;
  layers.push({pts: V.map(v => [v[cx], v[cy]]), color: "var(--c2)", r: 1.8, opacity: 0.75});
  chart(host, {x: rx, y: ry, H: cellH(host), xlabel: NAMES[cx], ylabel: NAMES[cy], label: `${SHORT[cx]} against ${SHORT[cy]}`,
    xticks: axisTicks(cx, rx), yticks: axisTicks(cy, ry), layers,
    hover: ({x, y}) => {
      const i = Math.floor((x - rx[0]) / dx), j = Math.floor((y - ry[0]) / dy);
      if (i < 0 || j < 0 || i >= n || j >= n) return null;
      let out = `<b>${physical(cx, x)} · ${physical(cy, y)}</b>` + trow("var(--c1)", `drawn density ${sig(P.drawn[j * n + i], 2)} of its peak`);
      const k = nearest(V.map(v => Math.hypot((v[cx] - x) / dx, (v[cy] - y) / dy)), 0);
      if (V.length && Math.hypot((V[k][cx] - x) / dx, (V[k][cy] - y) / dy) < 1.5) out += trow("var(--c2)", `visible source: SNR ${sig(V[k][3])}, ${physical(0, V[k][0])}`);
      return {html: out};
    }});
}

function cornerCard() {
  [0, 1, 2].forEach(diagonal);
  pair(0, 1); pair(0, 2); pair(1, 2);
  const b = D();
  html("lg-corner", legendBox("color-mix(in srgb, var(--c1) 45%, transparent)", `drawn sources (${b.sources.toLocaleString()}${kNow() ? `, SNR ≥ ${kNow()}` : ""})`) + legendItem("var(--c1)", "their 39% and 86% regions")
    + legendItem("var(--ink-2)", "catalogue Gaussian in the band", "5 3") + legendDot("var(--c2)", `SNR ≥ ${PG.visible_snr} (${b.visible_count.toLocaleString()}${b.visible_count > b.visible.length ? `, ${b.visible.length.toLocaleString()} shown` : ""})`)
    + legendItem("var(--ink)", "conditional mean given f₀", "2 3") + legendItem("var(--c2)", "A at SNR 7 for a sky- and orientation-average source") + (kNow() ? legendItem("var(--c3)", `A at SNR ${kNow()}`) : ""));
}

function visibleCard() {
  const V = {...D().centre, edges: B().edges}, mid = V.edges.slice(0, -1).map((e, i) => (e + V.edges[i + 1]) / 2), range = [V.edges[0], V.edges.at(-1)];
  let host = document.getElementById("p-centre"); host.innerHTML = "";
  chart(host, {x: range, y: [0, 1.15 * Math.max(...V.share)], H: 120, ylabel: "share of windows", label: "window centres",
    xticks: fTicks(range), layers: [{bars: V.share.map((v, i) => [V.edges[i] + 0.06 * (V.edges[1] - V.edges[0]), V.edges[i + 1] - 0.06 * (V.edges[1] - V.edges[0]), v]), color: "var(--bar)"}],
    hover: ({x}) => { const i = clamp(Math.floor((x - range[0]) / (range[1] - range[0]) * mid.length), 0, mid.length - 1);
      return {cross: mid[i], html: `<b>centre ${mhz(V.edges[i])}–${mhz(V.edges[i + 1])} mHz</b>` + trow("var(--bar)", `${pct(V.share[i], 2)} of windows`)}; }});
  host = document.getElementById("p-visible"); host.innerHTML = "";
  const colour = k => rampAt(0.15 + 0.25 * (k - 1)), steps = arr => arr.flatMap((v, i) => v === null ? [] : [[V.edges[i], v], [V.edges[i + 1], v]]);
  const layers = V.at_least.map((arr, k) => ({line: steps(arr), color: colour(k + 1), width: 2}));
  layers.push({line: steps(V.source), color: "var(--ink)", width: 1.4, dash: "4 3"}, {hline: 0.5, color: "var(--axis)", width: 1, dash: "2 3"});
  chart(host, {x: range, y: [0, 1], H: 300, xlabel: "window centre [mHz]", ylabel: `share with SNR ≥ ${PG.visible_snr}`, label: "visible sources per window",
    xticks: fTicks(range), yticks: [0, 0.25, 0.5, 0.75, 1].map(t => [t, pct(t, 0)]), layers,
    hover: ({x}) => { const i = clamp(Math.floor((x - range[0]) / (range[1] - range[0]) * mid.length), 0, mid.length - 1);
      if (V.source[i] === null) return {cross: mid[i], html: `<b>centre ${mhz(V.edges[i])}–${mhz(V.edges[i + 1])} mHz</b>` + trow(null, "no windows here")};
      return {cross: mid[i], html: `<b>centre ${mhz(V.edges[i])}–${mhz(V.edges[i + 1])} mHz</b>` + V.at_least.map((arr, k) => trow(colour(k + 1), `${k + 1 === PG.n_sources ? "all" : `≥ ${k + 1}`} of ${PG.n_sources}: ${pct(arr[i], 1)} of windows`)).join("")
        + trow("var(--ink)", `${pct(V.source[i], 2)} of sources`)}; }});
  html("lg-visible", legendBox("var(--bar)", "share of windows centred here (top)") + V.at_least.map((a, k) => legendItem(colour(k + 1), k + 1 === PG.n_sources ? `all ${PG.n_sources} sources visible` : `≥ ${k + 1} visible`)).join("")
    + legendItem("var(--ink)", "a single source", "4 3"));
}

function flowCard() {
  const F = D().flow, grid = document.getElementById("p-z");
  const one = (host, h, o) => {
    host.innerHTML = "";
    const line = arr => h.x.map((x, i) => [x, arr[i]]), top = 1.15 * Math.max(...h.drawn, ...(h.reference ?? [0]));
    const layers = [...(o.spans ?? []), {area: line(h.drawn), color: "var(--c1)", opacity: 0.15, base: 0}, {line: line(h.drawn), color: "var(--c1)", width: 1.6}];
    if (h.reference) layers.push({line: line(h.reference), color: "var(--ink-2)", width: 1.3, dash: "5 3"});
    layers.push(...(o.marks ?? []));
    chart(host, {x: [h.x[0] - h.dx / 2, h.x.at(-1) + h.dx / 2], y: [0, top], H: 180, xlabel: o.xlabel, ylabel: "density", label: o.xlabel, yticks: [], ml: 16, layers,
      hover: ({x}) => { const i = nearest(h.x, x); return {cross: h.x[i], html: `<b>${o.xlabel} ${sig(h.x[i])}</b>` + trow("var(--c1)", `drawn ${sig(h.drawn[i])}`) + trow("var(--ink-2)", `N(0, 1) ${sig(h.reference[i])}`)
        + trow("var(--ink)", `std ${sig(h.std)}, |z| > ${sig(h.x.at(-1) + h.dx / 2)}: ${pct(h.outside, 2)}`)}; }});
  };
  grid.innerHTML = F.coords.map(() => `<div><div></div><div class="tip"></div></div>`).join("");
  F.coords.forEach((h, c) => one(grid.children[c].firstChild, h, {xlabel: h.name}));
  html("lg-flow", legendItem("var(--c1)", "drawn sources") + legendItem("var(--ink-2)", "N(0, 1), the flow's base", "5 3"));
}

// one density against the uncut draws, on fixed axes
function versus(id, h, o) {
  const host = document.getElementById(id); host.innerHTML = "";
  const line = arr => h.x.map((x, i) => [x, arr[i]]), top = 1.15 * Math.max(...h.drawn, ...h.reference);
  const label = kNow() ? `SNR ≥ ${kNow()}` : "uncut";
  chart(host, {x: [h.x[0] - h.dx / 2, h.x.at(-1) + h.dx / 2], y: [0, top], H: 220, xlabel: o.xlabel, ylabel: "density", label: o.xlabel, yticks: [], ml: 16, xticks: o.xticks,
    layers: [{area: line(h.drawn), color: "var(--c1)", opacity: 0.15, base: 0}, {line: line(h.drawn), color: "var(--c1)", width: 1.6},
      {line: line(h.reference), color: "var(--ink-2)", width: 1.3, dash: "5 3"}, ...(o.marks ?? [])],
    hover: ({x}) => { const i = nearest(h.x, x); return {cross: h.x[i], html: `<b>${o.value(h.x[i])}</b>` + trow("var(--c1)", `${label} ${sig(h.drawn[i])}`) + trow("var(--ink-2)", `uncut ${sig(h.reference[i])}`)}; }});
}
// a per-f0-bin line of steps, null bins left out
const steps = (edges, arr) => arr.flatMap((v, i) => v === null ? [] : [[edges[i], v], [edges[i + 1], v]]);
function perF0(id, arrays, o) {
  const host = document.getElementById(id), b = B(), E = b.edges, range = [E[0], E.at(-1)]; host.innerHTML = "";
  if (!arrays) { host.innerHTML = `<p class="note">No cut at k = 0: the catalogue prior has no weights.</p>`; return; }
  const mid = E.slice(0, -1).map((e, i) => (e + E[i + 1]) / 2);
  chart(host, {x: range, y: o.y, H: 220, xlabel: o.xlabel ?? "f₀ [mHz]", ylabel: o.ylabel, label: o.ylabel, xticks: fTicks(range), yticks: o.yticks,
    layers: [...arrays.map(a => ({line: steps(E, a.v), color: a.color, width: a.width ?? 1.8, dash: a.dash})), ...(o.marks ?? [])],
    hover: ({x}) => { const i = clamp(Math.floor((x - range[0]) / (range[1] - range[0]) * mid.length), 0, mid.length - 1);
      return {cross: mid[i], html: `<b>${o.xname ?? "f₀"} ${mhz(E[i])}–${mhz(E[i + 1])} mHz</b>` + arrays.map(a => trow(a.color, `${a.name} ${a.v[i] === null ? "–" : o.fmt(a.v[i])}`)).join("")}; }});
}
function cutCard() {
  const C = D().cut, k = kNow(), W = C.weights;
  const kMark = k ? [{vline: Math.log10(k), color: "var(--c3)", width: 1.4, text: `k = ${k}`, anchor: "end", tdx: -4}] : [];
  versus("p-snr", C.snr, {xlabel: "SNR", xticks: snrTicks([C.snr.x[0], C.snr.x.at(-1)]), value: x => `SNR ${sig(10 ** x)}`,
    marks: [{vline: Math.log10(PG.visible_snr), color: "var(--c2)", width: 1.2, text: "7", anchor: "start", tdx: 4, dy: 14}, ...kMark]});
  versus("p-iota", C.sin_iota, {xlabel: "sin ι", value: x => `sin ι ${sig(x, 2)}`});
  versus("p-b", C.b, {xlabel: "galactic latitude b [°]", xticks: [-90, -45, 0, 45, 90].map(t => [t, `${t}`]), value: x => `b ${sig(x, 2)}°`});
  const lows = W ? [...W.mass, ...W.mass_uncut].filter(v => v !== null) : [0], floor = Math.min(-1, Math.floor(Math.min(...lows)));
  perF0("p-mass", W && [{v: W.mass, color: "var(--c3)", name: "from the weights"}, {v: W.mass_uncut, color: "var(--ink-2)", name: "uncut draws", dash: "5 3", width: 1.4}],
    {y: [floor, 0.2], ylabel: `prior mass at SNR ≥ ${k}`, fmt: v => pct(10 ** v, 2),
      yticks: Array.from({length: -floor + 1}, (_, i) => floor + i).filter((t, i, a) => a.length < 8 || (t - floor) % 2 === 0).map(t => [t, decade(t)])});
  perF0("p-ess", W && [{v: W.ess_median, color: "var(--c1)", name: "median"}, {v: W.ess_q05, color: "var(--c4)", name: "5% quantile"}],
    {y: [0, 1], ylabel: "ESS / M", xlabel: "window centre [mHz]", xname: "centre", fmt: v => sig(v, 2), yticks: [0, 0.25, 0.5, 0.75, 1].map(t => [t, sig(t)]),
      marks: [{hline: 1 / PG.candidates, color: "var(--axis)", width: 1, dash: "2 3", text: `1 / M`, left: true}]});
  html("lg-cut", legendItem("var(--c1)", k ? `SNR ≥ ${k}` : "uncut") + legendItem("var(--ink-2)", "uncut draws", "5 3")
    + legendItem("var(--c3)", "prior mass from the weights") + legendItem("var(--c1)", "ESS / M median") + legendItem("var(--c4)", "ESS / M 5% quantile"));
  const rows = B().cuts.map((c, i) => c.row ? `<tr data-i="${i}"${i === K() ? ' class="on"' : ""}><td>SNR ≥ ${PG.ks[i]}</td><td>${pct(c.row.mass, 2)}</td><td>${pct(c.row.mass_uncut, 2)}</td>`
    + `<td>${sig(c.row.ess_median, 2)}</td><td>${sig(c.row.ess_q05, 2)}</td><td>${pct(c.row.windows_k, 1)}</td><td>${pct(c.row.visible, 1)}</td></tr>` : "").join("");
  const table = document.getElementById("cut-table");
  table.innerHTML = `<table><tr><th>cut</th><th>prior mass (weights)</th><th>(uncut draws)</th><th>ESS / M median</th><th>5% quantile</th><th>windows with SNR ≥ k</th><th>at SNR ≥ ${PG.visible_snr}</th></tr>${rows}</table>`;
  table.onclick = e => { const tr = e.target.closest("tr[data-i]"); if (!tr) return; S.k = String(PG.ks[+tr.dataset.i]); slider.value = K(); save(); schedule(); };
}

function segs() {
  for (const [key, opts] of Object.entries(OPTIONS)) {
    const box = document.getElementById(key);
    box.innerHTML = opts.map(([v, t]) => `<button type="button" data-v="${v}" aria-pressed="${S[key] === v}">${t}</button>`).join("");
    box.onclick = e => { const b = e.target.closest("button"); if (!b) return; S[key] = b.dataset.v; save(); schedule(); };
  }
}
function save() { history.replaceState(null, "", "#" + Object.entries(S).map(([k, v]) => `${k}=${v}`).join("&")); }
function load() {
  for (const part of location.hash.slice(1).split("&")) {
    const [k, v] = part.split("=");
    if (k in OPTIONS && OPTIONS[k].some(o => o[0] === v)) S[k] = v;
    if (k === "k" && PG.ks.includes(+v)) S.k = String(+v);
  }
}
function navLinks() {
  document.querySelectorAll(".nav a[data-page]").forEach(a => a.href = `${a.dataset.page}#band=${S.band}`);
}

function render() {
  for (const key of Object.keys(OPTIONS)) document.querySelectorAll(`#${key} button`).forEach(b => b.setAttribute("aria-pressed", b.dataset.v === S[key]));
  navLinks();
  html("k-label", kNow() ? `SNR ≥ ${kNow()}` : "none (catalogue prior)");
  html("tiles", D().tiles);
  cornerCard(); visibleCard(); flowCard(); cutCard();
}
let pending = 0;
const schedule = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; render(); }); };
addEventListener("resize", schedule);
addEventListener("hashchange", () => { load(); slider.value = K(); schedule(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
load();
segs();
// drag by pointer capture: VS Code's webview drops the native range drag (clicks and keys still go native)
function dragSlider(slider) {
  const at = ev => { const r = slider.getBoundingClientRect(); return Math.round(clamp((ev.clientX - r.left) / r.width, 0, 1) * +slider.max); };
  slider.addEventListener("pointerdown", ev => {
    if (ev.button !== 0) return;
    ev.preventDefault(); slider.focus(); slider.setPointerCapture(ev.pointerId);
    const move = e => { const v = at(e); if (v !== +slider.value) { slider.value = v; slider.dispatchEvent(new Event("input")); } };
    const up = () => {
      for (const [t, f] of [["pointermove", move], ["pointerup", up], ["pointercancel", up]]) slider.removeEventListener(t, f);
      slider.dispatchEvent(new Event("change"));
    };
    for (const [t, f] of [["pointermove", move], ["pointerup", up], ["pointercancel", up]]) slider.addEventListener(t, f);
    move(ev);
  });
}
const slider = document.getElementById("k");
slider.max = PG.ks.length - 1;
slider.value = K();
slider.addEventListener("input", () => { S.k = String(PG.ks[+slider.value]); schedule(); });
slider.addEventListener("change", save);
dragSlider(slider);
render();
</script>
</body>
</html>
"""


def sig(v: float | None, n: int = 3) -> str:
    """v to n significant figures, plain at or above 10**n."""
    if v is None or not math.isfinite(v):
        return "–"
    return f"{round(v):,}" if abs(v) >= 10**n else f"{v:.{n}g}"


def pct(v: float, d: int = 1) -> str:
    """A share as a percentage, in exponent form when it would round to zero."""
    if v == 0:
        return "0%"
    return f"{100 * v:.1e}%" if v < 10 ** -(d + 2) else f"{100 * v:.{d}f}%"


def mhz(log_f: float) -> str:
    return sig(10**log_f * 1e3)


def rounded(a, digits: int = 4) -> list:
    """An array as a list of floats to `digits` significant figures, for the page's JSON."""
    return [float(f"{v:.{digits}g}") for v in np.asarray(a, dtype=float).ravel()]


def smooth(a: np.ndarray) -> np.ndarray:
    """corner.html's one-cell Gaussian along a 1-D array, reflected at both ends."""
    return np.convolve(np.pad(a, 2, mode="reflect"), SMOOTH, mode="valid")


def smooth2(h: np.ndarray) -> np.ndarray:
    """corner.html's one-cell Gaussian along both axes of a 2-D histogram, zero beyond it."""
    for axis in (0, 1):
        h = np.apply_along_axis(np.convolve, axis, h, SMOOTH, mode="same")
    return h


def phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def catalogue(
    prior: dict, lo: float, hi: float, rng: np.random.Generator
) -> np.ndarray:
    """The catalogue's Gaussian with log10 f0 in [lo, hi], as sample_sources draws it."""
    draws = rng.multivariate_normal(prior["mean"], prior["cov"], CATALOGUE_DRAWS)
    keep = (draws[:, 0] >= lo) & (draws[:, 0] <= hi)
    return draws[keep]


def density2(x, y, rx, ry) -> tuple[np.ndarray, list]:
    """A smoothed 2-D histogram over CELLS², scaled to its peak, with the density levels holding MASSES of it."""
    h = smooth2(np.histogram2d(x, y, CELLS, [rx, ry])[0].T)
    ordered = np.sort(h.ravel())[::-1]
    held = np.cumsum(ordered) / ordered.sum()
    levels = [ordered[np.searchsorted(held, m)] / ordered[0] for m in MASSES]
    return h / ordered[0], levels


def histogram(a: np.ndarray, lo: float, hi: float, bins: int = BINS) -> tuple:
    """The smoothed density of a over [lo, hi], with the bin centres."""
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.histogram(a, edges)[0]
    return (edges[1:] + edges[:-1]) / 2, smooth(
        counts / (len(a) * (edges[1] - edges[0]))
    )


def flow_histogram(a: np.ndarray, lo: float, hi: float, reference=None) -> dict:
    x, drawn = histogram(a, lo, hi, 60)
    out = dict(x=rounded(x, 5), dx=float(x[1] - x[0]), drawn=rounded(drawn, 3))
    if reference is not None:
        out["reference"] = rounded(reference(x), 3)
    return out


def columns(z) -> np.ndarray:
    """Per source, (log10 f0, log10 fdot, log10 A)."""
    return np.stack([z["log_f0"].ravel(), z["log_fdot"].ravel(), z["log_a"].ravel()], 1)


def quantile_range(arrays: list, q: float = 5e-4) -> tuple[float, float]:
    """The widest q to 1 - q range over the arrays."""
    return min(float(np.quantile(a, q)) for a in arrays), max(
        float(np.quantile(a, 1 - q)) for a in arrays
    )


def binned(where: np.ndarray, values: np.ndarray, reduce) -> list:
    """reduce(values) per centre bin, None where a bin holds fewer than 5 or it is not finite."""
    out = []
    for b in range(CENTRE_BINS):
        with np.errstate(divide="ignore"):
            v = (
                float(reduce(values[where == b]))
                if (where == b).sum() >= 5
                else math.nan
            )
        out.append(round(v, 5) if math.isfinite(v) else None)
    return out


def band_shared(
    rec: dict, zs: list, prior: dict, rng: np.random.Generator
) -> tuple[dict, dict]:
    """What every cut of a band shares: the page's axes, the catalogue's densities and the uncut references."""
    lo, hi = (math.log10(f) for f in rec["problem"]["f0_range"])
    z0 = zs[0]
    cat = catalogue(prior, lo, hi, rng)
    cols0 = columns(z0)
    visible0 = z0["snr"].ravel() >= VISIBLE_SNR

    # ranges: the band in f0, else the 0.05-99.95% of the catalogue and every cut, widened to every uncut visible source
    ranges = [[lo, hi]]
    for c in (1, 2):
        a, b = quantile_range([cat[:, c]] + [columns(z)[:, c] for z in zs])
        if visible0.any():
            a, b = min(a, cols0[visible0, c].min()), max(b, cols0[visible0, c].max())
        pad = 0.04 * (b - a)
        ranges.append([float(a - pad), float(b + pad)])
    marginal = []
    for c in range(3):
        x, density = histogram(cat[:, c], *ranges[c])
        marginal.append(dict(x=rounded(x, 5), catalogue=rounded(density, 3)))
    pairs = {}
    for cx, cy in ((0, 1), (0, 2), (1, 2)):
        grid, levels = density2(cat[:, cx], cat[:, cy], ranges[cx], ranges[cy])
        pairs[f"{cx}{cy}"] = dict(
            catalogue=np.round(grid.ravel(), 3).tolist(),
            catalogue_levels=rounded(levels),
        )

    # the cut card's axes, fixed across cuts so the slider moves the data, not the frame
    log_snr = quantile_range([np.log10(z["snr"].ravel()) for z in zs])
    sd = math.sqrt(prior["cov"][0][0])
    mass = lambda a, b: phi((b - prior["mean"][0]) / sd) - phi(
        (a - prior["mean"][0]) / sd
    )
    axes = dict(
        lo=lo,
        hi=hi,
        in_band=mass(lo, hi) / mass(prior["cut"], math.inf),
        cat=cat,
        ranges=ranges,
        edges=np.linspace(lo, hi, CENTRE_BINS + 1),
        log_snr=(log_snr[0] - 0.1, log_snr[1] + 0.1),
    )
    C = rec["curves"]
    page = dict(
        range=[[float(a), float(b)] for a, b in ranges],
        marginal=marginal,
        pairs=pairs,
        curves={
            k: rounded(C[k], 6) for k in ("log_f0", "fdot_mean", "amp_mean", "amp_snr7")
        },
        edges=rounded(axes["edges"], 6),
    )
    return page, axes


def cut_page(
    rec: dict, z, z0, k: float, axes: dict, prior: dict, rng: np.random.Generator
) -> tuple[dict, dict]:
    """The page's data for one band at snr cut k (0: the catalogue prior), and the numbers its text quotes."""
    hi, ranges, edges = axes["hi"], axes["ranges"], axes["edges"]
    n_sources = rec["problem"]["n_sources"]
    cols = columns(z)
    snr = z["snr"].ravel()
    visible = snr >= VISIBLE_SNR

    marginal = []
    for c in range(3):
        m = dict(drawn=rounded(histogram(cols[:, c], *ranges[c])[1], 3))
        if visible.sum() >= 30:
            m["visible"] = rounded(histogram(cols[visible, c], *ranges[c])[1], 3)
        marginal.append(m)
    pairs = {}
    for cx, cy in ((0, 1), (0, 2), (1, 2)):
        drawn, drawn_levels = density2(cols[:, cx], cols[:, cy], ranges[cx], ranges[cy])
        pairs[f"{cx}{cy}"] = dict(
            drawn=np.round(drawn.ravel(), 3).tolist(),
            drawn_levels=rounded(drawn_levels),
        )
    shown = np.flatnonzero(visible)
    if shown.size > VISIBLE_POINTS:
        shown = np.sort(rng.choice(shown, VISIBLE_POINTS, replace=False))
    points = [rounded(np.append(cols[i], snr[i]), 5) for i in shown]

    # per window centre: the share of windows there, and of those with at least n visible sources
    where = np.clip(np.digitize(np.log10(z["f"]), edges) - 1, 0, CENTRE_BINS - 1)
    count = (z["snr"] >= VISIBLE_SNR).sum(1)
    in_bin = [where == b for b in range(CENTRE_BINS)]
    share_of = lambda mask, v: float(v[mask].mean()) if mask.any() else None
    at_least = [
        [share_of(m, count >= n) for m in in_bin] for n in range(1, n_sources + 1)
    ]
    centre = dict(
        share=[float(m.mean()) for m in in_bin],
        at_least=at_least,
        source=[share_of(m, (z["snr"] >= VISIBLE_SNR).mean(1)) for m in in_bin],
    )

    # the flow's z per coordinate against its N(0, 1) base
    gauss = lambda x: np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    zc = rearrange(z["z"], "w s c -> c (w s)")
    z_std, z_outside = zc.std(1), (np.abs(zc) > Z_RANGE).mean(1)
    flow = dict(
        coords=[
            flow_histogram(zc[c], -Z_RANGE, Z_RANGE, gauss)
            | dict(name=name, std=float(z_std[c]), outside=float(z_outside[c]))
            for c, name in enumerate(Z_NAMES)
        ]
    )

    # what the cut does: snr, sky latitude and inclination against the uncut draws, and its weights per f0
    snr0 = z0["snr"].ravel()
    snr_hist = flow_histogram(
        np.log10(snr),
        *axes["log_snr"],
        lambda x: histogram(np.log10(snr0), *axes["log_snr"], 60)[1],
    )
    iota_hist = flow_histogram(
        z["sin_iota"].ravel(),
        -1.0,
        1.0,
        lambda x: histogram(z0["sin_iota"].ravel(), -1.0, 1.0, 60)[1],
    )
    b_hist = flow_histogram(
        np.degrees(z["b"].ravel()),
        -90.0,
        90.0,
        lambda x: histogram(np.degrees(z0["b"].ravel()), -90.0, 90.0, 60)[1],
    )
    weights, row = None, None
    if k > 0:
        # the candidates are log-uniform: p(f0) reweights their P(snr >= k) to the catalogue's prior mass
        cand_log_f0, ess = z["cand_log_f0"].ravel(), z["ess"]
        p_cut = 10.0 ** z["log_mass"].ravel()
        p_f0 = np.exp(-0.5 * (cand_log_f0 - prior["mean"][0]) ** 2 / prior["cov"][0][0])
        where_cand = np.clip(np.digitize(cand_log_f0, edges) - 1, 0, CENTRE_BINS - 1)
        where_centre = np.clip(
            np.digitize(np.log10(z["f"][: ess.size]), edges) - 1, 0, CENTRE_BINS - 1
        )
        where0 = np.clip(
            np.digitize(z0["log_f0"].ravel(), edges) - 1, 0, CENTRE_BINS - 1
        )
        weights = dict(
            mass=binned(where_cand, p_cut, lambda v: np.log10(v.mean())),
            mass_uncut=binned(where0, snr0 >= k, lambda v: np.log10(v.mean())),
            ess_median=binned(where_centre, ess, np.median),
            ess_q05=binned(where_centre, ess, lambda v: np.quantile(v, 0.05)),
        )
        row = dict(
            mass=float((p_f0 * p_cut).sum() / p_f0.sum()),
            mass_uncut=float((snr0 >= k).mean()),
            ess_median=float(np.median(ess)),
            ess_q05=float(np.quantile(ess, 0.05)),
            windows_k=float((z["snr"] >= k).any(1).mean()),
            windows_k_uncut=float((z0["snr"] >= k).any(1).mean()),
            visible=float(visible.mean()),
        )

    num = dict(
        sources=int(snr.size),
        windows=int(count.size),
        visible=float(visible.mean()),
        visible_count=int(visible.sum()),
        snr_median=float(np.median(snr)),
        none=float((count == 0).mean()),
        all_four=float((count == n_sources).mean()),
        mean_count=float(count.mean()),
        drift_over=float((z["drift"] > 1).mean()),
        drift_median=float(np.median(z["drift"])),
        span_max=float(z["span"].max()),
        z_std=z_std.tolist(),
        z_outside=z_outside.tolist(),
        a_median=float(np.median(z["log_a_unit"])),
    ) | (row or {})
    tile = (
        lambda key, v, note: f'<div class="tile"><div class="k">{key}</div><div class="v">{v}</div><div class="d">{note}</div></div>'
    )
    head = (
        [
            tile(
                f"prior mass at SNR ≥ {sig(k)}",
                pct(row["mass"], 2),
                f"from the window weights; the catalogue draws put {pct(row['mass_uncut'], 2)} there",
            ),
            tile(
                "ESS / M of the window candidates",
                sig(row["ess_median"], 2),
                f"median; 5% of windows below {sig(row['ess_q05'], 2)}, out of M = {rec['cut']['candidates']}",
            ),
            tile(
                f"windows with a source at SNR ≥ {sig(k)}",
                pct(row["windows_k"], 1),
                f"the catalogue's windows {pct(row['windows_k_uncut'], 2)}",
            ),
        ]
        if k > 0
        else [
            tile(
                "catalogue in the band",
                pct(axes["in_band"], 1),
                f"of the catalogue above its {mhz(prior['cut'])} mHz cut; the band ends at {mhz(hi)} mHz",
            ),
            tile(
                f"windows with none at SNR ≥ {sig(VISIBLE_SNR)}",
                pct(num["none"], 2),
                f"{pct(num['all_four'], 2)} have all {n_sources}; {sig(num['mean_count'])} visible per window on average",
            ),
        ]
    )
    tiles = "".join(
        head
        + [
            tile(
                f"sources at SNR ≥ {sig(VISIBLE_SNR)}",
                pct(num["visible"], 2),
                f"median source SNR {sig(num['snr_median'])}",
            ),
            tile(
                "drift over a bin",
                pct(num["drift_over"], 2),
                f"ḟT² median {sig(num['drift_median'])} bins; widest response {sig(num['span_max'])} of {rec['problem']['response_points']} bins",
            ),
        ][(1 if k > 0 else 0) :]
    )
    page = dict(
        sources=num["sources"],
        marginal=marginal,
        pairs=pairs,
        visible=points,
        visible_count=num["visible_count"],
        centre=centre,
        flow=flow,
        cut=dict(snr=snr_hist, sin_iota=iota_hist, b=b_hist, weights=weights, row=row),
        tiles=tiles,
    )
    return page, num


def band_numbers(rec: dict, z0, axes: dict) -> dict:
    """The uncut band's numbers the text quotes beyond cut_page's."""
    edges, cat = axes["edges"], axes["cat"]
    C = rec["curves"]
    crossing = np.flatnonzero(np.asarray(C["amp_mean"]) >= np.asarray(C["amp_snr7"]))
    where = np.clip(np.digitize(np.log10(z0["f"]), edges) - 1, 0, CENTRE_BINS - 1)
    count = (z0["snr"] >= VISIBLE_SNR).sum(1)
    shares = [
        float((count[where == b] >= 1).mean()) if (where == b).any() else None
        for b in range(CENTRE_BINS)
    ]
    half = [i for i, v in enumerate(shares) if v is not None and v >= 0.5]
    shift = np.abs(columns(z0).mean(0) - cat.mean(0)) / cat.std(0)
    return dict(
        in_band=axes["in_band"],
        response_points=rec["problem"]["response_points"],
        crossing=float(C["log_f0"][crossing[0]]) if crossing.size else None,
        half=float(edges[half[0]]) if half else None,
        top_share=float(np.mean(where >= CENTRE_BINS // 2)),
        shift=float(shift.max()),
        f_top=10 ** axes["hi"],
    )


def fit_table(prior: dict) -> str:
    """The catalogue fit: the Gaussian's mean, std and correlations, and (fdot, A) given f0."""
    rows = "".join(
        f"<tr><td>log₁₀ {name}</td><td>{prior['mean'][i]:.3f}</td><td>{prior['std'][i]:.3f}</td>"
        + "".join(f"<td>{prior['corr'][i][j]:.3f}</td>" for j in range(3))
        + "</tr>"
        for i, name in enumerate(("f₀", "ḟ", "A"))
    )
    given = "".join(
        f"<tr><td>log₁₀ {name}</td><td>{prior['slope'][i]:.3f}</td><td>{prior['intercept'][i]:.2f}</td><td>{prior['residual_std'][i]:.3f}</td></tr>"
        for i, name in enumerate(("ḟ", "A"))
    )
    return (
        "<b>The catalogue's Gaussian</b> (parent, before the cut)"
        f"<table><tr><th></th><th>mean</th><th>std</th><th>corr f₀</th><th>corr ḟ</th><th>corr A</th></tr>{rows}</table>"
        "<b>Given log₁₀ f₀</b>"
        f"<table><tr><th></th><th>slope</th><th>intercept</th><th>residual std</th></tr>{given}</table>"
        f"The residuals correlate at r = {prior['residual_corr']:.3f}."
    )


def main() -> None:
    d = json.loads((DATA / "f0-fdot-amplitude.json").read_text())
    prior = d["prior"]
    rng = np.random.default_rng(0)
    bands, nums, cut_nums = {}, {}, {}
    for band in BANDS:
        rec = d["bands"][band]
        z0, cut = np.load(DATA / f"{band}.npz"), np.load(DATA / f"{band}-cut.npz")
        ks = [0.0] + cut["visible_snr"].tolist()
        zs = [z0] + [
            {key: cut[key][i] for key in cut.files if key != "visible_snr"}
            for i in range(len(ks) - 1)
        ]
        bands[band], axes = band_shared(rec, zs, prior, rng)
        cuts = [cut_page(rec, z, z0, k, axes, prior, rng) for k, z in zip(ks, zs)]
        bands[band]["cuts"] = [c[0] for c in cuts]
        nums[band] = cuts[0][1] | band_numbers(rec, z0, axes)
        cut_nums[band] = [c[1] for c in cuts[1:]]
    n, f = nums["narrow"], nums["full"]
    tk = d["bands"]["full"]["training_visible_snr"]
    n_sources = d["bands"]["full"]["problem"]["n_sources"]
    cut_rec = d["bands"]["full"]["cut"]
    page = dict(
        bands=bands,
        ks=ks,
        candidates=cut_rec["candidates"],
        cells=CELLS,
        visible_snr=VISIBLE_SNR,
        training_k=tk,
        n_sources=n_sources,
    )

    sub = (
        "(log₁₀ f₀, log₁₀ ḟ, log₁₀ A) is one Gaussian, fitted to the double white dwarfs of a galaxy catalogue. "
        f"The catalogue prior (k = 0) draws a window centre from its f₀ marginal (<code>sample_f</code>) and {n_sources} sources from the Gaussian inside the window. "
        f"This page shows that joint prior, which of its sources the data see, and what the flow is given. "
        "The cut slider's k > 0 centres windows by <code>sample_window</code> at <code>LisaGB(visible_snr=k)</code>, about the f₀ of a source at SNR ≥ k; the sources inside stay catalogue draws. "
        + (
            f"Training uses k = {sig(tk)}, the slider's default. "
            if tk
            else "Training uses the catalogue, k = 0. "
        )
        + f"Measured on {f['windows']:,} windows per band ({f['sources']:,} sources), {cut_rec['windows']:,} per cut. "
        f"Built {date.today().isoformat()} from <code>data/</code> (commit <code>{d['commit'][:7]}</code>{', src dirty' if d['src_dirty'] else ''})."
    )
    corner_note = (
        "The corner of the three log parameters. The diagonal holds each one's density; the cells below it hold each pair's, smoothed by a one-cell Gaussian "
        f"on a {CELLS} × {CELLS} grid, filled more opaque where denser and outlined where it holds 39% and 86% of the sources (1σ and 2σ in 2-D). "
        "The dashed contours are the catalogue's Gaussian truncated to the band and to the ḟ ceiling, from a million numpy draws: where they meet the drawn ones, training follows the catalogue. "
        f"The orange dots are the sources at SNR ≥ {sig(VISIBLE_SNR)}, with their own density on the diagonal; "
        "the orange line in (f₀, A) is the amplitude a source needs for SNR 7, averaged over sky and orientation: 7 · <code>amplitude_unit</code> / √⟨κ²⟩, with κ the SNR of one amplitude unit; at a cut, the green line is the same amplitude at SNR k."
    )
    cut_note = (
        f"At cut k, <code>sample_window</code> draws M = {cut_rec['candidates']} log-uniform window-centre candidates, each a source alone at the centre with one prior sky and orientation, "
        "and resamples them by p(f₀) · P(SNR ≥ k | f₀, sky, orientation), the latter Φ(−cut) with cut where SNR = k falls in log₁₀ A given f₀, in stds. "
        "The window's sources stay catalogue draws, so the cut moves windows, not sources. "
        "Top: source SNR, sin ι and galactic latitude, against the catalogue draws (dashed). "
        f"Bottom, over the first {cut_rec['weight_windows']:,} windows: per f₀ bin, P(SNR ≥ k) from the candidates' Φ(−cut) (solid) and the catalogue draws' share (dashed); "
        "per window-centre bin, the share of the M candidates the resample keeps (ESS / M). "
        "The table holds every cut of the band; click a row to show it."
    )
    visible_note = (
        f"Top: where window centres fall, in {CENTRE_BINS} log bins: at k = 0 the catalogue's f₀ marginal, so most windows sit low; at a cut, <code>sample_window</code>'s. "
        f"Bottom: per centre bin, the share of windows holding at least k sources at SNR ≥ {sig(VISIBLE_SNR)}, and (dashed) the share of single sources. "
        "A window is 4 response bands wide, so its centre and its sources' f₀ agree to a few percent."
    )
    flow_note = (
        "Per source in f₀ order, what the flow learns: z = (x − μ) / σ of each chart coordinate in the prior's frame (<code>chart.prior_moments</code>), "
        "the frame the network starts from before the data shift it, and its best answer where the data see nothing. "
        "The flow transports its N(0, 1) base (dashed) to z. z± is the helicity coordinate, asinh(A/unit · (1 ∓ sin ι)²/4) times its phase; hover for each z's std and its share beyond the axis."
    )
    meaning = [
        f"<b>The draws follow the catalogue.</b> The drawn means sit within {sig(n['shift'], 2)} (narrow-B) and {sig(f['shift'], 2)} (full-B) stds of the catalogue Gaussian truncated to the band, "
        f"and the contours overlap. {pct(n['in_band'], 1)} of the catalogue lies in narrow-B, below {sig(n['f_top'] * 1e3)} mHz.",
        f"<b>Almost every source is invisible.</b> {pct(n['visible'], 2)} of narrow-B sources and {pct(f['visible'], 2)} of full-B ones reach SNR {sig(VISIBLE_SNR)}; "
        f"{pct(n['none'], 2)} and {pct(f['none'], 2)} of windows hold none, so for them the flow's best answer is the prior."
        + (
            f" The conditional mean A reaches the SNR-7 amplitude from {mhz(f['crossing'])} mHz."
            if f["crossing"] is not None
            else " The conditional mean A stays below the SNR-7 amplitude across full-B: the visible sources are its loud tail."
        ),
        "<b>The visible sources sit at the top of the band, where the catalogue's windows rarely go.</b> "
        + (
            f"Half the full-B windows hold a visible source only from about {mhz(f['half'])} mHz, "
            if f["half"] is not None
            else "No full-B centre bin has half its windows holding a visible source, "
        )
        + f"while the upper half of the band in log f₀ gets {pct(f['top_share'], 2)} of the windows"
        + (
            f": hence training's k = {sig(tk)} window prior."
            if tk
            else ". If the posterior there turns out miscalibrated, a flatter window prior reweighted back to the catalogue is the fix to try."
        ),
        f"<b>ḟ matters only in full-B.</b> {pct(f['drift_over'], 2)} of full-B sources drift by more than a bin over the observation (narrow-B {pct(n['drift_over'], 2)}); "
        f"every response still fits: the widest spans {sig(f['span_max'])} of {f['response_points']} bins (narrow-B {sig(n['span_max'])} of {n['response_points']}).",
    ]

    at7 = ks.index(VISIBLE_SNR) - 1
    fc = cut_nums["full"]
    meaning.append(
        "<b>The cut moves windows to where sources are visible; it does not make them visible.</b> "
        f"At SNR ≥ {sig(VISIBLE_SNR)}, {pct(fc[at7]['windows_k'], 1)} of full-B windows hold a source that loud (catalogue {pct(fc[at7]['windows_k_uncut'], 2)}), "
        f"narrow-B {pct(cut_nums['narrow'][at7]['windows_k'], 1)} (catalogue {pct(cut_nums['narrow'][at7]['windows_k_uncut'], 2)}). "
        f"Full-B median ESS / M falls from {sig(fc[0]['ess_median'], 2)} at SNR ≥ {sig(ks[1])} to {sig(fc[-1]['ess_median'], 2)} at SNR ≥ {sig(ks[-1])} "
        f"(5% quantile {sig(fc[0]['ess_q05'], 2)} to {sig(fc[-1]['ess_q05'], 2)}). "
        f"The candidates' prior mass at SNR ≥ 7 is {pct(fc[at7]['mass'], 2)} against the uncut draws' {pct(fc[at7]['mass_uncut'], 2)} (full-B), "
        f"and narrow-B {pct(cut_nums['narrow'][at7]['mass'], 3)} against {pct(cut_nums['narrow'][at7]['mass_uncut'], 3)}."
    )
    zs7, zo7 = np.array(fc[at7]["z_std"]), np.array(fc[at7]["z_outside"])
    meaning.append(
        f"<b>What the flow is given is a unit normal, at every cut.</b> At SNR ≥ {sig(VISIBLE_SNR)} (full-B) all eight z have std {sig(zs7.min())} to {sig(zs7.max())}, "
        f"and {pct(zo7[4:].mean(), 2)} of z± lies beyond ±{sig(Z_RANGE)} (a unit normal: 6e-5%). "
        f"The cut moves the median source from 10<sup>{f['a_median']:.2f}</sup> to 10<sup>{fc[at7]['a_median']:.2f}</sup> amplitude units, and the prior's frame follows it."
    )

    out_html = PAGE
    for key, value in dict(
        SUB=sub,
        N_CORNER=corner_note,
        FIT=fit_table(prior),
        N_VISIBLE=visible_note,
        N_FLOW=flow_note,
        N_CUT=cut_note,
        MEANING="".join(f"<li>{t}</li>" for t in meaning),
        PAGE=json.dumps(page, separators=(",", ":")),
    ).items():
        out_html = out_html.replace(f"__{key}__", value)
    out = FOLDER / "f0-fdot-amplitude.html"
    out.write_text(out_html)
    for band, num in nums.items():
        print(
            band,
            {k: (round(v, 5) if isinstance(v, float) else v) for k, v in num.items()},
        )
    for band, rows in cut_nums.items():
        for k, row in zip(ks[1:], rows):
            print(
                band,
                f"snr >= {k:g}:",
                {
                    key: round(row[key], 5)
                    for key in (
                        "mass",
                        "mass_uncut",
                        "ess_median",
                        "ess_q05",
                        "windows_k",
                    )
                },
            )
    print(f"wrote {out} ({len(out_html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
