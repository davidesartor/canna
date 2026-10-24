"""Write sky.html from data/sky.json: LisaGB's galactic sky prior, per component, in three frames, against the chart.

Standard library only; run from the repo root:
    uv run --frozen python reports/prior/sky/scripts/build_page.py
"""

import json
import math
import sys
from datetime import date
from pathlib import Path

REPORT = Path(__file__).parents[1]
BANDS = ("narrow", "full")
# the retired Gaussian sky proxy's centre: the galactic centre in ecliptic coordinates, which jaxgb read as (RA, Dec)
OLD_PROXY_RADEC = (4.6574, -0.0966)
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior: sky</title>
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
  --q0: #fcfcfb; --q1: #cde2fb; --q2: #86b6ef; --q3: #2a78d6; --q4: #0d366b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --bar: #52514e;
    --q0: #1a1a19; --q1: #0d366b; --q2: #184f95; --q3: #3987e5; --q4: #cde2fb;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main > .card + .card { margin-top: 16px; }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
h3 { font-size: 14px; margin: 12px 0 4px; color: var(--ink-2); }
.nav { font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.nav a { color: var(--c1); text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.sub { color: var(--ink-2); margin: 0 0 24px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; }
.note a, .meaning a { color: var(--c1); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 0 0 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 22px; flex: none; }
.row { display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.row .k { color: var(--ink-2); font-size: 13px; }
.bar { position: sticky; top: 0; z-index: 5; background: var(--page); padding: 8px 0; margin-bottom: 8px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 12px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.card > .seg, .card > .row > .seg { margin: 0 0 4px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
svg text.lab { fill: var(--ink-2); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; white-space: nowrap; z-index: 2; }
.tip .row { color: var(--ink-2); font-variant-numeric: tabular-nums; margin: 0; gap: 6px; flex-wrap: nowrap; }
.sky { position: relative; }
.sky canvas { width: 100%; height: auto; display: block; }
.sky svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.pair > div { min-width: 0; }
@media (max-width: 640px) { .pair { grid-template-columns: 1fr; } }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
td.on, th.on { background: color-mix(in srgb, var(--c1) 10%, transparent); }
.meaning ul { margin: 8px 0 0; padding-left: 20px; }
.meaning li { margin: 0 0 8px; max-width: 80ch; }
#err { position: sticky; top: 0; z-index: 10; background: #c0392b; color: #fff; padding: 10px 16px; font-size: 14px; }
</style>
</head>
<body>
<div id="err" role="alert" hidden></div>
<main>
<p class="nav">The LisaGB prior by parameter group: <a data-page="../f0-fdot-amplitude/f0-fdot-amplitude.html" href="../f0-fdot-amplitude/f0-fdot-amplitude.html">f₀, ḟ, A</a> · <a data-page="../polarization/polarization.html" href="../polarization/polarization.html">polarization</a> · <b>sky</b> · <a data-page="../response-span/response-span.html" href="../response-span/response-span.html">response span</a> · <a data-page="../source-count/source-count.html" href="../source-count/source-count.html">source count</a></p>
<h1>The LisaGB prior: sky</h1>
<p class="sub">Where on the sky a source sits: the galactic mixture of <code>sample_sky</code> (a bulge Gaussian, a thick and a thin Laplace disk in (l, sin b), and an isotropic floor), each part on its own and summed, in the galactic, ecliptic and equatorial frames, against the quadrature of <code>sky_log_density</code>; then why the frame matters to LISA, whether the rotations are exact, and how (l, b) reach the flow. The sky does not depend on the band, so there is no band picker. __SUB__</p>
<div class="bar"><div class="row" style="margin:0"><span class="k">Frame</span><span class="seg" id="frame"></span><span class="k">Part</span><span class="seg" id="part"></span></div></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>Where on the sky does the prior put sources?</h2>
  <p class="note" id="n-sky"></p>
  <div class="sky" id="p-sky"><canvas width="960" height="480"></canvas><svg viewBox="0 0 960 480" role="img" aria-label="sky prior density"></svg></div><div class="tip"></div>
  <div class="legend" id="lg-sky" style="margin-top:6px"></div>
  <div id="ramp"></div>
</div>
<div class="card">
  <h2>What does each part of the mixture contribute?</h2>
  <p class="note" id="n-comp"></p>
  <div class="tablewrap" id="t-comp"></div>
</div>
<div class="card">
  <h2>Do the draws match the density in every frame?</h2>
  <div class="row"><span class="k">Draws</span><span class="seg" id="src"></span></div>
  <p class="note" id="n-marg"></p>
  <div class="legend" id="lg-marg"></div>
  <div class="pair"><div><div id="p-lat"></div><div class="tip"></div></div><div><div id="p-lon"></div><div class="tip"></div></div></div>
  <h3>χ² per degree of freedom, drawn against quadrature (bins expecting ≥ 5 draws; dof in brackets)</h3>
  <div class="tablewrap" id="t-chi"></div>
</div>
<div class="card">
  <h2>How far from the galactic centre do sources sit?</h2>
  <p class="note" id="n-dist"></p>
  <div class="legend" id="lg-dist"></div>
  <div id="p-dist"></div><div class="tip"></div>
  <div class="tablewrap" id="t-dist"></div>
</div>
<div class="card">
  <h2>Why does the frame matter to LISA?</h2>
  <p class="note" id="n-lisa"></p>
  <div class="legend" id="lg-lisa"></div>
  <div id="p-lisa"></div><div class="tip"></div>
  <div class="tablewrap" id="t-lisa"></div>
</div>
<div class="card">
  <h2>Are the frame rotations and the mixture exact?</h2>
  <p class="note" id="n-rot"></p>
  <div class="pair"><div class="tablewrap" id="t-rot"></div><div class="tablewrap" id="t-mix"></div></div>
</div>
<div class="card">
  <h2>What do l and b look like in the flow's coordinates?</h2>
  <p class="note" id="n-flow"></p>
  <div class="legend" id="lg-flow"></div>
  <div id="p-flow"></div><div class="tip"></div>
  <h3>How much mass sits near the wrap at l = ±π?</h3>
  <p class="note" id="n-wrap"></p>
  <div class="tablewrap" id="t-wrap"></div>
  <h3>Is every source's sky the same, and is it independent of the other coordinates?</h3>
  <p class="note" id="n-slots"></p>
  <div class="pair"><div class="tablewrap" id="t-slots"></div><div class="tablewrap" id="t-corr"></div></div>
</div>
<div class="card meaning">
  <h2>What this means</h2>
  <ul id="meaning"></ul>
</div>
</main>
<script>
const PG = __PAGE__;
const DEG = Math.PI / 180, TAU = 2 * Math.PI, deg = v => v / DEG;
const FRAMES = [["galactic", "galactic (l, b)", "l", "b"], ["ecliptic", "ecliptic (λ, β)", "λ", "β"], ["equatorial", "equatorial (RA, Dec)", "RA", "Dec"]];
const NAMES = PG.sky.components.names;
const PARTS = [["sum", "sum", "var(--ink)"], ["bulge", NAMES[0], "var(--c2)"], ["thick", NAMES[1], "var(--c1)"], ["thin", NAMES[2], "var(--c3)"], ["floor", NAMES[3], "var(--c4)"]];
const CHOICES = {frame: FRAMES.map(f => f[0]), part: PARTS.map(p => p[0]), src: ["sky", "windows"], band: ["narrow", "full"]};
const S = {frame: "galactic", part: "sum", src: "sky", band: "full"};
const partName = id => PARTS.find(p => p[0] === id)[1];
const partColor = id => PARTS.find(p => p[0] === id)[2];
const partIndex = id => PARTS.findIndex(p => p[0] === id) - 1;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sum = a => a.reduce((x, y) => x + y, 0);
const sup = e => String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const W1 = PG.bands.full, SD = PG.sky_draws, CH = PG.chart, LI = PG.lisa, RT = PG.checks.rotation;

const marks = {};
function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; marks[key](); schedule(); }));
  marks[key]();
}
seg("frame", FRAMES.map(([k, n]) => [k, n.replace(/ \(.*/, "")]), "frame");
seg("part", PARTS.map(([k, n]) => [k, n]), "part");
seg("src", [["sky", `2²⁰ sky draws, per part`], ["windows", `${W1.sources.toLocaleString()} training-window sources`]], "src");

// state in the address bar: #frame=galactic&part=sum&src=sky (and band, carried to the sibling pages)
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  for (const [k, vs] of Object.entries(CHOICES)) if (vs.includes(q.get(k))) S[k] = q.get(k);
  Object.values(marks).forEach(m => m());
}
function writeHash() {
  history.replaceState(null, "", "#" + new URLSearchParams(Object.entries(S).filter(([, v]) => v)).toString());
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
const table = (head, rows, on) => `<table><thead><tr>${head.map((h, j) => `<th${on?.(j) ? ' class="on"' : ""}>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c, j) => `<td${on?.(j) ? ' class="on"' : ""}>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

// generic plots: layers in data units
function niceTicks(lo, hi, n = 4) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag;
  const step = (r >= 5 ? 10 : r >= 2 ? 5 : r >= 1 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(+v.toPrecision(12));
  return out;
}
const logLabel = v => { const x = 10 ** v, e = Math.floor(v + 1e-9), m = Math.round(x / 10 ** e); return x >= 1e-2 && x < 1e4 ? String(+x.toPrecision(2)) : `${m > 1 ? m + "×" : ""}10${sup(e)}`; };
const centres = e => e.slice(1).map((v, i) => (v + e[i]) / 2);
const nearest = (xs, v) => xs.reduce((best, x, i) => Math.abs(x - v) < Math.abs(xs[best] - v) ? i : best, 0);
function chart(host, o) {
  const W = Math.max(260, o.W ?? host.clientWidth), H = o.H ?? 280, m = {l: 54, r: 16, t: 18, b: 38};
  const [x0, x1] = o.x, [y0, y1] = o.y;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const pts = arr => arr.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label}">`;
  niceTicks(y0, y1, o.yn ?? 4).forEach(t => s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${sig(t, 3)}</text>`);
  niceTicks(x0, x1, o.xn ?? 5).forEach(t => s += `<line x1="${X(t)}" x2="${X(t)}" y1="${H - m.b}" y2="${H - m.b + 4}" stroke="var(--axis)"/><text x="${X(t)}" y="${H - m.b + 17}" text-anchor="middle">${sig(t, 3)}</text>`);
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 3}" text-anchor="middle">${o.xlabel}</text><text x="${m.l}" y="${m.t - 6}">${o.ylabel}</text>`;
  for (const L of o.layers) {
    if (L.bars) L.bars.v.forEach((v, i) => { if (v > 0) { const a = X(L.bars.e[i]) + 1, b = X(L.bars.e[i + 1]) - 1; s += `<rect x="${a.toFixed(1)}" y="${Y(Math.min(v, y1)).toFixed(1)}" width="${Math.max(b - a, 0.5).toFixed(1)}" height="${(Y(Math.max(y0, 0)) - Y(Math.min(v, y1))).toFixed(1)}" fill="${L.color}" rx="1"/>`; } });
    if (L.line && L.width !== 0) s += `<polyline points="${pts(L.line)}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 2}" stroke-linejoin="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.marks) L.line.forEach(([x, y]) => s += `<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="4" fill="${L.color}" stroke="var(--surface)" stroke-width="1.5"/>`);
  }
  s += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" visibility="hidden"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  host.insertAdjacentHTML("beforeend", s + "</svg>");
  const svg = host.lastElementChild, cross = svg.querySelector(".cross"), hit = svg.querySelector(".hit");
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    const out = o.hover({x: x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0)});
    cross.setAttribute("visibility", out?.cross !== undefined ? "visible" : "hidden");
    if (!out) return untip(host);
    cross.setAttribute("x1", X(out.cross)); cross.setAttribute("x2", X(out.cross));
    tip(host, ev, out.html);
  });
  hit.addEventListener("pointerleave", () => { untip(host); cross.setAttribute("visibility", "hidden"); });
}
const density = h => { const w = h.edges[1] - h.edges[0], n = sum(h.counts) + (h.outside ?? 0); return {w, n, v: h.counts.map(c => c / n / w)}; };

// Mollweide: plot longitude λ in [-π, π] (positive to the right) and latitude to the ellipse |x| ≤ 2√2, |y| ≤ √2
function mollweide(lambda, phi) {
  let t = phi;
  if (Math.abs(phi) < Math.PI / 2 - 1e-9) for (let k = 0; k < 30; k++) {
    const d = (2 * t + Math.sin(2 * t) - Math.PI * Math.sin(phi)) / (2 + 2 * Math.cos(2 * t));
    t -= d; if (Math.abs(d) < 1e-10) break;
  }
  return [2 * Math.SQRT2 / Math.PI * lambda * Math.cos(t), Math.SQRT2 * Math.sin(t)];
}
const MAP = PG.sky.map, SW = 960, SH = 480;
const toPx = ([x, y]) => [SW / 2 + x / (2 * Math.SQRT2) * (SW / 2 - 2), SH / 2 - y / Math.SQRT2 * (SH / 2 - 2)];
const wrap = v => ((v + Math.PI) % TAU + TAU) % TAU - Math.PI;
// byte 0: below the scale (under the isotropic floor's density); 1..255: log10_lo..log10_hi
const decode = b64 => Uint8Array.from(atob(b64), ch => ch.charCodeAt(0));
const FLOOR_Q = 1 + Math.round(clamp((MAP.floor_log10 - MAP.log10_lo) / (MAP.log10_hi - MAP.log10_lo), 0, 1) * 254);
const GRIDS = Object.fromEntries(Object.entries(PG.sky.frames).map(([k, f]) => [k, {
  sum: decode(f.map), bulge: decode(f.component_maps[NAMES[0]]), thick: decode(f.component_maps[NAMES[1]]), thin: decode(f.component_maps[NAMES[2]]),
  floor: new Uint8Array(MAP.width * MAP.height).fill(FLOOR_Q)}]));
const log10At = q => q ? MAP.log10_lo + (q - 1) / 254 * (MAP.log10_hi - MAP.log10_lo) : -Infinity;
function rampLut() {
  const stops = [0, 1, 2, 3, 4].map(i => { const h = css(`--q${i}`); return [1, 3, 5].map(j => parseInt(h.slice(j, j + 2), 16)); });
  const lut = new Uint8ClampedArray(256 * 3);
  for (let q = 0; q < 256; q++) {
    const u = Math.max(0, q - 1) / 254 * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(u)), w = u - i;
    for (let c = 0; c < 3; c++) lut[3 * q + c] = stops[i][c] * (1 - w) + stops[i + 1][c] * w;
  }
  return lut;
}
// inverse projection of a pixel to plot (λ, φ) and the map cell under it
function cellAt(px, py) {
  const x = (px - SW / 2) / (SW / 2 - 2) * 2 * Math.SQRT2, y = -(py - SH / 2) / (SH / 2 - 2) * Math.SQRT2;
  if (x * x / 8 + y * y / 2 > 1) return null;
  const t = Math.asin(clamp(y / Math.SQRT2, -1, 1));
  const phi = Math.asin(clamp((2 * t + Math.sin(2 * t)) / Math.PI, -1, 1));
  const lambda = Math.PI * x / (2 * Math.SQRT2 * Math.cos(t));
  if (Math.abs(lambda) > Math.PI) return null;
  const col = Math.min(MAP.width - 1, Math.floor((lambda + Math.PI) / TAU * MAP.width));
  const rw = Math.min(MAP.height - 1, Math.floor((Math.PI / 2 - phi) / Math.PI * MAP.height));
  return {lambda, phi, k: rw * MAP.width + col};
}
const skyImages = {};
function skyPixels(frame, part) {
  const key = frame + part + css("--q4");
  if (skyImages[key]) return skyImages[key];
  const img = new ImageData(SW, SH), lut = rampLut(), grid = GRIDS[frame][part];
  for (let py = 0; py < SH; py++) for (let px = 0; px < SW; px++) {
    const c = cellAt(px + 0.5, py + 0.5); if (!c) continue;
    const q = grid[c.k], k = 4 * (py * SW + px);
    img.data[k] = lut[3 * q]; img.data[k + 1] = lut[3 * q + 1]; img.data[k + 2] = lut[3 * q + 2]; img.data[k + 3] = 255;
  }
  return skyImages[key] = img;
}

const partKey = id => id === "sum" ? "sum" : partName(id);

function tiles() {
  const P = W1.sky.abs_lat_below_10deg, W = W1.sky.within_rad_of_centre, fl = W1.flow, wr = CH.near_wrap["0.1"];
  document.getElementById("tiles").innerHTML = [
    ["|b| < 10°, galactic", pct(P.galactic), "the plane holds the draws"],
    ["|β| < 10°, ecliptic", pct(P.ecliptic), `the plane crosses the ecliptic at ${LI.galactic_to_ecliptic_tilt_deg.toFixed(0)}°`],
    ["|Dec| < 10°, equatorial", pct(P.equatorial), "jaxgb's frame"],
    ["within 0.5 rad of the centre", pct(W["0.5"], 0), `${pct(W["0.1"], 0)} within 0.1 rad`],
    ["flow std of l / b", fl.map(f => f.std.toFixed(3)).join(" / "), "galactic (l, b) over GALACTIC_SCALE"],
    ["within 0.1 rad of l = ±π", pct(wr.draws, 2), `where the flow's l jumps by ${(2 * CH.wrap_flow[0]).toFixed(1)}`],
    ["frame round trip", `${sig(RT.round_trip_rad.equatorial_max, 2)} rad`, `centre ${sig(RT.centre_err_arcsec, 2)}″ from Hipparcos`],
  ].map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}

function skyCard() {
  const [frame, fname] = FRAMES.find(f => f[0] === S.frame), F = PG.sky.frames[frame], grids = GRIDS[frame];
  const host = document.getElementById("p-sky"), svg = host.querySelector("svg");
  host.querySelector("canvas").getContext("2d").putImageData(skyPixels(frame, S.part), 0, 0);
  // graticule every 60° in longitude and 30° in latitude, longitude increasing to the left
  const plotLon = lon => wrap(F.lon0 - lon), shownLon = l => { const v = ((l % 360) + 360) % 360; return frame === "galactic" && v > 180 ? v - 360 : v; };
  const path = ll => "M" + ll.map(([l, p]) => toPx(mollweide(l, p)).map(v => v.toFixed(1)).join(",")).join("L");
  let s = "";
  for (let k = -2; k <= 2; k++) { const p = []; for (let i = 0; i <= 90; i++) p.push([k * Math.PI / 3 - 1e-9 * Math.sign(k), -Math.PI / 2 + i * Math.PI / 90]); s += `<path d="${path(p)}" fill="none" stroke="var(--axis)" stroke-opacity="0.5"/>`; }
  for (let k = -2; k <= 2; k++) { const p = []; for (let i = 0; i <= 120; i++) p.push([-Math.PI + i * Math.PI / 60, k * Math.PI / 6]); s += `<path d="${path(p)}" fill="none" stroke="var(--axis)" stroke-opacity="0.5"/>`; }
  s += `<ellipse cx="${SW / 2}" cy="${SH / 2}" rx="${SW / 2 - 2}" ry="${SH / 2 - 2}" fill="none" stroke="var(--axis)"/>`;
  // the other frames' equators, split where they cross the map's edge
  const PLANES = {galactic: ["galactic plane", "var(--c2)"], ecliptic: ["ecliptic (LISA's orbit plane)", "var(--c4)"], equatorial: ["celestial equator", "var(--c3)"]};
  for (const [other, pts] of Object.entries(F.planes)) {
    const runs = [[]];
    pts.forEach(([lon, lat], i) => { const x = plotLon(lon); if (i && Math.abs(x - plotLon(pts[i - 1][0])) > Math.PI) runs.push([]); runs[runs.length - 1].push([x, lat]); });
    runs.filter(r => r.length > 1).forEach(r => s += `<path d="${path(r)}" fill="none" stroke="var(--surface)" stroke-width="4"/><path d="${path(r)}" fill="none" stroke="${PLANES[other][1]}" stroke-width="2" stroke-dasharray="6 4"/>`);
  }
  for (let k = -2; k <= 2; k++) { const [x, y] = toPx(mollweide(k * Math.PI / 3, 0)); s += `<text class="lab" x="${x}" y="${y + 16}" text-anchor="middle">${shownLon(deg(F.lon0) - k * 60).toFixed(0)}°</text>`; }
  [-60, -30, 30, 60].forEach(b => { const [x, y] = toPx(mollweide(-Math.PI, b * DEG)); s += `<text x="${x - 4}" y="${y + 4}" text-anchor="end">${b}°</text>`; });
  const marker = (lon, lat, text, below) => {
    const [cx, cy] = toPx(mollweide(plotLon(lon), lat));
    return `<path d="M${cx - 9},${cy}H${cx + 9}M${cx},${cy - 9}V${cy + 9}" stroke="var(--surface)" stroke-width="5"/><path d="M${cx - 9},${cy}H${cx + 9}M${cx},${cy - 9}V${cy + 9}" stroke="var(--ink)" stroke-width="2"/><text class="lab" x="${cx + 12}" y="${cy + (below ? 18 : -8)}">${text}</text>`;
  };
  s += marker(F.centre[0], F.centre[1], "galactic centre", frame === "equatorial");
  if (frame === "equatorial") s += marker(PG.old_proxy[0], PG.old_proxy[1], "old Gaussian proxy: the ecliptic centre read as (RA, Dec)", false);
  s += `<rect class="hit" width="${SW}" height="${SH}" fill="transparent"/>`;
  svg.innerHTML = s;
  const hit = svg.querySelector(".hit");
  hit.onpointermove = ev => {
    const r = svg.getBoundingClientRect(), c = cellAt((ev.clientX - r.left) / r.width * SW, (ev.clientY - r.top) / r.height * SH);
    if (!c) return untip(host);
    const total = 10 ** log10At(grids.sum[c.k]);
    const rows = PARTS.slice(1).map(([id, name, color]) => { const q = grids[id][c.k]; return trow(color, `${name}: ${q ? pct(10 ** log10At(q) / total, 1) : "&lt; " + pct(10 ** MAP.log10_lo / total, 1)}`); });
    tip(host, ev, `<b>(${shownLon(deg(F.lon0 - c.lambda)).toFixed(1)}°, ${deg(c.phi).toFixed(1)}°)</b>` + trow(null, `p<sub>Ω</sub> ≈ ${sig(total, 2)} sr⁻¹, of which`) + rows.join(""));
  };
  hit.onpointerleave = () => untip(host);
  html("lg-sky", Object.entries(PLANES).filter(([k]) => k !== frame).map(([, [n, c]]) => legendItem(c, n, "6 4")).join("") + `<span>✛ galactic centre</span>`);
  // the ramp's scale
  const lo = MAP.log10_lo, hi = MAP.log10_hi, RW = 300, RX = v => 8 + (v - lo) / (hi - lo) * (RW - 16);
  let k = `<svg width="${RW}" height="30" viewBox="0 0 ${RW} 30" style="width:${RW}px;max-width:100%;display:inline-block;vertical-align:middle"><defs><linearGradient id="ramp-fill">` + [0, 1, 2, 3, 4].map(i => `<stop offset="${i * 25}%" style="stop-color:var(--q${i})"/>`).join("") + `</linearGradient></defs><rect x="8" y="2" width="${RW - 16}" height="10" rx="2" fill="url(#ramp-fill)" stroke="var(--border)"/>`;
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) k += `<line x1="${RX(e)}" x2="${RX(e)}" y1="12" y2="16" stroke="var(--axis)"/><text x="${RX(e)}" y="27" text-anchor="middle">${logLabel(e)}</text>`;
  html("ramp", k + `</svg> <span class="note">p<sub>Ω</sub> per steradian, log scale; the bottom of the scale is the isotropic floor's ${sig(10 ** MAP.floor_log10, 2)} sr⁻¹, and the blank sky is below it</span>`);
  const c = PG.sky.components, latName = FRAMES.find(f => f[0] === frame)[3], P = SD.abs_lat_below_10deg[frame][partKey(S.part)];
  const what = S.part === "sum" ? `Density per steradian of the whole prior (${c.names.map((n, i) => `${n} ${pct(c.weights[i])}`).join(", ")})` : `The ${partName(S.part)}'s weighted density w·p per steradian (weight ${pct(c.weights[partIndex(S.part)])}), on the sum's colour scale`;
  html("n-sky", `${what} in the ${fname} frame, Mollweide, longitude increasing to the left. ${pct(P.quadrature)} of ${S.part === "sum" ? "the prior" : "this part"} has |${latName}| &lt; 10° here. Hover for each part's share of the density.${frame === "equatorial" ? " jaxgb reads the sky in this frame; the retired Gaussian proxy sat at the galactic centre's ecliptic coordinates read as (RA, Dec), ${PG.old_proxy_offset_deg.toFixed(1)}° from the true centre." : frame === "ecliptic" ? ` LISA's orbit lies in this frame's plane; the galactic plane crosses it at ${LI.galactic_to_ecliptic_tilt_deg.toFixed(1)}°.` : ""}`);
}

function compCard() {
  const c = PG.sky.components, n = SD.n, on = j => j > 0 && PARTS[j - 1][0] === S.part;
  const scale = s => s ? `${sig(s[0], 3)} rad, ${sig(s[1], 3)}` : "–";
  const P = CH.parts, Wd = SD.within_rad_of_centre;
  const rows = [
    ["weight w", ...PARTS.map(([id], j) => j ? pct(c.weights[j - 1], 2) : "1")],
    ["drawn share (2²⁰ draws)", ...PARTS.map(([id], j) => j ? pct(SD.component_counts[j - 1] / n, 2) : pct(1, 0))],
    ["scales (l, sin b)", "–", ...c.scales.map(scale)],
    ...FRAMES.map(([f, , , lat]) => [`|${lat}| &lt; 10°`, ...PARTS.map(([id]) => pct(SD.abs_lat_below_10deg[f][partKey(id)].quadrature))]),
    ...["0.1", "0.5"].map(r => [`within ${r} rad of the centre`, ...PARTS.map(([id]) => pct(Wd[partKey(id)][r].quadrature))]),
    ["flow std of l / b", ...PARTS.map(([id]) => P[partKey(id)].flow_std.map(v => sig(v, 3)).join(" / "))],
    ["share of l² / b²", ...PARTS.map(([id]) => P[partKey(id)].variance_share.map(v => pct(v, 1)).join(" / "))],
  ];
  html("t-comp", table(["", ...PARTS.map(p => p[1])], rows, on));
  const b = P.bulge, fl = P[NAMES[3]];
  html("n-comp", `Each part of <code>SKY_WEIGHTS</code> on its own, fitted to a ${c.catalogue_sources.toLocaleString()}-source WD–WD catalogue (the catalogue is not in the repo, so the fit itself is not re-checked here). Shares near the plane and the centre are quadratures of each part's density; the flow columns come from 2²⁰ draws split by the component <code>sample_sky</code> picked. The bulge is ${sig(b.flow_std[0], 2)} flow units wide in l. The isotropic floor holds ${pct(c.weights[3], 2)} of the sources but ${pct(fl.variance_share[1], 0)} of b's second moment (the thick disk ${pct(P[NAMES[1]].variance_share[1], 0)}), so the two set the scale <code>GALACTIC_SCALE</code> divides b by, and the flow's b is far wider than the thin disk the most sources sit in.`);
}

function margCard() {
  const [frame, , lonName, latName] = FRAMES.find(f => f[0] === S.frame);
  const windows = S.src === "windows", M = windows ? W1.sky.marginals[frame] : SD.marginals[frame][partKey(S.part)];
  const color = windows ? "var(--bar)" : (S.part === "sum" ? "var(--bar)" : partColor(S.part));
  [["lat", latName], ["lon", lonName]].forEach(([k, name]) => {
    const host = document.getElementById(`p-${k}`), h = M[k], e = h.edges, x = centres(e);
    host.innerHTML = "";
    chart(host, {H: 210, x: [e[0], e[e.length - 1]], y: [0, Math.max(...h.counts, ...h.reference) * 1.1], xn: 4, yn: 3, xlabel: `${name} [deg]`, ylabel: "sources", label: `sky ${k} marginal`,
      layers: [{bars: {e, v: h.counts}, color}, {line: x.map((v, i) => [v, h.reference[i]]), color: "var(--ink-2)", width: 1.5, dash: "4 3"}],
      hover: ({x: v}) => { const i = nearest(x, v); return {cross: x[i], html: `<b>${name} ${sig(e[i])}° – ${sig(e[i + 1])}°</b>` + trow(color, `${h.counts[i].toLocaleString()} drawn, quadrature ${sig(h.reference[i], 4)}`)}; }});
  });
  const out = M.lat.outside, total = sum(M.lat.counts) + out, X = PG.chi2;
  html("lg-marg", legendBox(color, `${total.toLocaleString()} drawn sources${windows ? "" : `, ${partName(S.part)}`}`) + legendItem("var(--ink-2)", windows || S.part === "sum" ? "quadrature of <code>sky_log_density</code>" : `quadrature of the ${partName(S.part)}'s density`, "4 3"));
  const chi = windows ? X.windows[frame] : X.sky[partKey(S.part)][frame];
  html("n-marg", `The drawn sources' latitude and longitude in the ${frame} frame against the quadrature marginals. ${windows ? `These are the ${W1.windows.toLocaleString()} full-B training windows' sources (the old page's draws), the sum only${S.part !== "sum" ? "; pick 2²⁰ sky draws to see one part" : ""}.` : `These are 2²⁰ draws of <code>sample_sky</code>, ${S.part === "sum" ? "all of them" : `the ones it drew from the ${partName(S.part)}`}.`} χ²/dof is ${chi.lat.v.toFixed(2)} (${latName}, ${chi.lat.dof} dof) and ${chi.lon.v.toFixed(2)} (${lonName}, ${chi.lon.dof} dof).${out ? ` The galactic latitude histogram spans ±30°; ${out.toLocaleString()} draws (${pct(out / total)}) lie beyond.` : ""}`);
  const cols = FRAMES.flatMap(([f, , lo, la]) => [[f, "lat", `${la}`], [f, "lon", `${lo}`]]);
  const row = (label, o) => [label, ...cols.map(([f, k]) => `${o[f][k].v.toFixed(2)} (${o[f][k].dof})`)];
  const rows = [row(`training windows (${W1.sources.toLocaleString()}), sum`, X.windows), ...PARTS.map(([id, name]) => row(`2²⁰ draws, ${name}`, X.sky[partKey(id)]))];
  html("t-chi", table(["", ...cols.map(([f, , n]) => `${f.slice(0, 3)}. ${n}`)], rows, j => j > 0 && cols[j - 1][0] === frame));
}

function distCard() {
  const host = document.getElementById("p-dist"), D = SD.from_centre, e = D.edges, d = D[partKey(S.part)], color = partColor(S.part);
  host.innerHTML = "";
  const rec = S.part === "sum" ? Object.entries(W1.sky.within_rad_of_centre).map(([r, v]) => [+r, v]) : Object.entries(SD.within_rad_of_centre[partKey(S.part)]).map(([r, v]) => [+r, v.draws]);
  chart(host, {x: [0, Math.PI], y: [0, 1], xlabel: "angle from the galactic centre [rad]", ylabel: "share of sources within", label: "distance from the galactic centre", H: 260,
    layers: [{line: e.map((v, i) => [v, d.draws[i]]), color, width: 2}, {line: e.map((v, i) => [v, d.quadrature[i]]), color: "var(--ink-2)", width: 1.5, dash: "4 3"}, {line: rec, color: "var(--c2)", width: 0, marks: true}],
    hover: ({x}) => { const i = nearest(e, x); return {cross: e[i], html: `<b>${e[i].toFixed(3)} rad (${deg(e[i]).toFixed(1)}°)</b>` + trow(color, `${pct(d.draws[i])} of the draws within`) + trow("var(--ink-2)", `${pct(d.quadrature[i])} by quadrature`)}; }});
  html("lg-dist", legendItem(color, `2²⁰ draws, ${partName(S.part)}`) + legendItem("var(--ink-2)", "quadrature", "4 3") + legendDot("var(--c2)", S.part === "sum" ? `the ${W1.sources.toLocaleString()} training-window sources, at 0.1, 0.2, 0.3 and 0.5 rad` : "the part's draws at 0.1, 0.2, 0.3 and 0.5 rad"));
  const W = W1.sky.within_rad_of_centre;
  html("n-dist", `The cumulative share of sources within an angle of the galactic centre; the largest gap between draws and quadrature (KS) is ${sig(d.ks, 2)} for the ${partName(S.part)}. The bulge puts ${pct(W["0.1"], 0)} of all sources within 0.1 rad, but the disks spread the rest along the plane: only ${pct(W["0.5"], 0)} lie within 0.5 rad, so a quiet source's sky posterior is mostly prior.`);
  const radii = Object.keys(W);
  html("t-dist", table(["within", ...radii.map(r => `${r} rad`)], [["training windows, sum", ...radii.map(r => pct(W[r]))], ...PARTS.map(([id, name]) => [`quadrature, ${name}`, ...radii.map(r => pct(SD.within_rad_of_centre[partKey(id)][r].quadrature))])], null));
}

function lisaCard() {
  const host = document.getElementById("p-lisa"), A = LI.abs_beta[partKey(S.part)], h = A.hist, e = h.edges, x = centres(e), dd = density(h), color = S.part === "sum" ? "var(--bar)" : partColor(S.part);
  const iso = x.map(v => [v, Math.cos(v * DEG) * DEG]);
  host.innerHTML = "";
  chart(host, {x: [0, 90], y: [0, Math.max(...dd.v, ...iso.map(p => p[1])) * 1.1], xn: 6, yn: 3, xlabel: "|β|, ecliptic latitude [deg]", ylabel: "density per degree", label: "ecliptic latitude", H: 230,
    layers: [{bars: {e, v: dd.v}, color}, {line: iso, color: "var(--ink-2)", width: 1.5, dash: "4 3"}],
    hover: ({x: v}) => { const i = nearest(x, v); return {cross: x[i], html: `<b>|β| ${e[i]}° – ${e[i + 1]}°</b>` + trow(color, `${pct(h.counts[i] / dd.n, 2)} of the ${partName(S.part)}`) + trow("var(--ink-2)", `${pct(Math.sin(e[i + 1] * DEG) - Math.sin(e[i] * DEG), 2)} if isotropic`)}; }});
  html("lg-lisa", legendBox(color, `2²⁰ draws, ${partName(S.part)}`) + legendItem("var(--ink-2)", "isotropic sky, cos β", "4 3"));
  const I = LI.isotropic, c = LI.centre_ecliptic_deg;
  html("n-lisa", `LISA's constellation orbits the Sun in the ecliptic, so the ecliptic (λ, β) is the frame its response is simple in. The yearly Doppler shift of a source's phase has amplitude 2π f<sub>0</sub> (1 AU / c) cos β, largest in the ecliptic and gone at its poles, and its phase in the year is set by λ; the antenna patterns of the rotating constellation also change with (λ, β) over the year. So β sets how well the orbit can localise a source and, with the inclination, how loud it is: the SNR side of this lives on <a href="../f0-fdot-amplitude/f0-fdot-amplitude.html">the f₀, ḟ, A page</a>. The galactic plane is tilted by ${LI.galactic_to_ecliptic_tilt_deg.toFixed(1)}° to the ecliptic (${LI.galactic_to_equator_tilt_deg.toFixed(1)}° to the equator), and the galactic centre sits at (λ, β) = (${c[0].toFixed(1)}°, ${c[1].toFixed(1)}°), close to the ecliptic: the prior's sources have a median |β| of ${A.median.toFixed(1)}° (${I.median_deg.toFixed(0)}° if isotropic) for the ${partName(S.part)}.`);
  const rows = PARTS.map(([id, name]) => { const a = LI.abs_beta[partKey(id)]; return [name, `${a.median.toFixed(1)}°`, pct(a.below_10deg), pct(a.below_30deg), a.cos_beta_mean.toFixed(3), a.cos_beta_median.toFixed(3)]; });
  rows.push(["isotropic sky", `${I.median_deg.toFixed(1)}°`, pct(I.below_10deg), pct(I.below_30deg), I.cos_beta_mean.toFixed(3), (Math.sqrt(3) / 2).toFixed(3)]);
  html("t-lisa", table(["", "median |β|", "|β| &lt; 10°", "|β| &lt; 30°", "mean cos β", "median cos β"], rows, null));
}

function rotCard() {
  const R = RT.round_trip_rad, rows = [
    ["|M Mᵀ − I|, max (<code>EQUATORIAL_TO_GALACTIC</code>)", sig(RT.orthonormal_max_err, 2)],
    ["det M", RT.det.toFixed(12)],
    ["|E Eᵀ − I|, max (ecliptic, page's own)", sig(RT.ecliptic_orthonormal_max_err, 2)],
    ["round trip eq → gal → eq, max / median [rad]", `${sig(R.equatorial_max, 2)} / ${sig(R.equatorial_median, 2)}`],
    ["round trip gal → eq → gal, max / median [rad]", `${sig(R.galactic_max, 2)} / ${sig(R.galactic_median, 2)}`],
    ["galactic centre vs Hipparcos", `${sig(RT.centre_err_arcsec, 3)}″`],
    ["north galactic pole vs Hipparcos", `${sig(RT.pole_err_arcsec, 3)}″`],
    [`astropy ICRS → Galactic, max over ${RT.astropy_draws.toLocaleString()}`, `${sig(RT.astropy_galactic_max_arcsec, 3)}″`],
    [`astropy ICRS → mean ecliptic J2000, max`, `${sig(RT.astropy_ecliptic_max_arcsec, 3)}″`],
  ];
  html("t-rot", table(["rotation check", "value"], rows, null));
  const C = PG.checks.frames, cl = CH.clip, mix = [
    ...FRAMES.map(([f]) => [`∫ p dΩ, ${f} quadrature`, PG.sky.frames[f].integral.toFixed(6)]),
    ...NAMES.map(n => [`∫ p dΩ, ${n} alone (galactic / ecliptic / equatorial)`, FRAMES.map(([f]) => PG.sky.frames[f].component_integrals[n].toFixed(6)).join(" / ")]),
    ...NAMES.slice(1, 3).map(n => [`… ${n}, galactic, midpoint rule on its cusp predicts`, PG.cusp[n].toFixed(6)]),
    ["max |log Σ w p − log p|, maps / quadrature", `${sig(Math.max(...FRAMES.map(([f]) => C[f].map_log_err)), 2)} / ${sig(Math.max(...FRAMES.map(([f]) => C[f].quadrature_log_err)), 2)}`],
    [`bulge draws clipped at |sin b| &gt; 1 / |l| &gt; π`, `${cl.bulge_sin_b_beyond_1} / ${cl.bulge_l_beyond_pi} of ${cl.bulge_draws.toLocaleString()}`],
    ["… expected probability", `10${sup(Math.round(cl.log10_p_sin_b_beyond_1))} / 10${sup(Math.round(cl.log10_p_l_beyond_pi))}`],
  ];
  html("t-mix", table(["mixture check", "value"], mix, null));
  html("n-rot", `The prior lives in galactic (l, b), jaxgb reads equatorial (RA, Dec) in ICRS, and the page draws the ecliptic with the J2000 obliquity ${RT.obliquity_deg.toFixed(7)}°. <code>EQUATORIAL_TO_GALACTIC</code> is orthonormal with determinant 1 to round-off; a round trip on all ${SD.n.toLocaleString()} draws returns every source within ${sig(Math.max(R.equatorial_max, R.galactic_max), 2)} rad; it puts the galactic centre and pole within ${sig(Math.max(RT.centre_err_arcsec, RT.pole_err_arcsec), 2)}″ of Hipparcos and agrees with astropy to ${sig(RT.astropy_galactic_max_arcsec, 2)}″. The right table checks the density: every frame integrates to one (the galactic grid comes out ${sig(1 - PG.sky.frames.galactic.integral, 2)} short because the disks' cusps at l = 0 and b = 0 sit on cell edges, where the midpoint rule falls short by (h/2s)/sinh(h/2s) per axis: it predicts the thin and thick disks' integrals to the digit), the parts written out by hand sum to <code>sky_log_density</code>, and the bulge Gaussian is never truncated in practice.`);
}

function flowCard() {
  const host = document.getElementById("p-flow"), fl = W1.flow, gs = PG.sky.galactic_scale, wf = CH.wrap_flow;
  const lo = -6.5, hi = 6.5, W = Math.max(320, host.clientWidth), row = 56, t = 4, l = 64, r = 16, H = t + row * fl.length + 30;
  const X = v => l + (clamp(v, lo, hi) - lo) / (hi - lo) * (W - l - r);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="flow sky coordinates of the prior">`;
  niceTicks(lo, hi, 8).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${t}" y2="${t + fl.length * row}" stroke="var(--grid)"/><text x="${X(v)}" y="${t + fl.length * row + 16}" text-anchor="middle">${v}</text>`);
  fl.forEach((f, i) => {
    const y = t + i * row, dd = density(f.hist), ref = f.hist.reference.map(v => v / dd.n / dd.w), top = Math.max(...dd.v, ...ref), e = f.hist.edges, x = centres(e);
    const Y = v => y + row - 4 - v / top * (row - 10);
    dd.v.forEach((v, k) => { if (v > 0) s += `<rect x="${X(e[k]) + 0.5}" width="${Math.max(0.5, X(e[k + 1]) - X(e[k]) - 1)}" y="${Y(v)}" height="${y + row - 4 - Y(v)}" fill="var(--bar)"/>`; });
    s += `<polyline points="${x.map((v, k) => `${X(v).toFixed(1)},${Y(ref[k]).toFixed(1)}`).join(" ")}" fill="none" stroke="var(--ink-2)" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    if (wf[i] < hi) [-1, 1].forEach(sg => s += `<line x1="${X(sg * wf[i])}" x2="${X(sg * wf[i])}" y1="${y + 2}" y2="${y + row - 4}" stroke="var(--c2)" stroke-width="2"/>`);
    s += `<text class="lbl" x="${l - 8}" y="${y + row / 2 + 2}" text-anchor="end">${f.name}</text><text x="${l - 8}" y="${y + row / 2 + 14}" text-anchor="end">std ${f.std.toFixed(2)}</text>`;
  });
  s += `<text x="${(l + W - r) / 2}" y="${H - 2}" text-anchor="middle">flow coordinate</text>`;
  host.innerHTML = s + "</svg>";
  html("lg-flow", legendBox("var(--bar)", `${W1.sources.toLocaleString()} training-window sources`) + legendItem("var(--ink-2)", "quadrature", "4 3") + legendItem("var(--c2)", "l = ±π, the wrap"));
  html("n-flow", `The flow sees galactic (l, b) divided by <code>GALACTIC_SCALE</code> = (${gs[0].toFixed(5)}, ${gs[1].toFixed(5)}) rad, the prior's own root-mean-square l and b (recomputed by <code>galactic_scales()</code>: ${CH.galactic_scale_recomputed.map(v => v.toFixed(5)).join(", ")}), so both come out whitened: std ${fl.map(f => `${f.std.toFixed(3)} (${f.name})`).join(" and ")} over the training windows, ${CH.parts.sum.flow_std.map(v => v.toFixed(3)).join(" and ")} over 2²⁰ draws. Both are flat coordinates and nothing clips them: 98% of the sources lie in ${fl.map(f => `[${f.quantiles[0].toFixed(2)}, ${f.quantiles[4].toFixed(2)}] (${f.name})`).join(" and ")}. l wraps at ±π/${gs[0].toFixed(3)} = ±${wf[0].toFixed(2)}; b's poles are at ±${wf[1].toFixed(1)}. ${PG.bands_same ? "The narrow-B windows draw the same sky from the same keys, so these rows hold for both bands; only the correlations below, with the band-dependent coordinates, differ." : "The narrow-B windows draw a different sky."}`);
  const N = CH.near_wrap, deltas = Object.keys(N), pd = CH.plane_density;
  html("t-wrap", table(["|l| &gt; π − δ", ...deltas.map(d => `δ = ${d} rad`)], [
    ["share of draws", ...deltas.map(d => pct(N[d].draws, 3))],
    ["quadrature", ...deltas.map(d => pct(N[d].quadrature, 3))],
    ...PARTS.slice(1).map(([id, name]) => [`… of the ${name}`, ...deltas.map(d => pct(N[d].by_part[partKey(id)], 3))]),
  ], null));
  html("n-wrap", `The flow's l jumps from +${wf[0].toFixed(2)} to −${wf[0].toFixed(2)} across the anticentre, so a posterior there would be split across the chart. Little prior mass sits there: ${pct(N["0.1"].draws, 2)} of draws lie within 0.1 rad of l = ±π, against ${pct(0.1 / Math.PI, 2)} for a uniform l. In the plane the density falls from ${sig(pd.centre, 3)} sr⁻¹ at the centre to ${sig(pd.quarter, 3)} at l = 90° and ${sig(pd.anticentre, 3)} at the anticentre, ${sig(pd.anticentre / pd.floor, 3)}× the floor's ${sig(pd.floor, 3)}.`);
  const slots = W1.slots;
  html("t-slots", table(["source", "std l", "std b", "|b| &lt; 10°"], slots.map((sl, k) => [k + 1, sl.l.std.toFixed(3), sl.b.std.toFixed(3), pct(sl.abs_b_below_10deg)]), null));
  const names = W1.corr_names, cols = names.map((n, j) => j).filter(j => j !== 2 && j !== 3);
  html("t-corr", table(["r with", ...cols.map(j => names[j]), "l ↔ b"], [
    ["l", ...cols.map(j => W1.corr_l[j].toFixed(3)), W1.corr_l[3].toFixed(3)],
    ["b", ...cols.map(j => W1.corr_b[j].toFixed(3)), ""],
  ], null));
  const offs = [...cols.map(j => W1.corr_l[j]), ...cols.map(j => W1.corr_b[j]), W1.corr_l[3]].map(Math.abs);
  html("n-slots", `The sources are unordered and the sky is drawn independently of everything else, so every source index should see the same sky. Correlations of l and b with the window's other flow coordinates stay within |r| ≤ ${Math.max(...offs).toFixed(3)}, against a noise level of 1/√${W1.sources.toLocaleString()} = ${(1 / Math.sqrt(W1.sources)).toFixed(4)}.`);
}

function meaning() {
  const P = W1.sky.abs_lat_below_10deg, W = W1.sky.within_rad_of_centre, fl = W1.flow, X = PG.chi2, b = CH.parts.bulge, N = CH.near_wrap;
  const chis = [...Object.values(X.windows), ...Object.values(X.sky).flatMap(p => Object.values(p))].flatMap(f => Object.values(f)), zs = chis.map(c => Math.abs(c.v - 1) / Math.sqrt(2 / c.dof));
  html("meaning", [
    `<b>The galactic sky prior traces the plane, not a point.</b> ${pct(P.galactic)} of draws have |b| &lt; 10°, but only ${["0.1", "0.2", "0.3", "0.5"].map(k => pct(W[k], 0)).join(", ")} lie within 0.1, 0.2, 0.3 and 0.5 rad of the centre, so a quiet source's sky posterior is mostly prior.`,
    `<b>Other frames see the plane tilted.</b> Only ${pct(P.ecliptic)} of draws have |β| &lt; 10° and ${pct(P.equatorial)} |Dec| &lt; 10°: in jaxgb's equatorial frame the galaxy is a band that crosses the equator, and in LISA's ecliptic it crosses at ${LI.galactic_to_ecliptic_tilt_deg.toFixed(0)}° with the centre ${Math.abs(LI.centre_ecliptic_deg[1]).toFixed(1)}° off the plane.`,
    `<b>Each part does one job.</b> The bulge (${pct(PG.sky.components.weights[0], 1)}) makes the central peak and is ${sig(b.flow_std[0], 2)} flow units wide in l; the disks make the band; the ${pct(PG.sky.components.weights[3], 2)} isotropic floor carries ${pct(CH.parts[NAMES[3]].variance_share[1], 0)} of b's second moment, and with the thick disk it sets the scale <code>GALACTIC_SCALE</code> divides b by, so the flow's b is far wider than the thin disk the most sources sit in.`,
    `<b>The draws match the density in every frame and for every part</b> (χ²/dof ${Math.min(...chis.map(c => c.v)).toFixed(2)}–${Math.max(...chis.map(c => c.v)).toFixed(2)}, every cell within ${Math.max(...zs).toFixed(1)}σ of 1 for its dof), and the rotations are exact to ${sig(Math.max(RT.round_trip_rad.equatorial_max, RT.round_trip_rad.galactic_max), 1)} rad and ${sig(RT.astropy_galactic_max_arcsec, 2)}″ of astropy.`,
    `<b>l and b reach the flow whitened</b> (std ${fl.map(f => `${f.std.toFixed(2)} (${f.name})`).join(" and ")}), with ${pct(N["0.1"].draws, 2)} of the mass within 0.1 rad of the wrap at l = ±π.`,
  ].map(t => `<li>${t}</li>`).join(""));
}

function render() { tiles(); skyCard(); compCard(); margCard(); distCard(); lisaCard(); rotCard(); flowCard(); meaning(); writeHash(); }
let pending = 0;
const schedule = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; render(); }); };
addEventListener("resize", schedule);
// the sky map's ramp is read from the theme's tokens
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
readHash();
render();
</script>
</body>
</html>
"""


def rounded(tree, digits: int = 5):
    """Every float in a JSON tree to `digits` significant figures, for the page's plots."""
    return json.loads(
        json.dumps(tree), parse_float=lambda s: float(f"{float(s):.{digits}g}")
    )


def chi2_per_bin(h: dict) -> dict:
    """χ²/dof of the drawn counts against the quadrature's, over bins expecting at least 5 draws."""
    pairs = [(c, r) for c, r in zip(h["counts"], h["reference"]) if r >= 5]
    x = sum((c - r) ** 2 / r for c, r in pairs)
    return dict(v=x / (len(pairs) - 1), dof=len(pairs) - 1)


def chi2(sky: dict) -> dict:
    """χ²/dof of every marginal: the full-B windows' sum, and each part of the sky draws."""
    windows = sky["bands"]["full"]["sky"]["marginals"]
    draws = sky["sky_draws"]["marginals"]
    return dict(
        windows={
            f: {k: chi2_per_bin(h) for k, h in m.items()} for f, m in windows.items()
        },
        sky={
            part: {
                f: {k: chi2_per_bin(h) for k, h in draws[f][part].items()}
                for f in draws
            }
            for part in draws["galactic"]
        },
    )


def old_proxy_offset(sky: dict) -> float:
    """The angle between the retired Gaussian proxy's centre and the true galactic centre, in equatorial."""
    unit = lambda lon, lat: (
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    )
    a, b = unit(*OLD_PROXY_RADEC), unit(*sky["sky"]["frames"]["equatorial"]["centre"])
    return math.degrees(math.acos(min(1.0, sum(x * y for x, y in zip(a, b)))))


def cusp_integrals(sky: dict) -> dict:
    """What the midpoint rule gives a disk whose cusps at l = 0 and b = 0 sit on the galactic grid's cell edges."""
    width, height = sky["sky"]["quadrature"]
    steps = (2 * math.pi / width, math.pi / height)
    out = {}
    for name, scales in zip(
        sky["sky"]["components"]["names"][1:3], sky["sky"]["components"]["scales"][1:3]
    ):
        x = [h / (2 * s) for h, s in zip(steps, scales)]
        out[name] = math.prod(v / math.sinh(v) for v in x)
    return out


def page_data(sky: dict) -> dict:
    """What the page plots: everything in sky.json but the narrow band, which the page only compares."""
    bands = sky["bands"]
    same = all(bands["narrow"][k] == bands["full"][k] for k in ("sky", "flow", "slots"))
    data = dict(
        sky=sky["sky"],
        sky_draws=sky["sky_draws"],
        checks=sky["checks"],
        lisa=sky["lisa"],
        chart=sky["chart"],
        bands=dict(full=bands["full"]),
        bands_same=same,
        chi2=chi2(sky),
        old_proxy=list(OLD_PROXY_RADEC),
        old_proxy_offset_deg=old_proxy_offset(sky),
        cusp=cusp_integrals(sky),
    )
    exact = dict(checks=data.pop("checks"), cusp=data.pop("cusp"))
    return dict(rounded(data, 7), **exact)


def report(sky: dict, data: dict) -> None:
    """Print every number the README's takeaways quote."""
    s, d, ch, li, rt = (
        sky["sky"],
        sky["sky_draws"],
        sky["chart"],
        sky["lisa"],
        sky["checks"]["rotation"],
    )
    bands = sky["bands"]
    print(
        f"draws {d['n']}, windows {bands['full']['windows']} ({bands['full']['sources']} sources); bands identical: {data['bands_same']}"
    )
    print(
        "components: "
        + "; ".join(
            f"{n} w={w:.5f} scales={sc} drawn={c / d['n']:.5f}"
            for n, w, sc, c in zip(
                s["components"]["names"],
                s["components"]["weights"],
                s["components"]["scales"],
                d["component_counts"],
            )
        )
    )
    print(
        f"old proxy offset {data['old_proxy_offset_deg']:.2f} deg; midpoint cusp predictions {data['cusp']}"
    )
    print(
        f"catalogue sources {s['components']['catalogue_sources']}, galactic scale {s['galactic_scale']}, recomputed {ch['galactic_scale_recomputed']}"
    )
    for frame in s["frames"]:
        f = s["frames"][frame]
        print(
            f"{frame}: integral {f['integral']:.6f}; components {', '.join(f'{v:.6f}' for v in f['component_integrals'].values())};"
            f" map/quadrature log err {sky['checks']['frames'][frame]['map_log_err']:.1e}/{sky['checks']['frames'][frame]['quadrature_log_err']:.1e}"
        )
        print(
            f"  |lat| < 10°: windows {bands['full']['sky']['abs_lat_below_10deg'][frame]:.4f}; "
            + ", ".join(
                f"{p} {v['quadrature']:.4f} ({v['draws']:.4f})"
                for p, v in d["abs_lat_below_10deg"][frame].items()
            )
        )
    print(
        "within rad of the centre, windows: "
        + ", ".join(
            f"{r} {v:.4f}"
            for r, v in bands["full"]["sky"]["within_rad_of_centre"].items()
        )
    )
    for part, w in d["within_rad_of_centre"].items():
        print(
            f"  {part}: "
            + ", ".join(f"{r} {v['quadrature']:.4f}" for r, v in w.items())
            + f"; KS {d['from_centre'][part]['ks']:.2e}"
        )
    X = data["chi2"]
    groups = dict(windows=X["windows"], **X["sky"])
    worst_z = 0.0
    for name, g in groups.items():
        cells = [c for f in g.values() for c in f.values()]
        vs = [c["v"] for c in cells]
        z = max(abs(c["v"] - 1) / math.sqrt(2 / c["dof"]) for c in cells)
        worst_z = max(worst_z, z)
        print(f"chi2/dof {name}: {min(vs):.3f}–{max(vs):.3f}, worst z {z:.2f}")
    print(f"chi2/dof worst z over all cells {worst_z:.2f}")
    print(
        f"lisa: tilt to ecliptic {li['galactic_to_ecliptic_tilt_deg']:.2f}°, to equator {li['galactic_to_equator_tilt_deg']:.2f}°;"
        f" centre (λ, β) = ({li['centre_ecliptic_deg'][0]:.2f}°, {li['centre_ecliptic_deg'][1]:.2f}°); isotropic {li['isotropic']}"
    )
    for part, a in li["abs_beta"].items():
        print(
            f"  {part}: median |β| {a['median']:.2f}°, <10° {a['below_10deg']:.4f}, <30° {a['below_30deg']:.4f}, mean cos β {a['cos_beta_mean']:.4f}"
        )
    r = rt["round_trip_rad"]
    print(
        f"rotation: |MM^T-I| {rt['orthonormal_max_err']:.1e}, det {rt['det']:.15f}, ecliptic |EE^T-I| {rt['ecliptic_orthonormal_max_err']:.1e};"
        f" round trip eq {r['equatorial_max']:.1e}/{r['equatorial_median']:.1e}, gal {r['galactic_max']:.1e}/{r['galactic_median']:.1e} rad;"
        f' centre {rt["centre_err_arcsec"]:.4f}", pole {rt["pole_err_arcsec"]:.4f}"; astropy gal {rt["astropy_galactic_max_arcsec"]:.4f}", ecl {rt["astropy_ecliptic_max_arcsec"]:.4f}" ({rt["astropy_draws"]})'
    )
    print(
        f"chart: draws std {ch['draws_std']}, wrap at {ch['wrap_flow']}"
    )
    for part, p in ch["parts"].items():
        print(
            f"  {part}: flow std {p['flow_std']}, variance share {p['variance_share']}"
        )
    for delta, n in ch["near_wrap"].items():
        print(
            f"  near wrap {delta}: draws {n['draws']:.5f}, quadrature {n['quadrature']:.5f}, uniform {float(delta) / math.pi:.5f}; {n['by_part']}"
        )
    print(f"  plane density {ch['plane_density']}; clip {ch['clip']}")
    for band, b in bands.items():
        offs = [abs(v) for j, v in enumerate(b["corr_l"]) if j != 2] + [
            abs(v) for j, v in enumerate(b["corr_b"]) if j not in (2, 3)
        ]
        print(
            f"{band}: {b['seconds']:.1f} s; |b| < 10° {b['sky']['abs_lat_below_10deg']['galactic']:.4f}; "
            + "; ".join(
                f"{f['name']} std {f['std']:.4f}, 1-99% {f['quantiles'][0]:.2f} to {f['quantiles'][4]:.2f}"
                for f in b["flow"]
            )
            + f"; max |r| {max(offs):.4f}; sources l std {[round(sl['l']['std'], 3) for sl in b['slots']]}, b std {[round(sl['b']['std'], 3) for sl in b['slots']]}"
        )


def main() -> None:
    sky = json.loads(
        Path(
            sys.argv[1] if len(sys.argv) > 1 else REPORT / "data" / "sky.json"
        ).read_text()
    )
    data = page_data(sky)
    page = PAGE.replace("__PAGE__", json.dumps(data, separators=(",", ":"))).replace(
        "__SUB__", f"Built {date.today().isoformat()} from <code>data/sky.json</code>."
    )
    out = REPORT / "sky.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")
    report(sky, data)


if __name__ == "__main__":
    main()
