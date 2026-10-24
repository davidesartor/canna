"""Write source-count.html from data/: how many prior sources, and how many visible ones, a window holds.

Run from the repo root:
    uv run --frozen python reports/prior/source-count/scripts/build_page.py
"""

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import binom, norm

from canna.lisa.physics import SKY_CATALOGUE_SOURCES, SOURCE_COVARIANCE, SOURCE_MEAN
from canna.lisa.problem import WINDOW_RESPONSES

FOLDER = Path(__file__).parents[1]
DATA = FOLDER / "data"
VISIBLE_SNR = 7.0
# the SNR thresholds the slider offers, and the log-f0 bins of the share reaching them
SNR_STEP = 0.5
SNR_STEPS = np.arange(0.0, 30.0 + SNR_STEP / 2, SNR_STEP)
F0_BINS = 96
WIDTHS = tuple(2**k for k in range(5, 13))
# window centres as cells of the band, and the points of the window integral
CENTRES = 600
SIMPSON = 9
PROBABILITIES = (0.5, 0.9, 0.99, 0.999, 0.9999)
# the sources in the band, one survival curve each
SOURCE_COUNTS = (1e6, 3e6, 1e7, 3e7, 1e8)

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior: source count</title>
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
.row { display: flex; align-items: center; justify-content: flex-start; gap: 8px; margin-bottom: 10px; }
.controls { position: sticky; top: 0; z-index: 3; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 24px; margin: 0 0 16px; padding: 8px 0; font-size: 13px; color: var(--ink-2); background: var(--page); }
.controls b { color: var(--ink); font-variant-numeric: tabular-nums; }
.controls b.val { display: inline-block; min-width: 4.5ch; }
.controls label { display: inline-flex; align-items: center; gap: 6px; }
.controls input[type=range] { width: 160px; accent-color: var(--c1); }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; vertical-align: middle; }
.seg button { font: inherit; color: var(--ink-2); background: transparent; border: 0; padding: 4px 10px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg text.lab { fill: var(--ink-2); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; white-space: nowrap; z-index: 2; }
.tip .row { color: var(--ink-2); font-variant-numeric: tabular-nums; margin: 0; gap: 6px; }
.meaning ul { margin: 8px 0 0; padding-left: 20px; }
.meaning li { margin: 0 0 8px; max-width: 80ch; }
#err { position: sticky; top: 0; z-index: 10; background: #c0392b; color: #fff; padding: 10px 16px; font-size: 14px; }
</style>
</head>
<body>
<div id="err" role="alert" hidden></div>
<main>
<p class="nav">The LisaGB prior by parameter group: <a data-page="../f0-fdot-amplitude/f0-fdot-amplitude.html" href="../f0-fdot-amplitude/f0-fdot-amplitude.html">f₀, ḟ, A</a> · <a data-page="../polarization/polarization.html" href="../polarization/polarization.html">polarization</a> · <a data-page="../sky/sky.html" href="../sky/sky.html">sky</a> · <a data-page="../response-span/response-span.html" href="../response-span/response-span.html">response span</a> · <b>source count</b></p>
<h1>The LisaGB prior: source count</h1>
<p class="sub">__SUB__</p>
<div class="tiles">__TILES__</div>
<div class="controls">
  <label>visible at SNR ≥ <b id="v-snr" class="val"></b> <input type="range" id="ctl-snr" min="0" step="__SNR_STEP__" max="__SNR_MAX__" aria-label="SNR threshold"></label>
  <span>window w <span class="seg" id="ctl-w"></span> bins</span>
  <span id="readout"></span>
</div>
<div class="card">
  <h2>How many sources does a window hold?</h2>
  <p class="note">__N_COUNT__</p>
  <div class="legend" id="lg-count"></div>
  <div id="p-count"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Where are the windows that hold the most?</h2>
  <p class="note">__N_CENTRE__</p>
  <div class="legend" id="lg-centre"></div>
  <div id="p-centre"></div><div class="tip"></div>
</div>
<div class="card meaning">
  <h2>What this means</h2>
  <ul>__MEANING__</ul>
</div>
</main>
<script>
const PG = __PAGE__;
const S = {snr: PG.default_snr, w: PG.default_w};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => v > 0 && v < 10 ** -(d + 2) ? `${(100 * v).toExponential(1)}%` : `${(100 * v).toFixed(d)}%`;
const share = v => `${+(100 * v).toPrecision(4)}%`;
const nearest = (xs, v) => xs.reduce((best, x, i) => Math.abs(x - v) < Math.abs(xs[best] - v) ? i : best, 0);
const mhz = f => sig(f * 1e3, 3);
const count = k => k >= 1e6 ? `${+(k / 1e6).toPrecision(3)}M` : k >= 1e4 ? `${+(k / 1e3).toPrecision(3)}k` : k.toLocaleString();

// state in the address bar: #snr=7&w=512
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.has("snr") && +q.get("snr") >= 0 && +q.get("snr") <= PG.snr_max) S.snr = Math.round(+q.get("snr") / PG.snr_step) * PG.snr_step;
  if (PG.widths.includes(+q.get("w"))) S.w = +q.get("w");
}
function writeHash() {
  history.replaceState(null, "", "#" + new URLSearchParams(Object.entries(S)).toString());
}

function tip(host, ev, html) {
  const t = host.nextElementSibling, card = host.closest(".card").getBoundingClientRect();
  t.innerHTML = html; t.style.display = "block";
  const x = ev.clientX - card.left + 14, y = ev.clientY - card.top + 14;
  t.style.left = `${Math.min(x, card.width - t.offsetWidth - 8)}px`; t.style.top = `${y}px`;
}
const untip = host => host.nextElementSibling.style.display = "none";
const trow = (color, text) => `<div class="row">${color ? `<i class="dot" style="background:${color};width:8px;height:8px"></i>` : ""}${text}</div>`;
const legendItem = (color, text, dash) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${color}" stroke-width="2"${dash ? ` stroke-dasharray="${dash}"` : ""}/></svg>${text}</span>`;
const html = (id, text) => document.getElementById(id).innerHTML = text;

function niceTicks(lo, hi, n = 4) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag;
  const step = (r >= 5 ? 10 : r >= 2 ? 5 : r >= 1 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(+v.toPrecision(12));
  return out;
}
// generic plot: layers in data units; ticks as [position, label] pairs
function chart(host, o) {
  const W = Math.max(300, host.clientWidth), H = o.H ?? 280, m = {l: 54, r: 16, t: 18, b: o.xlabel ? 38 : 8};
  const [x0, x1] = o.x, [y0, y1] = o.y;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const cx = v => clamp(X(v), m.l, W - m.r), cy = v => clamp(Y(v), m.t, H - m.b);
  const pts = arr => arr.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label}">`;
  (o.yticks ?? niceTicks(y0, y1, 4).map(t => [t, sig(t, 3)])).forEach(([t, l]) => s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${l}</text>`);
  (o.xticks ?? niceTicks(x0, x1, 5).map(t => [t, sig(t, 3)])).forEach(([t, l]) => s += `${o.xgrid ? `<line x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--grid)"/>` : ""}<line x1="${X(t)}" x2="${X(t)}" y1="${H - m.b}" y2="${H - m.b + 4}" stroke="var(--axis)"/>${o.xlabel ? `<text x="${X(t)}" y="${H - m.b + 17}" text-anchor="middle">${l}</text>` : ""}`);
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--axis)"/>`;
  s += `${o.xlabel ? `<text x="${(m.l + W - m.r) / 2}" y="${H - 3}" text-anchor="middle">${o.xlabel}</text>` : ""}<text x="${m.l}" y="${m.t - 6}">${o.ylabel}</text>`;
  s += `<clipPath id="clip-${host.id}"><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/></clipPath><g clip-path="url(#clip-${host.id})">`;
  for (const L of o.layers) {
    if (L.area && L.area.length) s += `<polygon points="${pts([[L.area[0][0], L.base ?? y0], ...L.area, [L.area.at(-1)[0], L.base ?? y0]])}" fill="${L.color}" opacity="${L.opacity ?? 1}"/>`;
    if (L.line && L.line.length) s += `<polyline points="${pts(L.line)}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 2}" stroke-linejoin="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.pts) L.pts.forEach(p => s += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="${L.r ?? 2}" fill="${L.color}" opacity="${L.opacity ?? 0.7}"/>`);
    if (L.hline !== undefined) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(L.hline)}" y2="${Y(L.hline)}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="${L.width ?? 1.2}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.vline !== undefined) s += `<line x1="${X(L.vline)}" x2="${X(L.vline)}" y1="${m.t}" y2="${H - m.b}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="${L.width ?? 1.2}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
  }
  s += "</g>";
  // labels outside the clip, so text at an edge stays readable
  for (const L of o.layers) {
    if (L.vline !== undefined && L.text) s += `<text class="lab" x="${cx(L.vline) + (L.tdx ?? 0)}" y="${m.t + 12 + (L.dy ?? 0)}" text-anchor="${L.anchor ?? "middle"}">${L.text}</text>`;
    if (L.hline !== undefined && L.text) s += `<text class="lab" x="${L.left ? m.l + 4 : W - m.r - 4}" y="${cy(L.hline) - 4}" text-anchor="${L.left ? "start" : "end"}">${L.text}</text>`;
  }
  s += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" visibility="hidden"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  host.insertAdjacentHTML("beforeend", s + "</svg>");
  const svg = host.lastElementChild, cross = svg.querySelector(".cross"), hit = svg.querySelector(".hit");
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    const out = o.hover({x: x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0)});
    cross.setAttribute("visibility", out ? "visible" : "hidden");
    if (!out) return untip(host);
    cross.setAttribute("x1", X(out.cross)); cross.setAttribute("x2", X(out.cross));
    tip(host, ev, out.html);
  });
  hit.addEventListener("pointerleave", () => { untip(host); cross.setAttribute("visibility", "hidden"); });
}

// Numerical Recipes' erfc, fractional error below 1.2e-7 everywhere, so tails stay accurate
function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
    + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const normCdf = z => 0.5 * erfc(-z / Math.SQRT2);
const normPdf = x => Math.exp(-0.5 * ((x - PG.mean) / PG.std) ** 2) / (PG.std * Math.sqrt(2 * Math.PI));

// per window centre, the prior share of sources with f0 in its w bins reaching SNR t: build_page.window_share
function windowShares(w, t) {
  const row = PG.reaching[Math.round(t / PG.snr_step)], [f_lo, f_hi] = PG.f0_range, n = PG.simpson;
  const x0 = PG.x[0], dx = PG.x[1] - PG.x[0], last = PG.x.length - 1;
  const reaching = x => { const u = clamp((x - x0) / dx, 0, last), i = Math.min(Math.floor(u), last - 1); return row[i] + (u - i) * (row[i + 1] - row[i]); };
  const half = w / (2 * PG.t_obs), inner = (w - w / PG.responses) / (2 * PG.t_obs);
  return PG.centres.map(c => {
    const f = clamp(10 ** c, f_lo + inner, f_hi - inner), lo = Math.log10(Math.max(f - half, f_lo)), hi = Math.log10(Math.min(f + half, f_hi));
    let s = 0;
    for (let i = 0; i < n; i++) { const x = lo + i / (n - 1) * (hi - lo); s += (i === 0 || i === n - 1 ? 1 : i % 2 ? 4 : 2) * normPdf(x) * reaching(x); }
    return s / PG.band * (hi - lo) / (3 * (n - 1));
  });
}

// Bin(N, p) as a cdf over integers k: exact by recurrence up to a mean of 500, normal past it
function binomial(N, p) {
  const mean = N * p, sd = Math.sqrt(mean * (1 - p)), top = Math.ceil(mean + 8 * sd + 10);
  if (mean > 500) return {mean, top, cdf: k => k < 0 ? 0 : normCdf((Math.floor(k) + 0.5 - mean) / sd)};
  const end = Math.ceil(mean + 12 * sd + 20), cum = new Float64Array(end + 1);
  let pmf = Math.exp(N * Math.log1p(-p)), s = 0;
  for (let k = 0; k <= end; k++) { s += pmf; cum[k] = Math.min(s, 1); pmf *= (N - k) / (k + 1) * p / (1 - p); }
  return {mean, top, cdf: k => k < 0 ? 0 : k > end ? 1 : cum[Math.floor(k)]};
}
// the smallest integer k with cdf(k) >= y, by bisection on [0, hi]
function quantile(cdf, y, hi) {
  const target = y * (1 - 1e-12);
  if (cdf(0) >= target) return 0;
  let lo = 0;
  while (hi - lo > 1) { const m = Math.floor((lo + hi) / 2); if (cdf(m) >= target) hi = m; else lo = m; }
  return hi;
}

// K over windows: per centre Bin(N, p), mixed with the centres' weights
let memo = {};
function distribution(n) {
  const key = `${S.snr}|${S.w}|${n}`;
  if (memo[key]) return memo[key];
  const sharesKey = `${S.snr}|${S.w}`;
  if (memo.sharesKey !== sharesKey) memo = {sharesKey, p: windowShares(S.w, S.snr)};
  const p = memo.p, B = p.map(v => binomial(n, v)), total = PG.weights.reduce((a, b) => a + b, 0);
  return memo[key] = {p, B, top: Math.max(...B.map(b => b.top)),
    mix: k => B.reduce((s, b, c) => s + PG.weights[c] * b.cdf(k), 0) / total,
    mean: B.reduce((s, b, c) => s + PG.weights[c] * b.mean, 0) / total};
}
// per centre, the median and each central interval's lower and upper quantile, as [lower, upper] per level
const levels = D => D.levels ??= PG.probabilities.map(y => D.B.map(b => [quantile(b.cdf, 1 - y, b.top), quantile(b.cdf, y, b.top)]));
const interval = y => y === 0.5 ? "median" : `${share(1 - y)}–${share(y)}`;

const shade = j => `color-mix(in srgb, var(--c1) ${[30, 45, 65, 85, 100][j]}%, transparent)`;
// the median darkest, wider intervals lighter
const levelShade = j => shade(PG.probabilities.length - 1 - j);
const at = k => Math.log10(k + 1);
function kTicks(top) {
  const out = [[0, "0"]];
  for (let e = 0; 10 ** e <= top; e++) for (const m of top > 1e4 ? [1] : [1, 2, 5]) if (m * 10 ** e <= top) out.push([at(m * 10 ** e), count(m * 10 ** e)]);
  return out;
}

function controls() {
  html("ctl-w", PG.widths.map(w => `<button data-v="${w}" aria-pressed="${S.w === w}">${w}</button>`).join(""));
  document.querySelectorAll("#ctl-w button").forEach(b => b.onclick = () => { S.w = +b.dataset.v; schedule(); });
  html("v-snr", S.snr === 0 ? "all" : sig(S.snr));
  html("readout", `w = ${PG.responses}·${S.w / PG.responses}: mean K <b>${sig(distribution(1e6).mean)}</b> per million sources in the band`);
}

// the survival of K over windows on log axes, one staircase per N, stepping where the k grid is dense
const FLOOR = -6;
function countCard() {
  const host = document.getElementById("p-count"); host.innerHTML = "";
  const Ds = PG.source_counts.map(distribution), top = Math.max(...Ds.map(D => D.top));
  const ks = [...new Set([...Array(Math.min(top, 127) + 1).keys(), ...Array.from({length: 400}, (_, i) => Math.round(128 * (top / 128) ** (i / 399))).filter(k => k >= 128 && k <= top)])];
  const sfs = Ds.map(D => ks.map(k => 1 - D.mix(k)));
  const layers = sfs.map((sf, j) => ({color: shade(j), width: 2,
    line: ks.flatMap((k, i) => { const y = Math.log10(Math.max(sf[i], 10 ** (FLOOR - 1))); return i + 1 === ks.length || ks[i + 1] === k + 1 ? [[at(k), y], [at(k + 1), y]] : [[at(k), y]]; })}));
  chart(host, {x: [0, at(top)], y: [FLOOR, 0], H: 380, xlabel: "sources in the window, k", ylabel: "P(K > k), share of windows", label: "survival of the window's source count",
    xticks: kTicks(top), yticks: Array.from({length: 1 - FLOOR}, (_, i) => [-i, `${(100 * 10 ** -i).toFixed(Math.max(0, i - 2))}%`]), xgrid: true, layers,
    hover: ({x}) => {
      const i = nearest(ks.map(at), x), k = ks[i];
      return {cross: at(k), html: `<b>more than ${k.toLocaleString()} sources</b>` + PG.source_counts.map((n, j) => trow(shade(j), `N = ${count(n)}: ${pct(sfs[j][i], 2)} of windows`)).join("")};
    }});
  html("lg-count", PG.source_counts.map((n, j) => legendItem(shade(j), `N = ${count(n)}`)).join(""));
}

function centreCard() {
  const host = document.getElementById("p-centre"), D = distribution(PG.sources), L = levels(D); host.innerHTML = "";
  const top = Math.max(at(Math.max(...L.at(-1).map(q => q[1]))), at(10));
  const layers = [{hline: at(PG.n_sources), color: "var(--ink)", width: 1.2, text: `${PG.n_sources} tokens`, left: true}];
  L.forEach((q, j) => [0, 1].forEach(side => { if (j || side) layers.push({line: PG.centres.map((c, i) => [c, at(q[i][side])]), color: levelShade(j), width: 1.8}); }));
  layers.push({line: PG.centres.map((c, i) => [c, at(D.B[i].mean)]), color: "var(--c2)", width: 1.8, dash: "6 3"});
  chart(host, {x: [PG.centres[0], PG.centres.at(-1)], y: [0, top * 1.04], H: 340, xlabel: "window centre f [mHz]", ylabel: "sources in the window, k", label: "source count quantiles against window centre",
    xticks: PG.f_ticks.map(f => [Math.log10(f), sig(f * 1e3, 2)]), yticks: kTicks(10 ** top - 1), layers,
    hover: ({x}) => {
      const i = nearest(PG.centres, x);
      return {cross: PG.centres[i], html: `<b>windows centred at ${mhz(10 ** PG.centres[i])} mHz</b>`
        + trow("var(--c2)", `mean K ${sig(D.B[i].mean)} (p = ${sig(D.p[i])})`)
        + PG.probabilities.map((y, j) => trow(levelShade(j), j ? `${interval(y)}: ${L[j][i][0].toLocaleString()} to ${L[j][i][1].toLocaleString()}` : `median ${L[0][i][1].toLocaleString()}`)).join("")};
    }});
  html("lg-centre", PG.probabilities.map((y, j) => legendItem(levelShade(j), interval(y))).join("") + legendItem("var(--c2)", "mean N·p", "6 3")
    + legendItem("var(--ink)", `n_sources = ${PG.n_sources}`));
}

function render() { controls(); countCard(); centreCard(); writeHash(); }
let pending = 0;
const schedule = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; render(); }); };
addEventListener("resize", schedule);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
readHash();
const snrInput = document.getElementById("ctl-snr");
snrInput.value = S.snr;
snrInput.oninput = () => { S.snr = +snrInput.value; schedule(); };
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


def mhz(f: float) -> str:
    return sig(f * 1e3)


def rounded(a, digits: int = 4) -> list:
    """An array as a list of floats to `digits` significant figures, for the page's JSON."""
    return [float(f"{v:.{digits}g}") for v in np.asarray(a, dtype=float).ravel()]


def window_share(
    centre_x: np.ndarray,
    width: int,
    count_x: np.ndarray,
    reaching: np.ndarray,
    t_obs: float,
    f0_range: list[float],
) -> np.ndarray:
    """Per window centre (log10 f), the prior share of sources with f0 in its width bins that reach the threshold; the page's JS does the same.

    As sample_f, the window keeps its outer half responses' worth of bins clear of the band edges, or less: the centre is clamped.
    """
    half = width / (2 * t_obs)
    inner = (width - width / WINDOW_RESPONSES) / (2 * t_obs)
    f = np.clip(10.0**centre_x, f0_range[0] + inner, f0_range[1] - inner)
    lo = np.log10(np.maximum(f - half, f0_range[0]))
    hi = np.log10(np.minimum(f + half, f0_range[1]))
    x = lo[:, None] + np.linspace(0, 1, SIMPSON) * (hi - lo)[:, None]
    mean, std = SOURCE_MEAN[0], math.sqrt(SOURCE_COVARIANCE[0, 0])
    band = np.diff(norm.cdf(np.log10(f0_range), mean, std))[0]
    density = norm.pdf(x, mean, std) / band * np.interp(x, count_x, reaching)
    simpson = np.r_[1, np.tile([4, 2], (SIMPSON - 3) // 2), 4, 1]
    return (density * simpson).sum(-1) * (hi - lo) / (3 * (SIMPSON - 1))


def count_quantiles(
    share: np.ndarray, weight: np.ndarray, n: int, ys: tuple[float, ...]
) -> list[int]:
    """The y-quantiles of K ~ Bin(n, share) mixed over the window centres' weights, exact to the k grid."""
    top = float(np.max(n * share + 8 * np.sqrt(n * share) + 10))
    ks = np.unique(np.r_[np.arange(256), np.geomspace(256, max(top, 256), 512).round()])
    cdf = weight @ binom.cdf(ks[None], n, share[:, None])
    return [int(ks[np.argmax(cdf >= y * (1 - 1e-12))]) for y in ys]


def main() -> None:
    d = json.loads((DATA / "source-count.json").read_text())
    z = np.load(DATA / "source-count.npz")
    f_lo, f_hi = d["f0_range"]
    tokens = d["n_sources"]
    window = WINDOW_RESPONSES * d["response_points"]

    # per SNR threshold and log-f0 bin, the prior share of sources at that f0 reaching it, A integrated out
    f0, snr_unit, f0_weight = z["f0"], z["snr_unit"], z["f0_weight"]
    edges = np.linspace(math.log10(f_lo), math.log10(f_hi), F0_BINS + 1)
    count_x = (edges[:-1] + edges[1:]) / 2
    bins = np.clip(np.digitize(np.log10(f0), edges) - 1, 0, F0_BINS - 1)
    reach = lambda t: (
        np.ones_like(snr_unit)
        if t == 0
        else ndtr((z["mean_log_a"] - np.log10(t / snr_unit)) / d["amp_std"])
    )
    bin_weight = np.bincount(bins, f0_weight, F0_BINS)
    reaching = np.array(
        [
            np.bincount(bins, f0_weight * reach(t), F0_BINS) / bin_weight
            for t in SNR_STEPS
        ]
    )
    visible = float(np.mean(f0_weight * reach(VISIBLE_SNR)))

    # window centres as training draws them, from the prior's f0 marginal on the band
    mean, std = SOURCE_MEAN[0], math.sqrt(SOURCE_COVARIANCE[0, 0])
    cells = np.linspace(math.log10(f_lo), math.log10(f_hi), CENTRES + 1)
    centre_x = (cells[:-1] + cells[1:]) / 2
    centre_w = norm.sf(cells[:-1], mean, std) - norm.sf(cells[1:], mean, std)
    centre_w /= centre_w.sum()

    # the takeaways' counts, exact binomials at the config's window and the catalogue's N
    N = SKY_CATALOGUE_SOURCES
    shares = {
        t: window_share(
            centre_x,
            window,
            count_x,
            reaching[round(t / SNR_STEP)],
            d["t_obs"],
            d["f0_range"],
        )
        for t in (0.0, VISIBLE_SNR)
    }
    counts = {
        t: count_quantiles(s, centre_w, N, PROBABILITIES) for t, s in shares.items()
    }
    seen = shares[VISIBLE_SNR]
    empty = float(centre_w @ binom.pmf(0, N, seen))
    over = float(centre_w @ binom.sf(tokens, N, seen))
    mean_seen = float(centre_w @ (N * seen))
    centre_99 = binom.ppf(0.99, N, seen)
    busiest = int(np.argmax(centre_99))
    binned = float(
        np.diff(norm.cdf(edges, mean, std))
        @ reaching[round(VISIBLE_SNR / SNR_STEP)]
        / np.diff(norm.cdf(edges[[0, -1]], mean, std))[0]
    )
    # the window width past which the 99.99% count tops the tokens
    wide = {
        w: count_quantiles(
            window_share(
                centre_x,
                w,
                count_x,
                reaching[round(VISIBLE_SNR / SNR_STEP)],
                d["t_obs"],
                d["f0_range"],
            ),
            centre_w,
            N,
            (0.9999,),
        )[0]
        for w in WIDTHS
    }

    page = dict(
        t_obs=d["t_obs"],
        f0_range=d["f0_range"],
        mean=mean,
        std=std,
        band=float(np.diff(norm.cdf(np.log10(d["f0_range"]), mean, std))[0]),
        snr_step=SNR_STEP,
        snr_max=float(SNR_STEPS[-1]),
        default_snr=VISIBLE_SNR,
        x=rounded(count_x, 6),
        reaching=[rounded(row) for row in reaching],
        centres=rounded(centre_x, 6),
        weights=rounded(centre_w),
        widths=list(WIDTHS),
        default_w=window,
        responses=WINDOW_RESPONSES,
        sources=N,
        source_counts=list(SOURCE_COUNTS),
        n_sources=tokens,
        probabilities=list(PROBABILITIES),
        simpson=SIMPSON,
        f_ticks=[
            f for f in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2] if f_lo <= f <= f_hi
        ],
    )

    tile = (
        lambda k, v, note: f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{note}</div></div>'
    )
    tiles = "".join(
        [
            tile(
                "sources in a window",
                f"{counts[0.0][0]:,}",
                f"median over windows of {window} bins; 99% hold at most {counts[0.0][2]:,}",
            ),
            tile(
                f"windows with none at SNR ≥ {sig(VISIBLE_SNR)}",
                pct(empty, 1),
                f"mean {sig(mean_seen)} visible sources per window",
            ),
            tile(
                "99.99% of windows hold",
                f"≤ {counts[VISIBLE_SNR][4]}",
                f"visible sources; 99.9% ≤ {counts[VISIBLE_SNR][3]}",
            ),
            tile(
                f"more than {tokens} visible",
                pct(over, 2),
                f"of windows; n_sources = {tokens} tokens",
            ),
        ]
    )
    count_note = (
        f"N sources in the band, each independently in a window of w bins centred at f with probability p(f, w) = ∫<sub>window</sub> prior(f₀) P(SNR ≥ t | f₀) df₀, "
        "so a window holds K ~ Bin(N, p(f, w)) sources at SNR ≥ t: the prior thinned to the visible ones, prior(f₀ | SNR ≥ t) with N·P(SNR ≥ t) sources. t = 0 counts every source. "
        "Window centres follow the prior's f₀ marginal, as training draws them; this mixes the binomials. "
        "Each curve is P(K > k), the share of windows holding more than k sources, on a log scale down to 10⁻⁶, for one N; "
        f"the catalogue the prior was fitted to has {N / 1e6:.0f}M. Past a mean of 500 a window's binomial is taken as normal."
    )
    near = float(centre_w[np.abs(centre_x - centre_x[busiest]) <= 0.1].sum())
    centre_note = (
        f"Per window centre, K's median and its 80, 98, 99.8 and 99.98% central intervals over the windows there, both bounds in one shade, and its mean N·p(f, w), at the catalogue's N = {N / 1e6:.0f}M. "
        "Every source counts most near the prior's f₀ peak; visible ones, a decade above it, where the SNR at a given A has grown faster than the prior has thinned."
    )
    meaning = [
        f"<b>A window holds hundreds of thousands of sources.</b> With N = {N:,} and w = {window} bins ({WINDOW_RESPONSES}·{d['response_points']}, {d['config']}), "
        f"a median {counts[0.0][0]:,} and up to {counts[0.0][2]:,} in 99% of windows.",
        f"<b>Most windows hold no visible source.</b> {pct(visible, 2)} of the prior's sources reach SNR {sig(VISIBLE_SNR)}, a mean of {sig(mean_seen)} per window, "
        f"but windows are centred where the prior is, mostly below where SNR {sig(VISIBLE_SNR)} is reachable: {pct(empty, 1)} of windows hold none.",
        f"<b>The rest can hold hundreds.</b> {pct(over, 1)} of windows hold more than the {tokens} source tokens; 99% hold at most {counts[VISIBLE_SNR][2]}, "
        f"99.9% at most {counts[VISIBLE_SNR][3]} and 99.99% at most {counts[VISIBLE_SNR][4]}. "
        f"The busiest centres, near {mhz(10 ** centre_x[busiest])} mHz, hold up to {int(centre_99[busiest])} in 99% of their windows, "
        f"and {pct(near, 1)} of windows are centred within 0.1 dex of them.",
        "<b>The count scales with the window.</b> At SNR ≥ "
        + sig(VISIBLE_SNR)
        + ", 99.99% of windows hold at most "
        + ", ".join(f"{k} at w = {w}" for w, k in wide.items())
        + ".",
    ]
    sub = (
        f"How many sources does one {d['config']} window hold, and how many of them can the data see? "
        f"Per source its SNR at A = 1 on {d['sources']:,} draws, f₀ log-uniform over the band and weighted back to the prior, each against a {d['reference_points']:,}-bin jaxgb reference; "
        "A is Gaussian in log₁₀ A given f₀ and ḟ, so P(SNR ≥ t) integrates it out in closed form. The controls set t and w. "
        f"Built {date.today().isoformat()} from <code>data/</code> (commit <code>{d['commit'][:7]}</code>{', src dirty' if d['src_dirty'] else ''})."
    )
    print(
        f"w = {window}, N = {N}, quantiles {PROBABILITIES}:",
        counts,
        f"\nSNR >= 7 share {visible:.4g} (binned {binned:.4g}), mean visible per window {mean_seen:.4g}",
        f"\nP(K = 0) {empty:.4g}, P(K > {tokens}) {over:.4g}, busiest {mhz(10 ** centre_x[busiest])} mHz: {int(centre_99[busiest])}",
        "\n99.99% visible count per w:",
        wide,
    )
    for t in meaning:
        print("-", t)

    out_html = PAGE
    for key, value in dict(
        SUB=sub,
        TILES=tiles,
        SNR_STEP=str(SNR_STEP),
        SNR_MAX=str(float(SNR_STEPS[-1])),
        N_COUNT=count_note,
        N_CENTRE=centre_note,
        MEANING="".join(f"<li>{t}</li>" for t in meaning),
        PAGE=json.dumps(page, separators=(",", ":")),
    ).items():
        out_html = out_html.replace(f"__{key}__", value)
    out = FOLDER / "source-count.html"
    out.write_text(out_html)
    print(f"wrote {out} ({len(out_html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
