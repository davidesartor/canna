"""Write response-span.html from data/: the R that keeps at least x of a source's above-noise power with probability y, and how wide that power is at each f0, per SNR cut.

Run from the repo root:
    uv run --frozen python reports/prior/response-span/scripts/build_page.py
"""

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from scipy.special import ndtr

FOLDER = Path(__file__).parents[1]
DATA = FOLDER / "data"
# the probabilities y the tiles and tooltips quote
PROBABILITIES = (0.5, 0.9, 0.99, 0.999)
VISIBLE_SNR = 7.0
# the slider's SNR cuts, as LisaGB.visible_snr; 0 counts every source with a bin above the noise
CUTS = tuple(range(21))
SUMMARY_CUTS = (0, 3, 5, 7, 10, 15, 20)
F0_BINS = 20
# the (log10 f0, log10 width) density: cells per axis, a one-cell Gaussian smoothing, the masses its contours hold
CELLS = 64
SMOOTH = np.array([0.054, 0.242, 0.399, 0.242, 0.054])
MASSES = (0.5, 0.9, 0.99)
# the density's integer scale on the page
DENSITY_SCALE = 10000
# the keep curves' x axis, -log10(1 - x) up to 99.99%, then x = 100% at FULL
CURVE_X = np.linspace(0.0, 4.0, 81)
FULL = 4.5
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior: response span</title>
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
h1 { font-size: 24px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 0 0 2px; }
.nav { font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.nav a { color: var(--c1); text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 80ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 2px 0 12px; max-width: 80ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; min-width: 0; position: relative; }
.sticky { position: sticky; top: 0; z-index: 3; background: var(--page); padding: 10px 0; border-bottom: 1px solid var(--border); }
input[type=range] { touch-action: none; }
.controls { flex-wrap: wrap; margin-bottom: 16px; font-size: 13px; color: var(--ink-2); }
.controls input[type=range] { width: min(320px, 60vw); accent-color: var(--c1); margin: 0; }
.controls b { color: var(--ink); font-variant-numeric: tabular-nums; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 0 0 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 13px; color: var(--ink-2); margin: 0 0 6px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 22px; flex: none; }
.row { display: flex; align-items: center; justify-content: flex-start; gap: 8px; margin-bottom: 10px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg text.lab { fill: var(--ink-2); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; white-space: nowrap; z-index: 2; }
.tip .row { color: var(--ink-2); font-variant-numeric: tabular-nums; margin: 0; gap: 6px; }
.meaning ul { margin: 8px 0 0; padding-left: 20px; }
.meaning li { margin: 0 0 8px; max-width: 80ch; }
details { margin-top: 16px; color: var(--ink-2); font-size: 13px; }
details summary { cursor: pointer; color: var(--ink); font-size: 14px; font-weight: 600; }
details p { max-width: 80ch; margin: 8px 0; }
#err { position: sticky; top: 0; z-index: 10; background: #c0392b; color: #fff; padding: 10px 16px; font-size: 14px; }
</style>
</head>
<body>
<div id="err" role="alert" hidden></div>
<main>
<p class="nav">The LisaGB prior by parameter group: <a data-page="../f0-fdot-amplitude/f0-fdot-amplitude.html" href="../f0-fdot-amplitude/f0-fdot-amplitude.html">f₀, ḟ, A</a> · <a data-page="../polarization/polarization.html" href="../polarization/polarization.html">polarization</a> · <a data-page="../sky/sky.html" href="../sky/sky.html">sky</a> · <b>response span</b> · <a data-page="../source-count/source-count.html" href="../source-count/source-count.html">source count</a></p>
<h1>The LisaGB prior: response span</h1>
<p class="sub">__SUB__</p>
<div class="row controls sticky"><span>SNR cut</span><input type="range" id="k" min="0" step="1" aria-label="SNR cut k"><b id="k-label"></b><span id="k-note"></span></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>How much of a source's above-noise power does each R keep?</h2>
  <p class="note">__N_KEEP__</p>
  <div class="legend" id="lg-keep"></div>
  <div id="p-keep"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How many bins does a source need at each f₀?</h2>
  <p class="note">__N_WIDTH__</p>
  <div class="legend" id="lg-width"></div>
  <div id="p-width"></div><div class="tip"></div>
  <div class="legend" id="lg-share" style="margin-top:12px"></div>
  <div id="p-share"></div><div class="tip"></div>
</div>
<div class="card meaning">
  <h2>What this means</h2>
  <ul>__MEANING__</ul>
  <details><summary>How it is measured</summary>__METHOD__</details>
</div>
</main>
<script>
const PG = __PAGE__;
const S = {k: 0};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => v > 0 && v < 10 ** -(d + 2) ? `${(100 * v).toExponential(1)}%` : `${(100 * v).toFixed(d)}%`;
const share = v => `${+(100 * v).toPrecision(4)}%`;
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const nearest = (xs, v) => xs.reduce((best, x, i) => Math.abs(x - v) < Math.abs(xs[best] - v) ? i : best, 0);
const pow2 = q => 2 ** Math.max(1, Math.ceil(Math.log2(q)));
const mhz = f => sig(f * 1e3, 3);

// state in the address bar: #k=5
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (PG.ks.includes(+q.get("k"))) S.k = +q.get("k");
}
function writeHash() {
  history.replaceState(null, "", S.k ? `#k=${S.k}` : location.pathname + location.search);
}
addEventListener("hashchange", () => { readHash(); slider.value = PG.ks.indexOf(S.k); schedule(); });

function tip(host, ev, html) {
  const t = host.nextElementSibling, card = host.closest(".card").getBoundingClientRect();
  t.innerHTML = html; t.style.display = "block";
  t.style.left = Math.max(4, Math.min(ev.clientX - card.left + 14, card.width - t.offsetWidth - 8)) + "px";
  t.style.top = (ev.clientY - card.top + 14) + "px";
}
const untip = host => host.nextElementSibling.style.display = "none";
const trow = (color, text) => `<div class="row">${color ? `<i class="dot" style="background:${color};width:8px;height:8px"></i>` : ""}${text}</div>`;
const legendItem = (color, text, dash, width = 2) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""}/></svg>${text}</span>`;
const legendBox = (color, text) => `<span><svg width="22" height="10"><rect x="1" y="1" width="20" height="8" rx="2" fill="${color}" opacity="0.35" stroke="${color}"/></svg>${text}</span>`;
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
    if (L.image) s += `<image href="${L.image.url}" x="${X(L.image.x[0])}" y="${Y(L.image.y[1])}" width="${X(L.image.x[1]) - X(L.image.x[0])}" height="${Y(L.image.y[0]) - Y(L.image.y[1])}" preserveAspectRatio="none"/>`;
    if (L.span) s += `<rect x="${cx(L.span[0])}" width="${Math.max(1.5, cx(L.span[1]) - cx(L.span[0]))}" y="${m.t}" height="${H - m.t - m.b}" fill="${L.color}" opacity="${L.opacity ?? 0.1}"/>`;
    if (L.area && L.area.length) s += `<polygon points="${pts([[L.area[0][0], L.base ?? y0], ...L.area, [L.area.at(-1)[0], L.base ?? y0]])}" fill="${L.color}" opacity="${L.opacity ?? 1}"/>`;
    if (L.line && L.line.length) s += `<polyline points="${pts(L.line)}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 2}" stroke-linejoin="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.segs) s += `<path d="${L.segs.map(([a, b]) => `M${X(a[0]).toFixed(1)},${Y(a[1]).toFixed(1)}L${X(b[0]).toFixed(1)},${Y(b[1]).toFixed(1)}`).join("")}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 1.4}" stroke-linecap="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.hline !== undefined) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(L.hline)}" y2="${Y(L.hline)}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="${L.width ?? 1.2}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.vline !== undefined) s += `<line x1="${X(L.vline)}" x2="${X(L.vline)}" y1="${m.t}" y2="${H - m.b}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="${L.width ?? 1.2}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
  }
  s += "</g>";
  // labels outside the clip, so text at an edge stays readable
  for (const L of o.layers) {
    if (L.vline !== undefined && L.text) s += `<text class="lab" x="${X(L.vline) + (L.tdx ?? 0)}" y="${m.t + 12 + (L.dy ?? 0)}" text-anchor="${L.anchor ?? "middle"}">${L.text}</text>`;
    if (L.hline !== undefined && L.text) s += `<text class="lab" x="${L.left ? m.l + 4 : W - m.r - 4}" y="${cy(L.hline) - 4}" text-anchor="${L.left ? "start" : "end"}">${L.text}</text>`;
    if (L.line && L.text && L.line.length) { const p = L.line[L.textAt ?? L.line.length - 1]; s += `<text class="lab" x="${cx(p[0]) + (L.tdx ?? 4)}" y="${cy(p[1]) + (L.tdy ?? 4)}" text-anchor="${L.anchor ?? "start"}">${L.text}</text>`; }
    if (L.note) s += `<text class="lab" x="${cx(L.note[0])}" y="${cy(L.note[1]) - 5}" text-anchor="middle">${L.note[2]}</text>`;
  }
  s += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" visibility="hidden"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  // swap the svg in one go: an emptied host would collapse the page for a layout and jump its scroll mid-drag
  host.innerHTML = s + "</svg>";
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
const cut = () => PG.cuts[PG.ks.indexOf(S.k)];
const need = (x, y, c = cut()) => pow2(c.table[PG.keep.indexOf(x)][PG.probabilities.indexOf(y)]);
const counted = () => S.k ? `SNR ≥ ${S.k}` : "power above the noise";
// a colour on the sequential ramp at t in [0, 1]
function rampAt(t) {
  const ramp = css("--ramp").split(",").map(h => [1, 3, 5].map(j => parseInt(h.trim().slice(j, j + 2), 16)));
  const u = clamp(t, 0, 1) * (ramp.length - 1), i = Math.min(Math.floor(u), ramp.length - 2), f = u - i;
  return `rgb(${ramp[i].map((v, n) => Math.round(v + f * (ramp[i + 1][n] - v))).join(",")})`;
}
const powerColour = p => p === PG.R ? "var(--ink)" : rampAt((Math.log2(p) - 2) / 9);
// the x axis counts nines up to 99.99%, then jumps to x = 100% (every above-noise bin) at C.full
const xName = (i, C = PG.curve) => i === C.x.length - 1 ? "100%" : share(1 - 10 ** -C.x[i]);
const yName = t => t > 8 ? "100%" : pct(1 - 10 ** -t, clamp(Math.ceil(t) - 1, 0, 6));
const f0Ticks = () => PG.f_ticks.map(f => [Math.log10(f), sig(f * 1e3, 2)]);
const f0Bin = x => clamp(PG.f0.edges.findIndex(e => e > x) - 1, 0, PG.f0.edges.length - 2);
const f0Range = b => `f₀ ${mhz(10 ** PG.f0.edges[b])}–${mhz(10 ** PG.f0.edges[b + 1])} mHz`;

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
// set up once: writing the slider's value or max mid-drag drops the drag, and the hash is written on release only
const slider = document.getElementById("k");
slider.max = PG.ks.length - 1;
slider.addEventListener("input", () => { S.k = PG.ks[+slider.value]; schedule(); });
slider.addEventListener("change", writeHash);
dragSlider(slider);
function controls() {
  html("k-label", S.k ? `SNR ≥ ${S.k}` : "off");
  html("k-note", S.k ? `where <code>LisaGB.visible_snr = ${S.k}</code> centres training windows` : "every source with a bin above the noise");
  const T = cut(), tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  html("tiles", tile("sources counted", pct(T.mass, 2), `of the prior's, with ${counted()}`)
    + tile("99% of the power, 99% of sources", `R = ${need(0.99, 0.99)}`, `99.9% of the power, 99.9% of sources: R = ${need(0.999, 0.999)}`)
    + tile("all of it, 99.9% of sources", `R = ${need(1.0, 0.999)}`, `99% of sources: R = ${need(1.0, 0.99)}`)
    + tile("in use · 5σ hard bound", `${PG.R} · ${PG.bound.toLocaleString()}`, `${PG.config}'s <code>response_points</code>, and what autosizing gave`));
}

// one line per R: the share y of the counted sources that keep at least x of their above-noise power, both axes in nines
function keepCard() {
  const host = document.getElementById("p-keep"), C = {...PG.curve, ...cut()};
  // every R past the widest source keeps everything: draw the first of those only
  const whole = r => C.y[r].every(v => v >= C.top);
  const shown = PG.powers.map((p, r) => r).filter(r => r === 0 || !whole(r - 1));
  const layers = [{span: [C.x.at(-2), C.full], color: "var(--muted)", opacity: 0.08}];
  for (const r of shown) {
    const p = PG.powers[r], line = C.x.map((x, i) => [x, C.y[r][i]]), at = nearest(C.y[r].map(v => Math.abs(v - 0.55 * C.top)), 0);
    layers.push({line, color: powerColour(p), width: p === PG.R ? 3 : 1.8, text: String(p), textAt: at, tdx: 5, tdy: -2});
  }
  const ticks = [[0, "0%"], [Math.log10(2), "50%"], [1, "90%"], [2, "99%"], [3, "99.9%"], [4, "99.99%"]];
  chart(host, {x: [0, C.full], y: [0, C.top], H: 380, xlabel: "keeps at least x of its above-noise power", ylabel: `share y of the sources with ${counted()}`, label: "kept above-noise power against probability, one line per R",
    xticks: [...ticks.filter(([t]) => t <= C.x.at(-2)), [C.full, "100%"]], yticks: ticks.filter(([t]) => t <= C.top), xgrid: true, layers,
    hover: ({x}) => {
      const i = nearest(C.x, x);
      return {cross: C.x[i], html: `<b>keeps ≥ ${xName(i)} of its above-noise power</b>` + shown.map(r => trow(powerColour(PG.powers[r]), `R = ${PG.powers[r]}: ${yName(C.y[r][i])} of sources`)).join("")};
    }});
  html("lg-keep", shown.map(r => PG.powers[r]).map(p => legendItem(powerColour(p), p === PG.R ? `R = ${p} (in use)` : `R = ${p}`, null, p === PG.R ? 3 : 2)).join(""));
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
    const v = grid[(n - 1 - r) * n + i] / PG.density.scale, k = 4 * (r * n + i);
    for (let c = 0; c < 3; c++) img.data[k + c] = rgb[c];
    img.data[k + 3] = v > 0 ? Math.round(255 * 0.55 * v ** 0.6) : 0;
  }
  ctx.putImageData(img, 0, 0);
  return images[key + colour] = cv.toDataURL();
}

// the counted sources' joint density of f0 and full above-noise width, with the R references
function widthCard() {
  const host = document.getElementById("p-width"), T = cut(), D = PG.density, G = PG.grid;
  const n = D.n, dx = (D.x[1] - D.x[0]) / n, dy = (D.y[1] - D.y[0]) / n;
  const at = ([i, j]) => [D.x[0] + (i + 0.5) * dx, D.y[0] + (j + 0.5) * dy];
  const layers = [{image: {url: densityImage(T.density, n, css("--c1"), S.k), x: D.x, y: D.y}}];
  T.levels.forEach((level, m) => {
    const segs = contour(T.density, n, level).map(([p, q]) => [at(p), at(q)]);
    layers.push({segs, color: "var(--c1)", width: [2, 1.5, 1.1][m]});
    if (segs.length) { const top = segs.flat().reduce((a, b) => b[1] > a[1] ? b : a); layers.push({note: [top[0], top[1], share(D.masses[m])]}); }
  });
  layers.push({line: G.x.map((x, i) => [x, Math.log10(G.ceiling[i])]), color: "var(--ink-2)", width: 1.4, text: "5σ hard bound", textAt: Math.round(0.55 * G.x.length), tdx: -6, tdy: -8, anchor: "end"},
    {vline: Math.log10(PG.narrow.f0_max), color: "var(--muted)", width: 1.2, dash: "3 3", text: "narrow-B top", anchor: "end", tdx: -4},
    {hline: Math.log10(PG.narrow.R), color: "var(--muted)", width: 1.2, dash: "3 3", text: `narrow-B R = ${PG.narrow.R}`, left: true},
    {hline: Math.log10(PG.R), color: "var(--ink)", width: 1.4, dash: "6 3", text: `R in use = ${PG.R}`, left: true});
  chart(host, {x: D.x, y: D.y, H: 400, xlabel: "f₀ [mHz]", ylabel: "bins holding all of a source's above-noise power", label: "density of f0 and full above-noise width",
    xticks: f0Ticks(), yticks: PG.w_ticks.filter(w => Math.log10(w) <= D.y[1]).map(w => [Math.log10(w), String(w)]), layers,
    hover: ({x, y}) => {
      const b = f0Bin(x), i = clamp(Math.floor((x - D.x[0]) / dx), 0, n - 1), j = clamp(Math.floor((y - D.y[0]) / dy), 0, n - 1), g = nearest(G.x, x);
      return {cross: x, html: `<b>${f0Range(b)}: ~${T.n[b].toLocaleString()} effective sources with ${counted()}</b>`
        + trow("var(--c1)", `density at ${sig(10 ** y, 3)} bins: ${pct(T.density[j * n + i] / D.scale, 1)} of its peak`)
        + PG.probabilities.map((p, m) => { const v = T.q[m][b]; return trow(null, v === null ? `${share(p)} of them: too few sources` : `${share(p)} of them within ${v} bins → R = ${pow2(v)}`); }).join("")
        + trow("var(--ink-2)", `5σ hard bound here: ${sig(G.ceiling[g], 3)} bins`)};
    }});
  html("lg-width", legendBox("var(--c1)", `density of the sources with ${counted()}; contours hold ${D.masses.map(share).join(" / ")} of them`)
    + legendItem("var(--ink)", "R in use", "6 3") + legendItem("var(--muted)", "narrow-B's R and f₀ top", "3 3")
    + legendItem("var(--ink-2)", "response_span at the old 5σ ḟ ceiling (the hard bound)", null, 1.4));
  shareCard();
}

// per f0 bin, the share of the prior's sources and of the counted ones, on the density's f0 axis
function shareCard() {
  const host = document.getElementById("p-share"), F = PG.f0, T = cut();
  const steps = arr => arr.flatMap((v, b) => [[F.edges[b], v], [F.edges[b + 1], v]]);
  chart(host, {x: PG.density.x, y: [0, 1.15 * Math.max(...F.prior, ...T.seen)], H: 170, xlabel: "f₀ [mHz]", ylabel: "share of sources per f₀ bin",
    label: "f0 distribution of the prior and of the counted sources", xticks: f0Ticks(), yticks: [],
    layers: [{area: steps(T.seen), color: "var(--c1)", opacity: 0.18, base: 0}, {line: steps(T.seen), color: "var(--c1)", width: 1.8},
      {line: steps(F.prior), color: "var(--ink-2)", width: 1.4, dash: "5 3"}],
    hover: ({x}) => {
      const b = f0Bin(x);
      return {cross: x, html: `<b>${f0Range(b)}</b>` + trow("var(--ink-2)", `prior: ${pct(F.prior[b], 2)} of its sources`)
        + trow("var(--c1)", `with ${counted()}: ${pct(T.seen[b], 2)} of them`)};
    }});
  html("lg-share", legendBox("var(--c1)", `f₀ of the sources with ${counted()}`)
    + legendItem("var(--ink-2)", `prior's f₀ (the catalogue, <code>sample_f</code>)`, "5 3", 1.4));
}

function render() { controls(); keepCard(); widthCard(); }
let pending = 0;
const schedule = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; render(); }); };
addEventListener("resize", schedule);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
readHash();
slider.value = PG.ks.indexOf(S.k);
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


def share(v: float) -> str:
    return f"{100 * v:.4g}%"


def mhz(f: float) -> str:
    return sig(f * 1e3)


def pow2(q: float) -> int:
    """The smallest power of two holding q bins."""
    return 1 << max(1, math.ceil(math.log2(max(q, 2))))


def quantile(a: np.ndarray, w: np.ndarray, y: float) -> float:
    """The smallest value with at least a share y of the weight w at or below it."""
    order = np.argsort(a, kind="stable")
    cumulative = np.cumsum(w[order])
    at = np.searchsorted(cumulative, y * cumulative[-1] * (1 - 1e-12))
    return float(a[order][min(at, a.size - 1)])


def effective(w: np.ndarray) -> float:
    """Kish's effective sample size of the weights w."""
    return float(w.sum() ** 2 / (w**2).sum()) if w.sum() > 0 else 0.0


def rounded(a, digits: int = 4) -> list:
    """An array as a list of floats to `digits` significant figures, for the page's JSON."""
    return [float(f"{v:.{digits}g}") for v in np.asarray(a, dtype=float).ravel()]


def density(x, y, w, rx, ry) -> tuple[np.ndarray, list]:
    """A smoothed weighted 2-D histogram over CELLS², scaled to its peak, with the density levels holding MASSES of it."""
    h = np.histogram2d(x, y, CELLS, [rx, ry], weights=w)[0].T
    for axis in (0, 1):
        h = np.apply_along_axis(np.convolve, axis, h, SMOOTH, mode="same")
    ordered = np.sort(h.ravel())[::-1]
    held = np.cumsum(ordered) / ordered.sum()
    levels = [ordered[np.searchsorted(held, m)] / ordered[0] for m in MASSES]
    return h / ordered[0], levels


def main() -> None:
    d = json.loads((DATA / "response-span.json").read_text())
    z = np.load(DATA / "response-span.npz")
    R, P, X, J = d["response_points"], d["powers"], d["keep"], d["amplitudes"]
    bound = d["bound_points"]
    f_lo, f_hi = d["f0_range"]
    k99, k100 = X.index(0.99), X.index(1.0)

    # per source its prior weight above the noise; per A draw a share of it, every draw lying above the noise
    f0, snr_unit = z["f0"], z["snr_unit"]
    source_w = z["f0_weight"] * z["seen"]
    w = np.repeat(source_w / J, J)
    width, loss = z["width"].reshape(w.size, -1), z["loss"].reshape(w.size, -1)
    # cumsum round-off leaves ~1e-16 where R holds everything
    loss = np.where(loss < 1e-9, 0.0, loss)
    snr = 10.0 ** z["log_a"].ravel().astype(float) * np.repeat(snr_unit, J)
    seen_share, n_eff = float(source_w.mean()), effective(source_w)
    # the prior share at SNR >= 7, with A integrated in closed form
    snr7 = float(
        np.mean(
            z["f0_weight"]
            * ndtr((z["mean_log_a"] - np.log10(VISIBLE_SNR / snr_unit)) / d["amp_std"])
        )
    )
    edges = np.linspace(math.log10(f_lo), math.log10(f_hi), F0_BINS + 1)
    bins = np.clip(np.digitize(np.log10(f0), edges) - 1, 0, F0_BINS - 1)
    draw_bins = np.repeat(bins, J)
    f0_share = np.bincount(bins, z["f0_weight"], F0_BINS)
    # the density's axes: log10 f0, and log10 of the full width spread over its even-bin step
    density_x = [math.log10(f_lo), math.log10(f_hi)]
    density_y = [0.0, math.log10(bound)]
    log_f0 = np.repeat(np.log10(f0), J)
    log_width = np.log10(
        width[:, k100] + np.random.default_rng(0).uniform(-1, 1, w.size)
    )

    def population(counted: np.ndarray) -> dict:
        """The page's numbers over the counted A draws: R(x, y), the keep curves and the per-f0 width quantiles."""
        sw = np.where(counted, w, 0.0).reshape(-1, J).sum(axis=1)
        wc, width_c, loss_c, bins_c = (
            w[counted],
            width[counted],
            loss[counted],
            draw_bins[counted],
        )
        table = [
            [quantile(width_c[:, k], wc, y) for y in PROBABILITIES]
            for k in range(len(X))
        ]

        # each R's keep curve: -log10 of the weight that loses more than 1 - x, then x = 100%: the whole above-noise width inside R
        top = round(math.log10(effective(sw)) + 0.1, 2)
        kept = []
        for r, p in enumerate(P):
            order = np.argsort(loss_c[:, r], kind="stable")
            cumulative = np.concatenate([[0.0], np.cumsum(wc[order])]) / wc.sum()
            at = np.searchsorted(loss_c[order, r], 10.0**-CURVE_X, side="right")
            kept.append(
                np.append(cumulative[at], wc[width_c[:, k100] <= p].sum() / wc.sum())
            )
        curve_y = np.round(
            np.minimum(-np.log10(np.maximum(1 - np.array(kept), 1e-12)), top), 3
        ).tolist()

        # the y-quantile of each x-width per f0 bin, where at least one effective source lies above it
        n_bin = [effective(sw[bins == b]) for b in range(F0_BINS)]
        f0_q = [
            [
                [
                    (
                        int(quantile(width_c[bins_c == b, k], wc[bins_c == b], y))
                        if n_bin[b] * (1 - y) >= 1
                        else None
                    )
                    for b in range(F0_BINS)
                ]
                for y in PROBABILITIES
            ]
            for k in range(len(X))
        ]
        seen_f0_share = np.bincount(bins, sw, F0_BINS)
        h, levels = density(
            log_f0[counted], log_width[counted], wc, density_x, density_y
        )
        return dict(
            mass=float(f"{sw.mean():.4g}"),
            table=table,
            top=top,
            y=curve_y,
            n=[round(v) for v in n_bin],
            q=f0_q,
            seen=rounded(seen_f0_share / seen_f0_share.sum(), 6),
            density=np.round(h * DENSITY_SCALE).astype(int).ravel().tolist(),
            levels=[round(v * DENSITY_SCALE, 2) for v in levels],
        )

    cuts = [population(snr >= k) for k in CUTS]
    table, f0_q = cuts[0]["table"], cuts[0]["q"]
    need = lambda x, y, c=cuts[0]: pow2(c["table"][X.index(x)][PROBABILITIES.index(y)])

    grid = d["grid"]
    page = dict(
        R=R,
        bound=bound,
        config=d["config"],
        powers=P,
        keep=X,
        probabilities=list(PROBABILITIES),
        narrow=dict(f0_max=d["narrow_f0_max"], R=d["narrow_response_points"]),
        ks=list(CUTS),
        # the page reads the width quantiles at x = 100% only
        cuts=[{**c, "q": c["q"][k100]} for c in cuts],
        density=dict(
            n=CELLS,
            x=rounded(density_x, 6),
            y=rounded(density_y, 6),
            masses=list(MASSES),
            scale=DENSITY_SCALE,
        ),
        curve=dict(x=rounded(np.append(CURVE_X, FULL)), full=FULL),
        f_ticks=[
            f for f in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2] if f_lo <= f <= f_hi
        ],
        w_ticks=[2**e for e in range(1, 12)],
        grid=dict(x=rounded(np.log10(grid["f0"]), 6), ceiling=rounded(grid["ceiling"])),
        f0=dict(edges=rounded(edges, 6), prior=rounded(f0_share / f0_share.sum(), 6)),
    )

    # f0 ranges by the R that keeps 99% for 99% of the sources there
    runs = []
    for b, q in enumerate(f0_q[k99][PROBABILITIES.index(0.99)]):
        if q is None:
            continue
        if runs and runs[-1][0] == pow2(q):
            runs[-1][2] = b + 1
        else:
            runs.append([pow2(q), b, b + 1])
    regimes = "; ".join(
        f"{r} for {mhz(10 ** edges[b0])}–{mhz(10 ** edges[b1])} mHz ({pct(source_w[(bins >= b0) & (bins < b1)].sum() / source_w.sum(), 0)} of them)"
        for r, b0, b1 in runs
    )

    # narrow-B's R on the full-B sources above the noise below its top
    narrow = np.repeat(f0 < d["narrow_f0_max"], J)
    narrow_r = d["narrow_response_points"]
    narrow_loss, narrow_w = loss[narrow, P.index(narrow_r)], w[narrow]
    narrow_full = pow2(quantile(width[narrow, k100], narrow_w, 0.999))
    below7 = w[snr < VISIBLE_SNR].sum() / w.sum()

    keep_note = (
        f"Each line is one R: the share y of the counted sources that keep at least a share x of their above-noise power inside their R bins. "
        "Both axes count nines, so 99% and 99.9% sit a decade apart; the last x is 100%, every bin above the noise. "
        "Only the first R that keeps everything is drawn; wider ones lie on it."
    )
    width_note = (
        "Where the counted sources lie in f₀ and in the bins their whole above-noise power spans, with contours holding "
        + " / ".join(share(m) for m in MASSES)
        + " of them. A source fits R when it lies below the R line. Below, how those sources and the prior's spread over f₀."
    )
    method = "".join(
        f"<p>{t}</p>"
        for t in [
            f"A bin's signal power is 2|h|²/(S<sub>n</sub>·T/2) per A, E, T channel, in units where it sums to SNR², and the noise there has expected power {sig(d['noise_power'])}; "
            "the above-noise power is the signal power past that, summed over bins and channels. "
            f"Sources with none, which the data cannot see, never count: off counts the {pct(seen_share, 2)} of the prior's sources that have some.",
            "SNR ≥ k counts only the sources that also reach SNR k. "
            "A source's SNR is linear in A, and its A draws are stratified, so the cut filters them without bias. "
            "<code>LisaGB.visible_snr = k</code> (default 7) centres training windows by about the prior's f₀ times P(SNR ≥ k | f₀), as the page weights it, toward high f₀ where sources are wider; "
            "the sources in those windows stay the catalogue's, and the fainter ones are narrower, so at each f₀ they need at most the R shown. "
            "The dashed curve is the catalogue's f₀ (<code>sample_f</code>).",
            f"R(x, y) is the next power of two above the y-quantile of the narrowest centred band, in bins of 1/T, holding a share x of the above-noise power. "
            f"The tooltip's quantiles are per f₀ bin ({F0_BINS} log bins), left out where fewer than 1/(1 − y) effective sources lie above them. "
            "Widths come in even bins; the density spreads each over its step before a one-cell smoothing.",
            "The hard bound is <code>response_span</code> at the ḟ ceiling the sampler used to cut at (5σ above the mean log ḟ), which autosizing read R off at the band's top; "
            "neither the draws here nor the sampler cut ḟ.",
        ]
    )
    meaning = [
        f"<b>Keeping 99% of what the data see needs {bound // need(0.99, 0.99)}× fewer bins than the hard bound.</b> R = {need(0.99, 0.99)} keeps 99% of the above-noise power for 99% of the sources, "
        f"against the hard bound's {bound}; 99.9% for 99.9% takes R = {need(0.999, 0.999)}. {d['config']} uses R = {R}.",
        f"<b>Keeping all of it costs little more.</b> Every above-noise bin fits in R = {need(1.0, 0.99)} for 99% of the sources and {need(1.0, 0.999)} for 99.9%; "
        f"99.99% of them span at most {sig(quantile(width[:, k100], w, 0.9999), 4)} bins. Leakage below the noise does not count, so x = 100% is finite.",
        f"<b>R follows f₀.</b> Keeping 99% for 99% of the sources at each f₀ takes R = {regimes}"
        + (
            f"; above {mhz(10 ** edges[runs[-1][2]])} mHz too few sources lie for a 99% quantile per bin."
            if runs and runs[-1][2] < F0_BINS
            else "."
        ),
        f"<b>narrow-B's R = {narrow_r}.</b> Below its {mhz(d['narrow_f0_max'])} mHz top ({pct(narrow_w.sum() / w.sum(), 1)} of the sources above the noise), "
        f"R = {narrow_r} loses {pct(quantile(narrow_loss, narrow_w, 0.5), 2)} of the above-noise power at the median, {pct(quantile(narrow_loss, narrow_w, 0.99), 2)} for 99% of sources "
        f"and {pct(quantile(narrow_loss, narrow_w, 0.999), 2)} for 99.9%; keeping all of it for 99.9% of them takes {narrow_full}.",
        f"<b>Which sources count.</b> {pct(seen_share, 2)} of the prior's sources have a bin above the noise, from SNR {sig(float(snr[w > 0].min()), 2)} up (median {sig(quantile(snr, w, 0.5), 2)}); "
        f"{pct(below7, 0)} of them stay below SNR {sig(VISIBLE_SNR)}, and {pct(snr7, 2)} of the prior's reach it.",
        "<b>A higher SNR cut widens the tail.</b> Brighter sources lift more sidelobes above the noise. Per k, the counted share of the prior's sources, then R(99%, 99%) / R(100%, 99.9%): "
        + "; ".join(
            f"{f'k = {k}' if k else 'off'}: {pct(c['mass'], 2)}, {need(0.99, 0.99, c)} / {need(1.0, 0.999, c)}"
            for k, c in zip(CUTS, cuts)
            if k in SUMMARY_CUTS
        )
        + ".",
    ]
    sub = (
        f"Autosizing gave every {d['config']} source R = <code>response_points</code> = {bound} bins, the next power of two above "
        f"<code>response_span</code> of the fastest chirp the sampler then allowed (ḟ cut 5σ above its mean) at the top of f₀, {mhz(f_hi)} mHz: a hard bound, {sig(d['span_top'], 4)} bins. "
        "This page asks instead for a fuzzy one per source: the smallest R that keeps at least a share x of its power above the noise inside its R bins, with probability y under the prior, ḟ uncut. "
        f"Measured on {d['sources']:,} sources, f₀ log-uniform over the band and weighted back to the prior ({sig(n_eff, 3)} effective sources above the noise), each against a {d['reference_points']:,}-bin jaxgb reference; "
        f"A enters linearly, so it is integrated out: the share of A above the noise in closed form, {J} stratified draws inside it. "
        f"Built {date.today().isoformat()} from <code>data/</code> (commit <code>{d['commit'][:7]}</code>{', src dirty' if d['src_dirty'] else ''})."
    )
    print(
        "R(x, y):",
        {x: [pow2(v) for v in table[k]] for k, x in enumerate(X)},
        "\nquantile widths:",
        {x: table[k] for k, x in enumerate(X)},
        "\n99.9% loss per R:",
        {p: round(quantile(loss[:, r], w, 0.999), 6) for r, p in enumerate(P)},
        "\nmedian loss per R:",
        {p: round(quantile(loss[:, r], w, 0.5), 6) for r, p in enumerate(P)},
        f"\nabove the noise {seen_share:.4g}, SNR >= 7 {snr7:.4g}, effective sources {n_eff:.0f}",
    )
    for t in meaning:
        print("-", t)

    out_html = PAGE
    for key, value in dict(
        SUB=sub,
        N_KEEP=keep_note,
        N_WIDTH=width_note,
        MEANING="".join(f"<li>{t}</li>" for t in meaning),
        METHOD=method,
        PAGE=json.dumps(page, separators=(",", ":")),
    ).items():
        out_html = out_html.replace(f"__{key}__", value)
    out = FOLDER / "response-span.html"
    out.write_text(out_html)
    print(f"wrote {out} ({len(out_html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
