"""Write sky.html from data/prior.json: LisaGB's galactic sky prior in the galactic, ecliptic and equatorial frames.

Standard library only; run from the repo root:
    uv run --frozen python reports/prior/scripts/build_sky.py
"""

import json
import math
from datetime import date
from pathlib import Path

HERE = Path(__file__).parents[1]
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
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --bar: #52514e;
    --q0: #1a1a19; --q1: #0d366b; --q2: #184f95; --q3: #3987e5; --q4: #cde2fb;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --bar: #52514e;
  --q0: #1a1a19; --q1: #0d366b; --q2: #184f95; --q3: #3987e5; --q4: #cde2fb;
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
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; }
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
<p class="nav">The LisaGB prior by parameter group: <a data-page="frequency.html" href="frequency.html">frequency</a> · <a data-page="amplitude.html" href="amplitude.html">amplitude</a> · <b>sky</b></p>
<h1>The LisaGB prior: sky</h1>
<p class="sub">Where on the sky a source sits: the galactic mixture of <code>sample_sky</code> (a bulge, thick and thin disk and an isotropic floor) in the galactic, ecliptic and equatorial frames, against the quadrature density, over 8192 drawn windows of four sources per band; then (l, b) in the flow's coordinates. The bands share the sky draws, so the band picker changes only the band carried to the other pages. __SUB__</p>
<div class="row"><span class="k">Band</span><span class="seg" id="band"></span><span class="k">Frame</span><span class="seg" id="frame"></span></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>Where on the sky does the prior put sources?</h2>
  <p class="note" id="n-sky"></p>
  <div class="sky" id="p-sky"><canvas width="960" height="480"></canvas><svg viewBox="0 0 960 480" role="img" aria-label="sky prior density"></svg></div><div class="tip"></div>
  <div id="ramp" style="margin-top:6px"></div>
</div>
<div class="card">
  <h2>Do the draws match the density in every frame?</h2>
  <p class="note" id="n-marg"></p>
  <div class="legend" id="lg-marg"></div>
  <div class="pair"><div><div id="p-lat"></div><div class="tip"></div></div><div><div id="p-lon"></div><div class="tip"></div></div></div>
</div>
<div class="card">
  <h2>How far from the galactic centre do sources sit?</h2>
  <p class="note" id="n-dist"></p>
  <div class="legend" id="lg-dist"></div>
  <div id="p-dist"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What do l and b look like in the flow's coordinates?</h2>
  <p class="note" id="n-flow"></p>
  <div class="legend" id="lg-flow"></div>
  <div id="p-flow"></div>
</div>
<div class="card meaning">
  <h2>What this means</h2>
  <ul id="meaning"></ul>
</div>
</main>
<script>
const PG = __PAGE__;
const DEG = Math.PI / 180, TAU = 2 * Math.PI, deg = v => v / DEG;
const FRAMES = [["galactic", "galactic (l, b)", "l", "b"], ["ecliptic", "ecliptic (λ, β)", "λ", "β"], ["equatorial", "equatorial (RA, Dec), jaxgb's", "RA", "Dec"]];
const CHOICES = {band: ["narrow", "full"], frame: FRAMES.map(f => f[0])};
const S = {band: "full", frame: "galactic", theme: ""};
const B = () => PG.bands[S.band];
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sum = a => a.reduce((x, y) => x + y, 0);
const sup = e => String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const marks = {};
function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; marks[key](); schedule(); }));
  marks[key]();
}
seg("band", [["narrow", "narrow-B"], ["full", "full-B"]], "band");
seg("frame", FRAMES.map(([k, n]) => [k, n.replace(/ \(.*/, "")]), "frame");

// state in the address bar: #band=full&frame=galactic (and theme=light|dark)
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  for (const [k, vs] of Object.entries(CHOICES)) if (vs.includes(q.get(k))) S[k] = q.get(k);
  if (["light", "dark"].includes(q.get("theme"))) { S.theme = q.get("theme"); document.documentElement.dataset.theme = S.theme; }
  Object.values(marks).forEach(m => m());
}
function writeHash() {
  history.replaceState(null, "", "#" + new URLSearchParams(Object.entries(S).filter(([, v]) => v)).toString());
  document.querySelectorAll(".nav a").forEach(a => a.href = `${a.dataset.page}#band=${S.band}${S.theme ? `&theme=${S.theme}` : ""}`);
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
    if (L.bars) L.bars.v.forEach((v, i) => { if (v > 0) { const a = X(L.bars.e[i]) + 1, b = X(L.bars.e[i + 1]) - 1; s += `<rect x="${a.toFixed(1)}" y="${Y(v).toFixed(1)}" width="${Math.max(b - a, 0.5).toFixed(1)}" height="${(Y(Math.max(y0, 0)) - Y(v)).toFixed(1)}" fill="${L.color}" rx="1"/>`; } });
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
const GRIDS = Object.fromEntries(Object.entries(PG.sky.frames).map(([k, f]) => [k, Uint8Array.from(atob(f.map), ch => ch.charCodeAt(0))]));
const log10At = q => MAP.log10_lo + q / 255 * (MAP.log10_hi - MAP.log10_lo);
function rampLut() {
  const stops = [0, 1, 2, 3, 4].map(i => { const h = css(`--q${i}`); return [1, 3, 5].map(j => parseInt(h.slice(j, j + 2), 16)); });
  const lut = new Uint8ClampedArray(256 * 3);
  for (let q = 0; q < 256; q++) {
    const u = q / 255 * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(u)), w = u - i;
    for (let c = 0; c < 3; c++) lut[3 * q + c] = stops[i][c] * (1 - w) + stops[i + 1][c] * w;
  }
  return lut;
}
// inverse projection of a pixel to plot (λ, φ) and the frame's map cell under it
function cellAt(grid, px, py) {
  const x = (px - SW / 2) / (SW / 2 - 2) * 2 * Math.SQRT2, y = -(py - SH / 2) / (SH / 2 - 2) * Math.SQRT2;
  if (x * x / 8 + y * y / 2 > 1) return null;
  const t = Math.asin(clamp(y / Math.SQRT2, -1, 1));
  const phi = Math.asin(clamp((2 * t + Math.sin(2 * t)) / Math.PI, -1, 1));
  const lambda = Math.PI * x / (2 * Math.SQRT2 * Math.cos(t));
  if (Math.abs(lambda) > Math.PI) return null;
  const col = Math.min(MAP.width - 1, Math.floor((lambda + Math.PI) / TAU * MAP.width));
  const rw = Math.min(MAP.height - 1, Math.floor((Math.PI / 2 - phi) / Math.PI * MAP.height));
  return {lambda, phi, q: grid[rw * MAP.width + col]};
}
const skyImages = {};
function skyPixels(frame) {
  const key = frame + css("--q4");
  if (skyImages[key]) return skyImages[key];
  const img = new ImageData(SW, SH), lut = rampLut(), grid = GRIDS[frame];
  for (let py = 0; py < SH; py++) for (let px = 0; px < SW; px++) {
    const c = cellAt(grid, px + 0.5, py + 0.5); if (!c) continue;
    const k = 4 * (py * SW + px);
    img.data[k] = lut[3 * c.q]; img.data[k + 1] = lut[3 * c.q + 1]; img.data[k + 2] = lut[3 * c.q + 2]; img.data[k + 3] = 255;
  }
  return skyImages[key] = img;
}

function tiles() {
  const P = PG.sky.abs_lat_below_10deg, W = PG.sky.within_rad_of_centre, fl = B().flow;
  document.getElementById("tiles").innerHTML = [
    ["|b| < 10°, galactic", pct(P.galactic), "the plane holds the draws"],
    ["|β| < 10°, ecliptic", pct(P.ecliptic), "the plane crosses the ecliptic at 60°"],
    ["|Dec| < 10°, equatorial", pct(P.equatorial), "jaxgb's frame"],
    ["within 0.5 rad of the centre", pct(W["0.5"], 0), `${pct(W["0.1"], 0)} within 0.1 rad`],
    ["l / b beyond the reach", fl.map(f => pct(f.beyond_reach, 2)).join(" / "), `of ±${fl[0].reach[1]} flow units`],
  ].map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}

function skyCard() {
  const [frame, fname] = FRAMES.find(f => f[0] === S.frame), F = PG.sky.frames[frame], grid = GRIDS[frame];
  const host = document.getElementById("p-sky"), svg = host.querySelector("svg");
  host.querySelector("canvas").getContext("2d").putImageData(skyPixels(frame), 0, 0);
  // graticule every 60° in longitude and 30° in latitude, longitude increasing to the left
  const plotLon = lon => wrap(F.lon0 - lon), shownLon = l => { const v = ((l % 360) + 360) % 360; return frame === "galactic" && v > 180 ? v - 360 : v; };
  const path = ll => "M" + ll.map(([l, p]) => toPx(mollweide(l, p)).map(v => v.toFixed(1)).join(",")).join("L");
  let s = "";
  for (let k = -2; k <= 2; k++) { const p = []; for (let i = 0; i <= 90; i++) p.push([k * Math.PI / 3 - 1e-9 * Math.sign(k), -Math.PI / 2 + i * Math.PI / 90]); s += `<path d="${path(p)}" fill="none" stroke="var(--axis)" stroke-opacity="0.5"/>`; }
  for (let k = -2; k <= 2; k++) { const p = []; for (let i = 0; i <= 120; i++) p.push([-Math.PI + i * Math.PI / 60, k * Math.PI / 6]); s += `<path d="${path(p)}" fill="none" stroke="var(--axis)" stroke-opacity="0.5"/>`; }
  s += `<ellipse cx="${SW / 2}" cy="${SH / 2}" rx="${SW / 2 - 2}" ry="${SH / 2 - 2}" fill="none" stroke="var(--axis)"/>`;
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
    const r = svg.getBoundingClientRect(), c = cellAt(grid, (ev.clientX - r.left) / r.width * SW, (ev.clientY - r.top) / r.height * SH);
    if (!c) return untip(host);
    tip(host, ev, `<b>(${shownLon(deg(F.lon0 - c.lambda)).toFixed(1)}°, ${deg(c.phi).toFixed(1)}°)</b>` + trow(null, `p<sub>Ω</sub> ≈ ${sig(10 ** log10At(c.q), 2)} sr⁻¹`));
  };
  hit.onpointerleave = () => untip(host);
  // the ramp's scale
  const lo = MAP.log10_lo, hi = MAP.log10_hi, RW = 300, RX = v => 8 + (v - lo) / (hi - lo) * (RW - 16);
  let k = `<svg width="${RW}" height="30" viewBox="0 0 ${RW} 30" style="width:${RW}px;max-width:100%;display:inline-block;vertical-align:middle"><defs><linearGradient id="ramp-fill">` + [0, 1, 2, 3, 4].map(i => `<stop offset="${i * 25}%" style="stop-color:var(--q${i})"/>`).join("") + `</linearGradient></defs><rect x="8" y="2" width="${RW - 16}" height="10" rx="2" fill="url(#ramp-fill)" stroke="var(--border)"/>`;
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) k += `<line x1="${RX(e)}" x2="${RX(e)}" y1="12" y2="16" stroke="var(--axis)"/><text x="${RX(e)}" y="27" text-anchor="middle">${logLabel(e)}</text>`;
  html("ramp", k + `</svg> <span class="note">p<sub>Ω</sub> per steradian, log scale</span>`);
  const c = PG.sky.components, P = PG.sky.abs_lat_below_10deg[frame], latName = FRAMES.find(f => f[0] === frame)[3];
  html("n-sky", `Density per steradian of the prior (${c.names.map((n, i) => `${n} ${pct(c.weights[i])}`).join(", ")}) in the ${fname} frame, Mollweide, longitude increasing to the left. ${pct(P)} of draws have |${latName}| &lt; 10° here.${frame === "equatorial" ? " jaxgb reads the sky in this frame; the retired Gaussian proxy sat at the galactic centre's ecliptic coordinates read as (RA, Dec), 23° off." : frame === "ecliptic" ? " LISA's orbit lies in this frame's plane; the galactic plane crosses it at about 60°." : ""}`);
}

function margCard() {
  const [frame, , lonName, latName] = FRAMES.find(f => f[0] === S.frame), M = PG.sky.marginals[frame];
  [["lat", latName], ["lon", lonName]].forEach(([k, name]) => {
    const host = document.getElementById(`p-${k}`), h = M[k], e = h.edges, x = centres(e);
    host.innerHTML = "";
    chart(host, {H: 210, x: [e[0], e[e.length - 1]], y: [0, Math.max(...h.counts, ...h.reference) * 1.1], xn: 4, yn: 3, xlabel: `${name} [deg]`, ylabel: "sources", label: `sky ${k} marginal`,
      layers: [{bars: {e, v: h.counts}, color: "var(--bar)"}, {line: x.map((v, i) => [v, h.reference[i]]), color: "var(--ink-2)", width: 1.5, dash: "4 3"}],
      hover: ({x: v}) => { const i = nearest(x, v); return {cross: x[i], html: `<b>${name} ${sig(e[i])}° – ${sig(e[i + 1])}°</b>` + trow("var(--bar)", `${h.counts[i].toLocaleString()} drawn, quadrature ${sig(h.reference[i], 4)}`)}; }});
  });
  const chi = PG.sky.chi2_per_bin, all = Object.values(chi).flatMap(f => Object.values(f)), out = M.lat.outside;
  html("lg-marg", legendBox("var(--bar)", `${(sum(M.lat.counts) + out).toLocaleString()} drawn sources`) + legendItem("var(--ink-2)", "quadrature of <code>sky_log_density</code>", "4 3"));
  html("n-marg", `The drawn sources' latitude and longitude in the ${S.frame} frame against the quadrature marginals of the density on the map. χ² per bin is ${chi[S.frame].lat.toFixed(2)} (${latName}) and ${chi[S.frame].lon.toFixed(2)} (${lonName}) here, ${Math.min(...all).toFixed(2)}–${Math.max(...all).toFixed(2)} across the frames.${out ? ` The galactic latitude histogram spans ±30°; ${out.toLocaleString()} draws (${pct(out / (sum(M.lat.counts) + out))}) lie beyond.` : ""}`);
}

function distCard() {
  const host = document.getElementById("p-dist"), d = PG.sky.from_centre, n = d.length, W = PG.sky.within_rad_of_centre;
  host.innerHTML = "";
  const line = d.map((v, i) => [v, (i + 1) / n]), rec = Object.entries(W).map(([r, v]) => [+r, v]);
  chart(host, {x: [0, Math.PI], y: [0, 1], xlabel: "angle from the galactic centre [rad]", ylabel: "share of sources within", label: "distance from the galactic centre", H: 260,
    layers: [{line, color: "var(--ink)", width: 2}, {line: rec, color: "var(--c2)", width: 0, marks: true}],
    hover: ({x}) => { const j = d.findIndex(v => v > x), i = j < 0 ? n - 1 : j; return {cross: d[i], html: `<b>${d[i].toFixed(2)} rad (${deg(d[i]).toFixed(0)}°)</b>` + trow("var(--ink)", `${pct((i + 1) / n)} of the scatter within`)}; }});
  html("lg-dist", legendItem("var(--ink)", `${n.toLocaleString()} prior sources`) + legendDot("var(--c2)", "every draw, at 0.1, 0.2, 0.3 and 0.5 rad"));
  html("n-dist", `The cumulative share of draws within an angle of the galactic centre. The bulge puts ${pct(W["0.1"], 0)} within 0.1 rad, but the disk spreads the rest along the plane: only ${pct(W["0.5"], 0)} lie within 0.5 rad, so a quiet source's sky posterior is mostly prior.`);
}

// the band's flow sky coordinates, one row each, against the network's reach
function flowCard() {
  const host = document.getElementById("p-flow"), fl = B().flow, gs = PG.sky.galactic_scale;
  const lo = -6, hi = 6, W = Math.max(320, host.clientWidth), row = 48, t = 4, l = 64, r = 16, H = t + row * fl.length + 30;
  const X = v => l + (clamp(v, lo, hi) - lo) / (hi - lo) * (W - l - r);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="flow sky coordinates of the prior">`;
  niceTicks(lo, hi, 8).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${t}" y2="${t + fl.length * row}" stroke="var(--grid)"/><text x="${X(v)}" y="${t + fl.length * row + 16}" text-anchor="middle">${v}</text>`);
  fl.forEach((f, i) => {
    const y = t + i * row, dd = density(f.hist), top = Math.max(...dd.v), e = f.hist.edges;
    s += `<rect x="${X(f.reach[0])}" width="${X(f.reach[1]) - X(f.reach[0])}" y="${y + 2}" height="${row - 6}" fill="var(--grid)" opacity="0.45"/>`;
    dd.v.forEach((v, k) => { if (v > 0) s += `<rect x="${X(e[k]) + 0.5}" width="${Math.max(0.5, X(e[k + 1]) - X(e[k]) - 1)}" y="${y + row - 4 - v / top * (row - 10)}" height="${v / top * (row - 10)}" fill="var(--bar)"/>`; });
    s += `<text class="lbl" x="${l - 8}" y="${y + row / 2 + 2}" text-anchor="end">${f.name}</text><text x="${l - 8}" y="${y + row / 2 + 14}" text-anchor="end">std ${f.std.toFixed(2)}</text>`;
  });
  s += `<text x="${(l + W - r) / 2}" y="${H - 2}" text-anchor="middle">flow coordinate</text>`;
  host.innerHTML = s + "</svg>";
  html("lg-flow", legendBox("var(--bar)", "prior") + legendBox("var(--grid)", "network's reach"));
  html("n-flow", `The flow sees galactic (l, b) divided by the prior's galactic scales, ${gs[0].toFixed(2)} and ${gs[1].toFixed(2)} rad, so both come out whitened (std ${fl.map(f => f.std.toFixed(2)).join(" and ")}). The disk's long tails reach past ±${fl[0].reach[1]}: ${fl.map(f => `${pct(f.beyond_reach, 2)} (${f.name})`).join(" and ")} of draws lie beyond the network's reach.`);
}

function meaning() {
  const P = PG.sky.abs_lat_below_10deg, W = PG.sky.within_rad_of_centre, fl = B().flow;
  html("meaning", [
    `<b>The galactic sky prior traces the plane, not a point.</b> ${pct(P.galactic)} of draws have |b| &lt; 10°, but only ${["0.1", "0.2", "0.3", "0.5"].map(k => pct(W[k], 0)).join(", ")} lie within 0.1, 0.2, 0.3 and 0.5 rad of the centre, so a quiet source's sky posterior is mostly prior.`,
    `<b>Other frames see the plane tilted.</b> Only ${pct(P.ecliptic)} of draws have |β| &lt; 10° and ${pct(P.equatorial)} |Dec| &lt; 10°: in jaxgb's equatorial frame the galaxy is a band that crosses the equator.`,
    `<b>The draws match the density in every frame</b> (χ² per bin ${Math.min(...Object.values(PG.sky.chi2_per_bin).flatMap(f => Object.values(f))).toFixed(2)}–${Math.max(...Object.values(PG.sky.chi2_per_bin).flatMap(f => Object.values(f))).toFixed(2)}).`,
    `<b>l and b reach the flow whitened</b> (std ${fl.map(f => `${f.std.toFixed(2)} (${f.name})`).join(" and ")}).`,
  ].map(t => `<li>${t}</li>`).join(""));
}

function render() { tiles(); skyCard(); margCard(); distCard(); flowCard(); meaning(); writeHash(); }
let pending = 0;
const schedule = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; render(); }); };
addEventListener("resize", schedule);
// the sky map's ramp is read from the theme's tokens
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
new MutationObserver(schedule).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
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


def chi2_per_bin(h: dict) -> float:
    """χ² of the drawn counts against the quadrature's, per bin with a nonzero reference."""
    pairs = [(c, r) for c, r in zip(h["counts"], h["reference"]) if r > 0]
    return sum((c - r) ** 2 / r for c, r in pairs) / len(pairs)


def page_data(prior: dict) -> dict:
    """What the page plots: the sky once (the bands share its draws), the flow's l and b per band."""
    b = prior["bands"]["full"]
    sc = b["scatter"]
    sky = dict(
        map=prior["sky"]["map"],
        frames={k: {kk: f[kk] for kk in ("lon0", "centre", "integral", "map")} for k, f in prior["sky"]["frames"].items()},
        components={k: prior["sky"]["components"][k] for k in ("names", "weights")},
        galactic_scale=prior["sky"]["galactic_scale"],
        marginals=b["sky"]["marginals"],
        abs_lat_below_10deg=b["sky"]["abs_lat_below_10deg"],
        within_rad_of_centre=b["sky"]["within_rad_of_centre"],
        chi2_per_bin={frame: {k: chi2_per_bin(h) for k, h in m.items()} for frame, m in b["sky"]["marginals"].items()},
        from_centre=sorted(
            math.acos(max(-1.0, min(1.0, math.cos(lat) * math.cos(lon))))
            for lon, lat in zip(sc["gal_l"], sc["gal_b"])
        ),
    )
    bands = {
        band: dict(
            flow=[
                {k: c[k] for k in ("name", "hist", "reach", "std", "beyond_reach")}
                for c in prior["bands"][band]["flow"]
                if c["name"] in ("l", "b")
            ]
        )
        for band in BANDS
    }
    return rounded(dict(sky=sky, bands=bands, old_proxy=list(OLD_PROXY_RADEC)))


def report(data: dict, prior: dict) -> None:
    """Print every number the README's sky takeaways quote."""
    sky = data["sky"]
    shared = all(prior["bands"][band]["sky"] == prior["bands"]["full"]["sky"] for band in BANDS)
    print(f"bands share the sky draws: {shared}")
    for frame, share in sky["abs_lat_below_10deg"].items():
        chi = sky["chi2_per_bin"][frame]
        print(
            f"{frame}: |lat| < 10° {share:.4f}; chi2/bin lat {chi['lat']:.3f}, lon {chi['lon']:.3f};"
            f" map integral {sky['frames'][frame]['integral']:.5f}"
        )
    print("within rad of the centre: " + ", ".join(f"{r} {v:.3f}" for r, v in sky["within_rad_of_centre"].items()))
    for band, b in data["bands"].items():
        print(f"{band}: " + "; ".join(f"{c['name']} std {c['std']:.3f}, beyond reach {c['beyond_reach']:.4f}" for c in b["flow"]))


def main() -> None:
    prior = json.loads((HERE / "data" / "prior.json").read_text())
    data = page_data(prior)
    page = PAGE.replace("__PAGE__", json.dumps(data, separators=(",", ":"))).replace(
        "__SUB__", f"Built {date.today().isoformat()} from <code>data/prior.json</code>."
    )
    out = HERE / "sky.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")
    report(data, prior)


if __name__ == "__main__":
    main()
