"""Write frequency.html from data/prior.json: LisaGB's prior on f₀, M_c and the drift, and where the bands sit in the noise.

Standard library only; run from the repo root:
    uv run --frozen python reports/prior/scripts/build_frequency.py
"""

import json
import math
from datetime import date
from pathlib import Path

HERE = Path(__file__).parents[1]
BANDS = ("narrow", "full")
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior: frequency</title>
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
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --bar: #52514e;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --bar: #52514e;
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
<p class="nav">The LisaGB prior by parameter group: <b>frequency</b> · <a data-page="amplitude.html" href="amplitude.html">amplitude</a> · <a data-page="sky.html" href="sky.html">sky</a></p>
<h1>The LisaGB prior: frequency</h1>
<p class="sub">Where a source sits in frequency and how it moves: f₀, the chirp mass M<sub>c</sub> and the drift ḟT² they set, over 8192 drawn windows of four sources per band, against the closed-form prior where there is one; then where the bands sit in the noise, how much one run draws, and f₀ and M<sub>c</sub> in the flow's coordinates. __SUB__</p>
<div class="row"><span class="k">Band</span><span class="seg" id="band"></span></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>What does the prior put on f₀, M<sub>c</sub> and the drift?</h2>
  <span class="seg" id="view"></span>
  <p class="note" id="n-view"></p>
  <div class="legend" id="lg-view"></div>
  <div id="p-view"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Where do the bands sit in the noise?</h2>
  <p class="note" id="n-psd"></p>
  <div class="legend" id="lg-psd"></div>
  <div id="p-psd"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How much does one training run draw?</h2>
  <p class="note" id="n-run"></p>
  <div class="tablewrap"><table id="t-run"></table></div>
</div>
<div class="card">
  <h2>What do f₀ and M<sub>c</sub> look like in the flow's coordinates?</h2>
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
const VIEWS = [["f0", "f₀"], ["mc", "M<sub>c</sub>"], ["drift", "drift ḟT²"]];
const CHOICES = {band: ["narrow", "full"], view: VIEWS.map(v => v[0])};
const S = {band: "full", view: "f0", theme: ""};
const B = () => PG.bands[S.band];
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sum = a => a.reduce((x, y) => x + y, 0);
const sup = e => String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
const big = v => { if (v < 1e5) return v.toLocaleString(); const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(2)}×10${sup(e)}`; };

const marks = {};
function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; marks[key](); schedule(); }));
  marks[key]();
}
seg("band", [["narrow", "narrow-B"], ["full", "full-B"]], "band");
seg("view", VIEWS, "view");

// state in the address bar: #band=full&view=f0 (and theme=light|dark)
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
  const cx = v => clamp(X(v), m.l, W - m.r);
  for (const L of o.layers) {
    if (L.span) s += `<rect x="${cx(L.span[0])}" width="${Math.max(1.5, cx(L.span[1]) - cx(L.span[0]))}" y="${m.t}" height="${H - m.t - m.b}" fill="${L.color}" opacity="${L.opacity ?? 0.1}"/>` + (L.text ? `<text class="lab" x="${L.textEnd ? cx(L.span[1]) - 4 : cx(L.span[0]) + 4}" y="${m.t + 12 + (L.dy ?? 0)}"${L.textEnd ? ' text-anchor="end"' : ""}>${L.text}</text>` : "");
    if (L.bars) L.bars.v.forEach((v, i) => { if (v > 0) { const a = X(L.bars.e[i]) + 1, b = X(L.bars.e[i + 1]) - 1; s += `<rect x="${a.toFixed(1)}" y="${Y(v).toFixed(1)}" width="${Math.max(b - a, 0.5).toFixed(1)}" height="${(Y(Math.max(y0, 0)) - Y(v)).toFixed(1)}" fill="${L.color}" rx="1"/>`; } });
    if (L.line) s += `<polyline points="${pts(L.line)}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 2}" stroke-linejoin="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.vline !== undefined) s += `<line x1="${X(L.vline)}" x2="${X(L.vline)}" y1="${m.t}" y2="${H - m.b}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="${L.width ?? 1.2}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>` + (L.text ? `<text class="lab" x="${X(L.vline)}" y="${m.t - 4 + (L.dy ?? 0)}" text-anchor="${L.anchor ?? "middle"}">${L.text}</text>` : "");
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
// a histogram's density per unit x
const density = h => { const w = h.edges[1] - h.edges[0], n = sum(h.counts) + (h.outside ?? 0); return {w, n, v: h.counts.map(c => c / n / w), ref: h.reference ? h.reference.map(c => c / n / w) : null}; };
const driftOver1 = b => { const h = b.frequency["log10 fdot [Hz/s]"], s = Math.log10(b.problem.t_obs ** 2); return sum(h.counts.filter((c, i) => (h.edges[i] + h.edges[i + 1]) / 2 + s > 0)) / (sum(h.counts) + (h.outside ?? 0)); };
// one parameter's histogram with the closed-form prior where there is one
function histChart(host, h, o) {
  const dd = density(h), e = h.edges.map(o.shift ? v => v + o.shift : v => v), x = centres(e);
  const top = Math.max(...dd.v, ...(dd.ref ?? [0]));
  const layers = [...(o.under ?? []), {bars: {e, v: dd.v}, color: "var(--bar)"}];
  if (dd.ref) layers.push({line: x.map((v, i) => [v, dd.ref[i]]), color: "var(--ink-2)", width: 1.5, dash: "4 3"});
  const vs = (o.under ?? []).filter(L => L.vline !== undefined).map(L => L.vline);
  const xr = [Math.min(e[0], ...vs), Math.max(e[e.length - 1], ...vs)];
  chart(host, {x: pad(xr, 0.02), y: [0, top * 1.15], xlog: o.xlog, xlabel: o.xlabel, ylabel: o.ylabel, label: o.label, layers,
    hover: ({x: v}) => { const i = nearest(x, v); return {cross: x[i], html: `<b>${o.xlabel}: ${o.fmt(e[i])} – ${o.fmt(e[i + 1])}</b>` + trow("var(--bar)", `${h.counts[i].toLocaleString()} drawn${dd.ref ? `, prior ${sig(h.reference[i], 4)}` : ""}`)}; }});
}

// the band's flow coordinates, one row each, against the network's reach
function flowPrior(host, lo, hi) {
  const fl = B().flow, W = Math.max(320, host.clientWidth), row = 48, t = 4, l = 64, r = 16, H = t + row * fl.length + 30;
  const X = v => l + (clamp(v, lo, hi) - lo) / (hi - lo) * (W - l - r);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="flow coordinates of the prior">`;
  niceTicks(lo, hi, 8).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${t}" y2="${t + fl.length * row}" stroke="var(--grid)"/><text x="${X(v)}" y="${t + fl.length * row + 16}" text-anchor="middle">${v}</text>`);
  fl.forEach((f, i) => {
    const y = t + i * row, dd = density(f.hist), top = Math.max(...dd.v), e = f.hist.edges;
    s += `<rect x="${X(f.reach[0])}" width="${X(f.reach[1]) - X(f.reach[0])}" y="${y + 2}" height="${row - 6}" fill="var(--grid)" opacity="0.45"/>`;
    dd.v.forEach((v, k) => { if (v > 0) s += `<rect x="${X(e[k]) + 0.5}" width="${Math.max(0.5, X(e[k + 1]) - X(e[k]) - 1)}" y="${y + row - 4 - v / top * (row - 10)}" height="${v / top * (row - 10)}" fill="var(--bar)"/>`; });
    s += `<text class="lbl" x="${l - 8}" y="${y + row / 2 + 2}" text-anchor="end">${f.name}</text><text x="${l - 8}" y="${y + row / 2 + 14}" text-anchor="end">std ${f.std.toFixed(2)}</text>`;
  });
  s += `<text x="${(l + W - r) / 2}" y="${H - 2}" text-anchor="middle">flow coordinate</text>`;
  host.innerHTML = s + "</svg>";
}

function tiles() {
  const b = B(), d = b.problem, ds = b.dataset, T = d.t_obs;
  document.getElementById("tiles").innerHTML = [
    ["f₀ range", `${sig(d.f0_range[0] * 1e3, 2)}–${sig(d.f0_range[1] * 1e3, 3)} mHz`, `${d.window_bins.toLocaleString()}-bin windows of ${d.n_sources} sources`],
    ["windows per run", big(ds.windows), `${ds.total_steps.toLocaleString()} steps × ${ds.windows_per_step}`],
    ["sources per run", big(ds.sources), `${big(ds.samples)} (x₀, t) samples`],
    ["drift beyond one bin", pct(driftOver1(b), 0), `99% drift less than ${sig(b.table["fdot [Hz/s]"].quantiles[4] * T * T, 2)} bins`],
    ["M<sub>c</sub> draws against the density", `χ²/dof ${b.mc_check.chi2_per_dof.toFixed(2)}`, `${b.mc_check.bins} bins of ln M<sub>c</sub>, ${big(b.mc_check.draws)} draws`],
  ].map(([k, v, dd]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${dd}</div></div>`).join("");
}

function viewCard() {
  const host = document.getElementById("p-view"), b = B(), d = b.problem, T = d.t_obs, v = S.view;
  host.innerHTML = "";
  const exp10 = (v, n = 3) => sig(10 ** v, n);
  if (v === "f0") {
    histChart(host, b.frequency["log10 f0 [Hz]"], {xlog: true, xlabel: "f₀ [Hz]", ylabel: "density per dex", label: "f0 prior", fmt: exp10});
    html("n-view", `f₀ is log-uniform over [${sig(d.f0_range[0] * 1e3)}, ${sig(d.f0_range[1] * 1e3)}] mHz: a window is snapped first, then its sources are drawn within its own f₀ range. ${S.band === "full" ? `At the bottom edge the ${d.window_bins}-bin window is wide in dex, so the snap moves sources up out of the lowest bins: mass moved, not lost.` : `The ${d.window_bins}-bin window is too short to reshape the marginal.`}`);
  } else if (v === "mc") {
    histChart(host, b.frequency["log10 Mc [Msun]"], {xlog: true, xlabel: "chirp mass [M☉]", ylabel: "density per dex", label: "chirp mass prior", fmt: exp10});
    html("n-view", `The chirp mass of two white dwarfs from <code>mass_range</code> = [${d.mass_range.join(", ")}] M☉. The draws match <code>chirp_mass_log_density</code>, the density the MCMC reference uses: χ²/dof = ${b.mc_check.chi2_per_dof.toFixed(2)} against quadrature.`);
  } else {
    const R = d.response_points, shift = Math.log10(T * T), q99 = b.table["fdot [Hz/s]"].quantiles[4] * T * T, q50 = b.table["fdot [Hz/s]"].quantiles[2] * T * T;
    histChart(host, b.frequency["log10 fdot [Hz/s]"], {shift, xlog: true, xlabel: "ḟT² [bins]", ylabel: "density per dex", label: "chirp drift in bins", fmt: exp10,
      under: [{vline: 0, color: "var(--muted)", dash: "2 3", text: "1 bin", anchor: "start"}, {vline: Math.log10(R), color: "var(--ink-2)", dash: "4 3", text: `response ${R} bins`, anchor: "end"}]});
    html("n-view", `How far a source's frequency moves over the observation, ḟT² in rfft bins, with ḟ from M<sub>c</sub> and f₀. The median source drifts ${sig(q50)} bins and 99% drift less than ${sig(q99)} bins, against a response of ${R} bins${S.band === "full" ? ": the loudest-chirping sources in the full band move tens of bins, so f₀ alone does not pin the token a source sits in" : ": in the narrow band every source is effectively monochromatic"}.`);
  }
  html("lg-view", legendBox("var(--bar)", `sources of ${b.windows.toLocaleString()} prior windows`) + (host.querySelector(`polyline[stroke="var(--ink-2)"]`) ? legendItem("var(--ink-2)", "closed-form prior", "4 3") : ""));
}

function psdCard() {
  const host = document.getElementById("p-psd"), b = B();
  host.innerHTML = "";
  const bandSpans = Object.entries(PG.bands).map(([name, bb]) => ({span: bb.problem.f0_range.map(Math.log10), color: "var(--ink)", opacity: name === S.band ? 0.1 : 0.04, text: `${name} band`, textEnd: name === "full"}));
  const fs = b.noise.f.map(Math.log10), ae = fs.map((lf, i) => [lf, 0.5 * Math.log10(b.noise.psd[i][0])]);
  const ys = [Math.min(...ae.map(p => p[1])) - 1, Math.max(...ae.map(p => p[1]))];
  const tt = fs.map((lf, i) => [lf, 0.5 * Math.log10(b.noise.psd[i][2])]).filter(p => p[1] >= ys[0] && p[1] <= ys[1]);
  chart(host, {x: [fs[0], fs.at(-1)], y: ys, xlog: true, ylog: true, xlabel: "f [Hz]", ylabel: "√S [1/√Hz, fractional frequency]", label: "noise PSD with the bands", H: 300,
    layers: [...bandSpans, {line: ae, color: "var(--ink)", width: 2}, {line: tt, color: "var(--ink-2)", width: 1.5, dash: "5 3"}],
    hover: ({x}) => { const i = nearest(fs, x); return {cross: fs[i], html: `<b>${sig(10 ** fs[i] * 1e3, 4)} mHz</b>` + trow("var(--ink)", `√S<sub>A</sub> = √S<sub>E</sub> ${sig(10 ** ae[i][1])}`) + trow("var(--ink-2)", `√S<sub>T</sub> ${sig(Math.sqrt(b.noise.psd[i][2]))}`)}; }});
  const n = PG.bands.narrow.problem.f0_range, bucket = 10 ** fs[ae.reduce((bi, p, i) => p[1] < ae[bi][1] ? i : bi, 0)];
  html("lg-psd", legendItem("var(--ink)", "A and E") + legendItem("var(--ink-2)", "T", "5 3"));
  html("n-psd", `<code>noise_psd</code>, TDI-1.5 in fractional frequency; shaded are the bands' f₀ ranges. The narrow band (${sig(n[0] * 1e3, 2)}–${sig(n[1] * 1e3, 3)} mHz) runs down the acceleration-noise slope towards the bucket; the full band reaches past its bottom near ${sig(bucket * 1e3, 2)} mHz. T is nearly blind at low f.`);
}

function runCard() {
  const rows = [["steps", "total_steps"], ["windows per step", "windows_per_step"], ["windows", "windows"], ["sources", "sources"], ["(x₀, t) samples", "samples"]];
  const cls = band => band === S.band ? ' class="on"' : "";
  html("t-run", `<tr><th>one run</th>${CHOICES.band.map(band => `<th${cls(band)}>${band}-B</th>`).join("")}</tr>` +
    rows.map(([label, k]) => `<tr><td>${label}</td>${CHOICES.band.map(band => `<td${cls(band)}>${big(PG.bands[band].dataset[k])}</td>`).join("")}</tr>`).join(""));
  const ds = B().dataset, perWindow = ds.samples / ds.windows;
  html("n-run", `One run of <code>total_steps</code> at <code>batch_size</code> ${ds.batch_size.toLocaleString()} samples, ${ds.draws} (x₀, t) draws per window, so ${ds.windows_per_step} fresh windows per step and each window seen ${perWindow} times. No window repeats: the prior is sampled afresh at every step.`);
}

function flowCard() {
  const b = B();
  flowPrior(document.getElementById("p-flow"), -5, 5);
  html("lg-flow", legendBox("var(--bar)", "prior") + legendBox("var(--grid)", "network's reach"));
  html("n-flow", `f₀ and M<sub>c</sub> as the flow sees them: f₀ is uniform on [−1, 1] across the window (std 1/√3 = 0.577) and M<sub>c</sub> is whitened (std ${b.flow[1].std.toFixed(3)}). No two of the eight flow coordinates correlate beyond |r| = ${b.corr_max.toFixed(3)}.`);
}

function meaning() {
  const n = PG.bands.narrow, f = PG.bands.full, T = f.problem.t_obs;
  const fh = f.frequency["log10 f0 [Hz]"], edge = fh.counts.slice(0, 4).map((c, i) => (c / fh.reference[i]).toFixed(2));
  const width = f.problem.window_bins / T, lo = f.problem.f0_range[0];
  html("meaning", [
    `<b>Sources drift, in the full band only.</b> 99% drift less than ${sig(f.table["fdot [Hz/s]"].quantiles[4] * T * T)} bins over the observation in the full band and ${sig(n.table["fdot [Hz/s]"].quantiles[4] * T * T, 2)} in the narrow one. M<sub>c</sub> enters the data only through that drift, and ${pct(driftOver1(f), 0)} of full-band sources drift more than one bin (${pct(driftOver1(n), 0)} narrow).`,
    `<b>The full band's f₀ marginal is reshuffled at its bottom edge.</b> Bins 1–4 hold ${edge.join(", ")} of their log-uniform share: window snapping moves sources up, since a ${f.problem.window_bins}-bin window is ${sig(width * 1e6, 2)} µHz, or ${Math.log10(1 + width / lo).toFixed(2)} dex at ${sig(lo * 1e3)} mHz.`,
    `<b>The chirp-mass draws match the density the MCMC reference uses</b> (χ²/dof = ${f.mc_check.chi2_per_dof.toFixed(2)} over ${f.mc_check.bins} bins of ln M<sub>c</sub>).`,
    `<b>f₀ and M<sub>c</sub> reach the flow whitened.</b> Stds ${f.flow.map(c => `${c.std.toFixed(3)} (${c.name})`).join(" and ")}, with every |r| &lt; ${Math.max(n.corr_max, f.corr_max).toFixed(3)} across the eight coordinates.`,
    `<b>One run draws every window afresh:</b> ${big(f.dataset.windows)} windows (${big(f.dataset.sources)} sources) in the full band and ${big(n.dataset.windows)} (${big(n.dataset.sources)}) in the narrow one.`,
  ].map(t => `<li>${t}</li>`).join(""));
}

function render() { tiles(); viewCard(); psdCard(); runCard(); flowCard(); meaning(); writeHash(); }
let pending = 0;
const schedule = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; render(); }); };
addEventListener("resize", schedule);
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


def drift_over_one_bin(frequency: dict, t_obs: float) -> float:
    """Share of sources whose ḟT² exceeds one rfft bin, from the log10 ḟ histogram's bin centres."""
    h = frequency["log10 fdot [Hz/s]"]
    shift = math.log10(t_obs**2)
    edges = h["edges"]
    over = sum(
        c
        for i, c in enumerate(h["counts"])
        if (edges[i] + edges[i + 1]) / 2 + shift > 0
    )
    return over / (sum(h["counts"]) + h["outside"])


def page_data(prior: dict) -> dict:
    """What the page plots, per band."""
    bands = {}
    for band in BANDS:
        b = prior["bands"][band]
        corr = b["corr"]
        bands[band] = {
            k: b[k] for k in ("problem", "windows", "table", "dataset", "noise")
        } | dict(
            frequency=b["physical"]["frequency"],
            mc_check={
                k: b["chirp_mass_check"][k] for k in ("bins", "draws", "chi2_per_dof")
            },
            flow=[
                {k: c[k] for k in ("name", "hist", "reach", "std", "beyond_reach")}
                for c in b["flow"]
                if c["name"] in ("f0", "Mc")
            ],
            corr_max=max(
                abs(corr[i][j])
                for i in range(len(corr))
                for j in range(len(corr))
                if i != j
            ),
        )
    return rounded(dict(bands=bands))


def report(data: dict) -> None:
    """Print every number the README's frequency takeaways quote."""
    for band, b in data["bands"].items():
        t_obs = b["problem"]["t_obs"]
        q = b["table"]["fdot [Hz/s]"]["quantiles"]
        stds = ", ".join(f"{c['name']} {c['std']:.3f}" for c in b["flow"])
        print(
            f"{band}: drift 99% {q[4] * t_obs**2:.3g} bins, beyond one bin {drift_over_one_bin(b["frequency"], t_obs):.3f};"
            f" Mc chi2/dof {b['mc_check']['chi2_per_dof']:.3f} over {b['mc_check']['bins']} bins;"
            f" flow std {stds}, |r| max {b['corr_max']:.4f};"
            f" per run {b['dataset']['windows']:.3g} windows, {b['dataset']['sources']:.3g} sources"
        )
    f = data["bands"]["full"]
    h = f["frequency"]["log10 f0 [Hz]"]
    width = f["problem"]["window_bins"] / f["problem"]["t_obs"]
    lo = f["problem"]["f0_range"][0]
    ratios = ", ".join(
        f"{c / r:.2f}" for c, r in zip(h["counts"][:4], h["reference"][:4])
    )
    print(
        f"full f0 bins 1-4 of log-uniform share: {ratios};"
        f" {f['problem']['window_bins']}-bin window {width * 1e6:.3g} µHz, {math.log10(1 + width / lo):.3f} dex at {lo * 1e3:.3g} mHz"
    )


def main() -> None:
    data = page_data(json.loads((HERE / "data" / "prior.json").read_text()))
    page = PAGE.replace("__PAGE__", json.dumps(data, separators=(",", ":"))).replace(
        "__SUB__",
        f"Built {date.today().isoformat()} from <code>data/prior.json</code>.",
    )
    out = HERE / "frequency.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")
    report(data)


if __name__ == "__main__":
    main()
