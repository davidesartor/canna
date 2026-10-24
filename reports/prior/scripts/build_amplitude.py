"""Write amplitude.html from data/prior.json: LisaGB's prior on the SNR, the amplitude and the orientation angles.

Standard library only; run from the repo root:
    uv run --frozen python reports/prior/scripts/build_amplitude.py
"""

import json
import math
from datetime import date
from pathlib import Path

HERE = Path(__file__).parents[1]
BANDS = ("narrow", "full")
SCATTER = 1500
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior: amplitude</title>
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
<p class="nav">The LisaGB prior by parameter group: <a data-page="frequency.html" href="frequency.html">frequency</a> · <b>amplitude</b> · <a data-page="sky.html" href="sky.html">sky</a></p>
<h1>The LisaGB prior: amplitude</h1>
<p class="sub">How loud a source is and how it is turned: the SNR each source is drawn at, the amplitude A that SNR needs, the inclination, ψ and φ₀, and the SNR constant C(f₀) that links them, over 8192 drawn windows of four sources per band; then Re and Im of z± in the flow's coordinates. __SUB__</p>
<div class="row"><span class="k">Band</span><span class="seg" id="band"></span></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>How loud is each source, and each window?</h2>
  <p class="note" id="n-snr"></p>
  <div class="legend" id="lg-snr"></div>
  <div id="p-snr"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What amplitude does that SNR need?</h2>
  <span class="seg" id="amp"></span>
  <p class="note" id="n-amp"></p>
  <div class="legend" id="lg-amp"></div>
  <div id="p-amp"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How are the orientation angles drawn?</h2>
  <span class="seg" id="angle"></span>
  <p class="note" id="n-angle"></p>
  <div class="legend" id="lg-angle"></div>
  <div id="p-angle"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How loud is a source at one <code>amplitude_unit</code>, across f₀?</h2>
  <p class="note" id="n-c"></p>
  <div class="legend" id="lg-c"></div>
  <div id="p-c"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What do z± look like in the flow's coordinates?</h2>
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
const SLOT = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"];
const AMPS = [["A", "A"], ["f0", "A against f₀"], ["kappa", "κ against |sin ι|"]];
const ANGLES = [["incl", "sin ι"], ["psi", "ψ"], ["phi0", "φ₀"]];
const CHOICES = {band: ["narrow", "full"], amp: AMPS.map(v => v[0]), angle: ANGLES.map(v => v[0])};
const S = {band: "full", amp: "A", angle: "incl", theme: ""};
const B = () => PG.bands[S.band];
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sum = a => a.reduce((x, y) => x + y, 0);
const sup = e => String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);

const marks = {};
function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; marks[key](); schedule(); }));
  marks[key]();
}
seg("band", [["narrow", "narrow-B"], ["full", "full-B"]], "band");
seg("amp", AMPS, "amp");
seg("angle", ANGLES, "angle");

// state in the address bar: #band=full&amp=A&angle=incl (and theme=light|dark)
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
  for (const L of o.layers) {
    if (L.bars) L.bars.v.forEach((v, i) => { if (v > 0) { const a = X(L.bars.e[i]) + 1, b = X(L.bars.e[i + 1]) - 1; s += `<rect x="${a.toFixed(1)}" y="${Y(v).toFixed(1)}" width="${Math.max(b - a, 0.5).toFixed(1)}" height="${(Y(Math.max(y0, 0)) - Y(v)).toFixed(1)}" fill="${L.color}" rx="1"/>`; } });
    if (L.line) s += `<polyline points="${pts(L.line)}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 2}" stroke-linejoin="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.marks) L.line.forEach(([x, y]) => s += `<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="3.5" fill="${L.color}" stroke="var(--surface)" stroke-width="1.5"/>`);
    if (L.dots) L.dots.forEach(([x, y]) => s += `<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="${L.r ?? 2}" fill="${L.color}" fill-opacity="${L.opacity ?? 0.5}"/>`);
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
  const top = Math.max(...dd.v, ...(dd.ref ?? [0]), ...(o.extra ?? []).flatMap(L => L.line.map(p => p[1])));
  const layers = [{bars: {e, v: dd.v}, color: "var(--bar)"}];
  if (dd.ref) layers.push({line: x.map((v, i) => [v, dd.ref[i]]), color: "var(--ink-2)", width: 1.5, dash: "4 3"});
  (o.extra ?? []).forEach(L => layers.push(L));
  chart(host, {x: pad(o.x ?? [e[0], e[e.length - 1]], 0.02), y: [0, top * 1.15], xlog: o.xlog, xlabel: o.xlabel, ylabel: o.ylabel, label: o.label, layers,
    hover: ({x: v}) => { const i = nearest(x, v); return {cross: x[i], html: `<b>${o.xlabel}: ${o.fmt(e[i])} – ${o.fmt(e[i + 1])}</b>` + trow("var(--bar)", `${h.counts[i].toLocaleString()} drawn${dd.ref ? `, prior ${sig(h.reference[i], 4)}` : ""}`) + (o.tipExtra ? o.tipExtra(i) : "")}; }});
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
  const b = B(), t = b.table, iso = b.sky_average.isotropic;
  document.getElementById("tiles").innerHTML = [
    ["source SNR, median", sig(t.snr.quantiles[2]), `log-uniform over [${b.problem.snr_range.join(", ")}]; ${pct(b.in_range, 0)} inside`],
    ["window SNR, median", sig(t["window snr"].quantiles[2]), `1% of windows below ${sig(t["window snr"].quantiles[0])}`],
    ["κ, 16–84%", `${sig(t.kappa.quantiles[1], 2)}–${sig(t.kappa.quantiles[3], 2)}`, `log κ against |sin ι|: r = ${b.kappa_r.toFixed(2)}`],
    ["C at low f₀", `${iso.C_low.toFixed(2)} ± ${iso.C_low_se.toFixed(2)}`, `isotropic sky; galactic ${b.sky_average.galactic.C_low.toFixed(2)}`],
    ["⟨κ²⟩ over the prior", `${b.C_prior.toFixed(2)} ± ${b.C_prior_se.toFixed(2)}`, "the training prior's sky and f₀"],
  ].map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}

function snrCard() {
  const host = document.getElementById("p-snr"), b = B(), d = b.problem, exp10 = (v, n = 3) => sig(10 ** v, n);
  host.innerHTML = "";
  const h = b.snr_hist, dd = density(h), x = centres(h.edges), wd = density(b.window_snr_hist), wx = centres(b.window_snr_hist.edges);
  const extra = b.snr_rank.map((r, k) => ({line: x.map((v, i) => [v, r.counts[i] / dd.n / dd.w]), color: SLOT[k], width: 1.5}));
  extra.push({line: wx.map((v, i) => [v, wd.v[i]]), color: "var(--ink)", width: 1.5});
  histChart(host, h, {xlog: true, x: [b.window_snr_hist.edges[0], b.window_snr_hist.edges.at(-1)], xlabel: "SNR", ylabel: "density per dex", label: "snr prior", fmt: exp10, extra,
    tipExtra: i => b.snr_rank.map((r, k) => trow(SLOT[k], `slot ${k + 1}: ${r.counts[i].toLocaleString()}`)).join("")});
  const med = b.snr_rank.map(r => 10 ** histQuantile(r, 0.5)), wt = b.table["window snr"].quantiles;
  html("lg-snr", legendBox("var(--bar)", "every source") + SLOT.map((c, k) => legendItem(c, `slot ${k + 1}`)).join("") + legendItem("var(--ink)", "window SNR") + legendItem("var(--ink-2)", "log-uniform prior", "4 3"));
  html("n-snr", `Each source's SNR is log-uniform over [${d.snr_range.join(", ")}] (bars); sorting loudest first makes slot k an order statistic (coloured lines, which sum to the bars), with medians ≈ ${med.map(v => sig(v, 2)).join(", ")}. The window's joint SNR (black) has median ${sig(wt[2])} and only 1% of windows fall below ${sig(wt[0])}: a window is rarely quiet.`);
}

function ampCard() {
  const host = document.getElementById("p-amp"), b = B(), d = b.problem, sc = b.scatter, v = S.amp, exp10 = (v, n = 3) => sig(10 ** v, n);
  host.innerHTML = "";
  if (v === "A") {
    const t = b.table.A.quantiles;
    histChart(host, b.amplitude["log10 A"], {xlog: true, xlabel: "A", ylabel: "density per dex", label: "amplitude prior", fmt: exp10});
    html("lg-amp", legendBox("var(--bar)", `sources of ${b.windows.toLocaleString()} prior windows`));
    html("n-amp", `A is not drawn: it is whatever gives the drawn SNR. Its 1–99% range spans ${(Math.log10(t[4] / t[0])).toFixed(1)} decades, of which ${Math.log10(d.snr_range[1] / d.snr_range[0]).toFixed(2)} is the SNR range; the rest is κ and the noise curve across <code>f0_range</code> (see A against f₀).`);
  } else if (v === "f0") {
    const pts = sc.f0.map((f, i) => [Math.log10(f), Math.log10(sc.A[i])]), ab = b.amplitude_band;
    chart(host, {x: pad(d.f0_range.map(Math.log10), 0.02), y: pad([Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))]), xlog: true, ylog: true, xlabel: "f₀ [Hz]", ylabel: "A", label: "amplitude prior against f0", H: 300,
      layers: [{dots: pts, color: "var(--muted)", opacity: 0.35}, {line: ab.f0.map((f, i) => [Math.log10(f), Math.log10(ab.lo[i])]), color: "var(--ink-2)", width: 1.5, dash: "4 3"}, {line: ab.f0.map((f, i) => [Math.log10(f), Math.log10(ab.hi[i])]), color: "var(--ink-2)", width: 1.5, dash: "4 3"}],
      hover: h => { const hit = nearestPoint(pts, h); if (!hit) return null; const [[x, y], i] = hit; return {ring: [x, y], html: trow("var(--muted)", `f₀ ${sig(10 ** x * 1e3, 4)} mHz, A ${sig(10 ** y)}, SNR ${sig(sc.snr[i])}`)}; }});
    html("lg-amp", legendDot("var(--muted)", `${pts.length.toLocaleString()} prior sources`) + legendItem("var(--ink-2)", `SNR ${d.snr_range.join(" and ")} at the sky average`, "4 3"));
    html("n-amp", "Each source's amplitude against its f₀. The dashed lines are <code>snr_range × amplitude_unit(f)</code>, where a sky-averaged source of SNR 7 or 1000 sits; the cloud follows the noise curve, and κ spreads it about the lines.");
  } else {
    const pts = sc.sin_iota.map((s, i) => [Math.abs(s), Math.log10(sc.kappa[i])]), kq = b.table.kappa.quantiles;
    chart(host, {x: [0, 1], y: pad([Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))]), ylog: true, xlabel: "|sin ι|", ylabel: "κ = ρ / A · amplitude_unit(f)", label: "kappa against inclination",
      layers: [{dots: pts, color: "var(--muted)", opacity: 0.35, r: 2}],
      hover: h => { const hit = nearestPoint(pts, h); if (!hit) return null; const [[x, y]] = hit; return {ring: [x, y], html: trow("var(--muted)", `|sin ι| ${x.toFixed(3)}, κ ${(10 ** y).toFixed(2)}`)}; }});
    html("lg-amp", legendDot("var(--muted)", `${pts.length.toLocaleString()} prior sources`));
    html("n-amp", `κ is the SNR at one <code>amplitude_unit</code>: amplitude and inclination together set the SNR. log κ follows |sin ι| at r = ${b.kappa_r.toFixed(2)} here; κ runs ${sig(kq[1], 2)}–${sig(kq[3], 2)} (16–84%), so <code>amplitude_unit</code> is the SNR-1 amplitude to within a factor of 1–3. The sky, ψ and M<sub>c</sub> barely move it (|r| &lt; ${b.kappa_r_other.toFixed(2)}).`);
  }
}

function angleCard() {
  const host = document.getElementById("p-angle"), b = B();
  host.innerHTML = "";
  const [key, label] = {incl: ["sin iota", "sin ι"], psi: ["psi [rad]", "ψ [rad]"], phi0: ["phi0 [rad]", "φ₀ at t = 0 [rad]"]}[S.angle];
  histChart(host, b.orientation[key], {xlabel: label, ylabel: "density", label: `${label} prior`, fmt: x => x.toFixed(2)});
  html("lg-angle", legendBox("var(--bar)", `sources of ${b.windows.toLocaleString()} prior windows`) + legendItem("var(--ink-2)", "closed-form prior", "4 3"));
  html("n-angle", {incl: "sin ι is uniform on [−1, 1]: ι is isotropic, with ι measured from the sky plane so that sin ι plays the role of the usual cos ι. It sets which helicity the data carry: face-on (+1) sources show only φ₋ = φ₀ − 2ψ, face-off (−1) only φ₊.",
    psi: "ψ is uniform. With φ₀ it enters only through φ± = φ₀ ± 2ψ, which is why the flow predicts z± instead of ψ and φ₀.",
    phi0: "φ₀ is uniform. It turns every bin of the window by the same phase, so it moves no |ỹ| and the tokens' magnitudes do not see it."}[S.angle]);
}

function cCard() {
  const host = document.getElementById("p-c"), sa = B().sky_average, xs = sa.f0.map(Math.log10), line = k => sa[k].C.map((v, i) => [xs[i], v]);
  host.innerHTML = "";
  const all = [...sa.isotropic.C, ...sa.galactic.C];
  chart(host, {x: [xs[0], xs.at(-1)], y: pad([Math.min(...all), Math.max(...all)], 0.1), xlog: true, xlabel: "f₀ [Hz]", ylabel: "C = ⟨ρ²⟩ at A = amplitude_unit(f₀)", label: "sky-averaged snr constant", H: 280,
    layers: [{line: line("isotropic"), color: "var(--ink)", marks: true}, {line: line("galactic"), color: "var(--ink-2)", dash: "5 3", marks: true}],
    hover: ({x}) => { const i = nearest(xs, x); return {cross: xs[i], html: `<b>f₀ ${sig(sa.f0[i] * 1e3, 3)} mHz</b>` + trow("var(--ink)", `isotropic C ${sa.isotropic.C[i].toFixed(3)} ± ${sa.isotropic.C_se[i].toFixed(3)}`) + trow("var(--ink-2)", `galactic C ${sa.galactic.C[i].toFixed(3)} ± ${sa.galactic.C_se[i].toFixed(3)}`)}; }});
  html("lg-c", legendItem("var(--ink)", "isotropic sky") + legendItem("var(--ink-2)", "galactic sky prior", "5 3"));
  const iso = sa.isotropic, gal = sa.galactic;
  html("n-c", `The mean ρ² of a monochromatic source at one <code>amplitude_unit</code>, over the same ${sa.sources.toLocaleString()} orientations at every f₀, so the curve's shape carries no sampling noise and only its level does (± ${iso.C_low_se.toFixed(2)}, 1 s.e.). It is flat at C = ${iso.C_low.toFixed(2)} ± ${iso.C_low_se.toFixed(2)} at low f and rises ${pct(iso.C.at(-1) / iso.C_low - 1, 1)} by ${sig(sa.f0.at(-1) * 1e3, 3)} mHz; the galactic prior is ${pct(gal.C_low / iso.C_low - 1)} louder than an isotropic sky. Over the whole training prior ⟨κ²⟩ = ${B().C_prior.toFixed(2)}.`);
}

function flowCard() {
  const b = B();
  flowPrior(document.getElementById("p-flow"), -9, 9);
  html("lg-flow", legendBox("var(--bar)", "prior") + legendBox("var(--grid)", "network's reach"));
  html("n-flow", `The helicity amplitudes z± = asinh-radius · e<sup>iφ±</sup>, as the flow sees them. Their radius follows the SNR draw (median |z+| ${sig(b.radius[0].quantiles[2])}, |z−| ${sig(b.radius[1].quantiles[2])}), so Re and Im are ${sig(Math.min(...b.flow.map(c => c.std)), 3)}–${sig(Math.max(...b.flow.map(c => c.std)), 3)} wide, about 2.5× the other coordinates: any loss or embedding that assumes unit scale sees z± dominate. ${pct(Math.max(...b.flow.map(c => c.beyond_reach)), 2)} fall beyond the reach of ±${b.flow[0].reach[1].toFixed(1)}.`);
}

function meaning() {
  const n = PG.bands.narrow, f = PG.bands.full, med = f.snr_rank.map(r => 10 ** histQuantile(r, 0.5));
  const decades = b => Math.log10(b.table.A.quantiles[4] / b.table.A.quantiles[0]), kdec = b => Math.log10(b.table.kappa.quantiles[4] / b.table.kappa.quantiles[0]);
  const iso = f.sky_average.isotropic;
  html("meaning", [
    `<b>Every window is crowded and loud.</b> Four sources, each with SNR log-uniform over [${f.problem.snr_range.join(", ")}] (median ${sig(f.table.snr.quantiles[2])} against √${f.problem.snr_range[0] * f.problem.snr_range[1]} = ${sig(Math.sqrt(f.problem.snr_range[0] * f.problem.snr_range[1]))}), sorted loudest first, so slot k is an order statistic (median SNRs ≈ ${med.map(v => sig(v, 2)).join(", ")}). The window's joint SNR has median ${sig(f.table["window snr"].quantiles[2])}; only 1% fall below ${sig(f.table["window snr"].quantiles[0])}.`,
    `<b>Inclination alone sets κ, the SNR at one <code>amplitude_unit</code>.</b> κ runs ${sig(f.table.kappa.quantiles[1], 2)}–${sig(f.table.kappa.quantiles[3], 2)} (16–84%); log κ correlates with |sin ι| at r = ${n.kappa_r.toFixed(2)} (narrow) and ${f.kappa_r.toFixed(2)} (full). A spans ${decades(n).toFixed(1)} decades in the narrow band and ${decades(f).toFixed(1)} in the full band (1–99%): ${Math.log10(f.problem.snr_range[1] / f.problem.snr_range[0]).toFixed(2)} is the SNR range, ${kdec(n).toFixed(2)}–${kdec(f).toFixed(2)} is κ, the rest the sensitivity curve.`,
    `<b>The sky-averaged SNR constant is C = ${iso.C_low.toFixed(2)} ± ${iso.C_low_se.toFixed(2)}.</b> It rises ${pct(n.sky_average.isotropic.C.at(-1) / n.sky_average.isotropic.C_low - 1, 2)} at the top of the narrow band and ${pct(iso.C.at(-1) / iso.C_low - 1, 0)} at ${sig(f.sky_average.f0.at(-1) * 1e3, 3)} mHz; the galactic sky is ${pct(f.sky_average.galactic.C_low / iso.C_low - 1)} louder. Over the training prior ⟨κ²⟩ is ${n.C_prior.toFixed(2)} (narrow) and ${f.C_prior.toFixed(2)} (full).`,
    `<b>z± reach the flow wider than the rest.</b> Re and Im of z± have std ${sig(Math.min(...f.flow.map(c => c.std)), 3)}–${sig(Math.max(...f.flow.map(c => c.std)), 3)}, about 2.5× the whitened f₀, M<sub>c</sub> and sky coordinates.`,
  ].map(t => `<li>${t}</li>`).join(""));
}

function render() { tiles(); snrCard(); ampCard(); angleCard(); cCard(); flowCard(); meaning(); writeHash(); }
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


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


def hist_quantile(h: dict, q: float) -> float:
    """Quantile of a histogram, interpolated linearly inside the bin."""
    target, acc = q * sum(h["counts"]), 0
    for i, c in enumerate(h["counts"]):
        if acc + c >= target:
            return h["edges"][i] + (target - acc) / c * (h["edges"][i + 1] - h["edges"][i])
        acc += c
    return h["edges"][-1]


def page_data(prior: dict) -> dict:
    """What the page plots, per band; κ's correlations use the whole recorded scatter."""
    keep = ("problem", "windows", "in_range", "snr_hist", "snr_rank", "window_snr_hist", "table",
            "amplitude_band", "C_prior", "C_prior_se", "sky_average")
    bands = {}
    for band in BANDS:
        b = prior["bands"][band]
        sc = b["scatter"]
        log_kappa = [math.log10(k) for k in sc["kappa"]]
        others = dict(psi=sc["psi"], l=sc["gal_l"], b=sc["gal_b"], Mc=[math.log10(m) for m in sc["mc"]])
        bands[band] = {k: b[k] for k in keep} | dict(
            amplitude=b["physical"]["amplitude"],
            orientation=b["physical"]["orientation"],
            kappa_r=pearson([abs(s) for s in sc["sin_iota"]], log_kappa),
            kappa_r_other=max(abs(pearson(x, log_kappa)) for x in others.values()),
            flow=[
                {k: c[k] for k in ("name", "hist", "reach", "std", "beyond_reach")}
                for c in b["flow"]
                if c["name"].startswith(("Re", "Im"))
            ],
            radius=[{k: r[k] for k in ("name", "quantiles")} for r in b["radius"]],
            scatter={k: sc[k][:SCATTER] for k in ("f0", "A", "snr", "kappa", "sin_iota")},
        )
    return rounded(dict(bands=bands))


def report(data: dict) -> None:
    """Print every number the README's amplitude takeaways quote."""
    for band, b in data["bands"].items():
        t = b["table"]
        snr_lo, snr_hi = b["problem"]["snr_range"]
        slots = ", ".join(f"{10 ** hist_quantile(r, 0.5):.3g}" for r in b["snr_rank"])
        decades = math.log10(t["A"]["quantiles"][4] / t["A"]["quantiles"][0])
        kappa_decades = math.log10(t["kappa"]["quantiles"][4] / t["kappa"]["quantiles"][0])
        sa = b["sky_average"]
        iso, gal = sa["isotropic"], sa["galactic"]
        stds = ", ".join(f"{c['std']:.3f}" for c in b["flow"])
        print(
            f"{band}: in range {b['in_range']}; SNR median {t['snr']['quantiles'][2]:.3g} vs sqrt {math.sqrt(snr_lo * snr_hi):.3g};"
            f" slot medians {slots}; window SNR median {t['window snr']['quantiles'][2]:.3g}, 1% {t['window snr']['quantiles'][0]:.3g}\n"
            f"  kappa 16-84% {t['kappa']['quantiles'][1]:.3g}-{t['kappa']['quantiles'][3]:.3g}, r(|sin i|, log kappa) {b['kappa_r']:.3f},"
            f" max |r| others {b['kappa_r_other']:.3f}; A {decades:.2f} decades, SNR {math.log10(snr_hi / snr_lo):.2f}, kappa {kappa_decades:.2f}\n"
            f"  C_low {iso['C_low']:.3f} ± {iso['C_low_se']:.3f}, rise at {sa['f0'][-1] * 1e3:.3g} mHz {iso['C'][-1] / iso['C_low'] - 1:.4f},"
            f" galactic {gal['C_low']:.3f} ({gal['C_low'] / iso['C_low'] - 1:.4f}); <kappa^2> {b['C_prior']:.3f} ± {b['C_prior_se']:.3f}; z std {stds}"
        )


def main() -> None:
    data = page_data(json.loads((HERE / "data" / "prior.json").read_text()))
    page = PAGE.replace("__PAGE__", json.dumps(data, separators=(",", ":"))).replace(
        "__SUB__", f"Built {date.today().isoformat()} from <code>data/prior.json</code>."
    )
    out = HERE / "amplitude.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")
    report(data)


if __name__ == "__main__":
    main()
