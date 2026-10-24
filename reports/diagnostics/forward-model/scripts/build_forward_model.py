"""Write forward-model.html from data/: the TDI, units, response-truncation and noise-generator checks of the simulator.

Numpy only; run from the repo root:
    uv run --frozen python reports/diagnostics/forward-model/scripts/build_forward_model.py
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).parents[1]
DATA = FOLDER / "data"
BANDS = ("narrow", "full")
STRIDE = 4
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The LisaGB forward model</title>
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
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main > .card + .card { margin-top: 16px; }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 24px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 22px; flex: none; }
.row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
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
.tip .row { color: var(--ink-2); font-variant-numeric: tabular-nums; margin: 0; justify-content: flex-start; gap: 6px; }
.pair { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
@media (max-width: 700px) { .pair { grid-template-columns: minmax(0, 1fr); } }
.tablewrap { overflow-x: auto; }
table.t { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
table.t th, table.t td { text-align: right; padding: 4px 12px; border-bottom: 1px solid var(--grid); }
table.t th:first-child, table.t td:first-child { text-align: left; color: var(--ink-2); }
table.t th { color: var(--ink-2); font-weight: 600; }
.meaning ul { margin: 8px 0 0; padding-left: 20px; }
.meaning li { margin: 0 0 8px; max-width: 80ch; }
#err { position: sticky; top: 0; z-index: 10; background: #c0392b; color: #fff; padding: 10px 16px; font-size: 14px; }
</style>
</head>
<body>
<div id="err" role="alert" hidden></div>
<main>
<h1>The LisaGB forward model</h1>
<p class="sub">Checks of the simulator behind <code>LisaGB</code>: jaxgb's TDI against the textbook equal-arm X, the sensitivity's units against Robson's curve, what the truncated response loses, and the noise generator against the PSD the likelihood whitens with. __SUB__</p>
<div class="card">
  <h2>Does jaxgb's TDI match the standard equal-arm X?</h2>
  <p class="note" id="n-tdi"></p>
  <div class="legend" id="lg-tdi"></div>
  <div id="p-tdi"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Are the sensitivity's units right?</h2>
  <p class="note" id="n-sky"></p>
  <div class="legend" id="lg-sky"></div>
  <div id="p-sky"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What does truncating the response cost?</h2>
  <p class="note" id="n-trunc"></p>
  <div class="row" style="justify-content:flex-start"><span class="k" style="color:var(--ink-2);font-size:13px">Band</span><span class="seg" id="band"></span></div>
  <div class="legend" id="lg-trunc"></div>
  <div id="p-trunc"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Is the noise the Gaussian the likelihood assumes?</h2>
  <p class="note" id="n-noise"></p>
  <div id="p-noise"></div>
</div>
<div class="card meaning">
  <h2>What this means</h2>
  <ul id="meaning"></ul>
</div>
</main>
<script>
const PG = __PAGE__;
const F = PG.forward;
const S = {band: "full"};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

const marks = {};
function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; marks[key](); schedule(); }));
  marks[key]();
}
seg("band", [["narrow", "narrow-B"], ["full", "full-B"]], "band");

// state in the address bar: #band=full
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (["narrow", "full"].includes(q.get("band"))) S.band = q.get("band");
  Object.values(marks).forEach(m => m());
}
const writeHash = () => history.replaceState(null, "", "#" + new URLSearchParams(S).toString());
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
const sup = e => String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
const logLabel = v => { const x = 10 ** v, e = Math.floor(v + 1e-9), m = Math.round(x / 10 ** e); return x >= 1e-2 && x < 1e4 ? String(+x.toPrecision(2)) : `${m > 1 ? m + "×" : ""}10${sup(e)}`; };
const pad = ([lo, hi], f = 0.04) => [lo - f * (hi - lo), hi + f * (hi - lo)];
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
    if (L.span) s += `<rect x="${cx(L.span[0])}" width="${Math.max(1.5, cx(L.span[1]) - cx(L.span[0]))}" y="${m.t}" height="${H - m.t - m.b}" fill="${L.color}" opacity="${L.opacity ?? 0.1}"/>` + (L.text ? `<text class="lab" x="${cx(L.span[0]) + 4}" y="${L.low ? H - m.b - 6 - (L.dy ?? 0) : m.t + 12 + (L.dy ?? 0)}">${L.text}</text>` : "");
    if (L.area) s += `<polygon points="${pts([...L.area.lo, ...L.area.hi.slice().reverse()])}" fill="${L.color}" fill-opacity="0.18"/>`;
    if (L.line) s += `<polyline points="${pts(L.line)}" fill="none" stroke="${L.color}" stroke-width="${L.width ?? 2}" stroke-linejoin="round"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.marks) L.line.forEach(([x, y]) => s += `<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="3.5" fill="${L.color}" stroke="var(--surface)" stroke-width="1.5"/>`);
    if (L.dots) L.dots.forEach(([x, y]) => s += `<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="${L.r ?? 2}" fill="${L.color}" fill-opacity="${L.opacity ?? 0.5}"/>`);
    if (L.hline !== undefined) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(L.hline)}" y2="${Y(L.hline)}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
    if (L.vline !== undefined) s += `<line x1="${X(L.vline)}" x2="${X(L.vline)}" y1="${m.t}" y2="${H - m.b}" stroke="${L.color ?? "var(--ink-2)"}" stroke-width="${L.width ?? 1.2}"${L.dash ? ` stroke-dasharray="${L.dash}"` : ""}/>`;
    if (L.text && L.at) s += `<text class="lab" x="${X(L.at[0]) + (L.dx ?? 0)}" y="${Y(L.at[1]) + (L.dy ?? 0)}" text-anchor="${L.anchor ?? "start"}">${L.text}</text>`;
  }
  s += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" visibility="hidden"/><circle class="ring" r="6" fill="none" stroke="var(--ink)" stroke-width="2" visibility="hidden"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  host.insertAdjacentHTML("beforeend", s + "</svg>");
  const svg = host.lastElementChild, cross = svg.querySelector(".cross"), ring = svg.querySelector(".ring"), hit = svg.querySelector(".hit");
  const tipHost = o.tipHost ?? host;
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W, py = (ev.clientY - r.top) / r.height * H;
    const out = o.hover({x: x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0), px, py, X, Y});
    cross.setAttribute("visibility", out?.cross !== undefined ? "visible" : "hidden");
    ring.setAttribute("visibility", out?.ring ? "visible" : "hidden");
    if (!out) return untip(tipHost);
    if (out.cross !== undefined) { cross.setAttribute("x1", X(out.cross)); cross.setAttribute("x2", X(out.cross)); }
    if (out.ring) { ring.setAttribute("cx", X(out.ring[0])); ring.setAttribute("cy", Y(out.ring[1])); }
    tip(tipHost, ev, out.html);
  });
  hit.addEventListener("pointerleave", () => { untip(tipHost); cross.setAttribute("visibility", "hidden"); ring.setAttribute("visibility", "hidden"); });
}
function nearestPoint(ps, h) {
  let best = null, bd = 14 ** 2;
  ps.forEach((p, i) => { const d = (h.X(p[0]) - h.px) ** 2 + (h.Y(p[1]) - h.py) ** 2; if (d < bd) { bd = d; best = [p, i]; } });
  return best;
}
const bandSpans = (opacity, low) => Object.entries(PG.bands).map(([name, b]) => ({span: b.f0_range.map(Math.log10), color: "var(--ink)", opacity: opacity[name], text: `${name} band`, dy: name === "full" ? 14 : 0, low}));
function table(rows, head) {
  return `<div class="tablewrap"><table class="t"><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function tdiCard() {
  const host = document.getElementById("p-tdi"), t = PG.tdi, pts = t.f0.map((f, i) => [Math.log10(f), Math.log10(t.rel[i])]);
  const anchor = Math.log10(F.tdi_rel_median_high) - 3 * Math.log10(0.01), cube = [Math.log10(1e-4), Math.log10(0.012)].map(x => [x, anchor + 3 * x]);
  host.innerHTML = "";
  chart(host, {x: pad([pts[0][0], pts.at(-1)[0]], 0.02), y: pad([Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))]), xlog: true, ylog: true, xlabel: "f₀ [Hz]", ylabel: "‖X_Babak − X_jaxgb‖ / ‖X_jaxgb‖", label: "TDI convention check", H: 300,
    layers: [{dots: pts, color: "var(--muted)", opacity: 0.4}, {line: cube, color: "var(--ink-2)", width: 1.5, dash: "4 3"}],
    hover: h => { const hit = nearestPoint(pts, h); if (!hit) return null; const [[x, y], i] = hit; return {ring: [x, y], html: `<b>f₀ ${sig(10 ** x * 1e3, 4)} mHz</b>` + trow("var(--muted)", `relative difference ${sig(10 ** y)}, power ${sig(t.power_dev[i])}`)}; }});
  html("lg-tdi", legendDot("var(--muted)", `${t.f0.length.toLocaleString()} of ${F.tdi_sources.toLocaleString()} full-band prior sources`) + legendItem("var(--ink-2)", "∝ f₀³", "4 3"));
  html("n-tdi", `jaxgb's TDI-1.5 X against the standard equal-arm X of Babak et al. 2021 (eq. 14), on the same link envelopes. They agree to ${sig(F.tdi_rel_max_low, 2)} up to ${sig(F.tdi_low_hz * 1e3, 2)} mHz; the difference grows as f₀<sup>${F.tdi_rel_loglog_slope.toFixed(2)}</sup>, to ${sig(F.tdi_rel_median_high, 2)} (median) at ${sig(F.tdi_high_lo_hz * 1e3, 2)}–${sig(F.tdi_high_hi_hz * 1e3, 2)} mHz and ${sig(F.tdi_rel_max, 2)} at most. Powers agree to ${pct(F.tdi_power_dev_max, 2)}, and the hand-built X equals <code>get_tdi</code>'s to ${sig(F.tdi_jaxgb_reimpl_max, 1)}.`);
}

function skyCard() {
  const host = document.getElementById("p-sky"), sa = PG.sky_average, xs = sa.f.map(Math.log10);
  host.innerHTML = "";
  chart(host, {x: pad([xs[0], xs.at(-1)], 0.02), y: [0.3, 0.52], xlog: true, xlabel: "f [Hz]", ylabel: "S_sky / S_Robson", label: "sky-averaged sensitivity against Robson", H: 280,
    layers: [...bandSpans({narrow: 0.07, full: 0.04}, true), {hline: 0.5, color: "var(--ink-2)"}, {line: xs.map((x, i) => [x, sa.ratio[i]]), color: "var(--ink)", marks: true}, {line: xs.map((x, i) => [x, sa.ratio_aet[i]]), color: "var(--muted)", width: 1.5, dash: "5 3"}],
    hover: ({x}) => { const i = nearest(xs, x); return {cross: xs[i], html: `<b>${sig(sa.f[i] * 1e3, 3)} mHz</b>` + trow("var(--ink)", `A + E: ${sa.ratio[i].toFixed(4)}`) + trow("var(--muted)", `A + E + T: ${sa.ratio_aet[i].toFixed(4)}`)}; }});
  html("lg-sky", legendItem("var(--ink)", "A + E") + legendItem("var(--muted)", "A + E + T", "5 3") + legendItem("var(--ink-2)", "1/2", "4 3"));
  html("n-sky", `The Monte-Carlo sky-averaged sensitivity (${F.sky_draws} sky draws per frequency) over the analytic curve of Robson et al. 2019 is flat at 1/2: between ${F.sky_ratio_min_below_4mHz.toFixed(4)} and ${F.sky_ratio_max_below_4mHz.toFixed(4)} below 4 mHz, log-log slope ${F.sky_slope.toFixed(4)} over ${sig(F.sky_grid_lo_hz * 1e3, 2)}–${sig(F.sky_fit_hi_hz * 1e3, 2)} mHz. Writing the noise as a fractional length instead tilts it to slope ${F.sky_slope_length.toFixed(3)}. It falls above 4 mHz, where Robson's curve has no transfer-function detail. Shaded are the bands' f₀ ranges.`);
}

function truncCard() {
  const host = document.getElementById("p-trunc"), tr = PG.bands[S.band].truncation, R = tr.response_points;
  const pts = tr.bin_offset.map((o, i) => [o, Math.log10(tr.rel[i])]);
  host.innerHTML = `<div class="pair"><div id="tr-a"></div><div id="tr-b"></div></div>`;
  chart(document.getElementById("tr-a"), {x: [-0.5, 0.5], y: pad([Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))]), ylog: true, xlabel: "f₀T − rint(f₀T) [bins]", ylabel: "relative L2 error of the response", label: "truncation error against bin offset", tipHost: host,
    layers: [{dots: pts, color: "var(--muted)", opacity: 0.35}],
    hover: h => { const hit = nearestPoint(pts, h); if (!hit) return null; const [[x, y], i] = hit; return {ring: [x, y], html: trow("var(--muted)", `offset ${x.toFixed(3)}, error ${sig(10 ** y)}, f₀ ${sig(tr.f0[i] * 1e3, 4)} mHz`)}; }});
  const lx = tr.ladder_points.map(Math.log10), q = tr.ladder_q.map(row => row.map(Math.log10));
  chart(document.getElementById("tr-b"), {x: pad([lx[0], lx.at(-1)], 0.04), y: pad([Math.min(...q[0]), Math.max(...q[2])]), xlog: true, ylog: true, xlabel: "response points R", ylabel: "relative L2 error, median and 16–84%", label: "truncation ladder", tipHost: host,
    layers: [{area: {lo: lx.map((x, i) => [x, q[0][i]]), hi: lx.map((x, i) => [x, q[2][i]])}, color: "var(--ink)"}, {line: lx.map((x, i) => [x, q[1][i]]), color: "var(--ink)", marks: true}, {vline: Math.log10(R), color: "var(--ink-2)", dash: "4 3"}, {text: `this band, R = ${R}`, at: [Math.log10(R), Math.min(...q[0])], dx: 6}],
    hover: ({x}) => { const i = nearest(lx, x); return {cross: lx[i], html: `<b>R = ${tr.ladder_points[i]}</b>` + trow("var(--ink)", `median ${sig(10 ** q[1][i])}, 16–84% ${sig(10 ** q[0][i])}–${sig(10 ** q[2][i])}`)}; }});
  html("lg-trunc", legendDot("var(--muted)", `${tr.f0.length.toLocaleString()} of ${tr.sources.toLocaleString()} ${S.band}-band prior sources`) + legendItem("var(--ink)", "median over 16–84%"));
  html("n-trunc", `A source's ${R}-point response against a ${PG.reference_points}-point jaxgb reference. The boxcar's 1/k leakage tails fold back onto the R bins, so the error is set by f₀'s offset from its bin (left): median ${pct(tr.rel_median_on_bin, 2)} on a bin, ${pct(tr.rel_median_off_bin, 1)} halfway, ${pct(tr.rel_median, 2)} overall. It falls as R<sup>${tr.ladder_slope.toFixed(2)}</sup> (right). The SNR is kept to ${sig(tr.rho_ratio_max_dev, 2)}, but the residual's own SNR has median ${sig(tr.rho_res_median, 2)} and exceeds 1 for ${pct(tr.rho_res_over_1, 0)} of sources. It bounds how real the simulator is, not the flow-vs-MCMC comparison, whose two sides share it.`);
}

function noiseCard() {
  const f4 = x => x.toFixed(5), e2 = x => sig(x, 2);
  html("p-noise", table([
    ["Var Re z", f4(F.noise_var_re), "0.5"], ["Var Im z", f4(F.noise_var_im), "0.5"], ["corr(Re z, Im z)", e2(F.noise_corr_re_im), `0 ± ${e2(F.noise_corr_floor)}`],
    ["E|z|² over A, E, T", f4(F.noise_abs2), "1"], ["E|z|², A / E / T", [F.noise_abs2_a, F.noise_abs2_e, F.noise_abs2_t].map(f4).join(" / "), "1"],
    ["corr, adjacent bins", e2(F.noise_corr_adjacent), `0 ± ${e2(F.noise_corr_floor)}`], ["corr, A with E / T", `${e2(F.noise_corr_a_e)} / ${e2(F.noise_corr_a_t)}`, `0 ± ${e2(F.noise_corr_floor)}`],
    ["PSD tilt, A / E / T", [F.noise_psd_tilt_a, F.noise_psd_tilt_e, F.noise_psd_tilt_t].map(e2).join(" / "), `< ${e2(F.noise_psd_tilt_floor)} (Monte-Carlo floor)`],
    ["⟨n|n⟩ over A, E: mean", F.noise_nn_mean.toFixed(1), `${F.noise_nn_dof} (χ²)`], ["⟨n|n⟩: variance", F.noise_nn_var.toFixed(0), `${2 * F.noise_nn_dof}`],
    [`KS p, ⟨n|n⟩ against χ²(${F.noise_nn_dof})`, F.noise_nn_ks_p.toFixed(3), "uniform"], ["KS p, Re z against N(0, 1/2)", F.noise_ks_re_p.toFixed(3), "uniform"],
  ], ["whitened noise", "measured", "expected"]));
  html("n-noise", `<code>sample_observation</code>'s noise, whitened with the PSD the likelihood uses: ${F.noise_draws.toLocaleString()} pure-noise windows of ${F.noise_window_bins} bins at ${sig(F.noise_f_hz * 1e3, 4)} mHz. It is a standard complex Gaussian at that PSD, with no tilt, cross-channel or adjacent-bin correlation beyond the Monte-Carlo floor.`);
}

function meaning() {
  const n = PG.bands.narrow.truncation, f = PG.bands.full.truncation;
  html("meaning", [
    `<b>jaxgb's TDI is the textbook one where the bands live.</b> It matches Babak et al.'s equal-arm X to ${sig(F.tdi_rel_max_low, 2)} below ${sig(F.tdi_low_hz * 1e3, 2)} mHz, and the gap grows as f₀³ to ${sig(F.tdi_rel_max, 2)} at worst; the powers agree to ${pct(F.tdi_power_dev_max, 2)}.`,
    `<b>The units are right.</b> The sky-averaged sensitivity over Robson et al.'s curve is flat at 1/2 below 4 mHz: ${F.sky_ratio_min_below_4mHz.toFixed(4)}–${F.sky_ratio_max_below_4mHz.toFixed(4)}, slope ${F.sky_slope.toFixed(4)}. Noise written as a fractional length would tilt it to slope ${F.sky_slope_length.toFixed(1)}.`,
    `<b>The noise is the Gaussian the likelihood assumes.</b> ⟨n|n⟩ against χ²(${F.noise_nn_dof}) gives KS p = ${F.noise_nn_ks_p.toFixed(2)}, Re z against N(0, 1/2) p = ${F.noise_ks_re_p.toFixed(2)}, and no correlation clears the Monte-Carlo floor.`,
    `<b>The truncated response keeps the SNR, not the shape.</b> SNR to ${sig(Math.max(n.rho_ratio_max_dev, f.rho_ratio_max_dev), 2)}, but the spectrum is off by ${pct(n.rel_median, 1)} (narrow-B) and ${pct(f.rel_median, 2)} (full-B), and the residual's SNR has median ${n.rho_res_median.toFixed(1)} (narrow-B) and ${sig(f.rho_res_median, 2)} (full-B). Training, injections and the MCMC reference share that response, so it bounds realism, not the comparison.`,
  ].map(t => `<li>${t}</li>`).join(""));
}

function render() { tdiCard(); skyCard(); truncCard(); noiseCard(); meaning(); writeHash(); }
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


def page_data() -> dict:
    """What the page plots, read from data/."""
    forward = json.loads((DATA / "forward-model.json").read_text())
    points = json.loads((DATA / "response-points.json").read_text())
    tdi, sky, trunc = (
        np.load(DATA / f"{n}.npz") for n in ("tdi", "sky", "response-points")
    )

    bands = {}
    for band in BANDS:
        summary = {
            k[len(band) + 1 :]: v for k, v in points.items() if k.startswith(f"{band}_")
        }
        bands[band] = dict(
            f0_range=[summary["f0_lo_hz"], summary["f0_hi_hz"]],
            truncation=summary
            | dict(
                f0=trunc[f"{band}_f0"][::STRIDE].tolist(),
                bin_offset=trunc[f"{band}_bin_offset"][::STRIDE].tolist(),
                rel=trunc[f"{band}_rel"][::STRIDE].tolist(),
                ladder_points=trunc[f"{band}_ladder_points"].tolist(),
                ladder_q=np.quantile(
                    trunc[f"{band}_ladder_rel"], [0.16, 0.5, 0.84], axis=1
                ).tolist(),
            ),
        )
    order = np.argsort(tdi["f0"])[::STRIDE]
    return rounded(
        dict(
            bands=bands,
            forward=forward,
            tdi=dict(
                f0=tdi["f0"][order].tolist(),
                rel=tdi["rel"][order].tolist(),
                power_dev=tdi["power_dev"][order].tolist(),
            ),
            sky_average={k: sky[k].tolist() for k in sky.files},
            reference_points=points["reference_points"],
        )
    )


def main() -> None:
    data = page_data()
    commit = data["forward"]["commit"][:7]
    page = PAGE.replace("__PAGE__", json.dumps(data, separators=(",", ":"))).replace(
        "__SUB__",
        f"Built {date.today().isoformat()} from the records of commit <code>{commit}</code>.",
    )
    out = FOLDER / "forward-model.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
