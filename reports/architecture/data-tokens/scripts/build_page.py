"""Write data-tokens.html from data/tokens.json: the STFT patch tokens per band and where each source's power lands among them.

Stdlib only; run from the repo root:
    uv run --frozen python reports/architecture/data-tokens/scripts/build_page.py
"""

import json
import statistics
from pathlib import Path

REPORT = Path(__file__).parents[1]
DAY = 86400.0


def quantiles(values: list[float], qs=(0.05, 0.5, 0.95)) -> list[float]:
    s = sorted(values)
    return [s[min(len(s) - 1, int(q * len(s)))] for q in qs]


def summary(band: dict) -> dict:
    """The numbers the tiles and the README quote."""
    pop = band["population"]
    col = lambda k: [s[k] for s in pop]
    loud = [s for s in pop if s["snr"] >= 100]
    f_lo, f_hi = band["f0_range"]
    top = [s for s in pop if s["f0"] >= f_lo * (f_hi / f_lo) ** 0.9]
    bottom = [s for s in pop if s["f0"] <= f_lo * (f_hi / f_lo) ** 0.1]
    pick = lambda rows, k: statistics.median(s[k] for s in rows) if rows else None
    return dict(
        sources=len(pop),
        tokens99=quantiles(col("tokens99")),
        tokens90=quantiles(col("tokens90")),
        greedy99=quantiles(col("greedy99")),
        greedy90=quantiles(col("greedy90")),
        width99=quantiles(col("width99")),
        own_share=quantiles(col("own_share")),
        top_share=quantiles(col("top_share")),
        tokens99_max=max(col("tokens99")),
        guard_share_max=max(col("guard_share")),
        guard_sources=sum(s["guard_share"] > 1e-3 for s in pop),
        tokens99_top=pick(top, "tokens99"),
        tokens99_bottom=pick(bottom, "tokens99"),
        greedy99_loud=pick(loud, "greedy99"),
        sample_days=band["t_obs"] / band["patch"] / DAY,
        token_nhz=band["patch"] / band["t_obs"] * 1e9,
        drawable_tokens=(band["window_bins"] - 2 * band["guard"]) // band["patch"],
        guard_tokens=band["guard"] / band["patch"],
    )


def report(data: dict) -> None:
    for name, band in data["bands"].items():
        print(name, json.dumps(band["summary"]))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA data tokens</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e;
  --s1: #2a78d6; --s2: #eb6834; --s3: #199e70; --s4: #a24bd1; --guard: rgba(11,11,11,0.06);
  --seq: #cde2fb,#9ec5f4,#6da7ec,#3987e5,#256abf,#184f95,#0d366b;
  --div: #1c5cab,#f0efec,#c2362b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7;
    --s1: #3987e5; --s2: #d95926; --s3: #1baf7a; --s4: #b366e0; --guard: rgba(255,255,255,0.07);
    --seq: #0d366b,#184f95,#256abf,#3987e5,#6da7ec,#9ec5f4,#cde2fb;
    --div: #5598e7,#383835,#e5675c;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
a { color: var(--s1); }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 90ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 16px; }
.bar .k, .controls .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: auto; display: inline-block; flex: none; overflow: visible; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; margin: 0 0 12px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; max-width: 320px; }
.tip b { color: var(--ink); font-weight: 600; }
table.t { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
table.t th, table.t td { padding: 3px 10px 3px 0; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
table.t th:first-child, table.t td:first-child { text-align: left; }
table.t th { color: var(--ink-2); font-weight: 600; }
.scroll { overflow-x: auto; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 80ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>What does the data stream see?</h1>
<p class="sub">How <code>stft_tokens</code> cuts a whitened window into patch tokens, and how many of them one galactic binary's power covers. Draws from the visible-window prior on the working-tree code; the per-bin window itself is in the <a href="../../diagnostics/window-viewer/window-viewer.html">window viewer</a>.</p>
<div class="bar">
  <div><span class="k">Band</span><span class="seg" id="band"></span></div>
</div>
<div class="tiles" id="tiles"></div>

<div class="card">
  <h2>How does a window become tokens?</h2>
  <p class="note" id="layout-note"></p>
  <div id="p-layout"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>What do the tokens of a drawn window look like?</h2>
  <p class="note">Top: the whitened magnitude per bin, asinh of the rms over A, E, T (one noise σ ≈ 0.88), with each source's central-99% SNR² span shaded and its f0 marked. Bottom: the tokens, one column each, rows the token's time samples over the observation. Grey columns are guard tokens. Hover for values.</p>
  <div class="controls">
    <div><span class="k">Window</span><span class="seg" id="obs"></span></div>
    <div><span class="k">Content</span><span class="seg" id="content"></span></div>
    <div><span class="k">Channel</span><span class="seg" id="chan"></span></div>
    <div><span class="k">Feature</span><span class="seg" id="feat"></span></div>
  </div>
  <div class="legend" id="lg-heat"></div>
  <div id="p-heat"></div><div class="tip"></div>
  <div class="scroll" id="p-src"></div>
</div>

<div class="card">
  <h2>How many tokens does one source cover, across the band and the sky?</h2>
  <p class="note">Each point is one prior source (four per window). The span is the tokens touched by the central 99% (or 90%) of its SNR²; "fewest" counts only the tokens needed to hold that share. The line is the analytic span at the ecliptic plane (|cos β| = 1) with ḟ 5σ above the prior's mean at f0: 2 × (Doppler + drift + antenna sidebands) bins.</p>
  <div class="controls"><div><span class="k">Count</span><span class="seg" id="measure"></span></div></div>
  <div class="legend" id="lg-span"></div>
  <div id="p-span"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>Does the analytic Doppler band predict the measured span?</h2>
  <p class="note">Measured central-99% width against 2 × (f0 v/c |cos β| T + Ḟ T² + 4 T/yr) bins, β the ecliptic latitude. On the line the analytic band is the 99% width.</p>
  <div class="legend" id="lg-pred"></div>
  <div id="p-pred"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>How concentrated is a source's power?</h2>
  <p class="note">Share of a source's SNR² in the token holding f0, and in its loudest token, as cumulative fractions of sources.</p>
  <div class="legend" id="lg-share"></div>
  <div id="p-share"></div><div class="tip"></div>
</div>

<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const S = {band: "full", obs: 0, content: "data", chan: 0, feat: "amp", measure: "tokens99"};
const marks = {};
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const B = () => D.bands[S.band];
const SRC = ["--s1", "--s2", "--s3", "--s4"];
const fmt = (v, n = 3) => v >= 1000 ? Math.round(v).toLocaleString("en") : Number(v.toPrecision(n)).toString();

function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = typeof S[key] === "number" ? Number(b.dataset.v) : b.dataset.v; schedule(); }));
}
seg("band", [["narrow", "narrow"], ["full", "full"]], "band");
seg("obs", D.bands.full.observations.map((_, i) => [i, String(i + 1)]), "obs");
seg("content", [["data", "data"], ["signal", "signal only"]], "content");
seg("chan", [[0, "A"], [1, "E"], [2, "T"]], "chan");
seg("feat", [["amp", "asinh|z|"], ["re", "gated Re"], ["im", "gated Im"]], "feat");
seg("measure", [["tokens99", "span 99%"], ["tokens90", "span 90%"], ["greedy99", "fewest 99%"], ["greedy90", "fewest 90%"]], "measure");

document.addEventListener("mousemove", e => {
  document.querySelectorAll(".tip").forEach(t => { t.style.display = "none"; });
  const t = e.target.closest("[data-tip]"), card = t && t.closest(".card");
  if (!t || !card) return;
  const tip = card.querySelector(".tip"), box = card.getBoundingClientRect();
  tip.innerHTML = t.dataset.tip;
  tip.style.display = "block";
  tip.style.left = Math.max(4, Math.min(e.clientX - box.left + 14, box.width - tip.offsetWidth - 8)) + "px";
  tip.style.top = e.clientY - box.top + 14 + "px";
});

// colour ramps read from the theme
function ramp(stops, v) {
  v = Math.max(0, Math.min(1, v)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(v)), f = v - i;
  const hex = s => [1, 3, 5].map(k => parseInt(s.slice(k, k + 2), 16));
  const a = hex(stops[i]), b = hex(stops[i + 1]);
  return `rgb(${a.map((x, k) => Math.round(x + f * (b[k] - x))).join(",")})`;
}
const seqStops = () => css("--seq").split(",").map(s => s.trim());
const divStops = () => css("--div").split(",").map(s => s.trim());
function rampLegend(stops, lo, hi, label) {
  const id = "g" + Math.random().toString(36).slice(2, 7);
  return `<span><svg width="120" height="12"><defs><linearGradient id="${id}">${stops.map((c, i) => `<stop offset="${i / (stops.length - 1)}" stop-color="${c}"/>`).join("")}</linearGradient></defs><rect width="120" height="12" rx="2" fill="url(#${id})"/></svg>${label}: ${lo} to ${hi}</span>`;
}

function tiles() {
  const b = B(), s = b.summary, tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  document.getElementById("tiles").innerHTML = [
    tile("Tokens per window", `${b.tokens}`, `${b.window_bins} bins ÷ patch ${b.patch}; ${b.features} features each`),
    tile("Drawable tokens", `${s.drawable_tokens}`, `${fmt(s.guard_tokens)} guard token${s.guard_tokens === 1 ? "" : "s"} each side (${b.guard} bins)`),
    tile("One token", `${fmt(s.token_nhz)} nHz`, `${b.patch} time samples, ${fmt(s.sample_days)} d apart`),
    tile("Source span, median", `${s.tokens99[1]} tokens`, `central 99% of SNR²; 5–95%: ${s.tokens99[0]}–${s.tokens99[2]}, max ${s.tokens99_max}`),
    tile("Fewest tokens for 99%", `${s.greedy99[1]}`, `median; 5–95%: ${s.greedy99[0]}–${s.greedy99[2]}`),
  ].join("");
}

// the window's bins, its guard and its tokens to scale
function layout() {
  const b = B(), s = b.summary, host = document.getElementById("p-layout");
  const W = Math.max(320, host.clientWidth), H = 120, l = 10, r = 10, X = v => l + v / b.window_bins * (W - l - r);
  document.getElementById("layout-note").innerHTML = `A ${S.band}-band window is ${b.window_bins} rfft bins of A, E, T (${fmt(b.window_bins / b.t_obs * 1e6)} µHz). <code>stft_tokens</code> groups them ${b.patch} at a time and takes the orthonormal inverse DFT of each group, so a token is the band's complex time series at ${b.patch} samples ${fmt(s.sample_days)} d apart over the ${fmt(b.t_obs / 86400 / 365.25)}-yr observation. Its ${b.features} features are asinh|z| and the gated phase z/√(1+|z|²) (real, imaginary) of each sample and channel. f0 is drawn only where the whole response fits, so ${b.guard} bins (${fmt(s.guard_tokens)} token${s.guard_tokens === 1 ? "" : "s"}) at each edge hold only the neighbours' tails.`;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="window layout">`;
  svg += `<rect x="${X(0)}" y="34" width="${X(b.guard) - X(0)}" height="36" fill="var(--guard)"/><rect x="${X(b.window_bins - b.guard)}" y="34" width="${X(b.window_bins) - X(b.window_bins - b.guard)}" height="36" fill="var(--guard)"/>`;
  const every = Math.max(1, Math.ceil(b.tokens / ((W - l - r) / 4)));
  for (let n = 0; n < b.tokens; n++) {
    const x0 = X(n * b.patch), x1 = X((n + 1) * b.patch), guard = n * b.patch < b.guard || (n + 1) * b.patch > b.window_bins - b.guard;
    if (n % every === 0) svg += `<line x1="${x0}" x2="${x0}" y1="34" y2="70" stroke="var(--axis)"/>`;
    svg += `<rect x="${x0}" y="34" width="${Math.max(x1 - x0, 1)}" height="36" fill="transparent" data-tip="${esc(`<b>token ${n}</b><br>bins ${n * b.patch}–${(n + 1) * b.patch - 1}${guard ? " (guard)" : ""}<br>RoPE position ${n * b.patch + (b.patch - 1) / 2} bins`)}"/>`;
  }
  svg += `<rect x="${X(0)}" y="34" width="${X(b.window_bins) - X(0)}" height="36" fill="none" stroke="var(--ink-2)"/>`;
  svg += `<text class="lbl" x="${X(b.window_bins / 2)}" y="26" text-anchor="middle">${b.tokens} tokens × ${b.patch} bins, drawable f0 in the middle ${s.drawable_tokens}</text>`;
  svg += `<text x="${X(0)}" y="88">bin 0</text><text x="${X(b.window_bins)}" y="88" text-anchor="end">${b.window_bins}</text>`;
  svg += `<text x="${X(b.guard / 2)}" y="104" text-anchor="${b.guard / b.window_bins < 0.05 ? "start" : "middle"}">guard</text><text x="${X(b.window_bins - b.guard / 2)}" y="104" text-anchor="${b.guard / b.window_bins < 0.05 ? "end" : "middle"}">guard</text>`;
  const med = s.width99[1];
  svg += `<rect x="${X(b.window_bins / 2 - med / 2)}" y="76" width="${X(med) - X(0)}" height="8" rx="2" fill="var(--s1)"/><text class="lbl" x="${X(b.window_bins / 2)}" y="104" text-anchor="middle">median source, central 99%: ${med} bins</text>`;
  host.innerHTML = svg + "</svg>";
}

function heat() {
  const b = B(), o = b.observations[Math.min(S.obs, b.observations.length - 1)], host = document.getElementById("p-heat");
  const W = Math.max(320, host.clientWidth), l = 44, r = 8, topH = 110, gap = 14, cell = 7, rows = b.patch, H = topH + gap + rows * cell + 30;
  const X = v => l + v / b.window_bins * (W - l - r);
  const bins = S.content === "data" ? o.bin_data : o.bin_signal;
  const ymax = Math.max(...o.bin_data, ...o.bin_signal) / 100 * 1.05;
  const Y = v => 8 + (1 - v / ymax) * (topH - 16);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="tokens of one window">`;
  // guard shading over both panels
  for (const [a, z] of [[0, b.guard], [b.window_bins - b.guard, b.window_bins]]) svg += `<rect x="${X(a)}" y="0" width="${X(z) - X(a)}" height="${topH + gap + rows * cell}" fill="var(--guard)"/>`;
  o.sources.forEach((s, i) => {
    svg += `<rect x="${X(s.lo99)}" y="4" width="${Math.max(1, X(s.hi99 + 1) - X(s.lo99))}" height="${topH - 8}" fill="var(${SRC[i]})" opacity="0.14"/>`;
    svg += `<line x1="${X(s.f0_bin)}" x2="${X(s.f0_bin)}" y1="4" y2="${topH - 4}" stroke="var(${SRC[i]})" stroke-width="2"/>`;
  });
  [0, 1, 2, 3, 4, 5, 6].filter(v => v <= ymax).forEach(v => svg += `<line x1="${l}" x2="${W - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  svg += `<path d="${bins.map((v, i) => `${i ? "L" : "M"}${X(i + 0.5).toFixed(1)},${Y(v / 100).toFixed(1)}`).join("")}" fill="none" stroke="var(--ink-2)" stroke-width="1"/>`;
  svg += `<text transform="translate(12 ${topH / 2}) rotate(-90)" text-anchor="middle">asinh|y|</text>`;
  // token heatmap
  const ch = o[S.content], y0 = topH + gap, stops = S.feat === "amp" ? seqStops() : divStops();
  const amax = Math.max(1, ...ch.amp[S.chan].flat()) / 100;
  for (let n = 0; n < b.tokens; n++) for (let p = 0; p < rows; p++) {
    const a = ch.amp[S.chan][n][p] / 100, ph = ch.phase[S.chan][n][p] / 100, z = Math.sinh(a), gate = z / Math.sqrt(1 + z * z);
    const v = S.feat === "amp" ? a / amax : S.feat === "re" ? (gate * Math.cos(ph) + 1) / 2 : (gate * Math.sin(ph) + 1) / 2;
    const tip = `<b>token ${n}, sample ${p}</b> (day ${Math.round(p * b.t_obs / b.patch / 86400)})<br>channel ${"AET"[S.chan]}, ${S.content === "data" ? "data" : "signal only"}<br>asinh|z| ${a.toFixed(2)}, phase ${ph.toFixed(2)} rad<br>gated Re ${(gate * Math.cos(ph)).toFixed(2)}, Im ${(gate * Math.sin(ph)).toFixed(2)}`;
    svg += `<rect x="${X(n * b.patch)}" y="${y0 + p * cell}" width="${X(b.patch) - X(0) + 0.3}" height="${cell + 0.3}" fill="${ramp(stops, v)}" data-tip="${esc(tip)}"/>`;
  }
  svg += `<text x="${l - 6}" y="${y0 + 8}" text-anchor="end">t=0</text><text x="${l - 6}" y="${y0 + rows * cell}" text-anchor="end">T</text>`;
  const tickEvery = b.tokens > 64 ? 32 : b.tokens > 16 ? 8 : 1;
  for (let n = 0; n <= b.tokens; n += tickEvery) svg += `<line x1="${X(n * b.patch)}" x2="${X(n * b.patch)}" y1="${y0 + rows * cell}" y2="${y0 + rows * cell + 4}" stroke="var(--axis)"/><text x="${X(n * b.patch)}" y="${y0 + rows * cell + 16}" text-anchor="middle">${n}</text>`;
  svg += `<text x="${(l + W - r) / 2}" y="${H - 2}" text-anchor="middle">token (window starts at bin ${o.start.toLocaleString("en")}, f = ${(o.f * 1e3).toFixed(4)} mHz)</text>`;
  host.innerHTML = svg + "</svg>";
  document.getElementById("lg-heat").innerHTML = o.sources.map((s, i) => `<span><svg width="12" height="12"><rect width="12" height="12" rx="2" fill="var(${SRC[i]})"/></svg>source ${i + 1}</span>`).join("")
    + (S.feat === "amp" ? rampLegend(stops, 0, amax.toFixed(1), "asinh|z|") : rampLegend(stops, -1, 1, S.feat === "re" ? "gated Re" : "gated Im"));
  const cosb = s => Math.abs(Math.cos(s.beta));
  document.getElementById("p-src").innerHTML = `<table class="t"><thead><tr><th>source</th><th>SNR</th><th>f0 bin</th><th>f0 token</th><th>99% bins</th><th>99% tokens</th><th>|cos β|</th></tr></thead><tbody>${o.sources.map((s, i) => `<tr><td>${i + 1}</td><td>${s.snr.toFixed(1)}</td><td>${s.f0_bin.toFixed(2)}</td><td>${Math.floor(s.f0_bin / b.patch)}</td><td>${s.lo99}–${s.hi99}</td><td>${Math.floor(s.lo99 / b.patch)}–${Math.floor(s.hi99 / b.patch)} (${Math.floor(s.hi99 / b.patch) - Math.floor(s.lo99 / b.patch) + 1})</td><td>${cosb(s).toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
}

// log-x scatter frame
function frame(host, o) {
  const W = Math.max(320, host.clientWidth), H = 300, M = {l: 56, r: 14, t: 10, b: 40};
  const lx = v => o.logx ? Math.log10(v) : v, ly = v => o.logy ? Math.log10(v) : v;
  const X = v => M.l + (lx(v) - lx(o.x[0])) / (lx(o.x[1]) - lx(o.x[0])) * (W - M.l - M.r);
  const Y = v => M.t + (ly(o.y[1]) - ly(v)) / (ly(o.y[1]) - ly(o.y[0])) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(o.label)}">`;
  o.yt.forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v)}</text>`);
  o.xt.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${o.xfmt ? o.xfmt(v) : fmt(v)}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${o.xlabel}</text><text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  return {W, H, X, Y, M, s};
}
const ticks = (lo, hi) => { const out = []; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); } return out; };

function spanPanel() {
  const b = B(), host = document.getElementById("p-span"), pop = b.population, key = S.measure, stops = seqStops().slice(1);
  const vals = pop.map(s => s[key]), c = b.curve;
  const analytic = c.f0.map((f0, i) => 2 * (c.doppler[i] + c.drift[i] + c.antenna) / b.patch + 1);
  const ymax = Math.max(...vals, ...analytic) * 1.3;
  const f = frame(host, {x: b.f0_range, y: [0.8, ymax], logx: true, logy: true, xt: ticks(...b.f0_range), yt: ticks(1, ymax), xfmt: v => (v * 1e3).toPrecision(1), xlabel: "f0 [mHz]", ylabel: "tokens", label: "tokens spanned against f0"});
  let s = f.s;
  s += `<path d="${c.f0.map((v, i) => `${i ? "L" : "M"}${f.X(v).toFixed(1)},${f.Y(analytic[i]).toFixed(1)}`).join("")}" fill="none" stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 4"/>`;
  for (const p of pop) {
    const cb = Math.abs(Math.cos(p.beta));
    const tip = `<b>f0 ${(p.f0 * 1e3).toFixed(4)} mHz, SNR ${p.snr.toFixed(1)}</b><br>99% span ${p.tokens99} tokens (${p.width99} bins), 90% ${p.tokens90}<br>fewest for 99%: ${p.greedy99}, for 90%: ${p.greedy90}<br>|cos β| ${cb.toFixed(2)}, ḟ ${p.fdot.toExponential(2)} Hz/s<br>Doppler ${p.doppler.toFixed(1)}, drift ${p.drift.toFixed(2)} bins`;
    s += `<circle cx="${f.X(p.f0)}" cy="${f.Y(p[key] * (1 + 0.06 * (Math.sin(p.f0 * 1e9) )))}" r="3.5" fill="${ramp(stops, cb)}" stroke="var(--surface)" stroke-width="1" data-tip="${esc(tip)}"/>`;
  }
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-span").innerHTML = rampLegend(stops, 0, 1, "|cos β|, ecliptic") + `<span><svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 4"/></svg>analytic span at |cos β| = 1, ḟ at +5σ</span><span>points jittered ±6% vertically</span>`;
}

function predPanel() {
  const b = B(), host = document.getElementById("p-pred"), pop = b.population, stops = seqStops().slice(1), ant = b.curve.antenna;
  const pred = p => 2 * (p.doppler + p.drift + ant);
  const all = pop.flatMap(p => [pred(p), p.width99]);
  const lo = Math.min(...all) / 1.3, hi = Math.max(...all) * 1.3;
  const f = frame(host, {x: [lo, hi], y: [lo, hi], logx: true, logy: true, xt: ticks(lo, hi), yt: ticks(lo, hi), xlabel: "analytic band 2(Doppler + drift + sidebands) [bins]", ylabel: "measured 99% width [bins]", label: "measured against analytic width"});
  let s = f.s + `<line x1="${f.X(lo)}" x2="${f.X(hi)}" y1="${f.Y(lo)}" y2="${f.Y(hi)}" stroke="var(--ref)" stroke-dasharray="5 4"/>`;
  for (const p of pop) {
    const cb = Math.abs(Math.cos(p.beta));
    s += `<circle cx="${f.X(pred(p))}" cy="${f.Y(p.width99)}" r="3.5" fill="${ramp(stops, cb)}" stroke="var(--surface)" stroke-width="1" data-tip="${esc(`<b>f0 ${(p.f0 * 1e3).toFixed(4)} mHz</b><br>measured ${p.width99} bins, analytic ${pred(p).toFixed(1)}<br>|cos β| ${cb.toFixed(2)}, SNR ${p.snr.toFixed(1)}`)}"/>`;
  }
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-pred").innerHTML = rampLegend(stops, 0, 1, "|cos β|, ecliptic") + `<span><svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="var(--ref)" stroke-dasharray="5 4"/></svg>measured = analytic</span>`;
}

function sharePanel() {
  const b = B(), host = document.getElementById("p-share"), pop = b.population;
  const f = frame(host, {x: [0, 1], y: [0, 1], xt: [0, 0.2, 0.4, 0.6, 0.8, 1], yt: [0, 0.25, 0.5, 0.75, 1], xlabel: "share of the source's SNR² in one token", ylabel: "fraction of sources below", label: "token shares"});
  let s = f.s;
  [["own_share", "--s1", "f0's own token"], ["top_share", "--s2", "loudest token"]].forEach(([k, c, name]) => {
    const v = pop.map(p => p[k]).sort((a, z) => a - z);
    s += `<path d="${v.map((x, i) => `${i ? "L" : "M"}${f.X(x).toFixed(1)},${f.Y((i + 1) / v.length).toFixed(1)}`).join("")}" fill="none" stroke="var(${c})" stroke-width="2"/>`;
    const med = v[Math.floor(v.length / 2)];
    s += `<circle cx="${f.X(med)}" cy="${f.Y(0.5)}" r="4" fill="var(${c})" stroke="var(--surface)" stroke-width="2" data-tip="${esc(`<b>${name}</b><br>median share ${med.toFixed(2)}`)}"/><text class="lbl" x="${f.X(med) + 7}" y="${f.Y(0.5) - 6}">${name}: median ${med.toFixed(2)}</text>`;
  });
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-share").innerHTML = `<span><svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="var(--s1)" stroke-width="2"/></svg>token holding f0</span><span><svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="var(--s2)" stroke-width="2"/></svg>loudest token</span>`;
}

function writeHash() { history.replaceState(null, "", "#" + new URLSearchParams({band: S.band, obs: S.obs, content: S.content, chan: S.chan, feat: S.feat, count: S.measure}).toString()); }
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (D.bands[h.get("band")]) S.band = h.get("band");
  if (h.get("obs") !== null && !isNaN(+h.get("obs"))) S.obs = +h.get("obs");
  if (["data", "signal"].includes(h.get("content"))) S.content = h.get("content");
  if (["0", "1", "2"].includes(h.get("chan"))) S.chan = +h.get("chan");
  if (["amp", "re", "im"].includes(h.get("feat"))) S.feat = h.get("feat");
  if (["tokens99", "tokens90", "greedy99", "greedy90"].includes(h.get("count"))) S.measure = h.get("count");
}
function render() {
  for (const k of Object.keys(marks)) marks[k]();
  tiles(); layout(); heat(); spanPanel(); predPanel(); sharePanel();
  writeHash();
}
let pending = false;
function schedule() { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; render(); }); }
addEventListener("resize", schedule);
addEventListener("hashchange", () => { readHash(); schedule(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
readHash();
render();
</script>
</body>
</html>
"""



def main() -> None:
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "tokens.json"
    data = json.loads((REPORT / "data" / name).read_text())
    for band in data["bands"].values():
        band["summary"] = summary(band)
    report(data)
    page = PAGE.replace("__MEANS__", MEANS_HTML).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = REPORT / "data-tokens.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


MEANS_HTML = """
<ul>
<li><b>A source is a few tokens, not one.</b> Its central 99% of SNR² spans a median 2 tokens on narrow and 3 on full (5–95%: 2–3 and 2–5, at most 3 and 7). On full the span grows with f0: a median 3 tokens (27 bins) below 1.5 mHz, 4 tokens (54 bins) above 3 mHz. Narrow sources all sit at 0.74–1.26 mHz.</li>
<li><b>The token holding f0 carries most of the power on narrow, little more than half on full.</b> It holds a median 78% of SNR² on narrow and 57% on full, and 4% (narrow) and 29% (full) of sources keep less than half there.</li>
<li><b>The sky barely changes the span under this prior.</b> The sources' sky prior puts them near the ecliptic, so |cos β| has median 0.989 and the Doppler band is nearly at its maximum for almost every source. Span varies with f0, not with sky position.</li>
<li><b>The analytic band is slightly tight.</b> The measured 99% width is 1.03–1.24 (narrow) and 0.96–1.20 (full) of 2 × (Doppler + drift + 4 T/yr) at 5–95%, median 1.13 and 1.07; the extremes are 0.79–2.1 and 0.87–2.9.</li>
<li><b>The guard holds real signal for some sources.</b> More than 1% of SNR² falls in a guard token for 22% of narrow sources (up to 64%) and 10% of full sources (up to 55%).</li>
</ul>
"""

if __name__ == "__main__":
    main()
