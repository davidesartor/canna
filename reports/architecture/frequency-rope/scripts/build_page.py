"""Write frequency-rope.html from data/rope.json: RoPE positions, periods and attention kernel on the shared frequency axis.

Stdlib only; run from the repo root:
    uv run --frozen python reports/architecture/frequency-rope/scripts/build_page.py
"""

import json
import sys
from pathlib import Path

REPORT = Path(__file__).parents[1]


def quantiles(values: list[float], qs=(0.05, 0.5, 0.95)) -> list[float]:
    s = sorted(values)
    return [s[min(len(s) - 1, int(q * len(s)))] for q in qs]


def summary(band: dict) -> dict:
    """The numbers the tiles and the README quote."""
    patch, fisher = band["patch"], band["fisher"]
    sigma = [s["sigma"] for s in fisher]
    loud = [s["sigma"] for s in fisher if s["snr"] >= 100]
    path = band["path"]
    i_1e3 = min(
        range(len(path["one_minus_s"])),
        key=lambda i: abs(path["one_minus_s"][i] - 1e-3),
    )
    return dict(
        pairs=len(band["rope_periods"]),
        below_two_tokens=sum(p < 2 * patch for p in band["rope_periods"]),
        below_token=sum(p < patch for p in band["rope_periods"]),
        lattice_blind=sum(a < 0.05 and r <= 2 * patch for a, r in zip(band["lattice_alias"], band["rope_periods"])),
        sigma=quantiles(sigma),
        sigma_loud=quantiles(loud) if loud else None,
        sigma_min=min(sigma),
        below_finest=sum(s < band["freq_finest"] for s in sigma),
        below_bf16=sum(s < band["bf16"]["noise_bf16"] for s in sigma),
        sources=len(sigma),
        path_1e3=dict(q50=path["q50"][i_1e3], q99=path["q99"][i_1e3]),
    )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frequency RoPE</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e;
  --s1: #2a78d6; --s2: #eb6834; --s3: #199e70; --s4: #a24bd1; --bad: #c2362b; --band: rgba(11,11,11,0.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7;
    --s1: #3987e5; --s2: #d95926; --s3: #1baf7a; --s4: #b366e0; --bad: #e5675c; --band: rgba(255,255,255,0.07);
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
table.t th, table.t td { padding: 3px 12px 3px 0; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
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
<h1>RoPE on the shared frequency axis</h1>
<p class="sub">Data tokens and source tokens share one rotary axis: absolute frequency in Hz, shown here in bins from the window start. Every number below comes from the working-tree <code>LisaFlow</code>: its periods, its <code>source_bins</code>, the (cos, sin) it hands the backbone, and <code>mmdit.rotate</code>. The tokens themselves are in <a href="../data-tokens/data-tokens.html">data tokens</a>.</p>
<div class="bar"><div><span class="k">Band</span><span class="seg" id="band"></span></div></div>
<div class="tiles" id="tiles"></div>

<div class="card">
  <h2>Which position does each stream's token get?</h2>
  <p class="note" id="pos-note"></p>
  <div class="legend" id="lg-pos"></div>
  <div id="p-pos"></div><div class="tip"></div>
  <div class="scroll" id="t-pos"></div>
</div>

<div class="card">
  <h2>Where do the periods sit against the scales f0 lives on?</h2>
  <p class="note">Each tick is one period, in frequency bins (log axis). RoPE: the 32 head-dim pairs, identical for every size XS–XL. f0, y: the Fourier features of f0 and of a data token's lowest and highest frequency, one shared set of periods (in bins). Vertical lines mark the scales they have to resolve.</p>
  <div class="legend" id="lg-ladder"></div>
  <div id="p-ladder"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>What does the position kernel resolve, and where does it alias?</h2>
  <p class="note">With q = k (all ones, every pair aligned) the logit is the mean over pairs of cos(2π Δ/P). This is the part of a score that depends on position only; learned q, k reweight the pairs. Dotted lines are multiples of the token spacing.</p>
  <div class="controls"><div><span class="k">Range</span><span class="seg" id="krange"></span></div></div>
  <div id="p-kernel"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>Does a source's own token win over every other data token?</h2>
  <p class="note">A source u bins from the centre of its token, scored with the aligned kernel against its own token and against the best other token in the window.</p>
  <div class="legend" id="lg-margin"></div>
  <div id="p-margin"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>Which pairs can tell data tokens apart?</h2>
  <p class="note">Data tokens sit on a lattice spaced by the patch. A pair whose period divides the patch (patch/P an integer) gives every data token the same angle, so it adds one constant to all of a source's data logits and softmax ignores it. Bar height is the distance of patch/P from the nearest integer; 0 at a period of at most two tokens means blind. For longer periods the bar is just the phase advance per token (patch/P turns): slow, not blind.</p>
  <div id="p-lattice"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>How finely does bf16 resolve a position?</h2>
  <p class="note" id="bf16-note"></p>
  <div class="legend" id="lg-stair"></div>
  <div id="p-stair"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>How narrow are the posteriors the positions must resolve?</h2>
  <p class="note">Fisher f0 standard deviation per source (four per window), in bins: marginal over the other seven parameters, with weak priors on log ḟ (the prior's conditional width) and the angles. Lines: the finest f0 Fourier period, the bf16 position noise, one bin and the token spacing.</p>
  <div class="controls"><div><span class="k">Width</span><span class="seg" id="sigkind"></span></div></div>
  <div class="legend" id="lg-sigma"></div>
  <div id="p-sigma"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>How far is the source position from the target along the path?</h2>
  <p class="note">The source token's RoPE position comes from x_t, not the prediction. Along x_t = x1 + (1 − s)(x0 − x1), x0 and x1 two prior draws, this is |bins(x_t) − bins(x1)|.</p>
  <div class="legend" id="lg-path"></div>
  <div id="p-path"></div><div class="tip"></div>
</div>

<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const S = {band: "full", krange: "fine", sigkind: "sigma"};
const marks = {};
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const B = () => D.bands[S.band];
const fmt = (v, n = 3) => v === null || v === undefined ? "–" : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en") : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(1) : Number(v.toPrecision(n)).toString();

function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; schedule(); }));
}
seg("band", [["narrow", "narrow"], ["full", "full"]], "band");
seg("krange", [["fine", "0–64 bins"], ["log", "whole window, log Δ"]], "krange");
seg("sigkind", [["sigma", "marginal"], ["conditional", "f0 alone (1/√F)"]], "sigkind");

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

const ticks = (lo, hi) => { const out = []; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); } return out; };
const decades = (lo, hi) => { const out = []; for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) out.push(10 ** e); return out; };
const sup = v => { const e = Math.round(Math.log10(v)); return Math.abs(e) <= 2 ? fmt(v) : `1e${e}`; };
function frame(host, o) {
  const W = Math.max(320, host.clientWidth), H = o.H || 300, M = {l: 60, r: 14, t: 10, b: 40};
  const lx = v => o.logx ? Math.log10(v) : v, ly = v => o.logy ? Math.log10(v) : v;
  const X = v => M.l + (lx(v) - lx(o.x[0])) / (lx(o.x[1]) - lx(o.x[0])) * (W - M.l - M.r);
  const Y = v => M.t + (ly(o.y[1]) - ly(v)) / (ly(o.y[1]) - ly(o.y[0])) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(o.label)}">`;
  o.yt.forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${o.yfmt ? o.yfmt(v) : fmt(v)}</text>`);
  o.xt.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${o.xfmt ? o.xfmt(v) : fmt(v)}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${o.xlabel}</text><text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  return {W, H, X, Y, M, s};
}
const line = (f, xs, ys, color, extra = "") => `<path d="${xs.map((x, i) => `${i ? "L" : "M"}${f.X(x).toFixed(1)},${f.Y(ys[i]).toFixed(1)}`).join("")}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" ${extra}/>`;
const swatch = (color, dash = "") => `<svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="${color}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ""}/></svg>`;
const dot = color => `<svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="${color}"/></svg>`;

function tiles() {
  const b = B(), s = b.summary, k = b.kernel, tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  document.getElementById("tiles").innerHTML = [
    tile("RoPE pairs", `${s.pairs}`, `periods ${fmt(b.rope_periods[0])}–${fmt(b.rope_periods.at(-1))} bins, same for XS–XL (head dim ${b.sizes.B.head_dim})`),
    tile("Pairs finer than two tokens", `${s.below_two_tokens}`, `period < ${2 * b.patch} bins; ${s.below_token} below one token, ${s.lattice_blind} blind to the token lattice`),
    tile("Kernel half-width", `${fmt(k.hwhm)} bins`, `first zero ${fmt(k.first_zero)} bins, largest sidelobe ${fmt(k.sidelobe)} at ${fmt(k.sidelobe_at)}`),
    tile("bf16 position noise", `${fmt(b.bf16.noise_bf16)} bins`, `float32 angles alone: ${fmt(b.bf16.noise_f32)} bins`),
    tile("Fisher σ(f0), median", `${fmt(s.sigma[1])} bins`, `5–95%: ${fmt(s.sigma[0])}–${fmt(s.sigma[2])}; ${s.below_bf16} of ${s.sources} below the bf16 noise`),
  ].join("");
}

function positions() {
  const b = B(), p = b.positions, sb = b.source_bins, host = document.getElementById("p-pos");
  document.getElementById("pos-note").innerHTML = `The (cos, sin) the XS flow hands <code>MMDiT.encode</code> and <code>decode</code> (backbone patched out), checked against absolute frequency (window start plus bins, over T) turned over each RoPE period. Data token n sits at its centre, ${b.patch}n + ${(b.patch - 1) / 2} bins; source s at <code>source_bins(x_t[s, 0], f)</code>. Over ${sb.windows} windows, <code>source_bins</code> matches f0·T − start from <code>flow_to_physical</code> to ${fmt(sb.max_err)} bins; a float32 x_f would quantise it in steps up to ${fmt(sb.f32_step)} bins (slope up to ${fmt(sb.max_slope)} bins per unit x_f), which is why it is computed before any cast. Largest captured-cosine error against that: data ${fmt(p.data_angle_err)}, sources ${fmt(p.source_angle_err)}.`;
  const W = Math.max(320, host.clientWidth), H = 110, l = 16, r = 16, X = v => l + v / b.window_bins * (W - l - r);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="token positions">`;
  svg += `<rect x="${X(0)}" y="20" width="${X(b.guard) - X(0)}" height="50" fill="var(--band)"/><rect x="${X(b.window_bins - b.guard)}" y="20" width="${X(b.window_bins) - X(b.window_bins - b.guard)}" height="50" fill="var(--band)"/>`;
  svg += `<line x1="${X(0)}" x2="${X(b.window_bins)}" y1="45" y2="45" stroke="var(--axis)"/>`;
  p.data_bins.forEach((v, i) => svg += `<line x1="${X(v)}" x2="${X(v)}" y1="26" y2="44" stroke="var(--s1)" stroke-width="1.5" data-tip="${esc(`<b>data token ${i}</b><br>position ${v} bins`)}"/>`);
  p.source_bins.forEach((v, i) => svg += `<circle cx="${X(v)}" cy="56" r="5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2" data-tip="${esc(`<b>source ${i}</b><br>RoPE position ${v} bins<br>source_bins ${fmt(p.source_bins_code[i], 7)}<br>f0 T − start ${fmt(p.source_bins_exact[i], 7)}`)}"/>`);
  svg += `<text x="${X(0)}" y="88">0</text><text x="${X(b.window_bins)}" y="88" text-anchor="end">${b.window_bins} bins</text><text x="${X(b.guard / 2)}" y="14" text-anchor="${b.guard / b.window_bins < 0.05 ? "start" : "middle"}">guard</text><text x="${X(b.window_bins - b.guard / 2)}" y="14" text-anchor="${b.guard / b.window_bins < 0.05 ? "end" : "middle"}">guard</text>`;
  host.innerHTML = svg + "</svg>";
  document.getElementById("lg-pos").innerHTML = `<span><svg width="10" height="14"><line x1="5" x2="5" y1="1" y2="13" stroke="var(--s1)" stroke-width="2"/></svg>data tokens (${p.data_bins.length})</span><span>${dot("var(--s2)")}sources of one prior draw x</span>`;
  document.getElementById("t-pos").innerHTML = `<table class="t"><thead><tr><th>source</th><th>RoPE position</th><th>source_bins</th><th>f0 T − start</th><th>token</th></tr></thead><tbody>${p.source_bins.map((v, i) => `<tr><td>${i}</td><td>${v.toFixed(3)}</td><td>${p.source_bins_code[i].toFixed(6)}</td><td>${p.source_bins_exact[i].toFixed(6)}</td><td>${Math.floor(p.source_bins_code[i] / b.patch)}</td></tr>`).join("")}</tbody></table>`;
}

function ladder() {
  const b = B(), s = b.summary, host = document.getElementById("p-ladder");
  const rows = [
    ["RoPE", b.rope_periods, "var(--s1)"],
    ["f0, y features", b.freq_periods, "var(--s2)"],
  ];
  const all = rows.flatMap(r => r[1]).concat([s.sigma[0]]);
  const lo = Math.min(...all) / 2, hi = Math.max(...all) * 2;
  const W = Math.max(320, host.clientWidth), rowH = 30, top = 64, H = top + rows.length * rowH + 40, l = 110, r = 14;
  const X = v => l + (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (W - l - r);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="period ladder">`;
  decades(lo, hi).forEach(v => svg += `<line x1="${X(v)}" x2="${X(v)}" y1="${top - 8}" y2="${H - 34}" stroke="var(--grid)"/><text x="${X(v)}" y="${H - 20}" text-anchor="middle">${sup(v)}</text>`);
  svg += `<rect x="${X(s.sigma[0])}" y="${top - 8}" width="${X(s.sigma[2]) - X(s.sigma[0])}" height="${H - 34 - top + 8}" fill="var(--band)" data-tip="${esc(`<b>Fisher σ(f0), 5–95%</b><br>${fmt(s.sigma[0])}–${fmt(s.sigma[2])} bins`)}"/>`;
  const refs = [
    [b.patch, "token", "4 3"], [b.half_width.low, "Doppler ½w low", "2 3"], [b.half_width.high, "Doppler ½w high", "2 3"], [b.window_bins, "window", "4 3"],
  ];
  refs.forEach(([v, name, dash], i) => svg += `<line x1="${X(v)}" x2="${X(v)}" y1="${top - 8}" y2="${H - 34}" stroke="var(--ref)" stroke-dasharray="${dash}"/><text class="lbl" x="${X(v)}" y="${12 + (i % 2) * 14}" text-anchor="middle">${name} ${fmt(v)}</text>`);
  svg += `<text class="lbl" x="${X(Math.sqrt(s.sigma[0] * s.sigma[2]))}" y="${top - 14}" text-anchor="middle">σ(f0) 5–95%</text>`;
  rows.forEach(([name, ps, color], i) => {
    const y = top + i * rowH + rowH / 2;
    svg += `<text class="lbl" x="${l - 8}" y="${y + 4}" text-anchor="end">${name}</text>`;
    ps.forEach((p, k) => svg += `<line x1="${X(p)}" x2="${X(p)}" y1="${y - 9}" y2="${y + 9}" stroke="${color}" stroke-width="2"/><rect x="${X(p) - 3}" y="${y - 11}" width="6" height="22" fill="transparent" data-tip="${esc(`<b>${name} period ${k}</b><br>${fmt(p, 4)} bins`)}"/>`);
  });
  svg += `<text x="${(l + W - r) / 2}" y="${H - 4}" text-anchor="middle">period [bins]</text>`;
  host.innerHTML = svg + "</svg>";
  document.getElementById("lg-ladder").innerHTML = rows.slice(0, 4).map(([n, , c]) => `<span>${swatch(c)}${n.replace(", band low", "")}</span>`).join("") + `<span>${swatch("var(--ref)", "4 3")}token, window</span><span>${swatch("var(--ref)", "2 3")}Doppler half-width at the band edges</span>`;
}

function kernel() {
  const b = B(), k = b.kernel, host = document.getElementById("p-kernel"), fine = S.krange === "fine";
  const src = fine ? k.fine : {delta: k.coarse.delta.filter(d => d > 0), k: k.coarse.k.filter((_, i) => k.coarse.delta[i] > 0)};
  const xr = fine ? [0, 64] : [src.delta[0], b.window_bins];
  const ymin = Math.min(-0.2, ...src.k) - 0.05;
  const f = frame(host, {x: xr, y: [ymin, 1.05], logx: !fine, xt: fine ? [0, 8, 16, 24, 32, 40, 48, 56, 64] : decades(...xr), yt: [-0.2, 0, 0.25, 0.5, 0.75, 1], xfmt: fine ? undefined : sup, xlabel: "frequency offset Δ [bins]", ylabel: "aligned logit / head dim", label: "position kernel"});
  let s = f.s;
  for (let m = b.patch; m <= xr[1]; m += fine ? b.patch : m) s += `<line x1="${f.X(m)}" x2="${f.X(m)}" y1="${f.Y(1.05)}" y2="${f.Y(ymin)}" stroke="var(--axis)" stroke-dasharray="1 3"/>`;
  s += `<line x1="${f.X(xr[0])}" x2="${f.X(xr[1])}" y1="${f.Y(0)}" y2="${f.Y(0)}" stroke="var(--axis)"/>`;
  s += line(f, src.delta, src.k, "var(--s1)");
  const pts = [[k.hwhm, 0.5, `half max at ${fmt(k.hwhm)} bins`], [k.first_zero, 0, `first zero ${fmt(k.first_zero)}`], [k.sidelobe_at, k.sidelobe, `largest sidelobe ${fmt(k.sidelobe)} at ${fmt(k.sidelobe_at)}`], [k.min_at, k.min_k, `minimum ${fmt(k.min_k)} at ${fmt(k.min_at)}`]];
  pts.filter(([x]) => x !== null && x >= xr[0] && x <= xr[1]).forEach(([x, y, t], i) => s += `<circle cx="${f.X(x)}" cy="${f.Y(y)}" r="4.5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2" data-tip="${esc(`<b>${t}</b>`)}"/><text class="lbl" x="${f.X(x) + 8}" y="${f.Y(y) - 6 - 12 * (i % 2)}">${t}</text>`);
  // hover targets along the curve
  const step = Math.max(1, Math.floor(src.delta.length / 200));
  for (let i = 0; i < src.delta.length; i += step) s += `<circle cx="${f.X(src.delta[i])}" cy="${f.Y(src.k[i])}" r="6" fill="transparent" data-tip="${esc(`Δ ${fmt(src.delta[i], 4)} bins: ${src.k[i].toFixed(4)}`)}"/>`;
  host.innerHTML = s + "</svg>";
}

function margin() {
  const b = B(), m = b.kernel.margin, host = document.getElementById("p-margin");
  const ys = m.own.concat(m.other);
  const f = frame(host, {x: [m.u[0], m.u.at(-1)], y: [Math.min(...ys) - 0.05, Math.max(...ys) + 0.05], xt: [-8, -4, 0, 4, 8].filter(v => v >= m.u[0] && v <= m.u.at(-1)), yt: ticks(0.05, 1).filter(v => v >= Math.min(...ys) - 0.05 && v <= Math.max(...ys) + 0.05), xlabel: "source offset from its token's centre u [bins]", ylabel: "aligned logit / head dim", label: "own token margin", H: 260});
  let s = f.s + line(f, m.u, m.own, "var(--s1)") + line(f, m.u, m.other, "var(--s2)");
  for (let i = 0; i < m.u.length; i += 4) s += `<rect x="${f.X(m.u[i]) - 4}" y="${f.M.t}" width="8" height="${f.H - f.M.t - f.M.b}" fill="transparent" data-tip="${esc(`<b>u = ${m.u[i].toFixed(2)} bins</b><br>own ${m.own[i].toFixed(3)}, best other ${m.other[i].toFixed(3)}<br>margin ${(m.own[i] - m.other[i]).toFixed(3)}`)}"/>`;
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-margin").innerHTML = `<span>${swatch("var(--s1)")}own token</span><span>${swatch("var(--s2)")}best other token</span><span>smallest margin ${m.min.toFixed(3)}</span>`;
}

function lattice() {
  const b = B(), host = document.getElementById("p-lattice"), a = b.lattice_alias, n = a.length;
  const W = Math.max(320, host.clientWidth), H = 220, M = {l: 60, r: 14, t: 10, b: 52};
  const bw = (W - M.l - M.r) / n, Y = v => M.t + (0.5 - v) / 0.5 * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="lattice alias per pair">`;
  [0, 0.25, 0.5].forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  a.forEach((v, i) => {
    const p = b.rope_periods[i], x = M.l + i * bw + 1, blind = v < 0.05 && p <= 2 * b.patch;
    s += `<rect x="${x}" y="${Y(Math.max(v, 0.004))}" width="${Math.max(bw - 2, 1)}" height="${Y(0) - Y(Math.max(v, 0.004))}" rx="2" fill="${blind ? "var(--bad)" : p < 2 * b.patch ? "var(--s2)" : "var(--s1)"}" data-tip="${esc(`<b>pair ${i}</b>, period ${fmt(p, 4)} bins<br>patch/P = ${(b.patch / p).toFixed(3)}, alias ${v.toFixed(3)}${blind ? "<br>blind to the token lattice" : ""}`)}"/>`;
    if (i % 4 === 0 || i === n - 1) s += `<text x="${x + bw / 2}" y="${H - M.b + 14}" text-anchor="middle">${fmt(p, 2)}</text>`;
  });
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 18}" text-anchor="middle">pair (period in bins)</text><text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">|patch/P − nearest int|</text>`;
  host.innerHTML = s + `</svg><div class="legend" style="margin-top:6px"><span>${dot("var(--bad)")}blind: period ≤ two tokens and alias < 0.05</span><span>${dot("var(--s2)")}period < two tokens</span><span>${dot("var(--s1)")}period ≥ two tokens</span></div>`;
}

function stair() {
  const b = B(), q = b.bf16, st = q.stair, host = document.getElementById("p-stair");
  document.getElementById("bf16-note").innerHTML = `Random rms-normalised q, k (head dim ${2 * b.rope_periods.length}) at random positions, rotated in float32 as <code>mmdit.rotate</code> does, then cast to bf16 (the default backbone) and dotted with float32 accumulation, against an exact float64 rotation. Logit error rms ${fmt(q.err_bf16)} over a logit-slope rms of ${fmt(q.slope_rms)} per bin makes a position noise of <b>${fmt(q.noise_bf16)} bins</b> under bf16 and ${fmt(q.noise_f32)} bins with float32 q, k. The plot shifts one query by ±0.01 bins.`;
  const all = st.f32.concat(st.bf16, st.exact), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08 || 1e-3;
  const f = frame(host, {x: [st.delta[0], st.delta.at(-1)], y: [lo - pad, hi + pad], xt: [-0.01, -0.005, 0, 0.005, 0.01], yt: [lo, (lo + hi) / 2, hi], yfmt: v => v.toFixed(4), xlabel: "query shift [bins]", ylabel: "logit", label: "bf16 staircase", H: 260});
  let s = f.s + line(f, st.delta, st.exact, "var(--ref)", 'stroke-dasharray="4 3"') + line(f, st.delta, st.f32, "var(--s1)") + line(f, st.delta, st.bf16, "var(--s2)");
  for (let i = 0; i < st.delta.length; i += 8) s += `<rect x="${f.X(st.delta[i]) - 4}" y="${f.M.t}" width="8" height="${f.H - f.M.t - f.M.b}" fill="transparent" data-tip="${esc(`<b>shift ${st.delta[i].toFixed(4)} bins</b><br>exact ${st.exact[i].toFixed(5)}<br>float32 ${st.f32[i].toFixed(5)}<br>bf16 ${st.bf16[i].toFixed(5)}`)}"/>`;
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-stair").innerHTML = `<span>${swatch("var(--ref)", "4 3")}exact (float64)</span><span>${swatch("var(--s1)")}float32 q, k</span><span>${swatch("var(--s2)")}bf16 q, k</span>`;
}

function sigma() {
  const b = B(), host = document.getElementById("p-sigma"), F = b.fisher, key = S.sigkind;
  const vals = F.map(s => s[key]), snr = F.map(s => s.snr);
  const refs = [[b.freq_finest, "finest f0 period", "var(--s3)"], [b.bf16.noise_bf16, "bf16 noise", "var(--s2)"], [1, "1 bin", "var(--ref)"], [b.patch, "token", "var(--ref)"]];
  const ylo = Math.min(...vals, ...refs.map(r => r[0])) / 2, yhi = Math.max(...vals, b.patch) * 2;
  const xlo = Math.min(...snr) / 1.2, xhi = Math.max(...snr) * 1.2;
  const f = frame(host, {x: [xlo, xhi], y: [ylo, yhi], logx: true, logy: true, xt: ticks(xlo, xhi), yt: decades(ylo, yhi), yfmt: sup, xlabel: "source SNR", ylabel: "σ(f0) [bins]", label: "Fisher f0 width"});
  let s = f.s;
  refs.forEach(([v, name, c], i) => s += `<line x1="${f.M.l}" x2="${f.W - f.M.r}" y1="${f.Y(v)}" y2="${f.Y(v)}" stroke="${c}" stroke-dasharray="4 3"/><text class="lbl" x="${f.W - f.M.r - 4}" y="${f.Y(v) - 4}" text-anchor="end">${name} ${fmt(v)}</text>`);
  F.forEach(p => s += `<circle cx="${f.X(p.snr)}" cy="${f.Y(p[key])}" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1" data-tip="${esc(`<b>SNR ${p.snr.toFixed(1)}</b>, f0 ${(p.f0 * 1e3).toFixed(4)} mHz<br>marginal σ ${fmt(p.sigma)} bins<br>f0 alone ${fmt(p.conditional)} bins<br>σ(log ḟ) ${fmt(p.sigma_log_fdot)}`)}"/>`);
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-sigma").innerHTML = `<span>${dot("var(--s1)")}source (${F.length})</span><span>${swatch("var(--s3)", "4 3")}finest f0 Fourier period</span><span>${swatch("var(--s2)", "4 3")}bf16 position noise</span><span>${swatch("var(--ref)", "4 3")}1 bin, one token</span>`;
}

function path() {
  const b = B(), p = b.path, host = document.getElementById("p-path"), s0 = b.summary;
  const all = p.q50.concat(p.q99).filter(v => v > 0);
  const ylo = Math.max(1e-9, Math.min(...all, s0.sigma[0]) / 2), yhi = Math.max(...all) * 2;
  const f = frame(host, {x: [p.one_minus_s[0], 1], y: [ylo, yhi], logx: true, logy: true, xt: decades(p.one_minus_s[0], 1), yt: decades(ylo, yhi), xfmt: sup, yfmt: sup, xlabel: "1 − s", ylabel: "|bins(x_t) − bins(x1)|", label: "path position error"});
  let s = f.s;
  [[b.patch, "token"], [s0.sigma[1], "median σ(f0)"]].forEach(([v, n]) => s += `<line x1="${f.M.l}" x2="${f.W - f.M.r}" y1="${f.Y(v)}" y2="${f.Y(v)}" stroke="var(--ref)" stroke-dasharray="4 3"/><text class="lbl" x="${f.M.l + 4}" y="${f.Y(v) - 4}">${n} ${fmt(v)}</text>`);
  const cl = v => Math.max(v, ylo);
  [["q50", "var(--s1)", "median"], ["q90", "var(--s3)", "90%"], ["q99", "var(--s2)", "99%"]].forEach(([k, c]) => s += line(f, p.one_minus_s, p[k].map(cl), c));
  p.one_minus_s.forEach((x, i) => s += `<rect x="${f.X(x) - 5}" y="${f.M.t}" width="10" height="${f.H - f.M.t - f.M.b}" fill="transparent" data-tip="${esc(`<b>1 − s = ${x.toExponential(1)}</b><br>median ${fmt(p.q50[i])}, 90% ${fmt(p.q90[i])}, 99% ${fmt(p.q99[i])} bins`)}"/>`);
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-path").innerHTML = `<span>${swatch("var(--s1)")}median</span><span>${swatch("var(--s3)")}90%</span><span>${swatch("var(--s2)")}99%</span>`;
}

function writeHash() { history.replaceState(null, "", "#" + new URLSearchParams({band: S.band, kernel: S.krange, width: S.sigkind}).toString()); }
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (D.bands[h.get("band")]) S.band = h.get("band");
  if (["fine", "log"].includes(h.get("kernel"))) S.krange = h.get("kernel");
  if (["sigma", "conditional"].includes(h.get("width"))) S.sigkind = h.get("width");
}
function render() {
  for (const k of Object.keys(marks)) marks[k]();
  tiles(); positions(); ladder(); kernel(); margin(); lattice(); stair(); sigma(); path();
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

MEANS_HTML = """
<ul>
<li><b>Attention places a source to about one token.</b> The aligned kernel has a half width of 2.4 bins on narrow and 4.4 on full, and its first zero is at 17 and 43 bins. Sources in visible windows are loud (median SNR 1.6 narrow, 7.2 full), so their Fisher widths in f0 are narrow: a median 0.36 bins (narrow) and 0.090 (full), 5–95% 0.11–0.82 and 0.036–0.40. All 256 sources per band are narrower than the kernel's half width, and 250 narrow and 256 full are under one bin.</li>
<li><b>Fine f0 is needed by most sources.</b> The finest f0 period is 1/64 bin and no Fisher width is below it (minimum 0.056 narrow, 0.020 full). bf16 position noise is 2.2e-3 (narrow) and 2.4e-3 (full) bins, also under every width.</li>
<li><b>A source's own token wins.</b> Its margin over the best other token is 0.82 on narrow and 0.70 on full at the token centre, and it falls to a tie only at the token edge. The kernel goes negative past its first zero (down to -0.30 and -0.38).</li>
<li><b>Many rope pairs are shorter than a token</b> (14 of 32 narrow, 11 full), and 18 and 14 are under two tokens, so they alias across data tokens; 4 narrow and 2 full pairs have a period that divides the patch.</li>
<li><b>Source positions come from x_t; on full the path error exceeds the median Fisher width.</b> At 1-s = 1e-3 the median position error is 0.11 bins (99%: 0.35) on full and 0.028 (0.086) on narrow; the median Fisher width is 0.36 (narrow) and 0.090 (full) bins.</li>
</ul>
"""


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "rope.json"
    data = json.loads((REPORT / "data" / name).read_text())
    for band in data["bands"].values():
        band["summary"] = summary(band)
    for name, band in data["bands"].items():
        print(name, json.dumps(band["summary"]))
        print(
            name,
            "kernel",
            {
                k: band["kernel"][k]
                for k in (
                    "hwhm",
                    "first_zero",
                    "min_k",
                    "min_at",
                    "sidelobe",
                    "sidelobe_at",
                    "at",
                )
            },
            "margin min",
            band["kernel"]["margin"]["min"],
        )
        print(
            name,
            "bf16",
            {k: v for k, v in band["bf16"].items() if k != "stair"},
            "source_bins",
            band["source_bins"],
        )
        print(
            name,
            "half_width",
            band["half_width"],
        )
    page = PAGE.replace("__MEANS__", MEANS_HTML).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = REPORT / "frequency-rope.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
