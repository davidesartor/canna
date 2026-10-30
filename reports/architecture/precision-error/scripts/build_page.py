"""Write precision-error.html from data/<run>-<platform>.json: each bf16 mode's error against float32, per block and in posterior widths.

Stdlib only; run from the repo root:
    uv run --frozen python reports/architecture/precision-error/scripts/build_page.py
"""

import json
import statistics
from pathlib import Path

REPORT = Path(__file__).parents[1]
SMOKE_STEPS = 100_000
COORDS = ["f0", "Mc", "l", "b", "Re z+", "Im z+", "Re z−", "Im z−"]


def flat(a) -> list:
    """Every finite number in a nested list."""
    if isinstance(a, list):
        return [v for row in a for v in flat(row)]
    return [] if a is None else [a]


def report(runs: dict) -> None:
    """Print the numbers the README quotes."""
    for name, r in runs.items():
        print(
            f"{name}: epoch {r['epoch']}, {r['platform']} ({r['device']}), recording check {r['patch_check']:.2g}"
        )
        for mode in r["modes"]:
            if mode == r["reference"]:
                continue
            out = [row[-1][0] for row in r["decode"][mode]]
            enc_k = r["encode"][mode]["k"][-1][0]
            print(
                f"  {mode}: output token rel err (median) per t {[f'{v:.2g}' for v in out]}, last-block k {enc_k:.2g}"
            )
            for i, t in enumerate(r["t_grid"]):
                vals = flat([obs[i] for obs in r["x1_over_width"][mode]])
                print(
                    f"    t={t}: |Δx1|/width median {statistics.median(vals):.3g}, max {max(vals):.3g};"
                    f" rel velocity err median {statistics.median(o[i] for o in r['velocity'][mode]):.3g}"
                )
            disp = flat(r["e2e_displacement"][mode])
            ratio = flat(r["e2e_width_ratio"][mode])
            print(
                f"    sampler: displacement/width median {statistics.median(disp):.3g}, max {max(disp):.3g};"
                f" width ratio {min(ratio):.3g}–{max(ratio):.3g}"
            )
        tok = flat(r["token_ratio"])
        print(
            f"  1σ token move / bf16 rounding: min {min(tok):.3g}, median {statistics.median(tok):.3g};"
            f" rounding / token norm median {statistics.median(flat(r['token_rounding'])):.2g}"
        )
        print("  snr", [[round(v, 1) for v in o] for o in r["snr"]])


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bf16 precision error</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e; --warn-bg: #fbf1dc; --warn: #7a5300;
  --g0: #2a78d6; --g1: #eb6834; --g2: #199e70; --g3: #8a5cd6; --g4: #b8860b; --bad: #c2362b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7; --warn-bg: #3a2f14; --warn: #f2c66d;
    --g0: #3987e5; --g1: #d95926; --g2: #1baf7a; --g3: #a07ce8; --g4: #d4a52a; --bad: #e5675c;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
a { color: var(--g0); }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 80ch; }
.warn { background: var(--warn-bg); color: var(--warn); border-radius: 8px; padding: 8px 12px; font-size: 13px; margin: 0 0 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.card + .card { margin-top: 16px; }
.bar, .controls { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 12px; }
.k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.tbl { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.tbl th, .tbl td { padding: 4px 6px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
.tbl th:first-child, .tbl td:first-child { text-align: left; }
.tbl th { color: var(--ink-2); font-weight: 600; }
.scroll { overflow-x: auto; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 80ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>How much does bf16 move the flow?</h1>
<p class="sub">The same trained parameters and inputs run through the network in each precision mode: <code>none</code> (float32, the reference), <code>attention</code> (bf16 attention cache and softmax-weighted values, float32 backbone) and <code>backbone</code> (bf16 backbone, the training default). A float64 backbone shows the reference's own floor. Errors are reported per block as relative token error and, at the output, in units of each source's posterior width. This page covers accuracy only; for speed and memory see <a href="../../benchmarks/step-time-memory/">step-time-memory</a>.</p>
<div class="bar"><div><span class="k">Run</span><span class="seg" id="run"></span></div></div>
<div id="smoke"></div>
<div id="platform"></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>Where in the network does the error enter?</h2>
  <p class="note">‖h<sub>mode</sub> − h<sub>f32</sub>‖ / ‖h<sub>f32</sub>‖ per token, median (solid) and 90th percentile (dashed) over tokens, draws and injections. The source stream is shown after each decode block, starting from its input (block 0) and ending at the backbone's output. The bf16 unit roundoff, 2<sup>−8</sup> ≈ 0.004, is the dotted line.</p>
  <div class="controls"><div><span class="k">t</span><span class="seg" id="t"></span></div></div>
  <div class="legend" id="lg-blocks"></div>
  <div id="p-blocks"></div>
  <div class="scroll" id="t-encode"></div>
</div>
<div class="card">
  <h2>How large is the x̂<sub>1</sub> error, in posterior widths?</h2>
  <p class="note">|x̂<sub>1,mode</sub> − x̂<sub>1,f32</sub>| per source and coordinate, median over 16 states on the straight path from a base draw to the truth at this t, divided by the source's Fisher posterior width in flow coordinates. Each point is one source of one eval injection, placed at its own SNR. The velocity error is this divided by 1 − t, and so is its effect on one step. The dashed line is a tenth of a width.</p>
  <div class="controls"><div><span class="k">t</span><span class="seg" id="t2"></span></div><div><span class="k">Coordinate</span><span class="seg" id="coord"></span></div></div>
  <div class="legend" id="lg-x1"></div>
  <div id="p-x1"></div>
  <div class="scroll" id="t-velocity"></div>
</div>
<div class="card">
  <h2>Does it survive the sampler?</h2>
  <p class="note">The default sampler (midpoint x-steps) from the same base draws in every mode. Left axis: the median over draws of |x<sub>mode</sub> − x<sub>f32</sub>| in posterior widths, per source and coordinate. The table gives the ratio of each mode's sample width (1.4826 × MAD) to the float32 sample width.</p>
  <div class="controls"><div><span class="k">Coordinate</span><span class="seg" id="coord2"></span></div></div>
  <div class="legend" id="lg-e2e"></div>
  <div id="p-e2e"></div>
  <div class="scroll" id="t-e2e"></div>
</div>
<div class="card">
  <h2>Can a bf16 token tell a 1σ move apart?</h2>
  <p class="note">At the truth (t = 0.99, x̂ = x), each coordinate in turn is moved by one posterior width. The plot shows the change this makes to that source's float32 decode-input token, divided by the error from rounding that token to bf16. Below 1 (dotted), a backbone-mode network cannot see the move in its input.</p>
  <div class="legend" id="lg-token"></div>
  <div id="p-token"></div>
</div>
<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const RUNS = Object.keys(D.runs);
const COORDS = D.coords;
const MCOL = {none: "var(--ink-2)", attention: "var(--g0)", backbone: "var(--g1)", f64: "var(--g2)"};
const CCOL = ["var(--g0)", "var(--g1)", "var(--g2)", "var(--g3)", "var(--g4)", "var(--bad)", "var(--ink-2)", "var(--muted)"];
const S = {run: RUNS[RUNS.length - 1], t: "3", t2: "3", coord: "all", coord2: "all"};
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const fmt = (v, n = 3) => v == null || !Number.isFinite(v) ? "–" : Number(v).toPrecision(n);
const lg = v => v > 0 ? Math.log10(v) : NaN;
const pow = v => Number.isInteger(v) ? (v === 0 ? "1" : `10${v < 0 ? "⁻" : ""}${String(Math.abs(v)).split("").map(c => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]).join("")}`) : "";
const median = a => { const v = a.filter(Number.isFinite).sort((x, y) => x - y); return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : NaN; };
const flat = a => Array.isArray(a) ? a.flatMap(flat) : [a == null ? NaN : a];

function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, label]) => `<button data-v="${esc(v)}">${esc(label)}</button>`).join("");
  host.onclick = e => { const b = e.target.closest("button"); if (!b) return; S[key] = b.dataset.v; render(); };
  return () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
}
const sw = (color, dash, dot) => dot ? `<svg width="14" height="10" style="width:14px;display:inline-block"><circle cx="7" cy="5" r="4" fill="${color}" stroke="${color}"/></svg>` : `<svg width="24" height="10" style="width:24px;display:inline-block"><line x1="1" x2="23" y1="5" y2="5" stroke="${color}" stroke-width="2" stroke-dasharray="${dash || ""}"/></svg>`;
const legend = (id, items) => document.getElementById(id).innerHTML = items.map(([c, d, label, dot]) => `<span>${sw(c, d, dot)}${esc(label)}</span>`).join("");

// a chart on linear axes; log quantities arrive as log10 and get power-of-ten ticks
function chart(host, o) {
  const M = {l: 58, r: 14, t: 10, b: 40};
  const W = Math.max(300, host.clientWidth), H = o.height || 260;
  const X = v => M.l + (v - o.x[0]) / (o.x[1] - o.x[0]) * (W - M.l - M.r);
  const Y = v => M.t + (o.y[1] - v) / (o.y[1] - o.y[0]) * (H - M.t - M.b);
  const ticks = (lo, hi, n) => { const step = 10 ** Math.floor(Math.log10((hi - lo) / n)); const m = [1, 2, 5, 10].find(m => (hi - lo) / (m * step) <= n) * step; const out = []; for (let v = Math.ceil(lo / m) * m; v <= hi + 1e-9; v += m) out.push(+v.toFixed(10)); return out; };
  const logTicks = (lo, hi) => { const out = []; for (let v = Math.ceil(lo); v <= Math.floor(hi); v++) out.push(v); return out.length >= 2 ? out : ticks(lo, hi, 4); };
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(o.label)}">`;
  (o.ylog ? logTicks(o.y[0], o.y[1]) : ticks(o.y[0], o.y[1], 6)).forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${o.ylog ? pow(v) : (o.yfmt || String)(v)}</text>`);
  (o.xticks || (o.xlog ? logTicks(o.x[0], o.x[1]) : ticks(o.x[0], o.x[1], Math.max(3, Math.floor((W - M.l) / 60))))).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${o.xlog ? pow(v) : (o.xfmt || String)(v)}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">${o.xlabel}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  (o.hlines || []).forEach(h => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(h.y)}" y2="${Y(h.y)}" stroke="var(--ref)" stroke-dasharray="${h.dash || "2 3"}"/><text x="${W - M.r - 4}" y="${Y(h.y) - 4}" text-anchor="end" style="fill:var(--ink-2)">${esc(h.label)}</text>`);
  (o.series || []).forEach(se => {
    const pts = se.pts.filter(p => Number.isFinite(p[1]));
    s += `<path d="${pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("")}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" stroke-dasharray="${se.dash || ""}"/>`;
    pts.forEach(([x, y], i) => s += `<circle cx="${X(x)}" cy="${Y(y)}" r="3" fill="${se.color}"><title>${esc(se.tip(i))}</title></circle>`);
  });
  (o.points || []).filter(p => Number.isFinite(p.y)).forEach(p => s += `<circle cx="${X(p.x)}" cy="${Y(Math.min(Math.max(p.y, o.y[0]), o.y[1]))}" r="3.5" fill="${p.color}" fill-opacity="0.75" stroke="${p.color}"><title>${esc(p.tip)}</title></circle>`);
  host.innerHTML = s + "</svg>";
}
const logSpan = (vals, pad = 0.2) => { const v = vals.filter(Number.isFinite); if (!v.length) return [-1, 1]; return [Math.floor(Math.min(...v) - pad), Math.ceil(Math.max(...v) + pad)]; };

const marks = [seg("run", RUNS.map(r => [r, r]), "run")];
function render() {
  const r = D.runs[S.run];
  if (marks.length === 1) {
    const tItems = r.t_grid.map((t, i) => [i, String(t)]);
    const cItems = [["all", "all"]].concat(COORDS.map((c, i) => [i, c]));
    marks.push(seg("t", tItems, "t"), seg("t2", tItems, "t2"), seg("coord", cItems, "coord"), seg("coord2", cItems, "coord2"));
  }
  marks.forEach(m => m());
  const modes = r.modes.filter(m => m !== r.reference);
  const smoke = r.total_steps < D.smoke_steps;
  document.getElementById("smoke").innerHTML = smoke ? `<div class="warn"><b>Smoke run.</b> ${esc(r.run)} has trained for ${r.total_steps.toLocaleString("en")} steps. The per-block errors show how bf16 behaves in this architecture. The posterior widths here come from the Fisher forecast, not from this flow, but the size of x̂<sub>1</sub>'s error depends on how sharp an untrained flow's predictions are, so read the numbers in widths as illustration only.</div>` : "";
  document.getElementById("platform").innerHTML = r.platform === "cpu" ? `<div class="warn"><b>Measured on CPU.</b> XLA on CPU computes bf16 matmuls with its own accumulation, which is not the GPU's tensor-core accumulation. The modes' casts are the same, but the size of the GPU error can differ; run the probe on a GPU (README) for the training numbers.</div>` : "";
  const tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  const ti = r.t_grid.indexOf(0.99);
  const x1b = flat(r.x1_over_width.backbone.map(o => o[ti]));
  const e2eb = flat(r.e2e_displacement.backbone);
  document.getElementById("tiles").innerHTML = [
    tile("Run", esc(r.config), `${esc(r.run)}, epoch ${r.epoch}, commit ${r.commit.slice(0, 7)}`),
    tile("Device", esc(r.platform), `${esc(r.device)}; float32 matmul precision ${esc(r.matmul_precision)}`),
    tile("backbone |Δx̂₁| at t = 0.99", fmt(median(x1b), 2) + " σ", `median over sources and coordinates; max ${fmt(Math.max(...x1b.filter(Number.isFinite)), 2)} σ`),
    tile("backbone after the sampler", fmt(median(e2eb), 2) + " σ", `median draw displacement; max ${fmt(Math.max(...e2eb.filter(Number.isFinite)), 2)} σ`),
  ].join("");

  // per-block relative error along the decode stream
  const t = Number(S.t);
  const nb = r.decode[r.reference][t].length;
  const series = modes.flatMap(m => [0, 1].map(q => ({pts: r.decode[m][t].map((v, b) => [b, lg(v[q])]), color: MCOL[m], dash: q ? "6 4" : "", width: q ? 1.4 : 2, tip: b => `${m}, ${b === 0 ? "decode input" : b === nb - 1 ? "backbone output" : "after block " + b}: ${q ? "p90" : "median"} ${fmt(r.decode[m][t][b][q])}`})));
  legend("lg-blocks", modes.map(m => [MCOL[m], "", m]).concat([["var(--ink-2)", "6 4", "90th percentile"]]));
  chart(document.getElementById("p-blocks"), {label: "relative error per block", x: [0, nb - 1], y: logSpan(series.flatMap(s => s.pts.map(p => p[1])).concat([Math.log10(2 ** -8)])), ylog: true, xticks: Array.from({length: nb}, (_, i) => i), xfmt: b => b === 0 ? "in" : b === nb - 1 ? "out" : String(b), xlabel: "decode block", ylabel: "relative token error", hlines: [{y: Math.log10(2 ** -8), label: "2⁻⁸"}], series});
  const nl = r.encode[r.reference].k.length;
  document.getElementById("t-encode").innerHTML = `<table class="tbl"><tr><th>data stream (median / p90)</th>${Array.from({length: nl}, (_, l) => `<th>block ${l + 1}</th>`).join("")}</tr>` +
    modes.flatMap(m => ["y", "k", "v"].map(key => `<tr><td>${m}, ${key === "y" ? "tokens entering" : key === "k" ? "cached keys" : "cached values"}</td>${r.encode[m][key].map(v => `<td>${fmt(v[0], 2)} / ${fmt(v[1], 2)}</td>`).join("")}</tr>`)).join("") +
    `</table><p class="note">The data stream is encoded once per observation and cached for every sampler step, so its error is shared by every draw. This is the same relative token error, for the data tokens entering each block and for each block's cached keys and values.</p>`;

  // x1 error per source in posterior widths, against SNR
  const t2 = Number(S.t2), pick = row => S.coord === "all" ? median(row) : row[Number(S.coord)];
  const pts = modes.flatMap(m => r.x1_over_width[m].flatMap((o, j) => o[t2].map((row, i) => ({x: lg(r.snr[j][i]), y: lg(pick(row)), color: MCOL[m], tip: `${m}, injection ${j + 1} source ${i + 1} (SNR ${fmt(r.snr[j][i], 3)}): ${fmt(pick(row), 3)} σ${S.coord === "all" ? " (median over coordinates)" : " in " + COORDS[S.coord]}`}))));
  legend("lg-x1", modes.map(m => [MCOL[m], "", m, true]));
  const snrX = logSpan(flat(r.snr).map(lg), 0.05);
  chart(document.getElementById("p-x1"), {label: "x1 error in widths against SNR", x: snrX, xlog: true, y: logSpan(pts.map(p => p.y).concat([-1])), ylog: true, xlabel: "source SNR", ylabel: "|Δx̂₁| / posterior width", hlines: [{y: -1, label: "0.1 σ", dash: "6 4"}], points: pts});
  document.getElementById("t-velocity").innerHTML = `<table class="tbl"><tr><th>median ‖Δv‖ / ‖v‖</th>${r.t_grid.map(t => `<th>t = ${t}</th>`).join("")}</tr>` +
    modes.map(m => `<tr><td>${m}</td>${r.t_grid.map((_, i) => `<td>${fmt(median(r.velocity[m].map(o => o[i])), 2)}</td>`).join("")}</tr>`).join("") + `</table>`;

  // after the whole sampler
  const pick2 = row => S.coord2 === "all" ? median(row) : row[Number(S.coord2)];
  const e2e = modes.flatMap(m => r.e2e_displacement[m].flatMap((o, j) => o.map((row, i) => ({x: lg(r.snr[j][i]), y: lg(pick2(row)), color: MCOL[m], tip: `${m}, injection ${j + 1} source ${i + 1} (SNR ${fmt(r.snr[j][i], 3)}): ${fmt(pick2(row), 3)} σ`}))));
  legend("lg-e2e", modes.map(m => [MCOL[m], "", m, true]));
  chart(document.getElementById("p-e2e"), {label: "sampler displacement in widths", x: snrX, xlog: true, y: logSpan(e2e.map(p => p.y).concat([-1])), ylog: true, xlabel: "source SNR", ylabel: "|x_mode − x_f32| / width", hlines: [{y: -1, label: "0.1 σ", dash: "6 4"}], points: e2e});
  const q = (a, p) => { const v = a.filter(Number.isFinite).sort((x, y) => x - y); return v.length ? v[Math.min(v.length - 1, Math.floor(p * v.length))] : NaN; };
  document.getElementById("t-e2e").innerHTML = `<table class="tbl"><tr><th>width ratio to float32</th>${COORDS.map(c => `<th>${esc(c)}</th>`).join("")}</tr>` +
    modes.map(m => `<tr><td>${m} (median [min, max])</td>${COORDS.map((_, c) => { const v = r.e2e_width_ratio[m].flatMap(o => o.map(row => row[c] == null ? NaN : row[c])); return `<td>${fmt(median(v), 3)} [${fmt(q(v, 0), 2)}, ${fmt(q(v, 1), 2)}]</td>`; }).join("")}</tr>`).join("") + `</table>`;

  // token resolution of a 1σ move
  const tok = r.token_ratio.flatMap((o, j) => o.flatMap((row, i) => row.map((v, c) => ({x: lg(r.snr[j][i]), y: lg(v), color: CCOL[c], tip: `injection ${j + 1} source ${i + 1} (SNR ${fmt(r.snr[j][i], 3)}), ${COORDS[c]}: 1σ move = ${fmt(v, 3)} × bf16 rounding`}))));
  legend("lg-token", COORDS.map((c, i) => [CCOL[i], "", c, true]));
  chart(document.getElementById("p-token"), {label: "1 sigma token move over bf16 rounding", x: snrX, xlog: true, y: logSpan(tok.map(p => p.y).concat([0])), ylog: true, xlabel: "source SNR", ylabel: "1σ token change / bf16 rounding", hlines: [{y: 0, label: "resolvable above"}], points: tok});
}
addEventListener("resize", render);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
render();
</script>
</body>
</html>
"""

MEANS = """
<ul>
<li><b>Read the output in widths, not as relative error.</b> A relative error of 10<sup>−3</sup> in x̂<sub>1</sub> does not matter at SNR 10, but it can be several posterior widths in f0 at SNR 1000. The second and third cards show which sources and coordinates cross 0.1 σ.</li>
<li><b>A late-t error moves samples further than an early one.</b> The velocity is (x̂<sub>1</sub> − x)/(1 − t), so at late t a fixed x̂<sub>1</sub> error becomes a large velocity error. But the step toward x̂<sub>1</sub> shrinks to match, so the end-to-end card is the number that decides.</li>
<li><b>The cached data stream is shared by every draw.</b> An error in the data stream's keys and values shifts every draw from that observation the same way, which shows up as a bias rather than as extra width.</li>
<li><b>CPU numbers bound the casts, not the GPU's arithmetic.</b> Training runs bf16 on GPU tensor cores. Rerun on a GPU for the numbers that describe a trained run's own sampler.</li>
</ul>
"""


def main() -> None:
    runs = {
        path.stem: json.loads(path.read_text())
        for path in sorted((REPORT / "data").glob("*.json"))
    }
    assert runs, f"no data in {REPORT / 'data'}"
    report(runs)
    data = dict(runs=runs, smoke_steps=SMOKE_STEPS, coords=COORDS)
    page = PAGE.replace("__MEANS__", MEANS).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = REPORT / "precision-error.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
