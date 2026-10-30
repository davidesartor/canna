"""Write sampler-convergence.html from data/*.json: per-source posterior widths against the ODE integrator, per run.

Stdlib only; run from the repo root:
    uv run --frozen python reports/posterior-eval/sampler-convergence/scripts/build_page.py
"""

import json
from pathlib import Path
from statistics import median

REPORT = Path(__file__).parents[1]
TRAINED_STEPS = 100_000


def reference(cells: list[dict]) -> dict:
    """The finest midpoint cell, on the log schedule when there is one: what every other cell is a ratio to."""
    midpoint = [c for c in cells if c["rule"] == "midpoint"]
    return max(midpoint, key=lambda c: (c["steps"], c["schedule"] == "log"))


def runs() -> dict:
    out = {}
    for path in sorted((REPORT / "data").glob("*.json")):
        d = json.loads(path.read_text())
        if not d.get("cells"):
            continue
        ref = reference(d["cells"])
        d["reference"] = dict(
            rule=ref["rule"], schedule=ref["schedule"], steps=ref["steps"]
        )
        d["untrained"] = d["total_steps"] < TRAINED_STEPS
        out[d["run"]] = d
    return out


def report(data: dict) -> None:
    """Print, per run, the median f0 width at each cell as a ratio to the reference."""
    for run, d in data.items():
        ref = next(
            c for c in d["cells"] if all(c[k] == v for k, v in d["reference"].items())
        )
        print(
            f"{run} ({d['total_steps']} steps{', untrained' if d['untrained'] else ''}), reference {d['reference']}"
        )
        print(f"  grid vs sample_posterior max |Δx|: {d.get('eval_match')}")
        for c in d["cells"] + [d["eval_cell"]]:
            ratio = median(
                w[0] / r[0] for w, r in zip(c["width"], ref["width"]) if r[0] > 0
            )
            print(
                f"  {c['rule']:8s} {c['schedule']:8s} {c['steps']:4d}: f0 width ratio {ratio:.3g}"
            )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA sampler convergence</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --warn: #fab219; --warn-ink: #0b0b0b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
a { color: var(--s1); }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 76ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 86ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 12px; }
.bar .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.banner { display: none; border: 1px solid var(--border); border-left: 4px solid var(--warn); background: var(--surface); border-radius: 8px; padding: 10px 14px; margin: 0 0 16px; font-size: 14px; }
.badge { display: inline-block; background: var(--warn); color: var(--warn-ink); border-radius: 4px; font-size: 11px; font-weight: 600; padding: 0 5px; margin-left: 6px; vertical-align: 1px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: auto; display: inline-block; flex: none; overflow: visible; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
svg [data-act] { cursor: pointer; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; max-width: 300px; }
.tip b { color: var(--ink); font-weight: 600; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 82ch; }
.means li { margin: 4px 0; }
.empty { color: var(--ink-2); }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>When does the flow's posterior stop depending on the integrator?</h1>
<p class="sub">Each of the eval's 10 injections is sampled from one trained flow with the ODE integrator varied: 4 to 256 steps, three step schedules and two update rules. A converged sampler gives per-source widths that no longer move as steps grow; a sampler that collapses gives widths that keep growing. Cost per call is in <a href="../../benchmarks/sampling-cost/sampling-cost.html">sampling cost</a>.</p>
<div class="bar"><div><span class="k">Run</span><span class="seg" id="run"></span></div></div>
<div class="banner" id="banner"></div>
<div id="body">
<div class="tiles" id="tiles"></div>
<div class="bar">
  <div><span class="k">Coordinate</span><span class="seg" id="col"></span></div>
</div>
<div class="card">
  <h2>At which step count do the widths stop changing?</h2>
  <p class="note" id="n-width"></p>
  <div class="legend" id="lg-width"></div>
  <div id="p-width"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Does the default step count collapse the narrow posteriors?</h2>
  <p class="note" id="n-snr"></p>
  <div class="bar">
    <div><span class="k">Schedule</span><span class="seg" id="sched"></span></div>
    <div><span class="k">Rule</span><span class="seg" id="rule"></span></div>
    <div><span class="k">Steps</span><span class="seg" id="steps"></span></div>
  </div>
  <div id="p-snr"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Does the score keep improving with more network calls?</h2>
  <p class="note">Median over sources of the CRPS at the truth, as a ratio to the reference cell, against network calls per draw (midpoint: two a step, Euler: one). Below 1 the cell scores better than the reference: with one injection per source that is noise unless it holds across schedules.</p>
  <div class="legend" id="lg-crps"></div>
  <div id="p-crps"></div><div class="tip"></div>
</div>
<div class="card means">
  <h2>What this means</h2>
  <div id="means"></div>
</div>
</div>
<div class="card empty" id="none" style="display:none">No probe output in <code>data/</code> yet: run <code>scripts/convergence.sbatch &lt;run&gt;</code>, then this page's <code>scripts/build_page.py</code>.</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const RUNS = Object.keys(D);
const SCHED = {uniform: "var(--s1)", power2: "var(--s2)", log: "var(--s3)"};
const SCHED_NAME = {uniform: "uniform", power2: "power 2", log: "log(1 − t)"};
const DASH = {midpoint: "", euler: "5 3"};
const S = {run: RUNS[RUNS.length - 1], col: 0, sched: "uniform", rule: "midpoint", steps: null};
const marks = {};

const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const act = o => esc(JSON.stringify(o));
const quant = (v, q) => { const s = v.filter(Number.isFinite).sort((a, b) => a - b); if (!s.length) return NaN; const i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo); };
const fmt = v => !Number.isFinite(v) ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.01 ? v.toFixed(Math.abs(v) >= 1 ? 2 : 3) : v.toExponential(1);
const run = () => D[S.run];
const refCell = d => d.cells.find(c => c.rule === d.reference.rule && c.schedule === d.reference.schedule && c.steps === d.reference.steps);
const cellOf = (d, rule, sched, steps) => d.cells.find(c => c.rule === rule && c.schedule === sched && c.steps === steps);
const series = d => [...new Set(d.cells.map(c => c.rule + "|" + c.schedule))].map(k => k.split("|"));
const ratios = (c, ref, key, col) => c[key].map((w, i) => w[col] / ref[key][i][col]).filter(r => Number.isFinite(r) && r > 0);
const refName = d => `${d.reference.rule}, ${SCHED_NAME[d.reference.schedule]}, ${d.reference.steps} steps`;

function seg(id, items, key, num) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = num ? Number(b.dataset.v) : b.dataset.v; setup(); schedule(); }));
}

function writeHash() {
  history.replaceState(null, "", "#" + new URLSearchParams({run: S.run, col: S.col, sched: S.sched, rule: S.rule, steps: S.steps}).toString());
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (RUNS.includes(h.get("run"))) S.run = h.get("run");
  if (h.get("col") !== null && Number(h.get("col")) >= 0 && Number(h.get("col")) < 8) S.col = Number(h.get("col"));
  if (SCHED[h.get("sched")]) S.sched = h.get("sched");
  if (DASH[h.get("rule")] !== undefined) S.rule = h.get("rule");
  if (Number(h.get("steps")) > 0) S.steps = Number(h.get("steps"));
}

document.addEventListener("click", e => {
  const t = e.target.closest("[data-act]");
  if (!t) return;
  Object.assign(S, JSON.parse(t.dataset.act));
  setup(); schedule();
});
document.addEventListener("mousemove", e => {
  document.querySelectorAll(".tip").forEach(t => { t.style.display = "none"; });
  const t = e.target.closest("[data-tip]"), card = t && t.closest(".card");
  if (!t || !card) return;
  const tip = card.querySelector(".tip"), box = card.getBoundingClientRect();
  tip.innerHTML = t.dataset.tip;
  tip.style.display = "block";
  tip.style.left = Math.min(e.clientX - box.left + 14, box.width - tip.offsetWidth - 8) + "px";
  tip.style.top = e.clientY - box.top + 14 + "px";
});

// controls that depend on the run: its coordinates, schedules and step counts
function setup() {
  const d = run();
  seg("run", RUNS.map(r => [r, esc(r) + (D[r].untrained ? '<span class="badge">untrained</span>' : "")]), "run");
  seg("col", d.columns.map((c, i) => [i, esc(c)]), "col", true);
  const schedules = [...new Set(d.cells.map(c => c.schedule))];
  if (!schedules.includes(S.sched)) S.sched = schedules[0];
  seg("sched", schedules.map(s => [s, SCHED_NAME[s] || s]), "sched");
  const rules = [...new Set(d.cells.filter(c => c.schedule === S.sched).map(c => c.rule))];
  if (!rules.includes(S.rule)) S.rule = rules[0];
  seg("rule", rules.map(r => [r, r]), "rule");
  if (!d.steps.includes(S.steps)) S.steps = d.steps.includes(d.eval_ode_steps) ? d.eval_ode_steps : d.steps[0];
  seg("steps", d.steps.map(s => [s, s + (s === d.eval_ode_steps ? " (eval)" : "")]), "steps", true);
}

function banner() {
  const d = run(), b = document.getElementById("banner");
  b.style.display = d.untrained ? "block" : "none";
  b.innerHTML = `<b>Pipeline test, not a result.</b> <code>${esc(d.run)}</code> trained for ${d.total_steps.toLocaleString("en")} steps (epoch ${d.epoch}), far short of a real run: its widths show that the probe runs end to end, not how a trained flow converges. The question is answered on narrow-B and full-B once those runs land.`;
}

function tiles() {
  const d = run(), ref = refCell(d), ev = d.eval_cell;
  const r = ratios(ev, ref, "width", S.col), m = quant(r, 0.5);
  const tile = (k, v, s) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${s}</div></div>`;
  document.getElementById("tiles").innerHTML = [
    tile("Run", esc(d.config), `${esc(d.run)}, epoch ${d.epoch}, ${esc(d.weights)} weights`),
    tile("Eval sampler", `${d.eval_ode_steps} steps`, `uniform midpoint, ${d.draws} draws per injection, ${esc(d.device)}`),
    tile(`${esc(d.columns[S.col])} width at eval steps`, fmt(m) + "×", `median over ${r.length} sources of the eval's width / the reference's (${refName(d)})`),
    tile("Grid sampler vs eval", d.eval_match === undefined ? "–" : "max |Δx| " + fmt(d.eval_match), "on the eval's own grid; near 0 means the probe's sampler is the eval's"),
  ].join("");
}

// log-y line chart over a log2 x axis, one line per (rule, schedule), with an interquartile band on the selected series
const M = {l: 56, r: 16, t: 10, b: 40};
function linePanel(hostId, legendId, key, xOf, xLabel, yLabel) {
  const d = run(), ref = refCell(d), host = document.getElementById(hostId);
  const W = Math.max(300, host.clientWidth), H = 300;
  const lines = series(d).map(([rule, sched]) => {
    const pts = d.steps.map(n => cellOf(d, rule, sched, n)).filter(Boolean).map(c => {
      const r = ratios(c, ref, key, S.col);
      return {c, x: xOf(c), med: quant(r, 0.5), lo: quant(r, 0.25), hi: quant(r, 0.75), n: r.length};
    }).filter(p => Number.isFinite(p.med));
    return {rule, sched, pts};
  });
  const xs = lines.flatMap(l => l.pts.map(p => p.x)), ys = lines.flatMap(l => l.pts.flatMap(p => [p.lo, p.hi])).filter(v => v > 0).concat([1]);
  const x0 = Math.min(...xs) / 1.2, x1 = Math.max(...xs) * 1.2;
  const y0 = Math.min(...ys) / 1.3, y1 = Math.max(...ys) * 1.3;
  const X = v => M.l + (Math.log2(v) - Math.log2(x0)) / (Math.log2(x1) - Math.log2(x0)) * (W - M.l - M.r);
  const Y = v => M.t + (Math.log10(y1) - Math.log10(v)) / (Math.log10(y1) - Math.log10(y0)) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(yLabel)} against ${esc(xLabel)}">`;
  const span = Math.log10(y1) - Math.log10(y0);
  const yticks = (span > 2.5 ? [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100] : span > 1.2 ? [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] : [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10]).filter(v => v >= y0 && v <= y1);
  yticks.forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${v === 1 ? "var(--axis)" : "var(--grid)"}"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v)}×</text>`);
  [...new Set(xs)].sort((a, b) => a - b).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${v}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(xLabel)}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
  if (key === "width") {
    const ev = xOf({steps: d.eval_ode_steps, rule: "midpoint"});
    s += `<line x1="${X(ev)}" x2="${X(ev)}" y1="${M.t}" y2="${H - M.b}" stroke="var(--ref)" stroke-dasharray="2 3"/><text x="${X(ev) + 4}" y="${M.t + 10}" class="lbl">eval</text>`;
  }
  for (const l of lines) {
    const sel = l.rule === S.rule && l.sched === S.sched, color = SCHED[l.sched] || "var(--ink-2)";
    if (sel) s += `<path d="${l.pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.hi).toFixed(1)}`).join("")}${l.pts.slice().reverse().map(p => `L${X(p.x).toFixed(1)},${Y(p.lo).toFixed(1)}`).join("")}Z" fill="${color}" opacity="0.13" pointer-events="none"/>`;
    s += `<path d="${l.pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.med).toFixed(1)}`).join("")}" fill="none" stroke="${color}" stroke-width="${sel ? 2.4 : 1.4}" stroke-dasharray="${DASH[l.rule]}" opacity="${sel ? 1 : 0.7}" pointer-events="none"/>`;
    for (const p of l.pts) {
      const tip = `<b>${l.rule}, ${SCHED_NAME[l.sched]}, ${p.c.steps} steps</b><br>${p.c.calls} calls per draw, ${fmt(p.c.seconds)} s for ${new Set(d.sources.map(q => q.injection)).size} windows<br>${esc(d.columns[S.col])} ${key === "width" ? "width" : "CRPS"} / reference: median ${fmt(p.med)}×, quartiles ${fmt(p.lo)}–${fmt(p.hi)} over ${p.n} sources`;
      s += `<circle cx="${X(p.x)}" cy="${Y(p.med)}" r="${sel ? 4.5 : 3.5}" fill="${l.rule === "euler" ? "var(--surface)" : color}" stroke="${color}" stroke-width="1.6" data-tip="${esc(tip)}" data-act="${act({rule: l.rule, sched: l.sched, steps: p.c.steps})}"/>`;
    }
  }
  host.innerHTML = s + "</svg>";
  const sw = (c, dash, fill) => `<svg width="28" height="10"><line x1="1" x2="27" y1="5" y2="5" stroke="${c}" stroke-width="2" stroke-dasharray="${dash}"/><circle cx="14" cy="5" r="3.5" fill="${fill || c}" stroke="${c}" stroke-width="1.6"/></svg>`;
  const scheds = [...new Set(lines.map(l => l.sched))], rules = [...new Set(lines.map(l => l.rule))];
  document.getElementById(legendId).innerHTML = scheds.map(k => `<span>${sw(SCHED[k], "")}${SCHED_NAME[k]}</span>`).join("") + rules.map(r => `<span>${sw("var(--ink-2)", DASH[r], r === "euler" ? "var(--surface)" : "")}${r}</span>`).join("");
}

// scatter: per-source width at the chosen cell / reference, against the source's SNR
function snrPanel() {
  const d = run(), ref = refCell(d), c = cellOf(d, S.rule, S.sched, S.steps) || d.eval_cell, host = document.getElementById("p-snr");
  document.getElementById("n-snr").innerHTML = `Each dot is one source: its ${esc(d.columns[S.col])} width at the chosen cell over its width at the reference (${refName(d)}), against its SNR. Below 1 the cell's posterior is narrower than the converged one; a collapse that worsens with SNR tilts the cloud down to the right. Click a point in the chart above to pick a cell.`;
  const pts = d.sources.map((src, i) => ({src, r: c.width[i][S.col] / ref.width[i][S.col], w: c.width[i][S.col], wr: ref.width[i][S.col]})).filter(p => Number.isFinite(p.r) && p.r > 0 && p.src.snr > 0);
  const W = Math.max(300, host.clientWidth), H = 300;
  const xv = pts.map(p => p.src.snr), yv = pts.map(p => p.r).concat([1]);
  const x0 = Math.min(...xv) / 1.15, x1 = Math.max(...xv) * 1.15, y0 = Math.min(...yv) / 1.3, y1 = Math.max(...yv) * 1.3;
  const X = v => M.l + (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - M.l - M.r);
  const Y = v => M.t + (Math.log10(y1) - Math.log10(v)) / (Math.log10(y1) - Math.log10(y0)) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="width ratio against source SNR">`;
  [1e-4, 1e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100].filter(v => v >= y0 && v <= y1).forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${v === 1 ? "var(--axis)" : "var(--grid)"}"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v)}×</text>`);
  [3, 5, 7, 10, 20, 30, 50, 100, 200, 500, 1000].filter(v => v >= x0 && v <= x1).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${v}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">source SNR (injected, against the noise PSD)</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">width / reference width</text>`;
  const color = SCHED[c.schedule] || "var(--ink-2)";
  for (const p of pts) {
    const tip = `<b>injection ${p.src.injection}, slot ${p.src.slot}</b><br>SNR ${fmt(p.src.snr)} (window ${fmt(p.src.window_snr)}), f0 at bin ${fmt(p.src.f0_bins)}<br>${esc(d.columns[S.col])} width ${fmt(p.w)} here, ${fmt(p.wr)} at the reference: ${fmt(p.r)}×`;
    s += `<circle cx="${X(p.src.snr)}" cy="${Y(p.r)}" r="4.5" fill="${color}" fill-opacity="0.75" stroke="var(--surface)" stroke-width="2" data-tip="${esc(tip)}"/>`;
  }
  host.innerHTML = s + "</svg>";
}

function means() {
  const d = run(), ref = refCell(d);
  const at = n => { const c = cellOf(d, "midpoint", "uniform", n); return c ? quant(ratios(c, ref, "width", 0), 0.5) : NaN; };
  const ev = quant(ratios(d.eval_cell, ref, "width", 0), 0.5);
  const lines = [
    `<li><b>How to read it.</b> Widths are per-source 1.4826 × MAD of the matched draws; every curve is a ratio to the finest cell (${refName(d)}). A flat curve near 1 from some step count on means the posterior has stopped depending on the integrator there; a curve still rising at the right edge means even the reference has not converged.</li>`,
    `<li><b>The step schedule is a sampling-only choice.</b> All schedules feed the network the time t it was trained on; only where the steps fall changes. The log schedule spaces 1 − t down to ${d.log_floor}, the loss's time floor, where narrow posteriors are resolved.</li>`,
  ];
  if (d.untrained) lines.push(`<li><b>No trained run here yet.</b> <code>${esc(d.run)}</code> is a ${d.total_steps.toLocaleString("en")}-step pipeline test, so none of its numbers is a finding (its f0 width at the eval's ${d.eval_ode_steps} steps is ${fmt(ev)}× the reference, uniform 128 steps ${fmt(at(128))}×). Read this page on narrow-B and full-B.</li>`);
  else lines.push(`<li><b>This run.</b> The median f0 width at the eval's ${d.eval_ode_steps} uniform steps is ${fmt(ev)}× the reference; uniform midpoint at 64 / 128 / 256 steps gives ${fmt(at(64))}× / ${fmt(at(128))}× / ${fmt(at(256))}×. A value well below 1 at the eval's steps means the eval's widths understate the flow's.</li>`);
  document.getElementById("means").innerHTML = `<ul>${lines.join("")}</ul>`;
}

function render() {
  for (const key of Object.keys(marks)) marks[key]();
  const d = run();
  banner(); tiles();
  document.getElementById("n-width").innerHTML = `Median over the ${d.sources.length} sources of each source's ${esc(d.columns[S.col])} width, as a ratio to its width at the reference cell (${refName(d)}); the band is the interquartile range of the selected line. Colour is the step schedule, dashed with open markers is Euler (one call a step), solid is the eval's midpoint rule. Click a point to show that cell in the SNR chart.`;
  linePanel("p-width", "lg-width", "width", c => c.steps, "ODE steps", "width / reference width");
  snrPanel();
  linePanel("p-crps", "lg-crps", "crps", c => (c.rule === "euler" ? 1 : 2) * c.steps, "network calls per draw", "CRPS / reference CRPS");
  means();
  writeHash();
}
let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}
if (!RUNS.length) {
  document.getElementById("body").style.display = "none";
  document.getElementById("none").style.display = "block";
} else {
  addEventListener("resize", schedule);
  addEventListener("hashchange", () => { readHash(); setup(); schedule(); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
  readHash();
  setup();
  render();
}
</script>
</body>
</html>
"""


def main() -> None:
    data = runs()
    report(data)
    page = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = REPORT / "sampler-convergence.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB, {len(data)} runs)")


if __name__ == "__main__":
    main()
