"""Write robustness.html from data/*.json: the flow's posteriors on misspecified windows against their in-distribution pairs, per run.

Stdlib only; run from the repo root:
    uv run --frozen python reports/posterior-eval/robustness/scripts/build_page.py
"""

import json
from pathlib import Path
from statistics import median

REPORT = Path(__file__).parents[1]
TRAINED_STEPS = 100_000


def runs() -> dict:
    out = {}
    for path in sorted((REPORT / "data").glob("*.json")):
        d = json.loads(path.read_text())
        if not d.get("sources"):
            continue
        d["untrained"] = d["total_steps"] < TRAINED_STEPS
        out[d["run"]] = d
    return out


def report(data: dict) -> None:
    """Print, per run and case, the median f0 width ratio to the paired baseline source and the 68% coverage of f0."""
    for run, d in data.items():
        print(
            f"{run} ({d['total_steps']} steps{', untrained' if d['untrained'] else ''}), matcher mismatch {d.get('matcher_mismatch')}"
        )
        base = {
            (r["injection"], r["slot"]): r
            for r in d["sources"]
            if r["case"] == "baseline"
        }
        for case in [c["name"] for c in d["cases"]]:
            rows = [r for r in d["sources"] if r["case"] == case and r["width"][0]]
            ratios = [
                r["width"][0] / base[r["injection"], r["base_slot"]]["width"][0]
                for r in rows
                if (r["injection"], r["base_slot"]) in base
                and base[r["injection"], r["base_slot"]]["width"][0]
            ]
            covered = [
                abs(r["rank"][0] - 0.5) < 0.34 for r in rows if r["rank"][0] is not None
            ]
            print(
                f"  {case:20s} {len(rows):4d} sources, f0 width ratio {median(ratios) if ratios else float('nan'):.3g},"
                f" f0 68% coverage {sum(covered) / max(len(covered), 1):.2f}"
            )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA flow robustness</title>
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
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
svg [data-act] { cursor: pointer; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; max-width: 320px; }
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
<h1>How does the flow behave on windows it was not trained on?</h1>
<p class="sub">Each case takes the in-distribution injections and changes one thing: the noise level, one source just outside the prior, a source fewer or more, or no source at all. The window, truth and noise realisation stay the same, so every source pairs with its own in-distribution posterior. Only the flow is used: the data come from the repo's simulator with modified parameters, the posterior from the eval's sampler.</p>
<div class="bar"><div><span class="k">Run</span><span class="seg" id="run"></span></div></div>
<div class="banner" id="banner"></div>
<div id="body">
<div class="tiles" id="tiles"></div>
<div class="bar"><div><span class="k">Coordinate</span><span class="seg" id="col"></span></div></div>
<div class="card">
  <h2>Do the posteriors widen or shift off-distribution?</h2>
  <p class="note" id="n-shift"></p>
  <div class="legend" id="lg-shift"></div>
  <div class="two"><div id="p-width"></div><div id="p-z"></div></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Is the truth still inside the flow's credible intervals?</h2>
  <p class="note">Left: the fraction of sources whose truth lies inside the central 68% and 95% of their matched draws, per case, against the nominal level (a calibrated flow sits on the line, an overconfident one below it). Right: the selected case's histogram of the truth's rank among the draws (flat when calibrated; a U means too narrow, a hump too wide). Click a case to select it.</p>
  <div class="legend" id="lg-cov"></div>
  <div class="two"><div id="p-cov"></div><div id="p-rank"></div></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What does the flow claim where there is nothing to find?</h2>
  <p class="note">Each dot is one of the flow's <span id="nslots"></span> slots in one window: the median SNR its draws would have (against the trained PSD) and the spread of its f0 draws as a fraction of the window's drawable band. A slot that found a source sits low and to the right; a slot spread uniformly over the window sits near 0.37 (1.4826 × MAD of a uniform). With no source or one missing, a slot has nothing to find: does it still claim a confident source?</p>
  <div class="bar"><div><span class="k">Cases</span><span class="seg" id="slotcase"></span></div></div>
  <div class="legend" id="lg-slot"></div>
  <div id="p-slot"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Does the flow find a fifth source it has no slot for?</h2>
  <p class="note">With one source more than the flow's slots, every draw leaves one true source without a slot. Per true source in that case, the fraction of draws that pair a slot to it, against its SNR: the added source against the four it shares the window with.</p>
  <div class="legend" id="lg-found"></div>
  <div id="p-found"></div><div class="tip"></div>
</div>
<div class="card means">
  <h2>What this means</h2>
  <div id="means"></div>
</div>
</div>
<div class="card empty" id="none" style="display:none">No probe output in <code>data/</code> yet: run <code>scripts/robustness.sbatch &lt;run&gt;</code>, then this page's <code>scripts/build_page.py</code>.</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const RUNS = Object.keys(D);
const GROUP = {perturbed: "var(--s2)", untouched: "var(--s1)"};
const S = {run: RUNS[RUNS.length - 1], col: 0, case: null, slotcase: "empty"};
const SLOTCASES = {empty: ["baseline", "empty window"], fewer: ["baseline", "one source fewer"], noise: ["noise ×0.5", "baseline", "noise ×2"]};
const SLOTCOLOR = ["var(--s1)", "var(--s2)", "var(--s3)"];
const marks = {};

const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const act = o => esc(JSON.stringify(o));
const ok = v => v !== null && v !== undefined && Number.isFinite(v);
const quant = (v, q) => { const s = v.filter(ok).sort((a, b) => a - b); if (!s.length) return NaN; const i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo); };
const fmt = v => !Number.isFinite(v) ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.01 ? v.toFixed(Math.abs(v) >= 1 ? 2 : 3) : v.toExponential(1);
const pct = v => Number.isFinite(v) ? Math.round(100 * v) + "%" : "–";
const run = () => D[S.run];
const caseNames = d => d.cases.map(c => c.name);

// each case's sources paired with their baseline selves
function paired(d) {
  const base = new Map(d.sources.filter(r => r.case === "baseline").map(r => [r.injection + ":" + r.slot, r]));
  return d.sources.map(r => ({...r, base: r.base_slot >= 0 ? base.get(r.injection + ":" + r.base_slot) : undefined}));
}

function seg(id, items, key, num) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${esc(v)}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = num ? Number(b.dataset.v) : b.dataset.v; setup(); schedule(); }));
}
function writeHash() {
  history.replaceState(null, "", "#" + new URLSearchParams({run: S.run, col: S.col, case: S.case, slots: S.slotcase}).toString());
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (RUNS.includes(h.get("run"))) S.run = h.get("run");
  if (h.get("col") !== null && Number(h.get("col")) >= 0 && Number(h.get("col")) < 8) S.col = Number(h.get("col"));
  if (h.get("case")) S.case = h.get("case");
  if (SLOTCASES[h.get("slots")]) S.slotcase = h.get("slots");
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

function setup() {
  const d = run();
  seg("run", RUNS.map(r => [r, esc(r) + (D[r].untrained ? '<span class="badge">untrained</span>' : "")]), "run");
  seg("col", d.columns.map((c, i) => [i, esc(c)]), "col", true);
  if (!caseNames(d).includes(S.case)) S.case = caseNames(d)[0];
  const avail = Object.entries(SLOTCASES).filter(([, cs]) => cs.every(c => caseNames(d).includes(c)));
  if (!avail.some(([k]) => k === S.slotcase) && avail.length) S.slotcase = avail[0][0];
  seg("slotcase", avail.map(([k, cs]) => [k, cs.filter(c => c !== "baseline").join(", ")]), "slotcase");
}

function banner() {
  const d = run(), b = document.getElementById("banner");
  b.style.display = d.untrained ? "block" : "none";
  b.innerHTML = `<b>Pipeline test, not a result.</b> <code>${esc(d.run)}</code> trained for ${d.total_steps.toLocaleString("en")} steps (epoch ${d.epoch}), far short of a real run: its posteriors are close to the prior everywhere, so they show that every case runs end to end, not how a trained flow handles misspecification. The question is answered on narrow-B and full-B once those runs land.`;
}

function tiles() {
  const d = run(), nInj = d.injections.length, nEval = d.injections.filter(i => i.eval).length;
  const tile = (k, v, s) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${s}</div></div>`;
  document.getElementById("tiles").innerHTML = [
    tile("Run", esc(d.config), `${esc(d.run)}, epoch ${d.epoch}, ${esc(d.device)}`),
    tile("Windows per case", nInj, `the eval's ${nEval}${nInj > nEval ? ` and ${nInj - nEval} of the scorecard's random ones` : ""}, ${d.n_sources} sources each`),
    tile("Sampler", `${d.ode_steps} steps`, `the eval's sampler, ${d.draws} draws per window`),
    tile("Matcher check", d.matcher_mismatch === undefined ? "–" : pct(d.matcher_mismatch), "baseline draws the page's matcher pairs unlike the eval's; 0% expected"),
  ].join("");
}

// one row per case: median and interquartile range of a per-source statistic, perturbed and untouched sources apart
function rowPanel(hostId, rows, {log, ref, label, xfmt, title}) {
  const host = document.getElementById(hostId), d = run(), names = caseNames(d);
  const W = Math.max(300, host.clientWidth), rowH = 26, L0 = Math.min(150, W * 0.38), H = names.length * rowH + 50;
  const vals = rows.flatMap(r => [r.lo, r.hi]).filter(v => ok(v) && (!log || v > 0)).concat([ref]);
  let x0 = Math.min(...vals), x1 = Math.max(...vals);
  if (log) { x0 /= 1.3; x1 *= 1.3; } else { const pad = (x1 - x0) * 0.08 || 0.5; x0 = Math.min(0, x0 - pad); x1 += pad; }
  const T = v => log ? Math.log10(v) : v;
  const X = v => L0 + (T(v) - T(x0)) / (T(x1) - T(x0)) * (W - L0 - 16);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(title)}">`;
  s += `<text x="${L0}" y="10" class="lbl">${esc(title)}</text>`;
  const ticks = log ? [1e-3, 1e-2, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1e3].filter(v => v >= x0 && v <= x1) : niceTicks(x0, x1);
  ticks.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="18" y2="${H - 30}" stroke="${v === ref ? "var(--axis)" : "var(--grid)"}"/><text x="${X(v)}" y="${H - 16}" text-anchor="middle">${xfmt(v)}</text>`);
  s += `<line x1="${X(ref)}" x2="${X(ref)}" y1="18" y2="${H - 30}" stroke="var(--ref)" stroke-dasharray="3 3"/>`;
  s += `<text x="${(L0 + W - 16) / 2}" y="${H - 2}" text-anchor="middle">${esc(label)}</text>`;
  names.forEach((name, i) => {
    const y = 22 + i * rowH, on = name === S.case;
    s += `<text class="lbl" x="${L0 - 8}" y="${y + 13}" text-anchor="end" font-weight="${on ? 700 : 400}" data-act="${act({case: name})}">${esc(name)}</text>`;
    for (const [g, dy] of [["perturbed", 6], ["untouched", 14]]) {
      const r = rows.find(q => q.case === name && q.group === g);
      if (!r || !ok(r.med) || (log && r.med <= 0)) continue;
      const tip = `<b>${esc(name)}, ${g} sources</b><br>${esc(label)}: median ${xfmt(r.med)}, quartiles ${xfmt(r.lo)}–${xfmt(r.hi)} over ${r.n}`;
      s += `<line x1="${X(Math.max(r.lo, log ? x0 : -Infinity))}" x2="${X(r.hi)}" y1="${y + dy}" y2="${y + dy}" stroke="${GROUP[g]}" stroke-width="2" pointer-events="none"/>`;
      s += `<circle cx="${X(r.med)}" cy="${y + dy}" r="4.5" fill="${GROUP[g]}" stroke="var(--surface)" stroke-width="2" data-tip="${esc(tip)}" data-act="${act({case: name})}"/>`;
    }
  });
  host.innerHTML = s + "</svg>";
}
function niceTicks(a, b) {
  const step = Math.pow(10, Math.floor(Math.log10((b - a) / 4))), m = [1, 2, 5, 10].find(k => (b - a) / (k * step) <= 6) * step;
  const out = []; for (let v = Math.ceil(a / m) * m; v <= b + 1e-12; v += m) out.push(Number(v.toPrecision(6)));
  return out;
}
function summary(rows, value) {
  const out = [];
  for (const name of caseNames(run())) for (const g of ["perturbed", "untouched"]) {
    const v = rows.filter(r => r.case === name && (r.perturbed ? "perturbed" : "untouched") === g).map(value).filter(ok);
    if (v.length) out.push({case: name, group: g, med: quant(v, 0.5), lo: quant(v, 0.25), hi: quant(v, 0.75), n: v.length});
  }
  return out;
}

function shiftPanels() {
  const d = run(), c = S.col, col = d.columns[c], rows = paired(d);
  document.getElementById("n-shift").innerHTML = `Left: each matched source's ${esc(col)} width over its own width in the baseline case (same window, truth and noise); 1 means unchanged. Right: the offset of its posterior median from the truth in units of its own width, |median − truth| / width; a calibrated posterior keeps this around 0.67 (the median |z| of a Gaussian), the dashed line. Orange is the source the case perturbs, blue the others in the window (the noise cases perturb the window, not a source). Dots are medians, bars the interquartile range.`;
  rowPanel("p-width", summary(rows, r => r.base && ok(r.width[c]) && ok(r.base.width[c]) && r.base.width[c] > 0 ? r.width[c] / r.base.width[c] : NaN), {log: true, ref: 1, label: `${col} width / baseline width`, xfmt: v => fmt(v) + "×", title: "width against the source's baseline"});
  rowPanel("p-z", summary(rows, r => ok(r.offset[c]) && ok(r.width[c]) && r.width[c] > 0 ? Math.abs(r.offset[c]) / r.width[c] : NaN), {log: true, ref: 0.6745, label: `|${col} offset| / width`, xfmt: fmt, title: "offset in posterior widths"});
  const sw = c => `<svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${c}" stroke-width="2"/><circle cx="11" cy="5" r="4" fill="${c}"/></svg>`;
  document.getElementById("lg-shift").innerHTML = `<span>${sw(GROUP.perturbed)}perturbed source</span><span>${sw(GROUP.untouched)}untouched sources</span>`;
}

// coverage per case from the truth's rank among the draws, then the selected case's rank histogram
function covPanels() {
  const d = run(), c = S.col, names = caseNames(d), host = document.getElementById("p-cov");
  const ranks = name => d.sources.filter(r => r.case === name).map(r => r.rank[c]).filter(ok);
  const W = Math.max(300, host.clientWidth), rowH = 26, L0 = Math.min(150, W * 0.38), H = names.length * rowH + 50;
  const X = v => L0 + v * (W - L0 - 16);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="coverage per case">`;
  s += `<text x="${L0}" y="10" class="lbl">truth inside the central interval</text>`;
  [0, 0.25, 0.5, 0.75, 1].forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="18" y2="${H - 30}" stroke="var(--grid)"/><text x="${X(v)}" y="${H - 16}" text-anchor="middle">${pct(v)}</text>`);
  [[0.68, "var(--s1)"], [0.95, "var(--s3)"]].forEach(([v, col]) => s += `<line x1="${X(v)}" x2="${X(v)}" y1="18" y2="${H - 30}" stroke="${col}" stroke-dasharray="3 3"/>`);
  s += `<text x="${(L0 + W - 16) / 2}" y="${H - 2}" text-anchor="middle">fraction of sources (${esc(d.columns[c])})</text>`;
  names.forEach((name, i) => {
    const y = 22 + i * rowH, r = ranks(name), on = name === S.case;
    s += `<text class="lbl" x="${L0 - 8}" y="${y + 13}" text-anchor="end" font-weight="${on ? 700 : 400}" data-act="${act({case: name})}">${esc(name)}</text>`;
    if (!r.length) { s += `<text x="${L0 + 4}" y="${y + 13}">no source to cover</text>`; return; }
    for (const [level, col, dy] of [[0.68, "var(--s1)", 6], [0.95, "var(--s3)", 14]]) {
      const cov = r.filter(v => Math.abs(v - 0.5) < level / 2).length / r.length;
      s += `<rect x="${L0}" y="${y + dy - 3}" width="${X(cov) - L0}" height="6" rx="2" fill="${col}" opacity="${on ? 1 : 0.75}" data-tip="${esc(`<b>${esc(name)}</b><br>${pct(cov)} of ${r.length} truths inside the central ${pct(level)}`)}" data-act="${act({case: name})}"/>`;
    }
  });
  host.innerHTML = s + "</svg>";
  const sw = c => `<svg width="22" height="10"><rect x="1" y="2" width="20" height="6" rx="2" fill="${c}"/></svg>`;
  document.getElementById("lg-cov").innerHTML = `<span>${sw("var(--s1)")}68% interval</span><span>${sw("var(--s3)")}95% interval</span><span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(--ink-2)" stroke-dasharray="3 3"/></svg>nominal level</span>`;

  // rank histogram of the selected case
  const r = ranks(S.case), bins = 10, counts = Array(bins).fill(0);
  r.forEach(v => counts[Math.min(bins - 1, Math.floor(v * bins))]++);
  const h2 = document.getElementById("p-rank"), W2 = Math.max(260, h2.clientWidth), H2 = H, Mg = {l: 40, r: 10, t: 22, b: 30};
  const ymax = Math.max(1, ...counts, r.length / bins * 1.5);
  const XB = i => Mg.l + i / bins * (W2 - Mg.l - Mg.r), YB = v => Mg.t + (1 - v / ymax) * (H2 - Mg.t - Mg.b);
  let t = `<svg viewBox="0 0 ${W2} ${H2}" width="${W2}" role="img" aria-label="rank histogram">`;
  t += `<text x="${Mg.l}" y="10" class="lbl">${esc(S.case)}: rank of the truth among the draws (${r.length} sources)</text>`;
  niceTicks(0, ymax).forEach(v => t += `<line x1="${Mg.l}" x2="${W2 - Mg.r}" y1="${YB(v)}" y2="${YB(v)}" stroke="var(--grid)"/><text x="${Mg.l - 6}" y="${YB(v) + 4}" text-anchor="end">${v}</text>`);
  counts.forEach((n, i) => t += `<rect x="${XB(i) + 1}" y="${YB(n)}" width="${XB(1) - XB(0) - 2}" height="${YB(0) - YB(n)}" rx="2" fill="var(--s1)" data-tip="${esc(`rank ${(i / bins).toFixed(1)}–${((i + 1) / bins).toFixed(1)}: <b>${n}</b> sources`)}"/>`);
  if (r.length) t += `<line x1="${Mg.l}" x2="${W2 - Mg.r}" y1="${YB(r.length / bins)}" y2="${YB(r.length / bins)}" stroke="var(--ref)" stroke-dasharray="3 3"/>`;
  [0, 0.5, 1].forEach(v => t += `<text x="${XB(v * bins)}" y="${H2 - 12}" text-anchor="middle">${v}</text>`);
  t += `<line x1="${Mg.l}" x2="${W2 - Mg.r}" y1="${YB(0)}" y2="${YB(0)}" stroke="var(--axis)"/>`;
  h2.innerHTML = t + "</svg>";
}

// every slot's claimed SNR against its f0 spread over the window
function slotPanel() {
  const d = run(), host = document.getElementById("p-slot"), cases = SLOTCASES[S.slotcase].filter(c => caseNames(d).includes(c));
  const band = new Map(d.injections.map(i => [i.index, i.f0_window_bins]));
  const pts = d.slots.filter(r => cases.includes(r.case)).map(r => ({...r, frac: r.f0_width / band.get(r.injection)})).filter(p => p.snr > 0 && p.frac > 0);
  const W = Math.max(300, host.clientWidth), H = 300, M = {l: 56, r: 16, t: 10, b: 40};
  if (!pts.length) { host.innerHTML = `<p class="note">No slots for these cases.</p>`; return; }
  const xv = pts.map(p => p.snr), yv = pts.map(p => p.frac).concat([0.37]);
  const x0 = Math.min(...xv) / 1.2, x1 = Math.max(...xv) * 1.2, y0 = Math.min(...yv) / 1.5, y1 = Math.max(...yv) * 1.5;
  const X = v => M.l + (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - M.l - M.r);
  const Y = v => M.t + (Math.log10(y1) - Math.log10(v)) / (Math.log10(y1) - Math.log10(y0)) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="slot SNR against f0 spread">`;
  [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1].filter(v => v >= y0 && v <= y1).forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v)}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(0.37)}" y2="${Y(0.37)}" stroke="var(--ref)" stroke-dasharray="3 3"/><text x="${W - M.r}" y="${Y(0.37) - 4}" text-anchor="end" class="lbl">uniform over the window</text>`;
  [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000].filter(v => v >= x0 && v <= x1).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${v}</text>`);
  const top = d.snr_top;
  if (top >= x0 && top <= x1) s += `<line x1="${X(top)}" x2="${X(top)}" y1="${M.t}" y2="${H - M.b}" stroke="var(--axis)" stroke-dasharray="2 4"/><text x="${X(top) - 4}" y="${M.t + 10}" text-anchor="end" class="lbl">prior's 99.9% source SNR</text>`;
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">claimed SNR (median over the slot's draws)</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">f0 width / window band</text>`;
  cases.forEach((name, k) => {
    for (const p of pts.filter(q => q.case === name)) {
      const tip = `<b>${esc(name)}</b>, window ${p.injection}, slot ${p.slot}<br>claimed SNR ${fmt(p.snr)}<br>f0 width ${fmt(p.f0_width)} bins of a ${fmt(band.get(p.injection))}-bin band`;
      s += `<circle cx="${X(p.snr)}" cy="${Y(p.frac)}" r="4" fill="${SLOTCOLOR[k]}" fill-opacity="0.75" stroke="var(--surface)" stroke-width="1.5" data-tip="${esc(tip)}"/>`;
    }
  });
  host.innerHTML = s + "</svg>";
  const sw = c => `<svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="${c}"/></svg>`;
  document.getElementById("lg-slot").innerHTML = cases.map((name, k) => `<span>${sw(SLOTCOLOR[k])}${esc(name)}</span>`).join("");
}

function foundPanel() {
  const d = run(), host = document.getElementById("p-found"), card = host.closest(".card");
  const rows = d.sources.filter(r => r.case === "one source more" && r.snr > 0);
  card.style.display = rows.length ? "" : "none";
  if (!rows.length) return;
  const W = Math.max(300, host.clientWidth), H = 240, M = {l: 56, r: 16, t: 10, b: 40};
  const xv = rows.map(r => r.snr), x0 = Math.min(...xv) / 1.2, x1 = Math.max(...xv) * 1.2;
  const X = v => M.l + (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - M.l - M.r);
  const Y = v => M.t + (1 - v) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="found fraction against SNR">`;
  [0, 0.25, 0.5, 0.75, 1].forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${pct(v)}</text>`);
  [0.3, 1, 3, 10, 30, 100, 300, 1000].filter(v => v >= x0 && v <= x1).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${v}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">true source SNR</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">draws with a slot on it</text>`;
  for (const r of rows) {
    const g = r.perturbed ? "perturbed" : "untouched";
    const tip = `<b>window ${r.injection}, ${r.perturbed ? "the added source" : "slot " + r.base_slot + " of the baseline"}</b><br>SNR ${fmt(r.snr)}<br>a slot pairs with it in ${pct(r.found)} of draws`;
    s += `<circle cx="${X(r.snr)}" cy="${Y(r.found)}" r="4.5" fill="${GROUP[g]}" fill-opacity="0.8" stroke="var(--surface)" stroke-width="1.5" data-tip="${esc(tip)}"/>`;
  }
  host.innerHTML = s + "</svg>";
  const sw = c => `<svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="${c}"/></svg>`;
  document.getElementById("lg-found").innerHTML = `<span>${sw(GROUP.perturbed)}the added source</span><span>${sw(GROUP.untouched)}the window's own sources</span>`;
}

function means() {
  const d = run();
  const lines = [
    `<li><b>How to read it.</b> Every case shares windows, truths and noise draws with the baseline, so a width ratio away from 1 or an offset above the dashed line is the case's doing, not the noise's. Coverage below the nominal line with offsets above it is the failure that matters: a confident wrong answer. Wider posteriors that still cover are the graceful one.</li>`,
    `<li><b>The noise cases are a PSD the flow does not know.</b> The data carry half or twice the trained noise power, but the flow whitens by its own PSD, so it sees every source at the wrong SNR; widths should scale roughly with √(PSD) if the flow reads the noise from the data, and stay put if it does not.</li>`,
    `<li><b>The slot chart is the flow's false-alarm behaviour.</b> The flow always returns ${d.n_sources} sources. In an empty window an honest flow spreads its slots over the window at prior-range SNR; tight f0 with high claimed SNR there would be a confident phantom source.</li>`,
  ];
  if (d.untrained) lines.push(`<li><b>No trained run here yet.</b> <code>${esc(d.run)}</code> is a ${d.total_steps.toLocaleString("en")}-step pipeline test, so none of its numbers is a finding. Read this page on narrow-B and full-B.</li>`);
  document.getElementById("means").innerHTML = `<ul>${lines.join("")}</ul>`;
}

function render() {
  for (const key of Object.keys(marks)) marks[key]();
  banner(); tiles(); shiftPanels(); covPanels(); slotPanel(); foundPanel(); means();
  document.getElementById("nslots").textContent = run().n_sources;
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
    out = REPORT / "robustness.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB, {len(data)} runs)")


if __name__ == "__main__":
    main()
