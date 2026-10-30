"""Write learned-loss-weights.html from data/<run>.json: the learned λ per time bin and coordinate group.

Stdlib only; run from the repo root:
    uv run --frozen python reports/architecture/learned-loss-weights/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]
# runs shorter than this are smoke tests: their λ cannot have reached the error
SMOKE_STEPS = 100_000


def report(runs: dict) -> None:
    """Print the numbers the README quotes."""
    for name, r in runs.items():
        print(
            f"{name}: epoch {r['epoch']} step {r['steps'][-1]}, max |λ - init| {r['moved'][-1]:.3g}"
            f" (AdamW reach {r['adam_reach'][-1]:.3g}); {r['unqueried_share']:.1%} of draws past the"
            f" {r['ode_steps']}-step sampler's last query (log10(1-s) = {r['sampler_min_decade']:.2f})"
        )
        print("  density per bin", [round(d, 4) for d in r["density"]])


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Learned loss weights</title>
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
<h1>What do the learned loss weights do?</h1>
<p class="sub">The trainer learns one log-variance λ per (time bin, coordinate group) and minimises <code>err·e<sup>−λ</sup> + λ</code>, where err is the group's mean squared velocity error in that bin of log<sub>10</sub>(1 − s). At equilibrium λ = log err, so every bin and group carries the same relative weight. This page follows λ over training and reads off what the final weights emphasise.</p>
<div class="bar"><div><span class="k">Run</span><span class="seg" id="run"></span></div></div>
<div id="smoke"></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>Where in time does training put its draws?</h2>
  <p class="note">Share of training draws per λ bin: uniform t mixed with 1 − s log-uniform on [floor, 1]. This follows from the config alone. The dashed line is the default sampler's last query (midpoint x-steps). Later bins train the flow at times that sampler never asks about.</p>
  <div id="p-density"></div>
</div>
<div class="card">
  <h2>How does λ move over training?</h2>
  <p class="note">λ per time bin for one group, at each logged epoch (darker is later), with its init (dashed). The dotted line is log err measured at the last epoch on a log-stratified diagnostic batch, which is where λ settles at equilibrium.</p>
  <div class="controls"><div><span class="k">Group</span><span class="seg" id="group"></span></div></div>
  <div class="legend" id="lg-lambda"></div>
  <div id="p-lambda"></div>
</div>
<div class="card">
  <h2>Has λ caught up with the error?</h2>
  <p class="note">λ − log err per bin and group at the last epoch. Zero means equilibrium. A positive gap means the bin is weighted less than its error warrants. AdamW moves λ by at most about its learning rate per step; the tiles above give that bound next to the largest move seen.</p>
  <div class="legend" id="lg-gap"></div>
  <div id="p-gap"></div>
</div>
<div class="card">
  <h2>What do the final weights emphasise?</h2>
  <p class="note">The share of the objective's error term that each time bin contributes: draws in the bin × err × weight, summed over groups. The learned weights (solid) are compared with the plain mean-squared flow loss (dashed). Under the plain loss, late bins dominate because the velocity error grows as 1/(1 − s).</p>
  <div class="legend" id="lg-share"></div>
  <div id="p-share"></div>
  <div class="scroll" id="t-share"></div>
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
const GCOL = ["var(--g0)", "var(--g1)", "var(--g2)", "var(--g3)", "var(--g4)"];
const S = {run: RUNS[RUNS.length - 1], group: 0};
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const fmt = (v, n = 3) => Number(v).toPrecision(n);

function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, label]) => `<button data-v="${esc(v)}">${esc(label)}</button>`).join("");
  host.onclick = e => { const b = e.target.closest("button"); if (!b) return; S[key] = typeof S[key] === "number" ? Number(b.dataset.v) : b.dataset.v; render(); };
  return () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
}
const sw = (color, dash, op = 1) => `<svg width="24" height="10" style="width:24px;display:inline-block"><line x1="1" x2="23" y1="5" y2="5" stroke="${color}" stroke-opacity="${op}" stroke-width="2" stroke-dasharray="${dash || ""}"/></svg>`;
const legend = (id, items) => document.getElementById(id).innerHTML = items.map(([c, d, label, op]) => `<span>${sw(c, d, op)}${esc(label)}</span>`).join("");

// a linear-axis chart: lines, bars and vertical markers, with a <title> on every point
function chart(host, o) {
  const M = {l: 58, r: 14, t: 10, b: 40};
  const W = Math.max(300, host.clientWidth), H = o.height || 260;
  const X = v => M.l + (v - o.x[0]) / (o.x[1] - o.x[0]) * (W - M.l - M.r);
  const Y = v => M.t + (o.y[1] - v) / (o.y[1] - o.y[0]) * (H - M.t - M.b);
  const ticks = (lo, hi, n) => { const step = 10 ** Math.floor(Math.log10((hi - lo) / n)); const m = [1, 2, 5, 10].find(m => (hi - lo) / (m * step) <= n) * step; const out = []; for (let v = Math.ceil(lo / m) * m; v <= hi + 1e-9; v += m) out.push(+v.toFixed(10)); return out; };
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(o.label)}">`;
  ticks(o.y[0], o.y[1], 6).forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${(o.yfmt || String)(v)}</text>`);
  ticks(o.x[0], o.x[1], Math.max(3, Math.floor((W - M.l) / 60))).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${(o.xfmt || String)(v)}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">${o.xlabel}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  if (o.zero) s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--ref)" stroke-width="1"/>`;
  (o.vlines || []).forEach(v => s += `<line x1="${X(v.x)}" x2="${X(v.x)}" y1="${M.t}" y2="${H - M.b}" stroke="var(--ref)" stroke-dasharray="4 3"/><text x="${X(v.x) + 4}" y="${M.t + 12}" style="fill:var(--ink-2)">${esc(v.label)}</text>`);
  (o.bars || []).forEach(b => s += `<rect x="${X(b.x0) + 1}" width="${Math.max(1, X(b.x1) - X(b.x0) - 2)}" y="${Y(Math.max(b.y, o.y[0]))}" height="${Math.max(0, Y(o.y[0]) - Y(Math.max(b.y, o.y[0])))}" fill="${b.color}" rx="2"><title>${esc(b.tip)}</title></rect>`);
  (o.series || []).forEach(se => {
    const pts = se.pts.filter(p => Number.isFinite(p[1]));
    s += `<path d="${pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("")}" fill="none" stroke="${se.color}" stroke-opacity="${se.op ?? 1}" stroke-width="${se.width || 2}" stroke-dasharray="${se.dash || ""}"/>`;
    if (se.dots !== false) pts.forEach(([x, y], i) => s += `<circle cx="${X(x)}" cy="${Y(y)}" r="3" fill="${se.color}" fill-opacity="${se.op ?? 1}"><title>${esc(se.tip ? se.tip(i) : `${se.name}: ${fmt(y)}`)}</title></circle>`);
  });
  host.innerHTML = s + "</svg>";
}
const gmean = a => Math.exp(a.reduce((s, v) => s + Math.log(v), 0) / a.length);
const span = (vals, pad = 0.06) => { const v = vals.filter(Number.isFinite); const lo = Math.min(...v), hi = Math.max(...v), d = (hi - lo) || 1; return [lo - pad * d, hi + pad * d]; };

const marks = [seg("run", RUNS.map(r => [r, r]), "run")];
function render() {
  const r = D.runs[S.run];
  if (!marks[1]) marks.push(seg("group", r.groups.map((g, i) => [i, g]), "group"));
  marks.forEach(m => m());
  const E = r.time_edges, centre = E.slice(0, -1).map((e, i) => (e + E[i + 1]) / 2);
  const smoke = r.total_steps < D.smoke_steps;
  document.getElementById("smoke").innerHTML = smoke ? `<div class="warn"><b>Smoke run.</b> ${esc(r.run)} trained ${r.total_steps.toLocaleString("en")} steps, so λ could move by at most ${fmt(r.adam_reach.at(-1), 2)}. Its λ shows the mechanism only and says nothing about where a trained run's weights settle.</div>` : "";
  const tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  document.getElementById("tiles").innerHTML = [
    tile("Run", esc(r.config), `${esc(r.run)}, commit ${r.commit.slice(0, 7)}`),
    tile("Trained", r.steps.at(-1).toLocaleString("en") + " steps", `epoch ${r.epoch} of ${r.total_steps.toLocaleString("en")} steps`),
    tile("Largest λ move", fmt(r.moved.at(-1), 3), `AdamW bound ≈ ${fmt(r.adam_reach.at(-1), 3)} (lr ${r.adam_learning_rate}, warmup ${r.warmup_steps})`),
    tile("Draws past the sampler", (100 * r.unqueried_share).toFixed(1) + "%", `1 − s below 10^${r.sampler_min_decade.toFixed(2)}, the ${r.ode_steps}-step sampler's last query`),
  ].join("");

  chart(document.getElementById("p-density"), {
    label: "training draws per time bin", x: [E[0], E.at(-1)], y: [0, Math.max(...r.density) * 1.1],
    xlabel: "log10(1 − s)", ylabel: "share of draws", yfmt: v => (100 * v).toFixed(0) + "%",
    bars: r.density.map((d, i) => ({x0: E[i], x1: E[i + 1], y: d, color: "var(--g0)", tip: `[${E[i].toFixed(3)}, ${E[i + 1].toFixed(3)}): ${(100 * d).toFixed(2)}% of draws`})),
    vlines: [{x: r.sampler_min_decade, label: `${r.ode_steps}-step sampler's last query`}],
  });

  const g = S.group, ep = r.log_var.length;
  const lam = r.log_var.map(e => e.map(b => b[g]));
  const target = r.bin_error[r.epoch - 1].map(b => b[g]);
  const series = lam.map((v, k) => ({name: `step ${r.steps[k]}`, pts: centre.map((c, i) => [c, v[i]]), color: "var(--ink)", op: 0.25 + 0.75 * (k + 1) / ep, width: 1.6, tip: i => `step ${r.steps[k]}, bin ${centre[i].toFixed(2)}: λ ${fmt(v[i])}`}));
  series.push({name: "init", pts: centre.map((c, i) => [c, r.init[i][g]]), color: GCOL[g], dash: "6 4", tip: i => `init, bin ${centre[i].toFixed(2)}: λ ${fmt(r.init[i][g])}`});
  series.push({name: "log err", pts: centre.map((c, i) => [c, target[i]]), color: GCOL[g], dash: "2 3", tip: i => `log err at epoch ${r.epoch}, bin ${centre[i].toFixed(2)}: ${fmt(target[i])}`});
  legend("lg-lambda", [["var(--ink)", "", `λ at each logged epoch (${ep})`, 1], [GCOL[g], "6 4", "init", 1], [GCOL[g], "2 3", "log err, last epoch", 1]]);
  chart(document.getElementById("p-lambda"), {label: "λ per bin over training", x: [E[0], E.at(-1)], y: span(series.flatMap(s => s.pts.map(p => p[1]))), xlabel: "log10(1 − s)", ylabel: `λ (${r.groups[g]})`, series});

  const last = r.log_var[r.epoch - 1];
  const gaps = r.groups.map((name, j) => ({name, pts: centre.map((c, i) => [c, last[i][j] - r.bin_error[r.epoch - 1][i][j]]), color: GCOL[j], tip: i => `${name}, bin ${centre[i].toFixed(2)}: λ − log err ${fmt(last[i][j] - r.bin_error[r.epoch - 1][i][j])}`}));
  legend("lg-gap", r.groups.map((name, j) => [GCOL[j], "", name, 1]));
  chart(document.getElementById("p-gap"), {label: "λ minus log err", x: [E[0], E.at(-1)], y: span(gaps.flatMap(s => s.pts.map(p => p[1])).concat([0])), xlabel: "log10(1 − s)", ylabel: "λ − log err", zero: true, series: gaps});

  const sum = rows => rows.map(b => b.reduce((a, v) => a + v, 0));
  const learned = sum(r.learned_share), plain = sum(r.plain_share);
  legend("lg-share", [["var(--ink)", "", "learned weights", 1], ["var(--ink)", "6 4", "plain flow loss", 1]]);
  chart(document.getElementById("p-share"), {
    label: "error share per time bin", x: [E[0], E.at(-1)], y: [0, Math.max(...learned, ...plain) * 1.1], xlabel: "log10(1 − s)", ylabel: "share of the error term", yfmt: v => (100 * v).toFixed(0) + "%",
    series: [{name: "learned", pts: centre.map((c, i) => [c, learned[i]]), color: "var(--ink)"}, {name: "plain", pts: centre.map((c, i) => [c, plain[i]]), color: "var(--ink)", dash: "6 4"}],
  });
  const col = (rows, j) => rows.reduce((a, b) => a + b[j], 0);
  document.getElementById("t-share").innerHTML = `<table class="tbl"><tr><th>group</th><th>coords</th><th>learned share</th><th>plain share</th><th>weight per coord, init</th><th>weight per coord, final</th></tr>` +
    r.groups.map((name, j) => `<tr><td>${esc(name)}</td><td>${r.group_sizes[j]}</td><td>${(100 * col(r.learned_share, j)).toFixed(1)}%</td><td>${(100 * col(r.plain_share, j)).toFixed(1)}%</td><td>${fmt(gmean(r.weight_init.map(b => b[j])))}</td><td>${fmt(gmean(r.weight_final.map(b => b[j])))}</td></tr>`).join("") +
    `</table><p class="note">Weights per coordinate are on the squared velocity error, relative to the plain loss's 1/8 per coordinate, as a geometric mean over the time bins (λ itself carries the 1/(1 − s)² scale).</p>`;
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
<li><b>The learned weights remove the error's scale, not its timing.</b> At equilibrium each (bin, group) term carries the same relative weight, so a bin's share of the gradient follows the share of draws that land in it, not the size of its error. Which times train the flow is then fixed by <code>late_time_mix</code> and <code>late_time_floor</code>.</li>
<li><b>λ is slow.</b> It sits on AdamW, so it moves by at most about the AdamW learning rate per step. A run must be well past distance/lr steps before its λ reflect the error instead of the init.</li>
<li><b>Some trained times go unused by the default sampler.</b> Draws later than the sampler's last query only matter if sampling uses more steps or a schedule that reaches them.</li>
</ul>
"""


def main() -> None:
    runs = {
        path.stem: json.loads(path.read_text())
        for path in sorted((REPORT / "data").glob("*.json"))
    }
    assert runs, f"no data in {REPORT / 'data'}"
    report(runs)
    data = dict(runs=runs, smoke_steps=SMOKE_STEPS)
    page = PAGE.replace("__MEANS__", MEANS).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = REPORT / "learned-loss-weights.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
