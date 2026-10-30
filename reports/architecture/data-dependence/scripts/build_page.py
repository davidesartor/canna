"""Write data-dependence.html from data/*.json: how much each source slot's velocity depends on the data.

Stdlib only; run from the repo root:
    uv run --frozen --no-sync python reports/architecture/data-dependence/scripts/build_page.py
"""

import json
import re
from pathlib import Path

REPORT = Path(__file__).parents[1]
# runs shorter than this are pipeline checks, never findings
SMOKE_STEPS = 50_000
REFERENCE_COMPARISON = "../../posterior-eval/reference-comparison/"
RUNS = REPORT.parents[2] / "outputs" / "runs"


def means(runs: list[dict]) -> str:
    """The closing card: what the measured runs can and cannot say yet."""
    trained = [r for r in runs if not r["smoke"]]
    status = (
        "<li><b>No trained run has been measured yet.</b> "
        + ", ".join(f"<code>{r['run']}</code> ({r['steps']:,} steps)" for r in runs)
        + " only checks the pipeline end to end; its numbers describe a network that has barely trained and say nothing about the architecture.</li>"
        if not trained
        else "<li>Measured trained runs: "
        + ", ".join(f"<code>{r['run']}</code> ({r['steps']:,} steps)" for r in trained)
        + ". Read the cards for those; smoke runs stay labelled.</li>"
    )
    return f"""<ul>
{status}
<li><b>Data share near 0 at late t</b> for a coordinate means its velocity there is set by x<sub>t</sub> alone: the flow has stopped reading the data and only contracts. That is right once x<sub>t</sub> already pins the source, wrong if it happens before the posterior width is reached (compare the late-time widths in <a href="{REFERENCE_COMPARISON}">reference comparison</a>).</li>
<li><b>Data share that grows with SNR</b> is expected: a loud source's data says more. A flat data share across SNR, at a t where the posterior is still wide, points at the conditioning path rather than the physics.</li>
<li><b>Own-band share well above its baseline</b> means a slot reads its own source's Doppler band, as the f0 RoPE intends. Own-band share near the baseline (uniform reading) or a large other-bands share means slots read each other's sources, which the slot sorting and <code>match_sources</code> then have to undo.</li>
<li><b>Enrichment that falls off within a few bins of f0</b> is a local reader; a flat curve is a global one.</li>
</ul>"""


def probe_steps(run: str) -> int | None:
    """The run's total_steps from its config.yaml on this machine, if its outputs are here."""
    path = RUNS / run / "config.yaml"
    found = re.search(r"^total_steps:\s*(\d+)", path.read_text(), re.M) if path.exists() else None
    return int(found.group(1)) if found else None


def main() -> None:
    runs = []
    for path in sorted((REPORT / "data").glob("*.json")):
        r = json.loads(path.read_text())
        r["tag"] = path.stem
        r["smoke"] = r["steps"] < SMOKE_STEPS
        runs.append(r)
    probes = []
    for path in sorted((REPORT / "data" / "velocity-probe").glob("*.json")):
        r = json.loads(path.read_text())
        r["tag"] = path.stem
        measured = next((m["total_steps"] for m in runs if m["run"] == r["run"]), None)
        r["total_steps"] = measured or probe_steps(r["run"])
        r["smoke"] = r["total_steps"] is not None and r["total_steps"] < SMOKE_STEPS
        probes.append(r)
    page = REPORT / "data-dependence.html"
    page.write_text(
        PAGE.replace("__DATA__", json.dumps(runs))
        .replace("__SMOKE__", str(SMOKE_STEPS))
        .replace("__PROBES__", json.dumps(probes))
        .replace("__MEANS__", means(runs) if runs else "<p>No run measured yet.</p>")
    )
    print(f"{page}: {len(runs)} run(s): {', '.join(r['tag'] for r in runs)}; {len(probes)} probe(s)")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA velocity data dependence</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e; --warn-bg: #fdf3dc; --warn: #7a5200;
  --c0: #2a78d6; --c1: #eb6834; --c2: #1baf7a; --c3: #eda100; --c4: #e87ba4; --c5: #008300; --c6: #6250d6; --c7: #e34948;
  --heat: 42, 120, 214;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7; --warn-bg: #3a2c0a; --warn: #f0c870;
    --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300; --c6: #9085e9; --c7: #e66767;
    --heat: 57, 135, 229;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
a { color: inherit; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 76ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 90ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.bar, .controls { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 12px; }
.k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.banner { background: var(--warn-bg); color: var(--warn); border-radius: 8px; padding: 8px 12px; margin: 0 0 16px; font-size: 14px; }
.meta { color: var(--muted); font-size: 13px; margin: 0 0 16px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: auto; display: inline-block; flex: none; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; max-width: 320px; }
.tip b { color: var(--ink); font-weight: 600; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 84ch; }
.means li { margin: 4px 0; }
.empty { color: var(--muted); }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>Does each source slot's velocity follow the data?</h1>
<p class="sub">The sampler's velocity of a trained LisaFlow, on fresh training observations at fixed path times t: how much of it moves with the data rather than with x<sub>t</sub>, and which data bins it is sensitive to. The last card is the velocity probe: whether the network has learned each coordinate and reads the data's phase. The first cards ask where the velocity's information comes from.</p>
<div class="bar">
  <div><span class="k">Run</span><span class="seg" id="run"></span></div>
  <div><span class="k">t</span><span class="seg" id="t"></span></div>
</div>
<div id="banner"></div>
<p class="meta" id="meta"></p>
<div class="card">
  <h2>How much of the velocity moves with the data, and how much with x<sub>t</sub>?</h2>
  <p class="note">Per coordinate, against path time t: the mean square change of the velocity when only the data change (another observation's y, same x<sub>t</sub>) or only x<sub>t</sub> changes (another base draw, same observation), each over twice the velocity's variance. A value of 1 is a fully independent redraw; 0 is no dependence. The two need not add to 1 (they are total effects, interactions count in both).</p>
  <div class="controls"><div><span class="k">Change</span><span class="seg" id="which"></span></div><div><span class="k">Slot</span><span class="seg" id="slot"></span></div></div>
  <div class="legend" id="lg-share"></div>
  <div id="p-share"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Does a louder source's velocity depend more on the data?</h2>
  <p class="note">The data share at the selected t, per coordinate, by the source's SNR (all slots pooled). Bins with few sources are noisy; hover for the count.</p>
  <div class="legend" id="lg-snr"></div>
  <div id="p-snr"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Is a slot sensitive to its own source's band?</h2>
  <p class="note">The squared Jacobian of the velocity in the whitened data, summed over channels, as shares of its total per slot and coordinate: on the bins holding <span id="bp"></span> of the slot's own source's power, on the other sources' bands, and on the rest. Dashed: the share each set would get from uniform sensitivity (its bin fraction). Mean over slots and the selected coordinates.</p>
  <div class="controls"><div><span class="k">Coordinate</span><span class="seg" id="coord"></span></div></div>
  <div class="legend" id="lg-band"></div>
  <div id="p-band"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How far from its f0 does a slot read?</h2>
  <p class="note">Sensitivity per bin at a distance from the slot's own f0 bin, relative to uniform (1 = uniform; 10 = ten times the uniform density), at the selected t, per coordinate. Rings are 0, 1, 2–3, 4–7, … bins.</p>
  <div class="legend" id="lg-ring"></div>
  <div id="p-ring"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Where does one observation's velocity look?</h2>
  <p class="note">One observation at the selected t: each slot's sensitivity per window bin (summed over coordinates, as a share of the slot's total; darker is more). Ticks above a row mark its own source's band, the triangle its f0.</p>
  <div id="p-example"></div><div class="tip"></div>
</div>
<div class="card" id="probe-card">
  <h2>Has the network learned each coordinate, and does it read the data's phase?</h2>
  <p class="note" id="probe-note"></p>
  <div class="controls">
    <div id="probe-run-box"><span class="k">Probe</span><span class="seg" id="probe"></span></div>
    <div><span class="k">Metric</span><span class="seg" id="metric"></span></div>
    <div id="change-box"><span class="k">Data change</span><span class="seg" id="change"></span></div>
  </div>
  <div id="probe-banner"></div>
  <div class="legend" id="probe-legend"></div>
  <div id="p-probe"></div><div class="tip"></div>
</div>
<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const RUNS = __DATA__;
const PROBES = __PROBES__;
const SMOKE = __SMOKE__;
const COLORS = Array.from({length: 8}, (_, i) => `var(--c${i})`);
const S = {run: null, t: 3, which: "data", slot: "all", coord: "all", probe: null, metric: "r2", change: "rotate 0.30"};
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const f3 = v => v == null || !isFinite(v) ? "—" : v.toFixed(3);
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const sw = (color, dash) => `<svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="${color}" stroke-width="2" stroke-dasharray="${dash || ""}"/></svg>`;
const legend = (id, items) => document.getElementById(id).innerHTML = items.map(([c, d, t]) => `<span>${sw(c, d)}${t}</span>`).join("");
function seg(id, items, key, numeric) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === String(S[key])}">${text}</button>`).join("");
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = numeric ? Number(b.dataset.v) : b.dataset.v; draw(); }));
}
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
// categorical x, linear or log y; series: {color, dash, pts: [[i, y, tip]]}
function chart(id, o) {
  const host = document.getElementById(id), W = Math.max(300, host.clientWidth), H = 260, M = {l: 52, r: 16, t: 10, b: 38};
  const n = o.x.length, X = i => M.l + (n === 1 ? 0.5 : i / (n - 1)) * (W - M.l - M.r);
  const [lo, hi] = o.y, log = o.log, tr = v => log ? Math.log10(v) : v;
  const Y = v => M.t + (tr(hi) - tr(Math.min(hi, Math.max(lo, v)))) / (tr(hi) - tr(lo)) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(o.label)}">`;
  o.ticks.forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  o.x.forEach((v, i) => s += `<text x="${X(i)}" y="${H - M.b + 16}" text-anchor="middle">${v}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${o.xlabel}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  if (o.mark != null) s += `<line x1="${X(o.mark)}" x2="${X(o.mark)}" y1="${M.t}" y2="${H - M.b}" stroke="var(--ref)" stroke-dasharray="3 3"/>`;
  o.series.forEach(se => {
    const pts = se.pts.filter(p => p[1] != null && isFinite(p[1]) && (!log || p[1] > 0));
    let d = "", gap = true;
    se.pts.forEach(p => { const ok = p[1] != null && isFinite(p[1]) && (!log || p[1] > 0); if (ok) { d += `${gap ? "M" : "L"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`; } gap = !ok; });
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2" stroke-dasharray="${se.dash || ""}"/>`;
    if (!se.dash) pts.forEach(p => s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="${se.color}" stroke="var(--surface)" stroke-width="2"/><circle cx="${X(p[0])}" cy="${Y(p[1])}" r="10" fill="transparent" data-tip="${esc(p[2])}"/>`);
  });
  host.innerHTML = s + "</svg>";
}
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const fmt = v => v == null || !isFinite(v) ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : Number(v.toPrecision(3)).toString();
const logTicks = (lo, hi) => { const out = [], e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi)), mults = e1 - e0 <= 2 ? [1, 2, 5] : [1]; for (let e = e0; e <= e1; e++) mults.forEach(m => { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }); return out; };
function linTicks(hi, count) {
  const raw = hi / count, e = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map(m => m * e).find(v => v >= raw), out = [];
  for (let i = 0; i * step <= hi + 1e-12; i++) out.push(i * step);
  return out;
}
const dotKey = c => `<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${c}"/></svg>`;
function drawProbe() {
  const host = document.getElementById("p-probe"), note = document.getElementById("probe-note");
  const empty = text => { host.innerHTML = `<p class="empty">${text}</p>`; note.textContent = ""; document.getElementById("probe-legend").innerHTML = ""; document.getElementById("probe-banner").innerHTML = ""; ["probe-run-box", "change-box"].forEach(i => { document.getElementById(i).style.display = "none"; }); };
  if (!PROBES.length) return empty("No velocity probe yet: <code>sbatch $CANNA_SBATCH_CPU reports/architecture/data-dependence/scripts/velocity_probe.sbatch &lt;run&gt;</code>");
  const pr = PROBES.find(p => p.tag === S.probe) || PROBES[0];
  seg("probe", PROBES.map(p => [p.tag, p.tag + (p.smoke ? " (smoke)" : "")]), "probe");
  seg("metric", [["r2", "R² by path position"], ["spread", "spread and head"], ["change", "response to the data"], ["phase", "head vs x_t phase"]], "metric");
  seg("change", Object.keys(pr.changes).filter(k => k.startsWith("rotate")).map(k => [k, `rotate ${k.split(" ")[1]} rad`]), "change");
  document.getElementById("probe-run-box").style.display = PROBES.length > 1 ? "" : "none";
  document.getElementById("change-box").style.display = S.metric === "change" ? "" : "none";
  document.getElementById("probe-banner").innerHTML = pr.smoke
    ? `<div class="banner"><b>Smoke run: ${pr.total_steps.toLocaleString()} steps.</b> A pipeline check, not a finding: these numbers describe a barely trained network.</div>` : "";
  const C = pr.coordinates.map(c => c.replace("z+", "z₊").replace("z-", "z₋"));
  const notes = {
    r2: "R² of the predicted velocity against the training target, per flow coordinate and band of path position s: 1 is the target exactly, 0 no better than its mean. A coordinate the network never learned stays near 0 at every s; near s = 1 the target is tiny and R² goes negative without meaning much.",
    spread: "The predicted velocity's spread over the target's (blue), and the output head's last-layer row norm over the median row (orange); the gray line at 1 is the target. A coordinate the head has dropped sits far below 1 in both.",
    change: "RMS change of the velocity, over the target's spread, when the whitened data are rotated by a phase (blue) and when another sample's data are swapped in (gray), at the same x_t and s. A rotation moves only the phases in the true posterior, so the z± rows should respond and the rest stay small; a network that ignores the data's phase shows z± near 0.",
    phase: "Median angle error of φ₊ and φ₋ in the x₁ the flow implies (blue), against x_t's own (gray), per band of path position s. Blue well below gray early in the path means the head reads the phases from the data rather than copying them from x_t.",
  };
  note.textContent = `${notes[S.metric]} Run ${pr.run}, epoch ${pr.epoch}, ${pr.weights} weights, ${pr.samples} fresh training samples on ${pr.device}.`;
  const W = Math.max(320, host.clientWidth), top = 8, svg = (H, body) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${body}</svg>`;
  let s = "";
  if (S.metric === "r2") {
    document.getElementById("probe-legend").innerHTML = "";
    const ML = 56, cw = (W - ML - 8) / pr.bands.length, RH = 26, H = top + 30 + C.length * RH + 6;
    pr.bands.forEach((b, j) => s += `<text x="${ML + (j + 0.5) * cw}" y="${top + 8}" text-anchor="middle" style="fill:var(--ink-2)">s ${b.lo}–${b.hi}</text><text x="${ML + (j + 0.5) * cw}" y="${top + 22}" text-anchor="middle">n ${b.n}</text>`);
    C.forEach((c, i) => {
      const y = top + 30 + i * RH;
      s += `<text x="${ML - 8}" y="${y + RH / 2 + 4}" text-anchor="end" style="fill:var(--ink-2)">${c}</text>`;
      pr.bands.forEach((b, j) => {
        const v = b.r2[i], t = clamp(v, 0, 1);
        s += `<rect x="${ML + j * cw + 1}" y="${y + 1}" width="${cw - 2}" height="${RH - 2}" rx="3" fill="rgba(var(--heat), ${t.toFixed(3)})" stroke="var(--grid)" data-tip="${esc(`${c}, s ${b.lo}–${b.hi}: R² ${fmt(v)}`)}"/>`;
        s += `<text x="${ML + (j + 0.5) * cw}" y="${y + RH / 2 + 4}" text-anchor="middle" style="fill:${t > 0.55 ? "#fff" : "var(--ink-2)"}">${v < 0 ? "< 0" : v.toFixed(2)}</text>`;
      });
    });
    host.innerHTML = svg(H, s);
    return;
  }
  let rows, series, label, log = true;
  if (S.metric === "spread") {
    rows = C; label = "over the target's spread / the median head row";
    series = [["velocity spread", "var(--c0)", pr.spread], ["head row norm", "var(--c1)", pr.head_rows]];
  } else if (S.metric === "change") {
    rows = C; label = "velocity change / target spread"; log = false;
    series = [[S.change + " rad", "var(--c0)", pr.changes[S.change]], ["other data", "var(--ref)", pr.changes["other data"]]];
  } else {
    rows = pr.bands.flatMap(b => ["φ+", "φ-"].map(k => `s ${b.lo}–${b.hi} ${k === "φ+" ? "φ₊" : "φ₋"}`));
    label = "median |angle error| [rad]";
    series = [["head (x₁ the flow implies)", "var(--c0)", pr.bands.flatMap(b => [b.head_error["φ+"], b.head_error["φ-"]])], ["x_t", "var(--ref)", pr.bands.flatMap(b => [b.xt_error["φ+"], b.xt_error["φ-"]])]];
  }
  document.getElementById("probe-legend").innerHTML = series.map(([n, c]) => `<span>${dotKey(c)}${n}</span>`).join("") + (S.metric === "spread" ? `<span>${sw("var(--ref)")}target</span>` : "");
  const vals = series.flatMap(x => x[2]), ML = 120, MR = 16, RH = 24, H = top + rows.length * RH + 40, y1 = top + rows.length * RH;
  let X, tv;
  if (log) { const lo = Math.min(...vals.filter(v => v > 0)) / 1.4, hi = Math.max(...vals, 1) * 1.4, a = Math.log(lo), b = Math.log(hi); X = v => ML + (Math.log(v) - a) / (b - a) * (W - MR - ML); tv = logTicks(lo, hi); }
  else { const hi = Math.max(...vals) * 1.1; X = v => ML + v / hi * (W - MR - ML); tv = linTicks(hi, 6); }
  tv.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${top}" y2="${y1}" stroke="var(--grid)"/>`);
  if (S.metric === "spread") s += `<line x1="${X(1)}" x2="${X(1)}" y1="${top}" y2="${y1}" stroke="var(--ref)" stroke-width="1.5"/>`;
  rows.forEach((name, i) => {
    const y = top + i * RH + RH / 2;
    s += `<text x="${ML - 8}" y="${y + 4}" text-anchor="end" style="fill:var(--ink-2)">${name}</text>`;
    const xs = series.map(x => x[2][i]).filter(v => !log || v > 0);
    if (xs.length > 1) s += `<line x1="${X(Math.min(...xs))}" x2="${X(Math.max(...xs))}" y1="${y}" y2="${y}" stroke="var(--axis)" stroke-width="2"/>`;
    series.forEach(([n, c, v]) => { if (!log || v[i] > 0) s += `<circle cx="${X(v[i])}" cy="${y}" r="5" fill="${c}" stroke="var(--surface)" stroke-width="1.5" data-tip="${esc(`${name}, ${n}: ${fmt(v[i])}`)}"/>`; });
  });
  s += `<line x1="${ML}" x2="${W - MR}" y1="${y1}" y2="${y1}" stroke="var(--axis)"/>`;
  tv.forEach(v => s += `<text x="${X(v)}" y="${y1 + 16}" text-anchor="middle">${log && v >= 1000 ? v / 1000 + "k" : fmt(v)}</text>`);
  s += `<text x="${(ML + W - MR) / 2}" y="${y1 + 31}" text-anchor="middle">${label}</text>`;
  host.innerHTML = svg(H, s);
}
function draw() {
  const R = RUNS.find(r => r.tag === S.run);
  seg("run", RUNS.map(r => [r.tag, r.tag + (r.smoke ? " (smoke)" : "")]), "run");
  seg("t", R.t_grid.map((t, i) => [i, String(t)]), "t", true);
  seg("which", [["data", "data"], ["xt", "x<sub>t</sub>"]], "which");
  seg("slot", [["all", "all"], ...R.shares[0].slot.map((_, s) => [s, String(s)])], "slot");
  seg("coord", [["all", "all"], ...R.coordinates.map((c, i) => [i, c])], "coord");
  document.getElementById("banner").innerHTML = R.smoke
    ? `<div class="banner"><b>Smoke run: ${R.steps.toLocaleString()} of ${R.total_steps.toLocaleString()} steps.</b> A pipeline check, not a finding: these numbers describe a barely trained network.</div>` : "";
  document.getElementById("meta").textContent = `${R.config}, commit ${R.commit.slice(0, 7)}, ${R.weights} weights at epoch ${R.epoch} (${R.steps.toLocaleString()} steps), bf16 ${R.bf16}; ${R.observations} observations (${R.jacobian_observations} with Jacobians) per t on ${R.device} in ${R.seconds} s.`;
  document.getElementById("bp").textContent = Math.round(100 * R.band_power) + "%";
  const ti = Math.min(S.t, R.t_grid.length - 1), tl = R.t_grid.map(String);
  const coords = R.coordinates.map((c, i) => [c, i, COLORS[i]]);

  // shares vs t
  legend("lg-share", coords.map(([c, , col]) => [col, "", c]));
  chart("p-share", {
    label: "share vs t", x: tl, xlabel: "path time t", ylabel: S.which === "data" ? "data share" : "x_t share", y: [0, 1.2], ticks: [0, 0.25, 0.5, 0.75, 1], mark: ti,
    series: coords.map(([c, i, col]) => ({color: col, pts: R.shares.map((sh, k) => {
      const g = S.slot === "all" ? sh.all : sh.slot[S.slot];
      return [k, g[S.which][i], `<b>${c}</b>, t ${sh.t}: ${S.which === "data" ? "data" : "x<sub>t</sub>"} share ${f3(g[S.which][i])} (n ${g.n})`];
    })})),
  });

  // data share by SNR
  const edges = R.snr_edges, bins = edges.slice(0, -1).map((e, i) => `${e}–${edges[i + 1]}`);
  legend("lg-snr", coords.map(([c, , col]) => [col, "", c]));
  chart("p-snr", {
    label: "data share by SNR", x: bins, xlabel: `source SNR (t = ${R.t_grid[ti]})`, ylabel: "data share", y: [0, 1.2], ticks: [0, 0.25, 0.5, 0.75, 1],
    series: coords.map(([c, i, col]) => ({color: col, pts: R.shares[ti].snr.map((g, k) => [k, g.n >= 4 ? g.data[i] : null, `<b>${c}</b>, SNR ${bins[k]}: data share ${f3(g.data[i])} (n ${g.n})`])})),
  });

  // own / other / rest sensitivity vs t
  const pick = split => { const rows = []; split.forEach(slot => slot.forEach((v, c) => { if (S.coord === "all" || String(S.coord) === String(c)) rows.push(v); })); return [0, 1, 2].map(j => mean(rows.map(r => r[j]))); };
  const base = [0, 1, 2].map(j => mean(R.baseline.map(b => b[j])));
  const parts = [["own band", "var(--c0)"], ["other sources' bands", "var(--c1)"], ["rest", "var(--c2)"]];
  legend("lg-band", [...parts.map(([n, c]) => [c, "", n]), ["var(--ref)", "4 3", "uniform baseline (colour per set)"]]);
  chart("p-band", {
    label: "sensitivity split vs t", x: tl, xlabel: "path time t", ylabel: "share of squared Jacobian", y: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], mark: ti,
    series: [
      ...parts.map(([n, col], j) => ({color: col, dash: "4 3", pts: tl.map((_, k) => [k, base[j], `${n}: uniform ${f3(base[j])}`])})),
      ...parts.map(([n, col], j) => ({color: col, pts: R.jacobian.map((J, k) => { const v = pick(J.split)[j]; return [k, v, `<b>${n}</b>, t ${J.t}: ${f3(v)} (uniform ${f3(base[j])}, ×${(v / base[j]).toFixed(1)})`]; })})),
    ],
  });

  // enrichment by ring
  const rings = R.rings.slice(0, -1).map((r, i) => R.rings[i + 1] - r === 1 ? String(r) : `${r}–${R.rings[i + 1] - 1}`);
  const en = R.jacobian[ti].enrichment, flat = en.flat().filter(v => v > 0);
  const top = 10 ** Math.ceil(Math.log10(Math.max(10, ...flat))), bot = 10 ** Math.floor(Math.log10(Math.min(0.1, ...flat)));
  const ticks = []; for (let v = bot; v <= top * 1.01; v *= 10) ticks.push(+v.toPrecision(1));
  legend("lg-ring", [...coords.map(([c, , col]) => [col, "", c]), ["var(--ref)", "4 3", "uniform"]]);
  chart("p-ring", {
    label: "enrichment by distance from f0", x: rings, xlabel: `bins from the slot's own f0 (t = ${R.t_grid[ti]})`, ylabel: "density / uniform", y: [bot, top], log: true, ticks,
    series: [{color: "var(--ref)", dash: "4 3", pts: rings.map((_, k) => [k, 1, ""])},
      ...coords.map(([c, i, col]) => ({color: col, pts: en[i].map((v, k) => [k, v, `<b>${c}</b>, ${rings[k]} bins from f0: ×${v.toFixed(2)} uniform`])}))],
  });

  // one observation's map
  const ex = R.jacobian[ti].example, E = R.example, host = document.getElementById("p-example");
  const W = Math.max(300, host.clientWidth), cols = ex[0].length, rowH = 22, band = 5, lft = 64, H = ex.length * (rowH + band + 4) + 30;
  const cw = (W - lft - 8) / cols, peak = Math.max(...ex.flat());
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="sensitivity map">`;
  ex.forEach((row, si) => {
    const y0 = si * (rowH + band + 4);
    s += `<text x="${lft - 6}" y="${y0 + band + rowH / 2 + 8}" text-anchor="end">slot ${si}</text>`;
    E.own[si].forEach((o, c) => { if (o) s += `<rect x="${lft + c * cw}" y="${y0}" width="${Math.max(cw, 1)}" height="${band - 1}" fill="var(--ink-2)"/>`; });
    const fx = lft + (E.f0_bin[si] / E.stride) * cw;
    s += `<path d="M${fx - 4} ${y0 - 4}L${fx + 4} ${y0 - 4}L${fx} ${y0 + 2}Z" fill="var(--ink)"/>`;
    row.forEach((v, c) => s += `<rect x="${lft + c * cw}" y="${y0 + band + 1}" width="${cw + 0.3}" height="${rowH}" fill="rgba(var(--heat), ${(Math.sqrt(v / peak)).toFixed(3)})" data-tip="slot ${si} (SNR ${E.snr[si].toFixed(0)}), bins ${c * E.stride}–${(c + 1) * E.stride - 1}: ${(100 * v).toFixed(2)}%"/>`);
  });
  s += `<text x="${lft}" y="${H - 6}">0</text><text x="${W - 8}" y="${H - 6}" text-anchor="end">${R.window_bins} window bins</text></svg>`;
  host.innerHTML = s;
  drawProbe();
}
const ok = RUNS.length > 0;
const hash = new URLSearchParams(location.hash.slice(1));
S.probe = PROBES.some(p => p.tag === hash.get("probe")) ? hash.get("probe") : (PROBES.filter(p => !p.smoke)[0] || PROBES[PROBES.length - 1] || {}).tag;
if (!ok) { document.querySelectorAll(".card:not(.means):not(#probe-card)").forEach(c => c.innerHTML = `<p class="empty">No run measured yet.</p>`); drawProbe(); window.addEventListener("resize", drawProbe); }
else {
  const trained = RUNS.filter(r => !r.smoke).sort((a, b) => b.steps - a.steps);
  const h = new URLSearchParams(location.hash.slice(1));
  S.run = RUNS.some(r => r.tag === h.get("run")) ? h.get("run") : (trained[0] || RUNS[RUNS.length - 1]).tag;
  draw();
  window.addEventListener("resize", draw);
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
