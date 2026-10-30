"""Write proper-scores.html from data/*.json (scores_probe.py): proper scores as skill against the fisher mock, per run and mock ladder.

Stdlib only; run from the repo root:
    uv run --frozen python reports/posterior-eval/proper-scores/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proper scores</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --lo: #eef4fc; --hi: #0b4a9e; --neg: #c2362b; --warn: #b8741a; --warn-bg: #fbf0dc; --warn-ink: #5d3b06;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --lo: #1c2330; --hi: #6aa8f5; --neg: #e0675c; --warn: #e0a447; --warn-bg: #2e2412; --warn-ink: #f0d9a8;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
a { color: var(--hi); }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 12px; max-width: 76ch; }
.nav { font-size: 13px; color: var(--ink-2); margin: 0 0 16px; display: flex; flex-wrap: wrap; gap: 4px 14px; }
.nav b { color: var(--ink); font-weight: 600; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 100ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.banner { background: var(--warn-bg); color: var(--warn-ink); border-radius: 8px; padding: 8px 12px; font-size: 13px; margin: 0 0 12px; max-width: 100ch; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 16px; }
.bar .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 0 0 14px; }
.tile { border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; }
.tile .t { color: var(--ink-2); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 600; }
.tile .d { color: var(--muted); font-size: 12px; }
.scroll { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 2px; font-size: 12px; font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; padding: 2px 6px; vertical-align: bottom; }
th.l { text-align: left; white-space: nowrap; font-weight: 500; }
th.g { text-align: left; color: var(--ink); padding-top: 12px; }
td { min-width: 64px; height: 24px; text-align: center; border-radius: 4px; color: var(--ink); padding: 0 4px; white-space: nowrap; }
td.dash { color: var(--muted); }
td.best { outline: 2px solid var(--ink); outline-offset: -2px; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; white-space: nowrap; }
.tip b { color: var(--ink); font-weight: 600; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 82ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>Does one run's posterior score better than another's?</h1>
<p class="sub">The energy score and per-coordinate CRPS of <code>canna.lisa.validate</code>, rescaled so that the fisher mock (a calibrated linear-Gaussian posterior) reads 1 and a data-blind posterior reads 0. The width ratio is the absolute yardstick beside them.</p>
<div class="nav"><b>Posterior evaluation</b>
<a href="../test-power/test-power.html">test-power</a> <a href="../sampler-convergence/sampler-convergence.html">sampler-convergence</a> <a href="../robustness/robustness.html">robustness</a> <a href="../mock-flows/mock-flows.html">mock-flows</a> <a href="../reference-comparison/reference-comparison.html">reference-comparison</a> <a href="../fisher-scorecard/fisher-scorecard.html">fisher-scorecard</a> <a href="../calibration/calibration.html">calibration</a> <a href="../data-space-checks/data-space-checks.html">data-space-checks</a> <b>proper-scores</b></div>
<div id="banners"></div>
<div class="bar">
  <div><span class="k">Setting</span><span class="seg" id="setting"></span></div>
  <div><span class="k">Statistic</span><span class="seg" id="stat"></span></div>
</div>

<div class="card">
  <h2>How do the runs compare?</h2>
  <p class="note" id="cmp-note"></p>
  <div class="tiles" id="tiles"></div>
  <div class="bar"><div><span class="k">SNR stratum</span><span class="seg" id="stratum"></span></div></div>
  <div class="scroll" id="cmp"></div>
  <div class="tip"></div>
</div>

<div class="card">
  <h2>Scores by SNR stratum and coordinate</h2>
  <p class="note" id="strata-note"></p>
  <div class="bar" id="flowbar"><div><span class="k">Flow</span><span class="seg" id="flow"></span></div></div>
  <div class="scroll" id="strata"></div>
  <div class="tip"></div>
</div>

<div class="card">
  <h2>Which defects does each score rank correctly?</h2>
  <p class="note" id="ladder-note"></p>
  <div class="scroll" id="ladder"></div>
  <div class="tip"></div>
</div>

<div class="card">
  <h2>What the reference is</h2>
  <div class="note" id="match"></div>
</div>

<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const SCORES = ["energy", "joint_energy", "crps_f0", "crps_fdot", "crps_lon", "crps_lat", "crps_re_zp", "crps_im_zp", "crps_re_zm", "crps_im_zm", "width_ratio"];
const NAME = {energy: "energy score (per slot)", joint_energy: "energy score (joint)", width_ratio: "f0 width / Fisher bound"};
const nm = s => NAME[s] || (s.startsWith("crps_") ? "CRPS " + s.slice(5) : s);
const GROUPS = [["Energy score", ["energy", "joint_energy"]], ["CRPS per coordinate", SCORES.slice(2, 10)], ["Yardstick, not a proper score", ["width_ratio"]]];
const STRATA = ["all", "0-15", "15-40", "40-100", ">=100"];
const runs = DATA.filter(r => r.kind === "run"), key = r => r.setting;
const S = {setting: key(runs[0] || DATA[0]), stat: "mean", stratum: "all", flow: null};
const rec = () => DATA.find(r => key(r) === S.setting);
const label = r => r.kind === "run" ? `${r.run}, epoch ${r.epoch}, ${r.ode_steps} steps` : `mock ladder, ${r.band} band`;
const fmt = (v, d = 2) => v == null ? "–" : v.toFixed(d);
const skillFill = v => v == null ? "transparent" : v >= 0
  ? `color-mix(in oklab, var(--hi) ${Math.round(100 * Math.min(1, v))}%, var(--lo))`
  : `color-mix(in oklab, var(--neg) ${Math.round(100 * Math.min(1, -v / 2))}%, var(--lo))`;
const ratioFill = r => r == null ? "transparent" : r >= 1
  ? `color-mix(in oklab, var(--warn) ${Math.round(100 * Math.min(1, Math.log2(r) / 3))}%, var(--lo))`
  : `color-mix(in oklab, var(--hi) ${Math.round(100 * Math.min(1, -Math.log2(r) / 2))}%, var(--lo))`;
const ink = pct => pct > 55 ? "var(--surface)" : "var(--ink)";

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (DATA.some(r => key(r) === h.get("setting"))) S.setting = h.get("setting");
  if (["mean", "median"].includes(h.get("stat"))) S.stat = h.get("stat");
  if (STRATA.includes(h.get("stratum"))) S.stratum = h.get("stratum");
}
function writeHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  h.set("setting", S.setting); h.set("stat", S.stat); h.set("stratum", S.stratum);
  history.replaceState(null, "", "#" + h.toString());
}
const seg = (id, items, cur, attr) => $(id).innerHTML = items.map(([k, t]) => `<button aria-pressed="${k === cur}" data-${attr}="${k}">${t}</button>`).join("");
function on(id, attr, fn) { $(id).addEventListener("click", e => { const b = e.target.closest(`button[data-${attr}]`); if (b) { fn(b.dataset[attr]); draw(); } }); }
on("setting", "setting", v => { S.setting = v; S.flow = null; });
on("stat", "stat", v => S.stat = v);
on("stratum", "stratum", v => S.stratum = v);
on("flow", "flow", v => S.flow = v);

const cellOf = (r, flow, score, stratum) => ((r.flows[flow] || {})[score] || {})[stratum];
const skillOf = c => c ? c["skill_" + S.stat] : null;
const raw = c => c ? c[S.stat] : null;
const mockOf = band => DATA.find(r => r.kind === "mocks" && r.band === band);

function drawBanners() {
  const r = rec(), out = [];
  if (DATA.some(d => d.smoke)) out.push(`<b>Smoke run.</b> The run numbers come from <code>${runs.filter(d => d.smoke)[0].run}</code>, a ${runs.filter(d => d.smoke)[0].total_steps} step narrow-XS pipeline test stopped at epoch ${runs.filter(d => d.smoke)[0].epoch}. They show the scores working end to end, not how good the method is; real findings need narrow-B and full-B.`);
  if (r.kind === "run" && !r.match.matched) out.push(`<b>No injection match.</b> This setting's injections differ from the mock's, so skill is relative to an in-setting data-blind floor only (no fisher reference).`);
  $("banners").innerHTML = out.map(t => `<div class="banner">${t}</div>`).join("");
}

function drawBar() {
  seg("setting", DATA.map(r => [key(r), label(r)]), S.setting, "setting");
  seg("stat", [["mean", "mean"], ["median", "median"]], S.stat, "stat");
  seg("stratum", STRATA.map(s => [s, s === "all" ? "all" : "ŝ " + s]), S.stratum, "stratum");
}

function drawCompare() {
  const band = rec().band, cols = runs.filter(r => r.band === band);
  $("cmp-note").innerHTML = `Skill = (floor − score) / (floor − fisher): 1 is the fisher mock's score, 0 the data-blind floor (the prior mock), below 0 worse than ignoring the data. Cells show the ${S.stat} of ${S.stratum === "all" ? "all injections" : "the draws' median SNR ŝ in " + S.stratum}; <i>Δ</i> is the paired change from the first column's setting with its bootstrap 95% interval over injections (negative is better). Skill is undefined (–) where fisher and the floor barely differ, as at low SNR where the data barely constrain the posterior.`;
  if (!cols.length) { $("tiles").innerHTML = ""; $("cmp").innerHTML = `<p class="note">No run in the ${band} band yet: run the sbatch in the README with its eval dirs.</p>`; return; }
  $("tiles").innerHTML = cols.map(r => {
    const c = cellOf(r, "run", "joint_energy", "all"), p = r.paired && r.paired.scores.joint_energy.all;
    return `<div class="tile"><div class="t">${r.run} · ${r.ode_steps} steps</div><div class="v">${fmt(skillOf(c))}</div><div class="d">joint energy skill; score ${fmt(raw(c), 3)}${p ? `<br>Δ ${(100 * p.rel).toFixed(1)}% vs ${p.base || r.paired.baseline}` : ""}</div></div>`;
  }).join("");
  let s = `<table><tr><th></th>${cols.map((r, i) => `<th colspan="${i ? 2 : 1}">${r.ode_steps} steps</th>`).join("")}</tr>`;
  GROUPS.forEach(([g, names]) => {
    s += `<tr><th class="g" colspan="${cols.length * 2}">${g}</th></tr>`;
    names.forEach(name => {
      s += `<tr><th class="l">${nm(name)}</th>`;
      cols.forEach((r, i) => {
        const c = cellOf(r, "run", name, S.stratum);
        if (name === "width_ratio") {
          s += `<td style="background:${ratioFill(c && raw(c))}" data-tip="<b>${nm(name)}</b><br>${r.ode_steps} steps, ${S.stratum}<br>${S.stat} <b>${fmt(c && raw(c), 2)}</b> (fisher mock ${fmt(c && c.fisher_median, 2)})<br>1 is the monochromatic Fisher bound">${fmt(c && raw(c), 2)}</td>`;
        } else {
          const v = skillOf(c);
          s += `<td style="background:${skillFill(v)};color:${ink(v == null ? 0 : 100 * Math.abs(v))}" data-tip="<b>${nm(name)}</b><br>${r.ode_steps} steps, ${S.stratum}, n=${c ? c.n : 0}<br>score ${fmt(raw(c), 3)}, fisher ${fmt(c && c["fisher_" + S.stat], 3)}, floor ${fmt(c && c["floor_" + S.stat], 3)}">${fmt(v)}</td>`;
        }
        if (i) {
          const p = r.paired && r.paired.scores[name] && r.paired.scores[name][S.stratum];
          const rel = p && p.rel != null ? p.rel : null, sig = p && p.lo != null && (p.lo > 0 || p.hi < 0);
          s += p ? `<td class="${sig ? "" : "dash"}" data-tip="<b>${nm(name)}</b><br>paired mean change ${fmt(p.diff, 4)}<br>95% [${fmt(p.lo, 4)}, ${fmt(p.hi, 4)}]${sig ? "" : "<br>interval spans 0"}">${sig ? "<b>" : ""}Δ ${rel == null ? "–" : (100 * rel).toFixed(1) + "%"}${sig ? "</b>" : ""}</td>` : `<td class="dash">–</td>`;
        }
      });
      s += `</tr>`;
    });
  });
  $("cmp").innerHTML = s + `</table>`;
}

function drawStrata() {
  const r = rec(), flows = Object.keys(r.flows);
  if (!flows.includes(S.flow)) S.flow = flows[0];
  $("flowbar").style.display = flows.length > 1 ? "" : "none";
  seg("flow", flows.map(f => [f, f]), S.flow, "flow");
  $("strata-note").innerHTML = `${label(r)}: ${r.injections} injections × ${r.draws} draws. Per-slot scores pool the four slots, each stratified by its own ŝ (the draws' median SNR, a function of the data alone); joint rows use the joint ŝ. Colour is skill (${S.stat}); hover for the score, the fisher mock's score and the floor on the same elements.`;
  let s = `<table><tr><th></th>${STRATA.map(t => `<th>${t === "all" ? "all" : "ŝ " + t}</th>`).join("")}</tr>`;
  GROUPS.forEach(([g, names]) => {
    s += `<tr><th class="g" colspan="${STRATA.length + 1}">${g}</th></tr>`;
    names.forEach(name => {
      s += `<tr><th class="l">${nm(name)}</th>`;
      STRATA.forEach(st => {
        const c = cellOf(r, S.flow, name, st);
        if (!c || !c.n) { s += `<td class="dash">–</td>`; return; }
        if (name === "width_ratio") {
          s += `<td style="background:${ratioFill(raw(c))}" data-tip="<b>${nm(name)}</b><br>${st}, n=${c.n}<br>${S.stat} <b>${fmt(raw(c), 2)}</b>, fisher mock ${fmt(c.fisher_median, 2)}">${fmt(raw(c), 2)}</td>`;
        } else {
          const v = skillOf(c);
          s += `<td style="background:${skillFill(v)};color:${ink(v == null ? 0 : 100 * Math.abs(v))}" data-tip="<b>${nm(name)}</b><br>${st}, n=${c.n}<br>score <b>${fmt(raw(c), 3)}</b>, fisher ${fmt(c["fisher_" + S.stat], 3)}, floor ${fmt(c["floor_" + S.stat], 3)}">${fmt(v)}</td>`;
        }
      });
      s += `</tr>`;
    });
  });
  $("strata").innerHTML = s + `</table>`;
}

const MOCK_NOTE = {fisher: "the reference", wide: "widths ×3", narrow: "widths ×0.5", biased: "f0 +1σ", prior: "identity flow", "phases-prior": "z± phases from prior", "loud-floor": "loud widths floored"};
function drawLadder() {
  const m = mockOf(rec().band);
  if (!m) { $("ladder").innerHTML = `<p class="note">No mock ladder for the ${rec().band} band yet.</p>`; $("ladder-note").textContent = ""; return; }
  $("ladder-note").innerHTML = `Each score's ratio to the fisher mock's, on its own ${m.injections} validation injections (${m.band} band), paired by injection; width ratio is the geometric mean of the per-source width ratios. A proper score should read above 1 for every defect: a <b>●</b> marks a 95% bootstrap interval that excludes 1, an outlined cell is the lowest in its column. The mocks are fixed checks that each score sees what it should; they say nothing about a run.`;
  const rows = Object.keys(m.ratios);
  let s = `<table><tr><th></th>${SCORES.map(c => `<th>${nm(c).replace("energy score ", "E ").replace("f0 width / Fisher bound", "width")}</th>`).join("")}<th>flagged</th></tr>`;
  const best = SCORES.map(c => rows.reduce((a, b) => m.ratios[b][c].ratio < m.ratios[a][c].ratio ? b : a));
  rows.forEach(name => {
    let flagged = 0;
    s += `<tr><th class="l">${name}<br><span style="font-weight:400;color:var(--muted)">${MOCK_NOTE[name] || ""}</span></th>`;
    SCORES.forEach((c, j) => {
      const q = m.ratios[name][c], sig = q.lo > 1 || q.hi < 1;
      if (name !== "fisher" && sig) flagged++;
      s += `<td class="${best[j] === name && c !== "width_ratio" ? "best" : ""}" style="background:${ratioFill(q.ratio)};color:${ink(Math.abs(Math.log2(q.ratio)) > 1.6 ? 100 : 0)}" data-tip="<b>${name}</b>, ${nm(c)}<br>ratio to fisher <b>${fmt(q.ratio, 3)}</b><br>95% [${fmt(q.lo, 3)}, ${fmt(q.hi, 3)}]">${fmt(q.ratio, q.ratio >= 10 ? 1 : 2)}${name !== "fisher" && sig ? " ●" : ""}</td>`;
    });
    s += `<td>${name === "fisher" ? "–" : flagged + " / " + SCORES.length}</td></tr>`;
  });
  $("ladder").innerHTML = s + `</table>`;
}

function drawMatch() {
  const r = rec();
  if (r.kind === "mocks") { $("match").innerHTML = `Each mock's own scores against the same band's fisher and prior mocks, on the same ${r.injections} validation injections (same keys by construction), so skill is exact: fisher is 1 and prior is 0.`; return; }
  const m = r.match, ck = Object.entries(m.checks).map(([k, v]) => `<code>${k}</code> ${v ? "✓" : "✗"}`).join(" · ");
  $("match").innerHTML = `Reference: ${m.reference}. The mock has ${m.mock_injections} validation injections and this setting ${r.injections}; ${m.n_used} were compared (the first ${m.n_used}). Checks on those: ${ck}. Skill cells use only those ${m.n_used} injections; the raw score over all ${r.injections} injections is in the data as <code>mean_all</code>.`;
}

function draw() { drawBanners(); drawBar(); drawCompare(); drawStrata(); drawLadder(); drawMatch(); writeHash(); }

document.querySelectorAll(".card").forEach(card => {
  const tip = card.querySelector(".tip");
  if (!tip) return;
  card.addEventListener("pointermove", ev => {
    const el = ev.target.closest("[data-tip]");
    if (!el) { tip.style.display = "none"; return; }
    const box = card.getBoundingClientRect();
    tip.innerHTML = el.dataset.tip; tip.style.display = "block";
    tip.style.left = Math.max(8, Math.min(ev.clientX - box.left + 14, box.width - tip.offsetWidth - 8)) + "px";
    tip.style.top = (ev.clientY - box.top + 14) + "px";
  });
  card.addEventListener("pointerleave", () => tip.style.display = "none");
});

readHash();
draw();
</script>
</body>
</html>
"""

MEANS = """
<ul>
<li><b>No run yet.</b> The page holds the mock ladder only (src at 8310cb3, 2026-10-30). Run rows drop in once a run has an eval of the current layout (<code>crps_fdot</code> in place of <code>crps_mc</code>); the 2026-10-24 <code>dusty-otter-narrow-XS</code> files stay in <code>data/</code> but are not shown.</li>
<li><b>Every eval source is faint</b> (the mocks ran with <code>visible_snr = 0</code>, catalogue windows; ŝ at most 2.6), so the prior mock (the skill floor) scores about as well as <code>fisher</code>, and skill is undefined (floor and fisher differ by under 10%). The ladder is read as ratios to <code>fisher</code>; rerun with <code>visible_snr</code> on.</li>
<li><b>The energy score ranks the ladder by magnitude.</b> Narrow band, ratio to <code>fisher</code>: wide 1.53, narrow 1.03, biased 1.04, phases-prior 1.00, loud-floor 1.00 (equal to <code>fisher</code>: no source reaches its cap). The <code>prior</code> mock scores below <code>fisher</code> (0.96; z± CRPS 0.43–0.47, fdot 0.89), because at faint SNR the Fisher mock is a worse posterior than the prior; only its f0 CRPS (1.11) is higher.</li>
<li><b>biased</b> is seen by the f0 CRPS only (1.32 narrow, 1.20 full); the energy score barely moves (1.04, 1.02). <b>phases-prior</b> is invisible to every score on the narrow band (z± CRPS 0.97–1.01).</li>
<li><b>The width ratio</b> separates over-wide from over-narrow (narrow-mock median 5.3 against fisher 15.0; wide 79) but its geometric mean collapses to 0 for <code>wide</code>, since 57 of 800 slots have a zero f0 width; read its median.</li>
<li><b>Caveat.</b> The scores are in unweighted flow coordinates, so the energy score is dominated by the large-scale coordinates (sky, amplitudes); read it with the per-coordinate CRPS.</li>
</ul>
"""


def main() -> None:
    records = [json.loads(p.read_text()) for p in sorted((REPORT / "data").glob("*.json"))]
    # records from before the (f0, fdot, A) chart have crps_mc: no run eval of the current layout yet
    records = [r for r in records if all("crps_fdot" in flow for flow in r["flows"].values())]
    records.sort(key=lambda r: (r["kind"] != "run", r.get("run", ""), r.get("ode_steps", 0), r["band"]))
    page = PAGE.replace("__MEANS__", MEANS).replace("__DATA__", json.dumps(records, separators=(",", ":")))
    out = REPORT / "proper-scores.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
