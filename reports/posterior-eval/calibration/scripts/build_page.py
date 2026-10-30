"""Write calibration.html from data/<setting>.json (calibration_probe.py): SBC ranks, TARP and joint pre-ranks per setting and SNR stratum.

Stdlib only; run from the repo root:
    uv run --frozen python reports/posterior-eval/calibration/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]
MOCK_ORDER = ["fisher", "wide", "narrow", "biased", "prior", "phases-prior", "loud-floor"]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Posterior calibration</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10); --axis: #898781;
  --s1: #1f5fbf; --band: #dcdad0; --bad: #c2362b; --warn-bg: #fbf0d6; --warn-ink: #6b4a05; --lo: #eef4fc;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --axis: #898781;
    --s1: #6aa8f5; --band: #343431; --bad: #e0675c; --warn-bg: #3a2f10; --warn-ink: #f0d58a; --lo: #1c2330;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding-block: 24px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
a { color: var(--s1); }
h1 { font-size: 24px; margin: 8px 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
nav { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; color: var(--ink-2); }
nav b { color: var(--ink); }
.sub { color: var(--ink-2); margin: 0 0 16px; max-width: 76ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 95ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 16px; }
.k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
select { font: inherit; font-size: 13px; padding: 5px 8px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--ink); max-width: 100%; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.banner { background: var(--warn-bg); color: var(--warn-ink); border-radius: 8px; padding: 8px 12px; font-size: 13px; margin: 0 0 12px; max-width: 100ch; }
.scroll { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 2px; font-size: 12px; font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; padding: 2px 6px; text-align: center; vertical-align: bottom; }
th.l { text-align: left; white-space: nowrap; font-weight: 500; }
th.g { text-align: left; color: var(--ink); padding-top: 12px; }
tr.pick th.l { cursor: pointer; }
tr.pick.on th.l { color: var(--ink); text-decoration: underline; }
td { min-width: 52px; height: 24px; text-align: center; border-radius: 4px; color: var(--ink); padding: 0 4px; }
td.na { color: var(--muted); }
td.rej { font-weight: 700; }
.panels { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
.panels svg { width: 100%; height: auto; display: block; }
svg text { fill: var(--ink-2); font-size: 10px; }
svg .t { fill: var(--ink); font-size: 11px; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; white-space: nowrap; }
.tip b { color: var(--ink); }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 85ch; }
.means li { margin: 5px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<nav>
  <a href="../test-power/test-power.html">test-power</a>
  <a href="../sampler-convergence/sampler-convergence.html">sampler-convergence</a>
  <a href="../robustness/robustness.html">robustness</a>
  <a href="../mock-flows/mock-flows.html">mock-flows</a>
  <a href="../reference-comparison/reference-comparison.html">reference-comparison</a>
  <a href="../fisher-scorecard/fisher-scorecard.html">fisher-scorecard</a>
  <b>calibration</b>
  <a href="../data-space-checks/data-space-checks.html">data-space-checks</a>
  <a href="../proper-scores/proper-scores.html">proper-scores</a>
</nav>
<h1>Is the posterior calibrated?</h1>
<p class="sub">SBC ranks of the truth among the flow's draws, TARP and the joint pre-ranks of <code>canna.lisa.validate</code>, scored by its ECDF γ test, per SNR stratum. Clean marginal SBC proves nothing alone: the <code>prior</code> mock, an identity flow, passes it.</p>
<div class="bar">
  <div><span class="k">Setting</span><select id="setting"></select></div>
  <div><span class="k">Draws' median SNR</span><span class="seg" id="stratum"></span></div>
</div>
<div id="banners"></div>
<div class="card">
  <h2>Which tests reject?</h2>
  <p class="note">γ-test p per test and slot (slot 1..4 as ordered by the eval, all slots pooled, and the joint rank over the window where there is one). Red: p below α = <span class="alpha"></span>, darker for smaller p; the cell shows <code>&lt;.001</code> at the test's simulated floor. Hover for the sample size, the mean rank offset (&gt; 0: the truth above the draws) and the share of ranks in the middle half (&gt; ½ too wide, &lt; ½ too narrow; for band depth a high rank is a central truth, so too wide). Click a row to draw its ECDF below.</p>
  <div class="scroll" id="cells"></div>
  <div class="tip"></div>
</div>
<div class="card">
  <h2>What does the failing test look like?</h2>
  <p class="note" id="panel-note"></p>
  <div class="panels" id="panels"></div>
  <div class="tip"></div>
</div>
<div class="card">
  <h2>Mock ladder: which tests see which defect?</h2>
  <p class="note">γ-test p of the selected stratum, all slots pooled (joint rows over the window), for each mock flow of the band and each run. Red below α. A mock's defect is fixed, so a test that misses it is blind to it; the findings are the run columns. Hover a column head for the mock's defect.</p>
  <div class="scroll" id="ladder"></div>
  <div class="tip"></div>
</div>
<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const DATA = __DATA__;
const MOCK_ORDER = __MOCK_ORDER__;
const MOCK_DESC = {
  "fisher": "linear-Gaussian posterior (Fisher + prior) at the data's MAP: calibrated only where the posterior is near-Gaussian (not at faint SNR)",
  "wide": "every width ×3",
  "narrow": "every width ×0.5",
  "biased": "f0 shifted +1σ",
  "prior": "identity flow: the prior, ignoring the data. Passes marginal SBC",
  "phases-prior": "z± phases redrawn from the prior",
  "loud-floor": "widths floored at the loud sources' f0 floor (no source reaches the floor at faint SNR)"
};
const LABEL = {f0: "f0", fdot: "fdot", lon: "lon", lat: "lat", re_zp: "Re z+", im_zp: "Im z+", re_zm: "Re z−", im_zm: "Im z−",
  f0_plus_fdot: "f0 + fdot", f0_minus_fdot: "f0 − fdot", abs_zp: "|z+|", abs_zm: "|z−|", tarp: "TARP", average_rank: "average rank", band_depth: "band depth"};
const GROUPS = [["Coordinate ranks", ["f0", "fdot", "lon", "lat", "re_zp", "im_zp", "re_zm", "im_zm"]],
  ["Derived ranks", ["f0_plus_fdot", "f0_minus_fdot", "abs_zp", "abs_zm"]],
  ["TARP", ["tarp"]], ["Joint pre-ranks", ["average_rank", "band_depth"]]];
const key = r => r.setting;
const label = r => r.kind === "run" ? `${r.run}, epoch ${r.epoch}, ${r.ode_steps} ODE steps` : `${r.band} band · mock ${r.mock}`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const S = {setting: DATA[0].setting, stratum: "all", test: "f0"};
const cur = () => DATA.find(r => key(r) === S.setting);

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (DATA.some(r => key(r) === h.get("setting"))) S.setting = h.get("setting");
  if (cur().strata.includes(h.get("stratum"))) S.stratum = h.get("stratum");
  if (cur().tests.includes(h.get("test"))) S.test = h.get("test");
}
function writeHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  h.set("setting", S.setting); h.set("stratum", S.stratum); h.set("test", S.test);
  history.replaceState(null, "", "#" + h.toString());
}

const pText = p => p <= 0.001 ? "&lt;.001" : p.toFixed(p < 0.1 ? 3 : 2);
const fill = p => p < cur().alpha ? `color-mix(in oklab, var(--bad) ${Math.round(100 * Math.min(0.6, 0.15 + -Math.log10(Math.max(p, 1e-3)) / 6))}%, var(--lo))` : "var(--lo)";
function td(c, what) {
  if (!c) return `<td class="na">–</td>`;
  const rej = c.p < cur().alpha;
  return `<td class="${rej ? "rej" : ""}" style="background:${fill(c.p)}" data-tip="<b>${esc(what)}</b><br>${c.n} ranks, p = ${c.p}<br>mean rank offset ${c.bias >= 0 ? "+" : ""}${c.bias.toFixed(3)}<br>middle-half share ${c.middle.toFixed(2)}">${pText(c.p)}</td>`;
}

function drawBar() {
  const sel = document.getElementById("setting");
  const group = (name, rows) => rows.length ? `<optgroup label="${name}">${rows.map(r => `<option value="${key(r)}">${esc(label(r))}</option>`).join("")}</optgroup>` : "";
  sel.innerHTML = group("Runs", DATA.filter(r => r.kind === "run")) + ["narrow", "full"].map(b =>
    group(`Mock flows, ${b} band`, DATA.filter(r => r.kind === "mock" && r.band === b).sort((a, c) => MOCK_ORDER.indexOf(a.mock) - MOCK_ORDER.indexOf(c.mock)))).join("");
  sel.value = S.setting;
  document.getElementById("stratum").innerHTML = cur().strata.map(s => `<button aria-pressed="${s === S.stratum}" data-s="${s}">${s === "all" ? "all" : "SNR " + s}</button>`).join("");
  document.querySelectorAll(".alpha").forEach(e => e.textContent = cur().alpha);
}
document.getElementById("setting").onchange = e => { S.setting = e.target.value; if (!cur().strata.includes(S.stratum)) S.stratum = "all"; draw(); };
document.getElementById("stratum").onclick = e => { const b = e.target.closest("button"); if (b) { S.stratum = b.dataset.s; draw(); } };

function drawBanners() {
  const r = cur(), out = [];
  if (r.kind === "run" && r.size === "XS") out.push(`<code>${esc(r.run)}</code> is a smoke run (narrow-XS, 2000 steps, epoch ${r.epoch}): none of its numbers is a finding. Real findings need narrow-B and full-B.`);
  if (r.tarp_ref_is_truth) out.push(`This eval's <code>tarp_ref</code> equals the truth (the fold_in/split key collision, fixed in 3d2413c): its TARP rows are invalid and reject by construction.`);
  if (r.kind === "mock") out.push(`Mock flow <code>${esc(r.mock)}</code>: ${esc(MOCK_DESC[r.mock] || "")}. A fixed check that each test sees what it should, not a finding.`);
  document.getElementById("banners").innerHTML = out.map(t => `<div class="banner">${t}</div>`).join("");
}

function drawCells() {
  const r = cur(), c = r.cells[S.stratum], n = r.counts[S.stratum];
  const nslots = n.sources.length;
  let s = `<table><tr><th></th>${n.sources.map((m, i) => `<th>slot ${i + 1}<br><span style="font-weight:400">${m}</span></th>`).join("")}<th>pooled<br><span style="font-weight:400">${n.sources.reduce((a, b) => a + b, 0)}</span></th><th>joint<br><span style="font-weight:400">${n.injections}</span></th></tr>`;
  for (const [g, tests] of GROUPS) {
    s += `<tr><th class="g" colspan="${nslots + 3}">${g}</th></tr>`;
    for (const t of tests) {
      s += `<tr class="pick ${t === S.test ? "on" : ""}" data-test="${t}"><th class="l">${LABEL[t]}</th>`;
      c[t].slot.forEach((x, i) => s += td(x, `${LABEL[t]}, slot ${i + 1}`));
      s += td(c[t].pooled, `${LABEL[t]}, pooled`) + td(c[t].joint, `${LABEL[t]}, joint`) + `</tr>`;
    }
  }
  const el = document.getElementById("cells");
  el.innerHTML = s + `</table>`;
  el.onclick = e => { const row = e.target.closest("tr[data-test]"); if (row) { S.test = row.dataset.test; draw(); } };
}

function panel(c, band, title) {
  const W = 230, H = 150, M = {l: 34, r: 8, t: 22, b: 22};
  const [i0, step] = cur().stride, k = cur().k, z = c.ecdf.map((_, i) => (i0 + i * step) / k);
  const lim = Math.max(0.05, ...c.ecdf.map(Math.abs), ...band[0].map(Math.abs), ...band[1].map(Math.abs)) * 1.08;
  const X = v => M.l + v * (W - M.l - M.r), Y = v => M.t + (1 - (v + lim) / (2 * lim)) * (H - M.t - M.b);
  const poly = (ys) => ys.map((y, i) => `${X(z[i]).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
  const bandPts = poly(band[1]) + " " + band[0].map((y, i) => `${X(z[z.length - 1 - i]).toFixed(1)},${Y(band[0][band[0].length - 1 - i]).toFixed(1)}`).join(" ");
  const rej = c.p < cur().alpha;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)} ECDF difference">`;
  s += `<text class="t" x="${M.l}" y="12">${esc(title)}: p ${c.p <= 0.001 ? "&lt; .001" : c.p.toFixed(3)}${rej ? " (rejects)" : ""}</text>`;
  s += `<polygon points="${bandPts}" fill="var(--band)"/>`;
  s += `<line x1="${X(0)}" x2="${X(1)}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)" stroke-dasharray="3 3"/>`;
  s += `<polyline points="${poly(c.ecdf)}" fill="none" stroke="${rej ? "var(--bad)" : "var(--s1)"}" stroke-width="2"/>`;
  [-lim, 0, lim].forEach(v => s += `<text x="${M.l - 4}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(2)}</text>`);
  [0, 0.5, 1].forEach(v => s += `<text x="${X(v)}" y="${H - 8}" text-anchor="middle">${v}</text>`);
  return s + `</svg>`;
}
function drawPanels() {
  const r = cur(), c = r.cells[S.stratum][S.test];
  document.getElementById("panel-note").innerHTML = `<b>${LABEL[S.test]}</b>, ${S.stratum === "all" ? "all SNR" : "SNR " + S.stratum}: the ECDF of the truth's rank minus the uniform diagonal, against the simultaneous ${100 * (1 - r.alpha)}% band of the γ test (<code>validate.gamma_test</code>, so p matches the table). A curve leaving the band upward at the left and downward at the right is a truth too often in the draws' middle (draws too narrow), the opposite shape too wide, a one-sided drift a bias.`;
  const items = c.slot.map((x, i) => [x, `slot ${i + 1}`]).concat([[c.pooled, "pooled"], [c.joint, "joint"]]).filter(([x]) => x);
  document.getElementById("panels").innerHTML = items.map(([x, t]) => panel(x, r.bands[x.n], t)).join("") || `<p class="note">Too few ranks in this stratum.</p>`;
}

function drawLadder() {
  const r = cur(), band = r.band;
  const cols = DATA.filter(d => d.band === band && (d.kind === "run" || d.kind === "mock"))
    .sort((a, b) => (a.kind === "run") - (b.kind === "run") || MOCK_ORDER.indexOf(a.mock) - MOCK_ORDER.indexOf(b.mock) || a.setting.localeCompare(b.setting));
  const head = cols.map(d => `<th data-tip="${esc(d.kind === "mock" ? MOCK_DESC[d.mock] : label(d))}" style="${d.setting === S.setting ? "color:var(--ink)" : ""}">${d.kind === "mock" ? d.mock : esc(d.run.replace(/-narrow|-full/, "")) + "<br>" + d.ode_steps + " steps"}</th>`).join("");
  let s = `<table><tr><th></th>${head}</tr>`;
  const row = (name, pick) => {
    s += `<tr><th class="l">${name}</th>`;
    for (const d of cols) { const c = (d.cells[S.stratum] || d.cells.all)[pick.test][pick.at]; s += td(c, `${name}, ${d.kind === "mock" ? d.mock : d.setting}`); }
    s += `</tr>`;
  };
  for (const [g, tests] of GROUPS) {
    s += `<tr><th class="g" colspan="${cols.length + 1}">${g}</th></tr>`;
    for (const t of tests) row(LABEL[t] + (t === "tarp" || t === "average_rank" || t === "band_depth" ? ", slots" : ""), {test: t, at: "pooled"});
    if (g === "TARP") row("TARP, joint", {test: "tarp", at: "joint"});
    if (g === "Joint pre-ranks") { row("average rank, joint", {test: "average_rank", at: "joint"}); row("band depth, joint", {test: "band_depth", at: "joint"}); }
  }
  document.getElementById("ladder").innerHTML = s + `</table>`;
}

function draw() { drawBar(); drawBanners(); drawCells(); drawPanels(); drawLadder(); writeHash(); }

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
addEventListener("hashchange", () => { readHash(); draw(); });
readHash();
draw();
</script>
</body>
</html>
"""

MEANS = """
<ul>
<li><b>No run yet.</b> The page holds only the mock ladder (src at 8310cb3, 2026-10-30). Run rows drop in once a run has an eval of the current layout (64-injection grid, <code>fdot</code> in place of Mc); the 2026-10-24 <code>dusty-otter-narrow-XS</code> files are kept in <code>data/</code> but not shown.</li>
<li><b>Every eval source is faint.</b> The mocks ran with <code>visible_snr = 0</code> (catalogue windows): the draws' median SNR (<code>snr_hat</code>) is at most 2.6, so every injection is in the 0–15 stratum and the loud strata are empty (a dash). Read the ladder as a faint-source ladder; rerun with <code>visible_snr</code> on.</li>
<li><b>Clean marginal SBC proves nothing alone.</b> The <code>prior</code> mock, an identity flow that ignores the data, passes f0, lon, lat, the z± components, TARP, average rank and band depth in the narrow band (fdot p = 0.009 is its only coordinate rejection); in the full band lat (p = 0.016) and Re z+ (0.044) are marginal. What catches it is in <a href="../data-space-checks/data-space-checks.html">data-space-checks</a>; the measured power of each test is in <a href="../test-power/test-power.html">test-power</a>.</li>
<li><b>The ladder behaves as designed.</b> <code>wide</code> and <code>narrow</code> reject f0, fdot, lon, lat, the z± components, the derived ranks, TARP and band depth in both bands. <code>biased</code> (f0 +1σ) rejects every test except fdot (p = 0.58 narrow, 0.86 full); f0 mean rank offset −0.19 narrow, −0.14 full.</li>
<li><b><code>fisher</code> is not a clean null at this SNR.</b> f0, fdot and f0 ± fdot pass, but lon, lat, the z± components, |z±|, TARP and the joint ranks reject (narrow, 200 injections): faint sources are non-Gaussian in the flow coordinates, so the linearised Gaussian is miscalibrated there. <code>phases-prior</code> therefore cannot be told from it, and <code>loud-floor</code> equals it (its cap starts at source SNR 18.4).</li>
</ul>
"""

def main() -> None:
    records = [json.loads(p.read_text()) for p in sorted((REPORT / "data").glob("*.json"))]
    # records from before the (f0, fdot, A) chart name f0_plus_mc: no run eval of the current layout yet
    records = [r for r in records if "f0_plus_fdot" in r["tests"]]
    records.sort(key=lambda r: (r["kind"] != "run", r["band"], r.get("run", ""), r["ode_steps"], MOCK_ORDER.index(r["mock"]) if r["kind"] == "mock" else 0))
    page = (
        PAGE.replace("__MEANS__", MEANS)
        .replace("__MOCK_ORDER__", json.dumps(MOCK_ORDER))
        .replace("__DATA__", json.dumps(records, separators=(",", ":")))
    )
    out = REPORT / "calibration.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB, {len(records)} settings)")


if __name__ == "__main__":
    main()
