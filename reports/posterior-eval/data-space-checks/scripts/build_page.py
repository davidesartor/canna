"""Write data-space-checks.html from data/*.json (data_space_checks.py): ranks of the draws as the simulator sees them.

Stdlib only; run from the repo root:
    uv run --frozen python reports/posterior-eval/data-space-checks/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data-space checks</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --lo: #eef4fc; --hi: #0b4a9e; --bad: #c2362b; --warn: #b36b00; --band: rgba(11,74,158,0.14); --pt: rgba(11,74,158,0.45);
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --lo: #1c2330; --hi: #6aa8f5; --bad: #ee7b6f; --warn: #e0a040; --band: rgba(106,168,245,0.18); --pt: rgba(106,168,245,0.5);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 24px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
a { color: var(--hi); }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 16px; max-width: 74ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 90ch; }
.nav { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; margin: 0 0 18px; }
.nav a { color: var(--ink-2); text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.nav b { font-weight: 600; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 14px; }
.bar .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.banner { display: none; border: 1px solid var(--border); border-left: 4px solid var(--warn); background: var(--surface); border-radius: 8px; padding: 10px 14px; margin: 0 0 16px; font-size: 14px; }
.scroll { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 2px; font-size: 12px; font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; padding: 2px 6px; vertical-align: bottom; }
th.l { text-align: left; white-space: nowrap; font-weight: 500; }
th.c { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; font-weight: 500; }
th.c.sel { color: var(--ink); font-weight: 700; }
th.g { text-align: left; color: var(--ink); padding-top: 12px; }
td { min-width: 52px; height: 24px; text-align: center; border-radius: 4px; color: var(--ink); }
td.sel { outline: 1.5px solid var(--ink); outline-offset: -1.5px; }
.grid5 { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }
.mini { border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; }
.mini .t { font-size: 12px; color: var(--ink-2); display: flex; justify-content: space-between; }
.mini .t b { color: var(--ink); font-weight: 600; }
.mini .rej { color: var(--bad); font-weight: 600; }
.grid4 { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
svg { display: block; width: 100%; height: auto; }
svg text { fill: var(--muted); font-size: 10px; font-family: inherit; }
svg .ax { stroke: var(--grid); stroke-width: 1; fill: none; }
svg .ref { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; fill: none; }
svg .line { stroke: var(--hi); stroke-width: 1.4; fill: none; }
svg .hb { fill: var(--hi); opacity: 0.75; }
svg .band { fill: var(--band); }
svg .pt { fill: var(--pt); }
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
<nav class="nav" id="nav"></nav>
<h1>Do re-simulated draws reproduce the data?</h1>
<p class="sub">The data-space ranks judge the flow's draws through the simulator: each draw's overlap with the data, its own SNR and its log-likelihood, ranked against the truth's. They see errors that leave every 1-D marginal exact. Marginal ranks are in <a href="../calibration/calibration.html">calibration</a>, the tests' measured power in <a href="../test-power/test-power.html">test-power</a>.</p>
<div class="bar">
  <div><span class="k">Setting</span><span class="seg" id="setting"></span></div>
  <div id="case-wrap"><span class="k">Mock</span><span class="seg" id="case"></span></div>
</div>
<div class="banner" id="banner"></div>

<div class="card" id="ladder-card">
  <h2>Which tests reject which defect?</h2>
  <p class="note" id="ladder-note"></p>
  <div class="scroll" id="ladder"></div>
  <div class="tip"></div>
</div>

<div class="card" id="rank-card">
  <h2>Ranks of the truth among the draws, by SNR stratum</h2>
  <p class="note">Each panel is one stratum (the draws' median SNR, a function of the data alone). Top: the ECDF of the ranks minus the diagonal, inside the simultaneous 95% band of uniform ranks; a curve leaving the band rejects. Below: the rank histogram in 16 bins, dashed at the uniform expectation. p is the repo's γ test.</p>
  <div class="bar">
    <div><span class="k">Test</span><span class="seg" id="test"></span></div>
    <div id="slot-wrap"><span class="k">Slot</span><span class="seg" id="slot"></span></div>
  </div>
  <div class="grid5" id="ranks"></div>
  <div class="tip"></div>
</div>

<div class="card" id="raw-card">
  <h2>The distributions behind the ranks</h2>
  <p class="note">Truth against the median of the 127 draws, for the first 300 injections of the setting, all 4 slots where per-slot (dashed: equal). Axes are logarithmic for SNR and symmetric-log for the signed overlap and the log-likelihood. A posterior that reproduces the data scatters about the diagonal; one far from the data sits below it. The own-SNR panel is also snr_hat against the true SNR, since snr_hat is the draws' median own SNR.</p>
  <div class="grid4" id="raw"></div>
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
const SIBLINGS = ["test-power", "sampler-convergence", "robustness", "mock-flows", "reference-comparison", "fisher-scorecard", "calibration", "data-space-checks", "proper-scores"];
const LADDER = [
  ["Data-space ranks (this page)", [["overlap", "overlap, all slots"], ["own_snr", "own SNR, all slots"], ["log_likelihood", "joint log-likelihood"]]],
  ["Marginal and geometric ranks (see calibration)", [["coords_min", "8 coordinates, smallest p"], ["abs_zp", "|z+| radius"], ["abs_zm", "|z-| radius"], ["tarp", "TARP, per slot"], ["band_depth", "band depth, per slot"], ["joint_tarp", "TARP, joint"], ["joint_band_depth", "band depth, joint"]]],
];
const TESTS = {overlap: "Overlap with the data", own_snr: "Own SNR", log_likelihood: "Joint log-likelihood"};
const S = {setting: DATA[0].setting, mock: null, test: "overlap", slot: "pooled"};
const $ = id => document.getElementById(id);
const setting = () => DATA.find(d => d.setting === S.setting);
const kase = () => { const s = setting(); return s.cases.find(c => c.name === S.mock) || s.cases[0]; };
const fmtP = p => p === null || p === undefined ? "–" : p < 0.001 ? "<.001" : p < 0.1 ? p.toFixed(3) : p.toFixed(2);
const caseLabel = (s, c) => c.mock ? c.name : `${s.setting}`;
const settingLabel = s => s.cases[0].mock ? `mock flows, ${s.band} band` : s.setting.replace(/-(\d+)$/, ", $1 ODE steps");

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (DATA.some(d => d.setting === h.get("setting"))) S.setting = h.get("setting");
  if (h.get("mock")) S.mock = h.get("mock");
  if (TESTS[h.get("test")]) S.test = h.get("test");
}
function writeHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  h.set("setting", S.setting); h.set("test", S.test);
  if (setting().cases.length > 1) h.set("mock", kase().name); else h.delete("mock");
  history.replaceState(null, "", "#" + h.toString());
}

const seg = (el, items, key) => {
  $(el).innerHTML = items.map(([v, l]) => `<button aria-pressed="${S[key] === v}" data-v="${v}">${l}</button>`).join("");
  $(el).onclick = e => { const b = e.target.closest("button"); if (b) { S[key] = b.dataset.v; if (key === "setting") S.mock = null; draw(); } };
};

function drawBar() {
  $("nav").innerHTML = SIBLINGS.map(n => n === "data-space-checks" ? `<b>${n}</b>` : `<a href="../${n}/${n}.html">${n}</a>`).join("");
  seg("setting", DATA.map(d => [d.setting, settingLabel(d)]), "setting");
  const s = setting();
  $("case-wrap").style.display = s.cases.length > 1 ? "" : "none";
  if (!s.cases.some(c => c.name === S.mock)) S.mock = s.cases[0].name;
  seg("case", s.cases.map(c => [c.name, c.name]), "mock");
  const b = $("banner");
  b.style.display = s.smoke ? "block" : "none";
  b.innerHTML = s.smoke ? `<b>Smoke run.</b> <code>${s.setting}</code> is a 2000-step narrow-XS pipeline test at epoch 4: its numbers show the tests working on a real flow, not how good the method is. Real findings need narrow-B and full-B; rerun the job in the README when they land.` : "";
}

// ladder: p of every test (rows) against every mock and run of the band (columns)
function drawLadder() {
  const s = setting(), cur = kase();
  const cols = [];
  DATA.filter(d => d.band === s.band).sort((a, b) => b.cases[0].mock - a.cases[0].mock).forEach(d => d.cases.forEach(c => cols.push([d, c])));
  $("ladder-note").innerHTML = `Pooled γ-test p over all injectionsof the ${s.band} band, per column: a mock flow (known defect; <code>fisher</code> is the near-calibrated base, details in <a href="../mock-flows/mock-flows.html">mock-flows</a>) or a run. Red: p &lt; 0.05, darker for smaller p. Per-slot tests are pooled over the 4 slots. ${cols.map(([d, c]) => `${caseLabel(d, c)}: ${c.n} injections`).join("; ")}.`;
  const red = p => p === null ? "transparent" : p >= 0.05 ? "color-mix(in oklab, var(--hi) 7%, var(--surface))" : `color-mix(in oklab, var(--bad) ${Math.round(100 * Math.min(1, 0.35 + Math.log10(0.05 / Math.max(p, 1e-4)) / 2.5 * 0.65))}%, var(--surface))`;
  let h = `<table><tr><th></th>${cols.map(([d, c]) => `<th class="c ${d === s && c === cur ? "sel" : ""}">${caseLabel(d, c)}</th>`).join("")}</tr>`;
  LADDER.forEach(([group, rows]) => {
    h += `<tr><th class="g" colspan="${cols.length + 1}">${group}</th></tr>`;
    rows.forEach(([key, label]) => {
      h += `<tr><th class="l">${label}</th>` + cols.map(([d, c]) => {
        const p = c.ladder[key];
        return `<td class="${d === s && c === cur ? "sel" : ""}" style="background:${red(p)}" data-tip="<b>${label}</b><br>${caseLabel(d, c)}<br>p = <b>${fmtP(p)}</b>">${fmtP(p)}</td>`;
      }).join("") + `</tr>`;
    });
  });
  $("ladder").innerHTML = h + `</table>`;
}

function drawRanks() {
  const c = kase(), s = setting();
  seg("test", Object.entries(TESTS), "test");
  $("slot-wrap").style.display = S.test === "log_likelihood" ? "none" : "";
  const slots = Array.from({length: 4}, (_, i) => [String(i), `slot ${i}`]).concat([["pooled", "pooled"]]);
  seg("slot", slots, "slot");
  const panels = c.panels[S.test === "log_likelihood" ? "log_likelihood" : `${S.test} ${S.slot}`];
  const yMax = Math.max(0.02, ...panels.filter(p => p.ecdf).flatMap(p => [...p.ecdf.map(Math.abs), ...p.band.flat().map(Math.abs)]));
  const W = 190, H1 = 96, H2 = 50, L = 26, R = 4, T = 6;
  $("ranks").innerHTML = panels.map(p => {
    const head = `<div class="t"><span>SNR ${p.stratum}</span><span>n=${p.n}</span></div>`;
    if (!p.ecdf) return `<div class="mini">${head}<div class="t"><b>too few ranks</b></div></div>`;
    const k = p.ecdf.length + 1, x = i => L + (W - L - R) * (i + 1) / k, y = v => T + (H1 - 2 * T) * (1 - (v + yMax) / (2 * yMax));
    const band = p.band[1].map((v, i) => `${x(i)},${y(v)}`).concat(p.band[0].map((v, i) => [x(i), y(v)]).reverse().map(a => a.join(","))).join(" ");
    const line = p.ecdf.map((v, i) => `${x(i)},${y(v)}`).join(" ");
    const hmax = Math.max(...p.hist, p.n / p.hist.length * 1.3), bw = (W - L - R) / p.hist.length;
    const bars = p.hist.map((v, i) => `<rect class="hb" x="${L + i * bw + 0.5}" y="${H2 - 4 - (H2 - 8) * v / hmax}" width="${bw - 1}" height="${(H2 - 8) * v / hmax}" data-tip="ranks ${Math.round(i * k / 16)}–${Math.round((i + 1) * k / 16) - 1}: <b>${v}</b> (uniform ${(p.n / 16).toFixed(1)})"/>`).join("");
    const e = H2 - 4 - (H2 - 8) * (p.n / p.hist.length) / hmax;
    return `<div class="mini"><div class="t"><span>SNR ${p.stratum}</span><span>n=${p.n}</span></div>
      <div class="t"><span>p</span><span class="${p.p < 0.05 ? "rej" : ""}"><b>${fmtP(p.p)}</b></span></div>
      <svg viewBox="0 0 ${W} ${H1}"><polygon class="band" points="${band}"/><line class="ax" x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}"/>
        <polyline class="line" points="${line}"/>
        <text x="${L - 3}" y="${y(yMax) + 8}" text-anchor="end">+${yMax.toFixed(2)}</text><text x="${L - 3}" y="${y(-yMax)}" text-anchor="end">−${yMax.toFixed(2)}</text><text x="${L - 3}" y="${y(0) + 3}" text-anchor="end">0</text></svg>
      <svg viewBox="0 0 ${W} ${H2}"><line class="ax" x1="${L}" x2="${W - R}" y1="${H2 - 4}" y2="${H2 - 4}"/>${bars}<line class="ref" x1="${L}" x2="${W - R}" y1="${e}" y2="${e}"/>
        <text x="${L - 3}" y="${H2 - 1}" text-anchor="end">0</text></svg></div>`;
  }).join("");
}

// transforms: log for SNR, symlog (linear within ±1) for signed quantities
const logT = v => Math.log10(Math.max(v, 1e-3)), symT = v => Math.sign(v) * Math.log10(1 + Math.abs(v));
function scatter(title, pts, T, ticks, xl, yl) {
  const W = 230, H = 230, L = 34, B = 22, T0 = 8, R = 8;
  const vals = pts.flatMap(p => [T(p[0]), T(p[1])]);
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.05 || 1;
  const sx = v => L + (W - L - R) * (T(v) - lo + pad) / (hi - lo + 2 * pad), sy = v => H - B - (H - B - T0) * (T(v) - lo + pad) / (hi - lo + 2 * pad);
  const tk = ticks.filter(v => T(v) >= lo - pad && T(v) <= hi + pad);
  const dots = pts.map(p => `<circle class="pt" cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="2" data-tip="truth <b>${p[0]}</b><br>draws' median <b>${p[1]}</b>"/>`).join("");
  const fmt = v => Math.abs(v) >= 1000 ? v.toExponential(0).replace("+", "") : v;
  return `<div><div class="t" style="font-size:12px;color:var(--ink-2)"><b style="color:var(--ink)">${title}</b></div><svg viewBox="0 0 ${W} ${H}">
    <rect x="${L}" y="${T0}" width="${W - L - R}" height="${H - B - T0}" class="ax"/>
    ${tk.map(v => `<line class="ax" x1="${sx(v)}" x2="${sx(v)}" y1="${T0}" y2="${H - B}"/><line class="ax" x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${sx(v)}" y="${H - B + 10}" text-anchor="middle">${fmt(v)}</text><text x="${L - 3}" y="${sy(v) + 3}" text-anchor="end">${fmt(v)}</text>`).join("")}
    <line class="ref" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${T0}"/>${dots}
    <text x="${(L + W - R) / 2}" y="${H - 1}" text-anchor="middle">${xl}</text><text x="8" y="${(T0 + H - B) / 2}" transform="rotate(-90 8 ${(T0 + H - B) / 2})" text-anchor="middle">${yl}</text></svg></div>`;
}
function drawRaw() {
  const sc = kase().scatter, dec = Array.from({length: 13}, (_, i) => 10 ** (i - 3));
  const sym = [0, ...dec, ...dec.map(v => -v)];
  $("raw").innerHTML = [
    scatter("Overlap with the data", sc.overlap, symT, sym, "truth", "draws' median"),
    scatter("Own SNR (the median is snr_hat)", sc.own_snr, logT, dec, "true SNR", "snr_hat"),
    scatter("Log-likelihood (per injection)", sc.log_likelihood, symT, sym, "truth", "draws' median"),
  ].join("");
}

function draw() { drawBar(); drawLadder(); drawRanks(); drawRaw(); writeHash(); }

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
<li><b>No run yet.</b> The page holds the mock ladder only (src at 8310cb3, 2026-10-30). Run rows drop in once a run has an eval of the current layout; the 2026-10-24 <code>dusty-otter-narrow-XS</code> files stay in <code>data/</code> but are not shown.</li>
<li><b>Every eval source is faint</b> (the mocks ran with <code>visible_snr = 0</code>, catalogue windows; <code>snr_hat</code> at most 2.6), so all injections are in the 0–15 stratum and the other strata are empty. The data-space tests are read at sub-threshold SNR; rerun with <code>visible_snr</code> on.</li>
<li><b>Only the data-space ranks catch <code>prior</code>.</b> The identity flow passes own SNR (p = 0.76 narrow, 0.57 full), TARP, average rank and band depth, and is rejected by the joint log-likelihood rank (p = 0.001 narrow, &lt; 0.001 full) and by the overlap rank (p = 0.003 narrow, 0.007 full).</li>
<li><b>Width and shift defects are not clean here.</b> <code>wide</code>, <code>narrow</code> and <code>biased</code> all reject own SNR and the joint log-likelihood rank (p &lt; 0.001), but overlap passes <code>wide</code> (p = 0.09 narrow, 0.10 full) and <code>biased</code> (0.21, 0.15), and <code>narrow</code> on the full band (0.18).</li>
<li><b><code>fisher</code> is not a null for these tests.</b> It rejects own SNR and the joint log-likelihood rank in both bands (p &lt; 0.001); overlap rejects narrow (0.003) and passes full (0.088). <code>phases-prior</code> and <code>loud-floor</code> read like it (<code>phases-prior</code> overlap on the full band rejects, p = 0.034; <code>loud-floor</code> equals it: no source reaches its SNR 18.4 cap). Compare a mock's p to <code>fisher</code>'s, not to 0.05.</li>
</ul>
"""


def main() -> None:
    records = [json.loads(p.read_text()) for p in sorted((REPORT / "data").glob("*.json"))]
    # records from before the (f0, fdot, A) chart have an Mc coordinate: no run eval of the current layout yet
    records = [r for r in records if "fdot" in r["cases"][0]["coord_p"]]
    records.sort(key=lambda r: (r["cases"][0]["mock"], r["band"] != "narrow", r["setting"]))
    page = PAGE.replace("__MEANS__", MEANS).replace("__DATA__", json.dumps(records, separators=(",", ":")))
    out = REPORT / "data-space-checks.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
