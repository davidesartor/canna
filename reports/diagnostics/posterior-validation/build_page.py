"""Render data/*.json (test_power.py) into posterior-validation.html.

Run from the repo root: uv run python reports/diagnostics/posterior-validation/build_page.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Posterior validation</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --lo: #eef4fc; --hi: #0b4a9e; --flag: #c2410c;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --lo: #1c2330; --hi: #6aa8f5; --flag: #f08a4b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --lo: #1c2330; --hi: #6aa8f5; --flag: #f08a4b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 16px 64px; }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub, .note { color: var(--ink-2); }
.note { font-size: 13px; margin: 4px 0 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 16px; position: relative; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 12px; font-size: 13px; color: var(--ink-2); }
select { font: inherit; background: var(--surface); color: var(--ink); border: 1px solid var(--border); border-radius: 6px; padding: 2px 6px; }
.wrap { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 2px; font-size: 12px; font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; padding: 2px 6px; vertical-align: bottom; }
th.l { text-align: left; white-space: nowrap; }
th.c { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; font-weight: 500; }
td { width: 46px; height: 24px; text-align: center; border-radius: 4px; color: var(--ink); }
td.na { color: var(--muted); }
td.flag { outline: 2px solid var(--flag); outline-offset: -2px; }
tr.gap td, tr.gap th { padding-top: 10px; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; white-space: nowrap; z-index: 2; color: var(--ink-2); }
.tip b { color: var(--ink); }
</style>
</head>
<body>
<main>
<h1>Posterior validation tests and their power</h1>
<p class="sub">Every test of <code>canna.lisa.validate</code>, first on the real injected truth, then on an exact null and known errors of the draws, to show which errors each test can see.</p>

<div class="card">
  <h2>Tests on the injected truth</h2>
  <p class="note">p-value of the ECDF γ test per slot (sources sorted loudest first) and on the whole window (joint), within a stratum of the draws' median SNR. Fill: −log₁₀ p (dark = strong evidence against calibration); outlined: rejected at α. Hover for the mean rank offset (−: truth below the draws) and the share of ranks in the middle half (0.5 when calibrated; above: draws too wide, below: too narrow).</p>
  <div class="controls">
    <label>eval <select id="eval"></select></label>
    <label>SNR stratum <select id="stratum"></select></label>
  </div>
  <div class="wrap" id="cells"></div>
  <div class="tip"></div>
</div>

<div class="card">
  <h2>Size and power of each test</h2>
  <p class="note">Share of __CHUNK__-injection chunks (and slots, for per-slot tests) in which a test rejects at α = __ALPHA__, with the draws perturbed as each column says. The first column is the exact null: it should read α for the rank tests. Score tests compare each perturbed set's score to the unperturbed draws' (one-sided paired Wilcoxon), so their null column is 0 by construction. <b>toy</b>: linear-Gaussian slots with exact posterior draws; <b>run</b>: one flow draw stands in for the truth, so the null holds whatever the flow's error.</p>
  <div class="controls"><label>setting <select id="setting"></select></label></div>
  <div class="wrap" id="power"></div>
  <div class="tip"></div>
</div>
</main>
<script>
const DATA = __DATA__;
const fill = v => `color-mix(in oklab, var(--hi) ${Math.round(100 * Math.min(1, Math.max(0, v)))}%, var(--lo))`;
const ink = v => v > 0.55 ? "var(--surface)" : "var(--ink)";
const label = r => r.setting === "toy" ? "toy" : `${r.run}, epoch ${r.epoch}, ${r.ode_steps} ODE steps`;

function hover(card) {
  const tip = card.querySelector(".tip");
  card.querySelectorAll("td[data-tip]").forEach(el => {
    el.addEventListener("pointermove", ev => {
      const box = card.getBoundingClientRect();
      tip.innerHTML = el.dataset.tip; tip.style.display = "block";
      tip.style.left = Math.min(ev.clientX - box.left + 14, box.width - tip.offsetWidth - 8) + "px";
      tip.style.top = (ev.clientY - box.top + 14) + "px";
    });
    el.addEventListener("pointerleave", () => tip.style.display = "none");
  });
}

// real-truth cells
const evals = DATA.filter(r => r.posterior);
const evalSel = document.getElementById("eval"), stratumSel = document.getElementById("stratum");
evalSel.innerHTML = evals.map((r, i) => `<option value="${i}">${label(r)}</option>`).join("");
function drawCells() {
  const r = evals[Number(evalSel.value)];
  if (!r) { document.getElementById("cells").textContent = "no run evaluated yet"; return; }
  if (!stratumSel.options.length) stratumSel.innerHTML = Object.keys(r.posterior).map(s => `<option${s === "all" ? " selected" : ""}>${s}</option>`).join("");
  const cellOf = r.posterior[stratumSel.value], tests = Object.entries(cellOf.tests), slots = cellOf.sources.length;
  let s = `<table><tr><th></th>${cellOf.sources.map((n, i) => `<th>slot ${i + 1}<br><span style="font-weight:400">${n}</span></th>`).join("")}<th>joint<br><span style="font-weight:400">${cellOf.injections}</span></th></tr>`;
  const td = (c, name, where, n) => {
    if (!c) return `<td class="na">–</td>`;
    const [p, bias, middle] = c, v = Math.min(1, -Math.log10(Math.max(p, 1e-6)) / 4);
    return `<td class="${p < r.alpha ? "flag" : ""}" style="background:${fill(v)};color:${ink(v)}" data-tip="<b>${name}, ${where}</b><br>${n} ranks<br>p = ${p.toPrecision(2)}<br>mean rank offset ${bias >= 0 ? "+" : ""}${bias.toFixed(3)}<br>middle-half share ${middle.toFixed(2)}">${p < 0.001 ? "<.001" : p.toFixed(2)}</td>`;
  };
  tests.forEach(([name, t]) => {
    s += `<tr><th class="l">${name}</th>`;
    for (let i = 0; i < slots; i++) s += t.slot ? td(t.slot[i], name, `slot ${i + 1}`, cellOf.sources[i]) : `<td class="na">–</td>`;
    s += td(t.joint, name, "joint", cellOf.injections) + `</tr>`;
  });
  document.getElementById("cells").innerHTML = s + `</table>`;
  hover(document.getElementById("cells").parentElement);
}
evalSel.onchange = () => { stratumSel.innerHTML = ""; drawCells(); };
stratumSel.onchange = drawCells;
drawCells();

// power heatmap
const settingSel = document.getElementById("setting");
settingSel.innerHTML = DATA.map((r, i) => `<option value="${i}">${label(r)} (${r.injections} injections, ${r.draws} draws)</option>`).join("");
function drawPower() {
  const r = DATA[Number(settingSel.value)], perts = r.perturbations, tests = Object.keys(r.power[perts[0]]);
  let s = `<table><tr><th></th>${perts.map(p => `<th class="c">${p}</th>`).join("")}</tr>`;
  tests.forEach((test, i) => {
    const gap = i && test.split(" ")[0] !== tests[i - 1].split(" ")[0];
    s += `<tr class="${gap ? "gap" : ""}"><th class="l">${test}</th>`;
    perts.forEach(p => {
      const v = r.power[p][test];
      s += `<td style="background:${fill(v)};color:${ink(v)}" data-tip="<b>${test}</b><br>${p}<br>rejects in <b>${(100 * v).toFixed(0)}%</b> of chunks">${(100 * v).toFixed(0)}</td>`;
    });
    s += `</tr>`;
  });
  document.getElementById("power").innerHTML = s + `</table>`;
  hover(document.getElementById("power").parentElement);
}
settingSel.onchange = drawPower;
drawPower();
</script>
</body>
</html>
"""


def main() -> None:
    records = [
        json.loads(p.read_text()) for p in sorted((HERE / "data").glob("*.json"))
    ]
    records.sort(key=lambda r: r["setting"] != "toy")
    page = PAGE.replace("__DATA__", json.dumps(records)).replace(
        "__CHUNK__", str(records[0]["chunk"])
    )
    page = page.replace("__ALPHA__", f"{records[0]['alpha']:g}")
    out = HERE / "posterior-validation.html"
    out.write_text(page)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
