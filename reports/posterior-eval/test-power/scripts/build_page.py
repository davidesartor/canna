"""Write test-power.html from data/*.json (test_power.py): the size and power of every validation test.

Stdlib only; run from the repo root:
    uv run --frozen --extra cuda python reports/posterior-eval/test-power/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validation test power</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --lo: #eef4fc; --hi: #0b4a9e;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --lo: #1c2330; --hi: #6aa8f5;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 16px; }
.bar .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.scroll { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 2px; font-size: 12px; font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; padding: 2px 6px; vertical-align: bottom; }
th.l { text-align: left; white-space: nowrap; font-weight: 500; }
th.c { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; font-weight: 500; }
th.g { text-align: left; color: var(--ink); padding-top: 12px; }
td { width: 46px; height: 24px; text-align: center; border-radius: 4px; color: var(--ink); }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; white-space: nowrap; }
.tip b { color: var(--ink); font-weight: 600; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 80ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>Which errors can each validation test see?</h1>
<p class="sub">Every test of <code>canna.lisa.validate</code> on an exact null and on known errors injected into the draws: the null column is the test's size, the others its power.</p>
<div class="bar"><div><span class="k">Setting</span><span class="seg" id="setting"></span></div></div>
<div class="card">
  <h2>How often does each test reject, under each error of the draws?</h2>
  <p class="note">Share of __CHUNK__-injection chunks (and slots, for per-slot tests) in which a test rejects at α = __ALPHA__, in %. The first column is the exact null: it should read α for the rank tests. Score tests compare each perturbed set's score to the unperturbed draws' (one-sided paired Wilcoxon), so their null column is 0 by construction. <b>toy</b>: linear-Gaussian slots with exact posterior draws; <b>run</b>: one flow draw stands in for the truth, so the null holds whatever the flow's error. <span id="meta"></span></p>
  <div class="scroll" id="power"></div>
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
const GROUPS = {slot: "Per-slot ranks (γ test per slot)", joint: "Joint ranks over the window (γ test)", score: "Proper scores (worse than unperturbed, paired Wilcoxon)"};
const fill = v => `color-mix(in oklab, var(--hi) ${Math.round(100 * Math.min(1, Math.max(0, v)))}%, var(--lo))`;
const ink = v => v > 0.55 ? "var(--surface)" : "var(--ink)";
const key = r => r.setting === "toy" ? "toy" : `${r.run}-${r.ode_steps}`;
const label = r => r.setting === "toy" ? "toy" : `${r.run}, epoch ${r.epoch}, ${r.ode_steps} ODE steps`;
const S = {setting: key(DATA[0])};

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (DATA.some(r => key(r) === h.get("setting"))) S.setting = h.get("setting");
}
function writeHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  h.set("setting", S.setting);
  history.replaceState(null, "", "#" + h.toString());
}

function drawBar() {
  document.getElementById("setting").innerHTML = DATA.map(r =>
    `<button aria-pressed="${key(r) === S.setting}" data-setting="${key(r)}">${label(r)}</button>`).join("");
}
document.getElementById("setting").addEventListener("click", e => {
  const b = e.target.closest("button[data-setting]");
  if (b) { S.setting = b.dataset.setting; draw(); }
});

function drawPower() {
  const r = DATA.find(d => key(d) === S.setting), perts = r.perturbations, tests = Object.keys(r.power[perts[0]]);
  const chunks = Math.floor(r.injections / r.chunk);
  document.getElementById("meta").textContent = `This setting: ${r.injections} injections × ${r.draws} draws, ${chunks} chunks.`;
  let s = `<table><tr><th></th>${perts.map(p => `<th class="c">${p}</th>`).join("")}</tr>`;
  tests.forEach((test, i) => {
    const [group, ...rest] = test.split(" "), name = rest.join(" ");
    if (!i || group !== tests[i - 1].split(" ")[0]) s += `<tr><th class="g" colspan="${perts.length + 1}">${GROUPS[group] || group}</th></tr>`;
    s += `<tr><th class="l">${name}</th>`;
    perts.forEach(p => {
      const v = r.power[p][test];
      s += `<td style="background:${fill(v)};color:${ink(v)}" data-tip="<b>${test}</b><br>${p}<br>rejects in <b>${(100 * v).toFixed(0)}%</b> of chunks">${(100 * v).toFixed(0)}</td>`;
    });
    s += `</tr>`;
  });
  document.getElementById("power").innerHTML = s + `</table>`;
}

function draw() { drawBar(); drawPower(); writeHash(); }

// one tooltip per card, for anything carrying data-tip
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
<li><b>The rank tests hold their size.</b> On the exact null they reject in 2–9% of the 50 chunks on the toy, against α = 5%. No run row yet: it drops in once a run has an eval of the current layout (the 2026-10-24 <code>dusty-otter-narrow-XS</code> files are kept in <code>data/</code> but not shown).</li>
<li><b>Per-coordinate SBC is blind to whole classes of error.</b> z+ phase or sky from the prior, and dependence shuffled, leave every 1-D marginal exact, so the coordinate ranks reject at the null rate; 10% prior draws reach only about 10%. The data-space ranks (overlap, log-likelihood, own SNR for the shuffle) reject in 97–100% of chunks, and so does the energy score; the CRPS of the affected coordinates catches the phase and sky errors (100%) and 10% prior draws (58–81%), but not the shuffle (4–9%). Posterior = prior passes marginal SBC: a clean rank histogram on its own proves nothing.</li>
<li><b>Mild width errors need the right test.</b> The coordinate ranks catch narrow ×0.8 in 55–67% of chunks and wide ×1.25 in 32–48%, while overlap and the joint log-likelihood catch both every time.</li>
<li><b>An f0 shift of 0.5σ</b> is caught by the f0 rank (99%) and by the log-likelihood (100%).</li>
<li><b>TARP and average rank add little.</b> Per-slot TARP mostly repeats the width tests (81% for ×0.8, 70% for ×1.25). Average rank is the weakest joint test against everything but the f0 shift.</li>
</ul>
"""


def main() -> None:
    records = [
        json.loads(p.read_text()) for p in sorted((REPORT / "data").glob("*.json"))
    ]
    # records from before the (f0, fdot, A) chart name slot mc: no run eval of the current layout yet
    records = [r for r in records if "slot fdot" in r["power"]["null"]]
    records.sort(
        key=lambda r: (r["setting"] != "toy", r.get("run", ""), r.get("ode_steps", 0))
    )
    page = (
        PAGE.replace("__MEANS__", MEANS)
        .replace("__DATA__", json.dumps(records, separators=(",", ":")))
        .replace("__CHUNK__", str(records[0]["chunk"]))
        .replace("__ALPHA__", f"{records[0]['alpha']:g}")
    )
    out = REPORT / "test-power.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
