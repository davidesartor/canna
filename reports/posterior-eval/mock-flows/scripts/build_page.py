"""Write mock-flows.html from data/<band>/<mock>/mock.json: what each mock is and whether the ladder behaves as designed.

Stdlib only; run from the repo root:
    uv run --frozen python reports/posterior-eval/mock-flows/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]
MOCKS = ("fisher", "wide", "narrow", "biased", "prior", "phases-prior", "loud-floor")


def main() -> None:
    data = {}
    for band in ("narrow", "full"):
        records = [
            json.loads(path.read_text())
            for mock in MOCKS
            if (path := REPORT / "data" / band / mock / "mock.json").exists()
        ]
        if records:
            data[band] = records
    page = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = REPORT / "mock-flows.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB, bands {list(data)})")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA mock flows</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10); --good: #199e70; --bad: #c2362b; --warn: #b07a12;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --good: #1baf7a; --bad: #e5675c; --warn: #d9a23a;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 76ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 90ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.card + .card { margin-top: 16px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 16px; }
.k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 6px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
td.desc { white-space: normal; text-align: left; color: var(--ink-2); max-width: 70ch; }
td.good { color: var(--good); } td.bad { color: var(--bad); font-weight: 600; } td.warn { color: var(--warn); }
td.na { color: var(--muted); }
.scroll { overflow-x: auto; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>Mock flows: stand-in posteriors with known defects</h1>
<p class="sub">The lisa eval pipeline (<code>eval</code>, <code>scorecard</code>, <code>validate</code>) run on the same injections and keys as a real run, with the flow's sampler replaced by a mock. Each mock's eval dir has a real run's format, so eval pages can be checked against a known answer.</p>
<div class="bar"><div><span class="k">Band</span><span class="seg" id="band"></span></div></div>
<div class="card">
  <h2>What is each mock?</h2>
  <p class="note" id="counts"></p>
  <div class="scroll" id="t-what"></div>
</div>
<div class="card">
  <h2>Are the widths what the ladder says?</h2>
  <p class="note">Median over the eval grid's 64 injections' sources (8 windows × 8 SNR ranks) of the matched mock width over the Fisher forecast's width (<code>draws.npz</code> <code>matched</code> vs <code>fisher</code>, 1.4826 MAD in flow coordinates). <code>fisher</code> should read ~1 where the data dominate; the forecast and the mock differ in centre and in prior handling, so prior-dominated coordinates (quiet sources, fdot) need not. Columns: f0 per SNR band, then every coordinate over all sources.</p>
  <div class="scroll" id="t-width"></div>
</div>
<div class="card">
  <h2>Does the scorecard see it?</h2>
  <p class="note">The scorecard's f0 columns over its eval + random injections (<code>scorecard.npz</code>), per true-SNR band: width over the monochromatic bound, 68% and 95% coverage. Calibrated: in68 ≈ 0.68, in95 ≈ 0.95.</p>
  <div class="scroll" id="t-score"></div>
</div>
<div class="card">
  <h2>Does SBC see it?</h2>
  <p class="note">γ-test p-values of the truth's rank (<code>scorecard.npz</code>'s validation scores), all slots pooled; red below 0.05. Marginal coordinates first, then data-dependent and joint quantities. The prior mock passes every marginal rank by construction (it is SBC's blind spot); the joint log-likelihood rank should catch it. Loud: the slots with snr_hat ≥ 100.</p>
  <div class="bar"><div><span class="k">Stratum</span><span class="seg" id="strat"></span></div></div>
  <div class="scroll" id="t-sbc"></div>
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const COORDS = ["f0", "fdot", "lon", "lat", "re_zp", "im_zp", "re_zm", "im_zm"];
const SLOT = ["f0", "fdot", "lon", "lat", "re_zp", "im_zp", "re_zm", "im_zm", "abs_zp", "abs_zm", "overlap", "own_snr", "tarp"];
const JOINT = ["log_likelihood", "tarp", "average_rank", "band_depth"];
let band = Object.keys(D)[0], strat = "all";
const fmt = (v, n = 2) => v == null || !isFinite(v) ? "—" : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(n));
const esc = s => String(s).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));

function seg(id, items, get, set) {
  const el = document.getElementById(id);
  el.innerHTML = items.map(([v, l]) => `<button data-v="${v}" aria-pressed="${v === get()}">${l}</button>`).join("");
  el.onclick = e => { const b = e.target.closest("button"); if (b) { set(b.dataset.v); render(); } };
}
function table(head, rows) {
  return `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.join("")}</tr>`).join("")}</tbody></table>`;
}
const ratio = v => { const c = v == null ? "na" : (v > 1.5 || v < 0.67 ? "bad" : (v > 1.15 || v < 0.87 ? "warn" : "good")); return `<td class="${c}">${fmt(v)}</td>`; };
const cover = (v, t) => { const c = v == null ? "na" : (Math.abs(v - t) > 0.12 ? "bad" : (Math.abs(v - t) > 0.06 ? "warn" : "good")); return `<td class="${c}">${fmt(v)}</td>`; };
const pval = c => c == null || c.p == null ? `<td class="na">${c ? "n " + c.n : "—"}</td>` : `<td class="${c.p < 0.05 ? "bad" : "good"}" title="n ${c.n}, bias ${fmt(c.bias)}, middle half ${fmt(c.middle)}">${c.p < 0.001 ? "<0.001" : c.p.toFixed(3)}</td>`;

function render() {
  seg("band", Object.keys(D).map(b => [b, b]), () => band, v => band = v);
  seg("strat", [["all", "all"], ["loud", "loud (snr_hat ≥ 100)"]], () => strat, v => strat = v);
  const R = D[band], r0 = R[0];
  document.getElementById("counts").textContent = `${band} band, seed ${r0.seed}: ${r0.n_eval} eval injections × ${r0.eval_draws} draws (draws.npz); ${r0.n_eval} + ${r0.n_random} scorecard injections × ${r0.score_draws} draws; ${r0.n_validation} validation injections × ${r0.validation_draws} draws. eval.yaml marks every mock with epoch −1, ode_steps 0.`;
  document.getElementById("t-what").innerHTML = table(["mock", "what it is"], R.map(r => [`<td><code>${r.mock}</code></td>`, `<td class="desc">${esc(r.description)}</td>`]));

  const bands = Object.keys(r0.checks.width_over_fisher).filter(b => b !== "all");
  document.getElementById("t-width").innerHTML = table(
    ["mock", ...bands.map(b => `f0, SNR ${b}`), ...COORDS.map(c => `${c}, all`)],
    R.map(r => { const w = r.checks.width_over_fisher; return [`<td><code>${r.mock}</code></td>`, ...bands.map(b => ratio(w[b].n ? w[b].f0 : null)), ...COORDS.map(c => ratio(w.all[c]))]; }));

  const sb = Object.keys(r0.checks.scorecard);
  document.getElementById("t-score").innerHTML = table(
    ["mock", ...sb.flatMap(b => [`SNR ${b}<br>width/ideal`, "in68", "in95"])],
    R.map(r => [`<td><code>${r.mock}</code></td>`, ...sb.flatMap(b => { const s = r.checks.scorecard[b]; return s.n ? [`<td>${fmt(s.width_over_ideal)}</td>`, cover(s.in68, 0.68), cover(s.in95, 0.95)] : ["<td class=na>—</td>", "<td class=na>—</td>", "<td class=na>—</td>"]; })]));

  const v = r => r.checks.validation;
  document.getElementById("t-sbc").innerHTML = table(
    ["mock", ...SLOT, ...JOINT.map(j => `joint ${j}`), "width ratio"],
    R.map(r => [`<td><code>${r.mock}</code></td>`, ...SLOT.map(k => pval(v(r).per_slot[k]?.[strat])), ...JOINT.map(k => pval(v(r).joint[k]?.[strat])), `<td>${fmt(v(r).width_ratio_median[strat])}</td>`]));
}
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
