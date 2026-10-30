"""Write source-pairing.html from data/pairing.json: path length per source under each pairing of base and target sources.

Stdlib only; run from the repo root:
    uv run --frozen python reports/architecture/source-pairing/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]


def report(data: dict) -> None:
    """Print the numbers the README quotes."""
    for band, b in data["bands"].items():
        base = b["total"]["f0"]
        print(f"{band}: {b['windows']} windows, {b['sources']} sources")
        for k in data["pairings"]:
            print(
                f"  {k:8s} {b['total'][k]:.3f} ± {b['total_se'][k]:.3f}"
                f" ({b['total'][k] / base - 1:+.1%} against f0 order)  "
                + " ".join(f"{g} {v:.3f}" for g, v in zip(data["groups"], b["mean"][k]))
            )
        print(
            f"  f0 order = nearest labelling {b['f0_is_nearest']:.1%}, = loudest first {b['f0_is_snr_order']:.1%}"
        )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Source pairing</title>
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
<h1>How should a base draw's sources pair with the target's?</h1>
<p class="sub">Each training path runs from a prior draw X<sub>0</sub> to the target X<sub>1</sub>, source by source. The sources are exchangeable, so which base source goes to which target source is a choice. This page measures the mean squared path length per source, in flow coordinates, under four pairings: both ends in f<sub>0</sub> order (what training uses, a sort), the nearest of the S! labellings over all coordinates, both ends loudest first, and a random labelling. No network is involved: it follows from the prior and the chart alone.</p>
<div class="bar"><div><span class="k">Band</span><span class="seg" id="band"></span></div></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>Where is the path length?</h2>
  <p class="note">Mean squared path length per source, split by coordinate group (f<sub>0</sub>, log ḟ, the two sky coordinates, the two helicity amplitudes z±). Hover a segment for its value.</p>
  <div class="legend" id="lg-groups"></div>
  <div id="p-groups"></div>
  <div class="scroll" id="t-groups"></div>
</div>
<div class="card">
  <h2>Which sources pay for a fixed order?</h2>
  <p class="note">The same length per target source, by its loudness rank (1 is the loudest of the window). The nearest labelling's length is the least in total, not for each rank.</p>
  <div class="legend" id="lg-rank"></div>
  <div id="p-rank"></div>
</div>
<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const BANDS = Object.keys(D.bands);
const GCOL = ["var(--g0)", "var(--g1)", "var(--g2)", "var(--g3)", "var(--g4)"];
const PCOL = {f0: "var(--g0)", nearest: "var(--g2)", snr: "var(--g1)", random: "var(--muted)"};
const PNAME = {f0: "f0 order (training)", nearest: "nearest labelling", snr: "loudest first", random: "random labels"};
const S = {band: BANDS[BANDS.length - 1]};
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const fmt = (v, n = 3) => Number(v).toPrecision(n);
const pct = v => (v >= 0 ? "+" : "−") + Math.abs(100 * v).toFixed(1) + "%";

const segHost = document.getElementById("band");
segHost.innerHTML = BANDS.map(b => `<button data-v="${esc(b)}">${esc(b)}</button>`).join("");
segHost.onclick = e => { const b = e.target.closest("button"); if (!b) return; S.band = b.dataset.v; render(); };
const sq = c => `<svg width="12" height="12" style="width:12px;display:inline-block"><rect width="12" height="12" rx="2" fill="${c}"/></svg>`;
const legend = (id, items) => document.getElementById(id).innerHTML = items.map(([c, label]) => `<span>${sq(c)}${esc(label)}</span>`).join("");
const ticks = (hi, n) => { const step = 10 ** Math.floor(Math.log10(hi / n)); const m = [1, 2, 5, 10].find(m => hi / (m * step) <= n) * step; const out = []; for (let v = 0; v <= hi + 1e-9; v += m) out.push(+v.toFixed(10)); return out; };

// one horizontal bar per pairing, stacked by coordinate group
function stacked(host, b) {
  const rowH = 30, M = {l: 190, r: 16, t: 6, b: 34};
  const W = Math.max(320, host.clientWidth), H = M.t + M.b + rowH * D.pairings.length;
  const hi = Math.max(...D.pairings.map(k => b.total[k])) * 1.05;
  const X = v => M.l + v / hi * (W - M.l - M.r);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="path length per pairing and group">`;
  ticks(hi, Math.max(3, Math.floor((W - M.l) / 70))).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${M.t}" y2="${H - M.b}" stroke="var(--grid)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${v}</text>`);
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">mean squared path length per source</text>`;
  D.pairings.forEach((k, i) => {
    const y = M.t + i * rowH + 5;
    let x0 = 0;
    s += `<text x="${M.l - 8}" y="${y + 14}" text-anchor="end" style="fill:var(--ink-2)">${esc(PNAME[k])}</text>`;
    b.mean[k].forEach((v, g) => { s += `<rect x="${X(x0)}" y="${y}" width="${Math.max(0, X(x0 + v) - X(x0) - 1)}" height="${rowH - 10}" fill="${GCOL[g]}" rx="2"><title>${esc(PNAME[k])}, ${esc(D.groups[g])}: ${fmt(v)}</title></rect>`; x0 += v; });
    s += `<text x="${X(x0) + 4}" y="${y + 14}">${fmt(b.total[k])}</text>`;
  });
  host.innerHTML = s + "</svg>";
}

// bars per loudness rank, one per pairing
function byRank(host, b) {
  const M = {l: 50, r: 12, t: 10, b: 40}, H = 260, W = Math.max(320, host.clientWidth);
  const totals = Object.fromEntries(D.pairings.map(k => [k, b.by_rank[k].map(r => r.reduce((a, v) => a + v, 0))]));
  const hi = Math.max(...Object.values(totals).flat()) * 1.08, n = b.sources, P = D.pairings.length;
  const Y = v => M.t + (hi - v) / hi * (H - M.t - M.b), cw = (W - M.l - M.r) / n, bw = cw * 0.8 / P;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="path length per loudness rank">`;
  ticks(hi, 6).forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  for (let r = 0; r < n; r++) {
    s += `<text x="${M.l + (r + 0.5) * cw}" y="${H - M.b + 16}" text-anchor="middle">${r + 1}</text>`;
    D.pairings.forEach((k, i) => { const v = totals[k][r], x = M.l + r * cw + cw * 0.1 + i * bw; s += `<rect x="${x}" y="${Y(v)}" width="${bw - 2}" height="${Y(0) - Y(v)}" fill="${PCOL[k]}" rx="2"><title>${esc(PNAME[k])}, rank ${r + 1}: ${fmt(v)}</title></rect>`; });
  }
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">target source's loudness rank in its window</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">squared path length</text>`;
  host.innerHTML = s + "</svg>";
}

function render() {
  segHost.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === S.band)));
  const b = D.bands[S.band], c = b.total.f0;
  const tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  document.getElementById("tiles").innerHTML = [
    tile("f0 order (training)", fmt(b.total.f0), `± ${fmt(b.total_se.f0, 2)}, ${b.windows.toLocaleString("en")} windows of ${b.sources} sources`),
    tile("Nearest labelling", pct(b.total.nearest / c - 1), `against f0 order (${fmt(b.total.nearest)})`),
    tile("Loudest first", pct(b.total.snr / c - 1), `against f0 order (${fmt(b.total.snr)})`),
    tile("f0 order is the nearest", (100 * b.f0_is_nearest).toFixed(1) + "%", `of windows; it is loudest first in ${(100 * b.f0_is_snr_order).toFixed(1)}%`),
  ].join("");
  legend("lg-groups", D.groups.map((g, i) => [GCOL[i], g]));
  stacked(document.getElementById("p-groups"), b);
  document.getElementById("t-groups").innerHTML = `<table class="tbl"><tr><th>pairing</th>${D.groups.map(g => `<th>${esc(g)}</th>`).join("")}<th>total</th><th>per window / f0 order, 10–50–90%</th></tr>` +
    D.pairings.map(k => `<tr><td>${esc(PNAME[k])}</td>${b.mean[k].map(v => `<td>${fmt(v)}</td>`).join("")}<td>${fmt(b.total[k])}</td><td>${b.ratio_quantiles[k].map(v => v.toFixed(2)).join(" – ")}</td></tr>`).join("") + `</table>`;
  legend("lg-rank", D.pairings.map(k => [PCOL[k], PNAME[k]]));
  byRank(document.getElementById("p-rank"), b);
}
addEventListener("resize", render);
render();
</script>
</body>
</html>
"""

MEANS = """
<ul>
<li><b>f<sub>0</sub> order is the optimal coupling of f<sub>0</sub> alone</b> (the monotone rearrangement), so no two paths cross in f<sub>0</sub>, and it costs a sort. The coordinates it ignores are paired at random.</li>
<li><b>The nearest labelling is the shortest pairing by construction</b>: it minimises the same sum over all S! labellings, at the cost of an assignment per draw. The page measures how much shorter it is, and in which coordinates.</li>
<li><b>Shorter, less crossed paths are an easier target.</b> The straight-path velocity field averages over the paths through each point; paths of exchanged sources crossing each other make that average blur two targets.</li>
</ul>
"""


def main() -> None:
    data = json.loads((REPORT / "data" / "pairing.json").read_text())
    report(data)
    page = PAGE.replace("__MEANS__", MEANS).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = REPORT / "source-pairing.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
