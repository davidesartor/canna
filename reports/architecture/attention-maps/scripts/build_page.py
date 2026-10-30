"""Write attention-maps.html from data/*.json: where each source slot's attention goes, per block and head, against t.

Stdlib only; run from the repo root:
    uv run --frozen --no-sync python reports/architecture/attention-maps/scripts/build_page.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).parents[1]
# runs shorter than this are pipeline checks, never findings
SMOKE_STEPS = 50_000
REFERENCE_COMPARISON = "../../posterior-eval/reference-comparison/"
DATA_DEPENDENCE = "../data-dependence/"


def means(runs: list[dict]) -> str:
    """The closing card: what the measured runs can and cannot say yet."""
    trained = [r for r in runs if not r["smoke"]]
    status = (
        "<li><b>No trained run has been measured yet.</b> "
        + ", ".join(f"<code>{r['run']}</code> ({r['steps']:,} steps)" for r in runs)
        + " only checks the pipeline end to end; its maps belong to a network that has barely trained and say nothing about the architecture.</li>"
        if not trained
        else "<li>Measured trained runs: "
        + ", ".join(f"<code>{r['run']}</code> ({r['steps']:,} steps)" for r in trained)
        + ". Read the cards for those; smoke runs stay labelled.</li>"
    )
    return f"""<ul>
{status}
<li><b>Own-band share well above its uniform baseline</b>, rising with t and depth, means slots find their own sources through the f0 RoPE: the attention does the source separation. Near-baseline share in every block means each slot reads the window as a whole, and separation must happen in the MLPs or not at all.</li>
<li><b>Most of the mass on the sinks</b> in a block means that block's attention is mostly switched off for the sources (the sinks are where a head parks unused mass). A block that is all sink at every t is a candidate to drop; one that turns on only late is resolving the posterior's last decades.</li>
<li><b>Low data entropy</b> (well under 1) is a head that reads a few tokens; entropy near 1 reads all of them evenly. Heads that stay at 1 at every t carry no data selection.</li>
<li>Whether that attention matters for the output is the <a href="{DATA_DEPENDENCE}">data-dependence</a> report's question (the velocity's sensitivity to the data), and whether the output is right is <a href="{REFERENCE_COMPARISON}">reference comparison</a>'s.</li>
</ul>"""


def main() -> None:
    runs = []
    for path in sorted((REPORT / "data").glob("*.json")):
        r = json.loads(path.read_text())
        r["tag"] = path.stem
        r["smoke"] = r["steps"] < SMOKE_STEPS
        runs.append(r)
    page = REPORT / "attention-maps.html"
    page.write_text(
        PAGE.replace("__DATA__", json.dumps(runs))
        .replace("__DEP__", DATA_DEPENDENCE)
        .replace("__MEANS__", means(runs) if runs else "<p>No run measured yet.</p>")
    )
    print(f"{page}: {len(runs)} run(s): {', '.join(r['tag'] for r in runs)}")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA attention maps</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e; --warn-bg: #fdf3dc; --warn: #7a5200;
  --c0: #2a78d6; --c1: #eb6834; --c2: #1baf7a; --c3: #eda100;
  --heat: 42, 120, 214;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7; --warn-bg: #3a2c0a; --warn: #f0c870;
    --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500;
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
<h1>Where do the source slots attend?</h1>
<p class="sub">Each block's softmax for the source-slot queries of a trained LisaFlow at one network call, recomputed from the block's own q and k on fresh training observations at fixed path times t. Keys are the source slots, the sink tokens and the data tokens; what the data tokens are worth to the output is in <a href="__DEP__">data dependence</a>.</p>
<div class="bar">
  <div><span class="k">Run</span><span class="seg" id="run"></span></div>
  <div><span class="k">t</span><span class="seg" id="t"></span></div>
  <div><span class="k">Head</span><span class="seg" id="head"></span></div>
</div>
<div id="banner"></div>
<p class="meta" id="meta"></p>
<div class="card">
  <h2>Where does a slot's attention go, block by block?</h2>
  <p class="note">At the selected t: each block's attention mass from a slot to itself, to the other slots, to the sinks and to the data tokens. Mean over observations, slots and the selected heads.</p>
  <div class="legend" id="lg-mass"></div>
  <div id="p-mass"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Does each slot attend to its own source's tokens?</h2>
  <p class="note">Of a slot's attention to the data, the share on the tokens that hold <span id="bp"></span> of its own source's power, against t, one line per block (darker is deeper). Dashed: the share uniform attention would give (those tokens' fraction of the window).</p>
  <div class="controls"><div><span class="k">Share on</span><span class="seg" id="part"></span></div></div>
  <div class="legend" id="lg-own"></div>
  <div id="p-own"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How much attention goes to the sinks, and when?</h2>
  <p class="note">The slots' attention mass on the sink tokens against t, one line per block.</p>
  <div class="legend" id="lg-sink"></div>
  <div id="p-sink"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>How focused is each head?</h2>
  <p class="note">Entropy of a slot's attention row over its keys, divided by the log of their number (1: uniform; 0: one key), at the selected t; mean over observations and slots. Darker is more focused.</p>
  <div class="controls"><div><span class="k">Row</span><span class="seg" id="ent"></span></div></div>
  <div id="p-ent"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Does a louder source get more of its slot's attention?</h2>
  <p class="note">Own-band share of the data attention at the selected t by the source's SNR (all slots pooled), one line per block; selected heads.</p>
  <div class="legend" id="lg-snr"></div>
  <div id="p-snr"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What does one observation's attention look like?</h2>
  <p class="note">One observation at the selected t and block: each slot's attention per data token, mean over heads (darker is more, scaled to the block's largest weight). Ticks above a row mark its own source's tokens, the triangle its f0 token.</p>
  <div class="controls"><div><span class="k">Block</span><span class="seg" id="block"></span></div></div>
  <div id="p-example"></div><div class="tip"></div>
</div>
<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const RUNS = __DATA__;
const S = {run: null, t: 3, head: "mean", part: 0, ent: "data", block: 0};
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const f3 = v => v == null || !isFinite(v) ? "—" : v.toFixed(3);
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const sw = (color, dash) => `<svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="${color}" stroke-width="2" stroke-dasharray="${dash || ""}"/></svg>`;
const box = color => `<svg width="12" height="12"><rect x="1" y="1" width="10" height="10" rx="2" fill="${color}"/></svg>`;
const legend = (id, items) => document.getElementById(id).innerHTML = items.map(([c, d, t, kind]) => `<span>${kind === "box" ? box(c) : sw(c, d)}${t}</span>`).join("");
// depth as one hue, light (first block) to dark (last)
const depth = (l, L) => `rgba(var(--heat), ${(0.3 + 0.7 * (L === 1 ? 1 : l / (L - 1))).toFixed(2)})`;
function seg(id, items, key, numeric) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === String(S[key])}">${text}</button>`).join("");
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = numeric ? Number(b.dataset.v) : b.dataset.v; draw(); }));
}
document.addEventListener("mousemove", e => {
  document.querySelectorAll(".tip").forEach(t => { t.style.display = "none"; });
  const t = e.target.closest("[data-tip]"), card = t && t.closest(".card");
  if (!t || !card) return;
  const tip = card.querySelector(".tip"), b = card.getBoundingClientRect();
  tip.innerHTML = t.dataset.tip;
  tip.style.display = "block";
  tip.style.left = Math.min(e.clientX - b.left + 14, b.width - tip.offsetWidth - 8) + "px";
  tip.style.top = e.clientY - b.top + 14 + "px";
});
const MG = {l: 52, r: 16, t: 10, b: 38};
function axes(o, W, H, X, Y) {
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(o.label)}">`;
  o.ticks.forEach(v => s += `<line x1="${MG.l}" x2="${W - MG.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${MG.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  o.x.forEach((v, i) => s += `<text x="${X(i)}" y="${H - MG.b + 16}" text-anchor="middle">${v}</text>`);
  s += `<line x1="${MG.l}" x2="${W - MG.r}" y1="${H - MG.b}" y2="${H - MG.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(MG.l + W - MG.r) / 2}" y="${H - 4}" text-anchor="middle">${o.xlabel}</text>`;
  return s + `<text transform="translate(12 ${(MG.t + H - MG.b) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
}
// categorical x, linear y in [0, 1]; series: {color, dash, pts: [[i, y, tip]]}
function chart(id, o) {
  const host = document.getElementById(id), W = Math.max(300, host.clientWidth), H = 260;
  const n = o.x.length, X = i => MG.l + (n === 1 ? 0.5 : i / (n - 1)) * (W - MG.l - MG.r);
  const Y = v => MG.t + (1 - Math.min(1, Math.max(0, v))) * (H - MG.t - MG.b);
  let s = axes(o, W, H, X, Y);
  if (o.mark != null) s += `<line x1="${X(o.mark)}" x2="${X(o.mark)}" y1="${MG.t}" y2="${H - MG.b}" stroke="var(--ref)" stroke-dasharray="3 3"/>`;
  o.series.forEach(se => {
    let d = "", gap = true;
    se.pts.forEach(p => { const ok = p[1] != null && isFinite(p[1]); if (ok) d += `${gap ? "M" : "L"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`; gap = !ok; });
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2" stroke-dasharray="${se.dash || ""}"/>`;
    if (!se.dash) se.pts.filter(p => p[1] != null && isFinite(p[1])).forEach(p => s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="${se.color}" stroke="var(--surface)" stroke-width="2"/><circle cx="${X(p[0])}" cy="${Y(p[1])}" r="10" fill="transparent" data-tip="${esc(p[2])}"/>`);
  });
  host.innerHTML = s + "</svg>";
}
// stacked bars in [0, 1]: rows [[label, [v...]]], parts [[name, color]]
function stacked(id, o) {
  const host = document.getElementById(id), W = Math.max(300, host.clientWidth), H = 260, n = o.rows.length;
  const step = (W - MG.l - MG.r) / n, bw = Math.min(56, step * 0.7), X = i => MG.l + (i + 0.5) * step;
  const Y = v => MG.t + (1 - v) * (H - MG.t - MG.b);
  let s = axes({...o, x: o.rows.map(r => r[0])}, W, H, X, Y);
  o.rows.forEach(([label, vals], i) => {
    let acc = 0;
    vals.forEach((v, j) => {
      const y0 = Y(acc + v), h = Math.max(0, Y(acc) - Y(acc + v) - 2);
      s += `<rect x="${X(i) - bw / 2}" y="${y0}" width="${bw}" height="${h}" rx="2" fill="${o.parts[j][1]}" data-tip="${esc(`<b>block ${label}</b>: ` + o.parts.map(([nm], k) => `${nm} ${f3(vals[k])}`).join(", "))}"/>`;
      acc += v;
    });
  });
  host.innerHTML = s + "</svg>";
}
const headPick = arr => S.head === "mean" ? mean(arr) : arr[Number(S.head)];
// [L][H][S] -> per block, slot-mean of the selected heads
const perBlock = (A, f = v => v) => A.map(hs => headPick(hs.map(ss => mean(ss.map(f)))));
function draw() {
  const R = RUNS.find(r => r.tag === S.run), L = R.blocks, ti = Math.min(S.t, R.t_grid.length - 1), P = R.per_t[ti];
  const tl = R.t_grid.map(String), blocks = Array.from({length: L}, (_, l) => l);
  seg("run", RUNS.map(r => [r.tag, r.tag + (r.smoke ? " (smoke)" : "")]), "run");
  seg("t", R.t_grid.map((t, i) => [i, String(t)]), "t", true);
  seg("head", [["mean", "mean"], ...Array.from({length: R.heads}, (_, h) => [h, String(h)])], "head");
  seg("part", [[0, "own band"], [1, "other sources' bands"]], "part", true);
  seg("ent", [["data", "data tokens only"], ["all", "all keys"]], "ent");
  seg("block", blocks.map(l => [l, String(l)]), "block", true);
  document.getElementById("banner").innerHTML = R.smoke
    ? `<div class="banner"><b>Smoke run: ${R.steps.toLocaleString()} of ${R.total_steps.toLocaleString()} steps.</b> A pipeline check, not a finding: these maps belong to a barely trained network.</div>` : "";
  document.getElementById("meta").textContent = `${R.config}, commit ${R.commit.slice(0, 7)}, ${R.weights} weights at epoch ${R.epoch} (${R.steps.toLocaleString()} steps), bf16 ${R.bf16}; ${R.blocks} blocks × ${R.heads} heads, ${R.slots} slots, ${R.sinks} sinks, ${R.tokens} data tokens of ${R.patch} bins; ${R.observations} observations per t on ${R.device} in ${R.seconds} s. Recomputed weights reproduce each block's x stream to ${Math.max(...R.check_block).toExponential(1)} of its largest entry, and the tapped flow the plain one to ${R.check_output.toExponential(1)} of the output's (bf16 rounding is ~4e-3).`;
  document.getElementById("bp").textContent = Math.round(100 * R.band_power) + "%";
  const blockLegend = id => legend(id, [...blocks.map(l => [depth(l, L), "", `block ${l}`])]);

  // mass per block
  const parts = [["self", "var(--c0)"], ["other slots", "var(--c1)"], ["sinks", "var(--c2)"], ["data", "var(--c3)"]];
  legend("lg-mass", parts.map(([n, c]) => [c, "", n, "box"]));
  stacked("p-mass", {label: "attention mass per block", xlabel: `block (t = ${R.t_grid[ti]})`, ylabel: "attention mass", ticks: [0, 0.25, 0.5, 0.75, 1], parts,
    rows: blocks.map(l => [String(l), [0, 1, 2, 3].map(j => headPick(P.mass[l].map(ss => mean(ss.map(m => m[j])))))])});

  // own-band share vs t
  const base = mean(R.baseline.map(b => b[S.part])), pname = S.part ? "other sources' bands" : "own band";
  legend("lg-own", [...blocks.map(l => [depth(l, L), "", `block ${l}`]), ["var(--ref)", "4 3", "uniform"]]);
  chart("p-own", {label: "own-band share vs t", x: tl, xlabel: "path time t", ylabel: `data attention on ${pname}`, ticks: [0, 0.25, 0.5, 0.75, 1], mark: ti,
    series: [{color: "var(--ref)", dash: "4 3", pts: tl.map((_, k) => [k, base, ""])},
      ...blocks.map(l => ({color: depth(l, L), pts: R.per_t.map((p, k) => { const v = perBlock(p.split, s => s[S.part])[l]; return [k, v, `<b>block ${l}</b>, t ${p.t}: ${f3(v)} on ${pname} (uniform ${f3(base)})`]; })}))]});

  // sink mass vs t
  blockLegend("lg-sink");
  chart("p-sink", {label: "sink mass vs t", x: tl, xlabel: "path time t", ylabel: "attention mass on sinks", ticks: [0, 0.25, 0.5, 0.75, 1], mark: ti,
    series: blocks.map(l => ({color: depth(l, L), pts: R.per_t.map((p, k) => { const v = perBlock(p.mass, m => m[2])[l]; return [k, v, `<b>block ${l}</b>, t ${p.t}: ${f3(v)} on the sinks`]; })}))});

  // entropy heatmap
  const E = S.ent === "data" ? P.data_entropy : P.entropy, host = document.getElementById("p-ent");
  const W = Math.max(300, host.clientWidth), lft = 64, cw = Math.min(64, (W - lft - 8) / R.heads), ch = 24, H = L * ch + 34;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="entropy per block and head">`;
  E.forEach((hs, l) => {
    s += `<text x="${lft - 6}" y="${l * ch + ch / 2 + 4}" text-anchor="end">block ${l}</text>`;
    hs.forEach((ss, h) => { const v = mean(ss); s += `<rect x="${lft + h * cw + 1}" y="${l * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="2" fill="rgba(var(--heat), ${(1 - v).toFixed(3)})" stroke="var(--grid)" data-tip="<b>block ${l}, head ${h}</b>: entropy ${f3(v)}"/>`; });
  });
  for (let h = 0; h < R.heads; h++) s += `<text x="${lft + (h + 0.5) * cw}" y="${L * ch + 14}" text-anchor="middle">${h}</text>`;
  s += `<text x="${lft + R.heads * cw / 2}" y="${H - 2}" text-anchor="middle">head</text></svg>`;
  host.innerHTML = s;

  // own-band share by SNR
  const edges = R.snr_edges, bins = edges.slice(0, -1).map((e, i) => `${e}–${edges[i + 1]}`);
  blockLegend("lg-snr");
  chart("p-snr", {label: "own-band share by SNR", x: bins, xlabel: `source SNR (t = ${R.t_grid[ti]})`, ylabel: "data attention on own band", ticks: [0, 0.25, 0.5, 0.75, 1],
    series: blocks.map(l => ({color: depth(l, L), pts: P.split_snr.map((g, k) => { const v = g ? headPick(g[l]) : null; return [k, v, `<b>block ${l}</b>, SNR ${bins[k]}: ${f3(v)}`]; })}))});

  // one observation's map
  const bl = Math.min(S.block, L - 1), ex = P.example[bl], X0 = R.example, hx = document.getElementById("p-example");
  const W2 = Math.max(300, hx.clientWidth), M = R.tokens, rowH = 22, band = 5, cw2 = (W2 - lft - 8) / M, peak = Math.max(1, ...ex.flat());
  const H2 = ex.length * (rowH + band + 8) + 24;
  let s2 = `<svg viewBox="0 0 ${W2} ${H2}" width="${W2}" role="img" aria-label="attention map">`;
  ex.forEach((row, si) => {
    const y0 = si * (rowH + band + 8) + 6;
    s2 += `<text x="${lft - 6}" y="${y0 + band + rowH / 2 + 4}" text-anchor="end">slot ${si}</text>`;
    X0.own[si].forEach((o, c) => { if (o) s2 += `<rect x="${lft + c * cw2}" y="${y0}" width="${Math.max(cw2 - 0.5, 1)}" height="${band - 1}" fill="var(--ink-2)"/>`; });
    const fx = lft + (X0.f0_token[si] + 0.5) * cw2;
    s2 += `<path d="M${fx - 4} ${y0 - 5}L${fx + 4} ${y0 - 5}L${fx} ${y0 + 1}Z" fill="var(--ink)"/>`;
    row.forEach((v, c) => s2 += `<rect x="${lft + c * cw2}" y="${y0 + band + 1}" width="${cw2 + 0.3}" height="${rowH}" fill="rgba(var(--heat), ${Math.sqrt(v / peak).toFixed(3)})" data-tip="slot ${si} (SNR ${X0.snr[si].toFixed(0)}), token ${c}: weight ${(v * R.example_unit).toFixed(4)}"/>`);
  });
  s2 += `<text x="${lft}" y="${H2 - 4}">token 0</text><text x="${W2 - 8}" y="${H2 - 4}" text-anchor="end">${M} tokens</text></svg>`;
  hx.innerHTML = s2;
}
if (!RUNS.length) document.querySelectorAll(".card:not(.means)").forEach(c => c.innerHTML = `<p class="empty">No run measured yet.</p>`);
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
