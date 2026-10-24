"""Write sampling-cost.html from data/: what one posterior costs from the flow against the MCMC reference.

Stdlib only; run from the repo root:
    uv run --frozen python reports/benchmarks/sampling-cost/scripts/build_page.py
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).parents[1]
CONFIGS = [f"{b}-{s}" for b in ("narrow", "full") for s in ("XS", "S", "B", "L", "XL")]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sig(v, n: int = 4):
    return None if v is None else float(f"{v:.{n}g}")


def flow() -> list[dict]:
    """The flow's sampler per config and draws, warm and with its first (compiling) call."""
    return [
        dict(
            config=r["config"],
            draws=r["draws"],
            params=r["params"],
            best=sig(r["best_s"]),
            first=sig(r["first_s"]),
        )
        for r in jsonl(HERE / "data/a100.jsonl")
    ]


def mcmc() -> dict:
    """Seconds per MCMC chain: per source, summed per window and seed, joint, and the convergence chains."""
    d = json.loads((HERE / "data/mcmc.json").read_text())
    window = defaultdict(float)
    for r in d["narrowband"]:
        window[r["inj"], r["seed"]] += r["seconds"]
    return dict(
        source=[sig(r["seconds"]) for r in d["narrowband"]],
        window=[sig(v) for v in window.values()],
        joint=[sig(r["seconds"]) for r in d["joint"]],
        convergence=[sig(r["seconds"]) for r in d["convergence"]],
    )


def report(data: dict) -> None:
    """Print the numbers the README quotes."""
    for r in sorted(
        data["flow"], key=lambda r: (r["draws"], CONFIGS.index(r["config"]))
    ):
        print(
            f"flow  {r['config']:9} {r['draws']:5} best {r['best']:.3g} s first {r['first']:.3g} s params {r['params'] / 1e6:.1f}M"
        )
    for k, v in data["mcmc"].items():
        print(
            f"mcmc  {k:12} n {len(v)} median {median(v) / 60:.1f} min range {min(v) / 60:.1f}-{max(v) / 60:.1f} min"
        )
    best = {(r["config"], r["draws"]): r["best"] for r in data["flow"]}
    for config in ("narrow-B", "full-B"):
        b = best[config, 256]
        print(
            f"ratio {config} 256: window {median(data['mcmc']['window']) / b:.0f}× joint {median(data['mcmc']['joint']) / b:.0f}×"
        )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA sampling cost</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e;
  --narrow: #2a78d6; --full: #eb6834; --cpu: #c98500;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7;
    --narrow: #3987e5; --full: #d95926; --cpu: #eda100;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7;
  --narrow: #3987e5; --full: #d95926; --cpu: #eda100;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; }
.calc { color: var(--ink-2); font-size: 14px; margin: 12px 0 0; max-width: 80ch; }
.calc b { color: var(--ink); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 12px; }
.bar .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.sl { display: block; font-size: 13px; color: var(--ink-2); max-width: 420px; margin: 0 0 16px; }
.sl .top { display: flex; justify-content: space-between; }
.sl .val { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
input[type=range] { width: 100%; accent-color: var(--narrow); margin: 4px 0 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: auto; display: inline-block; flex: none; overflow: visible; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
svg [data-act] { cursor: pointer; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; max-width: 300px; }
.tip b { color: var(--ink); font-weight: 600; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 80ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>What does one posterior cost, flow against MCMC?</h1>
<p class="sub">The flow samples a window's posterior on an A100 80 GB with the bf16 backbone: 32 ODE steps, x-prediction with self-conditioning, so 65 network calls per draw. The MCMC reference runs on 2 CPU cores per chain. The hardware differs: read the gap as the cost of a result, not as a like-for-like speedup.</p>
<div class="bar">
  <div><span class="k">Band</span><span class="seg" id="band"></span></div>
  <div><span class="k">Size</span><span class="seg" id="size"></span></div>
  <div><span class="k">Draws</span><span class="seg" id="draws"></span></div>
</div>
<label class="sl"><span class="top"><span>Windows to sample</span><span class="val" id="v-win"></span></span><input type="range" id="win"></label>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>How does the flow's cost grow with the network?</h2>
  <p class="note">Seconds per window, warm (best of 3), against the network's parameters, per band; solid at 256 draws, dashed at 1024. Click a point to select its config.</p>
  <div class="legend" id="lg-flow"></div>
  <div id="p-flow"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Flow or MCMC: how long until a window's posterior?</h2>
  <p class="note">Seconds per window on a log axis. The flow per config at the chosen draws; the MCMC reference per source, summed over a window's 4 sources, joint over the window, and the convergence report's long joint chains, as the median with a min–max whisker. Click a flow bar to select its config.</p>
  <div id="p-samp"></div><div class="tip"></div>
  <p class="calc" id="calc"></p>
</div>
<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const SIZES = ["XS", "S", "B", "L", "XL"];
const COLOR = {narrow: "var(--narrow)", full: "var(--full)"};
const WINS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
const S = {band: "full", size: "B", draws: "256", win: 100};
const marks = {};

const cfg = () => `${S.band}-${S.size}`;
const fmtS = v => v >= 86400 ? (v / 86400).toFixed(1) + " d" : v >= 3600 ? (v / 3600).toFixed(1) + " h" : v >= 60 ? (v / 60).toFixed(v >= 600 ? 0 : 1) + " min" : v >= 10 ? v.toFixed(1) + " s" : v >= 1 ? v.toFixed(2) + " s" : (1e3 * v).toFixed(0) + " ms";
const fmtM = v => v >= 1e9 ? (v / 1e9).toFixed(1) + "B" : (v / 1e6).toFixed(v >= 1e8 ? 0 : 1) + "M";
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const act = o => esc(JSON.stringify(o));
const median = v => { const s = [...v].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const flowOf = (config, draws) => D.flow.find(r => r.config === config && r.draws === Number(draws));

// controls
function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; schedule(); }));
}
seg("band", [["narrow", "narrow"], ["full", "full"]], "band");
seg("size", SIZES.map(s => [s, s]), "size");
seg("draws", [["256", "256"], ["1024", "1024"]], "draws");
const winInput = document.getElementById("win");
winInput.min = 0; winInput.max = WINS.length - 1;
winInput.addEventListener("input", () => { S.win = WINS[Number(winInput.value)]; schedule(); });

function writeHash() {
  history.replaceState(null, "", "#" + new URLSearchParams({cfg: cfg(), draws: S.draws, win: S.win}).toString());
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const [band, size] = (h.get("cfg") || "").split("-"); if (["narrow", "full"].includes(band) && SIZES.includes(size)) { S.band = band; S.size = size; }
  if (["256", "1024"].includes(h.get("draws"))) S.draws = h.get("draws");
  if (WINS.includes(Number(h.get("win")))) S.win = Number(h.get("win"));
  if (["light", "dark"].includes(h.get("theme"))) document.documentElement.dataset.theme = h.get("theme");
}

// clicks and hover on anything carrying data-act / data-tip
document.addEventListener("click", e => {
  const t = e.target.closest("[data-act]");
  if (!t) return;
  const a = JSON.parse(t.dataset.act);
  if (a.config) { [a.band, a.size] = a.config.split("-"); delete a.config; }
  Object.assign(S, a);
  schedule();
});
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

function tiles() {
  const tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  const fl = flowOf(cfg(), S.draws), mw = median(D.mcmc.window), mj = median(D.mcmc.joint);
  document.getElementById("tiles").innerHTML = [
    tile("Flow, one window", fmtS(fl.best), `${cfg()}, ${S.draws} draws, warm on an A100 80 GB`),
    tile("First call", fmtS(fl.first), "compiles the sampler once per process"),
    tile("MCMC per source", fmtS(mw), `a window's 4 sources, median: ${Math.round(mw / fl.best).toLocaleString("en")}× the flow`),
    tile("Joint MCMC", fmtS(mj), `one window, median: ${Math.round(mj / fl.best).toLocaleString("en")}× the flow`),
  ].join("");
}

// log-log chart: seconds per window against parameters, per band and draws
const M = {l: 56, r: 16, t: 10, b: 40};
function flowPanel() {
  const host = document.getElementById("p-flow"), W = Math.max(300, host.clientWidth), H = 280;
  const x0 = 1e7, x1 = 1e9, y0 = 0.05, y1 = 5;
  const X = v => M.l + (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - M.l - M.r);
  const Y = v => M.t + (Math.log10(y1) - Math.log10(v)) / (Math.log10(y1) - Math.log10(y0)) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="flow sampling time against parameters">`;
  [0.05, 0.1, 0.2, 0.5, 1, 2, 5].forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${fmtS(v)}</text>`);
  [1e7, 3e7, 1e8, 3e8, 1e9].forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${fmtM(v)}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">parameters</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">seconds per window</text>`;
  for (const band of ["narrow", "full"]) for (const draws of ["256", "1024"]) {
    const pts = SIZES.map(z => flowOf(`${band}-${z}`, draws)).filter(Boolean);
    const sel = band === S.band && draws === S.draws;
    s += `<path d="${pts.map((r, i) => `${i ? "L" : "M"}${X(r.params).toFixed(1)},${Y(r.best).toFixed(1)}`).join("")}" fill="none" stroke="${COLOR[band]}" stroke-width="${sel ? 2.4 : 1.3}" stroke-dasharray="${draws === "1024" ? "6 3" : ""}" opacity="${sel ? 1 : 0.55}"/>`;
    for (const r of pts) {
      const on = r.config === cfg() && draws === S.draws;
      s += `<circle cx="${X(r.params)}" cy="${Y(r.best)}" r="${on ? 5 : 3.5}" fill="${on ? COLOR[band] : "var(--surface)"}" stroke="${COLOR[band]}" stroke-width="1.5" data-tip="${esc(`<b>${r.config}, ${draws} draws</b><br>${fmtM(r.params)} params<br>${fmtS(r.best)} per window, first call ${fmtS(r.first)}`)}" data-act="${act({config: r.config, draws})}"/>`;
    }
  }
  host.innerHTML = s + "</svg>";
  const sw = (c, dash) => `<svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="${c}" stroke-width="2" stroke-dasharray="${dash}"/></svg>`;
  document.getElementById("lg-flow").innerHTML = `<span>${sw(COLOR.narrow, "")}narrow</span><span>${sw(COLOR.full, "")}full</span><span>${sw("var(--ink-2)", "")}256 draws</span><span>${sw("var(--ink-2)", "6 3")}1024 draws</span>`;
}

function sampPanel() {
  const host = document.getElementById("p-samp"), mc = D.mcmc;
  const flow = D.configs.map(c => flowOf(c, S.draws)).filter(Boolean);
  const mrows = [["MCMC, one source", mc.source, "per source and seed"], ["MCMC, a window's 4 sources", mc.window, "sum of the 4 per-source chains, per window and seed"], ["joint MCMC, one window", mc.joint, "all 4 sources at once, per window and seed"], ["joint MCMC, convergence chains", mc.convergence, "the convergence report's longer joint chains"]];
  const W = Math.max(300, host.clientWidth), rowH = 22, L0 = Math.min(200, W * 0.42), H = (flow.length + mrows.length) * rowH + 60;
  const all = flow.map(r => r.best).concat(...mrows.map(m => m[1]));
  const lo = Math.min(...all) / 1.5, hi = Math.max(...all) * 1.5;
  const X = v => L0 + (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (W - L0 - 70);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="posterior cost">`;
  [0.1, 1, 10, 60, 600, 3600, 36000].filter(v => v > lo && v < hi).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="4" y2="${H - 30}" stroke="var(--grid)"/><text x="${X(v)}" y="${H - 14}" text-anchor="middle">${fmtS(v)}</text>`);
  flow.forEach((r, i) => {
    const y = 6 + i * rowH, on = r.config === cfg(), band = r.config.split("-")[0];
    const tip = `<b>flow, ${r.config}, ${r.draws} draws</b><br>${fmtS(r.best)} per window (best of 3)<br>first call with compile: ${fmtS(r.first)}`;
    s += `<text class="lbl" x="${L0 - 8}" y="${y + 13}" text-anchor="end" font-weight="${on ? 700 : 400}">flow ${r.config}</text>`;
    s += `<rect x="${L0}" y="${y + 2}" width="${X(r.best) - L0}" height="14" fill="${COLOR[band]}" opacity="${on ? 1 : 0.45}" data-tip="${esc(tip)}" data-act="${act({config: r.config})}"/>`;
    s += `<text x="${X(r.best) + 5}" y="${y + 13}">${fmtS(r.best)}</text>`;
  });
  mrows.forEach(([label, v, what], j) => {
    const y = 6 + (flow.length + j) * rowH + 12, med = median(v);
    const tip = `<b>${label}</b><br>${what}, 2 CPU cores each<br>median ${fmtS(med)}, ${fmtS(Math.min(...v))} to ${fmtS(Math.max(...v))} over ${v.length}`;
    s += `<text class="lbl" x="${L0 - 8}" y="${y + 13}" text-anchor="end">${label}</text>`;
    s += `<rect x="${L0}" y="${y + 2}" width="${X(med) - L0}" height="14" fill="var(--cpu)" opacity="0.8" data-tip="${esc(tip)}"/>`;
    s += `<line x1="${X(Math.min(...v))}" x2="${X(Math.max(...v))}" y1="${y + 9}" y2="${y + 9}" stroke="var(--ink-2)" stroke-width="1.5" pointer-events="none"/>`;
    s += `<text x="${X(Math.max(...v)) + 5}" y="${y + 13}">${fmtS(med)}</text>`;
  });
  host.innerHTML = s + "</svg>";
  const fl = flowOf(cfg(), S.draws), n = S.win, flowT = fl.first + (n - 1) * fl.best;
  const mw = median(mc.window), mj = median(mc.joint);
  document.getElementById("calc").innerHTML = `${n.toLocaleString("en")} window${n > 1 ? "s" : ""} with ${cfg()}: flow <b>${fmtS(flowT)}</b> on one A100 (the first call compiles, ${fmtS(fl.first)}), against <b>${fmtS(n * mw)}</b> of 2-core time for per-source MCMC and <b>${fmtS(n * mj)}</b> for joint MCMC (medians, one chain each): ${Math.round(n * mw / flowT).toLocaleString("en")}× and ${Math.round(n * mj / flowT).toLocaleString("en")}× the flow.`;
}

function render() {
  for (const key of Object.keys(marks)) marks[key]();
  winInput.value = WINS.indexOf(S.win);
  document.getElementById("v-win").textContent = S.win.toLocaleString("en");
  tiles(); flowPanel(); sampPanel();
  writeHash();
}
let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}
addEventListener("resize", schedule);
addEventListener("hashchange", () => { readHash(); schedule(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
new MutationObserver(schedule).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
readHash();
render();
</script>
</body>
</html>
"""

MEANS = """
<ul>
<li><b>A posterior from the flow costs a fraction of a second.</b> 256 draws of one window take 0.22 s (narrow-B) and 0.25 s (full-B), warm on an A100 80 GB, linear in draws, and 9–10× the time from XS to XL for 51× the parameters.</li>
<li><b>Compiling dominates a handful of windows.</b> The first call takes 27–37 s; after that the flow does a window in under a second at every size.</li>
<li><b>MCMC costs tens of minutes to hours per window.</b> 19–28 min for a window's 4 per-source chains and 3.4–5.7 h joint, on 2 CPU cores each: the medians are 5,800–6,500× and 59,000–66,000× the flow at size B, as the cost of a result rather than a like-for-like speedup.</li>
</ul>
"""


def main() -> None:
    data = dict(configs=CONFIGS, flow=flow(), mcmc=mcmc())
    report(data)
    page = PAGE.replace("__MEANS__", MEANS).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = HERE / "sampling-cost.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
