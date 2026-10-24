"""Write training-overhead.html from data/: what restarts and checkpoints cost a 24 h training job.

Stdlib only; run from the repo root:
    uv run --frozen python reports/benchmarks/training-overhead/scripts/build_page.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE / "scripts"))
import overhead as model  # noqa: E402

CONFIGS = [f"{b}-{s}" for b in ("narrow", "full") for s in ("XS", "S", "B", "L", "XL")]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sig(v, n: int = 4):
    return None if v is None else float(f"{v:.{n}g}")


def defaults() -> dict:
    """The trainer's log_interval and total_steps, read from config.py and the band yamls without importing canna."""
    text = (REPO / "src/canna/lisa/config.py").read_text()
    log_interval = int(
        re.search(r"^\s+log_interval: int = ([\d_]+)", text, re.M)[1].replace("_", "")
    )
    total = {
        band: int(
            re.search(
                r"^total_steps: (\d+)",
                (REPO / f"src/canna/lisa/configs/band/{band}.yaml").read_text(),
                re.M,
            )[1]
        )
        for band in ("narrow", "full")
    }
    return dict(log_interval=log_interval, total=total)


def overhead_rows() -> list[dict]:
    """One trainer-loop record per card, precision and config, with its chunk-model costs."""
    paths = sorted((HERE / "data").glob("*.jsonl"))
    # a record that ran after the other band's configs in a shared process reused their compiles
    after_other_band = set()
    for path in paths:
        if path.stem.endswith("-alone"):
            continue
        seen = set()
        for r in jsonl(path):
            band = r["config"].split("-")[0]
            if seen - {band}:
                after_other_band.add((r["job"], r["config"]))
            seen.add(band)
    rows = []
    for (config, series), r in model.latest(paths).items():
        card, prec = series.split(" · bf16 ")
        row = dict(
            card=card,
            prec=prec,
            config=config,
            job=r["job"],
            plan=sig(r.get("plan_gib")),
            lower=(r["job"], config) in after_other_band,
        )
        if r.get("oom"):
            rows.append(row | dict(oom=r["oom"].split(":")[0]))
            continue
        c = model.costs(r)
        rows.append(
            row
            | dict(
                c=dict(
                    step=sig(c["step"]),
                    R=sig(c["R"]),
                    restart={k: sig(v) for k, v in c["restart"].items()},
                    block=sig(c["block"]),
                    background=sig(c["background"]),
                    contention=sig(c["contention"]),
                    diagnostics=sig(c["diagnostics"]),
                    history={k: sig(v) for k, v in c["history"].items()},
                    total=c["total"],
                    gib=sig(c["gib"]),
                    state_gib=sig(r["state_gib"]),
                )
            )
        )
    return rows


def report(data: dict) -> None:
    """Print the numbers the README quotes."""
    d = data["defaults"]
    print("defaults", d)
    for o in data["overhead"]:
        if "oom" in o:
            print(
                f"loop  {o['card']:11} {o['prec']:9} {o['config']:9} OOM in {o['oom']} plan {o['plan']}"
            )
            continue
        c = dict(o["c"], history={int(k): v for k, v in o["c"]["history"].items()})
        at = model.chunk(c, d["log_interval"])
        best = min(
            (L for L in model.NICE if L <= c["total"]),
            key=lambda L: model.chunk(c, L)["lost"],
        )
        share = c["restart"]["compiles"] / c["R"]
        wall = c["total"] * c["step"] / (1 - at["lost"]) / 86400
        print(
            f"loop  {o['card']:11} {o['prec']:9} {o['config']:9} step {c['step']:.4g} R {c['R']:.0f}{'≥' if o['lower'] else ''} compiles {100 * share:.0f}% "
            f"lost@{d['log_interval']} {100 * at['lost']:.2f}% best {best} {100 * model.chunk(c, best)['lost']:.2f}% per-epoch {at['per_epoch']:.2f} "
            f"block {c['block']:.2f} write {c['background']:.1f} wall {wall:.3g} d ckpt {c['gib']:.2f} GiB"
        )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA training overhead</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e; --s1: #2a78d6;
  --a80: #2a78d6; --a40: #7a5bd6; --l40s: #eb6834; --bad: #c2362b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7; --s1: #3987e5;
    --a80: #3987e5; --a40: #9a80ee; --l40s: #d95926; --bad: #e5675c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7; --s1: #3987e5;
  --a80: #3987e5; --a40: #9a80ee; --l40s: #d95926; --bad: #e5675c;
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
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 12px; }
.bar .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.sliders { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px 24px; margin: 0 0 16px; }
.sl { display: block; font-size: 13px; color: var(--ink-2); }
.sl .top { display: flex; justify-content: space-between; }
.sl .val { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
.sl .sn { color: var(--muted); font-size: 12px; }
input[type=range] { width: 100%; accent-color: var(--s1); margin: 4px 0 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.tile .v.na { color: var(--muted); font-weight: 500; }
.tile .v.bad { color: var(--bad); }
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
.cov { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.cov th, .cov td { padding: 4px 5px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
.cov th:first-child, .cov td:first-child { text-align: left; }
.cov th { color: var(--ink-2); font-weight: 600; }
.cov td[data-act] { cursor: pointer; }
.cov td.na { color: var(--muted); }
.cov td.oom { color: var(--bad); }
.cov td.sel { outline: 2px solid var(--ink); outline-offset: -2px; border-radius: 4px; }
.cov td small { display: block; color: var(--muted); font-size: 11px; }
.scroll { overflow-x: auto; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 80ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>What do restarts and checkpoints cost a training run?</h1>
<p class="sub">Training runs as a chain of 24 h jobs. Each job pays a restart (uv, imports, a cold restore, compiles), a per-epoch cost every log interval (diagnostics, <code>history.npz</code>, the checkpoint) and the half epoch the time limit kills. Measured in the trainer's own loop on its 2 CPUs at batch 256; the chunk model adds up the parts.</p>
<div class="bar">
  <div><span class="k">Card</span><span class="seg" id="card"></span></div>
  <div><span class="k">Precision</span><span class="seg" id="prec"></span></div>
  <div><span class="k">Band</span><span class="seg" id="band"></span></div>
  <div><span class="k">Size</span><span class="seg" id="size"></span></div>
</div>
<div class="sliders">
  <label class="sl"><span class="top"><span>Log interval</span><span class="val" id="v-L"></span></span><input type="range" id="L"><span class="sn">steps per epoch: diagnostics, checkpoint and <code>history.npz</code> after each</span></label>
  <label class="sl"><span class="top"><span>Total steps</span><span class="val" id="v-T"></span></span><input type="range" id="T"><span class="sn" id="n-T"></span></label>
</div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>How much of a 24 h job is lost, and at which log interval?</h2>
  <p class="note">Every trainer-loop series of this config; the selected one is bold. Dots mark each series' best interval; click anywhere on a curve to set the interval.</p>
  <div class="legend" id="lg-curve"></div>
  <div id="p-curve"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Where does a 24 h job go besides steps?</h2>
  <p class="note">The lost share of one chunk at the chosen interval: the 300 s requeue margin, the restart, the per-epoch costs and the killed epoch (the whole epoch when the kill lands inside its background write).</p>
  <div class="legend" id="lg-split"></div>
  <div id="p-split"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What does a restart cost?</h2>
  <p class="note">Before the first step: the <code>uv run</code> start, the imports with the git check and <code>setup</code>, a cold restore of the checkpoint, and the diagnostics and epoch compiles. ≥ marks lower bounds: the record ran after other configs in one process and reused their compiles.</p>
  <div class="legend" id="lg-restart"></div>
  <div id="p-restart"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What has been measured?</h2>
  <p class="note">Share of a 24 h job lost at the chosen interval, with the restart below it, for every series and config. OOM: the trainer loop ran out of memory (in the epoch, the setup or the diagnostics); —: not run.</p>
  <div class="scroll" id="p-cov"></div>
</div>
<div class="card means">
  <h2>What this means</h2>
  __MEANS__
</div>
</main>
<script>
window.onerror = (msg, src, line) => { const e = document.getElementById("err"); e.style.display = "block"; e.textContent = `Script error: ${msg} (line ${line})`; };
const D = __DATA__;
const CARDS = ["A100 80 GB", "A100 40 GB", "L40S"];
const CARD_ID = {"A100 80 GB": "a80", "A100 40 GB": "a40", "L40S": "l40s"};
const COLOR = {"A100 80 GB": "var(--a80)", "A100 40 GB": "var(--a40)", "L40S": "var(--l40s)"};
const PRECS = [["backbone", "bf16 backbone"], ["attention", "bf16 attention"]];
const PREC_NAME = Object.fromEntries(PRECS);
const DASH = {backbone: "", attention: "6 3"};
const SIZES = ["XS", "S", "B", "L", "XL"];
const DEF = D.defaults;
const S = {card: "A100 80 GB", prec: "backbone", band: "full", size: "B", L: DEF.log_interval, T: null};
const marks = {};

const cfg = () => `${S.band}-${S.size}`;
const total = () => S.T ?? DEF.total[S.band];
const fmtS = v => v == null ? "—" : v >= 86400 ? (v / 86400).toFixed(v >= 864000 ? 0 : 2) + " d" : v >= 3600 ? (v / 3600).toFixed(1) + " h" : v >= 100 ? (v / 60).toFixed(v >= 600 ? 0 : 1) + " min" : v >= 10 ? v.toFixed(1) + " s" : v >= 1 ? v.toFixed(2) + " s" : (1e3 * v).toFixed(0) + " ms";
const fmtStep = v => v >= 1 ? v.toFixed(2) + " s" : v.toFixed(3) + " s";
const fmtG = v => v >= 10 ? v.toFixed(1) : v.toFixed(2);
const fmtP = v => (100 * v).toFixed(100 * v >= 10 ? 0 : 1) + "%";
const fmtN = v => v >= 1e6 ? (v / 1e6).toFixed(v % 1e6 ? 2 : 0) + "M" : v >= 1e3 ? (v / 1e3).toFixed(v % 1e3 ? 1 : 0) + "k" : String(v);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const act = o => esc(JSON.stringify(o));
const name = r => `${r.card}, ${PREC_NAME[r.prec]}`;
const on = r => r.card === S.card && r.prec === S.prec;

// chunk model, a port of overhead.py's chunk()
function chunk(c, interval) {
  const hs = Object.entries(c.history).map(([k, v]) => [Number(k), v]).sort((a, b) => a[0] - b[0]);
  const [[l0, h0], [l1, h1]] = hs;
  const e0 = c.total / l0, e1 = c.total / l1;
  const history = Math.max(0, h1 + (h0 - h1) * (c.total / interval - e1) / (e0 - e1));
  const train = interval * c.step;
  const gap = history + train + c.contention + c.diagnostics;
  const wait = Math.max(0, c.background - gap);
  const per_epoch = c.diagnostics + history + c.block + c.contention + wait;
  const epoch = train + per_epoch;
  const n = (D.chunk_s - D.margin_s - c.R) / epoch - 0.5;
  const base = {per_epoch, wait, history, n: Math.max(n, 0)};
  if (n <= 0) return {...base, lost: 1, margin: D.margin_s / D.chunk_s, restart: c.R / D.chunk_s, epochs: 0, killed: 0};
  const unsaved = Math.min(c.background, epoch) / epoch * train;
  const out = {margin: D.margin_s / D.chunk_s, restart: c.R / D.chunk_s, epochs: n * per_epoch / D.chunk_s, killed: (epoch / 2 + unsaved) / D.chunk_s};
  return {...base, ...out, lost: out.margin + out.restart + out.epochs + out.killed};
}
const withTotal = c => ({...c, total: total()});
const bestInterval = c => D.nice.filter(L => L <= c.total).reduce((a, L) => chunk(c, L).lost < chunk(c, a).lost ? L : a, D.nice[0]);
const configRows = () => D.overhead.filter(r => r.config === cfg() && r.c).sort((a, b) => CARDS.indexOf(a.card) - CARDS.indexOf(b.card) || a.prec.localeCompare(b.prec));

// controls
function seg(id, items, key, after) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    S[key] = b.dataset.v;
    if (after) after();
    schedule();
  }));
}
seg("card", CARDS.map(c => [c, c]), "card");
seg("prec", PRECS, "prec");
seg("band", [["narrow", "narrow"], ["full", "full"]], "band", () => { S.T = null; });
seg("size", SIZES.map(s => [s, s]), "size");

const TSTEPS = Array.from({length: 61}, (_, k) => Number((10 ** (4 + k / 25)).toPrecision(2)));
const sl = {
  L: {vals: () => D.nice.filter(L => L <= total()), get: () => S.L, set: v => { S.L = v; }},
  T: {vals: () => TSTEPS, get: () => total(), set: v => { S.T = v; }},
};
const nearest = (vals, v) => vals.reduce((b, x, i) => Math.abs(Math.log(x / v)) < Math.abs(Math.log(vals[b] / v)) ? i : b, 0);
for (const [id, s] of Object.entries(sl)) {
  const input = document.getElementById(id);
  input.addEventListener("input", () => { s.set(s.vals()[Number(input.value)]); schedule(); });
}
function syncControls() {
  for (const key of Object.keys(marks)) marks[key]();
  for (const [id, s] of Object.entries(sl)) {
    const input = document.getElementById(id), vals = s.vals();
    input.min = 0; input.max = vals.length - 1; input.value = nearest(vals, s.get());
  }
  document.getElementById("v-L").textContent = S.L.toLocaleString("en");
  document.getElementById("v-T").textContent = total().toLocaleString("en");
  document.getElementById("n-T").textContent = S.T == null ? `the ${S.band} band's configured total_steps` : `configured: ${DEF.total[S.band].toLocaleString("en")}`;
}

function writeHash() {
  const h = {card: CARD_ID[S.card], prec: S.prec, cfg: cfg(), L: S.L};
  if (S.T != null) h.T = S.T;
  history.replaceState(null, "", "#" + new URLSearchParams(h).toString());
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const card = CARDS.find(c => CARD_ID[c] === h.get("card")); if (card) S.card = card;
  if (PREC_NAME[h.get("prec")]) S.prec = h.get("prec");
  const [band, size] = (h.get("cfg") || "").split("-"); if (["narrow", "full"].includes(band) && SIZES.includes(size)) { S.band = band; S.size = size; }
  if (D.nice.includes(Number(h.get("L")))) S.L = Number(h.get("L"));
  if (Number(h.get("T")) > 0) S.T = Number(h.get("T"));
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

// chart frame, log x and linear y
const M = {l: 56, r: 16, t: 10, b: 40};
function frame(host, o) {
  const W = Math.max(300, host.clientWidth), H = 280;
  const X = v => M.l + (Math.log10(v) - Math.log10(o.x[0])) / (Math.log10(o.x[1]) - Math.log10(o.x[0])) * (W - M.l - M.r);
  const Y = v => M.t + (o.y[1] - v) / (o.y[1] - o.y[0]) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(o.label)}">`;
  o.yticks.forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${o.yfmt(v)}</text>`);
  for (let e = Math.ceil(Math.log10(o.x[0])); e <= Math.log10(o.x[1]); e++) {
    const v = 10 ** e;
    s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${fmtN(v)}</text>`;
  }
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">${o.xlabel}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  return {W, H, X, Y, s};
}
const pathOf = (pts, X, Y) => pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("");
const sw = (color, dash, kind) => kind === "dot" ? `<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${color}"/></svg>` : `<svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="${color}" stroke-width="2" stroke-dasharray="${dash || ""}"/></svg>`;
const empty = (host, lg, what) => {
  const ooms = D.overhead.filter(r => r.config === cfg() && r.oom);
  host.innerHTML = `<p class="note">${ooms.length ? `Every trainer-loop run of ${cfg()} ran out of memory.` : `The trainer loop was not run for ${cfg()}.`}</p>`;
  document.getElementById(lg).innerHTML = "";
};

function tiles() {
  const tile = (k, v, d, cls = "") => `<div class="tile"><div class="k">${k}</div><div class="v ${cls}">${v}</div><div class="d">${d}</div></div>`;
  const r = D.overhead.find(r => r.config === cfg() && r.card === S.card && r.prec === S.prec), out = [];
  if (!r || r.oom) {
    const why = r ? `the trainer loop ran out of memory in ${r.oom}${r.plan ? ` (plan ${fmtG(r.plan)} GiB)` : ""}` : `no trainer-loop run of ${S.card}, ${PREC_NAME[S.prec]}, ${cfg()}`;
    out.push(tile("Loop step", r ? "Out of memory" : "Not measured", why, r ? "bad" : "na"));
    document.getElementById("tiles").innerHTML = out.join("");
    return;
  }
  const c = withTotal(r.c), ch = chunk(c, S.L), best = bestInterval(c), bl = chunk(c, best).lost;
  const wall = c.total * c.step / (1 - ch.lost);
  out.push(tile("Loop step", fmtStep(c.step), `best long epoch per step, batch 256`));
  out.push(tile("Restart", (r.lower ? "≥ " : "") + fmtS(c.R), `${fmtP(c.restart.compiles / c.R)} compiles`));
  out.push(tile("Lost per 24 h job", fmtP(ch.lost), `at interval ${S.L.toLocaleString("en")}; per epoch ${fmtS(ch.per_epoch)} beyond its steps`));
  out.push(tile("Best interval", best.toLocaleString("en"), `${fmtP(bl)} lost, ${(100 * (ch.lost - bl)).toFixed(1)} points under the chosen one`));
  out.push(tile("Wall time", fmtS(wall), `${c.total.toLocaleString("en")} steps with restarts and saves; ${Math.ceil(wall / D.chunk_s)} jobs`));
  out.push(tile("Checkpoint", `${fmtG(c.gib)} GiB`, `on disk; state ${fmtG(c.state_gib)} GiB in memory`));
  document.getElementById("tiles").innerHTML = out.join("");
}

function curvePanel() {
  const host = document.getElementById("p-curve"), rows = configRows();
  if (!rows.length) return empty(host, "lg-curve");
  const curves = rows.map(r => { const c = withTotal(r.c); return {r, c, pts: D.curve.filter(L => L <= c.total).map(L => [L, chunk(c, L).lost])}; });
  const top = Math.min(1, Math.max(0.05, 4 * Math.max(...curves.map(k => chunk(k.c, bestInterval(k.c)).lost)), ...curves.filter(k => on(k.r)).map(k => 1.2 * chunk(k.c, S.L).lost)));
  const f = frame(host, {x: [10, 1e6], y: [0, top], yfmt: v => (100 * v).toFixed(top < 0.05 ? 1 : 0) + "%", yticks: Array.from({length: 6}, (_, k) => Number((k * top / 5).toPrecision(2))), xlabel: "log interval (steps per epoch)", ylabel: "share of a 24 h job lost", label: "lost walltime against log interval"});
  let s = f.s + `<line x1="${f.X(S.L)}" x2="${f.X(S.L)}" y1="${M.t}" y2="${f.H - M.b}" stroke="var(--ref)" stroke-dasharray="3 3"/>`;
  for (const {r, c, pts} of curves.sort((a, b) => on(a.r) - on(b.r))) {
    s += `<path d="${pathOf(pts.filter(p => p[1] <= top), f.X, f.Y)}" fill="none" stroke="${COLOR[r.card]}" stroke-width="${on(r) ? 2.4 : 1.3}" stroke-dasharray="${DASH[r.prec]}" opacity="${on(r) ? 1 : 0.55}"/>`;
    const best = bestInterval(c), bl = chunk(c, best).lost, cur = chunk(c, S.L).lost;
    for (const L of D.nice.filter(L => L <= c.total && L >= 10)) {
      const v = chunk(c, L).lost;
      if (v <= top) s += `<circle cx="${f.X(L)}" cy="${f.Y(v)}" r="7" fill="transparent" data-tip="${esc(`<b>${name(r)}</b><br>interval ${L.toLocaleString("en")}: ${fmtP(v)} of each 24 h job lost`)}" data-act="${act({card: r.card, prec: r.prec, L})}"/>`;
    }
    if (bl <= top) s += `<circle cx="${f.X(best)}" cy="${f.Y(bl)}" r="4" fill="${COLOR[r.card]}" data-tip="${esc(`<b>${name(r)}</b><br>best interval ${best.toLocaleString("en")}: ${fmtP(bl)} lost`)}" data-act="${act({card: r.card, prec: r.prec, L: best})}"/>`;
    if (cur <= top) s += `<circle cx="${f.X(S.L)}" cy="${f.Y(cur)}" r="3" fill="var(--surface)" stroke="${COLOR[r.card]}" stroke-width="1.5" pointer-events="none"/>`;
  }
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-curve").innerHTML = rows.map(r => `<span>${sw(COLOR[r.card], DASH[r.prec])}${name(r)}</span>`).join("") + `<span>${sw("var(--ink-2)", "", "dot")}best interval</span>`;
}

// stacked bars, one per series: a chunk's lost parts or a restart's parts
function barsPanel(id, parts, valOf, totOf, unit) {
  const host = document.getElementById("p-" + id), rows = configRows();
  if (!rows.length) return empty(host, "lg-" + id);
  const W = Math.max(300, host.clientWidth), rowH = 34, H = rows.length * rowH + 44, L0 = Math.min(190, W * 0.4);
  const vals = rows.map(valOf);
  const scaleMax = Math.max(...vals.map(v => parts.reduce((a, [k]) => a + v[k], 0))) * 1.08;
  const X = v => L0 + v / scaleMax * (W - L0 - 60);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${id}">`;
  Array.from({length: 5}, (_, k) => k * scaleMax / 4).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="4" y2="${H - 30}" stroke="var(--grid)"/><text x="${X(v)}" y="${H - 14}" text-anchor="middle">${unit(v)}</text>`);
  rows.forEach((r, i) => {
    const y = 8 + i * rowH, v = vals[i];
    let x = 0;
    s += `<text class="lbl" x="${L0 - 8}" y="${y + 15}" text-anchor="end" font-weight="${on(r) ? 700 : 400}">${name(r)}</text>`;
    for (const [k, label, op, fmt] of parts) {
      const tip = `<b>${name(r)}</b><br>${label}: ${fmt(v[k])}${id === "restart" && r.lower ? "<br>lower bound: compiles reused from the configs before it in one process" : ""}`;
      s += `<rect x="${X(x)}" y="${y}" width="${Math.max(X(x + v[k]) - X(x), 0.5)}" height="22" fill="${COLOR[r.card]}" opacity="${op}" stroke="var(--surface)" data-tip="${esc(tip)}" data-act="${act({card: r.card, prec: r.prec})}"/>`;
      x += v[k];
    }
    s += `<text class="lbl" x="${X(x) + 6}" y="${y + 15}">${totOf(r, v)}</text>`;
  });
  host.innerHTML = s + "</svg>";
  document.getElementById("lg-" + id).innerHTML = parts.map(([k, label, op]) => `<span><svg width="14" height="12"><rect width="14" height="12" fill="var(--ink-2)" opacity="${op}"/></svg>${label}</span>`).join("");
}
const ofDay = v => `${fmtP(v)} of 24 h (${fmtS(v * D.chunk_s)})`;
const SPLIT = [["margin", "requeue margin", 0.25, ofDay], ["restart", "restart", 0.5, ofDay], ["epochs", "per-epoch costs", 0.75, ofDay], ["killed", "killed epoch", 1, ofDay]];
const RESTART = [["uv", "uv run", 0.25, fmtS], ["imports", "imports and setup", 0.5, fmtS], ["restore", "cold restore", 0.75, fmtS], ["compiles", "compiles", 1, fmtS]];

function covPanel() {
  const series = [];
  for (const card of CARDS) for (const [p] of PRECS) if (D.overhead.some(r => r.card === card && r.prec === p)) series.push([card, p]);
  let s = `<table class="cov"><thead><tr><th>interval ${S.L.toLocaleString("en")}</th>${D.configs.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
  for (const [card, p] of series) {
    s += `<tr><td>${card}<small>${PREC_NAME[p]}</small></td>`;
    for (const c of D.configs) {
      const r = D.overhead.find(r => r.card === card && r.prec === p && r.config === c);
      const sel = card === S.card && p === S.prec && c === cfg() ? " sel" : "", a = act({card, prec: p, config: c, T: null});
      if (r && r.c) {
        const cc = {...r.c, total: DEF.total[c.split("-")[0]]};
        s += `<td class="${sel}" data-act="${a}">${fmtP(chunk(cc, S.L).lost)}<small>${r.lower ? "≥" : ""}${Math.round(r.c.R)} s</small></td>`;
      } else if (r) s += `<td class="oom${sel}" data-act="${a}" title="out of memory in ${esc(r.oom)}">OOM<small>${r.oom}</small></td>`;
      else s += `<td class="na${sel}" data-act="${a}">—</td>`;
    }
    s += "</tr>";
  }
  document.getElementById("p-cov").innerHTML = s + "</tbody></table>";
}

function render() {
  syncControls();
  tiles(); curvePanel();
  barsPanel("split", SPLIT, r => chunk(withTotal(r.c), S.L), (r, v) => fmtP(v.lost), v => (100 * v).toFixed(1) + "%");
  barsPanel("restart", RESTART, r => r.c.restart, r => (r.lower ? "≥ " : "") + fmtS(r.c.R), v => Math.round(v) + " s");
  covPanel();
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
<li><b>Restarts and checkpoints cost 1.2–2.3% of each 24 h job at the default interval of 1000</b>, with the bf16 backbone: a B run takes 1.03–1.04 d on the A100 80 GB and 1.2–1.6 d on the L40S.</li>
<li><b>The restart is mostly compiles on the job's 2 CPUs</b> (73–96% of 400–705 s). The async save blocks for under a second; its write overlaps the next epoch.</li>
<li><b>The log interval is not worth tuning</b> with a fast step: the best interval saves at most 0.6 points over 1000. With bf16 attention a 1000-step epoch takes hours, and 50–200 is the right interval.</li>
<li><b>XL is borderline in the trainer's own process</b>: full-XL on the A100 80 GB and narrow-XL on the L40S ran out of memory in the loop although they fit alone in the step check.</li>
</ul>
"""


def main() -> None:
    data = dict(
        defaults=defaults(),
        configs=CONFIGS,
        overhead=overhead_rows(),
        chunk_s=model.CHUNK_S,
        margin_s=model.MARGIN_S,
        nice=model.NICE,
        curve=model.CURVE,
    )
    report(data)
    page = PAGE.replace("__MEANS__", MEANS).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = HERE / "training-overhead.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
