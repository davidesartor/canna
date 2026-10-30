"""Write step-time-memory.html from data/: the training step's time and memory per card, precision, config and batch.

Stdlib only; run from the repo root:
    uv run --frozen python reports/benchmarks/step-time-memory/scripts/build_page.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parents[1]
REPO = HERE.parents[2]
CARDS = {
    "NVIDIA A100-SXM4-80GB": "A100 80 GB",
    "NVIDIA L40S": "L40S",
    "NVIDIA GeForce GTX 1080 Ti": "GTX 1080 Ti",
}
CONFIGS = [f"{b}-{s}" for b in ("narrow", "full") for s in ("XS", "S", "B", "L", "XL")]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sig(v, n: int = 4):
    return None if v is None else float(f"{v:.{n}g}")


# the batch the data were swept at: the 10-22 default of 4096 samples (HEAD's is 1024, 64 observations)
SWEEP_BATCH, SWEEP_DRAWS = 256, 16


def defaults() -> dict:
    """The sweep's batch and draws, and the band yamls' total_steps, without importing canna."""
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
    return dict(batch=SWEEP_BATCH, draws=SWEEP_DRAWS, total=total)


def step_rows() -> list[dict]:
    """Every sweep point (frozen 2026-10-22 network) and every batch-256 check point (the 10-23 network)."""
    rows = []
    for path in sorted((HERE / "data").glob("*.jsonl")):
        if path.name == "params.jsonl":
            continue
        src = "check" if path.stem.endswith("-b256") else "sweep"
        for r in jsonl(path):
            rows.append(
                dict(
                    card=CARDS[r["chip"]],
                    prec=r["bf16"],
                    config=r["config"],
                    batch=r["batch"],
                    mb=r["micro_batches"],
                    src=src,
                    step=sig(r.get("step_s")),
                    plan=sig(r["plan_gib"]),
                    cap=sig(r["limit_gib"]),
                    draw_plan=sig(r.get("draw_plan_gib")),
                    peak=sig(r.get("peak_gib")),
                    oom=bool(r.get("oom")),
                )
            )
    return rows


def report(data: dict) -> None:
    """Print the numbers the README quotes."""
    d = data["defaults"]
    print("defaults", d)
    for r in data["step"]:
        if r["src"] != "check":
            continue
        if r["oom"]:
            print(
                f"check {r['card']:11} {r['config']:9} mb {r['mb']} OOM plan {r['plan']:.3g}"
            )
            continue
        days = d["total"][r["config"].split("-")[0]] * r["step"] / 86400
        print(
            f"check {r['card']:11} {r['config']:9} mb {r['mb']} step {r['step']:.4g} plan {r['plan']:.3g} peak {r['peak']:.3g} cap {r['cap']:.4g} days {days:.3g}"
        )
    sweep = {
        (r["card"], r["prec"], r["config"]): r["step"]
        for r in data["step"]
        if r["src"] == "sweep"
        and r["batch"] == d["batch"]
        and r["mb"] == 1
        and r["step"]
    }
    for (card, prec, config), step in sorted(sweep.items()):
        if prec != "backbone" and (card, "backbone", config) in sweep:
            print(
                f"sweep {card:11} {config:9} {prec:9} {step:.4g} s, {step / sweep[card, 'backbone', config]:.3g}x the bf16 backbone"
            )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LISA step time and memory</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --ref: #52514e;
  --a80: #2a78d6; --l40s: #eb6834; --gtx: #199e70; --bad: #c2362b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --ref: #c3c2b7;
    --a80: #3987e5; --l40s: #d95926; --gtx: #1baf7a; --bad: #e5675c;
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
.bar .k, .controls .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.tile .v.na { color: var(--muted); font-weight: 500; }
.tile .v.bad { color: var(--bad); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: auto; display: inline-block; flex: none; overflow: visible; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; margin: 0 0 12px; }
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
.cov sup { color: var(--muted); font-size: 10px; margin-left: 1px; }
.scroll { overflow-x: auto; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 80ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>How long is a training step, and does it fit?</h1>
<p class="note"><b>Timed on the 10-22 network</b> (sweep 2026-10-22/24, step check on the 10-23 network). Both predate the catalogue prior, unordered sources, the x1 head and the per-source frame; the scripts follow HEAD and were not rerun. The sweep's default batch of 256 observations (4096 samples) was the old default: HEAD's is 1024 samples, 64 observations at 16 draws.</p>
<p class="sub">The trainer's fused step (batch draw, micro-batch scan, schedule-free update) timed per card, precision, config and batch, with XLA's memory plan. Pick a point with the bar or by clicking any chart or cell.</p>
<div class="bar">
  <div><span class="k">Card</span><span class="seg" id="card"></span></div>
  <div><span class="k">Precision</span><span class="seg" id="prec"></span></div>
  <div><span class="k">Band</span><span class="seg" id="band"></span></div>
  <div><span class="k">Size</span><span class="seg" id="size"></span></div>
  <div><span class="k">Batch</span><span class="seg" id="batch"></span></div>
</div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>How long is a step, and how does it grow with the batch?</h2>
  <p class="note">Every card and precision the sweep ran for this config; the selected one is bold. Hollow points needed gradient accumulation. The sweep ran the frozen network of 2026-10-22 (old narrow window); diamonds are the 10-23 network at batch 256 in the trainer's environment (the step check).</p>
  <div class="controls"><div><span class="k">Show</span><span class="seg" id="yunit"></span></div></div>
  <div class="legend" id="lg-step"></div>
  <div id="p-step"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>Does it fit in memory, and what fills it?</h2>
  <p class="note">XLA's memory plan for the selected card, precision and config at each batch and micro-batch split (×: out of memory). The dashed line is the batch draw's own plan: the trainer draws the whole batch before splitting it, so accumulation cannot shrink that part. The sweep preallocated 95% of the card; the trainer gets JAX's default 75%.</p>
  <div class="legend" id="lg-mem"></div>
  <div id="p-mem"></div><div class="tip"></div>
</div>
<div class="card">
  <h2>What has been measured?</h2>
  <p class="note">Seconds per step at the selected batch, from the best source: <sup>c</sup> the step check (the 10-23 network), <sup>s</sup> the sweep (2026-10-22 network); ÷n marks n micro-batches. OOM: out of memory at every split tried; —: not measured.</p>
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
const CARDS = ["A100 80 GB", "L40S", "GTX 1080 Ti"];
const CARD_ID = {"A100 80 GB": "a80", "L40S": "l40s", "GTX 1080 Ti": "gtx"};
const COLOR = {"A100 80 GB": "var(--a80)", "L40S": "var(--l40s)", "GTX 1080 Ti": "var(--gtx)"};
const PRECS = [["backbone", "bf16 backbone"], ["attention", "bf16 attention"], ["none", "fp32"]];
const PREC_NAME = Object.fromEntries(PRECS);
const DASH = {backbone: "", attention: "6 3", none: "2 3"};
const SIZES = ["XS", "S", "B", "L", "XL"];
const BATCHES = Array.from({length: 15}, (_, k) => 2 ** k);
const DEF = D.defaults;
const S = {card: "A100 80 GB", prec: "backbone", band: "full", size: "B", batch: DEF.batch, yunit: "step"};
const marks = {};

const cfg = () => `${S.band}-${S.size}`;
const fmtS = v => v >= 86400 ? (v / 86400).toFixed(1) + " d" : v >= 3600 ? (v / 3600).toFixed(1) + " h" : v >= 100 ? (v / 60).toFixed(v >= 600 ? 0 : 1) + " min" : v >= 10 ? v.toFixed(1) + " s" : v >= 1 ? v.toFixed(2) + " s" : v >= 0.01 ? (1e3 * v).toFixed(0) + " ms" : (1e3 * v).toFixed(1) + " ms";
const fmtStep = v => v >= 1 ? v.toFixed(2) + " s" : v.toFixed(3) + " s";
const fmtG = v => v >= 10 ? v.toFixed(1) : v.toFixed(2);
const fmtB = v => v >= 1024 ? (v / 1024) + "k" : String(v);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const act = o => esc(JSON.stringify(o));

// every measurement of one point; the step check wins over the sweep
function lookup(card, prec, config, batch) {
  const match = r => r.card === card && r.prec === prec && r.config === config && r.batch === batch;
  const pick = rows => { const ran = rows.filter(r => r.step).sort((a, b) => a.mb - b.mb); return {rows, ran: ran[0] || null, oom: rows.length > 0 && !ran.length}; };
  const check = pick(D.step.filter(r => r.src === "check" && match(r)));
  const sweep = pick(D.step.filter(r => r.src === "sweep" && match(r)));
  let head = null;
  if (check.ran) head = {step: check.ran.step, src: "c", mb: check.ran.mb, what: "step check, the 10-23 network"};
  else if (sweep.ran) head = {step: sweep.ran.step, src: "s", mb: sweep.ran.mb, what: "sweep, 2026-10-22 network"};
  return {check, sweep, head};
}
function fit(m) {
  const {check, sweep} = m;
  if (check.ran) {
    const r = check.ran;
    return {v: r.mb > 1 ? `Fits ÷${r.mb}` : "Fits", d: `plan ${fmtG(r.plan)}, peak ${fmtG(r.peak)} of the trainer's ${fmtG(r.cap)} GiB${r.mb > 1 ? "; out of memory in one pass" : ""}`};
  }
  if (check.oom) return {v: "Out of memory", bad: true, d: "at every split the step check tried"};
  if (sweep.ran) {
    const r = sweep.ran, cap = r.cap / 0.95 * 0.75;
    const over = r.plan > cap ? `; over the trainer's ${fmtG(cap)} GiB cap` : ` of the trainer's ${fmtG(cap)} GiB`;
    return {v: r.mb > 1 ? `Fits ÷${r.mb}` : "Fits", bad: r.plan > cap, d: `sweep plan ${fmtG(r.plan)} GiB${over}${r.mb > 1 ? `; needs ${r.mb} micro-batches` : ""}`};
  }
  if (sweep.oom) return {v: "Out of memory", bad: true, d: `at every split the sweep tried (smallest plan ${fmtG(Math.min(...sweep.rows.map(r => r.plan)))} GiB)`};
  return {v: "Not measured", na: true, d: "no record of this card, precision, config and batch"};
}

// controls
function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    S[key] = typeof S[key] === "number" ? Number(b.dataset.v) : b.dataset.v;
    schedule();
  }));
}
seg("card", CARDS.map(c => [c, c.replace("GTX ", "")]), "card");
seg("prec", PRECS, "prec");
seg("band", [["narrow", "narrow"], ["full", "full"]], "band");
seg("size", SIZES.map(s => [s, s]), "size");
seg("batch", [16, 64, 256, 1024, 4096].map(b => [b, String(b)]), "batch");
seg("yunit", [["step", "s per step"], ["obs", "ms per observation"]], "yunit");

function writeHash() {
  history.replaceState(null, "", "#" + new URLSearchParams({card: CARD_ID[S.card], prec: S.prec, cfg: cfg(), batch: S.batch, y: S.yunit}).toString());
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const card = CARDS.find(c => CARD_ID[c] === h.get("card")); if (card) S.card = card;
  if (PREC_NAME[h.get("prec")]) S.prec = h.get("prec");
  const [band, size] = (h.get("cfg") || "").split("-"); if (["narrow", "full"].includes(band) && SIZES.includes(size)) { S.band = band; S.size = size; }
  if (BATCHES.includes(Number(h.get("batch")))) S.batch = Number(h.get("batch"));
  if (["step", "obs"].includes(h.get("y"))) S.yunit = h.get("y");
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

// log-log chart frame
const M = {l: 68, r: 16, t: 10, b: 40};
function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
  return out;
}
function frame(host, o) {
  const W = Math.max(300, host.clientWidth), H = 280;
  const X = v => M.l + (Math.log10(v) - Math.log10(o.x[0])) / (Math.log10(o.x[1]) - Math.log10(o.x[0])) * (W - M.l - M.r);
  const Y = v => M.t + (Math.log10(o.y[1]) - Math.log10(v)) / (Math.log10(o.y[1]) - Math.log10(o.y[0])) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(o.label)}">`;
  const yt = logTicks(o.y[0], o.y[1]);
  const yThin = yt.length > 9 ? yt.filter(v => /^[15]/.test(v.toPrecision(1))) : yt;
  yThin.forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${o.yfmt(v)}</text>`);
  const xThin = BATCHES.length > (W - M.l) / 46 ? BATCHES.filter(v => /^1/.test(v.toPrecision(1)) || [256, 4096].includes(v)) : BATCHES;
  xThin.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${fmtB(v)}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">batch (observations, ${DEF.draws} draws each)</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  s += `<line x1="${X(S.batch)}" x2="${X(S.batch)}" y1="${M.t}" y2="${H - M.b}" stroke="var(--ref)" stroke-dasharray="3 3"/>`;
  return {W, H, X, Y, s};
}
const pathOf = (pts, X, Y) => pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("");
const span = vals => [Math.min(...vals) / 1.25, Math.max(...vals) * 1.25];
const sw = (color, dash, kind) => kind === "dot" ? `<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${color}"/></svg>` : kind === "hollow" ? `<svg width="12" height="12"><circle cx="6" cy="6" r="3.5" fill="var(--surface)" stroke="${color}" stroke-width="1.6"/></svg>` : kind === "diamond" ? `<svg width="12" height="12"><path d="M6 1L11 6L6 11L1 6Z" fill="${color}"/></svg>` : kind === "x" ? `<svg width="12" height="12"><path d="M2 2L10 10M10 2L2 10" stroke="${color}" stroke-width="1.8"/></svg>` : `<svg width="24" height="10"><line x1="1" x2="23" y1="5" y2="5" stroke="${color}" stroke-width="2" stroke-dasharray="${dash || ""}"/></svg>`;
const diamond = (x, y, r, fill, extra = "") => `<path d="M${x} ${y - r}L${x + r} ${y}L${x} ${y + r}L${x - r} ${y}Z" fill="${fill}" stroke="var(--surface)" stroke-width="1" ${extra}/>`;
const cross = (x, y, r, color, extra = "") => `<path d="M${x - r} ${y - r}L${x + r} ${y + r}M${x + r} ${y - r}L${x - r} ${y + r}" stroke="${color}" stroke-width="2" ${extra}/><circle cx="${x}" cy="${y}" r="${r + 2}" fill="transparent" ${extra}/>`;
const fewestSplits = rows => { const best = new Map(); rows.filter(r => r.step).sort((a, b) => a.mb - b.mb).forEach(r => { if (!best.has(r.batch)) best.set(r.batch, r); }); return [...best.values()].sort((a, b) => a.batch - b.batch); };

function tiles(m, f) {
  const tile = (k, v, d, cls = "") => `<div class="tile"><div class="k">${k}</div><div class="v ${cls}">${v}</div><div class="d">${d}</div></div>`;
  const T = DEF.total[S.band], params = D.params[cfg()], out = [];
  out.push(m.head ? tile("Step", fmtStep(m.head.step), `${m.head.what}${m.head.mb > 1 ? `, ${m.head.mb} micro-batches` : ""}`) : tile("Step", "Not measured", `${S.card}, ${PREC_NAME[S.prec]}, ${cfg()} at batch ${S.batch}`, "na"));
  out.push(tile("Memory", f.v, f.d, f.bad ? "bad" : f.na ? "na" : ""));
  out.push(m.head ? tile("Steps to total_steps", fmtS(T * m.head.step), `${T.toLocaleString("en")} steps, no restarts or checkpoints`) : tile("Steps to total_steps", "—", "no step measured", "na"));
  out.push(tile("Network", `${(params / 1e6).toFixed(params < 1e8 ? 1 : 0)}M params`, `${cfg()}, the 10-23 network`));
  document.getElementById("tiles").innerHTML = out.join("");
}

function stepPanel() {
  const host = document.getElementById("p-step"), config = cfg(), per = S.yunit === "obs";
  const yv = r => per ? 1e3 * r.step / r.batch : r.step;
  const rows = D.step.filter(r => r.config === config && r.step);
  if (!rows.length) { host.innerHTML = `<p class="note">No step was measured for ${config}.</p>`; document.getElementById("lg-step").innerHTML = ""; return; }
  const f = frame(host, {x: [0.8, 20000], y: span(rows.map(yv)), yfmt: v => per ? fmtS(v / 1e3) : fmtS(v), ylabel: per ? "ms per observation" : "seconds per step", label: "step time against batch"});
  let s = f.s;
  const keys = [...new Set(rows.filter(r => r.src === "sweep").map(r => `${r.card}|${r.prec}`))], sel = `${S.card}|${S.prec}`;
  for (const k of keys.filter(k => k !== sel).concat(keys.includes(sel) ? [sel] : [])) {
    const [card, prec] = k.split("|"), on = k === sel, color = COLOR[card];
    const pts = fewestSplits(rows.filter(r => r.src === "sweep" && r.card === card && r.prec === prec));
    s += `<path d="${pathOf(pts.map(r => [r.batch, yv(r)]), f.X, f.Y)}" fill="none" stroke="${color}" stroke-width="${on ? 2.4 : 1.2}" stroke-dasharray="${DASH[prec]}" opacity="${on ? 1 : 0.35}"/>`;
    for (const r of pts) {
      const tip = `<b>${card}, ${PREC_NAME[prec]}</b><br>${config}, batch ${r.batch}${r.mb > 1 ? ` in ${r.mb} micro-batches` : ""}<br>${fmtStep(r.step)} per step, ${(1e3 * r.step / r.batch).toPrecision(3)} ms per observation<br>sweep (2026-10-22 network)`;
      const fill = r.mb > 1 ? "var(--surface)" : color;
      s += `<circle cx="${f.X(r.batch)}" cy="${f.Y(yv(r))}" r="${on ? 4 : 3}" fill="${fill}" stroke="${color}" stroke-width="1.6" opacity="${on ? 1 : 0.5}" data-tip="${esc(tip)}" data-act="${act({card, prec, batch: r.batch})}"/>`;
    }
  }
  for (const r of rows.filter(r => r.src === "check")) {
    const on = r.card === S.card && r.prec === S.prec;
    const tip = `<b>${r.card}, ${PREC_NAME[r.prec]}</b><br>${config}, batch ${r.batch}${r.mb > 1 ? ` in ${r.mb} micro-batches` : ""}<br>${fmtStep(r.step)} per step<br>step check, the 10-23 network`;
    s += diamond(f.X(r.batch) - 7, f.Y(yv(r)), on ? 7 : 5.5, COLOR[r.card], `data-tip="${esc(tip)}" data-act="${act({card: r.card, prec: r.prec, batch: r.batch})}"`);
  }
  host.innerHTML = s + "</svg>";
  const cards = CARDS.filter(c => rows.some(r => r.card === c));
  document.getElementById("lg-step").innerHTML = cards.map(c => `<span>${sw(COLOR[c], "", "dot")}${c}</span>`).join("")
    + PRECS.map(([p, n]) => `<span>${sw("var(--ink-2)", DASH[p])}${n}</span>`).join("")
    + `<span>${sw("var(--ink-2)", "", "hollow")}accumulated</span><span>${sw("var(--ink-2)", "", "diamond")}step check</span>`;
}

function memPanel() {
  const host = document.getElementById("p-mem"), config = cfg(), color = COLOR[S.card], lg = document.getElementById("lg-mem");
  const rows = D.step.filter(r => r.config === config && r.card === S.card && r.prec === S.prec);
  if (!rows.length) { host.innerHTML = `<p class="note">No memory plan of ${config} on the ${S.card} with ${PREC_NAME[S.prec]} at any batch.</p>`; lg.innerHTML = ""; return; }
  const sweep = rows.filter(r => r.src === "sweep"), sweepCap = sweep[0]?.cap;
  const trainer = rows.find(r => r.src === "check")?.cap ?? (sweepCap ? sweepCap / 0.95 * 0.75 : null);
  const vals = rows.map(r => r.plan).concat(rows.filter(r => r.draw_plan).map(r => r.draw_plan), [sweepCap, trainer].filter(Boolean), rows.filter(r => r.peak).map(r => r.peak));
  const f = frame(host, {x: [0.8, 20000], y: span(vals.filter(v => v > 0)), yfmt: v => v >= 10 ? String(Math.round(v)) : String(Number(v.toPrecision(v >= 1 ? 2 : 1))), ylabel: "GiB", label: "memory plan against batch"});
  let s = f.s;
  if (sweepCap) s += `<line x1="${M.l}" x2="${f.W - M.r}" y1="${f.Y(sweepCap)}" y2="${f.Y(sweepCap)}" stroke="var(--ref)" stroke-dasharray="2 3"/><text class="lbl" x="${M.l + 4}" y="${f.Y(sweepCap) - 4}">sweep cap ${fmtG(sweepCap)} GiB (95%)</text>`;
  if (trainer) s += `<line x1="${M.l}" x2="${f.W - M.r}" y1="${f.Y(trainer)}" y2="${f.Y(trainer)}" stroke="var(--ref)"/><text class="lbl" x="${f.W - M.r - 4}" y="${f.Y(trainer) - 4}" text-anchor="end">trainer cap ${fmtG(trainer)} GiB (75%)</text>`;
  const bp = fewestSplits(sweep);
  s += `<path d="${pathOf(bp.map(r => [r.batch, r.plan]), f.X, f.Y)}" fill="none" stroke="${color}" stroke-width="2"/>`;
  s += `<path d="${pathOf(bp.filter(r => r.draw_plan > 0).map(r => [r.batch, r.draw_plan]), f.X, f.Y)}" fill="none" stroke="${color}" stroke-width="1.6" stroke-dasharray="5 4"/>`;
  for (const r of sweep) {
    const tip = `<b>batch ${r.batch}${r.mb > 1 ? `, ${r.mb} micro-batches` : ", one pass"}</b><br>plan ${fmtG(r.plan)} GiB${r.draw_plan ? `, of which the draw ${fmtG(r.draw_plan)} GiB` : ""}<br>${r.oom ? "out of memory" : `${fmtStep(r.step)} per step`}`;
    const a = `data-tip="${esc(tip)}" data-act="${act({batch: r.batch})}"`, x = f.X(r.batch), y = f.Y(r.plan);
    s += r.oom ? cross(x, y, 4, "var(--bad)", a) : `<circle cx="${x}" cy="${y}" r="${r.mb > 1 ? 3.5 : 4}" fill="${r.mb > 1 ? "var(--surface)" : color}" stroke="${color}" stroke-width="1.6" ${a}/>`;
  }
  for (const r of rows.filter(r => r.src === "check")) {
    const tip = `<b>step check, batch ${r.batch}${r.mb > 1 ? `, ${r.mb} micro-batches` : ""}</b><br>the 10-23 network, trainer's environment<br>plan ${fmtG(r.plan)} GiB${r.peak ? `, peak ${fmtG(r.peak)} GiB` : ""}<br>${r.oom ? "out of memory" : `${fmtStep(r.step)} per step`}`;
    s += r.oom ? cross(f.X(r.batch) - 9, f.Y(r.plan), 4, "var(--bad)", `data-tip="${esc(tip)}"`) : diamond(f.X(r.batch) - 9, f.Y(r.plan), 6, color, `data-tip="${esc(tip)}"`);
  }
  host.innerHTML = s + "</svg>" + (sweep.length ? "" : `<p class="note">The sweep has no record of ${config} on the ${S.card} with ${PREC_NAME[S.prec]}; only the step check at batch ${DEF.batch}.</p>`);
  lg.innerHTML = `<span>${sw(color)}plan, fewest micro-batches that ran</span><span>${sw(color, "5 4")}the batch draw's plan</span><span>${sw(color, "", "dot")}one pass</span><span>${sw(color, "", "hollow")}accumulated</span><span>${sw("var(--bad)", "", "x")}out of memory</span><span>${sw(color, "", "diamond")}step check</span>`;
}

function covPanel() {
  const series = [];
  for (const card of CARDS) for (const [p] of PRECS) if (D.step.some(r => r.card === card && r.prec === p)) series.push([card, p]);
  let s = `<table class="cov"><thead><tr><th>batch ${S.batch}</th>${D.configs.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
  for (const [card, p] of series) {
    s += `<tr><td>${card}<small>${PREC_NAME[p]}</small></td>`;
    for (const c of D.configs) {
      const m = lookup(card, p, c, S.batch), sel = card === S.card && p === S.prec && c === cfg() ? " sel" : "", a = act({card, prec: p, config: c});
      if (m.head) s += `<td class="${sel}" data-act="${a}">${m.head.step.toPrecision(2)}${m.head.mb > 1 ? `÷${m.head.mb}` : ""}<sup>${m.head.src}</sup></td>`;
      else if (m.check.oom || m.sweep.oom) s += `<td class="oom${sel}" data-act="${a}">OOM</td>`;
      else s += `<td class="na${sel}" data-act="${a}">—</td>`;
    }
    s += "</tr>";
  }
  document.getElementById("p-cov").innerHTML = s + "</tbody></table>";
}

function render() {
  for (const key of Object.keys(marks)) marks[key]();
  const m = lookup(S.card, S.prec, cfg(), S.batch);
  tiles(m, fit(m)); stepPanel(); memPanel(); covPanel();
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
readHash();
render();
</script>
</body>
</html>
"""

MEANS = """
<ul>
<li><b>A B-size network trains in about a day of steps on an A100 80 GB with the bf16 backbone, in either band.</b> narrow-B: 650k steps × 0.136 s = 1.02 d; full-B: 240k × 0.372 s = 1.03 d. The L40S takes 1.16 d and 1.61 d. XL takes 5.2 d (narrow) and 4.5 d (full) on the A100, 7.6 d and 8.0 d on the L40S, so XL runs belong on A100 80 GB cards.</li>
<li><b>The bf16 backbone is not optional.</b> fp32 is 24× slower than the bf16 backbone for narrow-B and 23× for full-B on the A100 80 GB, and casting only attention to bf16 recovers almost none of it: the dense layers, not attention, carry the cost.</li>
<li><b>Memory decides only at XL.</b> At batch 256 everything up to L fits on both cards in one pass. full-XL on the L40S needs 2 micro-batches; full-XL on the A100 80 GB and narrow-XL on the L40S fit alone but are borderline in the trainer's own loop (see the training-overhead report).</li>
<li><b>On the full band the batch draw sets the memory.</b> The trainer draws the whole batch before splitting it, so gradient accumulation cannot shrink the draw's plan.</li>
</ul>
"""


def main() -> None:
    data = dict(
        defaults=defaults(),
        configs=CONFIGS,
        params={r["config"]: r["params"] for r in jsonl(HERE / "data/params.jsonl")},
        step=step_rows(),
    )
    report(data)
    page = PAGE.replace("__MEANS__", MEANS).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = HERE / "step-time-memory.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
