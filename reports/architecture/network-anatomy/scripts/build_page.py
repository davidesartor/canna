"""Write network-anatomy.html from data/anatomy.jsonl: LisaFlow's shapes, parameters and FLOPs per band x size.

Stdlib only; run from the repo root:
    uv run --frozen python reports/architecture/network-anatomy/scripts/build_page.py
"""

import json
import math
from pathlib import Path

REPORT = Path(__file__).parents[1]
CONFIGS = [f"{b}-{s}" for b in ("narrow", "full") for s in ("XS", "S", "B", "L", "XL")]
# parameter groups, in the page's stacking order
GROUPS = [
    ("data_embed", "data embedding"),
    ("data", "data stream blocks"),
    ("cond", "conditioning MLPs"),
    ("source_embed", "source embedding"),
    ("sources", "source stream blocks"),
    ("sinks", "sink stream"),
    ("registers", "register stream + frame head"),
    ("head", "head"),
]
TOP_GROUP = dict(
    y_embed="data_embed", y_mlp="data_embed", f_mlp="cond", c_mlp="cond",
    x_embed="source_embed", f0_mlp="source_embed",
    s_slot="sinks", frame_slot="registers", frame_modulation="registers", frame_unembed="registers",
    x_modulation="head", x_unembed="head",
)
STREAMS = ("data", "registers", "sources", "sinks")
KINDS = ("modulation", "qkv", "core", "out", "mlp")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sig(v: float, n: int = 4) -> float:
    return float(f"{v:.{n}g}")


def backbone_part(names: list[str]) -> tuple[str, str]:
    """(stream, kind) of a backbone leaf from its module names; qk_gain scales every stream's queries."""
    field = names[3] if names[2] == "attention" else names[2]
    if field == "qk_gain":
        return "sources", "qk gain"
    stream = {"x": "sources", "y": "data", "s": "sinks", "r": "registers"}[field.split("_")[1] if field.startswith("mod_") else field[-1]]
    kind = "modulation" if field.startswith("mod_") else "mlp" if field.startswith("mlp") else "attention"
    return stream, kind


def params(r: dict) -> dict:
    """Parameter counts by group, per block part, by optimizer, and as a module tree."""
    groups = dict.fromkeys((g for g, _ in GROUPS), 0)
    per_block = {}
    optim = {"Muon": 0, "AdamW": 0}
    tree = {}
    unread = 0
    L = r["num_blocks"]
    for leaf in r["leaves"]:
        names = leaf["path"].lstrip(".").split(".")
        size = math.prod(leaf["shape"])
        optim["Muon" if leaf["muon"] else "AdamW"] += size
        # the last block's y and sink queries and gates: one of the three stacked slices, never read
        if names[:2] == ["backbone", "last"] and {"mod_y_attn", "qkv_proj_y", "qkv_proj_s", "mod_s_attn"} & set(names):
            unread += size // 3
        if names[0] == "backbone":
            stream, kind = backbone_part(names)
            groups[stream] += size
            where = names[1]
            one = size // (L - 1) if where == "blocks" else size
            per_block.setdefault(where, {}).setdefault(stream, {}).setdefault(kind, 0)
            per_block[where][stream][kind] += one
        else:
            groups[TOP_GROUP[names[0]]] += size
        node = tree
        for name in names[:-1]:
            node = node.setdefault(name, {})
        node[names[-1]] = [leaf["shape"], size, leaf["muon"]]
    return dict(total=sum(groups.values()), groups=groups, per_block=per_block, optim=optim, tree=tree, unread=unread)


def flops(r: dict) -> dict:
    """XLA flops per part summed into the encode and decode, beside the scan-aware matmul walk."""
    f, L = r["flops"], r["num_blocks"]
    e, b, m = f["embed"], f["blocks"], f["matmul"]
    stream_sum = lambda part, streams: sum(part["parts"][s][k] for s in streams for k in part["parts"][s])
    enc_embed = e["stft"] + e["y_embed"] + e["y_mlp"] + e["f_mlp"] + e["frame_modulation"] + e["frame_unembed"]
    dec_embed = e["c_mlp"] + e["f_mlp"] + e["x_embed"] + e["f0_mlp"]
    dec_head = e["x_modulation"] + e["x_unembed"]
    enc_blocks = (L - 1) * stream_sum(b["middle"], ["data", "registers"]) + stream_sum(b["last"], ["data", "registers"])
    dec_blocks = (L - 1) * stream_sum(b["middle"], ["sources", "sinks"]) + stream_sum(b["last"], ["sources", "sinks"])
    per_draw = m["sampler_2"] - m["sampler_1"]
    return dict(
        embed={k: sig(v) for k, v in e.items()},
        block={w: {s: {k: sig(v) for k, v in b[w]["parts"][s].items()} for s in STREAMS} for w in ("middle", "last")},
        encode=dict(embed=sig(enc_embed), blocks=sig(enc_blocks), total=sig(enc_embed + enc_blocks)),
        decode=dict(embed=sig(dec_embed), blocks=sig(dec_blocks), head=sig(dec_head), total=sig(dec_embed + dec_blocks + dec_head)),
        xla_whole={k: sig(v) for k, v in f["xla_whole"].items()},
        block_check={
            w: dict(
                encode=sig(b[w]["xla"]["encode"] / b[w]["matmul"]["encode"]),
                decode=sig(b[w]["xla"]["decode"] / b[w]["matmul"]["decode"]),
            )
            for w in ("middle", "last")
        },
        matmul={k: sig(v) for k, v in m.items()},
        sampler=dict(
            per_draw=sig(per_draw),
            per_observation=sig(m["sampler_1"] - per_draw),
            shared_conditioning=sig(m["sampler_1"] - per_draw - m["encode"]),
            calls=2 * r["ode_steps"],
            decodes_per_draw=sig(per_draw / m["decode"]),
            per_step=sig(per_draw / r["ode_steps"]),
        ),
    )


def record(r: dict) -> dict:
    keep = ("config", "hidden_dim", "num_heads", "num_blocks", "head_dim", "patch", "window_bins",
            "response_points", "data_tokens", "sources", "sinks", "draws", "ode_steps", "bf16",
            "cache_bytes", "stages", "commit")
    return {k: r[k] for k in keep} | dict(params=params(r), flops=flops(r))


def report(data: dict) -> None:
    """Print the numbers the README quotes."""
    for c in data["configs"]:
        r = data["records"].get(c)
        if r is None:
            print(f"{c}: missing")
            continue
        p, f = r["params"], r["flops"]
        g = p["groups"]
        mm, s = f["matmul"], f["sampler"]
        print(
            f"{c:9} params {p['total'] / 1e6:7.2f}M  data_embed {g['data_embed'] / p['total']:.1%} "
            f"blocks data {g['data'] / p['total']:.1%} src {g['sources'] / p['total']:.1%} sinks {g['sinks'] / p['total']:.1%} registers+frame head {g['registers'] / p['total']:.1%} "
            f"muon {p['optim']['Muon'] / p['total']:.1%}"
        )
        mid = r["params"]["per_block"]["blocks"]
        mod = sum(mid[st].get("modulation", 0) for st in mid)
        blk = sum(sum(mid[st].values()) for st in mid)
        print(f"{'':9} middle block {blk / 1e6:.2f}M, modulation {mod / blk:.1%}; cache {r['cache_bytes'] / 2**20:.2f} MiB; unread in last block {p['unread'] / 1e6:.3f}M ({p['unread'] / p['total']:.2%})")
        print(
            f"{'':9} matmul GFLOP encode {mm['encode'] / 1e9:.4g} decode {mm['decode'] / 1e9:.4g} "
            f"ratio {mm['encode'] / mm['decode']:.3g}; xla parts encode {f['encode']['total'] / 1e9:.4g} decode {f['decode']['total'] / 1e9:.4g}; "
            f"xla whole encode {f['xla_whole']['encode'] / 1e9:.4g} decode {f['xla_whole']['decode'] / 1e9:.4g}"
        )
        print(
            f"{'':9} block xla/matmul {f['block_check']}; sampler decodes/draw {s['decodes_per_draw']:.4g} "
            f"per draw {s['per_draw'] / 1e9:.4g} GFLOP, per obs {s['per_observation'] / 1e9:.4g} (shared t-conditioning {s['shared_conditioning'] / 1e9:.4g}, {s['shared_conditioning'] / (s['calls'] * mm['decode']):.1%} of a decode); "
            f"train fwd {mm['train_forward'] / 1e9:.4g} grad {mm['train_grad'] / 1e9:.4g} "
            f"encode share of fwd {mm['encode'] / mm['train_forward']:.1%}; y_embed {f['embed']['y_embed'] / mm['encode']:.1%} of encode"
        )
        blk = f["block"]["middle"]
        tot_d = sum(blk["data"].values())
        tot_x = sum(sum(blk[st].values()) for st in ("sources", "sinks"))
        print(
            f"{'':9} middle block MFLOP data {tot_d / 1e6:.4g} (core {blk['data']['core'] / tot_d:.1%}, mlp {blk['data']['mlp'] / tot_d:.1%}, mod {blk['data']['modulation'] / tot_d:.1%}) "
            f"src+sinks {tot_x / 1e6:.4g} (core {blk['sources']['core'] / tot_x:.1%}, mod {(blk['sources']['modulation'] + blk['sinks']['modulation']) / tot_x:.1%})"
        )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LisaFlow anatomy</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --box: #f1f0ec;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100; --c5: #e87ba4; --c6: #008300; --c7: #6250d6; --c8: #a0522d;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --box: #242422;
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #cd8a5f;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; max-width: 90ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.card + .card { margin-top: 16px; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; margin: 0 0 16px; }
.bar .k, .controls .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; margin: 0 0 12px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 10px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
svg .ink { fill: var(--ink); }
svg [data-cfg] { cursor: pointer; }
.dia text { font-size: 12px; fill: var(--ink-2); }
.dia .t { fill: var(--ink); font-weight: 600; }
.dia .s { fill: var(--muted); font-size: 11px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.dia .box { fill: var(--box); stroke: var(--axis); }
.dia .arrow { stroke: var(--muted); fill: none; stroke-width: 1.5; }
.dia .data { stroke: var(--c1); stroke-width: 2; }
.dia .src { stroke: var(--c2); stroke-width: 2; }
.dia .sink { stroke: var(--c3); stroke-width: 2; }
.dia .reg { stroke: var(--c8); stroke-width: 2; }
.dia .cond { stroke: var(--c4); stroke-width: 2; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 5; max-width: 320px; }
.tip b { color: var(--ink); font-weight: 600; }
table.t { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
table.t th, table.t td { padding: 4px 6px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
table.t th:first-child, table.t td:first-child { text-align: left; }
table.t td.l, table.t th.l { text-align: left; white-space: normal; }
table.t th { color: var(--ink-2); font-weight: 600; }
table.t tr.sel td { background: var(--box); }
table.t td small { color: var(--muted); }
table.t tr[data-cfg] { cursor: pointer; }
.scroll { overflow-x: auto; }
.wide { min-width: 760px; }
.tree { font-size: 13px; font-variant-numeric: tabular-nums; }
.tree details { margin-left: 14px; }
.tree summary { cursor: pointer; list-style-position: outside; }
.tree .row { display: flex; gap: 10px; justify-content: space-between; border-bottom: 1px dotted var(--grid); padding: 1px 0; }
.tree .leaf { margin-left: 28px; color: var(--ink-2); }
.tree .n { color: var(--ink); white-space: nowrap; }
.tree .sh { color: var(--muted); font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }
.tree .mu { color: var(--muted); font-size: 11px; }
.means ul { margin: 6px 0; padding-left: 20px; max-width: 80ch; }
.means li { margin: 4px 0; }
#err { display: none; background: #c2362b; color: #fff; padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; }
</style>
</head>
<body>
<main>
<div id="err"></div>
<h1>What is inside LisaFlow?</h1>
<p class="sub">The flow network of every band × size, measured from the code: tensor shapes from <code>jax.eval_shape</code>, parameters from the module's leaves, FLOPs from XLA's cost analysis of each part and from a scan-aware walk of the jaxpr. Pick a config with the bar or by clicking a chart row.</p>
<div class="bar">
  <div><span class="k">Band</span><span class="seg" id="band"></span></div>
  <div><span class="k">Size</span><span class="seg" id="size"></span></div>
</div>
<div class="tiles" id="tiles"></div>

<div class="card">
  <h2>How is the network wired?</h2>
  <p class="note">Two token streams through the same <span id="d-L"></span> MMDiT blocks. The data stream (blue) sees only the window and its frequency, and the frame registers (brown, one per f0-ordered source) read it alongside; no data or register token attends to the sources. Every block leaves its data and register keys and values in a cache, and the registers' final state goes through the frame head to the frame (μ, log σ): one <code>encode</code> per observation. The source stream (orange, in z = (x − μ) / σ) and the attention sinks (green) run <code>decode</code> per draw and per ODE step, attending jointly to their own tokens and the cached keys. Shapes are for the selected config.</p>
  <div class="scroll"><div class="wide" id="p-dia"></div></div>
</div>

<div class="card">
  <h2>What shape is each tensor?</h2>
  <p class="note">Every stage of one example's forward (batching is the caller's <code>vmap</code>). The cache is in the trainer's precision; FLOPs elsewhere on this page are counted in float32.</p>
  <div class="scroll" id="p-shapes"></div>
</div>

<div class="card">
  <h2>Where do the parameters live?</h2>
  <p class="note">Share of the parameters by group, for every config (bar length is the share, the label the total). Backbone parameters are split by the stream they act on; <code>qk_gain</code> counts with the sources, and <code>frame_slot</code> and the frame head with the registers.</p>
  <div class="legend" id="lg-par"></div>
  <div id="p-par"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>What is one block made of?</h2>
  <p class="note">Parameters and float32 FLOPs per forward of one middle block and of the last block, by stream and part. FLOPs are XLA's count of each part lowered alone; the attention core (scores, softmax, weighted sums, rotary turns, residual adds) is the block's total minus its projections, modulation and MLP. The last block drops the data stream's attention output, its MLP, and the sinks' updates: nothing reads them. The registers keep updating there, since the frame head reads them.</p>
  <div class="scroll" id="p-block"></div>
</div>

<div class="card">
  <h2>What does the module tree hold?</h2>
  <p class="note">Every parameter leaf of the selected config, as <code>eqx</code> holds it. Stacked blocks carry a leading axis of <code>num_blocks − 1</code>. <span class="mu">M</span>: Muon takes it; the rest is AdamW.</p>
  <div class="tree" id="p-tree"></div>
</div>

<div class="card">
  <h2>How does the compute of a forward scale?</h2>
  <p class="note">Matmul FLOPs of one <code>encode</code> (per observation) and one <code>decode</code> (per draw and network call), from the jaxpr walk; the log axis spans all ten configs. Hover for XLA's count.</p>
  <div class="legend" id="lg-scale"></div>
  <div id="p-scale"></div><div class="tip"></div>
</div>

<div class="card">
  <h2>What does sampling and training cost?</h2>
  <p class="note">Matmul FLOPs from the jaxpr walk of <code>sample_posterior</code> (one observation with 1 and with 2 draws, so the per-draw and per-observation parts separate) and of the training objective and its gradient for one observation of <code>draws</code> samples. The midpoint sampler makes 2 network calls per ODE step, all vmapped over the draws. The draws share <code>t</code>, so <code>c_mlp</code> and every adaLN modulation run once per call for all draws: that part lands in the per-observation cost beside the encode, and a draw costs fewer than one decode per call.</p>
  <div class="scroll" id="p-cost"></div>
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
const GROUPS = D.groups;
const COLORS = ["var(--c1)", "var(--c5)", "var(--c4)", "var(--c7)", "var(--c2)", "var(--c3)", "var(--c8)", "var(--c6)"];
const S = {band: "full", size: "B"};
const marks = {};
const cfg = () => `${S.band}-${S.size}`;
const R = () => D.records[cfg()];
const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const fmtN = v => v >= 1e9 ? (v / 1e9).toFixed(v >= 1e10 ? 1 : 2) + "G" : v >= 1e6 ? (v / 1e6).toFixed(v >= 1e7 ? 1 : 2) + "M" : v >= 1e3 ? (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + "k" : String(Math.round(v));
const fmtF = v => v >= 1e12 ? (v / 1e12).toFixed(2) + " TFLOP" : v >= 1e9 ? (v / 1e9).toFixed(v >= 1e10 ? 1 : 2) + " GFLOP" : v >= 1e6 ? (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + " MFLOP" : (v / 1e3).toFixed(0) + " kFLOP";
const fmtB = v => v >= 2 ** 30 ? (v / 2 ** 30).toFixed(2) + " GiB" : v >= 2 ** 20 ? (v / 2 ** 20).toFixed(2) + " MiB" : (v / 1024).toFixed(0) + " KiB";
const pct = v => (100 * v).toFixed(v >= 0.1 ? 0 : 1) + "%";
const shp = a => "[" + a.join(", ") + "]";
const stage = name => R().stages.find(s => s.name === name);

function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === S[key])));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { S[key] = b.dataset.v; schedule(); }));
}
seg("band", [["narrow", "narrow"], ["full", "full"]], "band");
seg("size", SIZES.map(s => [s, s]), "size");
function pick(c) { const [b, s] = c.split("-"); S.band = b; S.size = s; schedule(); }
document.addEventListener("click", e => { const t = e.target.closest("[data-cfg]"); if (t) pick(t.dataset.cfg); });

function tip(card, html, ev) {
  const t = card.querySelector(".tip");
  if (!html) { t.style.display = "none"; return; }
  t.innerHTML = html; t.style.display = "block";
  const box = card.getBoundingClientRect();
  let x = ev.clientX - box.left + 12, y = ev.clientY - box.top + 12;
  x = Math.min(x, box.width - t.offsetWidth - 8);
  t.style.left = x + "px"; t.style.top = y + "px";
}

function tiles() {
  const r = R(), p = r.params, f = r.flops;
  const items = [
    ["Parameters", fmtN(p.total), `${pct(p.optim.Muon / p.total)} on Muon`],
    ["Tokens", `${r.data_tokens} + ${r.sources} + ${r.sinks}`, `data (${r.window_bins} bins / ${r.patch}) + sources + sinks; ${r.sources} registers`],
    ["Width × blocks", `${r.hidden_dim} × ${r.num_blocks}`, `${r.num_heads} heads of ${r.head_dim}`],
    ["Encode", fmtF(f.matmul.encode), "matmul, once per observation"],
    ["Decode", fmtF(f.matmul.decode), "matmul, per draw and network call"],
    ["Data cache", fmtB(r.cache_bytes), `keys + values per observation, ${stage("cache").dtype}`],
  ];
  document.getElementById("tiles").innerHTML = items.map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}

function diagram() {
  const r = R(), st = n => shp(stage(n).shape);
  const L = r.num_blocks;
  document.getElementById("d-L").textContent = L;
  const box = (x, y, w, h, title, sub, cls = "") => `<rect class="box ${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="7"/>` +
    `<text class="t" x="${x + 10}" y="${y + 18}">${title}</text>` + (sub ? `<text class="s" x="${x + 10}" y="${y + 34}">${esc(sub)}</text>` : "");
  const arr = (x1, y1, x2, y2, cls = "") => `<path class="arrow ${cls}" d="M${x1},${y1} L${x2},${y2}" marker-end="url(#ah)"/>`;
  const path = (d, cls = "") => `<path class="arrow ${cls}" d="${d}" marker-end="url(#ah)"/>`;
  const keys = 2 * r.sources + r.sinks + r.data_tokens;
  const svg = `<svg class="dia" viewBox="0 0 960 640" role="img" aria-label="LisaFlow architecture diagram">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>
<text class="t" x="20" y="20">encode: once per observation</text>
<text class="t" x="520" y="20">decode: per draw, per network call</text>
${box(20, 36, 300, 44, "window y", st("y") + "  (A, E, T; re, im)")}
${arr(170, 80, 170, 100, "data")}
${box(20, 100, 300, 44, "STFT tokens", st("stft") + "  asinh|z|, phase")}
${arr(170, 144, 170, 164, "data")}
${box(20, 164, 300, 58, "y_embed (linear) + y_mlp(f)", st("stft") + " → " + st("tokens_y"))}
<text class="s" x="30" y="214">y_mlp: features of each token's lowest, highest f</text>
${box(340, 36, 150, 58, "f_mlp(window f)", st("c_f") + " → [" + r.hidden_dim + "]")}
<text class="s" x="350" y="86">c_f</text>
${box(340, 120, 150, 58, "c_mlp(t, log(1−t))", "")}
<text class="s" x="350" y="154">${st("c")} → [${r.hidden_dim}]</text>
<text class="s" x="350" y="170">c = c_mlp + f_mlp</text>
${arr(170, 222, 170, 250, "data")}
<rect class="box" x="20" y="250" width="300" height="200" rx="7" style="stroke: var(--c1)"/>
<text class="t" x="30" y="270">${L} blocks, data + register streams</text>
<text x="30" y="292">adaLN(c_f) → q, k, v (rms, qk gain, RoPE)</text>
<text x="30" y="310">one softmax over ${r.data_tokens} data + ${r.sources} register keys</text>
<text x="30" y="328">registers unrotated; adaLN(c_f) → SwiGLU MLP (2×)</text>
<text class="s" x="30" y="352">data ${st("tokens_y")}, registers ${st("registers")}</text>
<text x="30" y="380">each block leaves its k, v (data, then registers)</text>
<text x="30" y="398">(last block: data stops, registers update)</text>
${box(340, 268, 150, 58, "registers", st("registers"))}
<text class="s" x="350" y="318">c_f + frame_slot</text>
${arr(340, 297, 322, 297, "reg")}
${box(20, 480, 180, 58, "data + register cache", st("cache") + " × 2")}
<text class="s" x="30" y="530">${stage("cache").dtype}, ${fmtB(r.cache_bytes)} per observation</text>
${arr(110, 450, 110, 480, "data")}
${path("M200,509 C 420,509 420,350 520,350", "data")}
${path("M490,65 C 400,65 400,260 320,260", "cond")}
${path("M490,149 C 505,149 505,300 520,300", "cond")}
${arr(270, 450, 270, 570, "reg")}
${box(20, 570, 470, 58, "frame head: frame_modulation(c_f) → frame_unembed", "")}
<text class="s" x="30" y="612">${st("registers")} → delta ${st("delta")} → frame (μ, log σ) ${st("frame")}</text>
${path("M490,599 C 515,599 505,430 520,430", "reg")}
${box(520, 36, 200, 44, "sources z", st("z") + "  x = μ + σ z")}
${box(740, 36, 200, 44, "sinks", st("sinks") + "  c + s_slot")}
${arr(620, 80, 620, 100, "src")}
${box(520, 100, 200, 58, "x_embed (linear)", st("features_x") + " → " + st("tokens_x"))}
<text class="s" x="530" y="150">+ f0_mlp ${st("f0_x")}, f0 in Hz</text>
${arr(620, 158, 620, 178, "src")}
${box(520, 178, 200, 44, "source tokens", st("tokens_x"))}
${arr(620, 222, 620, 250, "src")}
${arr(840, 80, 840, 250, "sink")}
<rect class="box" x="520" y="250" width="420" height="200" rx="7" style="stroke: var(--c2)"/>
<text class="t" x="530" y="270">${L} blocks, sources + sinks</text>
<text x="530" y="292">adaLN(c) per stream → q, k, v</text>
<text x="530" y="310">one softmax over ${keys} keys: ${r.sources} sources + ${r.sinks} sinks,</text>
<text x="530" y="326">${r.data_tokens} cached data + ${r.sources} cached registers</text>
<text x="530" y="344">absolute RoPE at f0 (sources) and token centres (data)</text>
<text x="530" y="360">sinks and registers unrotated; adaLN(c) → SwiGLU MLP</text>
<text class="s" x="530" y="384">sources ${st("tokens_x")}, sinks ${st("sinks")}</text>
<text x="530" y="404">(last block: sinks give keys only)</text>
${arr(620, 450, 620, 480, "src")}
${box(520, 480, 200, 58, "x_modulation(c) → x_unembed", "")}
<text class="s" x="530" y="530">SwiGLU head, [${r.hidden_dim}] → [${stage("out").shape[1]}]</text>
${arr(720, 509, 750, 509, "src")}
${box(750, 480, 190, 58, "m, log s of z₁", st("out"))}
</svg>`;
  document.getElementById("p-dia").innerHTML = svg;
}

function shapes() {
  const r = R();
  const rows = r.stages.map(s => `<tr><td><code>${s.name}</code></td><td><code>${shp(s.shape)}</code></td><td>${s.dtype}</td><td class="l">${esc(s.what)}</td></tr>`).join("");
  document.getElementById("p-shapes").innerHTML = `<table class="t"><tr><th>stage</th><th>shape</th><th>dtype</th><th class="l">what</th></tr>${rows}</table>`;
}

function paramChart() {
  const host = document.getElementById("p-par"), card = host.closest(".card");
  document.getElementById("lg-par").innerHTML = GROUPS.map(([g, name], i) => `<span><i style="background:${COLORS[i]}"></i>${name}</span>`).join("");
  const W = Math.max(host.clientWidth, 320), left = 76, right = 64, rowH = 26, top = 4;
  const configs = D.configs.filter(c => D.records[c]);
  const w = W - left - right;
  let s = `<svg viewBox="0 0 ${W} ${top + rowH * configs.length + 22}">`;
  configs.forEach((c, j) => {
    const p = D.records[c].params, y = top + j * rowH;
    let x = left;
    s += `<g data-cfg="${c}"><rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`;
    s += `<text x="${left - 8}" y="${y + 17}" text-anchor="end" class="${c === cfg() ? "ink" : "lbl"}" style="${c === cfg() ? "font-weight:600" : ""}">${c}</text>`;
    GROUPS.forEach(([g], i) => {
      const share = p.groups[g] / p.total, len = share * w;
      if (len > 0.5) s += `<rect x="${x + 1}" y="${y + 4}" width="${Math.max(len - 2, 0.5)}" height="${rowH - 8}" rx="2" fill="${COLORS[i]}" data-g="${i}"/>`;
      x += len;
    });
    s += `<text x="${W - right + 6}" y="${y + 17}" class="lbl">${fmtN(p.total)}</text></g>`;
  });
  [0, 0.25, 0.5, 0.75, 1].forEach(v => { s += `<text x="${left + v * w}" y="${top + rowH * configs.length + 14}" text-anchor="middle">${pct(v)}</text>`; });
  host.innerHTML = s + "</svg>";
  host.querySelectorAll("[data-cfg]").forEach(g => {
    g.addEventListener("mousemove", ev => {
      const c = g.dataset.cfg, p = D.records[c].params;
      tip(card, `<b>${c}</b> · ${fmtN(p.total)}<br>` + GROUPS.map(([k, name]) => `${name}: ${fmtN(p.groups[k])} (${pct(p.groups[k] / p.total)})`).join("<br>"), ev);
    });
    g.addEventListener("mouseleave", () => tip(card, null));
  });
}

function blockTable() {
  const r = R(), pb = r.params.per_block, fb = r.flops.block;
  const KINDS = [["modulation", "adaLN modulation", "modulation"], ["qkv", "q, k, v projections", "attention"], ["core", "attention core + residuals", null], ["out", "out projection", null], ["mlp", "SwiGLU MLP", "mlp"]];
  const streams = [["data", "data (encode)"], ["registers", "registers (encode)"], ["sources", "sources (decode)"], ["sinks", "sinks (decode)"]];
  let rows = "";
  const tot = {middle: {p: 0, f: 0}, last: {p: 0, f: 0}};
  for (const [s, sname] of streams) {
    for (const [k, kname, pk] of KINDS) {
      if (k === "core" && (s === "sinks" || s === "registers")) continue;
      const cells = ["blocks", "last"].map(w => {
        const where = w === "blocks" ? "middle" : "last";
        const par = pk === null ? null : (pk === "attention" ? ((pb[w][s] || {}).attention || 0) + (s === "sources" ? ((pb[w][s] || {})["qk gain"] || 0) : 0) : ((pb[w][s] || {})[pk] || 0));
        const fl = fb[where][s][k];
        if (par) tot[where].p += par;
        tot[where].f += fl;
        return `<td>${par === null ? "<small>in qkv</small>" : par ? fmtN(par) : "—"}</td><td>${fl > 0 ? fmtF(fl) : "—"}</td>`;
      }).join("");
      const label = k === "core" ? (s === "data" ? "attention core + residuals (data and registers, one softmax)" : "attention core + residuals (sources and sinks, one softmax)") : kname;
      rows += `<tr><td>${sname}</td><td class="l">${label}</td>${cells}</tr>`;
    }
  }
  rows += `<tr><td><b>block</b></td><td></td><td><b>${fmtN(tot.middle.p)}</b></td><td><b>${fmtF(tot.middle.f)}</b></td><td><b>${fmtN(tot.last.p)}</b></td><td><b>${fmtF(tot.last.f)}</b></td></tr>`;
  const e = r.flops.embed, f = r.flops;
  const outside = [["stft tokens", e.stft, "encode"], ["y_embed + y_mlp", e.y_embed + e.y_mlp, "encode"], ["f_mlp", e.f_mlp, "encode and decode"], ["frame_modulation + frame_unembed", e.frame_modulation + e.frame_unembed, "encode"], ["c_mlp", e.c_mlp, "decode"], ["x_embed + f0_mlp", e.x_embed + e.f0_mlp, "decode"], ["x_modulation + x_unembed", e.x_modulation + e.x_unembed, "decode"]];
  const orows = outside.map(([n, v, w]) => `<tr><td class="l">${n}</td><td>${w}</td><td>${fmtF(v)}</td></tr>`).join("");
  const enc = f.encode, dec = f.decode;
  document.getElementById("p-block").innerHTML =
    `<table class="t"><tr><th>stream</th><th class="l">part</th><th>params / middle block</th><th>FLOPs</th><th>params, last block</th><th>FLOPs</th></tr>${rows}</table>` +
    `<p class="note" style="margin-top:14px">Outside the blocks (XLA FLOPs per call):</p>` +
    `<table class="t"><tr><th class="l">part</th><th>runs in</th><th>FLOPs</th></tr>${orows}</table>` +
    `<p class="note" style="margin-top:12px">Summed with ${r.num_blocks - 1} middle blocks and the last: encode ${fmtF(enc.total)} (${pct(enc.blocks / enc.total)} in blocks), decode ${fmtF(dec.total)} (${pct(dec.blocks / dec.total)} in blocks). The scan-aware matmul walk gives ${fmtF(f.matmul.encode)} and ${fmtF(f.matmul.decode)}; XLA's own count of the whole jitted encode and decode is ${fmtF(f.xla_whole.encode)} and ${fmtF(f.xla_whole.decode)} (a scanned block body counts once there). One middle block: XLA / matmul = ${f.block_check.middle.encode.toFixed(3)} (encode), ${f.block_check.middle.decode.toFixed(3)} (decode).</p>` +
    `<p class="note">The last block still holds the data and sink streams' queries and attention gates, which nothing reads: ${fmtN(r.params.unread)} parameters (${pct(r.params.unread / r.params.total)} of the network) with zero gradient.</p>`;
}

function treeView() {
  const r = R();
  const count = n => Array.isArray(n) ? n[1] : Object.values(n).reduce((a, b) => a + count(b), 0);
  const total = r.params.total;
  const node = (name, n, depth) => {
    if (Array.isArray(n)) return `<div class="row leaf"><span>${name} <span class="sh">${shp(n[0])}</span> ${n[2] ? '<span class="mu">M</span>' : ""}</span><span class="n">${fmtN(n[1])}</span></div>`;
    const c = count(n);
    const kids = Object.entries(n).sort((a, b) => count(b[1]) - count(a[1])).map(([k, v]) => node(k, v, depth + 1)).join("");
    return `<details${depth < 1 ? " open" : ""}><summary><span class="row" style="display:inline-flex;width:calc(100% - 20px)"><span>${name}</span><span class="n">${fmtN(c)} <small>${pct(c / total)}</small></span></span></summary>${kids}</details>`;
  };
  document.getElementById("p-tree").innerHTML = node(`LisaFlow (${cfg()})`, r.params.tree, 0);
}

function scaleChart() {
  const host = document.getElementById("p-scale"), card = host.closest(".card");
  const series = [["encode", "encode (per observation)", "var(--c1)"], ["decode", "decode (per call)", "var(--c2)"]];
  document.getElementById("lg-scale").innerHTML = series.map(([, n, c]) => `<span><i style="background:${c};border-radius:50%"></i>${n}</span>`).join("");
  const configs = D.configs.filter(c => D.records[c]);
  const vals = configs.flatMap(c => series.map(([k]) => D.records[c].flops.matmul[k]));
  const lo = Math.floor(Math.log10(Math.min(...vals))), hi = Math.ceil(Math.log10(Math.max(...vals)));
  const W = Math.max(host.clientWidth, 320), left = 76, right = 16, rowH = 26, top = 4;
  const w = W - left - right, X = v => left + (Math.log10(v) - lo) / (hi - lo) * w;
  let s = `<svg viewBox="0 0 ${W} ${top + rowH * configs.length + 22}">`;
  for (let k = lo; k <= hi; k++) {
    s += `<line x1="${X(10 ** k)}" x2="${X(10 ** k)}" y1="${top}" y2="${top + rowH * configs.length}" stroke="var(--grid)"/>`;
    s += `<text x="${X(10 ** k)}" y="${top + rowH * configs.length + 14}" text-anchor="middle">${fmtF(10 ** k).replace(" ", "")}</text>`;
  }
  configs.forEach((c, j) => {
    const m = D.records[c].flops.matmul, y = top + j * rowH + rowH / 2;
    s += `<g data-cfg="${c}"><rect x="0" y="${y - rowH / 2}" width="${W}" height="${rowH}" fill="transparent"/>`;
    s += `<text x="${left - 8}" y="${y + 4}" text-anchor="end" class="${c === cfg() ? "ink" : "lbl"}" style="${c === cfg() ? "font-weight:600" : ""}">${c}</text>`;
    s += `<line x1="${X(m.decode)}" x2="${X(m.encode)}" y1="${y}" y2="${y}" stroke="var(--axis)" stroke-width="2"/>`;
    series.forEach(([k, , col]) => { s += `<circle cx="${X(m[k])}" cy="${y}" r="5" fill="${col}" stroke="var(--surface)" stroke-width="2"/>`; });
    s += `</g>`;
  });
  host.innerHTML = s + "</svg>";
  host.querySelectorAll("[data-cfg]").forEach(g => {
    g.addEventListener("mousemove", ev => {
      const c = g.dataset.cfg, f = D.records[c].flops;
      tip(card, `<b>${c}</b><br>encode ${fmtF(f.matmul.encode)} matmul · ${fmtF(f.encode.total)} XLA parts<br>decode ${fmtF(f.matmul.decode)} matmul · ${fmtF(f.decode.total)} XLA parts<br>encode = ${(f.matmul.encode / f.matmul.decode).toFixed(1)} decodes`, ev);
    });
    g.addEventListener("mouseleave", () => tip(card, null));
  });
}

function costTable() {
  const configs = D.configs.filter(c => D.records[c]);
  const head = `<tr><th>config</th><th>encode</th><th>decode</th><th>encode ÷ decode</th><th>sampler / obs</th><th>of which shared adaLN</th><th>sampler / draw</th><th>in decodes (of calls)</th><th>/ ODE step</th><th>train fwd / obs</th><th>train grad / obs</th><th>encode share of train fwd</th></tr>`;
  const rows = configs.map(c => {
    const r = D.records[c], m = r.flops.matmul, s = r.flops.sampler;
    return `<tr data-cfg="${c}" class="${c === cfg() ? "sel" : ""}"><td>${c}</td><td>${fmtF(m.encode)}</td><td>${fmtF(m.decode)}</td><td>${(m.encode / m.decode).toFixed(1)}</td><td>${fmtF(s.per_observation)}</td><td>${fmtF(s.shared_conditioning)}</td><td>${fmtF(s.per_draw)}</td><td>${s.decodes_per_draw.toFixed(1)} <small>(${s.calls})</small></td><td>${fmtF(s.per_step)}</td><td>${fmtF(m.train_forward)}</td><td>${fmtF(m.train_grad)}</td><td>${pct(m.encode / m.train_forward)}</td></tr>`;
  }).join("");
  const r = R();
  document.getElementById("p-cost").innerHTML = `<table class="t">${head}${rows}</table><p class="note" style="margin-top:10px">${r.ode_steps} ODE steps (the sampler's default); training draws ${r.draws} samples per observation. The gradient includes the recomputation of the checkpointed (remat) blocks.</p>`;
}

function render() {
  for (const k of Object.keys(marks)) marks[k]();
  if (!R()) { document.getElementById("tiles").innerHTML = `<div class="tile"><div class="k">${cfg()}</div><div class="v">Not measured</div></div>`; return; }
  history.replaceState(null, "", "#" + new URLSearchParams({cfg: cfg()}).toString());
  tiles(); diagram(); shapes(); paramChart(); blockTable(); treeView(); scaleChart(); costTable();
}
let pending = false;
function schedule() { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; render(); }); }
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const [b, s] = (h.get("cfg") || "").split("-");
  if (["narrow", "full"].includes(b) && SIZES.includes(s)) { S.band = b; S.size = s; }
}
addEventListener("resize", schedule);
addEventListener("hashchange", () => { readHash(); schedule(); });
readHash();
render();
</script>
</body>
</html>
"""

def means(data: dict) -> str:
    """The closing card's bullets, with numbers from the records."""
    rec = data["records"]
    fb, nb, fxs, fxl = rec["full-B"], rec["narrow-B"], rec["full-XS"], rec["full-XL"]
    m, mn = fb["flops"]["matmul"], nb["flops"]["matmul"]
    g = fb["params"]["groups"]
    s = fb["flops"]["sampler"]
    y_share = lambda r: r["flops"]["embed"]["y_embed"] / r["flops"]["matmul"]["encode"]
    ratio = lambda r: r["flops"]["matmul"]["encode"] / r["flops"]["matmul"]["decode"]
    blk = fb["flops"]["block"]["middle"]["data"]
    tot = fb["params"]["total"]
    frame_head = lambda r: (r["flops"]["embed"]["frame_modulation"] + r["flops"]["embed"]["frame_unembed"]) / r["flops"]["matmul"]["encode"]
    block_gap = max(abs(1 - r["flops"]["block_check"]["middle"][k]) for r in rec.values() for k in ("encode", "decode"))
    items = [
        f"<b>The encode outweighs a decode more on the full band.</b> One full-B encode costs {ratio(fb):.1f} decodes "
        f"({ratio(fxs):.1f} at XS, {ratio(fxl):.1f} at XL); on the narrow band it is {ratio(nb):.1f} at B. "
        f"In training the encode is {m['encode'] / m['train_forward']:.0%} of a full-B forward and {mn['encode'] / mn['train_forward']:.0%} of a narrow-B one.",
        f"<b>The cache makes sampling cheap per draw.</b> A full-B draw costs {s['per_draw'] / 1e9:.1f} GFLOP over {s['calls']} network calls, "
        f"against {s['per_observation'] / 1e9:.1f} GFLOP once per observation; past a few draws the encode no longer matters.",
        f"<b>The data embedding is one linear map.</b> <code>y_embed</code> maps the {fb['stages'][1]['shape'][1]:,} STFT numbers of a token to D: "
        f"{y_share(fb):.1%} of the full-B encode, {y_share(fxs):.1%} at XS.",
        f"<b>The frame is nearly free.</b> The {fb['sources']} registers add {fb['sources']} tokens to a stream of {fb['data_tokens']}, and the frame head "
        f"(<code>frame_modulation</code>, <code>frame_unembed</code>) is {frame_head(fb):.2%} of the full-B encode ({frame_head(nb):.2%} narrow-B), "
        f"yet the register stream holds {g['registers'] / tot:.0%} of B's parameters: it has its own projections, MLP and modulation in every block.",
        f"<b>Parameters split four ways, compute does not.</b> At B the data, register, source and sink streams hold "
        f"{g['data'] / tot:.0%}, {g['registers'] / tot:.0%}, {g['sources'] / tot:.0%} and {g['sinks'] / tot:.0%} of the parameters "
        f"({g['registers'] / tot:.1%} for the registers includes the frame head), "
        f"so the {fb['sinks']} sink tokens carry as many weights as the {fb['data_tokens']} data tokens. "
        f"Modulation is {nb['params']['per_block']['blocks']['data']['modulation'] / sum(nb['params']['per_block']['blocks']['data'].values()):.1%} of every block's weights "
        f"but {blk['modulation'] / sum(blk.values()):.1%} of a full-B data block's FLOPs.",
        f"<b>Matmuls are the compute.</b> XLA's count of a lone middle block is within {block_gap:.1%} of the matmul walk in every config, and the attention core (scores, softmax, weighted sums, residuals) "
        f"is {blk['core'] / sum(blk.values()):.0%} of a full-B data block: the SwiGLU MLP ({blk['mlp'] / sum(blk.values()):.0%}) and projections dominate.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def main() -> None:
    rows = {r["config"]: r for r in jsonl(REPORT / "data/anatomy.jsonl")}
    data = dict(
        configs=CONFIGS,
        groups=GROUPS,
        records={c: record(rows[c]) for c in CONFIGS if c in rows},
    )
    report(data)
    page = PAGE.replace("__MEANS__", means(data)).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = REPORT / "network-anatomy.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
