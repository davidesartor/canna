"""Every lisa page, drawn from saved data: a run's training page, an eval dir's corner and validation pages."""

import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from einops import rearrange

from .samples import robust_width

# the obliquity of the ecliptic at J2000 [rad]: LisaGB's sky is equatorial (lisaorbits v3), papers quote ecliptic
OBLIQUITY = np.deg2rad(23.4392911)
# draws per source the corner report keeps of each reference set; the flow keeps all of its own
REFERENCE_DRAWS = {"fisher": 512, "mcmc": 1024}
# float16 holds offsets from the truth in robust flow widths; past this they are only "off the page"
HALF_LIMIT = 6.0e4
COORDINATES = {
    "physical": ["f₀", "M<sub>c</sub>", "log₁₀ A", "λ", "β", "ψ", "ι", "φ₀"],
    "flow": ["x<sub>f</sub>", "m", "l / σ<sub>l</sub>", "b / σ<sub>b</sub>", "Re z<sub>+</sub>", "Im z<sub>+</sub>", "Re z<sub>−</sub>", "Im z<sub>−</sub>"],
}
UNITS = {
    "physical": ["bins", "M☉", "", "rad", "rad", "rad", "rad", "rad"],
    "flow": [""] * 8,
}


def gb_picks(source_snr: np.ndarray) -> dict[str, tuple[int, int]]:
    """(injection, source) of the loudest, median and faintest eval source, from SNRs [J, S]."""
    order = sorted(np.ndindex(source_snr.shape), key=lambda js: source_snr[js])
    return {
        "loudest": order[-1],
        "median": order[len(order) // 2],
        "faintest": order[0],
    }


def fold_orientation(p: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fold draws [..., 8] onto the (psi, phi0) copy nearest the truth: psi within pi/4, phi0 pi.

    By the likelihood's exact symmetries psi -> psi + pi and (psi, phi0) -> (psi + pi/2, phi0 + pi).
    """
    p = np.array(p, np.float64)
    k = np.round((p[..., 5] - truth[..., 5]) / (np.pi / 2))
    p[..., 5] -= k * np.pi / 2
    p[..., 7] = (
        truth[..., 7]
        + (p[..., 7] - k * np.pi - truth[..., 7] + np.pi) % (2 * np.pi)
        - np.pi
    )
    return p


def equatorial_to_ecliptic(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equatorial (RA, Dec) to ecliptic (λ in [0, 2π), β) [rad], a rotation about the equinox by the obliquity."""
    x, y, z = np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    y, z = c * y + s * z, -s * y + c * z
    return np.arctan2(y, x) % (2 * np.pi), np.arcsin(np.clip(z, -1.0, 1.0))


def physical_columns(p: np.ndarray, truth: np.ndarray, start_bin: float, t_obs: float) -> np.ndarray:
    """The report's physical columns of draws [..., S, 8] folded onto truth [S, 8]: f0 in bins from the window's
    first, Mc, log10 A, ecliptic λ (unwrapped around the truth's) and β, ψ, ι, φ0."""
    lam, beta = equatorial_to_ecliptic(p[..., 3], p[..., 4])
    lam_true, _ = equatorial_to_ecliptic(truth[..., 3], truth[..., 4])
    lam = lam_true + (lam - lam_true + np.pi) % (2 * np.pi) - np.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        log_a = np.log10(p[..., 2])
    return np.stack(
        [p[..., 0] * t_obs - start_bin, p[..., 1], log_a, lam, beta, p[..., 5], p[..., 6], p[..., 7]],
        axis=-1,
    )


def corner_data(eval_dir: Path) -> tuple[dict, bytes]:
    """The corner report's index and its draws: every eval injection's flow, Fisher and MCMC draws per source, in
    physical and flow coordinates, as float16 offsets from the truth in units of the flow's robust width."""
    import jax.numpy as jnp
    import yaml

    from .chart import physical_to_flow
    from .problem import LisaGB, window_start
    from .samples import load as load_samples

    run_dir = eval_dir.parents[1]
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    stage = yaml.safe_load((eval_dir / "eval.yaml").read_text()) if (eval_dir / "eval.yaml").exists() else {}
    problem = LisaGB(**config["problem"])
    d = load_samples(eval_dir / "draws.npz")
    mcmc = load_samples(eval_dir / "mcmc.npz") if (eval_dir / "mcmc.npz").exists() else {}
    rng = np.random.default_rng(0)
    n_injections, n_sources = d["source_snr"].shape

    blob, injections = [], []
    offset = 0
    for j in range(n_injections):
        truth, f = d["truth"][j], d["f"][j]
        chains = [mcmc.get(f"{j}_{s}") for s in range(n_sources)]
        sets = {"flow": d["matched"][j], "fisher": d["fisher"][j]}
        if all(c is not None for c in chains):
            sets["mcmc"] = np.stack(chains, axis=1)
        for name, keep in REFERENCE_DRAWS.items():
            if name in sets and len(sets[name]) > keep:
                sets[name] = sets[name][np.sort(rng.choice(len(sets[name]), keep, replace=False))]
        start_bin = float(window_start(problem, f))
        views = {}
        for name, p in [("truth", truth[None]), *sets.items()]:
            p = fold_orientation(p, truth)
            views[name] = {
                "physical": physical_columns(p, truth, start_bin, problem.t_obs),
                # the chart takes the folded draws whole: its z± are the same for every (ψ, φ0) copy
                "flow": np.asarray(physical_to_flow(problem, jnp.asarray(p), f)),
            }
        sources = []
        for s in range(n_sources):
            source = {"snr": float(d["source_snr"][j, s]), "truth": {}, "scale": {}, "sets": {}}
            for view in COORDINATES:
                centre = views["truth"][view][0, s]
                scale = robust_width(views["flow"][view][:, s], 0)
                # a flow collapsed onto one value in a column falls back to a reference's width, then to 1
                for name in ("mcmc", "fisher"):
                    if name in views:
                        fallback = robust_width(views[name][view][:, s], 0)
                        scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
                scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
                source["truth"][view] = centre.tolist()
                source["scale"][view] = scale.tolist()
                for name in sets:
                    v = (views[name][view][:, s] - centre) / scale
                    v = np.where(np.isfinite(v), np.clip(v, -HALF_LIMIT, HALF_LIMIT), np.nan)
                    blob.append(v.astype("<f2").tobytes())
                    source["sets"].setdefault(name, {})[view] = [offset, len(v)]
                    offset += v.size
            sources.append(source)
        injections.append(
            {
                "quantile": float(d["quantiles"][j]),
                "snr": float(d["snr"][j]),
                "f_mhz": float(f * 1e3),
                "start_bin": start_bin,
                "window_bins": int(problem.window_bins),
                "sources": sources,
            }
        )
    meta = {
        "run": run_dir.name,
        "config": config.get("name", ""),
        "commit": str(config.get("commit", ""))[:7],
        "eval": eval_dir.name,
        "epoch": int(d["epoch"]),
        "ode_steps": stage.get("ode_steps", config.get("ode_steps")),
        "weights": stage.get("weights", config.get("weights")),
        "t_obs": float(problem.t_obs),
        "draws": {name: int(len(p)) for name, p in sets.items()},
        "coordinates": COORDINATES,
        "units": UNITS,
        "picks": {k: list(map(int, v)) for k, v in gb_picks(d["source_snr"]).items()},
    }
    return {"meta": meta, "injections": injections}, b"".join(blob)


CORNER_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --flow: #eb6834; --fisher: #2a78d6; --mcmc: #1b9a6a; --truth: #0b0b0b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --flow: #d95926; --fisher: #3987e5; --mcmc: #2fb383; --truth: #f0efec;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --flow: #d95926; --fisher: #3987e5; --mcmc: #2fb383; --truth: #f0efec;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding-block: 28px 64px; padding-inline: 16px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 4px; }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 90ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 10px; }
.layout { display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 20px; align-items: start; }
.side { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto; }
@media (max-width: 900px) { .layout { grid-template-columns: minmax(0, 1fr); } .side { position: static; max-height: none; } }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.content > .card + .card { margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.stack { margin-bottom: 14px; }
.stack > .k { display: block; color: var(--ink-2); font-size: 13px; margin-bottom: 6px; }
.seg { display: grid; width: 100%; grid-template-columns: repeat(var(--cols), minmax(0, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { font: inherit; font-size: 13px; padding: 5px 6px; border: 0; background: var(--surface); color: var(--ink-2); cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.seg .sn { display: block; font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
.seg button[aria-pressed="true"] .sn { color: inherit; opacity: 0.8; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend button { font: inherit; display: inline-flex; align-items: center; gap: 6px; padding: 2px 6px; border: 0; border-radius: 6px; background: none; color: var(--ink-2); cursor: pointer; }
.legend button:hover { background: var(--grid); }
.legend button[aria-pressed="false"] { opacity: 0.4; text-decoration: line-through; }
.row { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
#corner-host { flex: 1 1 720px; min-width: 0; overflow-x: auto; }
canvas { display: block; max-width: 100%; }
#corner-host canvas { cursor: crosshair; max-width: none; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 500; }
td.hi { color: var(--flow); font-weight: 600; }
tr.sel td { background: var(--grid); }
details { font-size: 13px; color: var(--ink-2); }
details summary { cursor: pointer; color: var(--ink); }
details li { margin-bottom: 4px; }
.banner { background: #c62828; color: #fff; padding: 8px 16px; font: 13px/1.4 system-ui, sans-serif; }
</style>
</head>
<body>
<div id="banner" class="banner" hidden></div>
<script>
window.onerror = (message, source, line) => {
  const b = document.getElementById("banner");
  b.hidden = false;
  b.textContent = `This page hit an error: ${message} (line ${line}).`;
};
</script>
<main>
<h1 id="title"></h1>
<p class="sub" id="sub"></p>
<div class="layout">
<aside class="card side">
  <div class="stack"><span class="k">Coordinates</span><div class="seg" id="view" style="--cols: 2"></div></div>
  <div class="stack"><span class="k">Injection (eval SNR quantile, window SNR)</span><div class="seg" id="inj" style="--cols: 5"></div></div>
  <div class="stack"><span class="k">Source (SNR, loudest first)</span><div class="seg" id="src" style="--cols: 5"></div></div>
  <div class="stack"><span class="k">Eval sources by SNR</span><div class="seg" id="picks" style="--cols: 3"></div></div>
  <p class="note">Click a corner cell to enlarge it, a legend entry to cycle it through hidden, shown and shown with points. The view lives in the URL, so a link reopens it.</p>
</aside>
<div class="content">
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <h2 id="corner-title"></h2>
    <p class="note" id="corner-note"></p>
    <div class="legend" id="legend"></div>
    <div class="row"><div id="corner-host"></div></div>
  </div>
  <div class="card">
    <h2>Widths</h2>
    <p class="note" id="table-note"></p>
    <div class="tablewrap" id="table"></div>
  </div>
  <div class="card">
    <details>
      <summary>The two coordinate systems</summary>
      <ul>
        <li><b>Physical:</b> f₀ in frequency bins from the window's first bin (per source: offset from the truth), the chirp mass M<sub>c</sub> in M☉, log₁₀ of the strain amplitude A, the <b>ecliptic</b> longitude λ and latitude β (rotated from the simulator's equatorial RA and Dec by the J2000 obliquity, 23.4393°; λ unwrapped around the truth), the polarisation ψ, the inclination ι (sin ι uniform in [−1, 1]) and the initial phase φ₀. ψ and φ₀ are folded onto the copy nearest the truth by the likelihood's exact symmetries ψ → ψ + π and (ψ, φ₀) → (ψ + π/2, φ₀ + π).</li>
        <li><b>Flow:</b> the 8 coordinates the network learns (<code>chart.physical_to_flow</code>): x<sub>f</sub> = (2 ln f₀ − ln f<sub>lo</sub> − ln f<sub>hi</sub>) / (ln f<sub>hi</sub> − ln f<sub>lo</sub>) over the window's drawable f₀ range (one unit is half of it); m = (ln M<sub>c</sub> − μ) / σ under the chirp-mass prior; galactic l and b over the galactic sky prior's stds; and the helicity amplitudes z± = asinh(A / A₁ · (1 ∓ sin ι)² / 4) · e<sup>i(φ₀′ ± 2ψ)</sup>, A₁ the amplitude of SNR ≈ 1 and φ₀′ the phase at mid-observation. They need no folding: every (ψ, φ₀) copy has the same z±.</li>
        <li><b>Sets:</b> the flow's draws matched onto the injected sources, the Fisher forecast (Hessian averaged over noise, plus the prior) and the narrowband MCMC (one source free, the others at their truth), each thinned as the header says. Contours hold 39% and 86% of each set (1σ and 2σ of a 2-D Gaussian); widths are 1.4826 × MAD.</li>
      </ul>
    </details>
  </div>
</div>
</div>
</main>
<script>const DATA = __DATA__;</script>
<script type="application/octet-stream" id="blob">__BLOB__</script>
<script>
"use strict";
const META = DATA.meta, INJ = DATA.injections;
const SETS = [
  {key: "flow", text: "flow", colour: "--flow"},
  {key: "fisher", text: "Fisher", colour: "--fisher", dash: [5, 3]},
  {key: "mcmc", text: "MCMC", colour: "--mcmc"},
];
const EXTRAS = [{key: "truth", text: "truth"}];
const S = {view: "physical", inj: 0, src: 0, sets: ["flow", "fisher", "mcmc", "truth"], points: [], cell: [3, 4]};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function rgba(name, alpha) {
  const c = css(name), m = c.match(/^#([0-9a-f]{6})$/i);
  if (!m) return c;
  const n = parseInt(m[1], 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${alpha})`;
}
const fmt = (v, d = 2) => v === null || v === undefined || !isFinite(v) ? "–" : Number(v).toFixed(d);
const sig = v => !isFinite(v) ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v.toPrecision(3)).toString();
const plain = html => html.replace(/<sub>(.*?)<\/sub>/g, "_$1").replace(/<[^>]+>/g, "");

// the draws: float16 offsets from the truth in robust flow widths, decoded once per block
const HALF = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  HALF[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}
const RAW = (() => {
  const text = atob(document.getElementById("blob").textContent.trim()), bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return new Uint16Array(bytes.buffer);
})();
const cache = new Map();
// columns [8][N] of one set of one source in one view, as plotted (per-source physical f0 as an offset from the truth)
function columns(j, s, set, view, pooled) {
  const key = `${j}/${s}/${set}/${view}/${pooled}`;
  if (cache.has(key)) return cache.get(key);
  const src = INJ[j].sources[s], where = src.sets[set]?.[view];
  if (!where) return null;
  const [off, n] = where, truth = src.truth[view], scale = src.scale[view], cols = [];
  for (let c = 0; c < 8; c++) {
    const centre = view === "physical" && c === 0 && !pooled ? 0 : truth[c], col = new Float64Array(n);
    for (let k = 0; k < n; k++) col[k] = centre + HALF[RAW[off + k * 8 + c]] * scale[c];
    cols.push(col);
  }
  cache.set(key, cols);
  return cols;
}
const truthOf = (j, s, view, pooled) => INJ[j].sources[s].truth[view].map((v, c) => view === "physical" && c === 0 && !pooled ? 0 : v);
function labelOf(c, pooled) {
  const name = META.coordinates[S.view][c], unit = META.units[S.view][c];
  const text = S.view === "physical" && c === 0 ? (pooled ? "f₀ − window start" : "Δf₀") : name;
  return unit ? `${text} [${unit}]` : text;
}

function sortedFinite(col) { return Float64Array.from(col.filter(Number.isFinite)).sort(); }
function quantile(sorted, q) { return sorted.length ? sorted[clamp(Math.round(q * (sorted.length - 1)), 0, sorted.length - 1)] : NaN; }
function width(col) {
  const s = sortedFinite(col);
  if (!s.length) return NaN;
  const med = quantile(s, 0.5), dev = Float64Array.from(s, v => Math.abs(v - med)).sort();
  return 1.4826 * quantile(dev, 0.5);
}

// everything one corner needs: the visible sets' columns (sources pooled or one), the truths and the axis ranges
function context() {
  const j = S.inj, pooled = S.src === "all", n = INJ[j].sources.length;
  const sources = pooled ? [...Array(n).keys()] : [S.src];
  const sets = [];
  for (const set of SETS) {
    if (!S.sets.includes(set.key)) continue;
    const parts = sources.map(s => columns(j, s, set.key, S.view, pooled));
    if (parts.some(p => !p)) continue;
    const cols = [];
    for (let c = 0; c < 8; c++) {
      const total = parts.reduce((a, p) => a + p[c].length, 0), col = new Float64Array(total);
      let at = 0;
      parts.forEach(p => { col.set(p[c], at); at += p[c].length; });
      cols.push(col);
    }
    sets.push({...set, cols});
  }
  const truths = sources.map(s => truthOf(j, s, S.view, pooled));
  // per column: the flow's central 99% (one source: held to its median ± 8 widths, so a few strays cannot squash
  // it), the MCMC's (else the Fisher's) and the truths, padded 5%
  const range = [];
  for (let c = 0; c < 8; c++) {
    let lo = Infinity, hi = -Infinity;
    const take = (a, b) => { if (isFinite(a) && isFinite(b)) { lo = Math.min(lo, a); hi = Math.max(hi, b); } };
    const flow = sets.find(x => x.key === "flow"), ref = sets.find(x => x.key === "mcmc") || sets.find(x => x.key === "fisher");
    for (const set of [flow, ref]) {
      if (!set) continue;
      const sorted = sortedFinite(set.cols[c]);
      let a = quantile(sorted, 0.005), b = quantile(sorted, 0.995);
      if (set.key === "flow" && !pooled) {
        const med = quantile(sorted, 0.5), w = width(set.cols[c]);
        if (w > 0) { a = Math.max(a, med - 8 * w); b = Math.min(b, med + 8 * w); }
      }
      take(a, b);
    }
    truths.forEach(t => take(t[c], t[c]));
    if (!isFinite(lo)) { lo = -1; hi = 1; }
    if (hi - lo <= Math.abs(hi) * 1e-12) { const d = Math.abs(hi) * 1e-6 || 1; lo -= d; hi += d; }
    const pad = 0.05 * (hi - lo);
    range.push([lo - pad, hi + pad]);
  }
  const flow = sets.find(x => x.key === "flow");
  let off = 0;
  if (flow) {
    const n = flow.cols[0].length;
    for (let k = 0; k < n; k++) if (flow.cols.some((col, c) => !(col[k] >= range[c][0] && col[k] <= range[c][1]))) off++;
    off /= n;
  }
  return {j, pooled, sources, sets, truths, range, off, bins: pooled ? 48 : 30, bins2: pooled ? 44 : 28};
}

function histogram(col, lo, hi, bins) {
  const h = new Float64Array(bins);
  let inside = 0;
  for (const v of col) { const i = Math.floor((v - lo) / (hi - lo) * bins); if (i >= 0 && i < bins) { h[i]++; inside++; } }
  const norm = inside * (hi - lo) / bins || 1;
  return h.map(v => v / norm);
}
// a 2-D histogram smoothed by a 1-bin Gaussian, with the density levels holding 39% and 86% of it
function density(x, y, rx, ry, bins) {
  const h = new Float64Array(bins * bins);
  for (let k = 0; k < x.length; k++) {
    const i = Math.floor((x[k] - rx[0]) / (rx[1] - rx[0]) * bins), j = Math.floor((y[k] - ry[0]) / (ry[1] - ry[0]) * bins);
    if (i >= 0 && i < bins && j >= 0 && j < bins) h[j * bins + i]++;
  }
  const kernel = [0.054, 0.242, 0.399, 0.242, 0.054], tmp = new Float64Array(bins * bins), out = new Float64Array(bins * bins);
  for (let j = 0; j < bins; j++) for (let i = 0; i < bins; i++) {
    let a = 0; for (let d = -2; d <= 2; d++) { const ii = i + d; if (ii >= 0 && ii < bins) a += kernel[d + 2] * h[j * bins + ii]; } tmp[j * bins + i] = a;
  }
  for (let j = 0; j < bins; j++) for (let i = 0; i < bins; i++) {
    let a = 0; for (let d = -2; d <= 2; d++) { const jj = j + d; if (jj >= 0 && jj < bins) a += kernel[d + 2] * tmp[jj * bins + i]; } out[j * bins + i] = a;
  }
  const sorted = Float64Array.from(out).sort().reverse(), total = sorted.reduce((a, v) => a + v, 0), levels = [];
  for (const mass of [0.865, 0.393]) {
    let acc = 0, level = sorted[0];
    for (const v of sorted) { acc += v; level = v; if (acc >= mass * total) break; }
    levels.push(level);
  }
  return {grid: out, levels, max: sorted[0] || 1};
}
// marching squares over the bin centres: line segments of one level, in bin units
function contour(grid, bins, level) {
  const segs = [], at = (i, j) => grid[j * bins + i];
  const lerp = (a, b) => (level - a) / (b - a);
  for (let j = 0; j < bins - 1; j++) for (let i = 0; i < bins - 1; i++) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    const code = (a >= level) | (b >= level) << 1 | (c >= level) << 2 | (d >= level) << 3;
    if (code === 0 || code === 15) continue;
    const e = [
      [i + lerp(a, b), j], [i + 1, j + lerp(b, c)], [i + lerp(d, c), j + 1], [i, j + lerp(a, d)],
    ];
    const pairs = {1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[3, 0], [1, 2]], 6: [[0, 2]], 7: [[3, 2]],
      8: [[2, 3]], 9: [[0, 2]], 10: [[0, 1], [2, 3]], 11: [[1, 2]], 12: [[1, 3]], 13: [[0, 1]], 14: [[0, 3]]}[code];
    for (const [p, q] of pairs) segs.push([e[p], e[q]]);
  }
  return segs;
}

// one cell of the corner: the marginal of c on the diagonal, else the pair (x = cx, y = cy)
function drawCell(g, ctx, cx, cy, x0, y0, size, pad) {
  const inner = size - 2 * pad, rx = ctx.range[cx], ry = ctx.range[cy];
  const X = v => x0 + pad + (v - rx[0]) / (rx[1] - rx[0]) * inner;
  const Y = v => y0 + pad + (ry[1] - v) / (ry[1] - ry[0]) * inner;
  g.save();
  g.beginPath(); g.rect(x0 + 1, y0 + 1, size - 2, size - 2); g.clip();
  if (cx === cy) {
    const curves = ctx.sets.map(set => [set, histogram(set.cols[cx], rx[0], rx[1], ctx.bins)]);
    const peak = Math.max(...curves.map(([, h]) => Math.max(...h)), 1e-300);
    const hy = v => y0 + size - pad - v / peak * inner * 0.92;
    for (const [set, h] of curves) {
      g.strokeStyle = css(set.colour); g.lineWidth = set.key === "flow" ? 1.6 : 1.2; g.setLineDash(set.dash || []);
      g.beginPath();
      h.forEach((v, b) => {
        const xa = X(rx[0] + b * (rx[1] - rx[0]) / ctx.bins), xb = X(rx[0] + (b + 1) * (rx[1] - rx[0]) / ctx.bins);
        b ? g.lineTo(xa, hy(v)) : g.moveTo(xa, hy(v)); g.lineTo(xb, hy(v));
      });
      g.stroke(); g.setLineDash([]);
    }
    if (S.sets.includes("truth")) {
      g.strokeStyle = css("--truth"); g.lineWidth = 1;
      ctx.truths.forEach(t => { g.beginPath(); g.moveTo(X(t[cx]), y0); g.lineTo(X(t[cx]), y0 + size); g.stroke(); });
    }
  } else {
    for (const set of ctx.sets) {
      const {grid, levels, max} = density(set.cols[cx], set.cols[cy], rx, ry, ctx.bins2), cellW = inner / ctx.bins2;
      if (set.key === "flow") {
        for (let j = 0; j < ctx.bins2; j++) for (let i = 0; i < ctx.bins2; i++) {
          const v = grid[j * ctx.bins2 + i] / max;
          if (v > 0.02) { g.fillStyle = rgba(set.colour, 0.45 * v); g.fillRect(x0 + pad + i * cellW, y0 + pad + inner - (j + 1) * cellW, cellW + 0.5, cellW + 0.5); }
        }
      }
      if (S.points.includes(set.key)) {
        g.fillStyle = rgba(set.colour, 0.35);
        const xs = set.cols[cx], ys = set.cols[cy];
        for (let k = 0; k < xs.length; k++) g.fillRect(X(xs[k]) - 0.7, Y(ys[k]) - 0.7, 1.4, 1.4);
      }
      g.strokeStyle = css(set.colour); g.lineWidth = set.key === "flow" ? 1.4 : 1.1; g.setLineDash(set.dash || []);
      const BX = b => x0 + pad + (b + 0.5) * cellW, BY = b => y0 + pad + inner - (b + 0.5) * cellW;
      g.beginPath();
      for (const level of levels) for (const [[ax, ay], [bx, by]] of contour(grid, ctx.bins2, level)) { g.moveTo(BX(ax), BY(ay)); g.lineTo(BX(bx), BY(by)); }
      g.stroke(); g.setLineDash([]);
    }
    if (S.sets.includes("truth")) {
      g.strokeStyle = css("--truth"); g.fillStyle = css("--truth"); g.lineWidth = 0.8;
      if (ctx.pooled) ctx.truths.forEach(t => g.fillRect(X(t[cx]) - 2.5, Y(t[cy]) - 2.5, 5, 5));
      else {
        const t = ctx.truths[0];
        g.beginPath(); g.moveTo(X(t[cx]), y0); g.lineTo(X(t[cx]), y0 + size); g.moveTo(x0, Y(t[cy])); g.lineTo(x0 + size, Y(t[cy])); g.stroke();
        g.fillRect(X(t[cx]) - 2.5, Y(t[cy]) - 2.5, 5, 5);
      }
    }
  }
  g.restore();
  return {X, Y};
}
function ticks(lo, hi, count) {
  const step0 = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(step0)), err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  out.step = step;
  return out;
}
// as many decimals as the tick step needs, so ticks on a narrow axis far from 0 stay distinct
const tickText = (v, step) => Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(clamp(-Math.floor(Math.log10(step) + 1e-9), 0, 10));
function canvasIn(host, W, H) {
  let cv = host.querySelector("canvas");
  if (!cv) { cv = document.createElement("canvas"); host.appendChild(cv); }
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  cv.style.width = `${W}px`; cv.style.height = `${H}px`;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  g.font = "10px system-ui, sans-serif";
  return [cv, g];
}

function drawCorner(ctx) {
  const host = document.getElementById("corner-host");
  // never narrower than CORNER_MIN, so the zoom always fits the empty upper triangle; a narrower panel scrolls
  const W = Math.max(CORNER_MIN, Math.min(host.clientWidth || 800, 940)), L = 62, B = 58, cell = (W - L - 4) / 8, H = 8 * cell + B + 4;
  const [cv, g] = canvasIn(host, W, H);
  cv.onclick = ev => {
    const r = cv.getBoundingClientRect(), j = Math.floor((ev.clientX - r.left - L) / cell), i = Math.floor((ev.clientY - r.top - 4) / cell);
    if (j >= 0 && j < 8 && i >= j && i < 8) { S.cell = [j, i]; writeHash(); render(); }
  };
  for (let i = 0; i < 8; i++) for (let j = 0; j <= i; j++) {
    const x0 = L + j * cell, y0 = 4 + i * cell;
    g.strokeStyle = css("--grid"); g.lineWidth = 1; g.strokeRect(x0 + 0.5, y0 + 0.5, cell - 1, cell - 1);
    drawCell(g, ctx, j, i, x0, y0, cell, 3);
    if (j === S.cell[0] && i === S.cell[1]) { g.strokeStyle = css("--ink"); g.lineWidth = 2; g.strokeRect(x0 + 1, y0 + 1, cell - 2, cell - 2); }
  }
  g.fillStyle = css("--muted");
  for (let c = 0; c < 8; c++) {
    const [lo, hi] = ctx.range[c], t = ticks(lo, hi, 2), inner = cell - 6;
    g.textAlign = "center"; g.textBaseline = "top";
    let right = -Infinity;
    for (const v of t) {
      const x = L + c * cell + 3 + (v - lo) / (hi - lo) * inner, text = tickText(v, t.step), half = g.measureText(text).width / 2;
      if (x - half > right + 3) { g.fillText(text, x, 4 + 8 * cell + 3); right = x + half; }
    }
    if (c > 0) {
      g.textAlign = "right"; g.textBaseline = "middle";
      for (const v of t) { const y = 4 + c * cell + 3 + (hi - v) / (hi - lo) * inner; g.fillText(tickText(v, t.step), L - 4, y); }
    }
  }
  g.fillStyle = css("--ink-2"); g.font = "12px system-ui, sans-serif";
  g.textAlign = "center"; g.textBaseline = "top";
  for (let c = 0; c < 8; c++) g.fillText(plain(labelOf(c, ctx.pooled)), L + (c + 0.5) * cell, 4 + 8 * cell + 18, cell - 2);
  for (let c = 1; c < 8; c++) {
    g.save(); g.translate(12, 4 + (c + 0.5) * cell); g.rotate(-Math.PI / 2); g.textBaseline = "middle";
    g.fillText(plain(labelOf(c, ctx.pooled)), 0, 0, cell - 2); g.restore();
  }
  // the zoom, labels included, in the top-right 4 × 4 cells: the largest block above the diagonal
  const size = Math.min(4 * cell - ZOOM_L, 4 * cell - ZOOM_B - 2);
  drawZoomAt(g, ctx, W - 4 - size, 6, size);
}

// the enlarged cell with its own ticks and labels, its box at (x0, y0); the labels take ZOOM_L to its left, ZOOM_B below
const ZOOM_L = 64, ZOOM_B = 44;
// the corner's smallest width: 8 cells of 57 px, so the zoom's box is at least 4 × 57 − ZOOM_L = 164 px
const CORNER_MIN = 62 + 4 + 8 * 57;
function drawZoomAt(g, ctx, x0, y0, size) {
  const [cx, cy] = S.cell;
  g.strokeStyle = css("--grid"); g.lineWidth = 1; g.strokeRect(x0 + 0.5, y0 + 0.5, size - 1, size - 1);
  const {X, Y} = drawCell(g, ctx, cx, cy, x0, y0, size, 6);
  g.font = "10px system-ui, sans-serif"; g.fillStyle = css("--muted");
  g.textAlign = "center"; g.textBaseline = "top";
  let right = -Infinity;
  const tx = ticks(...ctx.range[cx], 4);
  for (const v of tx) {
    const text = tickText(v, tx.step), half = g.measureText(text).width / 2;
    if (X(v) - half > right + 4) { g.fillText(text, X(v), y0 + size + 4); right = X(v) + half; }
  }
  if (cx !== cy) {
    g.textAlign = "right"; g.textBaseline = "middle";
    const ty = ticks(...ctx.range[cy], 4);
    for (const v of ty) g.fillText(tickText(v, ty.step), x0 - 5, Y(v));
  }
  g.fillStyle = css("--ink-2"); g.font = "12px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "top";
  g.fillText(plain(labelOf(cx, ctx.pooled)), x0 + size / 2, y0 + size + 22);
  g.save(); g.translate(x0 - ZOOM_L + 12, y0 + size / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle";
  g.fillText(cx === cy ? "density" : plain(labelOf(cy, ctx.pooled)), 0, 0); g.restore();
}

function table(ctx) {
  const j = S.inj, n = INJ[j].sources.length, have = name => INJ[j].sources[0].sets[name];
  const ref = have("mcmc") ? "mcmc" : "fisher", refText = ref === "mcmc" ? "MCMC" : "Fisher";
  const names = META.coordinates[S.view];
  let h = `<table><tr><th>source (SNR)</th>${names.map(t => `<th>${t}</th>`).join("")}</tr>`;
  for (let s = 0; s < n; s++) {
    const f = columns(j, s, "flow", S.view, false), r = columns(j, s, ref, S.view, false);
    const cls = S.src === s ? ' class="sel"' : "";
    h += `<tr${cls}><td>${s + 1} (${fmt(INJ[j].sources[s].snr, 1)})</td>${names.map((_, c) => { const v = width(f[c]) / width(r[c]); return `<td${v > 3 ? ' class="hi"' : ""}>${fmt(v)}</td>`; }).join("")}</tr>`;
  }
  h += "</table>";
  let detail = "";
  if (!ctx.pooled) {
    const s = S.src, truth = truthOf(j, s, S.view, false), rows = [["truth", c => sig(truth[c])]];
    const f = columns(j, s, "flow", S.view, false);
    rows.push(["flow median − truth, in flow widths", c => { const sorted = sortedFinite(f[c]); return fmt((quantile(sorted, 0.5) - truth[c]) / width(f[c])); }]);
    rows.push(["truth's rank in the flow draws", c => fmt(f[c].filter(v => v < truth[c]).length / f[c].length)]);
    for (const set of SETS) if (have(set.key)) { const cols = columns(j, s, set.key, S.view, false); rows.push([`${set.text} width`, c => sig(width(cols[c]))]); }
    if (have("mcmc")) { const m = columns(j, s, "mcmc", S.view, false); rows.push(["flow ÷ MCMC width", c => fmt(width(f[c]) / width(m[c]))]); }
    const fi = columns(j, s, "fisher", S.view, false); rows.push(["flow ÷ Fisher width", c => fmt(width(f[c]) / width(fi[c]))]);
    detail = `<h2 style="margin-top:16px">Source ${s + 1}</h2><table><tr><th></th>${names.map(t => `<th>${t}</th>`).join("")}</tr>` +
      rows.map(([k, fn]) => `<tr><td>${k}</td>${names.map((_, c) => `<td>${fn(c)}</td>`).join("")}</tr>`).join("") + "</table>";
  }
  document.getElementById("table-note").textContent = `Flow width over the ${refText} width per source and coordinate, in the ${S.view} coordinates (above 3 in colour).` + (ctx.pooled ? "" : " Below, the selected source in detail.");
  document.getElementById("table").innerHTML = h + detail;
}

function seg(id, items, active, onPick) {
  const host = document.getElementById(id);
  host.innerHTML = items.map((it, k) => `<button type="button" data-k="${k}" aria-pressed="${active(it)}">${it.html}</button>`).join("");
  host.querySelectorAll("button").forEach(b => b.onclick = () => { onPick(items[+b.dataset.k]); writeHash(); render(); });
}
function pickers() {
  seg("view", [{v: "physical", html: "physical (λ, β)"}, {v: "flow", html: "flow"}], it => it.v === S.view, it => { S.view = it.v; });
  seg("inj", INJ.map((w, k) => ({v: k, html: `q${w.quantile.toFixed(2)}<span class="sn">${fmt(w.snr, 0)}</span>`})), it => it.v === S.inj, it => { S.inj = it.v; });
  const src = INJ[S.inj].sources.map((x, k) => ({v: k, html: `${k + 1}<span class="sn">${fmt(x.snr, 0)}</span>`}));
  seg("src", [...src, {v: "all", html: `all<span class="sn">pooled</span>`}], it => it.v === S.src, it => { S.src = it.v; });
  seg("picks", Object.entries(META.picks).map(([k, [j, s]]) => ({j, s, html: `${k}<span class="sn">${fmt(INJ[j].sources[s].snr, 0)}</span>`})),
    it => it.j === S.inj && it.s === S.src, it => { S.inj = it.j; S.src = it.s; });
}
// the legend doubles as the toggles: a set cycles hidden → shown → shown with its draws as points; the truth toggles
function legend() {
  const sets = SETS.filter(x => INJ[S.inj].sources[0].sets[x.key]);
  const line = x => `<line x1="1" x2="21" y1="5" y2="5" stroke="var(${x.colour})" stroke-width="2"${x.dash ? ' stroke-dasharray="5 3"' : ""}/>`;
  const dots = x => [3, 8, 13, 18].map((d, k) => `<rect x="${d - 1}" y="${k % 2 ? 0 : 7}" width="2.4" height="2.4" fill="var(${x.colour})"/>`).join("");
  const host = document.getElementById("legend");
  host.innerHTML = sets.map(x => {
    const points = S.sets.includes(x.key) && S.points.includes(x.key);
    return `<button type="button" data-k="${x.key}" aria-pressed="${S.sets.includes(x.key)}" title="hide → show → show with points">` +
      `<svg width="22" height="10">${line(x)}${points ? dots(x) : ""}</svg>${x.text}${points ? " + points" : ""}</button>`;
  }).join("") + EXTRAS.map(x => `<button type="button" data-k="${x.key}" aria-pressed="${S.sets.includes(x.key)}">` +
    `<svg width="12" height="12"><rect x="3" y="3" width="6" height="6" fill="var(--truth)"/></svg>${x.text}</button>`).join("");
  host.querySelectorAll("button").forEach(b => b.onclick = () => {
    const k = b.dataset.k, shown = S.sets.includes(k), points = S.points.includes(k);
    if (k === "truth") S.sets = shown ? S.sets.filter(v => v !== k) : [...S.sets, k];
    else if (!shown) { S.sets = [...S.sets, k]; S.points = S.points.filter(v => v !== k); }
    else if (!points) S.points = [...S.points, k];
    else { S.sets = S.sets.filter(v => v !== k); S.points = S.points.filter(v => v !== k); }
    writeHash(); render();
  });
}
function tiles(ctx) {
  const w = INJ[S.inj], t = [
    ["window", `q${w.quantile.toFixed(2)}`, `SNR ${fmt(w.snr, 1)} · ${fmt(w.f_mhz, 3)} mHz`],
    ["window bins", `${w.start_bin}–${w.start_bin + w.window_bins}`, `${w.window_bins} bins`],
  ];
  if (!ctx.pooled) {
    const src = w.sources[S.src], phys = src.truth.physical;
    t.push(["source", `${S.src + 1} of ${w.sources.length}`, `SNR ${fmt(src.snr, 1)}`]);
    t.push(["f₀ (truth)", `${fmt(phys[0], 2)} bins`, `from the window start; f₀ = ${fmt((w.start_bin + phys[0]) / META.t_obs * 1e3, 6)} mHz`]);
    const f = columns(S.inj, S.src, "flow", "physical", false), m = columns(S.inj, S.src, "mcmc", "physical", false);
    t.push(["f₀ width, flow", `${sig(width(f[0]))} bins`, m ? `MCMC ${sig(width(m[0]))} bins` : "no MCMC"]);
  } else t.push(["sources", `${w.sources.length} pooled`, w.sources.map(x => fmt(x.snr, 0)).join(" · ")]);
  t.push(["flow draws off the axes", `${fmt(100 * ctx.off, 1)}%`, "any coordinate outside"]);
  document.getElementById("tiles").innerHTML = t.map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}
function render() {
  if (S.src !== "all" && S.src >= INJ[S.inj].sources.length) S.src = 0;
  pickers();
  const ctx = context();
  tiles(ctx);
  const src = ctx.pooled ? `all ${ctx.sources.length} sources pooled` : `source ${S.src + 1} (SNR ${fmt(INJ[S.inj].sources[S.src].snr, 1)})`;
  document.getElementById("corner-title").textContent = `q${INJ[S.inj].quantile.toFixed(2)}, ${src}, ${S.view} coordinates`;
  document.getElementById("corner-note").textContent = (ctx.pooled
    ? "Every source of the window in the same axes; squares are the truths."
    : "Axes hold the flow's central 99% (within 8 robust widths of its median), the reference's and the truth.") +
    ` Contours: 39% and 86% of each set; the flow's density is shaded.`;
  legend();
  drawCorner(ctx);
  table(ctx);
}

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has("view") && ["physical", "flow"].includes(h.get("view"))) S.view = h.get("view");
  if (h.has("inj")) S.inj = clamp(+h.get("inj") || 0, 0, INJ.length - 1);
  if (h.has("src")) S.src = h.get("src") === "all" ? "all" : clamp(+h.get("src") || 0, 0, 7);
  if (h.has("sets")) S.sets = h.get("sets").split(",").filter(Boolean);
  if (h.has("points")) S.points = h.get("points").split(",").filter(Boolean);
  if (h.has("cell")) { const [a, b] = h.get("cell").split(",").map(Number); if (a >= 0 && b >= a && b < 8) S.cell = [a, b]; }
  if (h.has("theme")) document.documentElement.dataset.theme = h.get("theme");
}
function writeHash() {
  const h = new URLSearchParams({view: S.view, inj: S.inj, src: S.src, sets: S.sets.join(","), points: S.points.join(","), cell: S.cell.join(",")});
  if (document.documentElement.dataset.theme) h.set("theme", document.documentElement.dataset.theme);
  history.replaceState(null, "", `#${h}`);
}

document.getElementById("title").textContent = `${META.run}: corners`;
document.getElementById("sub").textContent = `${META.config} at ${META.commit}, eval ${META.eval} (epoch ${META.epoch}, ${META.ode_steps} ODE steps, ${META.weights} weights). Draws per source: ` +
  Object.entries(META.draws).map(([k, v]) => `${k === "mcmc" ? "MCMC" : k === "fisher" ? "Fisher" : k} ${v}`).join(", ") + `. Layout ${META.layout}.`;
readHash();
let pending = false;
const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } };
addEventListener("resize", schedule);
addEventListener("hashchange", () => { readHash(); schedule(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
render();
</script>
</body>
</html>
"""

def corner_report(eval_dir: Path) -> Path:
    """eval_dir/corner.html: one interactive page of every eval injection's corners, physical and flow coordinates."""
    index, blob = corner_data(eval_dir)
    page = CORNER_PAGE
    # the layout's version on the page, so a stale build is told from the current page at a glance
    index["meta"]["layout"] = hashlib.sha1(page.encode()).hexdigest()[:8]
    page = page.replace("__TITLE__", f"{index['meta']['run']} corners")
    page = page.replace("__DATA__", json.dumps(index, separators=(",", ":")))
    page = page.replace("__BLOB__", base64.b64encode(blob).decode())
    path = eval_dir / "corner.html"
    path.write_text(page)
    return path


def significant(values, digits: int = 4) -> list:
    """values as nested lists rounded to `digits` significant figures, so their JSON stays short; NaN and inf kept."""
    v = np.asarray(values, np.float64)
    finite = np.isfinite(v) & (v != 0)
    # scale by an exact power of ten (multiplying or dividing by its sign), so the rounded floats print short
    shift = (
        digits - 1 - np.floor(np.log10(np.abs(np.where(finite, v, 1.0)))).astype(int)
    )
    up, down = 10.0 ** np.maximum(shift, 0), 10.0 ** np.maximum(-shift, 0)
    rounded = np.where(shift >= 0, np.round(v * up) / up, np.round(v / down) * down)
    return np.where(finite, rounded, v).tolist()


RUN_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding-block: 28px 64px; padding-inline: 16px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 4px; }
h3 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
a { color: var(--s1); }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 90ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 10px; max-width: 110ch; }
.layout { display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 20px; align-items: start; }
.side { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto; }
@media (max-width: 900px) { .layout { grid-template-columns: minmax(0, 1fr); } .side { position: static; max-height: none; } }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.content > .card + .card { margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 20px; font-weight: 600; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.stack { margin-bottom: 14px; }
.stack > .k { display: block; color: var(--ink-2); font-size: 13px; margin-bottom: 6px; }
.seg { display: grid; width: 100%; grid-template-columns: repeat(var(--cols), minmax(0, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { font: inherit; font-size: 13px; padding: 5px 6px; border: 0; background: var(--surface); color: var(--ink-2); cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; align-items: center; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px 20px; }
.chart { position: relative; min-width: 0; }
canvas { display: block; max-width: 100%; }
.cross { position: absolute; width: 1px; background: var(--ink-2); pointer-events: none; display: none; }
.slider { display: grid; grid-template-columns: 110px minmax(0, 1fr) 170px; gap: 10px; align-items: center; font-size: 13px; color: var(--ink-2); margin: 4px 0; }
.slider input { width: 100%; }
.slider output { font-variant-numeric: tabular-nums; color: var(--ink); }
#tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 6px 9px; font-size: 12px; line-height: 1.45; box-shadow: 0 2px 10px rgba(0,0,0,0.12); display: none; }
#tip table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
#tip td { padding: 0 0 0 10px; text-align: right; }
#tip td:first-child { padding: 0; text-align: left; color: var(--ink-2); }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
ul.links { margin: 0; padding-left: 18px; font-size: 13px; }
details { font-size: 13px; color: var(--ink-2); }
details summary { cursor: pointer; color: var(--ink); }
details li { margin-bottom: 4px; }
.banner { background: #c62828; color: #fff; padding: 8px 16px; font: 13px/1.4 system-ui, sans-serif; }
</style>
</head>
<body>
<div id="banner" class="banner" hidden></div>
<script>
window.onerror = (message, source, line) => {
  const b = document.getElementById("banner");
  b.hidden = false;
  b.textContent = `This page hit an error: ${message} (line ${line}).`;
};
</script>
<div id="tip"></div>
<main>
<h1 id="title"></h1>
<p class="sub" id="sub"></p>
<div class="layout">
<aside class="card side">
  <div class="stack"><span class="k">Step axis</span><div class="seg" id="xscale" style="--cols: 2"></div></div>
  <div class="stack"><span class="k">Per flow time, show</span><div class="seg" id="mode" style="--cols: 1"></div></div>
  <div class="stack"><span class="k">Evals</span><ul class="links" id="evals"></ul></div>
  <p class="note">Hover a chart for values; in the flow-time charts, click to pick the epoch or the time bin. The view lives in the URL.</p>
</aside>
<div class="content">
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <h2>Training</h2>
    <p class="note">Per chunk of steps, the median (line) and 10–90% (band) of the per-step values.</p>
    <div class="grid" id="train"></div>
  </div>
  <div class="card">
    <h2>Held out</h2>
    <p class="note">Once per epoch, on a fresh batch. The flow loss of the schedule-free average x (what eval samples from) against the raw iterate y; the backbone's weight RMS; the RMS of z − x, the schedule-free spread.</p>
    <div class="grid" id="heldout"></div>
  </div>
  <div class="card">
    <h2>Per flow time</h2>
    <p class="note" id="time-note"></p>
    <div class="legend" id="time-legend"></div>
    <div class="slider"><label for="epoch">epoch</label><input type="range" id="epoch" min="0" step="1"><output id="epoch-out"></output></div>
    <div class="slider"><label for="bin">time bin</label><input type="range" id="bin" min="0" step="1"><output id="bin-out"></output></div>
    <div class="grid" id="time"></div>
  </div>
  <div class="card">
    <details>
      <summary>What the curves are</summary>
      <ul>
        <li><b>flow</b>: the training loss, the mean squared velocity error over the batch; <b>x</b>: the mean squared error of the predicted x₁; <b>objective</b>: what the optimizer descends, Σ e·e<sup>−λ</sup> + λ over (time bin, group) with the learned λ, else the flow loss; <b>grad norm</b>: before clipping.</li>
        <li><b>Per flow time</b>: the held-out velocity error per coordinate group (f₀, M<sub>c</sub>, sky, z₊, z₋) on a batch spread evenly over the time bins, binned in log₁₀(1 − s), 1 − s the flow time left. Its x-space version multiplies by (1 − s)², the scale between a velocity and an x₁ error. With learned weights, e<sup>λ</sup> (dashed) is the error each bin's weight expects; at its optimum λ = ln e, so error ÷ e<sup>λ</sup> = 1 where λ has caught up.</li>
      </ul>
    </details>
  </div>
</div>
</div>
</main>
<script>const DATA = __DATA__;</script>
<script>
"use strict";
const META = DATA.meta, E = META.epochs, G = META.groups, NT = META.time_edges.length - 1;
const LOSSES = {flow: "flow loss", x: "x₁ error", objective: "objective", grad_norm: "grad norm"};
const HELDOUT = {w_rms: "backbone weight RMS", z_x: "RMS of z − x"};
const MODES = [
  {v: "velocity", html: "velocity error"},
  {v: "x", html: "x error, × (1 − s)²"},
  ...(META.has_lambda ? [{v: "ratio", html: "error ÷ e<sup>λ</sup>"}] : []),
];
const S = {xlog: false, mode: "velocity", epoch: E - 1, bin: 0};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function rgba(name, alpha) {
  const c = css(name), m = c.match(/^#([0-9a-f]{6})$/i);
  if (!m) return c;
  const n = parseInt(m[1], 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${alpha})`;
}
const sig = v => v === null || !isFinite(v) ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v.toPrecision(3)).toString();
const kilo = v => v >= 1e6 ? `${sig(v / 1e6)}M` : v >= 1e3 ? `${sig(v / 1e3)}k` : String(v);

function ticks(lo, hi, count) {
  const step0 = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(step0)), err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  out.step = step;
  return out;
}
const tickText = (v, step) => Math.abs(v) >= 1e5 ? v.toExponential(1) : v.toFixed(clamp(-Math.floor(Math.log10(step) + 1e-9), 0, 10));
const SUP = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"};
const decade = e => e >= -3 && e <= 4 ? String(+(10 ** e).toPrecision(1)) : "10" + String(e).replace(/./g, ch => SUP[ch]);
// ticks in plot space (log₁₀ on a log axis): decades, or 1-2-5 within two of them
function axisTicks(lo, hi, log, count) {
  if (!log) { const t = ticks(lo, hi, count); return t.map(v => ({at: v, text: tickText(v, t.step)})); }
  const a = Math.ceil(lo - 1e-9), b = Math.floor(hi + 1e-9);
  if (b - a >= 2) {
    const step = Math.max(1, Math.ceil((b - a) / count)), out = [];
    for (let e = a; e <= b; e += step) out.push({at: e, text: decade(e)});
    return out;
  }
  const out = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) for (const m of [1, 2, 5]) {
    const at = Math.log10(m) + e;
    if (at >= lo - 1e-9 && at <= hi + 1e-9) out.push({at, text: sig(m * 10 ** e)});
  }
  return out;
}
function canvasIn(host, W, H) {
  let cv = host.querySelector("canvas");
  if (!cv) { cv = document.createElement("canvas"); host.appendChild(cv); }
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  cv.style.width = `${W}px`; cv.style.height = `${H}px`;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  g.font = "11px system-ui, sans-serif";
  return [cv, g];
}
const tip = document.getElementById("tip");
function showTip(ev, html) {
  tip.innerHTML = html; tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${ev.clientX + 14 + w > innerWidth ? ev.clientX - w - 14 : ev.clientX + 14}px`;
  tip.style.top = `${clamp(ev.clientY - h / 2, 4, innerHeight - h - 4)}px`;
}
const hideTip = () => { tip.style.display = "none"; };
const row = (colour, name, value) => `<tr><td>${colour ? `<span class="key" style="background:var(${colour})"></span>` : ""}${name}</td><td>${value}</td></tr>`;

// one chart: layers of lines and bands over a shared x, hovered by the nearest x; charts of one sync group share the crosshair
let SYNC = {};
function plot(host, o) {
  const W = host.clientWidth || 400, H = o.height || 220, L = 52, R = 10, T = 8, B = 34;
  const [cv, g] = canvasIn(host, W, H);
  const fx = o.x.log ? v => v > 0 ? Math.log10(v) : NaN : v => v, fy = o.y.log ? v => v > 0 ? Math.log10(v) : NaN : v => v;
  const range = (values, f, fixed) => {
    if (fixed) return fixed.map(f);
    let lo = Infinity, hi = -Infinity;
    for (const v of values) { const t = f(v); if (isFinite(t)) { lo = Math.min(lo, t); hi = Math.max(hi, t); } }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
    const pad = 0.04 * (hi - lo);
    return [lo - pad, hi + pad];
  };
  const xs = o.layers.flatMap(l => l.x || []), ys = o.layers.flatMap(l => [...(l.y || []), ...(l.lo || []), ...(l.hi || [])]);
  const [x0, x1] = range(xs, fx, o.x.domain), [y0, y1] = range(ys, fy, o.y.domain);
  const X = v => L + (fx(v) - x0) / (x1 - x0) * (W - L - R), Y = v => T + (y1 - fy(v)) / (y1 - y0) * (H - T - B);
  // grid, ticks and labels
  g.strokeStyle = css("--grid"); g.lineWidth = 1; g.fillStyle = css("--muted");
  g.textAlign = "right"; g.textBaseline = "middle";
  for (const t of axisTicks(y0, y1, o.y.log, 4)) {
    const y = T + (y1 - t.at) / (y1 - y0) * (H - T - B);
    g.beginPath(); g.moveTo(L, y + 0.5); g.lineTo(W - R, y + 0.5); g.stroke(); g.fillText(t.text, L - 5, y);
  }
  g.textAlign = "center"; g.textBaseline = "top";
  let right = -Infinity;
  for (const t of axisTicks(x0, x1, o.x.log, 5)) {
    const x = L + (t.at - x0) / (x1 - x0) * (W - L - R), half = g.measureText(t.text).width / 2;
    g.beginPath(); g.moveTo(x + 0.5, T); g.lineTo(x + 0.5, H - B); g.stroke();
    if (x - half > right + 4) { g.fillText(t.text, x, H - B + 4); right = x + half; }
  }
  g.strokeStyle = css("--axis"); g.beginPath(); g.moveTo(L, H - B + 0.5); g.lineTo(W - R, H - B + 0.5); g.stroke();
  g.fillStyle = css("--ink-2"); g.font = "12px system-ui, sans-serif"; g.fillText(o.x.label, (L + W - R) / 2, H - 16);
  g.save(); g.beginPath(); g.rect(L, T, W - L - R, H - T - B); g.clip();
  for (const l of o.layers) {
    if (l.type === "band") {
      g.fillStyle = rgba(l.colour, l.alpha ?? 0.18); g.beginPath();
      let open = false;
      const ok = k => isFinite(fx(l.x[k])) && isFinite(fy(l.lo[k])) && isFinite(fy(l.hi[k]));
      for (let k = 0; k <= l.x.length; k++) {
        if (k < l.x.length && ok(k)) {
          if (!open) { open = k; g.moveTo(X(l.x[k]), Y(l.hi[k])); } else g.lineTo(X(l.x[k]), Y(l.hi[k]));
          if (open === false) open = k;
        } else if (open !== false) {
          for (let m = k - 1; m >= open; m--) g.lineTo(X(l.x[m]), Y(l.lo[m]));
          g.closePath(); open = false;
        }
      }
      g.fill();
    } else if (l.type === "line") {
      g.strokeStyle = css(l.colour); g.lineWidth = l.width || 2; g.setLineDash(l.dash || []); g.lineJoin = "round";
      g.beginPath();
      let pen = false;
      for (let k = 0; k < l.x.length; k++) {
        const x = X(l.x[k]), y = Y(l.y[k]);
        if (!isFinite(x) || !isFinite(y)) { pen = false; continue; }
        pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true;
      }
      g.stroke(); g.setLineDash([]);
    } else if (l.type === "vline") {
      g.strokeStyle = css(l.colour || "--ink"); g.lineWidth = 1; g.setLineDash(l.dash || [3, 3]);
      g.beginPath(); g.moveTo(X(l.at) + 0.5, T); g.lineTo(X(l.at) + 0.5, H - B); g.stroke(); g.setLineDash([]);
    } else if (l.type === "hline") {
      g.strokeStyle = css(l.colour || "--ink-2"); g.lineWidth = 1; g.setLineDash(l.dash || [4, 3]);
      g.beginPath(); g.moveTo(L, Y(l.at) + 0.5); g.lineTo(W - R, Y(l.at) + 0.5); g.stroke(); g.setLineDash([]);
    }
  }
  g.restore();
  // hover: the nearest x of o.hover.x; the sync group's other charts follow with their crosshair
  let cross = host.querySelector(".cross");
  if (!cross) { cross = document.createElement("div"); cross.className = "cross"; host.appendChild(cross); }
  cross.style.top = `${T}px`; cross.style.height = `${H - T - B}px`; cross.style.display = "none";
  const xsHover = o.hover.x;
  const nearest = px => {
    let best = -1, d = Infinity;
    xsHover.forEach((v, k) => { const e = Math.abs(X(v) - px); if (e < d) { d = e; best = k; } });
    return best;
  };
  const chart = {mark: v => {
    if (v === null) { cross.style.display = "none"; return; }
    const k = nearest(X(v));
    if (k < 0 || !isFinite(X(xsHover[k]))) return;
    cross.style.left = `${X(xsHover[k])}px`; cross.style.display = "block";
  }};
  if (o.sync) (SYNC[o.sync] ||= []).push(chart);
  const peers = () => o.sync ? SYNC[o.sync] : [chart];
  cv.onmousemove = ev => {
    const r = cv.getBoundingClientRect(), px = ev.clientX - r.left;
    if (px < L || px > W - R) { peers().forEach(c => c.mark(null)); hideTip(); return; }
    const k = nearest(px);
    peers().forEach(c => c.mark(xsHover[k]));
    showTip(ev, o.hover.html(k));
  };
  cv.onmouseleave = () => { peers().forEach(c => c.mark(null)); hideTip(); };
  cv.style.cursor = o.click ? "pointer" : "crosshair";
  cv.onclick = o.click ? ev => { const r = cv.getBoundingClientRect(); o.click(nearest(ev.clientX - r.left)); } : null;
}
function chartIn(parent, id, title) {
  let box = document.getElementById(id);
  if (!box) {
    box = document.createElement("div");
    box.innerHTML = `<h3>${title}</h3><div class="chart" id="${id}"></div>`;
    parent.appendChild(box);
    box = document.getElementById(id);
  }
  return box;
}

function training() {
  const t = DATA.train, parent = document.getElementById("train"), x = {log: S.xlog, label: "step"};
  for (const name of META.loss_names) {
    const q = t.losses[name], label = LOSSES[name] || name;
    plot(chartIn(parent, `train-${name}`, label), {
      x, y: {log: q.lo.every(v => !(v <= 0)), label}, sync: "step",
      layers: [{type: "band", x: t.step, lo: q.lo, hi: q.hi, colour: "--s1"}, {type: "line", x: t.step, y: q.median, colour: "--s1", width: 1.5}],
      hover: {x: t.step, html: k => `step ${kilo(t.step[k])}<table>${row("", "90%", sig(q.hi[k]))}${row("--s1", "median", sig(q.median[k]))}${row("", "10%", sig(q.lo[k]))}</table>`},
    });
  }
}
function heldout() {
  const h = DATA.heldout, step = DATA.epoch_step, parent = document.getElementById("heldout"), x = {log: S.xlog, label: "step"};
  const pairs = [["heldout_x", "--s1", "x (average)"], ["heldout_y", "--s2", "y (iterate)"]].filter(([k]) => h[k]);
  plot(chartIn(parent, "heldout-loss", "held-out flow loss"), {
    x, y: {log: pairs.every(([k]) => h[k].every(v => !(v <= 0))), label: "flow loss"}, sync: "step",
    layers: pairs.map(([k, colour]) => ({type: "line", x: step, y: h[k], colour})),
    hover: {x: step, html: k => `epoch ${k + 1}, step ${kilo(step[k])}<table>${pairs.map(([n, c, label]) => row(c, label, sig(h[n][k]))).join("")}</table>`},
  });
  for (const [name, label] of Object.entries(HELDOUT)) {
    if (!h[name]) continue;
    plot(chartIn(parent, `heldout-${name}`, label), {
      x, y: {log: h[name].every(v => !(v <= 0)), label}, sync: "step",
      layers: [{type: "line", x: step, y: h[name], colour: "--s1"}],
      hover: {x: step, html: k => `epoch ${k + 1}, step ${kilo(step[k])}<table>${row("--s1", label, sig(h[name][k]))}</table>`},
    });
  }
}

// the flow-time view: error [epoch][bin][group] in the chosen mode, and e^λ in the same units (null in the ratio mode)
const CENTRES = META.time_edges.slice(0, -1).map((e, k) => 10 ** ((e + META.time_edges[k + 1]) / 2));
function shown(e, b, grp) {
  const err = DATA.time.error[e][b][grp], lam = META.has_lambda ? Math.exp(DATA.time.log_var[e][b][grp]) : NaN;
  const scale = S.mode === "x" ? CENTRES[b] ** 2 : 1;
  return S.mode === "ratio" ? [err / lam, NaN] : [err * scale, lam * scale];
}
const COLOURS = ["--s1", "--s2", "--s3", "--s4", "--s5"];
const GROUP_TEXT = {f0: "f₀", Mc: "M<sub>c</sub>", sky: "sky", "z+": "z₊", "z-": "z₋"};
function flowTime() {
  const parent = document.getElementById("time"), lam = META.has_lambda && S.mode !== "ratio";
  const yLabel = {velocity: "velocity error", x: "x error", ratio: "error ÷ e^λ"}[S.mode];
  const bins = [...CENTRES.keys()], epochs = [...Array(E).keys()], step = DATA.epoch_step;
  const layers = (index, xsOf, pick) => G.flatMap((grp, gi) => {
    const v = index.map(i => pick(i, gi));
    return [
      {type: "line", x: xsOf, y: v.map(p => p[0]), colour: COLOURS[gi]},
      ...(lam ? [{type: "line", x: xsOf, y: v.map(p => p[1]), colour: COLOURS[gi], dash: [5, 3], width: 1.5}] : []),
    ];
  });
  const table = vals => `<table>${G.map((grp, gi) => row(COLOURS[gi], GROUP_TEXT[grp] || grp, lam ? `${sig(vals[gi][0])} <span style="color:var(--muted)">e^λ ${sig(vals[gi][1])}</span>` : sig(vals[gi][0]))).join("")}</table>`;
  const ratio = S.mode === "ratio" ? [{type: "hline", at: 1}] : [];
  const binText = b => `1 − s ∈ [${decade(META.time_edges[b])}, ${decade(META.time_edges[b + 1])}]`;
  plot(chartIn(parent, "time-at-epoch", "over flow time, at the picked epoch"), {
    x: {log: true, label: "1 − s (flow time left)"}, y: {log: true, label: yLabel},
    layers: [...ratio, ...layers(bins, CENTRES, (b, gi) => shown(S.epoch, b, gi)), {type: "vline", at: CENTRES[S.bin]}],
    hover: {x: CENTRES, html: b => `epoch ${S.epoch + 1}, ${binText(b)}${table(G.map((_, gi) => shown(S.epoch, b, gi)))}`},
    click: b => { S.bin = b; writeHash(); render(); },
  });
  plot(chartIn(parent, "time-over-steps", "over training, in the picked time bin"), {
    x: {log: S.xlog, label: "step"}, y: {log: true, label: yLabel},
    layers: [...ratio, ...layers(epochs, step, (e, gi) => shown(e, S.bin, gi)), {type: "vline", at: step[S.epoch]}],
    hover: {x: step, html: e => `epoch ${e + 1}, step ${kilo(step[e])}, ${binText(S.bin)}${table(G.map((_, gi) => shown(e, S.bin, gi)))}`},
    click: e => { S.epoch = e; writeHash(); render(); },
  });
  document.getElementById("time-note").innerHTML = `Held-out error per coordinate group and log-spaced bin of flow time; ` +
    (S.mode === "ratio" ? "the ratio is 1 where a bin's learned weight has caught up with its error." : lam ? "solid: measured, dashed: e<sup>λ</sup>, the error the learned weight expects." : "no learned weights in this run.") +
    " Click the left chart to pick the time bin, the right one to pick the epoch.";
  document.getElementById("time-legend").innerHTML = G.map((grp, gi) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(${COLOURS[gi]})" stroke-width="2"/></svg>${GROUP_TEXT[grp] || grp}</span>`).join("") +
    (lam ? `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(--ink-2)" stroke-width="1.5" stroke-dasharray="5 3"/></svg>e<sup>λ</sup></span>` : "");
  const ep = document.getElementById("epoch"), bin = document.getElementById("bin");
  ep.max = E - 1; ep.value = S.epoch; bin.max = NT - 1; bin.value = S.bin;
  document.getElementById("epoch-out").textContent = `${S.epoch + 1} of ${E}, step ${kilo(step[S.epoch])}`;
  document.getElementById("bin-out").textContent = binText(S.bin);
}
document.getElementById("epoch").oninput = ev => { S.epoch = +ev.target.value; writeHash(); schedule(); };
document.getElementById("bin").oninput = ev => { S.bin = +ev.target.value; writeHash(); schedule(); };

function seg(id, items, active, onPick) {
  const host = document.getElementById(id);
  host.innerHTML = items.map((it, k) => `<button type="button" data-k="${k}" aria-pressed="${active(it)}">${it.html}</button>`).join("");
  host.querySelectorAll("button").forEach(b => b.onclick = () => { onPick(items[+b.dataset.k]); writeHash(); render(); });
}
function tiles() {
  const q = DATA.train.losses, h = DATA.heldout, last = a => a[a.length - 1], step = DATA.epoch_step, t = [];
  t.push(["trained", `${E} epochs`, `${kilo(last(step))} steps`]);
  if (q.flow) t.push(["flow loss, last chunk", sig(last(q.flow.median)), `10–90%: ${sig(last(q.flow.lo))} – ${sig(last(q.flow.hi))}`]);
  if (h.heldout_x) {
    const best = h.heldout_x.reduce((b, v, k) => v < h.heldout_x[b] ? k : b, 0);
    t.push(["held-out flow loss (x)", sig(last(h.heldout_x)), `best ${sig(h.heldout_x[best])} at epoch ${best + 1}`]);
  }
  if (q.x) t.push(["x₁ error, last chunk", sig(last(q.x.median)), "median"]);
  if (q.grad_norm) t.push(["grad norm, last chunk", sig(last(q.grad_norm.median)), `90%: ${sig(last(q.grad_norm.hi))}`]);
  document.getElementById("tiles").innerHTML = t.map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}
function render() {
  SYNC = {};
  seg("xscale", [{v: false, html: "linear"}, {v: true, html: "log"}], it => it.v === S.xlog, it => { S.xlog = it.v; });
  seg("mode", MODES, it => it.v === S.mode, it => { S.mode = it.v; });
  tiles();
  training();
  heldout();
  flowTime();
}

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has("x")) S.xlog = h.get("x") === "log";
  if (h.has("mode") && MODES.some(m => m.v === h.get("mode"))) S.mode = h.get("mode");
  if (h.has("epoch")) S.epoch = clamp(+h.get("epoch") - 1 || 0, 0, E - 1);
  if (h.has("bin")) S.bin = clamp(+h.get("bin") || 0, 0, NT - 1);
  if (h.has("theme")) document.documentElement.dataset.theme = h.get("theme");
}
function writeHash() {
  const h = new URLSearchParams({x: S.xlog ? "log" : "linear", mode: S.mode, epoch: S.epoch + 1, bin: S.bin});
  if (document.documentElement.dataset.theme) h.set("theme", document.documentElement.dataset.theme);
  history.replaceState(null, "", `#${h}`);
}

document.getElementById("title").textContent = `${META.run}: training`;
document.getElementById("sub").textContent = `${META.config} at ${META.commit}, ${E} epochs of ${kilo(META.steps_per_epoch)} steps. Layout ${META.layout}.`;
document.getElementById("evals").innerHTML = META.evals.length ? META.evals.map(e => `<li>${e.name}: ` +
  [e.validation ? `<a href="eval/${e.name}/validation.html">validation</a>` : "", e.corner ? `<a href="eval/${e.name}/corner.html">corners</a>` : ""].filter(Boolean).join(" · ") + "</li>").join("") : "<li>none yet</li>";
// the last bin with data at the picked epoch, by default
S.bin = Math.max(0, DATA.time.error[E - 1].findIndex(r => r.some(Number.isFinite)));
readHash();
let pending = false;
const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } };
addEventListener("resize", schedule);
addEventListener("hashchange", () => { readHash(); schedule(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
render();
</script>
</body>
</html>
"""

def run_report(run_dir: Path) -> Path:
    """run_dir/run.html: one interactive page of history.npz's training and held-out losses and per-time-bin error and λ."""
    import yaml

    h = dict(np.load(run_dir / "history.npz"))
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    steps, losses = h["steps"], h["losses"]
    n_epochs, logged = losses.shape[:2]

    # each epoch's logged steps in chunks, ~1000 over the run, each kept as its 10/50/90% quantiles
    chunks = max(1, min(logged // 20, round(1000 / n_epochs)))
    size = logged // chunks
    chunked = rearrange(losses[:, : chunks * size], "e (c p) l -> (e c) p l", c=chunks)
    lo, median, hi = np.nanpercentile(chunked, [10, 50, 90], axis=1)
    epoch_steps = np.diff(steps, prepend=0)
    chunk_step = steps[:, None] - epoch_steps[:, None] * (
        1 - (np.arange(chunks) + 0.5) / chunks
    )

    evals = [
        {
            "name": path.parent.name,
            "corner": (path.parent / "draws.npz").exists(),
            "validation": any(
                (path.parent / name).exists()
                for name in (
                    "validation_scores.npz",
                    "scorecard.npz",
                    "mcmc_compare.npz",
                )
            ),
        }
        for path in sorted((run_dir / "eval").glob("*/eval.yaml"))
    ]
    loss_names = [str(name) for name in h["loss_names"]]
    index = {
        "meta": {
            "run": run_dir.name,
            "config": config.get("name", ""),
            "commit": str(config.get("commit", ""))[:7],
            "epochs": int(n_epochs),
            "steps_per_epoch": int(epoch_steps[-1]),
            "groups": [str(g) for g in h["groups"]],
            "time_edges": significant(h["time_edges"]),
            "has_lambda": bool(np.isfinite(h["log_var"]).any()),
            "loss_names": loss_names,
            "evals": evals,
        },
        "train": {
            "step": rearrange(chunk_step, "e c -> (e c)").round().astype(int).tolist(),
            "losses": {
                name: {
                    "lo": significant(lo[:, c]),
                    "median": significant(median[:, c]),
                    "hi": significant(hi[:, c]),
                }
                for c, name in enumerate(loss_names)
            },
        },
        "heldout": {
            str(name): significant(h["heldout"][:, c])
            for c, name in enumerate(h["heldout_names"])
        },
        "epoch_step": steps.astype(int).tolist(),
        "time": {
            "error": significant(h["bin_error"]),
            "log_var": significant(h["log_var"]),
        },
    }
    page = RUN_PAGE
    # the layout's version on the page, so a stale build is told from the current page at a glance
    index["meta"]["layout"] = hashlib.sha1(page.encode()).hexdigest()[:8]
    page = page.replace("__TITLE__", f"{run_dir.name} training")
    page = page.replace("__DATA__", json.dumps(index, separators=(",", ":")))
    path = run_dir / "run.html"
    path.write_text(page)
    return path


VALIDATION_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
  --q1: #86b6ef; --q2: #3987e5; --q3: #1c5cab; --q4: #0d366b; --bad: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
    --q1: #1c5cab; --q2: #3987e5; --q3: #86b6ef; --q4: #cfe2f8; --bad: #e25c5c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  --q1: #1c5cab; --q2: #3987e5; --q3: #86b6ef; --q4: #cfe2f8; --bad: #e25c5c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding-block: 28px 64px; padding-inline: 16px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 4px; }
h3 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
a { color: var(--s1); }
.sub { color: var(--ink-2); margin: 0 0 20px; max-width: 90ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 10px; max-width: 110ch; }
.layout { display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 20px; align-items: start; }
.side { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto; }
@media (max-width: 900px) { .layout { grid-template-columns: minmax(0, 1fr); } .side { position: static; max-height: none; } }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.content > .card + .card { margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 20px; font-weight: 600; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.stack { margin-bottom: 14px; }
.stack > .k { display: block; color: var(--ink-2); font-size: 13px; margin-bottom: 6px; }
.seg { display: grid; width: 100%; grid-template-columns: repeat(var(--cols), minmax(0, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { font: inherit; font-size: 13px; padding: 5px 6px; border: 0; background: var(--surface); color: var(--ink-2); cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; align-items: center; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px 20px; }
.grid.small { grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
.chart { position: relative; min-width: 0; }
canvas { display: block; max-width: 100%; }
.cross { position: absolute; width: 1px; background: var(--ink-2); pointer-events: none; display: none; }
.ring { position: absolute; width: 10px; height: 10px; margin: -5px 0 0 -5px; border: 2px solid var(--ink); border-radius: 50%; pointer-events: none; display: none; }
#tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 6px 9px; font-size: 12px; line-height: 1.45; box-shadow: 0 2px 10px rgba(0,0,0,0.12); display: none; }
#tip table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
#tip td { padding: 0 0 0 10px; text-align: right; }
#tip td:first-child { padding: 0; text-align: left; color: var(--ink-2); }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.scroll { overflow-x: auto; }
table.data { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
table.data th, table.data td { padding: 3px 9px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
table.data th { color: var(--ink-2); font-weight: 600; }
table.data th:first-child, table.data td:first-child { text-align: left; }
table.heat td { min-width: 64px; cursor: pointer; }
table.heat td.empty { color: var(--muted); cursor: default; }
table.heat td.picked { outline: 2px solid var(--ink); outline-offset: -2px; }
table.heat tr.picked th { color: var(--ink); }
.facts { font-size: 13px; color: var(--ink-2); margin: 0; padding-left: 18px; }
ul.links { margin: 0; padding-left: 18px; font-size: 13px; }
details { font-size: 13px; color: var(--ink-2); }
details summary { cursor: pointer; color: var(--ink); }
details li { margin-bottom: 4px; }
.banner { background: #c62828; color: #fff; padding: 8px 16px; font: 13px/1.4 system-ui, sans-serif; }
</style>
</head>
<body>
<div id="banner" class="banner" hidden></div>
<script>
window.onerror = (message, source, line) => {
  const b = document.getElementById("banner");
  b.hidden = false;
  b.textContent = `This page hit an error: ${message} (line ${line}).`;
};
</script>
<div id="tip"></div>
<main>
<h1 id="title"></h1>
<p class="sub" id="sub"></p>
<div class="layout">
<aside class="card side">
  <div class="stack"><span class="k">SNR band (draws' median SNR)</span><div class="seg" id="band" style="--cols: 3"></div></div>
  <div class="stack" id="mcmc-pick"><span class="k">Against the MCMC, show</span><div class="seg" id="mcmc-mode" style="--cols: 1"></div></div>
  <div class="stack"><span class="k">Pages</span><ul class="links" id="links"></ul></div>
  <div class="stack"><span class="k">Eval</span><ul class="facts" id="facts"></ul></div>
  <p class="note">Click a cell of the γ table to plot its quantity and column below. Hover a chart for values. The view lives in the URL.</p>
</aside>
<div class="content">
  <div class="tiles" id="tiles"></div>
  <div class="card" id="calibration-card">
    <h2>Rank calibration</h2>
    <p class="note" id="calibration-note"></p>
    <div class="scroll" id="calibration"></div>
  </div>
  <div class="card" id="ecdf-card">
    <h2 id="ecdf-title">ECDF − z</h2>
    <p class="note" id="ecdf-note"></p>
    <div class="legend" id="ecdf-legend"></div>
    <div class="grid small" id="ecdf"></div>
  </div>
  <div class="card" id="sharpness-card">
    <h2>Sharpness</h2>
    <p class="note">The f₀ posterior's robust width over the monochromatic Fisher bound √3 / (π SNR) at the draws' median SNR, per slot (sources loudest first); lines are running medians, the shade is the picked SNR band.</p>
    <div class="legend" id="sharpness-legend"></div>
    <div class="grid" id="sharpness"></div>
  </div>
  <div class="card" id="scores-card">
    <h2>Proper scores</h2>
    <p class="note" id="scores-note"></p>
    <div class="scroll" id="scores"></div>
  </div>
  <div class="card" id="scorecard-card">
    <h2>f₀ scorecard</h2>
    <p class="note" id="scorecard-note"></p>
    <div class="legend" id="scorecard-legend"></div>
    <div class="grid" id="scorecard"></div>
    <div class="scroll" id="scorecard-table" style="margin-top: 12px"></div>
  </div>
  <div class="card" id="mcmc-card">
    <h2>Against the MCMC</h2>
    <p class="note" id="mcmc-note"></p>
    <div class="grid small" id="mcmc"></div>
    <div class="scroll" id="mcmc-table" style="margin-top: 12px"></div>
  </div>
  <div class="card">
    <details>
      <summary>What the checks are</summary>
      <ul>
        <li><b>Rank calibration</b> (SBC): for a calibrated posterior the truth's rank among the draws is uniform. Per quantity and column, the p of Säilynoja+22's γ test of that uniformity; ✕ marks p below the level. Columns are the slots (sources loudest first), each source's quantity on its own, and the joint draw, a statistic of all sources at once. <b>TARP</b> is the rank in distance to a random reference point; <b>band depth</b> and <b>average rank</b> rank how central the truth is among the draws, so there a high rank is a central truth, a too-wide posterior.</li>
        <li><b>Mean rank − ½</b>: &gt; 0 when the truth sits above the draws. <b>Share in the middle half</b>: &gt; ½ when the posterior is too wide, &lt; ½ when too narrow.</li>
        <li><b>ECDF − z</b>: the ranks' empirical CDF minus the uniform's at fractional rank z, with the simultaneous band the γ test accepts; a curve above zero at small z has too many low ranks (the truth below the draws).</li>
        <li><b>SNR bands</b> are of the draws' median SNR, a function of the data alone, so stratifying by them keeps each stratum's ranks uniform under calibration.</li>
        <li><b>Proper scores</b>: the energy score of the draws against the truth, per source and for the joint draw, and the CRPS per coordinate; lower is better, and only comparable between evals of the same injections.</li>
        <li><b>Scorecard</b>: random and eval injections scored on f₀ alone by true SNR, the eval job's quick look; <b>against the MCMC</b>: the eval injections' narrowband MCMC, one source free and the others at their truth.</li>
      </ul>
    </details>
  </div>
</div>
</div>
</main>
<script>const DATA = __DATA__;</script>
<script>
"use strict";
const META = DATA.meta, C = DATA.calibration, SC = DATA.scorecard, MC = DATA.mcmc, NB = META.bands.length;
const BAND_COLOURS = ["--q1", "--q2", "--q3", "--q4", "--ink"];
const COLOURS = ["--s1", "--s2", "--s3", "--s4", "--s5"];
const MCMC_MODES = [
  {v: 1, html: "flow / MCMC width"},
  {v: 2, html: "flow W₁ / MCMC sd"},
  {v: 0, html: "MCMC width"},
];
const S = {band: NB - 1, q: 0, col: 0, mcmc: 1};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function rgba(name, alpha) {
  const c = css(name), m = c.match(/^#([0-9a-f]{6})$/i);
  if (!m) return c;
  const n = parseInt(m[1], 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${alpha})`;
}
const sig = v => v === null || v === undefined || !isFinite(v) ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v.toPrecision(3)).toString();
const signed = v => v === null || !isFinite(v) ? "–" : (v >= 0 ? "+" : "") + sig(v);
const nameText = name => name.replaceAll("_", " ");
const median = a => {
  const s = a.filter(Number.isFinite).sort((x, y) => x - y), n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN;
};
const inBand = (v, b) => v >= META.band_edges[b][0] && (META.band_edges[b][1] === null || v < META.band_edges[b][1]);

function ticks(lo, hi, count) {
  const step0 = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(step0)), err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  out.step = step;
  return out;
}
const tickText = (v, step) => Math.abs(v) >= 1e5 ? v.toExponential(1) : v.toFixed(clamp(-Math.floor(Math.log10(step) + 1e-9), 0, 10));
const SUP = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"};
const decade = e => e >= -3 && e <= 4 ? String(+(10 ** e).toPrecision(1)) : "10" + String(e).replace(/./g, ch => SUP[ch]);
// ticks in plot space (log₁₀ on a log axis): decades, or 1-2-5 within two of them
function axisTicks(lo, hi, log, count) {
  if (!log) { const t = ticks(lo, hi, count); return t.map(v => ({at: v, text: tickText(v, t.step)})); }
  const a = Math.ceil(lo - 1e-9), b = Math.floor(hi + 1e-9);
  if (b - a >= 2) {
    const step = Math.max(1, Math.ceil((b - a) / count)), out = [];
    for (let e = a; e <= b; e += step) out.push({at: e, text: decade(e)});
    return out;
  }
  const out = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) for (const m of [1, 2, 5]) {
    const at = Math.log10(m) + e;
    if (at >= lo - 1e-9 && at <= hi + 1e-9) out.push({at, text: sig(m * 10 ** e)});
  }
  return out;
}
function canvasIn(host, W, H) {
  let cv = host.querySelector("canvas");
  if (!cv) { cv = document.createElement("canvas"); host.appendChild(cv); }
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  cv.style.width = `${W}px`; cv.style.height = `${H}px`;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  g.font = "11px system-ui, sans-serif";
  return [cv, g];
}
const tip = document.getElementById("tip");
function showTip(ev, html) {
  tip.innerHTML = html; tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${ev.clientX + 14 + w > innerWidth ? ev.clientX - w - 14 : ev.clientX + 14}px`;
  tip.style.top = `${clamp(ev.clientY - h / 2, 4, innerHeight - h - 4)}px`;
}
const hideTip = () => { tip.style.display = "none"; };
const row = (colour, name, value) => `<tr><td>${colour ? `<span class="key" style="background:var(${colour})"></span>` : ""}${name}</td><td>${value}</td></tr>`;
const lineKey = (colour, label, dash) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(${colour})" stroke-width="2"${dash ? ` stroke-dasharray="${dash}"` : ""}/></svg>${label}</span>`;
const dotKey = (colour, label) => `<span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(${colour})"/></svg>${label}</span>`;

// one chart of layers (bands, lines, points, rules, x spans); hovered by the nearest x (o.hover.x) or the nearest point (o.hover.points)
let SYNC = {};
function plot(host, o) {
  const W = host.clientWidth || 400, H = o.height || 220, L = 52, R = 10, T = 8, B = 34;
  const [cv, g] = canvasIn(host, W, H);
  const fx = o.x.log ? v => v > 0 ? Math.log10(v) : NaN : v => v, fy = o.y.log ? v => v > 0 ? Math.log10(v) : NaN : v => v;
  const range = (values, f, fixed) => {
    if (fixed) return fixed.map(f);
    let lo = Infinity, hi = -Infinity;
    for (const v of values) { const t = f(v); if (isFinite(t)) { lo = Math.min(lo, t); hi = Math.max(hi, t); } }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
    const pad = 0.04 * (hi - lo);
    return [lo - pad, hi + pad];
  };
  const data = o.layers.filter(l => !l.decor);
  const xs = data.flatMap(l => l.x || []), ys = data.flatMap(l => [...(l.y || []), ...(l.lo || []), ...(l.hi || [])]);
  const [x0, x1] = range(xs, fx, o.x.domain), [y0, y1] = range(ys, fy, o.y.domain);
  const X = v => L + (fx(v) - x0) / (x1 - x0) * (W - L - R), Y = v => T + (y1 - fy(v)) / (y1 - y0) * (H - T - B);
  // grid, ticks and labels
  g.strokeStyle = css("--grid"); g.lineWidth = 1; g.fillStyle = css("--muted");
  g.textAlign = "right"; g.textBaseline = "middle";
  for (const t of axisTicks(y0, y1, o.y.log, 4)) {
    const y = T + (y1 - t.at) / (y1 - y0) * (H - T - B);
    g.beginPath(); g.moveTo(L, y + 0.5); g.lineTo(W - R, y + 0.5); g.stroke(); g.fillText(t.text, L - 5, y);
  }
  g.textAlign = "center"; g.textBaseline = "top";
  let right = -Infinity;
  for (const t of axisTicks(x0, x1, o.x.log, 5)) {
    const x = L + (t.at - x0) / (x1 - x0) * (W - L - R), half = g.measureText(t.text).width / 2;
    g.beginPath(); g.moveTo(x + 0.5, T); g.lineTo(x + 0.5, H - B); g.stroke();
    if (x - half > right + 4) { g.fillText(t.text, x, H - B + 4); right = x + half; }
  }
  g.strokeStyle = css("--axis"); g.beginPath(); g.moveTo(L, H - B + 0.5); g.lineTo(W - R, H - B + 0.5); g.stroke();
  g.fillStyle = css("--ink-2"); g.font = "12px system-ui, sans-serif"; g.fillText(o.x.label, (L + W - R) / 2, H - 16);
  g.save(); g.beginPath(); g.rect(L, T, W - L - R, H - T - B); g.clip();
  for (const l of o.layers) {
    if (l.type === "span") {
      const a = isFinite(X(l.from)) ? X(l.from) : L, b = l.to === null || !isFinite(X(l.to)) ? W - R : X(l.to);
      g.fillStyle = rgba(l.colour || "--ink-2", l.alpha ?? 0.07); g.fillRect(a, T, b - a, H - T - B);
    } else if (l.type === "band") {
      g.fillStyle = rgba(l.colour, l.alpha ?? 0.18); g.beginPath();
      let open = false;
      const ok = k => isFinite(fx(l.x[k])) && isFinite(fy(l.lo[k])) && isFinite(fy(l.hi[k]));
      for (let k = 0; k <= l.x.length; k++) {
        if (k < l.x.length && ok(k)) {
          if (open === false) { open = k; g.moveTo(X(l.x[k]), Y(l.hi[k])); } else g.lineTo(X(l.x[k]), Y(l.hi[k]));
        } else if (open !== false) {
          for (let m = k - 1; m >= open; m--) g.lineTo(X(l.x[m]), Y(l.lo[m]));
          g.closePath(); open = false;
        }
      }
      g.fill();
    } else if (l.type === "line") {
      g.strokeStyle = css(l.colour); g.lineWidth = l.width || 2; g.setLineDash(l.dash || []); g.lineJoin = "round";
      g.beginPath();
      let pen = false;
      for (let k = 0; k < l.x.length; k++) {
        const x = X(l.x[k]), y = Y(l.y[k]);
        if (!isFinite(x) || !isFinite(y)) { pen = false; continue; }
        pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true;
      }
      g.stroke(); g.setLineDash([]);
    } else if (l.type === "points") {
      g.fillStyle = rgba(l.colour, l.alpha ?? 0.5);
      const r = l.size || 2.5;
      for (let k = 0; k < l.x.length; k++) {
        const x = X(l.x[k]), y = Y(l.y[k]);
        if (!isFinite(x) || !isFinite(y)) continue;
        g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
      }
    } else if (l.type === "hline") {
      g.strokeStyle = css(l.colour || "--ink-2"); g.lineWidth = 1; g.setLineDash(l.dash || [4, 3]);
      g.beginPath(); g.moveTo(L, Y(l.at) + 0.5); g.lineTo(W - R, Y(l.at) + 0.5); g.stroke(); g.setLineDash([]);
    }
  }
  g.restore();
  if (o.hover.points) {
    // the nearest point of the hoverable point layers, ringed
    let ring = host.querySelector(".ring");
    if (!ring) { ring = document.createElement("div"); ring.className = "ring"; host.appendChild(ring); }
    ring.style.display = "none";
    const layers = o.layers.filter(l => l.type === "points" && l.hover);
    cv.onmousemove = ev => {
      const r = cv.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
      let best = null, d = 400;
      layers.forEach((l, li) => l.x.forEach((v, k) => {
        const e = (X(v) - px) ** 2 + (Y(l.y[k]) - py) ** 2;
        if (e < d) { d = e; best = [li, k]; }
      }));
      if (!best) { ring.style.display = "none"; hideTip(); return; }
      const [li, k] = best, l = layers[li];
      ring.style.left = `${X(l.x[k])}px`; ring.style.top = `${Y(l.y[k])}px`; ring.style.display = "block";
      showTip(ev, l.hover(k));
    };
    cv.onmouseleave = () => { ring.style.display = "none"; hideTip(); };
    cv.style.cursor = "crosshair";
    return;
  }
  // hover: the nearest x of o.hover.x; the sync group's other charts follow with their crosshair
  let cross = host.querySelector(".cross");
  if (!cross) { cross = document.createElement("div"); cross.className = "cross"; host.appendChild(cross); }
  cross.style.top = `${T}px`; cross.style.height = `${H - T - B}px`; cross.style.display = "none";
  const xsHover = o.hover.x;
  const nearest = px => {
    let best = -1, d = Infinity;
    xsHover.forEach((v, k) => { const e = Math.abs(X(v) - px); if (e < d) { d = e; best = k; } });
    return best;
  };
  const chart = {mark: v => {
    if (v === null) { cross.style.display = "none"; return; }
    const k = nearest(X(v));
    if (k < 0 || !isFinite(X(xsHover[k]))) return;
    cross.style.left = `${X(xsHover[k])}px`; cross.style.display = "block";
  }};
  if (o.sync) (SYNC[o.sync] ||= []).push(chart);
  const peers = () => o.sync ? SYNC[o.sync] : [chart];
  cv.onmousemove = ev => {
    const r = cv.getBoundingClientRect(), px = ev.clientX - r.left;
    if (px < L || px > W - R) { peers().forEach(c => c.mark(null)); hideTip(); return; }
    const k = nearest(px);
    peers().forEach(c => c.mark(xsHover[k]));
    showTip(ev, o.hover.html(k));
  };
  cv.onmouseleave = () => { peers().forEach(c => c.mark(null)); hideTip(); };
  cv.style.cursor = "crosshair";
}
function chartIn(parent, id, title) {
  let box = document.getElementById(id);
  if (!box) {
    box = document.createElement("div");
    box.innerHTML = `<h3></h3><div class="chart" id="${id}"></div>`;
    parent.appendChild(box);
    box = document.getElementById(id);
  }
  box.previousElementSibling.innerHTML = title;
  return box;
}
const table = (header, rows) => `<table class="data"><thead><tr>${header.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
  rows.map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("") + "</tbody></table>";

// the γ table of the picked band: p per quantity and column, shaded by how far below the level it falls
function calibration() {
  const p = C.p[S.band], level = META.alpha;
  document.getElementById("calibration-note").innerHTML = `Per quantity and column, the p of the γ test of uniform ranks of the truth among the ${C.k - 1} draws, ` +
    `for the sources whose draws' median SNR is in ${META.bands[S.band]}; ✕ marks p &lt; ${level}. Hover a cell for its count, the mean rank's offset and the share of ranks in the middle half.`;
  const cell = (q, c) => {
    const v = p[q][c];
    if (v === null) return `<td class="empty">–</td>`;
    const strength = v < level ? clamp(0.3 + 0.35 * Math.log10(level / Math.max(v, 1e-3)), 0.3, 1) : 0;
    const style = strength ? `background:${rgba("--bad", strength)};${strength > 0.6 ? "color:#fff" : ""}` : "";
    return `<td data-q="${q}" data-c="${c}" style="${style}" class="${q === S.q && c === S.col ? "picked" : ""}">${v.toFixed(2)}${v < level ? " ✕" : ""}</td>`;
  };
  const host = document.getElementById("calibration");
  host.innerHTML = `<table class="data heat"><thead><tr><th>quantity</th>${C.columns.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
    C.names.map((name, q) => `<tr class="${q === S.q ? "picked" : ""}"><th>${nameText(name)}</th>${C.columns.map((_, c) => cell(q, c)).join("")}</tr>`).join("") +
    "</tbody></table>";
  host.querySelectorAll("td[data-q]").forEach(td => {
    const q = +td.dataset.q, c = +td.dataset.c;
    td.onclick = () => { S.q = q; S.col = c; writeHash(); render(); };
    td.onmousemove = ev => showTip(ev, `${nameText(C.names[q])}, ${C.columns[c]}<table>${row("", "p", p[q][c].toFixed(3))}${row("", "sources", C.n[S.band][c])}` +
      `${row("", "mean rank − ½", signed(C.bias[S.band][q][c]))}${row("", "share in the middle half", sig(C.middle[S.band][q][c]))}</table>`);
    td.onmouseleave = hideTip;
  });
}

// the ranks of a quantity in column c (a slot, or the joint draw last) for the sources in band b
function ranksIn(name, c, b) {
  if (c < C.n_sources) return C.rank[name] ? C.rank[name].flatMap((r, n) => inBand(C.snr_hat[n][c], b) ? [r[c]] : []) : null;
  return C.joint_rank[name] ? C.joint_rank[name].filter((_, n) => inBand(C.joint_snr_hat[n], b)) : null;
}
const Z = C ? Array.from({length: C.k - 1}, (_, i) => (i + 1) / C.k) : [];
function ecdfMinusZ(ranks) {
  if (!ranks || ranks.length < 10) return null;
  const counts = new Array(C.k).fill(0);
  for (const r of ranks) counts[r]++;
  let below = 0;
  return Z.map((z, i) => { below += counts[i]; return below / ranks.length - z; });
}
function ecdf() {
  const name = C.names[S.q], parent = document.getElementById("ecdf");
  document.getElementById("ecdf-title").textContent = `ECDF − z: ${nameText(name)}`;
  document.getElementById("ecdf-note").innerHTML = `The rank ECDF minus the uniform's per SNR band (the picked band thicker), with the picked band's simultaneous ${Math.round(100 * (1 - META.alpha))}% band shaded. ` +
    "A curve above zero at small z has too many low ranks. Click a cell of the γ table to change the quantity.";
  document.getElementById("ecdf-legend").innerHTML = META.bands.map((label, b) => lineKey(BAND_COLOURS[b], `SNR ${label}`)).join("") +
    `<span><span class="key" style="background:${rgba("--ink-2", 0.18)}"></span>${Math.round(100 * (1 - META.alpha))}% band</span>`;
  C.columns.forEach((column, c) => {
    const curves = META.bands.map((_, b) => ecdfMinusZ(ranksIn(name, c, b))), env = C.envelope[S.band][c];
    const host = chartIn(parent, `ecdf-${c}`, `${column}${C.p[S.band][S.q][c] === null ? "" : `, p ${C.p[S.band][S.q][c].toFixed(2)}`}`);
    host.parentElement.style.display = curves.some(Boolean) ? "" : "none";
    if (!curves.some(Boolean)) return;
    plot(host, {
      x: {log: false, label: "fractional rank z", domain: [0, 1]}, y: {log: false}, sync: "ecdf",
      layers: [
        ...(env && curves[S.band] ? [{type: "band", x: Z, lo: env[0], hi: env[1], colour: "--ink-2"}] : []),
        {type: "hline", at: 0, decor: true},
        ...curves.flatMap((v, b) => v && b !== S.band ? [{type: "line", x: Z, y: v, colour: BAND_COLOURS[b], width: 1.2}] : []),
        ...(curves[S.band] ? [{type: "line", x: Z, y: curves[S.band], colour: BAND_COLOURS[S.band], width: 2.5}] : []),
      ],
      hover: {x: Z, html: k => `${column}, z = ${sig(Z[k])}<table>${META.bands.map((label, b) => curves[b] ? row(BAND_COLOURS[b], `SNR ${label}`, signed(curves[b][k])) : "").join("")}</table>`},
    });
  });
}

function sharpness() {
  const parent = document.getElementById("sharpness"), [lo, hi] = META.band_edges[S.band];
  document.getElementById("sharpness-legend").innerHTML = C.columns.slice(0, C.n_sources).map((label, s) => dotKey(COLOURS[s % 5], label)).join("") + lineKey("--ink-2", "Fisher bound", "4 3");
  const xs = s => C.snr_hat.map(r => r[s]), ys = s => C.width_ratio.map(r => r[s]);
  plot(chartIn(parent, "sharpness-chart", "f₀ width / monochromatic Fisher bound, by the draws' median SNR"), {
    x: {log: true, label: "draws' median SNR"}, y: {log: true}, height: 360,
    layers: [
      ...(S.band < NB - 1 ? [{type: "span", from: lo, to: hi, decor: true}] : []),
      ...C.columns.slice(0, C.n_sources).map((label, s) => ({
        type: "points", x: xs(s), y: ys(s), colour: COLOURS[s % 5], alpha: 0.35,
        hover: k => `${label}, injection ${k}<table>${row(COLOURS[s % 5], "draws' median SNR", sig(C.snr_hat[k][s]))}${row("", "width / Fisher bound", sig(C.width_ratio[k][s]))}</table>`,
      })),
      ...C.running.map((m, s) => ({type: "line", x: m.map(p => p[0]), y: m.map(p => p[1]), colour: COLOURS[s % 5], width: 2})),
      {type: "hline", at: 1, decor: true},
    ],
    hover: {points: true},
  });
}

function scores() {
  const rows = C.proper[S.band];
  document.getElementById("scores-note").innerHTML = `Medians over the sources whose draws' median SNR is in ${META.bands[S.band]} (lower is better); the CRPS and width only per slot.`;
  document.getElementById("scores").innerHTML = table(["median", ...C.columns], rows.map(r => [r[0], ...r.slice(1).map(v => r[0] === "sources" ? String(v) : sig(v))]));
}

function scorecard() {
  const parent = document.getElementById("scorecard"), found = SC.width.map(w => w < SC.found_bins);
  document.getElementById("scorecard-note").innerHTML = `Every source of ${SC.n_random} random and the eval injections (epoch ${SC.epoch}, ${SC.ode_steps} ODE steps), by true SNR. ` +
    `A source is found when its f₀ width is under ${SC.found_bins} bin; in the table widths, × ideal and offsets are medians over found sources, the 68% / 95% coverage and the mean rank of the truth (½ is centred) over all.`;
  document.getElementById("scorecard-legend").innerHTML = dotKey("--s1", "random") + dotKey("--s2", "eval") + lineKey("--ink-2", "Fisher bound / found threshold / ±1", "4 3");
  const sets = [["random", "--s1", k => !SC.is_eval[k]], ["eval", "--s2", k => SC.is_eval[k]]];
  const pick = (keep, f) => SC.snr.flatMap((_, k) => keep(k) ? [f(k)] : []);
  const hover = k => `${SC.is_eval[k] ? "eval" : "random"} injection ${SC.injection[k]}<table>${row("", "SNR", sig(SC.snr[k]))}${row("", "width [bins]", sig(SC.width[k]))}` +
    `${row("", "offset [bins]", signed(SC.offset[k]))}${row("", "offset / width", signed(SC.offset[k] / SC.width[k]))}${row("", "rank", sig(SC.rank[k]))}</table>`;
  const snr = SC.snr.filter(v => v > 0), lo = Math.min(...snr), hi = Math.max(...snr);
  const bound = Array.from({length: 40}, (_, i) => lo * (hi / lo) ** (i / 39));
  plot(chartIn(parent, "scorecard-width", "f₀ width [bins]"), {
    x: {log: true, label: "source SNR"}, y: {log: true}, height: 320,
    layers: [
      ...sets.map(([, colour, keep]) => {
        const idx = pick(keep, k => k);
        return {type: "points", x: idx.map(k => SC.snr[k]), y: idx.map(k => SC.width[k]), colour, alpha: 0.55, hover: i => hover(idx[i])};
      }),
      {type: "line", x: bound, y: bound.map(v => Math.sqrt(3) / (Math.PI * v)), colour: "--ink-2", width: 1.5, dash: [4, 3], decor: true},
      {type: "hline", at: SC.found_bins, decor: true},
    ],
    hover: {points: true},
  });
  plot(chartIn(parent, "scorecard-offset", "found sources' f₀ offset / width"), {
    x: {log: true, label: "source SNR"}, y: {log: false}, height: 320,
    layers: [
      ...sets.map(([, colour, keep]) => {
        const idx = pick(k => keep(k) && found[k], k => k);
        return {type: "points", x: idx.map(k => SC.snr[k]), y: idx.map(k => SC.offset[k] / SC.width[k]), colour, alpha: 0.55, hover: i => hover(idx[i])};
      }),
      {type: "hline", at: 1, decor: true}, {type: "hline", at: -1, decor: true}, {type: "hline", at: 0, dash: [], decor: true},
    ],
    hover: {points: true},
  });
  document.getElementById("scorecard-table").innerHTML = table(["set", "SNR", "n", "found", "width", "× ideal", "offset", "in 68%", "in 95%", "mean rank"],
    SC.table.map(([set, band, n, ...v]) => [set, band, String(n), ...v.map((x, i) => i === 3 ? signed(x) : sig(x))]));
}

function mcmc() {
  const parent = document.getElementById("mcmc"), mode = MCMC_MODES.find(m => m.v === S.mcmc);
  document.getElementById("mcmc-note").innerHTML = `The ${MC.sources.length} eval sources with a narrowband MCMC (one source free, the others at their truth): per coordinate the flow's robust width over the MCMC's, ` +
    "the 1-Wasserstein distance between them over the MCMC's sd, or the MCMC's width; angles folded, dotted lines at 1. The sidebar picks the quantity.";
  const snr = MC.sources.map(s => s.snr);
  MC.columns.forEach((column, c) => {
    const y = MC.sources.map(s => s.values[S.mcmc][c]);
    plot(chartIn(parent, `mcmc-${c}`, `${column}: ${mode.html}`), {
      x: {log: true, label: "source SNR"}, y: {log: y.every(v => !(v <= 0))}, height: 200,
      layers: [
        ...(S.mcmc ? [{type: "hline", at: 1, decor: true}] : []),
        {type: "points", x: snr, y, colour: "--s1", alpha: 0.8, size: 3.5,
          hover: k => `${MC.sources[k].label}<table>${row("", "SNR", sig(snr[k]))}${MCMC_MODES.map(m => row(m.v === S.mcmc ? "--s1" : "", m.html, sig(MC.sources[k].values[m.v][c]))).join("")}</table>`},
      ],
      hover: {points: true},
    });
  });
  document.getElementById("mcmc-table").innerHTML = table(["median over sources", ...MC.columns],
    MCMC_MODES.map(m => [m.html, ...MC.columns.map((_, c) => sig(median(MC.sources.map(s => s.values[m.v][c]))))]));
}

function seg(id, items, active, onPick) {
  const host = document.getElementById(id);
  host.innerHTML = items.map((it, k) => `<button type="button" data-k="${k}" aria-pressed="${active(it)}">${it.html}</button>`).join("");
  host.querySelectorAll("button").forEach(b => b.onclick = () => { onPick(items[+b.dataset.k]); writeHash(); render(); });
}
function tiles() {
  const t = [];
  if (C) {
    const cells = C.p[S.band].flat().filter(v => v !== null), failed = cells.filter(v => v < META.alpha).length;
    t.push(["prior injections", String(C.n_injections), `${C.k - 1} draws each, epoch ${C.epoch}`]);
    t.push([`γ tests failed, SNR ${META.bands[S.band]}`, `${failed} of ${cells.length}`, `p < ${META.alpha}; ${Math.round(META.alpha * cells.length * 10) / 10} expected by chance`]);
    const ratio = C.width_ratio.flatMap((r, n) => r.filter((_, s) => inBand(C.snr_hat[n][s], S.band)));
    t.push([`f₀ width / Fisher bound, SNR ${META.bands[S.band]}`, sig(median(ratio)), `median of ${ratio.length} sources`]);
  }
  if (SC) {
    const n = SC.width.length, found = SC.width.filter(w => w < SC.found_bins).length;
    t.push(["scorecard: found", sig(found / n), `${found} of ${n} sources, width < ${SC.found_bins} bin`]);
  }
  if (MC) t.push(["f₀ flow / MCMC width", sig(median(MC.sources.map(s => s.values[1][0]))), `median of ${MC.sources.length} sources`]);
  document.getElementById("tiles").innerHTML = t.map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}
function render() {
  SYNC = {};
  seg("band", META.bands.map((label, b) => ({v: b, html: label})), it => it.v === S.band, it => { S.band = it.v; });
  if (MC) seg("mcmc-mode", MCMC_MODES, it => it.v === S.mcmc, it => { S.mcmc = it.v; });
  tiles();
  if (C) { calibration(); ecdf(); sharpness(); scores(); }
  if (SC) scorecard();
  if (MC) mcmc();
}

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has("band")) S.band = clamp(META.bands.indexOf(h.get("band")), 0, NB - 1);
  if (C && h.has("q") && C.names.includes(h.get("q"))) S.q = C.names.indexOf(h.get("q"));
  if (C && h.has("col") && C.columns.includes(h.get("col"))) S.col = C.columns.indexOf(h.get("col"));
  if (h.has("mcmc") && MCMC_MODES.some(m => String(m.v) === h.get("mcmc"))) S.mcmc = +h.get("mcmc");
  if (h.has("theme")) document.documentElement.dataset.theme = h.get("theme");
}
function writeHash() {
  const h = new URLSearchParams({band: META.bands[S.band], mcmc: S.mcmc});
  if (C) { h.set("q", C.names[S.q]); h.set("col", C.columns[S.col]); }
  if (document.documentElement.dataset.theme) h.set("theme", document.documentElement.dataset.theme);
  history.replaceState(null, "", `#${h}`);
}

document.getElementById("title").textContent = `${META.run}: validation`;
document.getElementById("sub").textContent = [META.config && `${META.config} at ${META.commit}`, `eval ${META.eval}`, `layout ${META.layout}`].filter(Boolean).join(" · ");
document.getElementById("links").innerHTML = [META.links.corner ? `<li><a href="corner.html">corners</a></li>` : "", META.links.run ? `<li><a href="../../run.html">training</a></li>` : ""].join("") || "<li>none</li>";
document.getElementById("facts").innerHTML = Object.entries(META.stage).map(([k, v]) => `<li>${k.replaceAll("_", " ")}: ${v}</li>`).join("") || "<li>no eval.yaml</li>";
for (const [id, present] of [["calibration-card", C], ["ecdf-card", C], ["sharpness-card", C], ["scores-card", C], ["scorecard-card", SC], ["mcmc-card", MC], ["mcmc-pick", MC]])
  document.getElementById(id).hidden = !present;
readHash();
let pending = false;
const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } };
addEventListener("resize", schedule);
addEventListener("hashchange", () => { readHash(); schedule(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
render();
</script>
</body>
</html>
"""

def validation_report(eval_dir: Path) -> Path:
    """eval_dir/validation.html: one interactive page of the eval's rank calibration, sharpness, proper scores,
    f0 scorecard and MCMC comparison, from whichever of their files exist."""
    import yaml

    from .validate import ALPHA, COORDS, SNR_BANDS, gamma_test

    run_dir = eval_dir.parents[1]
    config = (
        yaml.safe_load((run_dir / "config.yaml").read_text())
        if (run_dir / "config.yaml").exists()
        else {}
    )
    stage = (
        yaml.safe_load((eval_dir / "eval.yaml").read_text())
        if (eval_dir / "eval.yaml").exists()
        else {}
    )
    bands = [*SNR_BANDS, (0, np.inf)]
    in_band = lambda snr, lo, hi: (snr >= lo) & (snr < hi)

    calibration = None
    if (eval_dir / "validation_scores.npz").exists():
        scores = dict(np.load(eval_dir / "validation_scores.npz"))
        rank = {
            k.removeprefix("rank_"): v
            for k, v in scores.items()
            if k.startswith("rank_")
        }
        joint_rank = {
            k.removeprefix("joint_rank_"): v
            for k, v in scores.items()
            if k.startswith("joint_rank_")
        }
        k = int(scores["draws"]) + 1
        snr_hat, joint_snr_hat = scores["snr_hat"], scores["joint_snr_hat"]
        n_injections, n_sources = snr_hat.shape
        names = list(rank) + [name for name in joint_rank if name not in rank]
        # per band and column (the slots, then the joint draw): which injections, and the γ test of every quantity
        masks = [
            [
                *(in_band(snr_hat[:, s], lo, hi) for s in range(n_sources)),
                in_band(joint_snr_hat, lo, hi),
            ]
            for lo, hi in bands
        ]
        p, bias, middle, envelope = [], [], [], []
        for band in masks:
            cells = [[None] * (n_sources + 1) for _ in names]
            band_bias = [[None] * (n_sources + 1) for _ in names]
            band_middle = [[None] * (n_sources + 1) for _ in names]
            band_envelope = [None] * (n_sources + 1)
            for q, name in enumerate(names):
                for c, mask in enumerate(band):
                    if c < n_sources:
                        ranks = rank[name][mask, c] if name in rank else None
                    else:
                        ranks = joint_rank[name][mask] if name in joint_rank else None
                    if ranks is None or len(ranks) < 10:
                        continue
                    p_value, limits = gamma_test(ranks, k)
                    cells[q][c] = round(float(p_value), 4)
                    band_bias[q][c] = round(float(ranks.mean() / (k - 1) - 0.5), 4)
                    band_middle[q][c] = round(
                        float(np.mean((ranks >= k / 4) & (ranks < 3 * k / 4))), 4
                    )
                    band_envelope[c] = significant(limits, 3)
            p.append(cells)
            bias.append(band_bias)
            middle.append(band_middle)
            envelope.append(band_envelope)

        # the f0 width ratio's running median per slot, over log-spaced bins of the draws' median SNR
        running = []
        for s in range(n_sources):
            x, y = snr_hat[:, s], scores["width_ratio"][:, s]
            keep = (x > 0) & (y > 0) & np.isfinite(y)
            if not keep.any():
                running.append([])
                continue
            edges = np.geomspace(x[keep].min(), x[keep].max() * (1 + 1e-9), 13)
            which = np.digitize(x[keep], edges)
            running.append(
                [
                    significant(
                        [np.median(x[keep][which == b]), np.median(y[keep][which == b])]
                    )
                    for b in range(1, len(edges))
                    if (which == b).sum() >= 5
                ]
            )

        # per band, the proper scores' medians per column: energy per slot and joint, CRPS per coordinate, f0 width
        median_in = lambda v, mask: float(np.median(v[mask])) if mask.any() else None
        proper = []
        for band in masks:
            slots, joint = band[:n_sources], band[n_sources]
            rows = [
                ["sources", *(int(m.sum()) for m in band)],
                [
                    "energy score",
                    *(
                        median_in(scores["energy"][:, s], m)
                        for s, m in enumerate(slots)
                    ),
                    median_in(scores["joint_energy"], joint),
                ],
                *(
                    [
                        f"CRPS {name}",
                        *(
                            median_in(scores["crps"][:, s, c], m)
                            for s, m in enumerate(slots)
                        ),
                        None,
                    ]
                    for c, name in enumerate(COORDS)
                ),
                [
                    "f₀ width / Fisher bound",
                    *(
                        median_in(scores["width_ratio"][:, s], m)
                        for s, m in enumerate(slots)
                    ),
                    None,
                ],
            ]
            proper.append(
                [
                    [
                        row[0],
                        *(
                            v if v is None or isinstance(v, int) else significant(v)
                            for v in row[1:]
                        ),
                    ]
                    for row in rows
                ]
            )

        calibration = {
            "k": k,
            "epoch": int(scores["epoch"]) if "epoch" in scores else None,
            "n_injections": int(n_injections),
            "n_sources": int(n_sources),
            "names": names,
            "columns": [f"slot {s + 1}" for s in range(n_sources)] + ["joint"],
            "p": p,
            "n": [[int(m.sum()) for m in band] for band in masks],
            "bias": bias,
            "middle": middle,
            "envelope": envelope,
            "rank": {name: v.tolist() for name, v in rank.items()},
            "joint_rank": {name: v.tolist() for name, v in joint_rank.items()},
            "snr_hat": significant(snr_hat, 3),
            "joint_snr_hat": significant(joint_snr_hat, 3),
            "width_ratio": significant(scores["width_ratio"], 3),
            "running": running,
            "proper": proper,
        }

    scorecard = None
    if (eval_dir / "scorecard.npz").exists():
        from .scorecard import FOUND_BINS

        sc = dict(np.load(eval_dir / "scorecard.npz"))
        # per set and true-SNR band, as scorecard.summary prints it: widths and offsets over found sources only
        table = []
        for set_name, which in [
            ("all", np.ones_like(sc["is_eval"])),
            ("eval", sc["is_eval"]),
            ("random", ~sc["is_eval"]),
        ]:
            for (lo, hi), label in zip(
                bands, ["0–15", "15–40", "40–100", "≥ 100", "all"]
            ):
                m = which & in_band(sc["snr"], lo, hi)
                if not m.any():
                    continue
                found = m & (sc["width"] < FOUND_BINS)
                med = lambda v: (
                    float(np.median(v[found])) if found.any() else float("nan")
                )
                values = [
                    found.sum() / m.sum(),
                    med(sc["width"]),
                    med(sc["width"] / sc["ideal"]),
                    med(sc["offset"]),
                    sc["in68"][m].mean(),
                    sc["in95"][m].mean(),
                    sc["rank"][m].mean(),
                ]
                table.append([set_name, label, int(m.sum()), *significant(values, 3)])
        scorecard = {
            "found_bins": FOUND_BINS,
            "epoch": int(sc["epoch"]),
            "ode_steps": int(sc["ode_steps"]),
            "n_random": int(sc["n_random"]),
            "snr": significant(sc["snr"]),
            "width": significant(sc["width"]),
            "offset": significant(sc["offset"]),
            "rank": significant(sc["rank"], 3),
            "injection": sc["injection"].astype(int).tolist(),
            "is_eval": sc["is_eval"].astype(bool).tolist(),
            "table": table,
        }

    mcmc = None
    if (eval_dir / "mcmc_compare.npz").exists():
        from .mcmc import COLUMNS

        compare = dict(np.load(eval_dir / "mcmc_compare.npz"))
        source_snr = (
            np.load(eval_dir / "draws.npz")["source_snr"]
            if (eval_dir / "draws.npz").exists()
            else None
        )
        sources = []
        for key, values in compare.items():
            j, s = map(int, key.split("_"))
            snr = float(source_snr[j, s]) if source_snr is not None else float("nan")
            sources.append(
                {
                    "label": f"injection {j}, source {s + 1}",
                    "snr": round(snr, 2),
                    "values": significant(values),
                }
            )
        mcmc = {
            "columns": COLUMNS,
            "sources": sorted(sources, key=lambda source: -source["snr"]),
        }

    index = {
        "meta": {
            "run": run_dir.name,
            "config": config.get("name", ""),
            "commit": str(config.get("commit", ""))[:7],
            "eval": eval_dir.name,
            "alpha": ALPHA,
            "bands": ["0–15", "15–40", "40–100", "≥ 100", "all"],
            "band_edges": [[lo, hi if np.isfinite(hi) else None] for lo, hi in bands],
            "stage": {str(key): str(value) for key, value in stage.items()},
            "links": {
                "corner": (eval_dir / "draws.npz").exists(),
                "run": (run_dir / "history.npz").exists(),
            },
        },
        "calibration": calibration,
        "scorecard": scorecard,
        "mcmc": mcmc,
    }
    page = VALIDATION_PAGE
    # the layout's version on the page, so a stale build is told from the current page at a glance
    index["meta"]["layout"] = hashlib.sha1(page.encode()).hexdigest()[:8]
    page = page.replace("__TITLE__", f"{run_dir.name} validation")
    page = page.replace("__DATA__", json.dumps(index, separators=(",", ":")))
    path = eval_dir / "validation.html"
    path.write_text(page)
    return path


def eval_pages(eval_dir: Path) -> list[Path]:
    """corner.html from draws.npz (with mcmc.npz's chains when `mcmc report` has run), and validation.html from the
    validation, scorecard and MCMC comparison files."""
    pages = []
    if (eval_dir / "draws.npz").exists():
        pages.append(corner_report(eval_dir))
    if any(
        (eval_dir / name).exists()
        for name in ("validation_scores.npz", "scorecard.npz", "mcmc_compare.npz")
    ):
        pages.append(validation_report(eval_dir))
    return pages


def main() -> int:
    """Write every page of the given run dirs (their evals too) or eval dirs, on the CPU."""
    import jax

    # the only JAX here is the corner report's chart conversion: no accelerator needed
    jax.config.update("jax_platforms", "cpu")

    pages = []
    for target in map(Path, sys.argv[1:]):
        if (target / "config.yaml").exists():
            pages += [run_report(target)] if (target / "history.npz").exists() else []
            for eval_dir in sorted((target / "eval").glob("*/eval.yaml")):
                pages += eval_pages(eval_dir.parent)
        elif (target / "eval.yaml").exists():
            pages += eval_pages(target)
    if not pages:
        print("nothing to draw: give a run dir or an eval dir", file=sys.stderr)
        return 1
    for page in pages:
        print(page, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
