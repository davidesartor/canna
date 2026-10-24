"""strub-source.html from data/: the Strub et al. verification binary through this repo's simulator and likelihood."""

import datetime
import json
from pathlib import Path

import numpy as np

REPORT = Path(__file__).parents[1]
DATA = REPORT / "data"
HIST_BINS = 60


def sig(a, digits: int = 4):
    """Nested lists of `digits` significant figures, NaN as null."""
    a = np.asarray(a, np.float64)
    flat = [None if not np.isfinite(v) else float(f"{v:.{digits}g}") for v in a.ravel()]
    return np.array(flat, dtype=object).reshape(a.shape).tolist()


def histogram(samples: dict, lo: float, hi: float) -> dict:
    edges = np.linspace(lo, hi, HIST_BINS + 1)
    return dict(
        edges=sig(edges),
        **{k: sig(np.histogram(v, edges, density=True)[0]) for k, v in samples.items()},
    )


def strub_record(strub: dict, seed0) -> dict:
    scalars = {k: v for k, v in strub.items() if not isinstance(v, (list, dict))}
    lists = {
        k: strub[k]
        for k in (
            "sweep_rho_obs_list",
            "sweep_rho_mle_list",
            "sweep_sigma_f0_nhz_list",
            "sweep_off_mode_fraction_list",
        )
    }
    f0 = {"strub": seed0["strub_f0_nhz"], "box": seed0["box_f0_nhz"]}
    cos_iota = {"strub": seed0["strub_cos_iota"], "box": seed0["box_cos_iota"]}
    lo, hi = np.percentile(np.concatenate(list(f0.values())), [0.2, 99.8])
    edge = max(-lo, hi)
    pct = lambda a, q: sig(np.percentile(a, q), 5)
    return dict(
        scalars,
        **lists,
        hist_f0=histogram(f0, -edge, edge),
        hist_cos_iota=histogram(cos_iota, -1.0, 1.0),
        f0_q=dict(
            strub=pct(f0["strub"], [16, 50, 84]), box=pct(f0["box"], [16, 50, 84])
        ),
        cos_iota_q=dict(
            strub=pct(cos_iota["strub"], [16, 50, 84]),
            box=pct(cos_iota["box"], [16, 50, 84]),
        ),
    )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strub source check</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --ref: #52514e;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --ref: #c3c2b7;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
a { color: var(--s1); }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 24px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.card + .card { margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: auto; display: inline-block; flex: none; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; margin: 0 0 16px; }
.controls .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 12px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
.banner { background: #c62828; color: #fff; padding: 8px 16px; font: 13px/1.4 system-ui, sans-serif; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 500; }
#t-strub td:last-child, #t-strub th:last-child { text-align: left; white-space: normal; min-width: 220px; }
ul.means { padding-left: 20px; margin: 8px 0 0; max-width: 80ch; }
ul.means li { margin-bottom: 8px; }
</style>
</head>
<body>
<div id="banner" class="banner" hidden></div>
<script>
window.onerror = (message, source, line) => {
  const b = document.getElementById("banner");
  b.hidden = false;
  b.textContent = `This page hit an error: ${message} (line ${line}). Some cards may be blank.`;
};
</script>
<main>
<h1>Does the pipeline reproduce a published LISA posterior?</h1>
<p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>How do its posterior and SNR gain compare with Strub et al.'s?</h2>
  <p class="note">The verification binary of Strub et al., analysed alone with this repo's simulator and likelihood but under their priors: one source at f₀ = 1.40457 mHz with an optimal SNR near 7.9, for which they publish the posterior's widths and the SNR gained by fitting.</p>
  <div class="controls"><div><span class="k">View</span><span class="seg" id="strub"></span></div></div>
  <p class="note" id="note-f"></p>
  <div class="legend" id="legend-f"></div>
  <div id="p-strub"></div>
  <div id="p-strub2"></div>
  <div class="tablewrap" id="t-strub"></div>
</div>
<div class="card">
  <h2>What this means</h2>
  <ul class="means" id="means"></ul>
</div>
</main>
<script>const DATA = __DATA__;</script>
<script>
const ST = DATA.strub;
const CHOICES = {strub: ["f0", "cosi", "gain", "pulls"]};
const S = {strub: "f0"};
const marks = {};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const fmt = (v, d = 2) => v === null || v === undefined || !isFinite(v) ? "–" : Number(v).toFixed(d);

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  for (const [k, allowed] of Object.entries(CHOICES)) if (allowed.includes(h.get(k))) S[k] = h.get(k);
}
function writeHash() {
  const p = {strub: S.strub};
  history.replaceState(null, "", "#" + new URLSearchParams(p).toString());
}
readHash();

function seg(id, items, key) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    S[key] = b.dataset.v;
    marks[key]();
    schedule();
  }));
  marks[key]();
}
seg("strub", [["f0", "f₀"], ["cosi", "cos ι"], ["gain", "SNR gain"], ["pulls", "noiseless pulls"]], "strub");
const legendSwatch = (colour, text, dashed) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(${colour})" stroke-width="2"${dashed ? ' stroke-dasharray="4 3"' : ""}/></svg>${text}</span>`;
const legendDot = (colour, text) => `<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(${colour})"/></svg>${text}</span>`;

const M = {l: 46, r: 14};
function ticks(lo, hi, count) {
  const raw = Math.max((hi - lo) / Math.max(count, 1), 1e-12), e = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(m => m * e).find(s => s >= raw);
  const out = [];
  for (let i = Math.ceil(lo / step); i * step <= hi; i++) out.push(i * step);
  return out;
}
const frame = (W, H) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img">`;
const hline = (W, y, colour, dash, width = 1) => `<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" stroke="var(${colour})" stroke-width="${width}"${dash ? ' stroke-dasharray="4 3"' : ""}/>`;
const vline = (x, t, bottom, colour, dash, width = 1) => `<line x1="${x}" x2="${x}" y1="${t}" y2="${bottom}" stroke="var(${colour})" stroke-width="${width}"${dash ? ' stroke-dasharray="4 3"' : ""}/>`;
const poly = (pts, colour, width = 1.25, extra = "") => pts.length ? `<path d="${pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("")}" fill="none" stroke="var(${colour})" stroke-width="${width}" stroke-linejoin="round" ${extra}/>` : "";
function xAxis(W, y, X, values, label, format = v => +v.toPrecision(3)) {
  let s = `<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" stroke="var(--axis)"/>`;
  values.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${y}" y2="${y + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${y + 16}" text-anchor="middle">${format(v)}</text>`);
  return s + `<text x="${(M.l + W - M.r) / 2}" y="${y + 31}" text-anchor="middle">${label}</text>`;
}
function yGrid(W, Y, values, format = v => +v.toPrecision(3)) {
  return values.map(v => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${format(v)}</text>`).join("");
}
const yLabel = (H, t, b, text) => `<text class="lbl" transform="translate(11 ${(t + H - b) / 2}) rotate(-90)" text-anchor="middle">${text}</text>`;

function strubHist(host, h, keys, colours, x0, x1, extra, xlabel) {
  const W = Math.max(280, host.clientWidth || 600), t = 10, b = 40, H = 250, X = v => M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r);
  const peak = 1.1 * Math.max(...keys.flatMap(k => h[k].filter(v => v !== null)));
  const Y = v => t + (peak - v) / peak * (H - t - b);
  let s = frame(W, H) + extra(X, t, H - b) + yGrid(W, Y, ticks(0, peak, 4));
  keys.forEach((k, n) => {
    const pts = [];
    h[k].forEach((v, i) => { if (h.edges[i] >= x0 && h.edges[i + 1] <= x1) pts.push([X(h.edges[i]), Y(v)], [X(h.edges[i + 1]), Y(v)]); });
    s += poly(pts, colours[n], 1.5);
  });
  s += xAxis(W, H - b, X, ticks(x0, x1, 8), xlabel) + yLabel(H, t, b, "density");
  host.innerHTML = s + "</svg>";
}
function strubPanel() {
  const host = document.getElementById("p-strub"), host2 = document.getElementById("p-strub2"), note = document.getElementById("note-f"), legend = document.getElementById("legend-f");
  host2.innerHTML = "";
  const band = (X, t, bottom, a, z, colour) => `<rect x="${X(a)}" width="${X(z) - X(a)}" y="${t}" height="${bottom - t}" fill="var(${colour})" opacity="0.14"/>`;
  if (S.strub === "f0") {
    const e = ST.hist_f0.edges, edge = Math.max(4, ST.strub_f0_plus_nhz * 2.2), q = ST.f0_q.strub;
    strubHist(host, ST.hist_f0, ["strub", "box"], ["--s1", "--ref"], Math.max(e[0], -edge), Math.min(e[e.length - 1], edge), (X, t, bottom) =>
      band(X, t, bottom, -ST.strub_f0_minus_nhz, ST.strub_f0_plus_nhz, "--s2") + vline(X(q[0] - q[1]), t, bottom, "--s1", true) + vline(X(q[2] - q[1]), t, bottom, "--s1", true), "f₀ − posterior median [nHz]");
    note.innerHTML = `Noise realisation 0. Under Strub's priors the 68% interval is +${fmt(ST.seed0_f0_plus_nhz)}/−${fmt(ST.seed0_f0_minus_nhz)} nHz (dashed); Strub et al. report +${ST.strub_f0_plus_nhz}/−${ST.strub_f0_minus_nhz} nHz (orange band, centred on our median). The Fisher box (grey) truncates f₀ at ±1σ, so its f₀ is narrower by construction. Below: σ<sub>f₀</sub> over ${ST.sweep_count} noise realisations, which shows how much a single realisation's width moves; hollow dots put over 1% of their draws in the sideband modes at ±31.7 nHz.`;
    legend.innerHTML = legendSwatch("--s1", "Strub's priors") + legendSwatch("--ref", "Fisher box") + `<span><svg width="22" height="10"><rect width="22" height="10" fill="var(--s2)" opacity="0.3"/></svg>Strub et al. 68%</span>`;
    // sigma_f0 over the noise sweep
    const sig = ST.sweep_sigma_f0_nhz_list, off = ST.sweep_off_mode_fraction_list, W = Math.max(280, host2.clientWidth || 600), t = 14, b = 40, H = 110;
    const lo = 0.5, hi = 20, X = v => M.l + Math.log(clamp(v, lo, hi) / lo) / Math.log(hi / lo) * (W - M.l - M.r);
    let s = frame(W, H) + vline(X((ST.strub_f0_plus_nhz + ST.strub_f0_minus_nhz) / 2), t - 6, H - b, "--s2", true) + `<text x="${X((ST.strub_f0_plus_nhz + ST.strub_f0_minus_nhz) / 2) + 4}" y="${t}">Strub's half-interval</text>`;
    s += vline(X(ST.sweep_sigma_f0_nhz_median), t - 6, H - b, "--ink", true) + `<text class="lbl" x="${X(ST.sweep_sigma_f0_nhz_median) - 4}" y="${t}" text-anchor="end">median ${fmt(ST.sweep_sigma_f0_nhz_median)}</text>`;
    sig.forEach((v, i) => { const y = t + 10 + (i % 6) * 6; s += `<circle cx="${X(v)}" cy="${y}" r="3" ${off[i] > 0.01 ? 'fill="none" stroke="var(--s1)" stroke-width="1.25"' : 'fill="var(--s1)"'}><title>σ = ${fmt(v, 2)} nHz${off[i] > 0.01 ? `, ${fmt(off[i] * 100, 1)}% in the sideband modes` : ""}</title></circle>`; });
    s += xAxis(W, H - b, X, [0.5, 1, 2, 5, 10, 20], "σ of f₀ per noise realisation [nHz]");
    host2.innerHTML = s + "</svg>";
  } else if (S.strub === "cosi") {
    const a = Math.cos(ST.strub_iota_median + ST.strub_iota_plus), z = Math.cos(ST.strub_iota_median - ST.strub_iota_minus), truth = Math.cos(ST.iota);
    strubHist(host, ST.hist_cos_iota, ["strub", "box"], ["--s1", "--ref"], -1, 1, (X, t, bottom) =>
      band(X, t, bottom, a, z, "--s2") + vline(X(Math.cos(ST.strub_iota_median)), t, bottom, "--s2") + vline(X(truth), t, bottom, "--ink", true) + `<text class="lbl" x="${X(truth) + 4}" y="${t + 10}">truth</text>`, "cos ι");
    note.innerHTML = `Under Strub's priors cos ι runs out to face-on: ι = ${fmt(ST.seed0_iota_median)} +${fmt(ST.seed0_iota_plus)}/−${fmt(ST.seed0_iota_minus)}. Inside the Fisher box at the maximum-likelihood point (±3σ, which narrows cos ι), ι = ${fmt(ST.box_iota_median)} +${fmt(ST.box_iota_plus)}/−${fmt(ST.box_iota_minus)}, against Strub's ${ST.strub_iota_median} +${ST.strub_iota_plus}/−${ST.strub_iota_minus} (orange). The box's 68% half-width of cos ι is ${fmt(ST.box_cos_iota_half_width68, 3)}; Strub's interval maps to about ${fmt((z - a) / 2, 3)}. cos ι and log A correlate at ${fmt(ST.seed0_corr_cos_iota_log10_A)}.`;
    legend.innerHTML = legendSwatch("--s1", "Strub's priors") + legendSwatch("--ref", "Fisher box") + `<span><svg width="22" height="10"><rect width="22" height="10" fill="var(--s2)" opacity="0.3"/></svg>Strub et al. ι interval</span>` + legendSwatch("--ink", "truth", true);
  } else if (S.strub === "gain") {
    const obs = ST.sweep_rho_obs_list, gain = ST.sweep_rho_mle_list.map((v, i) => v - obs[i]);
    const W = Math.max(280, host.clientWidth || 600), t = 10, b = 40, H = 270;
    const x0 = Math.floor(Math.min(...obs)) - 0.5, x1 = Math.ceil(Math.max(...obs)) + 0.5, y0 = Math.min(0, ...gain) - 0.05, y1 = Math.max(...gain, ST.strub_gain) + 0.1;
    const X = v => M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r), Y = v => t + (y1 - v) / (y1 - y0) * (H - t - b);
    const m = ST.sweep_gain_obs_mean, e = ST.sweep_gain_obs_sem;
    let s = frame(W, H) + `<rect x="${M.l}" width="${W - M.l - M.r}" y="${Y(m + e)}" height="${Y(m - e) - Y(m + e)}" fill="var(--s1)" opacity="0.15"/>` + yGrid(W, Y, ticks(y0, y1, 5));
    s += hline(W, Y(m), "--s1", false, 1.5) + hline(W, Y(ST.strub_gain), "--s2", true, 1.5);
    s += `<text x="${W - M.r - 4}" y="${Y(m) - 5}" text-anchor="end">mean ${fmt(m, 3)} ± ${fmt(e, 3)}</text><text x="${W - M.r - 4}" y="${Y(ST.strub_gain) + 14}" text-anchor="end">Strub et al. ${fmt(ST.strub_gain)}</text>`;
    obs.forEach((v, i) => s += `<circle cx="${X(v)}" cy="${Y(gain[i])}" r="3.5" fill="var(--s1)" opacity="0.85"><title>ρ_obs ${fmt(v, 2)}, gain ${fmt(gain[i], 3)}</title></circle>`);
    s += xAxis(W, H - b, X, ticks(x0, x1, 8), "observed SNR ρ<tspan dy=\"3\" font-size=\"9\">obs</tspan>") + yLabel(H, t, b, "ρ_MLE − ρ_obs");
    host.innerHTML = s + "</svg>";
    note.innerHTML = `${ST.sweep_count} noise realisations of the same source: the SNR gained by fitting, ρ<sub>MLE</sub> − ρ<sub>obs</sub>, against the observed SNR. The mean is ${fmt(m, 3)} ± ${fmt(e, 3)} with a scatter of ${fmt(ST.sweep_gain_obs_std)} per realisation, so Strub's ${fmt(ST.strub_gain)} from one realisation is ordinary. Refining the maximum gives ${fmt(ST.sweep_gain_obs_refined_mean, 3)} ± ${fmt(ST.sweep_gain_obs_refined_sem, 3)}, and Δρ² = ${fmt(ST.sweep_delta_rho2_refined_mean)} ± ${fmt(ST.sweep_delta_rho2_refined_sem)}.`;
    legend.innerHTML = legendDot("--s1", "one realisation") + legendSwatch("--s1", "mean ± s.e.") + legendSwatch("--s2", "Strub et al.", true);
  } else {
    const keys = [["f0", "f₀"], ["log10_fdot", "log ḟ"], ["log10_A", "log A"], ["ecl_lon", "λ"], ["ecl_lat", "β"], ["psi", "ψ"], ["cos_iota", "cos ι"], ["phi0", "φ₀"]];
    const W = Math.max(280, host.clientWidth || 600), t = 10, b = 40, row = 24, H = t + b + row * keys.length, L = 60;
    const X = v => L + (clamp(v, -2, 2) + 2) / 4 * (W - L - 110);
    let s = frame(W, H);
    [-2, -1, 0, 1, 2].forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${t}" y2="${H - b}" stroke="var(${v ? "--grid" : "--axis"})"/><text x="${X(v)}" y="${H - b + 16}" text-anchor="middle">${v}</text>`);
    keys.forEach(([k, name], i) => {
      const v = ST[`noiseless_pull_${k}`], qn = ST[`noiseless_truth_quantile_${k}`], y = t + i * row + 4;
      s += `<text class="lbl" x="${L - 8}" y="${y + 12}" text-anchor="end">${name}</text><rect x="${Math.min(X(0), X(v))}" width="${Math.abs(X(v) - X(0))}" y="${y}" height="${row - 8}" fill="var(${Math.abs(v) > 1 ? "--s2" : "--s1"})"/><text x="${W - 100}" y="${y + 12}">truth at q ${fmt(qn, 3)}</text>`;
    });
    s += `<text x="${(L + W - 110) / 2}" y="${H - b + 31}" text-anchor="middle">(median − truth) / posterior std</text>`;
    host.innerHTML = s + "</svg>";
    note.innerHTML = `Without noise, Strub's priors: how far each marginal's median sits from the truth, in posterior standard deviations. ${keys.filter(([k]) => Math.abs(ST[`noiseless_pull_${k}`]) < 0.25).length} of ${keys.length} are within 0.25. log A and cos ι move together along their degeneracy, towards face-on and lower A. The likelihood still peaks at the truth: the refined maximum-likelihood SNR is ${fmt(ST.noiseless_rho_mle_refined, 5)} against ρ<sub>opt</sub> = ${fmt(ST.noiseless_rho_opt, 5)}, so the offset is prior volume, not a biased peak.`;
    legend.innerHTML = "";
  }
  const rows = [
    ["Optimal SNR", fmt(ST.snr, 4), fmt(ST.snr_strub, 2), `${fmt(ST.snr_relative_gap * 100, 2)}% gap; ${fmt(ST.snr_unrotated_rp64, 4)} if the sky frame is not rotated`],
    ["f₀ 68% interval [nHz]", `+${fmt(ST.seed0_f0_plus_nhz)} / −${fmt(ST.seed0_f0_minus_nhz)}`, `+${ST.strub_f0_plus_nhz} / −${ST.strub_f0_minus_nhz}`, `one realisation; σ over ${ST.sweep_count}: median ${fmt(ST.sweep_sigma_f0_nhz_median)} (${fmt(ST.sweep_sigma_f0_nhz_q16)}–${fmt(ST.sweep_sigma_f0_nhz_q84)})`],
    ["ι, Fisher box", `${fmt(ST.box_iota_median)} +${fmt(ST.box_iota_plus)} / −${fmt(ST.box_iota_minus)}`, `${ST.strub_iota_median} +${ST.strub_iota_plus} / −${ST.strub_iota_minus}`, `full prior: ${fmt(ST.seed0_iota_median)} +${fmt(ST.seed0_iota_plus)} / −${fmt(ST.seed0_iota_minus)}`],
    ["SNR gain ρ_MLE − ρ_obs", `${fmt(ST.sweep_gain_obs_mean, 3)} ± ${fmt(ST.sweep_gain_obs_sem, 3)}`, fmt(ST.strub_gain), `scatter ${fmt(ST.sweep_gain_obs_std)} per realisation`],
    ["R̂, all chains", `< ${fmt(Math.max(ST.sweep_rhat_max, ST.noiseless_rhat_max, ST.seed0_rhat_max, ST.box_rhat_max), 4)}`, "–", `${ST.sweep_count} sweep chains and the noiseless, realisation-0 and box chains`],
  ];
  document.getElementById("t-strub").innerHTML = `<table><thead><tr><th>quantity</th><th>this pipeline</th><th>Strub et al.</th><th>note</th></tr></thead><tbody>${rows.map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function tiles() {
  const tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  document.getElementById("tiles").innerHTML = [
    tile("Optimal SNR", fmt(ST.snr, 3), `Strub et al. ${ST.snr_strub}`),
    tile("f₀ 68% interval", `+${fmt(ST.seed0_f0_plus_nhz)}/−${fmt(ST.seed0_f0_minus_nhz)}`, `nHz; Strub et al. +${ST.strub_f0_plus_nhz}/−${ST.strub_f0_minus_nhz}`),
    tile("SNR gain", `${fmt(ST.sweep_gain_obs_mean, 3)} ± ${fmt(ST.sweep_gain_obs_sem, 3)}`, `over ${ST.sweep_count} realisations; Strub et al. ${fmt(ST.strub_gain)}`),
    tile("ι, Fisher box", fmt(ST.box_iota_median), `full prior ${fmt(ST.seed0_iota_median)}; Strub et al. ${ST.strub_iota_median}`),
  ].join("");
}

function texts() {
  document.getElementById("sub").innerHTML = `Strub et al. 2022 (arXiv:2204.04467, Table VII) publish the posterior of one faint verification binary. Here the same source runs through this repo's simulator and likelihood, under their priors, with their published widths and SNR gain as the target. Built ${DATA.built} from <code>data/</code>.`;
  const items = [
    `<b>The pipeline reproduces Strub et al.'s verification binary.</b> Its optimal SNR is ${fmt(ST.snr, 4)} against their ${ST.snr_strub}; the f₀ interval is +${fmt(ST.seed0_f0_plus_nhz)}/−${fmt(ST.seed0_f0_minus_nhz)} nHz against +${ST.strub_f0_plus_nhz}/−${ST.strub_f0_minus_nhz}, within the realisation-to-realisation spread; the SNR gain ${fmt(ST.sweep_gain_obs_mean, 2)} ± ${fmt(ST.sweep_gain_obs_sem, 2)} against their single ${fmt(ST.strub_gain)}. Their ι (${ST.strub_iota_median}) is approached only inside a Fisher box (${fmt(ST.box_iota_median)}); under their full prior ι runs towards face-on (${fmt(ST.seed0_iota_median)}).`,
    `<b>So the simulator, noise and likelihood behind the MCMC reference</b> (<a href="../../mcmc/mcmc.html">mcmc</a>) reproduce a published analysis' SNR and widths; the ι difference comes from the prior volume, since without noise the likelihood still peaks at the truth.`,
  ];
  document.getElementById("means").innerHTML = items.map(v => `<li>${v}</li>`).join("");
}

function render() {
  tiles(); strubPanel(); texts();
  writeHash();
}

let pending = false;
function schedule() {  // one render per frame however fast the pickers change
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}
addEventListener("resize", schedule);
addEventListener("hashchange", () => { readHash(); Object.values(marks).forEach(m => m()); schedule(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
render();
</script>
</body>
</html>
"""


def main() -> int:
    strub = strub_record(
        json.loads((DATA / "strub_source.json").read_text()),
        np.load(DATA / "seed0.npz"),
    )
    page = dict(built=datetime.date.today().isoformat(), strub=strub)
    html = PAGE.replace("__DATA__", json.dumps(page, separators=(",", ":")))
    (REPORT / "strub-source.html").write_text(html)
    keys = (
        "snr",
        "seed0_f0_plus_nhz",
        "seed0_f0_minus_nhz",
        "sweep_gain_obs_mean",
        "sweep_gain_obs_sem",
        "box_iota_median",
        "seed0_iota_median",
        "noiseless_pull_log10_A",
        "noiseless_pull_cos_iota",
    )
    pulls = [v for k, v in strub.items() if k.startswith("noiseless_pull_")]
    print(
        json.dumps(
            {k: strub[k] for k in keys}
            | {
                "pulls_below_0.25": f"{sum(abs(v) < 0.25 for v in pulls)} of {len(pulls)}"
            },
            indent=1,
        )
    )
    print(f"strub-source.html: {len(html) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
