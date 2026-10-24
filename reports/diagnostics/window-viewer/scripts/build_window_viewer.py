"""Write window-viewer.html: a LisaGB prior window, its sources, noise and data as LisaFlow reads them.

Run from the repo root (on a compute node: it builds LisaGB and checks its port under node, see window-viewer.sbatch):
    uv run --frozen python reports/diagnostics/window-viewer/scripts/build_window_viewer.py [--no-check] [--out page.html]
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import date
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa.config import load
from canna.lisa.network import stft_tokens
from canna.lisa.physics import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT, SUN_MASS
from canna.lisa.problem import WINDOW_STRIDE, LisaGB, sample_f, window_start
from canna.lisa.simulator import (
    clean_signal,
    preprocess,
    sample_physical,
    snr,
    source_snrs,
)

FOLDER = Path(__file__).parents[1]
BANDS = ("narrow", "full")
PATCHES = (4, 16)
WINDOWS = 8192
BANK = 128
CHECKED_WINDOWS = 4
# worst disagreement allowed between the port and canna, over every checked window
TOLERANCES = dict(window=1e-8, window_snr=1e-8, source_snrs=1e-8, tokens=1e-6)
JS_CORE = r"""
// numpy's round half to even
const rint = x => { const r = Math.round(x); return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r; };

function fft(re, im) {  // in place, radix 2, forward (exp(-i...)) like numpy
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

// jaxgb's JaxGB.get_tdi(p, 1.5, "AET") for one source in jaxgb's columns
// [f0, fdot, A, ra, dec, psi, iota (inclination), phi0]: kmin and the fftshifted A/E/T bins
function gbResponse(p, D) {
  const [f0, fdot, amp, ra, dec, psi, iota, phi0] = p;
  const n = D.n, T = D.t_obs, c = D.c, fstar = D.fstar;
  const sra = Math.sin(ra), cra = Math.cos(ra), sde = Math.sin(dec), cde = Math.cos(dec);
  const ci = Math.cos(iota), c2p = Math.cos(2 * psi), s2p = Math.sin(2 * psi);
  const aplus = -amp * (1 + ci * ci), across = -2 * amp * ci;
  // complex h+ / hx amplitudes in the SSB frame
  const ap = [aplus * c2p, across * s2p], ac = [aplus * s2p, -across * c2p];
  const u = [sra, -cra, 0], v = [-sde * cra, -sde * sra, cde], k = [-cde * cra, -cde * sra, -sde];
  const ep = [[], [], []], ec = [[], [], []];
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    ep[a][b] = u[a] * u[b] - v[a] * v[b];
    ec[a][b] = u[a] * v[b] + v[a] * u[b];
  }
  const q = Math.round(f0 * T), om = 2 * Math.PI * f0, df = 2 * Math.PI * q / T;
  const SC = [0, 1, 2, 0, 2, 1], AIJ = [0, 1, 2, 2, 1, 0];  // links 12, 23, 31, 13, 32, 21
  const g = Array.from({length: 6}, () => [new Float64Array(n), new Float64Array(n)]);
  const kp = new Float64Array(3), xi = new Float64Array(3), ph = new Float64Array(3);
  const kr = new Float64Array(6), aij = [[0, 0], [0, 0], [0, 0]];
  for (let t = 0; t < n; t++) {
    const tm = t * T / n;
    for (let s = 0; s < 3; s++) {
      const P = D.pos[s];
      kp[s] = (k[0] * P[0][t] + k[1] * P[1][t] + k[2] * P[2][t]) / c;
      xi[s] = tm - kp[s];
      ph[s] = om * kp[s] - (-phi0 + (om - df) * tm + Math.PI * fdot * xi[s] * xi[s]);
    }
    for (let l = 0; l < 3; l++) {
      // unit link vector r_l = (x_l - x_{l+1}) / L and its two polarisation projections
      const P1 = D.pos[l], P2 = D.pos[(l + 1) % 3], r = [0, 0, 0];
      for (let a = 0; a < 3; a++) r[a] = (P1[a][t] - P2[a][t]) / D.L;
      let rpr = 0, rcr = 0;
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        rpr += r[a] * ep[a][b] * r[b]; rcr += r[a] * ec[a][b] * r[b];
      }
      aij[l] = [rpr * ap[0] + rcr * ac[0], rpr * ap[1] + rcr * ac[1]];
      kr[l] = k[0] * r[0] + k[1] * r[1] + k[2] * r[2];
    }
    kr[3] = -kr[2]; kr[4] = -kr[1]; kr[5] = -kr[0];
    for (let l = 0; l < 6; l++) {
      const fonfs = (f0 + fdot * xi[SC[l]]) / fstar;
      const alpha = 0.5 * fonfs * (1 - kr[l]);
      const sinc = alpha === 0 ? 1 : Math.sin(alpha) / alpha;
      const arg = -(alpha + ph[SC[l]]);
      const er = Math.cos(arg), ei = Math.sin(arg);
      const [ar, ai] = aij[AIJ[l]];
      // -0.5i * a * fonfs * sinc * exp(i arg)
      const m = 0.5 * fonfs * sinc, zr = ar * er - ai * ei, zi = ar * ei + ai * er;
      g[l][0][t] = m * zi; g[l][1][t] = -m * zr;
    }
  }
  // TDI 1.5 X, Y, Z from the slow links, then to A, E, T
  const X = [new Float64Array(n), new Float64Array(n)];
  const Y = [new Float64Array(n), new Float64Array(n)];
  const Z = [new Float64Array(n), new Float64Array(n)];
  const comb = (out, t, a, b, c1, d, fr, fi, f2r, f2i) => {
    // out = f2 * (g[a] - g[b] + (g[c1] - g[d]) * f)
    const ur = g[c1][0][t] - g[d][0][t], ui = g[c1][1][t] - g[d][1][t];
    const sr = g[a][0][t] - g[b][0][t] + ur * fr - ui * fi;
    const si = g[a][1][t] - g[b][1][t] + ur * fi + ui * fr;
    out[0][t] = f2r * sr - f2i * si; out[1][t] = f2r * si + f2i * sr;
  };
  for (let t = 0; t < n; t++) {
    const oml = (f0 + fdot * t * T / n) / fstar;
    const fr = Math.cos(oml), fi = -Math.sin(oml);  // exp(-i oml)
    const s = Math.sin(oml), f2r = -2 * s * s, f2i = -2 * s * fr;  // -2i sin(oml) exp(-i oml)
    comb(X, t, 5, 2, 0, 3, fr, fi, f2r, f2i);
    comb(Y, t, 4, 0, 1, 5, fr, fi, f2r, f2i);
    comb(Z, t, 3, 1, 2, 4, fr, fi, f2r, f2i);
  }
  const r2 = Math.SQRT2, r3 = Math.sqrt(3), r6 = Math.sqrt(6);
  const slow = {A: [[], []], E: [[], []], T: [[], []]};
  for (let j = 0; j < 2; j++) for (let t = 0; t < n; t++) {
    slow.A[j][t] = (Z[j][t] - X[j][t]) / r2;
    slow.E[j][t] = (X[j][t] - 2 * Y[j][t] + Z[j][t]) / r6;
    slow.T[j][t] = (X[j][t] + Y[j][t] + Z[j][t]) / r3;
  }
  const spec = {};
  for (const ch of ["A", "E", "T"]) {
    const re = Float64Array.from(slow[ch][0]), im = Float64Array.from(slow[ch][1]);
    fft(re, im);
    const sr = new Array(n), si = new Array(n), h = n / 2, norm = 0.5 * T / n;
    for (let i = 0; i < n; i++) {  // fftshift and the half-sided normalisation
      sr[i] = re[(i + h) % n] * norm; si[i] = im[(i + h) % n] * norm;
    }
    spec[ch] = [sr, si];
  }
  return {kmin: Math.round(q - n / 2), spec: spec};
}

// canna.lisa.problem: single_link_noise, noise_psd and the window's placement
function singleLink(f, D) {
  f = Math.abs(f);
  return [D.oms ** 2 * 1e-24 * (1 + (0.002 / f) ** 4),
          D.acc ** 2 * 1e-30 * (1 + (0.0004 / f) ** 2) * (1 + (f / 0.008) ** 4)];
}

function noisePsd(f, D) {
  f = Math.abs(f);
  const phase = 2 * Math.PI * D.L / D.c * f, tr = Math.cos(phase);
  const dop = (2 * Math.PI * f / D.c) ** 2;
  const [oms, acc] = singleLink(f, D);
  const om = oms * dop, tm = acc * dop / (2 * Math.PI * f) ** 4;
  const ae = 2 * tm * (1 + tr + tr * tr) + 0.5 * om * (2 + tr);
  const t = 2 * tm * (1 - tr) ** 2 + om * (1 - tr);
  const g = 16 * Math.sin(phase) ** 2;
  return [ae * g, ae * g, t * g];
}

function windowStart(f, D) {
  return rint((f * D.t_obs - D.window_bins / 2) / D.stride) * D.stride;
}

function fdotFromChirpMass(mc, f0, D) {
  return 96 / 5 * Math.PI ** (8 / 3) * f0 ** (11 / 3) * (D.G * D.sun / D.c ** 3 * mc) ** (5 / 3);
}

// canna's [f0, Mc, A, lon, lat, psi, iota, phi0] to jaxgb's columns: the chirp, and iota as colatitude
function jaxgbColumns(p, D) {
  return [p[0], fdotFromChirpMass(p[1], p[0], D), p[2], p[3], p[4], p[5], Math.PI / 2 - p[6], p[7]];
}

// preprocess(clean_signal(p, f), f) for one source: re A, E, T then im A, E, T over the window
function whitenedWindow(p, f, D) {
  const r = gbResponse(jaxgbColumns(p, D), D), start = windowStart(f, D), wb = D.window_bins;
  const y = Array.from({length: 6}, () => new Float64Array(wb));
  for (let i = 0; i < D.n; i++) {
    const k = r.kmin - start + i;
    if (k < 0 || k >= wb) continue;
    const psd = noisePsd((start + k) / D.t_obs, D);
    ["A", "E", "T"].forEach((ch, c) => {
      const scale = 1 / Math.sqrt(psd[c] * D.t_obs / 2 / 2);
      y[c][k] += r.spec[ch][0][i] * scale; y[c + 3][k] += r.spec[ch][1][i] * scale;
    });
  }
  return {y: y, start: start, kmin: r.kmin};
}

function windowSnr(y) {
  let sum = 0;
  for (const a of y) for (const v of a) sum += v * v;
  return Math.sqrt(sum);
}

// canna.lisa.network.stft_tokens: token n is [asinh|z| (p c), Re gated (p c), Im gated (p c)]
function stftTokens(y, P) {
  const N = y[0].length / P, out = [], re = new Float64Array(P), im = new Float64Array(P), root = Math.sqrt(P);
  for (let n = 0; n < N; n++) {
    const tok = new Float64Array(9 * P);
    for (let c = 0; c < 3; c++) {
      // orthonormal inverse DFT as conj(fft(conj z)) / sqrt(P)
      for (let k = 0; k < P; k++) { re[k] = y[c][n * P + k]; im[k] = -y[c + 3][n * P + k]; }
      fft(re, im);
      for (let t = 0; t < P; t++) {
        const zr = re[t] / root, zi = -im[t] / root, m2 = zr * zr + zi * zi, gate = 1 / Math.sqrt(1 + m2);
        tok[t * 3 + c] = Math.asinh(Math.sqrt(m2));
        tok[3 * P + t * 3 + c] = zr * gate;
        tok[6 * P + t * 3 + c] = zi * gate;
      }
    }
    out.push(tok);
  }
  return out;
}

"""

CHECK_JS = r"""
const D = __DATA__, PATCHES = __PATCHES__, WINDOWS = __WINDOWS__;
const keep = a => Array.from(a);
// the page's whole window: every source's whitened window, summed, and its tokens
const windows = WINDOWS.map(([f, ps]) => {
  const ws = ps.map(p => whitenedWindow(p, f, D)), ys = ws.map(w => w.y);
  const y = ys[0].map((a, c) => a.map((_, k) => ys.reduce((t, s) => t + s[c][k], 0)));
  return {
    start: ws[0].start, y: y.map(keep), snr: windowSnr(y), source_snrs: ys.map(windowSnr),
    tokens: Object.fromEntries(PATCHES.map(P => [P, stftTokens(y, P).map(keep)])),
  };
});
console.log(JSON.stringify({windows: windows}));
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Window viewer</title>
<script>
window.onerror = (msg, src, line) => {
  const b = document.getElementById("err");
  if (b) { b.hidden = false; b.textContent = `This page hit an error and may show stale views: ${msg} (line ${line}).`; }
};
</script>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --ref: #52514e;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --ref: #c3c2b7;
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --ref: #c3c2b7;
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 24px; max-width: 72ch; }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0 12px; }
.layout { display: grid; grid-template-columns: 290px minmax(0, 1fr); gap: 20px; align-items: start; }
.side { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto; }
.content { min-width: 0; }
.content > .card + .card { margin-top: 16px; }
@media (max-width: 900px) {
  .layout { grid-template-columns: minmax(0, 1fr); }
  .side { position: static; max-height: none; }
}
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; position: relative; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.tile .d { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 22px; flex: none; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; margin: 0 0 16px; }
.controls .k, .row .k { color: var(--ink-2); font-size: 13px; margin-right: 8px; }
.row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; padding: 5px 12px; border: 0; background: transparent; color: var(--ink-2); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.card > .seg { margin: 0 0 12px; }
button.plain { font: inherit; font-size: 13px; padding: 5px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--ink-2); cursor: pointer; }
.sl { display: block; margin: 14px 0; }
.sl .top { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; }
.sl .name { color: var(--ink-2); }
.sl .val { font-variant-numeric: tabular-nums; }
.sl .sn { display: block; color: var(--muted); font-size: 12px; }
input[type=range] { width: 100%; accent-color: var(--s1); margin: 4px 0 0; }
.flowrow { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
#p-flow { flex: 1 1 320px; min-width: 0; }
#p-dials { flex: 0 1 320px; min-width: 0; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); }
svg text.lab { fill: var(--ink-2); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
svg image { image-rendering: pixelated; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; white-space: nowrap; z-index: 2; }
.tip .row { color: var(--ink-2); font-variant-numeric: tabular-nums; margin: 0; justify-content: flex-start; gap: 6px; }
.winrow { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; }
.winrow .num { flex: 1; text-align: center; font-variant-numeric: tabular-nums; font-size: 13px; }
.tablewrap { overflow-x: auto; }
table.t { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
table.t th, table.t td { text-align: right; padding: 4px 12px; border-bottom: 1px solid var(--grid); }
table.t th:first-child, table.t td:first-child { text-align: left; color: var(--ink-2); }
table.t th { color: var(--ink-2); font-weight: 600; }
.meaning ul { margin: 8px 0 0; padding-left: 20px; }
.meaning li { margin: 0 0 8px; max-width: 80ch; }
#err { position: sticky; top: 0; z-index: 10; background: #c0392b; color: #fff; padding: 10px 16px; font-size: 14px; }
</style>
</head>
<body>
<div id="err" role="alert" hidden></div>
<main>
<h1>Window viewer</h1>
<p class="sub">A prior window of <code>LisaGB</code> as <code>LisaFlow</code> reads it: four galactic binaries sorted loudest first, each alone and summed, a seeded noise draw and the data, all whitened, with each source's SNR and the STFT tokens of the whole window. The page recomputes everything in the browser; the address bar keeps the state. One source and how each parameter moves it: the <a href="../source-viewer/source-viewer.html">source viewer</a>. __SUB__</p>
<div class="layout">
<aside class="card side">
  <div class="row"><span class="k">Band</span><span class="seg" id="band"></span></div>
  <div class="k" style="color:var(--ink-2);font-size:13px">Prior window</div>
  <div class="winrow"><button type="button" class="plain" id="prev" aria-label="previous window">‹</button><span class="num" id="winlabel"></span><button type="button" class="plain" id="next" aria-label="next window">›</button><button type="button" class="plain" id="rand">random</button></div>
  <div class="row"><span class="k">Source</span><span class="seg" id="src"></span></div>
  <p class="note" id="n-src"></p>
  <p class="note"><a id="open-source" href="../source-viewer/source-viewer.html">Open this source in the source viewer</a></p>
</aside>
<div class="content">
  <div class="controls">
    <div><span class="k">Channel</span><span class="seg" id="ch"></span></div>
    <div><span class="k">Patch</span><span class="seg" id="patch"></span></div>
    <div><span class="k">View</span><span class="seg" id="view"></span></div>
    <div><span class="k">Noise</span><span class="seg" id="noise"></span> <button type="button" class="plain" id="redraw">New draw</button></div>
  </div>
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <h2>Which sources share this window, and where do they overlap?</h2>
    <p class="note">Top: each source's own response bins (thin) and the bins holding 99% of its ρ² (thick), with its f₀ (tick); shaded where two or more sources share bins. Click a row to pick that source. Bottom: the whitened amplitude over A, E and T, √Σ|ỹ|², of each source alone and of the data, on an asinh axis; the dashed line is the noise's √6. In the source view the bracket is the span the cards below show.</p>
    <div class="legend" id="lg-overview"></div>
    <div id="p-overview"></div><div class="tip"></div>
  </div>
  <div class="card">
    <h2>How do the sources and the noise add up to the data?</h2>
    <p class="note">One channel of the whitened window (<code>preprocess</code>), each source alone, their sum, the noise draw and the data = sum + noise. Every strip shares one asinh axis; the grey band is ±1, the noise's standard deviation per part. The gray tick is each source's f₀.</p>
    <div class="legend">__PARTS__</div>
    <div id="p-layers"></div>
  </div>
  <div class="card">
    <h2>How loud is each source, alone and in the data?</h2>
    <p class="note">Bars: each source's own SNR ρ = ‖ỹ‖ (<code>source_snrs</code>), and the window's, ‖Σỹ‖ (<code>snr</code>), whose tick is √Σρ², what it would be if the sources shared no bins. Diamonds: the matched-filter SNR of the source's own template on the data, ⟨d, ỹ⟩/‖ỹ‖. With the noise off it moves off ρ only through the other sources' overlap; with the noise on it also scatters by about ±1.</p>
    <div id="p-snr"></div>
    <div id="t-snr"></div>
  </div>
  <div class="card">
    <h2>What do the tokens of the whole window hold?</h2>
    <p class="note">The data tokens (<code>stft_tokens</code>) of the data. Each column is one token: its <i>patch</i> bins taken back to the time domain by an orthonormal inverse DFT, so the rows run through the observation, from 0 at the bottom to T at the top. The token holds asinh|z| and the gated phase z/√(1+|z|²) of all three channels; this view shows the selected channel. Noise has E|z|² = 2. The ticks are the sources' f₀, in their colours.</p>
    <div id="p-tokens"></div>
  </div>
  <div class="card">
    <h2>Does the page's port agree with canna?</h2>
    <p class="note" id="n-port"></p>
    <div id="p-port"></div>
  </div>
  <div class="card meaning">
    <h2>What this means</h2>
    <ul id="meaning"></ul>
  </div>
</div>
</div>
</main>
<script>
__CORE__
const DATA = __DATA__;
const BANK = __BANK__;
const PG = __PAGE__;
const TAU = 2 * Math.PI;
const PATCH_CHOICES = [2, 4, 8, 16, 32, 64];
const SLOT = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"];
const S = {band: "full", win: 0, src: 0, view: 0, noise: 1, seed: 1, ch: 0, patch: 16};
const D = () => DATA[S.band];
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sig = (v, n = 3) => Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(n - 1) : (+v.toPrecision(n)).toString()) : "–";
const sum = a => a.reduce((x, y) => x + y, 0);
const dot = (a, b) => sum(a.map((row, c) => row.reduce((t, v, k) => t + v * b[c][k], 0)));

const marks = {};
function seg(id, items, key, after) {
  const host = document.getElementById(id);
  host.innerHTML = items.map(([v, text]) => `<button type="button" data-v="${v}">${text}</button>`).join("");
  marks[key] = () => host.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(S[key]))));
  host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    S[key] = typeof S[key] === "number" ? Number(b.dataset.v) : b.dataset.v;
    if (after) after();
    Object.values(marks).forEach(m => m());
    schedule();
  }));
  marks[key]();
}
const clampWindow = () => S.win = clamp(S.win, 0, BANK[S.band].length - 1);
seg("band", [["narrow", "narrow"], ["full", "full"]], "band", clampWindow);
seg("src", [0, 1, 2, 3].map(s => [s, `<i class="dot" style="background:${SLOT[s]}"></i> ${s + 1}`]), "src");
seg("ch", [[0, "A"], [1, "E"], [2, "T"]], "ch");
seg("patch", PATCH_CHOICES.map(P => [P, P]), "patch");
seg("view", [[0, "window"], [1, "source"]], "view");
seg("noise", [[0, "off"], [1, "on"]], "noise");
const stepWindow = k => { S.win = ((k % BANK[S.band].length) + BANK[S.band].length) % BANK[S.band].length; schedule(); };
document.getElementById("prev").addEventListener("click", () => stepWindow(S.win - 1));
document.getElementById("next").addEventListener("click", () => stepWindow(S.win + 1));
document.getElementById("rand").addEventListener("click", () => stepWindow(Math.floor(Math.random() * BANK[S.band].length)));
document.getElementById("redraw").addEventListener("click", () => { S.seed += 1; S.noise = 1; marks.noise(); schedule(); });

// state in the address bar: #band=full&win=3&src=2&noise=1&seed=4 ..., windows and sources counted from 1
const HASH = ["band", "win", "src", "view", "noise", "seed", "ch", "patch"];
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1)), ok = (k, list) => list.map(String).includes(q.get(k));
  if (ok("band", ["narrow", "full"])) S.band = q.get("band");
  if (q.has("win")) S.win = clamp(Math.round(Number(q.get("win")) - 1) || 0, 0, BANK[S.band].length - 1);
  if (ok("src", [1, 2, 3, 4])) S.src = Number(q.get("src")) - 1;
  if (q.has("seed")) S.seed = Math.max(1, Math.round(Number(q.get("seed"))) || 1);
  for (const [k, list] of [["view", [0, 1]], ["noise", [0, 1]], ["ch", [0, 1, 2]], ["patch", PATCH_CHOICES]]) if (ok(k, list)) S[k] = Number(q.get(k));
  Object.values(marks).forEach(m => m());
}
function writeHash() {
  const q = new URLSearchParams();
  HASH.forEach(k => q.set(k, k === "win" || k === "src" ? S[k] + 1 : S[k]));
  history.replaceState(null, "", "#" + q.toString());
  document.getElementById("open-source").href = `../source-viewer/source-viewer.html#band=${S.band}&start=bank&win=${S.win + 1}&src=${S.src + 1}&ch=${S.ch}&patch=${S.patch}`;
}
addEventListener("hashchange", () => { readHash(); schedule(); });

// mulberry32 and Box-Muller: the whitened noise is unit normal per part, as sample_observation's after preprocess
function normals(count, seed) {
  let a = seed >>> 0;
  const next = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const out = new Float64Array(count);
  for (let i = 0; i < count; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(1 - next())), angle = TAU * next();
    out[i] = radius * Math.cos(angle);
    if (i + 1 < count) out[i + 1] = radius * Math.sin(angle);
  }
  return out;
}
let noiseCache = {};
function noiseFor(d) {
  const key = `${S.band}-${S.seed}`, wb = d.window_bins;
  if (!noiseCache[key]) { const z = normals(6 * wb, S.seed); noiseCache = {[key]: [0, 1, 2, 3, 4, 5].map(c => z.subarray(c * wb, (c + 1) * wb))}; }
  return noiseCache[key];
}

// per-bin power over A, E, T and the smallest set of bins that holds 99% of a source's rho^2
const power = y => y[0].map((_, k) => y.reduce((t, a) => t + a[k] * a[k], 0));
function footprint(pw) {
  const order = pw.map((v, k) => k).sort((a, b) => pw[b] - pw[a]), total = sum(pw), keep = new Uint8Array(pw.length);
  let acc = 0;
  for (const k of order) { if (acc >= 0.99 * total) break; keep[k] = 1; acc += pw[k]; }
  return keep;
}
const runs = mask => { const out = []; let a = -1; mask.forEach((v, k) => { if (v && a < 0) a = k; if (!v && a >= 0) { out.push([a, k - 1]); a = -1; } }); if (a >= 0) out.push([a, mask.length - 1]); return out; };

// the window's sources do not change with the toggles: their whitened windows and footprints are kept
let bankCache = {};
function bankSources() {
  const key = `${S.band}-${S.win}`, w = BANK[S.band][S.win], d = D(), wb = d.window_bins;
  if (bankCache[key]) return bankCache[key];
  const sources = w.p.map((p, s) => {
    const o = whitenedWindow(p, w.f, d), pw = power(o.y);
    return {s, p, y: o.y, start: o.start, seg0: o.kmin - o.start, bin: p[0] * d.t_obs - o.start, rho: windowSnr(o.y), pw, keep: footprint(pw)};
  });
  const count = new Uint8Array(wb);
  sources.forEach(q => q.keep.forEach((v, k) => count[k] += v));
  sources.forEach(q => q.shared = sum(q.pw.map((v, k) => q.keep[k] && count[k] >= 2 ? v : 0)) / q.rho ** 2);
  const clean = sources[0].y.map((a, c) => a.map((_, k) => sum(sources.map(q => q.y[c][k]))));
  bankCache = {[key]: {sources, count, clean, start: sources[0].start}};
  return bankCache[key];
}

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const rgb = name => { const h = css(name).replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); };
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * clamp(t, 0, 1));
function hue(angle) {
  const h = (((angle / TAU) % 1) + 1) % 1 * 6, s = 0.7, l = 0.52;
  const C = (1 - Math.abs(2 * l - 1)) * s, X = C * (1 - Math.abs((h % 2) - 1)), m = l - C / 2;
  const [r, g, b] = h < 1 ? [C, X, 0] : h < 2 ? [X, C, 0] : h < 3 ? [0, C, X] : h < 4 ? [0, X, C] : h < 5 ? [X, 0, C] : [C, 0, X];
  return [r + m, g + m, b + m].map(v => 255 * v);
}
function raster(w, h, colour) {
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const g = cv.getContext("2d"), img = g.createImageData(w, h);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const o = 4 * (j * w + i), c = colour(i, j);
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return cv.toDataURL();
}

const M = {l: 46, r: 14};
function xScale(host, b0, b1) {
  const W = Math.max(280, host.clientWidth);
  return {W, X: b => M.l + (b - b0 + 0.5) / (b1 - b0) * (W - M.l - M.r)};
}
function ticks(lo, hi, count) {
  const raw = Math.max((hi - lo) / Math.max(count, 1), 1e-12), e = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(m => m * e).find(s => s >= raw);
  const out = [];
  for (let i = Math.ceil(lo / step); i * step <= hi; i++) out.push(i * step);
  return out;
}
function binAxis(X, W, y, b0, b1) {
  let s = `<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" stroke="var(--axis)"/>`;
  ticks(b0, b1 - 1, Math.floor((W - M.l - M.r) / 80)).filter(v => Number.isInteger(v)).forEach(v =>
    s += `<line x1="${X(v)}" x2="${X(v)}" y1="${y}" y2="${y + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${y + 16}" text-anchor="middle">${v}</text>`);
  return s + `<text x="${(M.l + W - M.r) / 2}" y="${y + 31}" text-anchor="middle">bin from the window start</text>`;
}
const marker = (X, bin, y0, y1, colour = "var(--ref)") => `<line x1="${X(bin)}" x2="${X(bin)}" y1="${y0}" y2="${y1}" stroke="${colour}" stroke-width="1.5"/>`;

function tip(host, ev, html) {
  const t = host.nextElementSibling, card = host.closest(".card").getBoundingClientRect();
  t.innerHTML = html; t.style.display = "block";
  t.style.left = Math.max(4, Math.min(ev.clientX - card.left + 14, card.width - t.offsetWidth - 8)) + "px";
  t.style.top = (ev.clientY - card.top + 14) + "px";
}
const untip = host => host.nextElementSibling.style.display = "none";
const trow = (color, text) => `<div class="row">${color ? `<i class="dot" style="background:${color};width:8px;height:8px"></i>` : ""}${text}</div>`;
const legendItem = (color, text, dash) => `<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${color}" stroke-width="2"${dash ? ` stroke-dasharray="${dash}"` : ""}/></svg>${text}</span>`;
const legendDot = (color, text) => `<span><i class="dot" style="background:${color}"></i>${text}</span>`;
const legendBox = (color, text) => `<span><i class="dot" style="background:${color};border-radius:2px"></i>${text}</span>`;

function tiles(c) {
  const tile = (k, v, d) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  const wb = c.d.window_bins, P = c.P, rhos = c.sources.map(q => q.rho), shared = c.count.filter(v => v >= 2).length;
  document.getElementById("tiles").innerHTML = [
    tile("Window", `${c.start}–${c.start + wb - 1}`, `${wb} rfft bins, f = ${(c.f * 1e3).toPrecision(5)} mHz, prior draw ${S.win + 1}`),
    tile("Source SNRs", rhos.map(v => v.toFixed(0)).join(" · "), "each source alone, loudest first"),
    tile("Window SNR", c.windowSnr.toFixed(1), `all four together; √Σρ² = ${c.orthogonal.toFixed(1)}`),
    tile("Shared bins", `${shared}`, `bins in two or more sources' 99% of ρ², of ${wb}`),
    tile("Noise", S.noise ? `draw ${S.seed}` : "off", "unit normal per part after whitening"),
    tile("Tokens", `${wb / P} × ${9 * P}`, `asinh|z|, Re and Im gated, of 3 channels × ${P} samples`),
  ].join("");
  document.getElementById("winlabel").textContent = `draw ${S.win + 1} of ${BANK[S.band].length}`;
  const q = c.sources[S.src];
  document.getElementById("n-src").innerHTML = `Source ${S.src + 1}: f₀ = ${(q.p[0] * 1e3).toPrecision(6)} mHz (bin ${q.bin.toFixed(1)}), ρ = ${q.rho.toFixed(1)}, ${(100 * q.shared).toFixed(0)}% of its ρ² in shared bins.`;
}

function overviewPanel(c) {
  const host = document.getElementById("p-overview"), wb = c.d.window_bins, {W, X} = xScale(host, 0, wb);
  const rowH = 22, t = 6, rows = 4 * rowH, gap = 16, ph = 190, b = 38, H = t + rows + gap + ph + b, y0 = t + rows + gap;
  const data = power(c.y).map(Math.sqrt);
  let top = Math.sqrt(6) * 2;
  data.forEach(v => top = Math.max(top, v));
  c.sources.forEach(q => q.pw.forEach(v => top = Math.max(top, Math.sqrt(v))));
  const at = Math.asinh(top * 1.1), Y = v => y0 + ph - Math.asinh(v) / at * ph;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="the window's sources and their overlaps">`;
  runs(Array.from(c.count, v => v >= 2)).forEach(([a, z]) => s += `<rect x="${X(a - 0.5)}" width="${Math.max(1, X(z + 0.5) - X(a - 0.5))}" y="${t}" height="${rows + gap + ph}" fill="var(--ink)" opacity="0.07"/>`);
  if (S.view) s += `<rect x="${X(c.b0 - 0.5)}" width="${X(c.b1 - 0.5) - X(c.b0 - 0.5)}" y="${t - 3}" height="${rows + gap + ph + 6}" fill="none" stroke="var(--ink-2)" stroke-dasharray="3 3"/>`;
  c.sources.forEach(q => {
    const yr = t + q.s * rowH, mid = yr + rowH / 2, a = Math.max(0, q.seg0), z = Math.min(wb - 1, q.seg0 + c.d.n - 1);
    if (q.s === S.src) s += `<rect x="${M.l}" width="${W - M.l - M.r}" y="${yr}" height="${rowH}" fill="var(--grid)" opacity="0.6"/>`;
    s += `<text x="${M.l - 6}" y="${mid + 4}" text-anchor="end" class="lbl">ρ ${q.rho.toFixed(0)}</text>`;
    if (z >= a) s += `<line x1="${X(a - 0.5)}" x2="${X(z + 0.5)}" y1="${mid}" y2="${mid}" stroke="${SLOT[q.s]}" stroke-width="1.5" opacity="0.5"/>`;
    runs(q.keep).forEach(([u, v]) => s += `<rect x="${X(u - 0.5)}" width="${Math.max(2, X(v + 0.5) - X(u - 0.5))}" y="${mid - 5}" height="10" rx="2" fill="${SLOT[q.s]}"/>`);
    s += `<line x1="${X(q.bin)}" x2="${X(q.bin)}" y1="${mid - 9}" y2="${mid + 9}" stroke="var(--ink)" stroke-width="1.5"/>`;
    s += `<rect class="pick" data-s="${q.s}" x="0" width="${W}" y="${yr}" height="${rowH}" fill="transparent" style="cursor:pointer"/>`;
  });
  [0, 1, 3, 10, 30, 100, 300, 1000, 3000].filter(v => v <= top * 1.1).forEach(v => s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(Math.sqrt(6))}" y2="${Y(Math.sqrt(6))}" stroke="var(--ink-2)" stroke-dasharray="4 3"/>`;
  const path = (a, lo = 0, hi = a.length - 1) => { let p = ""; for (let k = Math.max(0, lo); k <= Math.min(a.length - 1, hi); k++) p += `${p ? "L" : "M"}${X(k).toFixed(1)},${Y(a[k]).toFixed(1)}`; return p; };
  s += `<path d="${path(data)}" fill="none" stroke="var(--muted)" stroke-width="1" stroke-linejoin="round"/>`;
  c.sources.forEach(q => s += `<path d="${path(q.pw.map(Math.sqrt), q.seg0, q.seg0 + c.d.n - 1)}" fill="none" stroke="${SLOT[q.s]}" stroke-width="1.25" stroke-linejoin="round"/>`);
  s += `<line class="cross" y1="${t}" y2="${H - b}" stroke="var(--axis)" visibility="hidden"/>`;
  s += binAxis(X, W, H - b, 0, wb) + "</svg>";
  host.innerHTML = s;
  document.getElementById("lg-overview").innerHTML = c.sources.map(q => legendDot(SLOT[q.s], `source ${q.s + 1}`)).join("") + legendItem("var(--muted)", S.noise ? "data, with noise" : "data (noise off)") + legendItem("var(--ink-2)", "noise √6", "4 3") + legendBox("var(--grid)", "bins shared by ≥ 2 sources");
  const svg = host.querySelector("svg"), cross = svg.querySelector(".cross");
  svg.querySelectorAll(".pick").forEach(r => r.addEventListener("click", () => { S.src = Number(r.dataset.s); Object.values(marks).forEach(m => m()); schedule(); }));
  svg.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W, k = Math.round((px - M.l) / (W - M.l - M.r) * wb - 0.5);
    if (k < 0 || k >= wb) { cross.setAttribute("visibility", "hidden"); return untip(host); }
    cross.setAttribute("visibility", "visible"); cross.setAttribute("x1", X(k)); cross.setAttribute("x2", X(k));
    tip(host, ev, `<b>bin ${k}, ${((c.start + k) / c.d.t_obs * 1e3).toPrecision(6)} mHz</b>` + trow("var(--muted)", `data ${data[k].toFixed(2)}`) + c.sources.map(q => trow(SLOT[q.s], `source ${q.s + 1}: ${Math.sqrt(q.pw[k]).toFixed(2)}${q.keep[k] ? ", in its 99%" : ""}`)).join("") + (c.count[k] >= 2 ? trow(null, `${c.count[k]} sources share this bin`) : ""));
  });
  svg.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); untip(host); });
}

function layerPanel(c) {
  const host = document.getElementById("p-layers"), {W, X} = xScale(host, c.b0, c.b1);
  const rows = [...c.sources.map(q => ({name: `source ${q.s + 1}, ρ ${q.rho.toFixed(1)}`, y: q.y, colour: SLOT[q.s], bin: q.bin})),
    {name: "sum of the four", y: c.clean}, {name: S.noise ? `noise, draw ${S.seed}` : "noise (off)", y: c.noise}, {name: "data = sum + noise", y: c.y}];
  const h = 64, gap = 10, t = 4, b = 38, H = t + rows.length * (h + gap) + b;
  let top = 3;
  rows.forEach(r => [r.y[S.ch], r.y[S.ch + 3]].forEach(a => { for (let k = c.b0; k < c.b1; k++) top = Math.max(top, Math.abs(a[k])); }));
  const at = Math.asinh(top * 1.05);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="sources, noise and data">`;
  rows.forEach((r, i) => {
    const y0 = t + i * (h + gap), mid = y0 + h / 2, Y = v => mid - Math.asinh(v) / at * h / 2;
    const path = a => { let p = ""; for (let k = c.b0; k < c.b1; k++) p += `${k === c.b0 ? "M" : "L"}${X(k).toFixed(1)},${Y(a[k]).toFixed(1)}`; return p; };
    s += `<rect x="${M.l}" width="${W - M.l - M.r}" y="${Y(1)}" height="${Y(-1) - Y(1)}" fill="var(--grid)" opacity="0.7"/>`;
    s += `<line x1="${M.l}" x2="${W - M.r}" y1="${y0 + h}" y2="${y0 + h}" stroke="var(--grid)"/>`;
    [-top, 0, top].forEach(v => s += `<text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v ? (v > 0 ? "" : "−") + Math.abs(v).toFixed(0) : 0}</text>`);
    if (r.bin !== undefined && r.bin >= c.b0 && r.bin < c.b1) s += marker(X, r.bin, y0, y0 + h);
    s += `<path d="${path(r.y[S.ch])}" fill="none" stroke="var(--s1)" stroke-width="1.1" stroke-linejoin="round"/>`;
    s += `<path d="${path(r.y[S.ch + 3])}" fill="none" stroke="var(--s2)" stroke-width="1.1" stroke-linejoin="round"/>`;
    s += `<text class="lab" x="${M.l + 6}" y="${y0 + 12}">${r.name}</text>`;
    if (r.colour) s += `<circle cx="${M.l - 36}" cy="${mid}" r="4" fill="${r.colour}"/>`;
  });
  s += binAxis(X, W, H - b, c.b0, c.b1) + "</svg>";
  host.innerHTML = s;
}

function snrPanel(c) {
  const host = document.getElementById("p-snr"), W = Math.max(280, host.clientWidth);
  const rows = c.sources.map(q => ({name: `source ${q.s + 1}`, colour: SLOT[q.s], bar: q.rho, mf: q.mf}));
  rows.push({name: "window", colour: "var(--ink-2)", bar: c.windowSnr, mf: c.mf, tick: c.orthogonal});
  const values = rows.flatMap(r => [r.bar, r.mf, r.tick || r.bar]).filter(v => v > 0);
  const lo = 10 ** Math.floor(Math.log10(Math.min(1, ...values))), hi = 10 ** Math.ceil(Math.log10(Math.max(...values) * 1.2));
  const l = 84, r = 54, rowH = 26, t = 6, H = t + rows.length * rowH + 30;
  const V = v => l + Math.log(clamp(v, lo, hi) / lo) / Math.log(hi / lo) * (W - l - r);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="source and window SNRs">`;
  for (let e = Math.log10(lo); e <= Math.log10(hi); e++) s += `<line x1="${V(10 ** e)}" x2="${V(10 ** e)}" y1="${t}" y2="${t + rows.length * rowH}" stroke="var(--grid)"/><text x="${V(10 ** e)}" y="${t + rows.length * rowH + 16}" text-anchor="middle">${sig(10 ** e)}</text>`;
  rows.forEach((q, i) => {
    const y = t + i * rowH, mid = y + rowH / 2;
    s += `<text class="lbl" x="${l - 8}" y="${mid + 4}" text-anchor="end">${q.name}</text>`;
    s += `<rect x="${V(lo)}" y="${y + 6}" width="${V(q.bar) - V(lo)}" height="${rowH - 12}" rx="2" fill="${q.colour}" opacity="0.75"/>`;
    if (q.tick) s += `<line x1="${V(q.tick)}" x2="${V(q.tick)}" y1="${y + 2}" y2="${y + rowH - 2}" stroke="var(--ink)" stroke-width="1.5" stroke-dasharray="3 2"/>`;
    const m = V(q.mf);
    s += `<path d="M${m},${mid - 6} l6,6 l-6,6 l-6,-6 z" fill="var(--surface)" stroke="var(--ink)" stroke-width="1.5"/>`;
    s += `<text x="${W - r + 6}" y="${mid + 4}">${q.bar.toFixed(1)}</text>`;
  });
  host.innerHTML = s + `<text x="${(l + W - r) / 2}" y="${H - 2}" text-anchor="middle">SNR</text></svg>`;
  const pct = v => `${(100 * v).toFixed(0)}%`;
  document.getElementById("t-snr").innerHTML = table([
    ...c.sources.map(q => [`<i class="dot" style="background:${SLOT[q.s]}"></i> source ${q.s + 1}`, q.rho.toFixed(2), q.mf.toFixed(2), pct(q.shared)]),
    ["window", `${c.windowSnr.toFixed(2)} (√Σρ² ${c.orthogonal.toFixed(2)})`, c.mf.toFixed(2), "–"],
  ], ["", "ρ alone", "⟨d, ỹ⟩/‖ỹ‖", "ρ² in shared bins"]);
}

function tokenPanel(c) {
  const host = document.getElementById("p-tokens"), {W, X} = xScale(host, c.b0, c.b1), P = c.P;
  const n0 = c.b0 / P, cols = (c.b1 - c.b0) / P, Hh = 128, gap = 26, t = 18, b = 38, H = t + 2 * Hh + gap + b;
  let vmax = 2;
  for (let i = 0; i < cols; i++) for (let q = 0; q < P; q++) vmax = Math.max(vmax, c.tokens[n0 + i][q * 3 + S.ch]);
  const surface = rgb("--surface"), blue = rgb("--s1");
  const magnitude = raster(cols, P, (i, j) => mix(surface, blue, c.tokens[n0 + i][(P - 1 - j) * 3 + S.ch] / vmax));
  const phase = raster(cols, P, (i, j) => {
    const tok = c.tokens[n0 + i], at = (P - 1 - j) * 3 + S.ch, re = tok[3 * P + at], im = tok[6 * P + at];
    return mix(surface, hue(Math.atan2(im, re)), Math.hypot(re, im));
  });
  const wheel = raster(64, 1, i => hue(-Math.PI + TAU * (i + 0.5) / 64));
  const x0 = X(c.b0 - 0.5), x1 = X(c.b1 - 0.5), y1 = t + Hh + gap;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="STFT tokens">`;
  [[magnitude, t, `asinh|z|, blank 0 to full ${vmax.toFixed(2)}`], [phase, y1, "gated phase: hue arg z, strength |z|/√(1+|z|²)"]].forEach(([url, y, title]) => {
    s += `<text class="lbl" x="${M.l}" y="${y - 6}">${title}</text>`;
    s += `<image href="${url}" x="${x0}" y="${y}" width="${x1 - x0}" height="${Hh}" preserveAspectRatio="none"/>`;
    s += `<rect x="${x0}" y="${y}" width="${x1 - x0}" height="${Hh}" fill="none" stroke="var(--axis)"/>`;
    s += `<text x="${M.l - 6}" y="${y + 10}" text-anchor="end">T</text><text x="${M.l - 6}" y="${y + Hh}" text-anchor="end">0</text>`;
    if (cols <= 96) for (let i = 1; i < cols; i++) s += `<line x1="${X(c.b0 + i * P - 0.5)}" x2="${X(c.b0 + i * P - 0.5)}" y1="${y}" y2="${y + Hh}" stroke="var(--surface)" stroke-width="1"/>`;
    c.sources.filter(q => q.bin >= c.b0 && q.bin < c.b1).forEach(q => s += marker(X, q.bin, y + Hh, y + Hh + 8, SLOT[q.s]));
  });
  s += `<image href="${wheel}" x="${W - M.r - 90}" y="${y1 - 15}" width="90" height="8" preserveAspectRatio="none"/>`;
  s += `<text x="${W - M.r - 94}" y="${y1 - 8}" text-anchor="end">−π</text><text x="${W - M.r}" y="${y1 - 19}" text-anchor="end">π</text>`;
  s += binAxis(X, W, H - b, c.b0, c.b1) + "</svg>";
  host.innerHTML = s;
}

function table(rows, head) {
  return `<div class="tablewrap"><table class="t"><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function portCard() {
  const host = document.getElementById("p-port"), P = PG.port;
  if (!P) { host.innerHTML = "<p class='note'>This build skipped the port check (<code>--no-check</code>).</p>"; document.getElementById("n-port").innerHTML = ""; return; }
  const NAMES = {window: "whole window, 4 sources (preprocess)", window_snr: "window SNR (snr)", source_snrs: "each source's SNR (source_snrs)", tokens: "the window's stft_tokens, patches 4 and 16"};
  const rows = Object.keys(NAMES).map(k => [NAMES[k], ...["narrow", "full"].map(b => sig(P[b].worst[k], 2)), sig(P.full.tolerances[k], 1)]);
  host.innerHTML = table(rows, ["worst disagreement with canna", "narrow", "full", "tolerance"]);
  document.getElementById("n-port").innerHTML = `Every card above is computed in this page by a JavaScript port of jaxgb and canna. The build runs the port under node on the first ${P.full.windows} prior windows per band and stops on any disagreement past the tolerance. The source viewer's build checks each source's response, PSD and flow coordinates.`;
}

function meaning() {
  const P = PG.port;
  const items = [`<b>Windows are crowded.</b> Four sources, sorted loudest first, share response bins (shaded in the overview), so the flow must separate them, not just find one.`];
  if (P) items.push(`<b>What this page draws is what canna computes.</b> The port agrees with canna to ${sig(Math.max(...Object.values(P.full.worst), ...Object.values(P.narrow.worst)), 1)} at worst, whole four-source windows included.`);
  document.getElementById("meaning").innerHTML = items.map(t => `<li>${t}</li>`).join("");
}

function render() {
  const d = D(), P = S.patch, wb = d.window_bins, w = BANK[S.band][S.win], bank = bankSources();
  const noise = S.noise ? noiseFor(d) : bank.clean.map(a => new Float64Array(a.length));
  const y = bank.clean.map((a, c) => a.map((v, k) => v + noise[c][k]));
  const sources = bank.sources.map(q => ({...q, mf: dot(y, q.y) / q.rho}));
  const windowSnr_ = windowSnr(bank.clean);
  // the source view: the picked source's response bins and a quarter of them either side, in whole tokens
  let b0 = 0, b1 = wb;
  if (S.view) {
    const seg0 = sources[S.src].seg0;
    b0 = Math.max(0, Math.floor((seg0 - d.n / 4) / P) * P);
    b1 = Math.min(wb, Math.ceil((seg0 + d.n + d.n / 4) / P) * P);
  }
  const c = {d, P, f: w.f, start: bank.start, y, noise, clean: bank.clean, count: bank.count, sources, b0, b1,
    windowSnr: windowSnr_, mf: dot(y, bank.clean) / windowSnr_, orthogonal: Math.sqrt(sum(sources.map(q => q.rho ** 2))), tokens: stftTokens(y, P)};
  tiles(c); overviewPanel(c); layerPanel(c); snrPanel(c); tokenPanel(c);
  portCard();
  writeHash();
}

let pending = false;
function schedule() {  // one render per frame however fast the toggles change
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}
addEventListener("resize", schedule);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
new MutationObserver(schedule).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
readHash();
meaning();
render();
</script>
</body>
</html>
"""


def band_data(band: str) -> tuple[LisaGB, dict]:
    """The band's B-size problem and the constants the page's port reads."""
    problem = LisaGB(**load(f"{band}-B").problem)
    response = problem.response
    assert int(response.n) == problem.response_points
    data = dict(
        n=problem.response_points,
        t_obs=float(problem.t_obs),
        c=SPEED_OF_LIGHT,
        L=float(response.arm_length),
        fstar=float(np.mean(response.fstar)),
        pos=np.asarray(response.position).tolist(),  # (spacecraft, xyz, time) [m]
        oms=float(problem.oms_noise),
        acc=float(problem.acceleration_noise),
        G=GRAVITATIONAL_CONSTANT,
        sun=SUN_MASS,
        window_bins=problem.window_bins,
        stride=WINDOW_STRIDE,
    )
    return problem, data


def draw(problem: LisaGB, key: jax.Array) -> dict:
    """One prior window as draw_prior.py draws it, so the keys match."""
    key_f, key_p = jr.split(key)
    f = sample_f(problem, key_f)
    return dict(f=f, p=sample_physical(problem, key_p, f))


def bank(problem: LisaGB) -> list[dict]:
    """The page's prior windows: draw_prior.py's first BANK keys, so window i is prior.json's draw i."""
    d = jax.lax.map(partial(draw, problem), jr.split(jr.key(0), WINDOWS)[:BANK])
    return [
        dict(f=float(d["f"][i]), p=np.asarray(d["p"][i]).tolist())
        for i in range(BANK)
    ]


def run_node(node: str, script: str):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
    try:
        return json.loads(
            subprocess.run(
                [node, fh.name], capture_output=True, text=True, check=True
            ).stdout
        )
    finally:
        Path(fh.name).unlink()


def check_port(
    band: str, problem: LisaGB, data: dict, windows: list[dict], node: str
) -> dict:
    """The page's port against canna on whole bank windows, under node; stops on any disagreement."""
    cases = [(w["f"], w["p"]) for w in windows[:CHECKED_WINDOWS]]
    script = (
        CHECK_JS.replace("__DATA__", json.dumps(data))
        .replace("__PATCHES__", json.dumps(PATCHES))
        .replace("__WINDOWS__", json.dumps(cases))
    )
    result = run_node(node, JS_CORE + script)

    rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b)))
    worst = dict.fromkeys(TOLERANCES, 0.0)
    for (f, p), r in zip(cases, result["windows"]):
        f, p = jnp.asarray(f), jnp.asarray(p)
        assert r["start"] == int(window_start(problem, f)), (band, r["start"])
        y = preprocess(problem, clean_signal(problem, p, f), f)
        rho, rhos = float(snr(problem, p, f)), np.asarray(source_snrs(problem, p, f))
        errors = dict(
            window=rel(np.array(r["y"]).T, np.asarray(y)),
            window_snr=abs(r["snr"] - rho) / rho,
            source_snrs=float(np.max(np.abs(np.array(r["source_snrs"]) - rhos) / rhos)),
            tokens=max(
                float(
                    np.max(
                        np.abs(
                            np.array(r["tokens"][str(patch)])
                            - np.asarray(stft_tokens(y, patch))
                        )
                    )
                )
                for patch in PATCHES
            ),
        )
        worst |= {k: max(worst[k], v) for k, v in errors.items()}

    print(
        f"{band}: worst disagreement of the JS port with canna over {len(cases)} bank windows"
    )
    for k, v in worst.items():
        print(f"    {k:12s} {v:.1e}  (tolerance {TOLERANCES[k]:.0e})")
    failed = [k for k, v in worst.items() if not v < TOLERANCES[k]]
    assert not failed, f"{band}: the JavaScript port disagrees with canna on {failed}"
    return dict(worst=worst, tolerances=TOLERANCES, windows=len(cases))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="write the page without checking the JavaScript port under node",
    )
    parser.add_argument("--out", type=Path, default=FOLDER / "window-viewer.html")
    args = parser.parse_args()

    bands = {band: band_data(band) for band in BANDS}
    banks = {band: bank(problem) for band, (problem, _) in bands.items()}
    port = None
    if args.no_check:
        checked = "Its port of canna was <b>not</b> checked for this build."
    else:
        node = shutil.which("node")
        if node is None:
            raise SystemExit(
                "node not found: load it (see window-viewer.sbatch) or pass --no-check"
            )
        port = {
            band: check_port(band, problem, data, banks[band], node)
            for band, (problem, data) in bands.items()
        }
        checked = (
            "Its port was checked under node against canna's <code>preprocess</code>,"
            " <code>snr</code>, <code>source_snrs</code> and <code>stft_tokens</code>"
            " on whole prior windows."
        )

    parts = (
        '<span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(--s1)"'
        ' stroke-width="2"/></svg>real part</span><span><svg width="22" height="10"><line x1="1"'
        ' x2="21" y1="5" y2="5" stroke="var(--s2)" stroke-width="2"/></svg>imaginary part</span>'
    )
    page = (
        PAGE.replace("__CORE__", JS_CORE)
        .replace("__DATA__", json.dumps({band: data for band, (_, data) in bands.items()}))
        .replace("__BANK__", json.dumps(banks))
        .replace("__PAGE__", json.dumps(dict(port=port)))
        .replace("__SUB__", f"Built {date.today().isoformat()}. {checked}")
        .replace("__PARTS__", parts)
    )
    args.out.write_text(page)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
