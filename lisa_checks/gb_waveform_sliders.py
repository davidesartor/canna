"""Interactive view of one galactic binary's LISA response, with a slider per parameter.

Writes gb_waveform_sliders.html next to this file: the TDI A/E/T spectra of a single
source (real/imaginary parts of A, amplitudes of A/E/T against the noise), two
time-frequency phase maps of one channel, from its short-time Fourier transform and from
its WDM transform, and the slow amplitude envelope of A and E over the observation. A
sidebar holds a slider per source parameter (f0, fdot, A, ra, dec, psi, iota, phi0), all
acting together, and the channel and window length shared by the two maps.

The sliders combine freely because the response is recomputed in the browser: the
HTML carries a JavaScript port of jaxgb's JaxGB.get_tdi (FastGB slow response, FFT,
TDI 1.5, AET) plus the spacecraft positions of LisaGB's orbit on jaxgb's time grid.
Before writing, the port is checked against jaxgb itself under node, for random
sources across the prior, the WDM port against wdm_transform, and the STFT against a
direct sum; the script stops if any disagree.

Both maps show the same quantity, the source phase relative to the q/T carrier at the
pixel time, so they differ only by the transform. The STFT runs on jaxgb's slow
(heterodyned) series: Hann window of L samples, hop L/8, zero-padded to 4L, phase
referenced to each window centre. The WDM coefficients are real, W[n,m] =
Re(conj(C_nm) z[n,m]) with C_nm = exp(i pi/4 (1 - (-1)^(n+m))); the map shows the phase of
the complex z (the band-passed signal at t_n = n dT, of which W is one quadrature), with
the carrier phase 2 pi q t_n / T removed. The WDM window is wdm_transform's Meyer window
with a = 1/3, as canna used, and its channel spacing is the STFT frequency step n/L bins,
so its time pixels are L/2 samples (dT dF = 1/2). The phase of the full-length spectrum
scrambles across the Doppler sidebands; these maps wind slowly with the Doppler and drift.

Parameters are jaxgb's own columns, so the sky is (ra, dec) in the equatorial frame of
lisaorbits v3 and iota is the inclination in [0, pi] (LisaGB stores pi/2 - iota).

    uv run python lisa_checks/gb_waveform_sliders.py [--no-check]
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from lisaconstants import c as SPEED_OF_LIGHT  # noqa: E402
from wdm_transform.transforms import from_freq_to_wdm_band  # noqa: E402

from canna.lisa.priors import fdot_from_chirp_mass  # noqa: E402
from canna.lisa.problem import LisaGB  # noqa: E402

HERE = Path(__file__).parent
DEG = np.pi / 180.0
MONTH = 86400 * 365.25 / 12

# fiducial source: a 0.5 Msun chirp-mass DWD at 5 mHz, loud enough to see the sidebands
F0 = 5e-3
FDOT = float(fdot_from_chirp_mass(0.5, F0))
AMPLITUDE = 10**-22.5

# slider grids, each with the fiducial value on it; angles are given in degrees
GRIDS = {
    "f0": dict(
        values=F0 * 2.0 ** (np.arange(-16, 6) / 4),  # quarter octaves, 0.31-11.9 mHz
        fiducial=F0, label="f0", unit="mHz", scale=1e3, fmt="{:.3f}",
    ),
    "fdot": dict(
        values=np.array(
            [-1e-15, -5e-16, -2e-16, 0.0, 1e-16, 2e-16, 4e-16, FDOT,
             1e-15, 2e-15, 5e-15, 1e-14, 2e-14]
        ),
        fiducial=FDOT, label="fdot", unit="Hz/s", scale=1.0, fmt="{:.2e}",
    ),
    "amplitude": dict(
        values=10.0 ** np.linspace(-23, -21, 17),
        fiducial=AMPLITUDE, label="A", unit="", scale=1.0, fmt="{:.2e}",
    ),
    "ra": dict(
        values=np.arange(0, 361, 10) * DEG,
        fiducial=60 * DEG, label="ra", unit="°", scale=1 / DEG, fmt="{:.0f}",
    ),
    "dec": dict(
        values=np.arange(-90, 91, 10) * DEG,
        fiducial=20 * DEG, label="dec", unit="°", scale=1 / DEG, fmt="{:.0f}",
    ),
    "psi": dict(
        values=np.arange(0, 181, 10) * DEG,
        fiducial=30 * DEG, label="psi", unit="°", scale=1 / DEG, fmt="{:.0f}",
    ),
    "iota": dict(
        values=np.arange(0, 181, 10) * DEG,
        fiducial=60 * DEG, label="iota", unit="°", scale=1 / DEG, fmt="{:.0f}",
    ),
    "phi0": dict(
        values=np.arange(0, 361, 15) * DEG,
        fiducial=90 * DEG, label="phi0", unit="°", scale=1 / DEG, fmt="{:.0f}",
    ),
}

# window lengths L in slow-series samples (n = 1024 over 2 yr, 0.71 d each); the WDM
# transform takes nt = 2n/L time pixels, so it must stay even
WINDOWS = [16, 32, 64, 128, 256]
WINDOW = 64
CHANNELS = ["A", "E", "T"]
# map cells this far below the map's peak modulus are left blank
FLOORS_DB = [-20, -30, -40, -50, -60]
FLOOR_DB = -40
WDM_A = 1 / 3

# categorical slots 1-3 of the dataviz reference palette, plus recessive ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NOISE, INK, MUTED, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE, CONTROL = "#fcfcfb", "#f2f1ee"

# The JaxGB.get_tdi port. `D` holds the orbit and the instrument constants; one source,
# boxcar window, equal-arm TDI 1.5, exactly as jaxgb builds it (see jaxgb/jaxgb.py)
JS_CORE = r"""
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

// Hann STFT of a complex series, hop L/8, zero-padded to 4L; each frame's phase is
// referenced to its window centre. Returns frame centres, shifted frequencies (cycles per
// sample) and the complex frames [frame][re/im][freq]
function stft(re, im, L) {
  const n = re.length, N = 4 * L, h = L / 2, hop = L / 8;
  const centres = [], frames = [];
  for (let c = h; c + h <= n; c += hop) {
    const br = new Float64Array(N), bi = new Float64Array(N);
    for (let k = -h; k < h; k++) {  // centre sample at index 0
      const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * (k + h) / L), j = (k + N) % N;
      br[j] = re[c + k] * w; bi[j] = im[c + k] * w;
    }
    fft(br, bi);
    const sr = new Array(N), si = new Array(N);
    for (let r = 0; r < N; r++) { sr[r] = br[(r + N / 2) % N]; si[r] = bi[(r + N / 2) % N]; }
    centres.push(c); frames.push([sr, si]);
  }
  return {centres: centres, freqs: Array.from({length: N}, (_, r) => (r - N / 2) / N),
          frames: frames};
}

// WDM interior channels touched by a compact one-sided spectrum [sr, si] on bins
// kmin..kmin+n-1, as wdm_transform's from_freq_to_wdm with nt time bins and roll-off a.
// Returns the channels and the complex z[m][re/im][n]; the real coefficients are
// W = Re(conj(C_nm) z), C_nm = exp(i pi/4 (1 - (-1)^(n+m)))
function wdm(sr, si, kmin, nt, a) {
  const n = sr.length, h = nt / 2, b = 1 - 2 * a;
  const phi = new Float64Array(nt);
  for (let l = 0; l < nt; l++) {  // the phi window on the fftfreq grid
    const u = Math.abs(2 * (l < h ? l : l - nt) / nt);
    phi[l] = (u <= a ? 1 : u <= a + b ? Math.cos(Math.PI / 2 * (u - a) / b) : 0) * Math.sqrt(2 / nt);
  }
  const channels = [], z = [];
  for (let m = Math.floor(kmin / h); m <= Math.ceil((kmin + n) / h); m++) {
    const re = new Float64Array(nt), im = new Float64Array(nt);
    for (let l = 0; l < nt; l++) {  // the block of bins [(m-1)h, (m+1)h), centred on m h
      const i = m * h + (l < h ? l : l - nt) - kmin;
      if (i >= 0 && i < n) { re[l] = sr[i] * phi[l]; im[l] = -si[i] * phi[l]; }
    }
    fft(re, im);  // nt * ifft(x) = conj(fft(conj(x)))
    const zr = new Array(nt), zi = new Array(nt);
    for (let k = 0; k < nt; k++) {
      const s = (k * m) % 2 ? -Math.SQRT2 : Math.SQRT2;
      zr[k] = s * re[k]; zi[k] = -s * im[k];
    }
    channels.push(m); z.push([zr, zi]);
  }
  return {channels: channels, z: z};
}

// p = [f0, fdot, A, ra, dec, psi, iota, phi0], jaxgb's columns
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
  const kmin = Math.round(q - n / 2);
  return {kmin: kmin, spec: spec, slow: slow};
}

// canna's LisaGB.noise_psd: TDI 1.5 A/E and T, fractional frequency [1/Hz]
function noisePsd(f, D) {
  f = Math.abs(f);
  const phase = 2 * Math.PI * D.L / D.c * f, tr = Math.cos(phase);
  const dop = (2 * Math.PI * f / D.c) ** 2;
  const oms = D.oms ** 2 * 1e-24 * (1 + (0.002 / f) ** 4);
  const acc = D.acc ** 2 * 1e-30 * (1 + (0.0004 / f) ** 2) * (1 + (f / 0.008) ** 4);
  const om = oms * dop, tm = acc * dop / (2 * Math.PI * f) ** 4;
  const ae = 2 * tm * (1 + tr + tr * tr) + 0.5 * om * (2 + tr);
  const t = 2 * tm * (1 - tr) ** 2 + om * (1 - tr);
  const g = 16 * Math.sin(phase) ** 2;
  return [ae * g, ae * g, t * g];
}
"""

# Browser side: builds the sidebar controls from D, keeps the state and redraws all
# traces, the two maps and the readout from the port above. Plotly runs it right after
# the plot div, before the sidebar is parsed, so it waits for the DOM
JS_PAGE = r"""
function start() {
const gd = document.getElementById('{plot_id}');
const D = __DATA__;
const state = D.names.map(name => D.grids[name].fiducial_index);
let win = D.windows.indexOf(D.window), channel = D.channels[0];
let floor = D.floors_db.indexOf(D.floor_db);
const MONTH = 86400 * 365.25 / 12;

// phase of a complex cell, or null (a gap in the map) below the floor of the peak modulus
function phaseMap(cells, rows, cols) {
  let peak = 0;
  for (const r of rows) for (const c of cols) peak = Math.max(peak, Math.hypot(...cells(r, c)));
  const cut = peak * 10 ** (D.floors_db[floor] / 20);
  return rows.map(r => cols.map(c => {
    const [re, im] = cells(r, c);
    return Math.hypot(re, im) > cut ? Math.atan2(im, re) : null;
  }));
}

function redraw() {
  const p = D.names.map((name, i) => D.grids[name].values[state[i]]);
  const [f0, fdot] = p;
  const out = gbResponse(p, D), n = D.n, T = D.t_obs;
  const x = [], amp = {A: [], E: [], T: []}, sigAE = [], sigT = [];
  let snr2 = 0;
  for (let i = 0; i < n; i++) {
    const f = (out.kmin + i) / T;
    x.push(f * T - f0 * T);  // bins from f0
    const S = noisePsd(f, D);
    sigAE.push(Math.sqrt(S[0] * T / 2)); sigT.push(Math.sqrt(S[2] * T / 2));
    ["A", "E", "T"].forEach((ch, j) => {
      const [re, im] = out.spec[ch];
      const a2 = re[i] * re[i] + im[i] * im[i];
      amp[ch].push(Math.sqrt(a2));
      snr2 += 2 * a2 / (S[j] * T / 2);
    });
  }
  // zoom on the central 99.9% of the A+E power (the boxcar leakage tails are left out)
  const power = amp.A.map((a, i) => a * a + amp.E[i] * amp.E[i]);
  const total = power.reduce((s, v) => s + v, 0);
  let lo = 0, hi = n - 1, acc = 0;
  while (acc + power[lo] < 5e-4 * total) acc += power[lo++];
  acc = 0;
  while (acc + power[hi] < 5e-4 * total) acc += power[hi--];
  const pad = Math.max(8, Math.round(0.3 * (hi - lo)));
  const range = [x[Math.max(lo - pad, 0)], x[Math.min(hi + pad, n - 1)]];
  const t = [], envA = [], envE = [];
  for (let i = 0; i < n; i++) {
    t.push(i * T / n / MONTH);
    envA.push(Math.hypot(out.slow.A[0][i], out.slow.A[1][i]));
    envE.push(Math.hypot(out.slow.E[0][i], out.slow.E[1][i]));
  }
  const L = D.windows[win], q = out.kmin + n / 2, days = s => (s * T / n / 86400).toFixed(0);
  const near = (v, margin) => v >= range[0] - margin && v <= range[1] + margin;

  // STFT of the slow series, which sits at the q/T carrier, in bins from f0
  const sl = out.slow[channel], st = stft(sl[0], sl[1], L);
  const fy = st.freqs.map(v => v * n + q - f0 * T);
  const sRows = fy.map((v, r) => r).filter(r => near(fy[r], n / L));
  const sCols = st.frames.map((_, c) => c);
  const sZ = phaseMap((r, c) => [st.frames[c][0][r], st.frames[c][1][r]], sRows, sCols);

  // WDM with nt = 2n/L time pixels, its channel spacing the STFT frequency step n/L
  const nt = 2 * n / L, h = nt / 2, [sr, si] = out.spec[channel];
  const wd = wdm(sr, si, out.kmin, nt, D.wdm_a);
  const wy = wd.channels.map(m => m * h - f0 * T);
  const wRows = wy.map((v, r) => r).filter(r => near(wy[r], h));
  const wCols = Array.from({length: nt}, (_, k) => k);
  const wZ = phaseMap((r, k) => {  // take the carrier phase 2 pi q t_k / T off z
    const [zr, zi] = wd.z[r], a = -2 * Math.PI * ((q % nt) * k % nt) / nt;
    const c = Math.cos(a), s = Math.sin(a);
    return [zr[k] * c - zi[k] * s, zr[k] * s + zi[k] * c];
  }, wRows, wCols);

  Plotly.restyle(gd, {
    x: [st.centres.map(c => c * T / n / MONTH), wCols.map(k => k * T / nt / MONTH)],
    y: [sRows.map(r => fy[r]), wRows.map(r => wy[r])],
    z: [sZ, wZ],
  }, [7, 8]);
  Plotly.update(gd, {
    x: [x, x, x, x, x, x, x, t, t],
    y: [out.spec.A[0], out.spec.A[1], amp.A, amp.E, amp.T, sigAE, sigT, envA, envE],
  }, {
    'xaxis.range': range, 'xaxis2.range': range, 'yaxis3.range': range, 'yaxis4.range': range,
    'annotations[2].text': `<b>STFT phase of ${channel}</b>   Hann window ${L} samples` +
      ` (${days(L)} d), hop ${L / 8}, frequency step ${n / L} bins`,
    'annotations[3].text': `<b>WDM phase of ${channel}</b>   a = 1/3, time pixel ${L / 2}` +
      ` samples (${days(L / 2)} d), channel spacing ${h} bins`,
  }, [0, 1, 2, 3, 4, 5, 6, 9, 10]);

  const mc = fdot > 0 ? (fdot / (D.chirp_k * f0 ** (11 / 3))) ** 0.6 : null;
  document.getElementById('readout').innerHTML =
    `SNR (A+E+T) = <b>${Math.sqrt(snr2).toFixed(1)}</b>` +
    `<span>drift fdot·T² = ${(fdot * T * T).toFixed(2)} bins</span>` +
    `<span>implied Mc = ${mc === null ? '—' : mc.toFixed(3) + ' M☉'}</span>` +
    `<span>f0·T = ${(f0 * T).toFixed(2)}</span>`;
}

let pending = false;
function schedule() {  // one redraw per frame however fast a slider is dragged
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; redraw(); });
}

function slider(parent, label, labels, index, onChange) {
  const box = document.createElement('div'), head = document.createElement('label');
  const name = document.createElement('span'), value = document.createElement('output');
  const input = document.createElement('input');
  box.className = 'ctl'; name.textContent = label; value.textContent = labels[index];
  Object.assign(input, {type: 'range', min: 0, max: labels.length - 1, step: 1, value: index});
  input.setAttribute('aria-label', label);
  input.addEventListener('input', () => {
    value.textContent = labels[input.value]; onChange(Number(input.value)); schedule();
  });
  head.append(name, value); box.append(head, input); parent.append(box);
  return i => { input.value = i; value.textContent = labels[i]; };
}

const setters = D.names.map((name, i) => slider(
  document.getElementById('source'), D.grids[name].label, D.grids[name].labels,
  state[i], v => { state[i] = v; }));
slider(document.getElementById('window'), 'window L', D.window_labels, win, v => { win = v; });
slider(document.getElementById('window'), 'blank cells below peak', D.floor_labels, floor,
  v => { floor = v; });
const seg = document.getElementById('channel');
for (const ch of D.channels) {
  const b = document.createElement('button');
  b.textContent = ch; b.setAttribute('aria-pressed', String(ch === channel));
  b.addEventListener('click', () => {
    channel = ch;
    for (const o of seg.children) o.setAttribute('aria-pressed', String(o === b));
    schedule();
  });
  seg.append(b);
}
document.getElementById('reset').addEventListener('click', () => {
  D.names.forEach((name, i) => { state[i] = D.grids[name].fiducial_index; setters[i](state[i]); });
  schedule();
});
redraw();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GB waveform sliders</title>
<style>
  :root {{ --ink: {ink}; --muted: {muted}; --grid: {grid}; --surface: {surface};
           --control: {control}; --accent: {accent}; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--surface); color: var(--ink);
          font: 14px/1.4 Inter, system-ui, sans-serif; }}
  header {{ padding: 16px 24px 0; }}
  h1 {{ font-size: 18px; font-weight: 600; margin: 0 0 6px; }}
  #readout {{ margin: 0; color: var(--ink); font-variant-numeric: tabular-nums; }}
  #readout span {{ color: var(--muted); margin-left: 18px; }}
  main {{ display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px;
          padding: 0 24px 24px 8px; }}
  aside {{ position: sticky; top: 0; align-self: start; max-height: 100vh; overflow-y: auto;
           padding: 16px 0 16px 8px; }}
  h2 {{ font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
        color: var(--muted); margin: 20px 0 10px; }}
  .ctl {{ margin-bottom: 12px; }}
  .ctl label {{ display: flex; justify-content: space-between; gap: 8px; font-size: 13px; }}
  .ctl output {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
  .ctl input {{ width: 100%; margin: 4px 0 0; accent-color: var(--accent); }}
  button {{ font: inherit; color: var(--ink); background: var(--control);
            border: 1px solid var(--grid); border-radius: 6px; padding: 5px 12px; cursor: pointer; }}
  button:hover {{ border-color: var(--muted); }}
  .seg {{ display: flex; gap: 4px; margin-bottom: 12px; }}
  .seg button {{ flex: 1; }}
  .seg button[aria-pressed="true"] {{ background: var(--accent); border-color: var(--accent);
                                      color: #fff; }}
  .note {{ color: var(--muted); font-size: 12px; margin: 4px 0 0; }}
  @media (max-width: 900px) {{
    main {{ grid-template-columns: 1fr; padding: 0 16px 16px; }}
    aside {{ position: static; max-height: none; order: -1; padding: 8px 0 0; }}
  }}
</style>
</head>
<body>
<header>
  <h1>A galactic binary seen by LISA: TDI 1.5 response (jaxgb, 2 yr)</h1>
  <p id="readout"></p>
</header>
<main>
  <div>{plot}</div>
  <aside>
    <button id="reset">Reset source to fiducial</button>
    <h2>Source</h2>
    <div id="source"></div>
    <h2>Time-frequency maps</h2>
    <div class="seg" id="channel" role="group" aria-label="channel"></div>
    <div id="window"></div>
    <p class="note">Both maps show the phase relative to the q/T carrier at each pixel's
    time, blank where the modulus is below the chosen fraction of the map's peak. The
    STFT uses a Hann window of L samples; the WDM (a = 1/3) has channel spacing n/L bins, the STFT
    frequency step, so its time pixels are L/2 samples. The WDM coefficients are real, one
    quadrature of the complex z whose phase is shown. Its first and last pixels light up
    across the band: the transform is periodic in time and the 2-yr boxcar is a jump
    there.</p>
  </aside>
</main>
</body>
</html>
"""


def build_data(problem: LisaGB) -> dict:
    response = problem.response
    grids = {}
    for name, g in GRIDS.items():
        values = np.asarray(g["values"], dtype=float)
        index = int(np.argmin(np.abs(values - g["fiducial"])))
        assert np.isclose(values[index], g["fiducial"], rtol=1e-12, atol=0), name
        unit = f" {g['unit']}" if g["unit"] not in ("", "°") else g["unit"]
        labels = [g["fmt"].format(v * g["scale"]) + unit for v in values]
        grids[name] = dict(values=values.tolist(), fiducial_index=index, label=g["label"],
                           labels=labels)
    n, t_obs = int(response.n), float(response.t_obs)
    assert all(2 * n % w == 0 and (2 * n // w) % 2 == 0 for w in WINDOWS)
    return dict(
        n=n,
        t_obs=t_obs,
        c=float(SPEED_OF_LIGHT),
        L=float(response.arm_length),
        fstar=float(np.mean(response.fstar)),
        pos=np.asarray(response.position).tolist(),  # (spacecraft, xyz, time) [m]
        oms=float(problem.oms_noise),
        acc=float(problem.acceleration_noise),
        chirp_k=float(fdot_from_chirp_mass(1.0, 1.0)),
        names=list(GRIDS),
        grids=grids,
        windows=WINDOWS,
        window=WINDOW,
        window_labels=[f"{w} samples ({w * t_obs / n / 86400:.0f} d)" for w in WINDOWS],
        channels=CHANNELS,
        floors_db=FLOORS_DB,
        floor_db=FLOOR_DB,
        floor_labels=[f"{f} dB ({100 * 10 ** (f / 20):.2g}%)".replace("-", "−")
                      for f in FLOORS_DB],
        wdm_a=WDM_A,
    )


def run_node(node: str, script: str):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
    try:
        return json.loads(subprocess.run(
            [node, fh.name], capture_output=True, text=True, check=True
        ).stdout)
    finally:
        Path(fh.name).unlink()


def check_port(problem: LisaGB, data: dict, n_sources: int = 12) -> None:
    """The JS port against jaxgb's get_tdi (AET) and canna's noise_psd, its WDM against
    wdm_transform's from_freq_to_wdm_band and its STFT against a direct sum, under node."""
    node = shutil.which("node")
    if node is None:
        print("node not found: skipping the check of the JavaScript port")
        return
    rng = np.random.default_rng(0)
    f0 = np.exp(rng.uniform(np.log(3e-4), np.log(1.2e-2), n_sources))
    params = np.stack([
        f0,
        rng.uniform(-1e-15, 2e-14, n_sources),
        np.exp(rng.uniform(np.log(1e-23), np.log(1e-21), n_sources)),
        rng.uniform(0, 2 * np.pi, n_sources),
        np.arcsin(rng.uniform(-1, 1, n_sources)),
        rng.uniform(0, np.pi, n_sources),
        np.arccos(rng.uniform(-1, 1, n_sources)),
        rng.uniform(0, 2 * np.pi, n_sources),
    ], axis=1)

    result = run_node(node, JS_CORE + f"""
const D = {json.dumps(data)};
const P = {json.dumps(params.tolist())};
const out = P.map(p => {{
  const r = gbResponse(p, D);
  return {{kmin: r.kmin, A: r.spec.A, E: r.spec.E, T: r.spec.T,
          psd: Array.from({{length: D.n}}, (_, i) => noisePsd((r.kmin + i) / D.t_obs, D))}};
}});
console.log(JSON.stringify(out));
""")
    response = problem.response
    aet = np.stack(response.get_tdi(jnp.asarray(params), 1.5, "AET"), axis=1)  # (S, 3, n)
    kmin = np.asarray(response.get_kmin(jnp.asarray(params[:, 0])))
    worst_tdi = worst_psd = 0.0
    for s, r in enumerate(result):
        assert r["kmin"] == kmin[s], (s, r["kmin"], kmin[s])
        js = np.stack([np.array(r[ch][0]) + 1j * np.array(r[ch][1]) for ch in "AET"])
        err = np.max(np.abs(js - aet[s])) / np.max(np.abs(aet[s]))
        freqs = (kmin[s] + np.arange(response.n)) / response.t_obs
        psd = np.asarray(problem.noise_psd(jnp.asarray(freqs)))
        err_psd = np.max(np.abs(np.array(r["psd"]) - psd) / psd)
        worst_tdi, worst_psd = max(worst_tdi, err), max(worst_psd, err_psd)
    print(f"JS port vs jaxgb over {n_sources} random sources: "
          f"max |dTDI|/max|TDI| = {worst_tdi:.1e}, max |dPSD|/PSD = {worst_psd:.1e}")
    assert worst_tdi < 1e-8 and worst_psd < 1e-10, "the JavaScript port disagrees with jaxgb"

    # WDM of jaxgb's A spectra at every window length; the full grid only sets which
    # channels exist, so any N = nt nf above the band does
    n, t_obs, n_full = data["n"], data["t_obs"], 2**21
    cases = [(s, 2 * n // w) for s in range(3) for w in WINDOWS]
    result = run_node(node, JS_CORE + f"""
const S = {json.dumps([[aet[s, 0].real.tolist(), aet[s, 0].imag.tolist(), int(kmin[s]), nt]
                       for s, nt in cases])};
console.log(JSON.stringify(S.map(([sr, si, k, nt]) => wdm(sr, si, k, nt, {WDM_A!r}))));
""")
    worst_wdm = 0.0
    for (s, nt), r in zip(cases, result):
        ch = np.array(r["channels"])
        z = np.array([np.array(a) + 1j * np.array(b) for a, b in r["z"]]).T  # (nt, channels)
        c_nm = np.exp(1j * np.pi / 4 * (1 - (-1.0) ** (np.arange(nt)[:, None] + ch)))
        w_js = np.real(np.conj(c_nm) * z)
        inside = ((ch - 1) * nt // 2 >= kmin[s]) & ((ch + 1) * nt // 2 <= kmin[s] + n)
        ref = np.asarray(from_freq_to_wdm_band(
            aet[s, 0], df=1 / t_obs, nfreqs_fourier=n_full // 2 + 1, kmin=int(kmin[s]),
            nfreqs_wdm=n_full // nt, ntimes_wdm=nt, mmin=int(ch[inside][0]),
            nf_sub_wdm=int(inside.sum()), a=WDM_A,
        ))
        worst_wdm = max(worst_wdm, np.max(np.abs(w_js[:, inside] - ref)) / np.max(np.abs(ref)))

    # STFT of a random complex series against the direct windowed sum
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    result = run_node(node, JS_CORE + f"""
const re = {json.dumps(x.real.tolist())}, im = {json.dumps(x.imag.tolist())};
console.log(JSON.stringify({json.dumps(WINDOWS)}.map(L => stft(re, im, L))));
""")
    worst_stft = 0.0
    for w, r in zip(WINDOWS, result):
        k = np.arange(-w // 2, w // 2)
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * (k + w // 2) / w)
        kernel = np.exp(-2j * np.pi * np.outer(k, r["freqs"]))
        for c, (fr, fi) in zip(r["centres"], r["frames"]):
            ref = (x[c + k] * hann) @ kernel
            err = np.max(np.abs(np.array(fr) + 1j * np.array(fi) - ref)) / np.max(np.abs(ref))
            worst_stft = max(worst_stft, err)
    print(f"JS WDM vs wdm_transform: max |dW|/max|W| = {worst_wdm:.1e}; "
          f"JS STFT vs direct sum: {worst_stft:.1e}")
    assert worst_wdm < 1e-10 and worst_stft < 1e-10, "the JavaScript transforms disagree"


def build_figure(data: dict) -> go.Figure:
    fig = go.Figure()
    line = dict(width=2)
    # trace order is the redraw's: 0-6 spectra, 7 STFT map, 8 WDM map, 9-10 envelope
    traces = [
        ("Re A", "x", "y", dict(line, color=BLUE)),
        ("Im A", "x", "y", dict(line, color=ORANGE)),
        ("|A|", "x2", "y2", dict(line, color=BLUE)),
        ("|E|", "x2", "y2", dict(line, color=ORANGE)),
        ("|T|", "x2", "y2", dict(line, color=AQUA)),
        ("noise σ A, E", "x2", "y2", dict(color=NOISE, width=1.5, dash="dash")),
        ("noise σ T", "x2", "y2", dict(color=NOISE, width=1.5, dash="dot")),
        ("|A(t)|", "x5", "y5", dict(line, color=BLUE)),
        ("|E(t)|", "x5", "y5", dict(line, color=ORANGE)),
    ]
    legend = {"x": "legend", "x2": "legend2", "x5": "legend3"}
    for k, (name, xa, ya, ln) in enumerate(traces):
        if k == 7:
            for xm, ym in (("x3", "y3"), ("x4", "y4")):
                fig.add_trace(go.Heatmap(
                    x=[], y=[], z=[], xaxis=xm, yaxis=ym, coloraxis="coloraxis",
                    hoverongaps=False, showlegend=False,
                    hovertemplate="t = %{x:.1f} months<br>(f − f0)·T = %{y:.1f} bins"
                                  "<br>phase = %{z:.2f} rad<extra></extra>",
                ))
        fig.add_trace(go.Scatter(
            x=[], y=[], name=name, mode="lines", line=ln, xaxis=xa, yaxis=ya,
            legend=legend[xa], hovertemplate=f"{name}: %{{y:.3e}}<extra></extra>",
        ))

    axis = dict(
        showgrid=True, gridcolor=GRID, zeroline=False, linecolor=MUTED, ticks="outside",
        tickcolor=MUTED, tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=12),
        showspikes=True, spikemode="across", spikethickness=1, spikecolor=MUTED, spikedash="dot",
    )
    bins, months = "(f − f0) · T_obs  [bins]", "t  [months]"
    rows = [[0.855, 1.0], [0.645, 0.79], [0.43, 0.575], [0.215, 0.36], [0.0, 0.145]]
    t_range = [0, data["t_obs"] / MONTH]
    fig.update_layout(
        xaxis=dict(axis, domain=[0, 1], anchor="y"),
        yaxis=dict(axis, domain=rows[0], anchor="x", title_text="Ã(f)  [1/Hz]",
                   exponentformat="power"),
        xaxis2=dict(axis, domain=[0, 1], anchor="y2", matches="x", title_text=bins),
        yaxis2=dict(axis, domain=rows[1], anchor="x2", type="log", exponentformat="power",
                    title_text="|Ã|, |Ẽ|, |T̃|  [1/Hz]"),
        xaxis3=dict(axis, domain=[0, 1], anchor="y3", matches="x5", showgrid=False),
        yaxis3=dict(axis, domain=rows[2], anchor="x3", showgrid=False, title_text=bins),
        xaxis4=dict(axis, domain=[0, 1], anchor="y4", matches="x5", showgrid=False),
        yaxis4=dict(axis, domain=rows[3], anchor="x4", showgrid=False, title_text=bins),
        xaxis5=dict(axis, domain=[0, 1], anchor="y5", range=t_range, title_text=months),
        yaxis5=dict(axis, domain=rows[4], anchor="x5", rangemode="tozero",
                    exponentformat="power", title_text="|A(t)|, |E(t)|  [frac. freq.]"),
        coloraxis=dict(
            colorscale="twilight", cmin=-np.pi, cmax=np.pi,
            colorbar=dict(
                x=1.01, xanchor="left", y=rows[3][0], yanchor="bottom",
                len=rows[2][1] - rows[3][0], thickness=12, outlinewidth=0,
                tickvals=np.pi * np.array([-1, -0.5, 0, 0.5, 1]),
                ticktext=["−π", "−π/2", "0", "π/2", "π"],
                tickfont=dict(color=MUTED, size=11),
                title=dict(text="phase", font=dict(color=MUTED, size=12)),
            ),
        ),
    )
    # annotations 2 and 3 are rewritten by the redraw with the channel and window
    for i, (row, title) in enumerate(zip(rows, [
        "TDI A spectrum, real and imaginary parts",
        "Amplitude spectra against the per-bin noise σ = √(S T/2)",
        "STFT phase", "WDM phase",
        "Amplitude modulation of A and E over the observation (heterodyned slow response)",
    ])):
        fig.add_annotation(
            text=f"<b>{title}</b>", x=0, y=row[1] + 0.008, xref="paper", yref="paper",
            xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=13, color=INK),
        )
    for key, row in zip(["legend", "legend2", "legend3"], [rows[0], rows[1], rows[4]]):
        fig.update_layout({key: dict(
            x=1.01, y=row[1], xanchor="left", yanchor="top",
            font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)",
        )})
    fig.update_layout(
        height=1520, margin=dict(l=80, r=110, t=50, b=50),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, system-ui, sans-serif", color=INK),
        hovermode="x unified", hoverlabel=dict(bgcolor=SURFACE, font_color=INK),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-check", action="store_true",
                        help="skip checking the JavaScript port against jaxgb under node")
    parser.add_argument("--out", type=Path, default=HERE / "gb_waveform_sliders.html")
    args = parser.parse_args()

    problem = LisaGB()
    data = build_data(problem)
    if not args.no_check:
        check_port(problem, data)

    fig = build_figure(data)
    script = JS_CORE + JS_PAGE.replace("__DATA__", json.dumps(data))
    plot = fig.to_html(full_html=False, include_plotlyjs="cdn", post_script=script,
                       div_id="gb-plot", config=dict(displaylogo=False, responsive=True))
    args.out.write_text(PAGE.format(
        plot=plot, ink=INK, muted=MUTED, grid=GRID, surface=SURFACE,
        control=CONTROL, accent=BLUE,
    ))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
