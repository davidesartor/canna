"""Signal vs noise PSD, checked against the analytic TDI 1.5 PSD.

Overlays the per-source signal periodogram at a few SNR percentiles against the
median noise periodogram and the analytic reference (all in |rfft|^2 units).
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from canna import lisa

from _helpers import problem_name, save_figure, save_text

SNR_PCTLS = [50, 75, 90, 100]
F_BAND = (1e-4, lisa.MAX_FREQUENCY)  # GB f0 prior range


def test_signal_noise(batch):
    signal = batch.signal  # (N, T, C), channel-last
    noise = batch.datastream - batch.signal
    channels = signal.shape[-1]
    assert np.isfinite(signal).all() and np.isfinite(noise).all()

    # representative injections: the sources at these SNR percentiles (100 = brightest)
    order = np.argsort(np.asarray(batch.snr))
    picks = {p: int(order[round(p / 100 * (batch.n - 1))]) for p in SNR_PCTLS}
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(picks)))

    n_t, dt = signal.shape[1], lisa.SAMPLING_STEP
    freqs = np.asarray(jnp.fft.rfftfreq(n_t, dt))
    p_sig = np.abs(np.fft.rfft(signal, axis=1)) ** 2  # (N, F, C)
    p_noise = np.abs(np.fft.rfft(noise, axis=1)) ** 2
    ref = np.asarray(lisa.noise_psd(jnp.asarray(freqs))).T * n_t / (2.0 * dt)  # (F, C)

    pos = freqs > 0
    band = (freqs >= F_BAND[0]) & (freqs <= F_BAND[1])
    picks_idx = np.array(list(picks.values()))
    fig, axes = plt.subplots(1, channels, figsize=(6 * channels, 4.5), squeeze=False)
    for ch in range(channels):
        ax = axes[0, ch]
        ax.loglog(freqs[pos], np.median(p_noise[:, pos, ch], axis=0), color="grey", lw=1.0, label="noise (median)")
        ax.loglog(freqs[pos], ref[pos, ch], "k--", lw=0.8, label="analytic PSD")
        for (p, idx), c in zip(picks.items(), colors):
            ax.loglog(freqs[pos], p_sig[idx, pos, ch], color=c, lw=0.9, label=f"p{p} snr={batch.snr[idx]:.0f}")
        # zoom to the GB band and the noise PSD range (signal is ~0 off its line bins)
        noise_band = np.median(p_noise[:, band, ch], axis=0)
        ax.set(xlim=F_BAND, ylim=(0.3 * noise_band.min(), 10.0 * noise_band.max()))
        ax.set(title=f"PSD · {lisa.CHANNEL_NAMES[ch]}", xlabel="f [Hz]", ylabel="|rfft|^2")
        ax.legend(fontsize=7)
    fig.suptitle(f"signal vs noise PSD — {problem_name()} (n={batch.n})")
    save_figure(fig, "signal_noise")

    lines = [f"signal vs noise — {problem_name()} (n={batch.n})"]
    for ch in range(channels):
        lines.append(
            f"  {lisa.CHANNEL_NAMES[ch]}: signal_std={signal[..., ch].std():.3g}  "
            f"noise_std={noise[..., ch].std():.3g}"
        )
    save_text("signal_noise", "\n".join(lines))
