"""the corner pages: full, pooled over sources, and one source, on synthetic draws."""

import matplotlib
import numpy as np
import pytest

from canna.lisa.plots import PARAM_IS_LOG, PARAM_LABELS, axis_ranges, corner_page, replot_page

LABELS = [f"${label}$" for label in PARAM_LABELS]
SCALE = ["log" if is_log else "linear" for is_log in PARAM_IS_LOG]


@pytest.fixture(scope="module")
def draws():
    """[draw, source, param] around two sources, f0 a few bins apart, amplitudes a decade apart."""
    matplotlib.use("Agg")
    truth = np.array(
        [[1.0e-3 + 1e-7 * s, 0.4, 1e-21 * 10.0**s, 1.0, 0.1, 2.0, 0.5, 3.0] for s in range(2)]
    )
    noise = 1 + 1e-4 * np.random.default_rng(0).standard_normal((256, 2, 8))
    return truth[None] * noise, truth


def test_ranges_pad_log_axes_in_log_space(draws):
    d, _ = draws
    flat = d.reshape(-1, 8)
    ranges = axis_ranges(flat, flat, SCALE)
    lo, hi = ranges[2]  # amplitude, a decade apart: 5% of a decade either side
    assert np.isclose(np.log10(flat[:, 2].min()) - np.log10(lo), 0.05 * np.log10(hi / lo) / 1.1, rtol=0.05)
    assert all(lo < hi for lo, hi in ranges)


@pytest.mark.parametrize("kind", ["full", "pooled", "source"])
def test_every_kind_of_page_renders(draws, tmp_path, kind):
    d, truth = draws
    n, s, p = d.shape
    if kind == "full":
        args = (d.reshape(n, -1), d.reshape(n, -1), LABELS * s, SCALE * s, truth.reshape(1, -1))
        kwargs = {}
    elif kind == "pooled":
        args = (d.reshape(-1, p), d.reshape(-1, p), LABELS, SCALE, truth)
        kwargs = dict(pooled=True, bins=50)
    else:
        args = (d[:, 0], d[:, 0], LABELS, SCALE, truth[:1])
        kwargs = {}
    path = tmp_path / f"{kind}.pdf"
    corner_page(path, *args[:4], kind, args[4], **kwargs)
    assert path.stat().st_size > 10_000


def test_pages_save_their_samples_and_replot_from_them(draws, tmp_path):
    d, truth = draws
    mcmc = d[:, 0] * (1 + 1e-5)
    path = tmp_path / "source.pdf"
    corner_page(path, d[:, 0], d[:, 1], LABELS, SCALE, "a title", truth[:1], mcmc=mcmc)
    saved = np.load(path.with_suffix(".npz"))
    assert np.array_equal(saved["flow"], d[:, 0]) and np.array_equal(saved["mcmc"], mcmc)
    assert list(saved["labels"]) == LABELS and str(saved["title"]) == "a title"
    again = replot_page(path.with_suffix(".npz"), tmp_path / "again.pdf")
    assert again.stat().st_size > 10_000


def test_a_few_strays_do_not_set_the_flow_axis():
    rng = np.random.default_rng(1)
    flow = rng.normal(0.0, 0.04, (1024, 8))
    flow[:12, 0] += 1.5  # 1.2% of the draws latched a bin and a half away
    mcmc = rng.normal(0.0, 0.01, (20000, 8))
    lo, hi = axis_ranges(flow, mcmc, ["linear"] * 8, mcmc)[0]
    assert hi < 0.5 and lo > -0.5  # the core, not the strays
    lo, hi = axis_ranges(flow, mcmc, ["linear"] * 8, mcmc, robust=False)[0]
    assert hi > 1.0  # the old rule, kept for pooled pages
