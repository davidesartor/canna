"""The packed sample files: float16 offsets from the truth keep the draws to ~1e-3 of a width, outliers included."""

import numpy as np

from canna.lisa import samples


def test_a_packed_eval_reads_back_close_and_plain_files_unchanged(tmp_path):
    rng = np.random.default_rng(0)
    truth = np.array([[1.234567e-3, 0.5, 1e-22, 1.0, 0.2, 0.3, 0.4, 0.5]] * 4)[None].repeat(3, 0)
    width = np.array([1e-10, 0.05, 1e-24, 1e-3, 1e-3, 0.1, 0.1, 0.1])
    draws = truth[:, None] + width * rng.standard_normal((3, 256, 4, 8))
    draws[0, :3, 0, 0] += 3e-5  # three draws a few hundred thousand widths away
    np.savez(tmp_path / "draws.npz", draws=draws, matched=draws, fisher=draws, truth=truth, epoch=1)
    np.savez(tmp_path / "plain.npz", draws=draws, truth=truth)

    samples.pack_eval_dir(tmp_path)
    back = samples.load(tmp_path / "draws.npz")
    assert set(back) == {"draws", "matched", "fisher", "truth", "epoch"}
    error = np.abs(back["matched"] - draws) / width
    assert np.max(error[:, 3:]) < 5e-3  # the core: a few thousandths of a width
    assert np.allclose(back["matched"][0, :3, 0, 0], draws[0, :3, 0, 0], rtol=1e-6)  # the strays survive
    assert np.array_equal(back["truth"], truth)
    assert np.array_equal(samples.load(tmp_path / "plain.npz")["draws"], draws)
    assert samples.pack_eval_dir(tmp_path) == []  # packed already


def test_validation_rows_keep_their_truth_exactly():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 9, 4, 8))
    packed = samples.pack_validation({"x": x, "snr": rng.random((5, 9)), "f": rng.random(5)})
    centre, half, scale = packed["x.centre"], packed["x.half"], packed["x.scale"]
    assert np.array_equal(centre, x[:, 0]) and np.all(half[:, 0] == 0)
    assert np.allclose(centre[:, None] + half * scale[:, None], x, atol=5e-3 * scale.max())
    assert packed["snr"].dtype == np.float32 and packed["f"].dtype == np.float64
