"""Draws saved small: float16 offsets from a centre (the truth) in units of their robust width.

A packed array `name` is stored as `name.half` (float16 offsets / scale), `name.centre`, `name.scale` and
`name.axis` (the draw axis, absent from centre and scale). Offsets keep ~5e-4 of their value, so a draw a
few widths out is good to ~1e-3 of a width: f0 in Hz as float32 would step by ~1e-3 bins at 1 mHz, wider than
a loud source's posterior. `load` reads packed and plain files alike, as float64.

    python -m canna.lisa.samples <eval dir>...   # pack an eval's draws.npz and scorecard.npz in place
"""

import sys
import warnings
from pathlib import Path

import numpy as np

# float16's largest finite is 65504: the scale grows so that the farthest draw of a slice stays below this
HALF_LIMIT = 6.0e4


def robust_width(x: np.ndarray, axis: int) -> np.ndarray:
    """1.4826 × the median absolute deviation along axis, ignoring non-finite draws."""
    median = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - median), axis=axis)


def pack(name: str, draws: np.ndarray, centre: np.ndarray, axis: int) -> dict[str, np.ndarray]:
    """The npz entries of draws [..., N (at axis), ...] around centre (draws' shape without axis)."""
    draws = np.asarray(draws, np.float64)
    centre = np.asarray(centre, np.float64)
    offset = draws - np.expand_dims(centre, axis)
    # all-NaN slices (a Fisher draw with no valid value) warn and give NaN, handled below
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = robust_width(draws, axis)
        far = np.nanmax(np.abs(offset), axis=axis) / HALF_LIMIT
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    scale = np.maximum(scale, np.where(np.isfinite(far), far, 0.0))
    return {
        f"{name}.half": (offset / np.expand_dims(scale, axis)).astype(np.float16),
        f"{name}.centre": centre,
        f"{name}.scale": scale,
        f"{name}.axis": np.asarray(axis),
    }


def load(path: Path | str) -> dict[str, np.ndarray]:
    """Every array of an npz, packed ones unpacked to float64."""
    with np.load(path) as d:
        return unpack({k: d[k] for k in d.files})


def unpack(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The plain arrays as they are, packed ones unpacked to float64."""
    out = {k: v for k, v in raw.items() if "." not in k}
    for key in raw:
        if key.endswith(".half"):
            name = key.removesuffix(".half")
            axis = int(raw[f"{name}.axis"])
            out[name] = np.expand_dims(raw[f"{name}.centre"], axis) + raw[key].astype(
                np.float64
            ) * np.expand_dims(raw[f"{name}.scale"], axis)
    return out


def pack_draws(d: dict) -> dict:
    """draws.npz: the flow's, matched and Fisher draws [J, N, S, 8] around the truth [J, S, 8]."""
    out = {k: v for k, v in d.items() if k not in ("draws", "matched", "fisher")}
    for name in ("draws", "matched", "fisher"):
        if name in d:
            out |= pack(name, d[name], d["truth"], axis=1)
    return out


def pack_validation(d: dict) -> dict:
    """The validation samples: x and f0_bins [I, 1 + draws, ...] around their row 0 (the truth); the per-draw scalars in float32."""
    out = {}
    for k, v in d.items():
        if k in ("x", "f0_bins"):
            out |= pack(k, v, v[:, 0], axis=1)
        elif k in ("overlap", "source_snr", "snr", "log_likelihood") and v.dtype == np.float64:
            out[k] = v.astype(np.float32)
        else:
            out[k] = v
    return out


def pack_eval_dir(eval_dir: Path) -> list[Path]:
    """Pack an eval dir's draws.npz and scorecard.npz in place (a packed file is left as it is)."""
    done = []
    for name, packer in (
        ("draws.npz", pack_draws),
        ("scorecard.npz", pack_validation),
    ):
        path = eval_dir / name
        if not path.exists():
            continue
        with np.load(path) as d:
            if any(k.endswith(".half") for k in d.files):
                continue
            raw = {k: d[k] for k in d.files}
        np.savez(path.with_suffix(".tmp.npz"), **packer(raw))
        path.with_suffix(".tmp.npz").replace(path)
        done.append(path)
    return done


if __name__ == "__main__":
    for target in map(Path, sys.argv[1:]):
        for path in pack_eval_dir(target):
            print(path, flush=True)
