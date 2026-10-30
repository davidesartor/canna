"""Numpy-only helpers shared by the probe and the one-off legacy converter: page-ready rounding and a joint window's record."""

import numpy as np

# draws per set and source the page keeps from each joint window
PAGE_DRAWS = 250


def sig(a, digits: int = 4):
    """Floats (any shape) -> nested lists rounded to `digits` significant figures, NaN -> None."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return None if not np.isfinite(a) else float(f"{a:.{digits}g}")
    return [sig(v, digits) for v in a]


def circular_width(phases: np.ndarray) -> np.ndarray:
    """Phases [N, K] -> sqrt(-2 log R) per column."""
    resultant = np.abs(np.mean(np.exp(1j * phases.astype(np.float64)), 0))
    return np.sqrt(-2 * np.log(np.clip(resultant, 1e-12, 1.0)))


def window_record(z, window: int, epoch: int) -> dict:
    """A joint window's arrays (compare_joint's npz layout) -> per source the comparison rows and some draws."""
    rng = np.random.default_rng(0)
    rows = [str(r) for r in z["rows"]]
    sources = []
    for s in np.argsort(-z["snr"]):
        sets = {
            name: np.concatenate(
                [z[f"s{s}_{name}_offsets"], z[f"s{s}_{name}_phases"]], 1
            )
            for name in ("mcmc", "flow", "fisher")
        }
        mcmc_w, flow_w, fisher_w = (
            circular_width(sets[n][:, 8:]) for n in ("mcmc", "flow", "fisher")
        )
        # phi± rows are appended to the eight linear columns
        extra = {
            "mcmc width": mcmc_w,
            "flow/mcmc width": flow_w / mcmc_w,
            "fisher/mcmc width": fisher_w / mcmc_w,
        }
        table = {
            name: sig(
                np.concatenate([z[f"s{s}"][i], extra.get(name, [np.nan, np.nan])])
            )
            for i, name in enumerate(rows)
        }
        draws = {
            name: sig(v[rng.choice(len(v), min(PAGE_DRAWS, len(v)), replace=False)])
            for name, v in sets.items()
        }
        sources.append(
            dict(
                s=int(s),
                snr=sig(z["snr"][s]),
                rows=table,
                in68=[bool(v) for v in z[f"s{s}_in68"]],
                in95=[bool(v) for v in z[f"s{s}_in95"]],
                far=sig(z[f"s{s}_far"]),
                draws=draws,
            )
        )
    return dict(
        window=window,
        window_snr=sig(z["window_snr"]),
        epoch=epoch,
        sources=sources,
    )
