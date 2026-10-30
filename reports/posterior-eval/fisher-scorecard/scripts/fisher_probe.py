"""Read each setting's scorecard.npz (and draws.npz where present) into data/<setting>.json: f0 width, pulls, coverage, flow-over-Fisher widths."""

import argparse
import json
from pathlib import Path

import numpy as np

REPORT = Path(__file__).parents[1]
ROOT = Path(__file__).parents[3].parent
MOCKS = REPORT.parent / "mock-flows" / "data"
SNR_BANDS = [(0, 15), (15, 40), (40, 100), (100, np.inf)]
PARAMS = ["f0", "fdot", "A", "lon", "lat", "psi", "iota", "phi0"]
PERIOD = {"lon": 2 * np.pi, "psi": 2 * np.pi, "phi0": 2 * np.pi}
FOUND_BINS = 1.0


def setting_of(path: Path) -> tuple[str, str, dict]:
    """Setting key, group and metadata from an eval dir (a real run's or a mock's)."""
    path = path.resolve()
    if MOCKS in path.parents:
        band, mock = path.relative_to(MOCKS).parts
        return f"mock-{band}-{mock}", "mock", dict(band=band, mock=mock)
    run, tag = path.parent.parent.name, path.name
    return f"{run}-{tag}", "run", dict(run=run, tag=tag)


def robust_width(x: np.ndarray, axis: int) -> np.ndarray:
    """1.4826 x MAD along an axis."""
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def wrapped(x: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Draws [N, S, 8] minus truth [S, 8] with the angle coordinates wrapped to (-period/2, period/2]."""
    out = x - truth[None]
    for k, name in enumerate(PARAMS):
        if name in PERIOD:
            out[..., k] = (out[..., k] + PERIOD[name] / 2) % PERIOD[name] - PERIOD[name] / 2
    return out


def band_table(sc: dict) -> list[dict]:
    """Per SNR band and for eval, random and all rows: counts, found fraction, width over ideal, offset over width, coverage, mean rank."""
    rows = []
    for subset, mask in [
        ("all", np.ones_like(sc["is_eval"])),
        ("eval", sc["is_eval"]),
        ("random", ~sc["is_eval"]),
    ]:
        for lo, hi in SNR_BANDS:
            m = mask & (sc["snr"] >= lo) & (sc["snr"] < hi)
            if not m.any():
                continue
            ok = m & (sc["width"] > 0)
            rows.append(
                dict(
                    subset=subset,
                    lo=lo,
                    hi=None if np.isinf(hi) else hi,
                    n=int(m.sum()),
                    found=float((m & (sc["width"] < FOUND_BINS)).sum() / m.sum()),
                    ratio=float(np.median(sc["width"][ok] / sc["ideal"][ok])),
                    pull=float(np.median(np.abs(sc["offset"][ok]) / sc["width"][ok])),
                    in68=float(sc["in68"][m].mean()),
                    in95=float(sc["in95"][m].mean()),
                    rank=float(sc["rank"][m].mean()),
                )
            )
    return rows


def fisher_part(draws_path: Path) -> dict:
    """Per injection, source and parameter: flow width, Fisher width and the source SNR, from draws.npz."""
    from canna.lisa.samples import load as load_samples

    d = load_samples(draws_path)
    flow_width = np.stack([robust_width(wrapped(m, t), 0) for m, t in zip(d["matched"], d["truth"])])
    fisher_width = np.stack([robust_width(wrapped(m, t), 0) for m, t in zip(d["fisher"], d["truth"])])
    ratio = flow_width / fisher_width
    snr = d["source_snr"]
    out = dict(
        snr=snr.round(3).tolist(),
        ratio=np.round(ratio, 4).tolist(),
        flow_width=[[[float(f"{v:.4g}") for v in r] for r in a] for a in flow_width],
        fisher_width=[[[float(f"{v:.4g}") for v in r] for r in a] for a in fisher_width],
    )
    loud = snr >= 100
    out["median_ratio"] = {
        name: dict(
            low=float(np.median(ratio[..., k][~loud])) if (~loud).any() else None,
            loud=float(np.median(ratio[..., k][loud])) if loud.any() else None,
            all=float(np.median(ratio[..., k])),
        )
        for k, name in enumerate(PARAMS)
    }
    return out


def probe(eval_dir: Path) -> None:
    key, group, meta = setting_of(eval_dir)
    with np.load(eval_dir / "scorecard.npz") as raw:
        sc = {k.removeprefix("card_"): raw[k] for k in raw.files if k.startswith("card_")}
        sc |= {k: raw[k] for k in ("epoch", "ode_steps", "n_random", "t_obs")}
    record = dict(
        setting=key,
        group=group,
        **meta,
        epoch=int(sc["epoch"]),
        ode_steps=int(sc["ode_steps"]),
        n_random=int(sc["n_random"]),
        n_eval=int(sc["is_eval"].sum()),
        n_rows=len(sc["snr"]),
        source=str(eval_dir.relative_to(ROOT)) if ROOT in eval_dir.resolve().parents else str(eval_dir),
        rows={
            k: [round(float(v), 5) if sc[k].dtype.kind == "f" else int(v) for v in sc[k]]
            for k in ["snr", "width", "ideal", "offset", "rank", "in68", "in95", "is_eval", "injection"]
        },
        bands=band_table(sc),
    )
    if (eval_dir / "draws.npz").exists():
        record["fisher"] = fisher_part(eval_dir / "draws.npz")
    path = REPORT / "data" / f"{key}.json"
    path.write_text(json.dumps(record, separators=(",", ":")))
    print(f"{key}: {record['n_rows']} rows, fisher part {'fisher' in record}, -> {path}")
    for r in record["bands"]:
        if r["subset"] == "all":
            hi = "inf" if r["hi"] is None else r["hi"]
            print(
                f"  snr {r['lo']}-{hi}: n={r['n']} found={r['found']:.2f} width/ideal={r['ratio']:.3g}"
                f" |pull|={r['pull']:.2f} in68={r['in68']:.2f} in95={r['in95']:.2f}"
            )
    if "fisher" in record:
        print("  flow/Fisher median (low|loud):", {k: (round(v["low"], 2) if v["low"] else None, round(v["loud"], 2) if v["loud"] else None) for k, v in record["fisher"]["median_ratio"].items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dirs", nargs="*", type=Path, help="eval dirs; default: every mock")
    args = parser.parse_args()
    dirs = args.eval_dirs or sorted(MOCKS.glob("*/*"))
    (REPORT / "data").mkdir(exist_ok=True)
    for eval_dir in dirs:
        probe(eval_dir)


if __name__ == "__main__":
    main()
