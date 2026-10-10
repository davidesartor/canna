"""Flow against MCMC on XS eval windows: the numbers and pages of the paper's MCMC appendix.

Reads the cold chains of lisa_checks/mcmc_xs.py (outputs/mcmc/window<j>_s<seed>.npz), the
flow's eval draws (outputs/lisa-XS-late-768-cool/eval_draws.npz, eval.py's keys, so the same
noise realisation) and the window's Fisher draws. Every sample set is relabelled onto the
true sources by eval.match_sources, and its angles folded by the likelihood's exact
symmetries: psi -> psi + pi, and (psi, phi0) -> (psi + pi/2, phi0 + pi). Per source and
parameter it prints
- the width ratio, flow / MCMC, of the robust standard deviation (1.4826 MAD);
- the 1-D Wasserstein distance between the two, in units of the MCMC standard deviation;
- whether the MCMC median lies inside the flow's central 68% and 95%;
and the same between the two MCMC seeds, which bounds what the MCMC itself can resolve.
Writes outputs/mcmc/compare_window<j>.npz and one corner page per source.

Run from the repo root with the project venv:  .venv/bin/python lisa_checks/compare_mcmc.py
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).parent))
from mcmc_xs import setup  # noqa: E402

from canna.lisa.eval import match_sources  # noqa: E402

COLUMNS = ["f0 [bins]", "log Mc", "log A", "lon", "sin lat", "psi", "sin iota", "phi0"]
FLOW_DRAWS = Path("outputs/lisa-XS-late-768-cool/eval_draws.npz")
FISHER = Path("outputs/lisa-XS/eval-tools/fisher_eval_injections.npz")


def fold(p, truth, t_obs):
    """(N, S, 8) physical -> (N, S, 8) comparison coordinates, folded around the truth."""
    p = np.array(p, np.float64)
    wrap = lambda a, c: c + (a - c + np.pi) % (2 * np.pi) - np.pi
    k = np.round((p[..., 5] - truth[:, 5]) / (np.pi / 2))
    psi, phi0 = p[..., 5] - k * np.pi / 2, p[..., 7] - k * np.pi
    return np.stack(
        [
            p[..., 0] * t_obs,
            np.log(p[..., 1]),
            np.log(np.abs(p[..., 2])),
            wrap(p[..., 3], truth[:, 3]),
            np.sin(p[..., 4]),
            psi,
            np.sin(p[..., 6]),
            wrap(phi0, truth[:, 7]),
        ],
        axis=-1,
    )


def width(a):
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=0)), axis=0)


def compare(a, ref):
    """a, ref: (N, 8) for one source -> width ratio, W1 / sd(ref), ref median in a's 68 / 95."""
    ratio = width(a) / width(ref)
    w1 = np.array([wasserstein_distance(a[:, c], ref[:, c]) for c in range(a.shape[1])])
    w1 /= width(ref)
    q = np.percentile(a, [2.5, 16, 84, 97.5], axis=0)
    med = np.median(ref, axis=0)
    return ratio, w1, (q[1] <= med) & (med <= q[2]), (q[0] <= med) & (med <= q[3])


def check_symmetry(s):
    """The folding is only right if the likelihood is invariant under it: check at the truth."""
    p = jnp.asarray(s["truth"])
    base = float(s["problem"].log_likelihood(p, s["obs"], s["f"]))
    for dpsi, dphi in [(np.pi, 0.0), (np.pi / 2, np.pi)]:
        q = p.at[:, 5].add(dpsi).at[:, 7].add(dphi)
        assert abs(float(s["problem"].log_likelihood(q, s["obs"], s["f"])) - base) < 1e-6 * abs(base)


def corner_source(path, flow, mcmc, fisher, truth, title):
    import corner
    import matplotlib.pyplot as plt

    lo = np.minimum(np.percentile(flow, 0.5, 0), np.percentile(mcmc, 0.5, 0))
    hi = np.maximum(np.percentile(flow, 99.5, 0), np.percentile(mcmc, 99.5, 0))
    pad = 0.1 * (hi - lo)
    ranges = list(zip(lo - pad, hi + pad))
    kw = dict(bins=40, range=ranges, plot_datapoints=False, plot_density=False,
              levels=(0.39, 0.86), smooth=1.0, labels=COLUMNS)
    fig = corner.corner(mcmc, color="black", **kw)
    corner.corner(flow, fig=fig, color="tab:orange", **kw)
    corner.corner(fisher, fig=fig, color="tab:blue", **kw)
    corner.overplot_lines(fig, truth, color="tab:red", lw=0.8)
    fig.suptitle(title + "\nMCMC (black), flow (orange), Fisher (blue), truth (red)", fontsize=10)
    for ax in fig.axes:
        ax.set_rasterized(True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=[9, 5])
    parser.add_argument("--out", type=Path, default=Path("outputs/mcmc"))
    parser.add_argument("--n_mcmc", type=int, default=40000, help="MCMC draws compared")
    args = parser.parse_args()

    flow_all = np.load(FLOW_DRAWS)["draws"]
    fisher_all = np.load(FISHER)["fisher"]
    for j in args.windows:
        s = setup(j)
        check_symmetry(s)
        problem, truth, f, t_obs = s["problem"], s["truth"], s["f"], s["problem"].t_obs
        runs = sorted(args.out.glob(f"window{j}_s*.npz"))
        assert runs, f"no MCMC runs for window {j}"
        rng = np.random.default_rng(0)
        mcmc = []
        for r in runs:
            z = np.load(r)
            print(f"{r.name}: {z['seconds'] / 60:.1f} min, R-hat max {z['rhat'].max():.4f}, "
                  f"tau max {z['tau'].max():.0f} iterations, ESS min {z['ess'].min():.0f}")
            pick = rng.choice(len(z["physical"]), min(args.n_mcmc, len(z["physical"])), replace=False)
            draws = z["physical"][pick].reshape(-1, *truth.shape)
            mcmc.append(fold(match_sources(problem, draws, truth, f), truth, t_obs))
        flow = fold(match_sources(problem, flow_all[j], truth, f), truth, t_obs)
        fisher = fold(fisher_all[j], truth, t_obs)
        ref = np.concatenate(mcmc)
        tru = fold(truth[None], truth, t_obs)[0]

        print(f"\nwindow {j}: window SNR {s['window_snr']:.1f}; {len(flow)} flow draws, "
              f"{len(ref)} MCMC draws over {len(runs)} runs")
        out = {}
        for src in np.argsort(-s["snr"]):
            r_flow = compare(flow[:, src], ref[:, src])
            r_fish = compare(fisher[:, src], ref[:, src])
            r_seed = compare(mcmc[1][:, src], mcmc[0][:, src]) if len(mcmc) > 1 else None
            print(f"\n source {src}, SNR {s['snr'][src]:.1f}")
            print(f"  {'':22s}" + "".join(f"{c:>11s}" for c in COLUMNS))
            rows = [("MCMC width", width(ref[:, src])), ("flow / MCMC width", r_flow[0]),
                    ("flow W1 / MCMC sd", r_flow[1]), ("Fisher / MCMC width", r_fish[0])]
            if r_seed is not None:
                rows += [("seed 1 / seed 0 width", r_seed[0]), ("seed W1 / sd", r_seed[1])]
            for name, v in rows:
                print(f"  {name:22s}" + "".join(f"{x:11.4g}" for x in v))
            print(f"  {'MCMC median in flow 68':22s}" + "".join(f"{str(bool(x)):>11s}" for x in r_flow[2]))
            print(f"  {'MCMC median in flow 95':22s}" + "".join(f"{str(bool(x)):>11s}" for x in r_flow[3]))
            out[f"s{src}"] = np.stack([width(ref[:, src]), *r_flow[:2], r_fish[0], *(r_seed[:2] if r_seed else [np.nan * r_flow[0]] * 2)])
            out[f"s{src}_in68"], out[f"s{src}_in95"] = r_flow[2], r_flow[3]
            corner_source(
                args.out / f"corner_window{j}_s{src}.pdf",
                flow[:, src], ref[:, src], fisher[:, src], tru[src],
                f"window {j} (window SNR {s['window_snr']:.0f}), source {src}, SNR {s['snr'][src]:.1f}",
            )
        np.savez(args.out / f"compare_window{j}.npz", snr=s["snr"], columns=COLUMNS,
                 rows=["mcmc width", "flow/mcmc width", "flow W1/sd", "fisher/mcmc width",
                       "seed1/seed0 width", "seed W1/sd"], **out)
        print(f"\nsaved {args.out / f'compare_window{j}.npz'} and corner_window{j}_s*.pdf")


if __name__ == "__main__":
    main()
