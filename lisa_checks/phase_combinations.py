"""psi and phi0 against the MCMC in the combinations the data measure (F52).

The signal is linear in A (1 + sin iota)^2 e^{-i(phi0 - 2 psi)} and A (1 - sin iota)^2
e^{-i(phi0 + 2 psi)} (F49): a face-on source measures phi- = phi0 - 2 psi, a face-off one
phi+ = phi0 + 2 psi, an edge-on one both. Both are invariant under the psi/phi0 symmetries, so
they need no folding. For each MCMC'd source this prints the circular widths of phi0, psi (mod
pi/2, i.e. 4 psi / 4), phi+ and phi- of the flow and of the MCMC, and their ratio. A flow with the
right marginals but no psi-phi0 correlation is as wide as the MCMC in phi0 and psi and much
wider in the combination the MCMC pins.

    .venv/bin/python lisa_checks/phase_combinations.py <config> <run folder> [more run folders]
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import jax
import numpy as np

from canna.lisa import mcmc as M
from canna.lisa.eval import eval_injections, match_sources


def circ_width(a, k=1):
    """Circular standard deviation of k a, divided by k (radians of a)."""
    r = np.abs(np.mean(np.exp(1j * k * a)))
    return np.sqrt(-2 * np.log(max(r, 1e-12))) / k


config, runs = sys.argv[1], sys.argv[2:]
problem, seed = M.load_problem(config)
latents, windows, *_ = eval_injections(problem, seed)
snr = np.asarray(jax.vmap(problem.source_snrs)(latents, windows))
chains = M.load_chains(problem, seed, n=10**9)
quantities = {
    "phi0": lambda p: (p[:, 7], 2),  # phi0 is defined mod pi once psi is free: width of 2 phi0 / 2
    "psi": lambda p: (p[:, 5], 4),  # psi mod pi/2
    "phi+": lambda p: (p[:, 7] + 2 * p[:, 5], 1),
    "phi-": lambda p: (p[:, 7] - 2 * p[:, 5], 1),
}
for run in runs:
    d = np.load(f"outputs/{run}/eval_draws.npz")
    print(f"\n{run}: circular widths [rad], flow / MCMC (ratio)")
    print(f"{'src':5s} {'SNR':>7s} {'sin i':>6s} " + "".join(f"{q:>22s}" for q in quantities))
    for (j, s), mc in sorted(chains.items(), key=lambda kv: -snr[kv[0]]):
        fl = match_sources(problem, d["draws"][j], d["truth"][j], windows[j])[:, s]
        cells = []
        for q, fn in quantities.items():
            (a, k), (b, _) = fn(fl), fn(mc)
            wf, wm = circ_width(a, k), circ_width(b, k)
            cells.append(f"{wf:6.3f}/{wm:6.3f} ({wf / wm:5.1f})")
        print(f"{j}_{s:<3d} {snr[j, s]:7.1f} {np.sin(d['truth'][j, s, 6]):6.2f} " + "".join(f"{c:>22s}" for c in cells))
