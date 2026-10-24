"""Size and power of canna.lisa.validate's tests: an exact null and known perturbations of the draws.

toy: an 8-dimensional linear-Gaussian model per slot, exact posterior draws against the true θ.
lisa: a run's validation.npz with one flow draw standing in for the truth, so the draws are
exchangeable with it under the flow itself; also records the tests on the real truth.
From the repo root, as power.sbatch does:
    uv run --frozen python reports/diagnostics/posterior-validation/test_power.py toy
    LISA_RUN=<run> uv run --frozen python reports/diagnostics/posterior-validation/test_power.py lisa
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from canna.lisa.validate import ALPHA, COORDS, SNR_BANDS, analyse, gamma_test

HERE = Path(__file__).parent
CHUNK = 100
TOY_INJECTIONS = 5000
TOY_DRAWS = 126
SCORES = ["energy", "joint_energy"] + [f"crps_{c}" for c in COORDS]
PERTURBATIONS = [
    "null",
    "narrow ×0.5",
    "narrow ×0.8",
    "wide ×1.25",
    "wide ×2",
    "f0 shift 0.25σ",
    "f0 shift 0.5σ",
    "z+ phase from prior",
    "sky from prior",
    "dependence shuffled",
    "10% prior draws",
    "30% prior draws",
]


def perturb(
    name: str, x: np.ndarray, prior: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """The draws x [N, M, S, 8] under one known error; prior [N, M, S, 8] are independent prior draws."""
    mean, std = x.mean(1, keepdims=True), x.std(1, keepdims=True)
    if name.startswith(("narrow", "wide")):
        return mean + float(name.split("×")[1]) * (x - mean)
    if name.startswith("f0 shift"):
        return x + np.eye(8)[0] * float(name.split()[2][:-1]) * std
    if name == "z+ phase from prior":
        angle = rng.uniform(0, 2 * np.pi, x.shape[:-1])
        z = (x[..., 4] + 1j * x[..., 5]) * np.exp(1j * angle)
        return np.concat(
            [x[..., :4], z.real[..., None], z.imag[..., None], x[..., 6:]], -1
        )
    if name == "sky from prior":
        return np.concat([x[..., :2], prior[..., 2:4], x[..., 4:]], -1)
    if name == "dependence shuffled":
        return rng.permuted(x, axis=1)
    if name.endswith("prior draws"):
        swap = rng.random(x.shape[:2]) < float(name.split("%")[0]) / 100
        return np.where(swap[..., None, None], prior, x)
    return x


def score_rows(rows: dict) -> dict:
    """Every proper score of analyse's rows, one value per injection and slot: [N, S] or [N]."""
    return {
        "energy": rows["energy"],
        "joint_energy": rows["joint_energy"],
        **{f"crps_{c}": rows["crps"][:, :, i] for i, c in enumerate(COORDS)},
    }


def rejections(rows: dict, reference: dict, n: int) -> dict:
    """Share of CHUNK-injection chunks (and slots) where each test rejects at ALPHA."""
    k, chunks = rows["draws"] + 1, [
        slice(i, i + CHUNK) for i in range(0, n - CHUNK + 1, CHUNK)
    ]
    out = {}
    for name, r in rows["rank"].items():
        out[f"slot {name}"] = np.mean(
            [
                gamma_test(r[c, s], k)[0] < ALPHA
                for c in chunks
                for s in range(r.shape[1])
            ]
        )
    for name, r in rows["joint_rank"].items():
        out[f"joint {name}"] = np.mean([gamma_test(r[c], k)[0] < ALPHA for c in chunks])
    # a proper score detects the error when it is worse than the unperturbed draws', paired per injection
    perturbed, unperturbed = score_rows(rows), score_rows(reference)
    worse = lambda a, b: not np.array_equal(a, b) and (
        wilcoxon(a, b, alternative="greater").pvalue < ALPHA
    )
    for name in SCORES:
        a, b = perturbed[name].reshape(n, -1), unperturbed[name].reshape(n, -1)
        out[f"score {name}"] = np.mean(
            [worse(a[c, s], b[c, s]) for c in chunks for s in range(a.shape[1])]
        )
    return {key: float(v) for key, v in out.items()}


def posterior_cells(rows: dict) -> dict:
    """validate.report's cells as data: per SNR stratum and test, [p, mean rank offset, middle share] per slot and joint."""
    k = rows["draws"] + 1
    cell = lambda r: (
        None
        if len(r) < 10
        else [
            float(gamma_test(r, k)[0]),
            float(r.mean() / (k - 1) - 0.5),
            float(np.mean((r >= k / 4) & (r < 3 * k / 4))),
        ]
    )
    out = {}
    for lo, hi in [*SNR_BANDS, (0, np.inf)]:
        band = (rows["snr_hat"] >= lo) & (rows["snr_hat"] < hi)
        joint = (rows["joint_snr_hat"] >= lo) & (rows["joint_snr_hat"] < hi)
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else (f">={lo:g}" if lo else "all")
        names = list(rows["rank"]) + [
            n for n in rows["joint_rank"] if n not in rows["rank"]
        ]
        out[label] = dict(
            sources=band.sum(0).tolist(),
            injections=int(joint.sum()),
            tests={
                name: dict(
                    slot=(
                        [
                            cell(rows["rank"][name][band[:, s], s])
                            for s in range(band.shape[1])
                        ]
                        if name in rows["rank"]
                        else None
                    ),
                    joint=(
                        cell(rows["joint_rank"][name][joint])
                        if name in rows["joint_rank"]
                        else None
                    ),
                )
                for name in names
            },
        )
    return out


def power(
    x: np.ndarray, prior: np.ndarray, requantify, tarp_ref: np.ndarray, seed: int = 0
) -> dict:
    """Rejection rates of every test under every perturbation; x [N, M + 1, S, 8] has the truth in row 0."""
    rng, n = np.random.default_rng(seed), len(x)
    samples = lambda draws: dict(
        requantify(np.concat([x[:, :1], draws], 1)), tarp_ref=tarp_ref
    )
    reference = analyse(samples(x[:, 1:]), seed)
    out = {}
    for name in PERTURBATIONS:
        out[name] = rejections(
            analyse(samples(perturb(name, x[:, 1:], prior, rng)), seed), reference, n
        )
        print(
            f"{name:>22}: " + " ".join(f"{v:.2f}" for v in out[name].values()),
            flush=True,
        )
    return out


def toy() -> dict:
    """Linear-Gaussian slots y = A θ + σ ε, θ ~ N(0, I_8), σ log-uniform per source: exact Gaussian posteriors."""
    rng, n, m, s = np.random.default_rng(0), TOY_INJECTIONS, TOY_DRAWS, 4
    mixing = rng.normal(size=(s, 8, 8)) / np.sqrt(8)
    noise = np.exp(rng.uniform(np.log(0.05), np.log(1.0), (n, s)))
    truth = rng.normal(size=(n, s, 8))
    y = np.einsum("sij,nsj->nsi", mixing, truth) + noise[..., None] * rng.normal(
        size=(n, s, 8)
    )

    # posterior N(P⁻¹ Aᵀ y / σ², P⁻¹) with precision P = I + Aᵀ A / σ²
    gram = np.einsum("sji,sjk->sik", mixing, mixing)
    precision = np.eye(8) + gram / noise[..., None, None] ** 2
    cov = np.linalg.inv(precision)
    mean = np.einsum("nsij,skj,nsk->nsi", cov, mixing, y) / noise[..., None] ** 2
    draws = mean[:, None] + np.einsum(
        "nsij,nmsj->nmsi", np.linalg.cholesky(cov), rng.normal(size=(n, m, s, 8))
    )

    def requantify(theta: np.ndarray) -> dict:
        signal = np.einsum("sij,nmsj->nmsi", mixing, theta)
        whitened = noise[:, None, :, None]
        source_snr = np.linalg.norm(signal / whitened, axis=-1)
        return dict(
            x=theta,
            f0_bins=theta[..., 0],
            source_snr=source_snr,
            overlap=np.sum(y[:, None] * signal / whitened**2, -1) / source_snr,
            log_likelihood=-0.5
            * np.sum(((y[:, None] - signal) / whitened) ** 2, (-2, -1)),
            snr=np.linalg.norm(source_snr, axis=-1),
        )

    x = np.concat([truth[:, None], draws], 1)
    prior = rng.normal(size=(n, m, s, 8))
    return dict(
        setting="toy",
        injections=n,
        draws=m,
        power=power(x, prior, requantify, rng.normal(size=(n, s, 8))),
    )


def lisa() -> dict:
    """The run's newest validation.npz: the real-truth cells, and power with draw 1 standing in for the truth."""
    import jax
    import jax.random as jr
    from canna.lisa import config as run_config
    from canna.lisa.chart import flow_to_physical, sample_flow
    from canna.lisa.problem import LisaGB, sample_f
    from canna.lisa.simulator import sample_observation, sample_physical
    from canna.lisa.validate import quantities

    config = run_config.load_run()
    problem = LisaGB(**config.problem)
    path = (
        config.run_dir
        / "eval"
        / f"{max(int(p.name.split('-')[0]) for p in (config.run_dir / 'eval').iterdir())}-{config.weights}-{config.ode_steps}"
        / "validation.npz"
    )
    with np.load(path) as d:
        samples = dict(d)
    n = len(samples["f"])
    print(f"{path}: {n} injections, {samples['x'].shape[1] - 1} draws", flush=True)
    cells = posterior_cells(analyse(samples, config.seed))

    # the observations again, from validate.sample's keys
    keys = jr.split(jr.fold_in(jr.key(config.seed), 1), n)

    @jax.jit
    def observe(key):
        key_f, key_p, key_n, _ = jr.split(key, 4)
        f = sample_f(problem, key_f)
        return f, sample_observation(
            problem, key_n, sample_physical(problem, key_p, f), f
        )

    f, o = jax.lax.map(observe, keys)
    assert np.allclose(
        np.asarray(f), samples["f"]
    ), "observations do not match validation.npz"

    @jax.jit
    def one(x, o, f):
        return quantities(problem, flow_to_physical(problem, x, f), o, f)

    def requantify(x: np.ndarray) -> dict:
        rows = [jax.tree.map(np.asarray, one(x[j], o[j], f[j])) for j in range(n)]
        return {k: np.stack([r[k] for r in rows]) for k in rows[0]}

    m = samples["x"].shape[1] - 2

    @jax.jit
    def prior_draws(key, f):
        return jax.vmap(lambda k: sample_flow(problem, k, f))(jr.split(key, m))

    prior_keys = jr.split(jr.key(7), n)
    prior = np.stack([np.asarray(prior_draws(prior_keys[j], f[j])) for j in range(n)])
    check = requantify(samples["x"][:, :2])
    assert np.allclose(
        check["overlap"], samples["overlap"][:, :2], rtol=1e-4, atol=1e-6
    ), "requantified draws do not match"
    return dict(
        setting="lisa",
        run=config.run,
        epoch=int(samples["epoch"]),
        ode_steps=int(samples["ode_steps"]),
        injections=n,
        draws=m,
        posterior=cells,
        power=power(
            samples["x"][:, 1:], prior, requantify, samples["tarp_ref"], config.seed
        ),
    )


def main() -> None:
    setting = sys.argv[1]
    record = toy() if setting == "toy" else lisa()
    record |= dict(chunk=CHUNK, alpha=ALPHA, perturbations=PERTURBATIONS)
    name = "toy" if setting == "toy" else f"{record['run']}-{record['ode_steps']}"
    out = HERE / "data" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
