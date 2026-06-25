import os
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import blackjax
import corner

from src import lisa, networks
from wdm_transform.transforms import from_freq_to_wdm

print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")


# problem
SEED = 0
N_SOURCES = 2
T_OBS = lisa.MONTH_s

# model
HIDDEN_DIM = 512
NUM_BLOCKS = 4
NUM_HEADS = 8

# training
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
EPOCH_TIME_BUDGET_s = 5 * 60  # 5 minutes per epoch
EPOCHS = (48 * 60 * 60) // EPOCH_TIME_BUDGET_s  # 48h total
CHECKPOINT_PATH = "checkpoint.eqx"

# evaluation
N_SAMPLES = int(os.environ.get("N_SAMPLES", 1024))

# --- Build the flow and load the checkpoint ---------------------------------
key = jr.key(SEED)
key, key_mock = jr.split(key)
x, dx, t, y = lisa.get_train_batch(
    key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
)
flow = networks.MMDiT(
    x_dim=x.shape[-1],
    y_dim=y.shape[-1],
    hidden_dim=HIDDEN_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS,
    key=key_mock,
)
flow = eqx.tree_deserialise_leaves(CHECKPOINT_PATH, flow)


# --- Build one self-consistent injection ------------------------------------
# We generate a ground-truth set of source parameters, the corresponding
# frequency-domain datastream (signal + noise) used by the MCMC likelihood,
# and the WDM-domain conditioning ``y`` fed to the flow. This way the flow and
# the MCMC are looking at the *same* observation.
key, key_truth, key_noise = jr.split(key, 3)

true_params = lisa.prior_inverse_cdf(jr.uniform(key_truth, shape=(N_SOURCES, 8)))
signal = lisa.clean_signal(true_params, t_obs=T_OBS)
noise = lisa.sample_noise(key_noise, t_obs=T_OBS)
datastream = signal + noise  # (F, 3) frequency domain
freqs = jnp.fft.rfftfreq(int(T_OBS / lisa.SAMPLING_STEP_s), lisa.SAMPLING_STEP_s)
freqs = freqs[: datastream.shape[0]]

# WDM-domain conditioning for the flow (same processing as get_train_batch).
y = from_freq_to_wdm(
    datastream.T,
    nt=32,
    nf=datastream.shape[0] // 32,
    a=1.0 / 3.0,
    d=1.0,
    dt=lisa.SAMPLING_STEP_s,
    backend="jax",
)
y = rearrange(y, "c t f -> t (f c)")


# --- Flow sampling ----------------------------------------------------------
# The flow samples in the *uniformed* (unit-cube) parameter space: it is
# trained on x1 ~ Uniform(0, 1), which get_train_batch maps to physical
# parameters via prior_inverse_cdf (see src/lisa.py). So we push uniform noise
# through the flow, then map the result to physical parameters to compare it
# against the MCMC posterior (which lives in physical space).
#
# This is the expensive part: each flow.push is a multi-step RK4 ODE solve
# (4 network evals per step), so cost ~ N_FLOW * FLOW_ODE_STEPS. To make it
# tractable on CPU we (1) use very few ODE steps and (2) process the samples in
# chunks through a single jitted+vmapped kernel (compiled once, reused) so we
# get a progress bar instead of one opaque multi-minute call. Accuracy is
# intentionally traded for speed.
@eqx.filter_jit
def flow_sample(x):
    return jax.vmap(lambda xi: flow.push(xi, y))(x)


key, key_x = jr.split(key)
x0 = jr.uniform(key_x, shape=(N_SAMPLES, N_SOURCES, 8))
x = flow_sample(x0)
print("flow samples shape:", x.block_until_ready().shape)

# Map the flow's uniform-space samples to physical parameters.
flow_physical = lisa.prior_inverse_cdf(x)
flow_samples = rearrange(flow_physical, "n s p -> n (s p)")


# --- MCMC sampling ----------------------------------------------------------
def loglik(pars):
    return 1.0
    pars_model = pars.reshape(N_SOURCES, 8)
    model = lisa.clean_signal(pars_model, t_obs=T_OBS)
    res = datastream - model
    toret = 0.0
    for i, ch in enumerate("AET"):
        PSD = lisa.noise_psd(ch)
        toret += jnp.sum(jnp.abs(res[1:, i]) ** 2 / PSD(freqs[1:]))
    return -0.5 * toret


# Initialise the walkers around the injected (true) parameters.
params = true_params
ndim = params.flatten().shape[0]
params_MCMC = params.flatten()

# Spread independent chains across CPU devices so all cores are used.
n_devices = jax.local_device_count()
walkers_per_device = max(1, round(4 * ndim / n_devices))
nwalkers = n_devices * walkers_per_device

p0 = params_MCMC * (1 + 1e-4 * jr.normal(jr.key(SEED + 2), shape=(nwalkers, ndim)))
p0 = p0.reshape(n_devices, walkers_per_device, ndim)

# Proposal scale: 0.1% of each parameter value (gradient-free, no AD needed)
sigma = jnp.abs(params_MCMC) * 1e-3
rwm = blackjax.normal_random_walk(loglik, sigma)


@partial(jax.pmap, axis_name="device")
def run_sampling(key, positions):
    # positions: (walkers_per_device, ndim) on this device.
    init_states = jax.vmap(rwm.init)(positions)

    def one_step(states, key):
        keys = jr.split(key, walkers_per_device)
        new_states, infos = jax.vmap(rwm.step)(keys, states)
        return new_states, (new_states.position, infos.acceptance_rate)

    _, (chain, acc) = jax.lax.scan(one_step, init_states, jr.split(key, N_SAMPLES))
    return chain, acc  # (N_SAMPLES, wpd, ndim), (N_SAMPLES, wpd)


print(
    f"Compiling and running {N_SAMPLES} samples x {nwalkers} walkers "
    f"on {n_devices} devices..."
)
keys = jr.split(jr.key(SEED + 3), n_devices)
chain, acc = run_sampling(keys, p0)
chain.block_until_ready()

# (n_devices, N_SAMPLES, wpd, ndim) -> (N_SAMPLES, nwalkers, ndim)
chain = rearrange(chain, "d n w p -> n (d w) p")
acc = rearrange(acc, "d n w -> n (d w)")
print(f"Done. Chain shape: {chain.shape}")
print(f"Mean acceptance rate: {acc.mean():.2f}")


# --- Corner plot: MCMC vs flow posterior ------------------------------------
# Discard the first half of each MCMC chain as burn-in, then flatten walkers.
burn_in = N_SAMPLES // 2
mcmc_samples = np.asarray(rearrange(chain[burn_in:], "n w p -> (n w) p"))
flow_samples = np.asarray(flow_samples)

param_names = ["f0", "fdot", "A", "ra", "dec", "psi", "iota", "phi0"]
labels = [f"{name}$_{s}$" for s in range(N_SOURCES) for name in param_names]
truths = np.asarray(true_params).flatten()


def combined_range(*datasets, frac=0.999):
    """Per-dimension plot range covering all datasets (robust to outliers)."""
    data = np.concatenate(datasets, axis=0)
    lo = np.quantile(data, (1 - frac) / 2, axis=0)
    hi = np.quantile(data, 1 - (1 - frac) / 2, axis=0)
    pad = 0.05 * (hi - lo)
    pad = np.where(pad == 0, np.maximum(np.abs(lo), 1e-30) * 1e-3, pad)
    return list(zip(lo - pad, hi + pad))


plot_range = combined_range(mcmc_samples, flow_samples)

MCMC_COLOR, FLOW_COLOR = "C0", "C1"
fig = corner.corner(
    mcmc_samples,
    labels=labels,
    truths=truths,
    truth_color="black",
    color=MCMC_COLOR,
    range=plot_range,
    show_titles=True,
    title_fmt=".2e",
    hist_kwargs={"density": True},
)
corner.corner(
    flow_samples,
    fig=fig,
    color=FLOW_COLOR,
    range=plot_range,
    hist_kwargs={"density": True},
)

fig.legend(
    handles=[
        mlines.Line2D([], [], color=MCMC_COLOR, label="MCMC"),
        mlines.Line2D([], [], color=FLOW_COLOR, label="Flow"),
        mlines.Line2D([], [], color="black", label="Injected truth"),
    ],
    loc="upper right",
    fontsize=14,
    frameon=False,
)
fig.suptitle("Posterior distribution: MCMC vs flow")
fig.savefig("corner.png", dpi=150, bbox_inches="tight")
print("Saved corner plot to corner.png")
plt.show()
