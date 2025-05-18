from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm

from networks import GWMMDiT


def sample_params_and_signal(
    rng: Key,
    N: int,  # Number of sources
    T: int,  # Number of time steps
) -> tuple[Float[Array, "N 3"], Float[Array, "T 2"]]:
    rng_1, rng_2, rng_3, rng_4 = jr.split(rng, 4)

    amplitude = 10 ** jr.uniform(rng_1, (N), minval=0.0, maxval=1.0)
    frequency = 10 ** jr.uniform(rng_2, (N), minval=0.0, maxval=1.0)
    phase = jr.uniform(rng_3, (N), minval=0.0, maxval=2 * jnp.pi)
    params = jnp.stack([amplitude, frequency, phase], axis=-1)

    times = jnp.linspace(0, 1, T)[..., None]
    angles = 2 * jnp.pi * frequency * times + phase
    h_plus = (amplitude * jnp.sin(angles)).sum(-1)
    h_cross = (amplitude * jnp.cos(angles)).sum(-1)
    h = jnp.stack([h_plus, h_cross], axis=-1)
    
    datastream = h + 1.0 * jr.normal(rng_4, h.shape)
    return params, datastream, times


if __name__ == "__main__":
    BATCH_SIZE = 128
    N = 2
    T = 128
    
    @jax.jit
    @jax.vmap
    def sample_batch(rng):
        rng_1, rng_2, rng_3 = jr.split(rng, 3)
        x1, c, _ = sample_params_and_signal(rng_1, N=N, T=T)
        x0 = jr.normal(rng_3, x1.shape)
        t = jax.nn.sigmoid(jr.normal(rng_2))
        return x1, x0, t, c

    model = GWMMDiT(dim=64*4, num_heads=4, num_blocks=8)
    optimizer = optax.adam(learning_rate=1e-4)

    x1, x0, t, c = sample_batch(jr.split(jr.key(0), BATCH_SIZE))
    params = model.init(jr.key(0), x1, t, c)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, batch):
        @jax.value_and_grad
        def loss(params, batch):
            x1, x0, t, c = batch
            xt = x1 * t[..., None, None] + x0 * (1 - t[..., None, None])
            dx = model.apply(params, xt, t, c)
            return optax.l2_loss(dx, x1 - x0).mean()

        loss, grads = loss(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in (pbar := tqdm(range(10000))):
        batch = sample_batch(jr.split(jr.key(0), BATCH_SIZE))
        params, opt_state, loss = train_step(params, opt_state, batch)
        pbar.set_postfix(loss=loss)
