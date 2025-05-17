from typing import NamedTuple
from jaxtyping import Float, Int, Array
import jax.numpy as jnp


class SourceParameters(NamedTuple):
    amplitude: float
    frequency: float
    phase: float


class PriorParameters(NamedTuple):
    a_min: float
    a_max: float
    f_min: float
    f_max: float


class SignalParameters(NamedTuple):
    time: Float[Array, "num_samples"]
    noise_std: float


def sample(rng_key, *, prior_params: PriorParameters) -> Parameters:
    amplitude = jax.random.uniform(rng_key, (num_samples,), minval=0.1, maxval=10.0)
    frequency = jax.random.uniform(rng_key, (num_samples,), minval=0.1, maxval=10.0)
    phase = jax.random.uniform(rng_key, (num_samples,), minval=0.0, maxval=2 * jnp.pi)
    return Parameters(amplitude=amplitude, frequency=frequency, phase=phase)


def signal(params: Parameters, *, time, noise_std: float) -> Float[Array, "num_samples"]:
    return params.amplitude * jnp.sin(
        2 * jnp.pi * params.frequency * time + params.phase
    )


def prior_loglikelihood(params: Parameters, prior_params: PriorParameters) -> float:
    return (
        -0.5 * jnp.sum((params.amplitude - 1) ** 2)
        - 0.5 * jnp.sum((params.frequency - 1) ** 2)
        - 0.5 * jnp.sum((params.phase - 0) ** 2)
    )


def loglikelihood(params: Parameters, signal, time, ):
    model = signal(params, time=time)
    return -jnp.sum((data - model) ** 2)
