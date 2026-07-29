"""fdot is not sampled: it is the radiation reaction the chirp mass and f0 imply."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.problems import LisaGB
from canna.problems.lisa import chirp_mass_from_fdot, fdot_from_chirp_mass

jax.config.update("jax_enable_x64", True)

TWO_YEARS = 2 * 365.25 * 24 * 60 * 60


def test_fdot_and_chirp_mass_round_trip():
    mc = jnp.array([0.15, 0.25, 0.5, 1.2])
    f0 = jnp.array([1e-4, 1e-3, 3e-3, 1e-2])
    assert jnp.allclose(chirp_mass_from_fdot(fdot_from_chirp_mass(mc, f0), f0), mc)


def test_fdot_follows_the_known_powers_of_chirp_mass_and_frequency():
    base = fdot_from_chirp_mass(0.25, 1e-3)
    assert jnp.allclose(fdot_from_chirp_mass(0.5, 1e-3), base * 2 ** (5 / 3))
    assert jnp.allclose(fdot_from_chirp_mass(0.25, 2e-3), base * 2 ** (11 / 3))


def test_a_verification_binary_lands_on_its_catalogue_chirp():
    # HM Cnc (f0 = 6.22 mHz, Mc ~ 0.32 Msun) chirps at a few 1e-16 Hz/s
    fdot = fdot_from_chirp_mass(0.32, 6.22e-3)
    assert 1e-16 < float(fdot) < 1e-15


def test_the_chirp_mass_column_drifts_the_waveform_by_the_predicted_amount():
    # the band's power-weighted centre sits at f0 + fdot * t_obs / 2, so the shift
    # between two chirp masses pins down the fdot clean_signal actually used
    problem = LisaGB(n_sources=1, t_obs=TWO_YEARS, f0_range=(9.5e-3, 1.0e-2))
    p = jnp.array([[9.9e-3, 0.2, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])

    def centre(chirp_mass: float) -> float:
        power = jnp.abs(problem.clean_signal(p.at[0, 1].set(chirp_mass))[:, 0]) ** 2
        return float(jnp.sum(power * jnp.arange(power.size)) / jnp.sum(power))

    light, heavy = 0.2, 1.2
    shift = centre(heavy) - centre(light)
    fdots = fdot_from_chirp_mass(jnp.array([light, heavy]), p[0, 0])
    predicted = float((fdots[1] - fdots[0]) * problem.t_obs**2 / 2)
    assert abs(shift - predicted) < 0.05 * predicted


def test_the_fdot_range_the_prior_implies_spans_the_whole_band():
    problem = LisaGB(n_sources=1)
    low, high = problem.fdot_range
    mc_lo, mc_hi = problem.chirp_mass_range
    assert low == fdot_from_chirp_mass(mc_lo, problem.f0_range[0])
    assert high == fdot_from_chirp_mass(mc_hi, problem.f0_range[1])

    # every draw sits inside it, unlike under the old fixed [1e-18, 1e-15] box: at the
    # bottom of the band radiation reaction is many orders of magnitude weaker
    p = jax.vmap(problem.sample_physical)(jr.split(jr.key(1), 256)).reshape(-1, 8)
    fdot = fdot_from_chirp_mass(p[:, 1], p[:, 0])
    assert jnp.all((fdot >= low) & (fdot <= high))
    assert float(jnp.min(fdot)) < 1e-18
