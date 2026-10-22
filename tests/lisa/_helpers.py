"""Every LisaGB function needs the f that produced its window, so the tests derive one."""

import jax.numpy as jnp

from canna.lisa.problem import WINDOW_STRIDE


def window(problem) -> jnp.ndarray:
    """Centre frequency of the bottom window on this problem's own index grid."""
    lo, _ = problem.window_index_range
    bins = lo * WINDOW_STRIDE + problem.window_bins / 2
    return jnp.asarray(bins / problem.t_obs)
