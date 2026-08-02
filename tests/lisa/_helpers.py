"""Every LisaGB method needs the f that produced its window, so the tests derive one."""

import jax.numpy as jnp


def window(problem) -> jnp.ndarray:
    """Centre frequency of the bottom window on this problem's own index grid."""
    lo, _ = problem.window_index_range
    bins = lo * problem.wdm_times + problem.window_bins / 2
    return jnp.asarray(bins / problem.t_obs)
