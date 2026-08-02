"""NoisySinusoid: superposed sinusoids in white noise, sources an unordered set."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.sinusoid import NoisySinusoid


@pytest.fixture
def problem():
    return NoisySinusoid(n_sources=3)


@pytest.fixture
def key():
    return jr.key(0)


# --- physical / point shapes and provenance ---


def test_sample_physical_shape(problem, key):
    p = problem.sample_physical(key)
    assert p.shape == (problem.n_sources, 3)


def test_sample_point_shape(problem, key):
    x = problem.sample_point(key)
    assert x.shape == (problem.n_sources, 4)


def test_sample_point_matches_physical_to_flow_of_physical(problem, key):
    """sample_point(key) == chart.forward(sample_physical(key)); confirmed in
    sinusoid.py: sample_point calls sample_physical(key) with the same key, no split."""
    p = problem.sample_physical(key)
    x_direct = problem.sample_point(key)
    x_via_chart = problem.physical_to_flow(p)
    assert jnp.allclose(x_direct, x_via_chart, atol=1e-4)


def test_amplitude_within_amp_range(problem, key):
    p = problem.sample_physical(key)
    lo, hi = problem.amp_range
    assert jnp.all(p[..., 0] >= lo - 1e-6) and jnp.all(p[..., 0] <= hi + 1e-6)


def test_frequency_within_freq_range(problem, key):
    p = problem.sample_physical(key)
    lo, hi = problem.freq_range
    assert jnp.all(p[..., 1] >= lo - 1e-6) and jnp.all(p[..., 1] <= hi + 1e-6)


# --- clean_signal / sample_observation / snr ---


def test_clean_signal_shape(problem, key):
    p = problem.sample_physical(key)
    signal = problem.clean_signal(p)
    n_samples = round(problem.t_obs / problem.sampling_step)
    assert signal.shape[-1] == 2
    assert signal.shape[-2] in (n_samples, n_samples + 1)


def test_clean_signal_sample_count_matches_arange_convention(problem, key):
    """t = jnp.arange(0, t_obs, sampling_step) is half-open: exactly t_obs/sampling_step
    samples when it divides evenly, never +1."""
    p = problem.sample_physical(key)
    signal = problem.clean_signal(p)
    assert signal.shape[-2] == round(problem.t_obs / problem.sampling_step)


def test_clean_signal_zero_amplitude_is_zero(problem):
    p = jnp.zeros((problem.n_sources, 3)).at[..., 1].set(0.03).at[..., 2].set(0.3)
    assert jnp.allclose(problem.clean_signal(p), 0.0, atol=1e-6)


def test_clean_signal_permutation_invariant(problem, key):
    p = problem.sample_physical(key)
    perm = jnp.roll(jnp.arange(problem.n_sources), shift=1)
    assert jnp.allclose(
        problem.clean_signal(p), problem.clean_signal(p[perm]), atol=1e-4
    )


def test_clean_signal_additive_superposition(key):
    combined = NoisySinusoid(n_sources=2)
    k1, k2 = jr.split(key)
    single_a, single_b = NoisySinusoid(n_sources=1), NoisySinusoid(n_sources=1)
    pa, pb = single_a.sample_physical(k1), single_b.sample_physical(k2)
    p_joint = jnp.concatenate([pa, pb], axis=0)
    lhs = combined.clean_signal(p_joint)
    rhs = single_a.clean_signal(pa) + single_b.clean_signal(pb)
    assert jnp.allclose(lhs, rhs, atol=1e-4)


def test_clean_signal_invariant_to_2pi_phase_shift():
    problem = NoisySinusoid(n_sources=1, t_obs=64.0)
    p = jnp.array([[0.4, 0.03, 0.9]])
    shifted = p.at[:, 2].add(2 * jnp.pi)
    assert jnp.allclose(
        problem.clean_signal(p), problem.clean_signal(shifted), atol=1e-5
    )


def test_clean_signal_single_source_obeys_pythagorean_identity():
    """One source: channels are amp*sin and amp*cos, so ch0**2 + ch1**2 == amp**2."""
    problem = NoisySinusoid(n_sources=1, t_obs=64.0)
    amp = 0.6
    out = problem.clean_signal(jnp.array([[amp, 0.02, 1.234]]))
    assert jnp.allclose(out[..., 0] ** 2 + out[..., 1] ** 2, amp**2, atol=1e-5)


def test_sample_observation_noiseless_matches_clean_signal(key):
    problem = NoisySinusoid(n_sources=2, noise_level=0.0)
    p = problem.sample_physical(key)
    assert jnp.allclose(
        problem.sample_observation(key, p), problem.clean_signal(p), atol=1e-5
    )


def test_snr_permutation_invariant(problem, key):
    p = problem.sample_physical(key)
    perm = jnp.roll(jnp.arange(problem.n_sources), shift=1)
    assert jnp.allclose(problem.snr(p), problem.snr(p[perm]), atol=1e-4)


def test_snr_decreases_with_noise_level(key):
    low_noise = NoisySinusoid(n_sources=2, noise_level=0.1)
    high_noise = NoisySinusoid(n_sources=2, noise_level=10.0)
    p = low_noise.sample_physical(key)
    assert low_noise.snr(p) > high_noise.snr(p)


def test_snr_increases_with_amplitude(problem, key):
    p = problem.sample_physical(key)
    p_loud = p.at[..., 0].multiply(10.0)
    assert problem.snr(p_loud) > problem.snr(p)


def test_snr_linear_in_amplitude():
    problem = NoisySinusoid(n_sources=1)
    p = jnp.array([[0.4, 0.05, 0.6]])
    assert jnp.isclose(
        problem.snr(p.at[:, 0].multiply(3.0)), 3.0 * problem.snr(p), rtol=1e-4
    )


def test_snr_zero_amplitude_is_zero():
    problem = NoisySinusoid(n_sources=1)
    assert jnp.isclose(problem.snr(jnp.array([[0.0, 0.05, 0.6]])), 0.0, atol=1e-8)


def test_snr_independent_of_sampling_step():
    p = jnp.array([[0.4, 0.05, 0.6]])
    fine = NoisySinusoid(n_sources=1, sampling_step=0.5)
    coarse = NoisySinusoid(n_sources=1, sampling_step=1.0)
    assert jnp.isclose(fine.snr(p), coarse.snr(p), rtol=1e-4)


def test_snr_scales_as_inverse_sqrt_noise_level():
    p = jnp.array([[0.4, 0.05, 0.6]])
    quiet = NoisySinusoid(n_sources=1, noise_level=0.2)
    loud = NoisySinusoid(n_sources=1, noise_level=0.4)
    assert jnp.isclose(quiet.snr(p), jnp.sqrt(2.0) * loud.snr(p), rtol=1e-4)


def test_snr_quadrature_sums_across_well_separated_sources():
    p1, p2 = jnp.array([[0.4, 0.02, 0.6]]), jnp.array([[0.7, 0.08, 1.9]])
    single, both = NoisySinusoid(n_sources=1), NoisySinusoid(n_sources=2)
    expected = jnp.sqrt(single.snr(p1) ** 2 + single.snr(p2) ** 2)
    assert jnp.isclose(both.snr(jnp.concatenate([p1, p2], axis=0)), expected, rtol=0.05)


def test_snr_silent_source_does_not_contribute():
    active, silent = jnp.array([[0.5, 0.03, 0.4]]), jnp.array([[0.0, 0.06, 1.0]])
    single, both = NoisySinusoid(n_sources=1), NoisySinusoid(n_sources=2)
    combined = jnp.concatenate([active, silent], axis=0)
    assert jnp.isclose(both.snr(combined), single.snr(active), rtol=1e-4)


def test_snr_independent_of_frequency(problem, key):
    p = problem.sample_physical(key)
    p_shifted_freq = p.at[..., 1].set(problem.freq_range[1])
    assert jnp.allclose(problem.snr(p), problem.snr(p_shifted_freq), atol=1e-6)


def test_snr_independent_of_phase(problem, key):
    p = problem.sample_physical(key)
    p_shifted_phase = p.at[..., 2].set(p[..., 2] + jnp.pi)
    assert jnp.allclose(problem.snr(p), problem.snr(p_shifted_phase), atol=1e-6)


def test_snr_scales_as_sqrt_t_obs(key):
    problem_short = NoisySinusoid(n_sources=2, t_obs=1000.0)
    problem_long = NoisySinusoid(n_sources=2, t_obs=4000.0)
    p = problem_short.sample_physical(key)
    ratio = problem_long.snr(p) / problem_short.snr(p)
    assert jnp.allclose(ratio, 2.0, atol=1e-4)


def test_noise_variance_matches_noise_level_over_2_sampling_step():
    problem = NoisySinusoid(n_sources=1, noise_level=0.7, t_obs=64.0, sampling_step=1.0)
    p = jnp.array([[0.0, 0.05, 0.0]])
    keys = jr.split(jr.key(0), 500)
    residuals = jnp.stack(
        [problem.sample_observation(k, p) - problem.clean_signal(p) for k in keys]
    )
    expected_var = problem.noise_level / (2.0 * problem.sampling_step)
    assert jnp.isclose(jnp.var(residuals), expected_var, rtol=0.2)


# --- preprocess (time-frequency image) ---


def test_preprocess_shape_channels_preserved(problem, key):
    p = problem.sample_physical(key)
    o = problem.sample_observation(key, p)
    img = problem.preprocess(o)
    assert img.shape[-1] == 2
    assert img.ndim == o.ndim + 1


def test_preprocess_deterministic(problem, key):
    p = problem.sample_physical(key)
    o = problem.sample_observation(key, p)
    assert jnp.array_equal(problem.preprocess(o), problem.preprocess(o))


def test_preprocess_exact_time_frequency_shape(problem, key):
    """preprocess slices the prior band out of the rfft spectrum and WDM-transforms it,
    so f == wdm_freq_bands exactly and nt = patch_downsample * ceil(2*band_bins /
    (wdm_freq_bands*patch_downsample)). For the defaults (t_obs=10000, sampling_step=1,
    freq_range=(0.01,0.1), wdm_freq_bands=512, patch_downsample=4) that is (4, 512)."""
    p = problem.sample_physical(key)
    o = problem.sample_observation(key, p)
    img = problem.preprocess(o)
    assert img.shape[-3:] == (4, problem.wdm_freq_bands, 2)


def test_batched_leading_dim_clean_signal(problem, key):
    keys = jr.split(key, 4)
    p_batch = jnp.stack([problem.sample_physical(k) for k in keys])
    signal_batch = problem.clean_signal(p_batch)
    assert signal_batch.shape[0] == 4
    assert signal_batch.shape[1:] == problem.clean_signal(p_batch[0]).shape


# --- train_sample ---


def test_train_sample_shapes(problem, key):
    sample = problem.train_sample(key)
    assert sample.xt.shape == (problem.n_sources, 4)
    assert sample.x_target.shape == (problem.n_sources, 4)
    assert sample.dx.shape == sample.xt.shape
    assert sample.t.shape == ()


def test_train_sample_t_in_unit_interval(problem, key):
    """Base Problem.train_sample draws t = jr.uniform(key, ()), half-open [0,1);
    already pinned down precisely for the generic case in test_base.py, kept loose here.
    """
    sample = problem.train_sample(key)
    assert 0.0 <= float(sample.t) <= 1.0


def test_train_sample_y_y_target_shape_match(problem, key):
    sample = problem.train_sample(key)
    assert sample.y.shape == sample.y_target.shape


def test_train_sample_deterministic_given_key(problem, key):
    a = problem.train_sample(key)
    b = problem.train_sample(key)
    # NoisySinusoid carries no context, so that field is None on both samples
    for field_a, field_b in zip(a, b):
        if field_a is not None:
            assert jnp.array_equal(field_a, field_b)


def test_train_sample_varies_with_key(problem, key):
    key2 = jr.key(999)
    a = problem.train_sample(key)
    b = problem.train_sample(key2)
    assert not jnp.array_equal(a.xt, b.xt)


def test_train_sample_x_target_on_manifold(problem, key):
    """Point layout is [logAffine(amp), logAffine(freq), cos(phase), sin(phase)]
    (confirmed from physical_to_flow's amp/freq/phase block order); the last 2 dims
    are the (cos,sin) pair, unit-norm."""
    sample = problem.train_sample(key)
    norm_sq = jnp.sum(sample.x_target[..., -2:] ** 2, axis=-1)
    assert jnp.allclose(norm_sq, 1.0, atol=1e-3)


def test_train_sample_xt_on_manifold_for_all_t(problem, key):
    sample = problem.train_sample(key)
    norm_sq = jnp.sum(sample.xt[..., -2:] ** 2, axis=-1)
    assert jnp.allclose(norm_sq, 1.0, atol=1e-3)


def test_train_sample_y_equals_y_target_when_noiseless(key):
    """y comes from a noisy observation and y_target from a clean one, so they coincide
    at noise_level=0, where the noise term is scaled by
    sqrt(noise_level/(2*sampling_step)) == 0."""
    problem = NoisySinusoid(n_sources=2, noise_level=0.0)
    sample = problem.train_sample(key)
    assert jnp.allclose(sample.y, sample.y_target, atol=1e-4)


def test_y_is_noisy_and_y_target_clean_through_wdm_conditioning(problem):
    """Through NoisySinusoid's nonlinear, WDM-transform-based preprocess: with
    noise_level > 0, y carries noise the clean y_target does not, so they differ."""
    sample = problem.train_sample(jr.key(3))
    assert not jnp.allclose(sample.y, sample.y_target, atol=1e-6)


def test_geodesic_t1_endpoint_can_differ_from_raw_x_target(problem):
    """geometries.Set.log_map reassigns x1's source rows to whichever
    permutation best matches x0 (geometries.py's `assign`) before taking the displacement.
    train_sample's x_target is the raw, un-reordered physical_to_flow(p) (sinusoid.py: x1 =
    self.physical_to_flow(p)). So geometry.geodesic(1, x0, x_target) is Set.assign(x0,
    x_target), not x_target itself, whenever the optimal assignment isn't already the
    identity permutation -- which is generic for two independently drawn point sets.
    Anything downstream that expects the geodesic to literally converge onto x_target at
    t=1 will see a source relabeling instead."""
    mismatches = 0
    for seed in range(20):
        key = jr.key(seed)
        # mirrors train_sample's split; unpacked in full so a change here breaks loudly
        key_p, key_o, key_x0, key_t = jr.split(key, 4)
        x0 = problem.sample_point(key_x0)
        sample = problem.train_sample(key)
        # x0 is the one train_sample actually drew, else the rest is vacuous
        assert jnp.allclose(
            problem.geodesic(sample.t, x0, sample.x_target),
            sample.xt,
            atol=1e-4,
        )
        g1 = problem.geodesic(jnp.array(1.0), x0, sample.x_target)
        matched = problem.geometry.assign(x0, sample.x_target)
        assert jnp.allclose(g1, matched, atol=1e-4)
        if not jnp.allclose(g1, sample.x_target, atol=1e-4):
            mismatches += 1
    assert mismatches > 0


# --- wdm band resolution ---


def test_prior_frequencies_resolve_within_the_band(problem):
    """The prior's frequency range must sit inside the resolvable band: above one WDM
    channel width and below Nyquist, so a tone lands in an interior channel."""
    nyquist = 0.5 / problem.sampling_step
    band_width = nyquist / problem.wdm_freq_bands
    low, high = problem.freq_range
    assert low > band_width
    assert high < nyquist


def test_clean_signal_peak_band_increases_with_frequency(problem):
    """A pure tone peaks in a WDM channel that grows monotonically with its frequency:
    sweeping the prior band low->high walks the argmax channel strictly upward. The exact
    channel offset is a WDM-window artifact, so this pins the ordering, not an index."""
    low, high = problem.freq_range
    fractions = jnp.linspace(0.0, 1.0, 6)
    freqs = jnp.exp(jnp.log(low) + fractions * (jnp.log(high) - jnp.log(low)))
    peaks = [
        int(
            jnp.argmax(
                jnp.abs(
                    problem.preprocess(
                        problem.clean_signal(jnp.array([[1.0, float(f), 0.0]]))
                    )
                ).sum(axis=(0, 2))
            )
        )
        for f in freqs
    ]
    assert all(lo < hi for lo, hi in zip(peaks[:-1], peaks[1:]))
