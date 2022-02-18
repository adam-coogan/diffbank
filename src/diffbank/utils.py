"""
Utilities for calculating properties of binaries, sampling their parameters,
comparing waveforms and generating template banks.

Warning:
    Much of this module will be refactored into its own package in the future.
"""
from contextlib import nullcontext
from math import pi
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
from tqdm.auto import tqdm, trange

from .constants import C, G

# TODO: what type should this be?
PRNGKeyArray = jax._src.prng.PRNGKeyArray  # type: ignore
Array = jnp.ndarray


def ms_to_Mc_eta(m):
    r"""
    Converts binary component masses to chirp mass and symmetric mass ratio.

    Args:
        m: the binary component masses ``(m1, m2)``

    Returns:
        :math:`(\mathcal{M}, \eta)`, with the chirp mass in the same units as
        the component masses
    """
    m1, m2 = m
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5), m1 * m2 / (m1 + m2) ** 2


def get_f_isco(m):
    r"""
    Computes the ISCO frequency for a black hole.

    Args:
        m: the black hole's mass in kg

    Returns:
        The ISCO frequency in Hz
    """
    return 1 / (6 ** (3 / 2) * pi * m / (C ** 3 / G))


def get_M_eta_sampler(
    M_range: Tuple[float, float], eta_range: Tuple[float, float]
) -> Callable[[PRNGKeyArray, int], Array]:
    """
    Uniformly values of the chirp mass and samples over the specified ranges.
    This function may be removed in the future since it is trivial.
    """

    def sampler(key, n):
        M_eta = random.uniform(
            key,
            minval=jnp.array([M_range[0], eta_range[0]]),
            maxval=jnp.array([M_range[1], eta_range[1]]),
            shape=(n, 2),
        )
        return M_eta

    return sampler


def get_m1_m2_sampler(
    m1_range: Tuple[float, float], m2_range: Tuple[float, float]
) -> Callable[[PRNGKeyArray, int], Array]:
    r"""
    Creates a function to uniformly sample two parameters, with the restriction
    that the first is larger than the second.

    Note:
        While this function is particularly useful for sampling masses in a
        binary, nothing in it is specific to that context.

    Args:
        m1_range: the minimum and maximum values of the first parameter
        m2_range: the minimum and maximum values of the second parameter

    Returns:
        The sampling function
    """

    def sampler(key, n):
        ms = random.uniform(
            key,
            minval=jnp.array([m1_range[0], m2_range[0]]),
            maxval=jnp.array([m1_range[1], m2_range[1]]),
            shape=(n, 2),
        )
        return jnp.stack([ms.max(axis=1), ms.min(axis=1)]).T

    return sampler


def get_eff_pads(fs: Array) -> Tuple[Array, Array]:
    r"""
    Gets arrays of zeros to pad a function evaluated on a frequency grid so the
    function values can be passed to ``jax.numpy.fft.ifft``.

    Args:
        fs: uniformly-spaced grid of frequencies. It is assumed that the first
            element in the grid must be an integer multiple of the grid spacing
            (i.e., ``fs[0] % df == 0``, where ``df`` is the grid spacing).

    Returns:
        The padding arrays of zeros. The first is of length ``fs[0] / df`` and
        the second is of length ``fs[-1] / df - 2``.
    """
    df = fs[1] - fs[0]
    N = 2 * jnp.array(fs[-1] / df - 1).astype(int)
    pad_low = jnp.zeros(jnp.array(fs[0] / df).astype(int))
    pad_high = jnp.zeros(N - jnp.array(fs[-1] / df).astype(int))
    return pad_low, pad_high


def get_phase_maximized_inner_product(
    theta1: Array,
    theta2: Array,
    del_t: Array,
    amp1: Callable[[Array, Array], Array],
    Psi1: Callable[[Array, Array], Array],
    amp2: Callable[[Array, Array], Array],
    Psi2: Callable[[Array, Array], Array],
    fs: Array,
    Sn: Callable[[Array], Array],
) -> Array:
    r"""
    Calculates the inner product between two waveforms, maximized over the difference
    in phase at coalescence. This is just the absolute value of the noise-weighted
    inner product.

    Args:
        theta1: parameters for the first waveform
        theta2: parameters for the second waveform
        del_t: difference in the time at coalescence for the waveforms
        amp1: amplitude function for first waveform
        Psi1: phase function for first waveform
        amp2: amplitude function for second waveform
        Psi2: phase function for second waveform
        fs: uniformly-spaced grid of frequencies used to perform the integration
        Sn: power spectral density of the detector noise

    Returns:
        The noise-weighted inner product between the waveforms, maximized over
        the phase at coalescence
    """
    # Evaluate all functions over frequency gradient
    amps = amp1(fs, theta1)
    amp_del = amp2(fs, theta2)
    amp_prods = amps * amp_del
    del_phases = (2.0 * pi * fs * del_t) + (Psi2(fs, theta2) - Psi1(fs, theta1))
    Sns = Sn(fs)

    # Normalize both waveforms
    norm = jnp.sqrt(4 * jnp.trapz(amps ** 2 / Sns, fs))
    norm_del = jnp.sqrt(4 * jnp.trapz(amp_del ** 2 / Sns, fs))

    # Compute unnormalized match, maximizing over phi_0 by taking the absolute value
    re_integral = 4.0 * jnp.trapz(amp_prods * jnp.cos(del_phases) / Sns, fs)
    im_integral = 4.0 * jnp.trapz(amp_prods * jnp.sin(del_phases) / Sns, fs)
    match_un = jnp.sqrt(re_integral ** 2 + im_integral ** 2)

    return match_un / (norm * norm_del)


def get_match(
    theta1: Array,
    theta2: Array,
    amp1: Callable[[Array, Array], Array],
    Psi1: Callable[[Array, Array], Array],
    amp2: Callable[[Array, Array], Array],
    Psi2: Callable[[Array, Array], Array],
    fs: Array,
    Sn: Callable[[Array], Array],
    pad_low: Array,
    pad_high: Array,
) -> Array:
    r"""
    Calculates the match between two waveforms with different parameters and of
    distinct types. The match is defined as the noise-weighted inner product maximized
    over the difference in time and phase at coalescence. The maximizations are
    performed using the absolute value of the inverse Fourier transform trick.

    Args:
        theta1: parameters for the first waveform
        theta2: parameters for the second waveform
        amp1: amplitude function for first waveform
        Psi1: phase function for first waveform
        amp2: amplitude function for second waveform
        Psi2: phase function for second waveform
        fs: uniformly-spaced grid of frequencies used to perform the integration
        Sn: power spectral density of the detector noise
        pad_low: array of zeros to pad the left side of the integrand before it
            is passed to ``jax.numpy.fft.ifft``
        pad_right: array of zeros to pad the right side of the integrand before
            it is passed to ``jax.numpy.fft.ifft``

    Returns:
        The match :math:`m[\theta_1, \theta_2]`
    """
    h1 = amp1(fs, theta1) * jnp.exp(1j * Psi1(fs, theta1))
    h2 = amp2(fs, theta2) * jnp.exp(1j * Psi2(fs, theta2))
    Sns = Sn(fs)
    return get_match_arr(h1, h2, Sns, fs, pad_low, pad_high)


def get_match_arr(
    h1: Array, h2: Array, Sns: Array, fs: Array, pad_low: Array, pad_high: Array
) -> Array:
    """
    Calculates the match between two frequency-domain complex strains. The maximizations
    over the difference in time and phase at coalescence are performed by taking
    the absolute value of the inverse Fourier transform.

    Args:
        h1: the first set of strains
        h2: the second set of strains
        Sns: the noise power spectral densities
        fs: frequencies at which the strains and noise PSDs were evaluated
        pad_low: array of zeros to pad the left side of the integrand before it
            is passed to ``jax.numpy.fft.ifft``
        pad_right: array of zeros to pad the right side of the integrand before
            it is passed to ``jax.numpy.fft.ifft``

    Returns:
        The match.
    """
    # Factors of 4 and df drop out due to linearity
    norm1 = jnp.sqrt(jnp.sum(jnp.abs(h1) ** 2 / Sns))
    norm2 = jnp.sqrt(jnp.sum(jnp.abs(h2) ** 2 / Sns))

    # Use IFFT trick to maximize over t_c. Ref: Maggiore's book, eq. 7.171.
    integrand_padded = jnp.concatenate((pad_low, h1.conj() * h2 / Sns, pad_high))
    # print(low_padding, high_padding, len(fs), N)
    return jnp.abs(len(integrand_padded) * jnp.fft.ifft(integrand_padded)).max() / (
        norm1 * norm2
    )


def sample_uniform_ball(key: PRNGKeyArray, dim: int, shape: Tuple[int] = (1,)) -> Array:
    r"""
    Uniformly sample from the unit ball.

    Args:
        key: key to pass to sampler
        dim: dimensionality of the ball
        shape: shape of samples to draw

    Returns:
        Samples from the ``dim``-dimensional unit ball of shape ``shape``
    """
    xs = random.normal(key, shape + (dim,))
    abs_xs = jnp.sqrt(jnp.sum(xs ** 2, axis=-1, keepdims=True))
    sphere_samples = xs / abs_xs
    rs = random.uniform(key, shape + (1,)) ** (1 / dim)
    return sphere_samples * rs


def sample_uniform_metric_ellipse(key: PRNGKeyArray, g: Array, n: int) -> Array:
    r"""
    Uniformly sample inside a metric ellipse centered at the origin.

    Args:
        key: key to pass to sampler
        g: the metric
        n: the number of samples to draw

    Returns:
        ``n`` samples from the metric ellipse centered at the origin
    """
    dim = g.shape[1]
    # radius = jnp.sqrt(m_star)
    ball_samples = sample_uniform_ball(key, dim, (n,))
    trafo = jnp.linalg.inv(jnp.linalg.cholesky(g))
    return ball_samples @ trafo.T


def get_template_frac_in_bounds(
    key: PRNGKeyArray,
    theta: Array,
    get_g: Callable[[Array], Array],
    m_star,
    is_in_bounds: Callable[[Array], Array],
    n: int,
) -> Tuple[Array, Array]:
    r"""
    Perform a Monte Carlo estimate of the fraction of a template's metric
    ellipse lying inside the parameter space.

    Args:
        key: key to pass to sampler
        theta: the template's location
        get_g: a function to compute the metric at a point
        m_star: the minimum match, used to scale the metric ellipse by
            ``sqrt(m_star)``
        is_in_bounds: callable that takes a point and returns ``1`` if it is in
            the parameter space and ``0`` if not
        n: number of points to sample for the Monte Carlo estimate

    Returns:
        Estimate and error for the fraction of the template ellipse at ``theta``
        lying inside the parameter space
    """
    # Rescale metric ellipse samples to have radius ``sqrt(m_star)`` and
    # recenter on ``theta``
    ellipse_samples_0 = sample_uniform_metric_ellipse(key, get_g(theta), n)
    ellipse_samples = jnp.sqrt(m_star) * ellipse_samples_0 + theta
    in_bounds = jnp.concatenate(
        (jnp.array([1.0]), jax.lax.map(is_in_bounds, ellipse_samples))
    )
    return in_bounds.mean(), in_bounds.std() / jnp.sqrt(n + 1)


def est_ratio_max(
    key: PRNGKeyArray,
    density_fun: Callable[[Array], Array],
    sample_base: Callable[[PRNGKeyArray, int], Array],
    density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
    n_iter: int = 1000,
    n_init: int = 200,
    show_progress: bool = True,
) -> Tuple[Array, Array]:
    r"""
    Estimate maximum of the ratio of target to base density using empirical
    supremum rejection sampling.

    Note:
        Not ``jit``-able.

    References:
        https://bookdown.org/rdpeng/advstatcomp/rejection-sampling.html

    Args:
        density_fun: ``jit``-able target density function
        sample_base: ``jit``-able base sampler taking a key and number of
            samples as arguments
        density_fun_base: ``jit``-able density function for ``sample_base``.
            Need not be normalized.
        n_iter: number of iterations of rejection sampling to perform to
            estimate the ratio of densities
        n_init: as an initial guess, the maximum ratio will be computed over
            by sampling ``n_init`` points from ``sample_base``
        show_progress: displays a ``tqdm`` progress bar if ``True``

    Returns:
        The estimated maximum ratio between ``density_fun`` and
        ``density_fun_base``, and point at which it was attained. This can be
        passed as ``ratio_max`` to other sampling functions.
    """
    # Get initial guess for ratio_max by computing it at random points
    key, subkey = random.split(key)
    thetas = sample_base(subkey, n_init)
    densities = jax.lax.map(density_fun, thetas)
    densities_base = jax.lax.map(density_fun_base, thetas)
    ratios = densities / densities_base
    idx_max = jnp.argmax(ratios)
    ratio_max = ratios[idx_max]
    theta_max = thetas[idx_max]

    @jax.jit
    def rejection_sample(key, ratio_max):
        """Generate ratio and point by rejection sampling."""

        def cond_fun(val):
            cond_key, ratio = val[1], val[2]
            u = random.uniform(cond_key)
            return u >= ratio / ratio_max

        def body_fun(val):
            key = val[0]
            key, theta_key, cond_key = random.split(key, 3)
            theta = sample_base(theta_key, 1)[0]
            ratio = density_fun(theta) / density_fun_base(theta)
            return (key, cond_key, ratio, theta)

        # Only first element of init_val matters
        init_val = body_fun((key, None, None, None))
        # Get max ratio and point at which it is attained
        _, _, ratio, theta = jax.lax.while_loop(cond_fun, body_fun, init_val)
        return ratio, theta

    iterator = trange(n_iter) if show_progress else range(n_iter)
    for _ in iterator:
        key, subkey = random.split(key)
        ratio, theta = rejection_sample(subkey, ratio_max)

        if ratio > ratio_max:
            ratio_max = ratio
            theta_max = theta

            if show_progress:
                iterator.set_postfix_str(f"{ratio:.3e} at {theta}")  # type: ignore

    if show_progress:
        iterator.close()  # type: ignore

    return ratio_max, theta_max


def gen_template_rejection(
    key: PRNGKeyArray,
    ratio_max: Union[float, Array],
    density_fun: Callable[[Array], Array],
    sample_base: Callable[[PRNGKeyArray, int], Array],
    density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
) -> Array:
    """
    Generates a single template using rejection sampling.

    Note:
        While this function is used to generate templates in ``diffbank``, it
        can be used for any rejection sampling task.

    Args:
        key: key to pass to sampler
        ratio_max: maximum value of the ratio of ``density_fun`` to
            ``density_fun_base``
        density_fun: ``jit``-able target density function
        sample_base: ``jit``-able base sampler taking a key and number of
            samples as arguments
        density_fun_base: ``jit``-able density function for ``sample_base``.
            Need not be normalized.

    Returns:
        A single sample from the distribution with density ``density_fun``
    """

    def cond_fun(val):
        cond_key, theta = val[1], val[2]
        u = random.uniform(cond_key)
        return u >= density_fun(theta) / (ratio_max * density_fun_base(theta))

    def body_fun(val):
        key = val[0]
        key, theta_key, cond_key = random.split(key, 3)
        theta = sample_base(theta_key, 1)[0]
        return (key, cond_key, theta)  # new val

    # Only first element of init_val matters
    init_val = body_fun((key, None, None))

    return jax.lax.while_loop(cond_fun, body_fun, init_val)[2]


def gen_templates_rejection(
    key: PRNGKeyArray,
    n_templates: int,
    ratio_max: Union[float, Array],
    density_fun: Callable[[Array], Array],
    sample_base: Callable[[PRNGKeyArray, int], Array],
    density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
) -> Array:
    """
    Generates multiple templates using rejection sampling.

    Note:
        While this function is used to generate templates in ``diffbank``, it
        can be used for any rejection sampling task.

    Args:
        key: key to pass to sampler
        n_templates: the number of templates to sample
        ratio_max: maximum value of the ratio of ``density_fun`` to
            ``density_fun_base``
        density_fun: ``jit``-able target density function
        sample_base: ``jit``-able base sampler taking a key and number of
            samples as arguments
        density_fun_base: ``jit``-able density function for ``sample_base``.
            Need not be normalized.

    Returns:
        ``n_templates`` samples from the distribution with density
        ``density_fun``
    """
    keys = random.split(key, n_templates)
    f = lambda key: gen_template_rejection(
        key, ratio_max, density_fun, sample_base, density_fun_base
    )
    return jax.lax.map(f, keys)


def _update_uncovered_eff(
    pt: Array,
    eff: Array,
    template: Array,
    minimum_match: Union[float, Array],
    match_fun: Callable[[Array, Array], Array],
) -> Array:
    """
    Helper function to compute match for a point only if it is not already
    covered by a template.
    """
    return jax.lax.cond(
        eff < minimum_match,
        lambda pt: match_fun(template, pt),
        lambda _: eff,
        pt,
    )


def gen_bank_random(
    key: PRNGKeyArray,
    minimum_match: Union[float, Array],
    eta: Union[float, Array],
    match_fun: Callable[[Array, Array], Array],
    ratio_max: Array,
    density_fun: Callable[[Array], Array],
    sample_base: Callable[[PRNGKeyArray, int], Array],
    density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
    eff_pt_sampler: Callable[[PRNGKeyArray], Array] = None,
    n_eff: int = 1000,
    show_progress: bool = True,
) -> Tuple[Array, Array]:
    r"""
    Generates a random bank using the method introduced in Coogan et al 2022.

    References:
        See also `Messenger, Prix & Papa 2008 <https://arxiv.org/abs/0809.5223>`_

    Note:
        Not ``jit``-able

    Args:
        key: key to pass to sampler
        minimum_match: the target minimum match match of the bank
        eta: the target fraction of parameters space to cover with the bank
        match_fun: ``jit``-able function to compute the match between two points
        ratio_max: maximum value of the ratio of ``density_fun`` to
            ``density_fun_base``
        density_fun: ``jit``-able target density function
        sample_base: ``jit``-able base sampler taking a key and number of
            samples as arguments
        density_fun_base: ``jit``-able density function for ``sample_base``.
            Need not be normalized.
        eff_pt_sampler: if provided, this function will be used to sample
            effectualness points instead of sampling them with the same density
            as templates
        n_eff: the number of effectualness points used to monitor convergence
        show_progress: displays a ``tqdm`` progress bar if ``True``

    Returns:
        The array of template positions and effectualness points used to monitor
        convergence
    """
    # Function for rejection sampling of templates
    gen_template = jax.jit(
        lambda key: gen_template_rejection(
            key, ratio_max, density_fun, sample_base, density_fun_base
        )
    )
    if eff_pt_sampler is None:
        eff_pt_sampler = gen_template

    # Generate points for effectualness monitoring
    key, subkey = random.split(key)
    eff_pts = jnp.array([eff_pt_sampler(k) for k in random.split(subkey, n_eff)])
    effs = jnp.zeros(n_eff)
    n_covered = 0

    # Close over eff_pts
    @jax.jit
    def update_uncovered_effs(template, effs):
        update = lambda ep: _update_uncovered_eff(
            ep["point"], ep["eff"], template, minimum_match, match_fun
        )
        return jax.vmap(update)({"point": eff_pts, "eff": effs})

    # Fill the bank!
    templates = []
    n_ko = int(jnp.ceil(n_eff * eta))
    with tqdm(total=n_ko) if show_progress else nullcontext() as pbar:
        while n_covered < n_ko:
            # Make template
            key, key_template = random.split(key)
            template = gen_template(key_template)
            templates.append(template)

            # Compute matches
            effs = update_uncovered_effs(template, effs)
            # Update coverage count
            dn_covered = (effs > minimum_match).sum() - n_covered
            n_covered += dn_covered

            if show_progress:  # pbar is a tqdm
                pbar.update(int(dn_covered))  # type: ignore
                pbar.set_postfix_str(f"n_templates = {len(templates)}")  # type: ignore

    return jnp.array(templates), eff_pts


def gen_bank_stochastic(
    key: PRNGKeyArray,
    minimum_match: Union[float, Array],
    eta: Union[float, Array],
    match_fun: Callable[[Array, Array], Array],
    propose_template: Callable[[PRNGKeyArray], Array],
    eff_pt_sampler: Callable[[PRNGKeyArray], Array],
    n_eff: int = 1000,
    show_progress: bool = True,
    n_acc_monitoring: int = 1,  # number of iterations for acc rate moving average
) -> Tuple[Array, Array]:
    r"""
    Generates a stochastic bank by adding an accept/reject step to the random
    bank generation method introduced in Coogan et al 2022. This step rejects
    the template if its effectualness is greater than the target minimum match
    with respect to the bank at the time it is sampled.

    References:
        See `this paper <https://arxiv.org/abs/0908.2090>`_, for example

    Note:
        Not ``jit``-able

    Args:
        key: key to pass to sampler
        minimum_match: the target minimum match match of the bank
        eta: the target fraction of parameters space to cover with the bank
        match_fun: ``jit``-able function to compute the match between two points
        propose_template: ``jit``-able function for sampling new templates
        eff_pt_sampler: if provided, this function will be used to sample
            effectualness points instead of sampling them with the same density
            as templates
        n_eff: the number of effectualness points used to monitor convergence
        show_progress: displays a ``tqdm`` progress bar if ``True``
        n_acc_monitoring: number of iterations to use for moving average
            calculation of the acceptance rate. Only relevant if
            ``show_progress`` is ``True``.

    Returns:
        The array of template positions and effectualness points used to monitor
        convergence
    """
    match_fun = jax.jit(match_fun)
    propose_template = jax.jit(propose_template)
    if eff_pt_sampler is None:
        eff_pt_sampler = propose_template

    # Generate points for effectualness monitoring
    key, subkey = random.split(key)
    eff_pts = jnp.array([eff_pt_sampler(k) for k in random.split(subkey, n_eff)])
    effs = jnp.zeros(n_eff)
    n_covered = 0

    # Close over eff_pts
    @jax.jit
    def update_uncovered_effs(template, effs):
        update = lambda ep: _update_uncovered_eff(
            ep["point"], ep["eff"], template, minimum_match, match_fun
        )
        return jax.vmap(update)({"point": eff_pts, "eff": effs})

    # Returns True if point is far from all templates (ie, has a low
    # effectualness to the bank)
    def accept(pt, templates):
        effs = map(lambda template: match_fun(template, pt), templates)
        return max(effs) < minimum_match

    # Add first template
    key, subkey = random.split(key)
    templates = [propose_template(subkey)]
    n_proposals = 1
    acc_rates = []
    n_ko = int(jnp.ceil(n_eff * eta))
    with tqdm(total=n_ko) if show_progress else nullcontext() as pbar:
        while n_covered < n_ko:
            # Make a template
            n_proposal_it = 0
            while True:
                key, subkey = random.split(key)
                template = propose_template(subkey)
                n_proposal_it += 1
                if accept(template, templates):
                    templates.append(template)
                    break

            # Update monitoring
            n_proposals += n_proposal_it
            acc_rate_it = 1 / n_proposal_it  # for the round
            if len(acc_rates) == n_acc_monitoring:
                acc_rates = acc_rates[1:]
            acc_rates.append(acc_rate_it)
            acc_rate = sum(acc_rates) / len(acc_rates)

            # Compute matches
            effs = update_uncovered_effs(template, effs)
            # Update coverage count
            dn_covered = (effs > minimum_match).sum() - n_covered
            n_covered += dn_covered

            if show_progress:  # pbar is a tqdm
                pbar.update(int(dn_covered))  # type: ignore
                pbar.set_postfix_str(  # type: ignore
                    f"acc rate = {acc_rate:.3f}, {len(templates)} / {n_proposals}"
                )

    return jnp.array(templates), eff_pts


def get_top_matches_templates(
    templates: Array,
    match_fun: Callable[[Array, Array], Array],
    points: Array,
    n_top: int = 1,
    show_progress: bool = True,
) -> Tuple[Array, Array]:
    r"""
    Find templates that have highest match with respect to a set of points.

    Args:
        templates: a template bank
        match_fun: ``jit``-able function to compute the match taking the template
            as the first argument and point as the second argument.
        points: a single point (1D array) or a set of points (2D array)
        n_top: number of best-matching templates to return for each point. If this
            is 1, the best template and its match (i.e., the bank's effectualness)
            are returned for each point. If this is greater than 1, the ``n_top``
            best-matching templates are returned for each point.
        show_progress: displays a ``tqdm`` progress bar if ``True``

    Raises:
        ValueError: if ``n_top < 1`` or if ``points`` is not 1D or 2D

    Returns:
        The top matches and corresponding templates in the bank for each point.
        If ``n_top`` is 1, these will have shapes ``(n,)`` and ``(n, p)``, where
        ``n`` is the length of ``points`` and ``p`` is the dimension of each of
        its elements. If ``n_top`` is greater than 1, their returned arrays will
        have shapes ``(n, n_top)`` and ``(n, n_top, p)``.
    """
    # Compile in the templates and waveform model
    if n_top == 1:

        @jax.jit
        def get_matches_idxs(pt: Array) -> Tuple[Array, Array]:
            matches = jax.lax.map(lambda template: match_fun(template, pt), templates)
            idx = jnp.argmax(matches)
            return (matches[idx], idx)

    elif n_top > 1:

        @jax.jit
        def get_matches_idxs(pt: Array) -> Tuple[Array, Array]:
            matches = jax.lax.map(lambda template: match_fun(template, pt), templates)
            idxs = jnp.argsort(matches, 0)[-n_top:]
            return (jnp.take_along_axis(matches, idxs, 0), idxs)

    else:
        raise ValueError("n_top must be >= 1")

    if points.ndim == 1:
        points = points.reshape(1, -1)
    elif points.ndim != 2:
        raise ValueError("points must be 1D or 2D")

    top_matches = []
    top_templates = []
    with tqdm(points) if show_progress else nullcontext() as pbar:
        for pt in pbar:  # type: ignore
            tms, idxs = get_matches_idxs(pt)
            top_matches.append(tms)
            if idxs.ndim == 0:
                tts = templates[idxs]
            else:
                tts = jnp.take_along_axis(templates, idxs[:, None], 0)
            top_templates.append(tts)

    return jnp.stack(top_matches), jnp.stack(top_templates)


def get_bank_effectualness(
    key: PRNGKeyArray,
    minimum_match: Union[float, Array],
    templates: Array,
    match_fun: Callable[[Array, Array], Array],
    eff_pt_sampler: Callable[[PRNGKeyArray], Array],
    n: int = 100,
    show_progress: bool = True,
) -> Tuple[Array, Array, Array, Array]:
    r"""
    Computes effectualness of a bank at random points.

    Args:
        key: key to pass to sampler
        minimum_match: the target minimum match match of the bank
        templates: the bank's templates' locations
        match_fun: ``jit``-able function to compute the match between two points
        eff_pt_sampler: function for sampling points at which to compute the
            bank's effectualness
        n: the number of points at which to compute the bank's effectualness
        show_progress: displays a ``tqdm`` progress bar if ``True``

    Returns:
        Effectualness, the corresponding points, and the Monte Carlo
        estimate and error for the fraction of parameter space covered by the
        bank
    """
    # Compile in the templates and waveform model
    @jax.jit
    def get_bank_eff(pt):
        return jax.lax.map(lambda template: match_fun(template, pt), templates).max()

    # Sample points and compute effectualnesses
    eff_pts = jnp.array([eff_pt_sampler(k) for k in random.split(key, n)])
    effs = []
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    eta_est, eta_est_err, M_2 = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
    with tqdm(eff_pts) if show_progress else nullcontext() as pbar:
        for n, pt in enumerate(pbar, start=1):  # type: ignore
            effs.append(get_bank_eff(pt))

            x = effs[-1] > minimum_match
            eta_prev = eta_est
            eta_est = eta_prev + (x - eta_prev) / n
            M_2 = M_2 + (x - eta_prev) * (x - eta_est)
            if n > 1:
                # Standard deviation of the mean
                eta_est_err = jnp.sqrt(M_2 / (n - 1)) / jnp.sqrt(n)

            if show_progress:  # pbar is a tqdm
                pbar.set_postfix_str(f"eta = {eta_est:.3f} +/- {eta_est_err:.3f}")  # type: ignore

    return jnp.array(effs), eff_pts, eta_est, eta_est_err
