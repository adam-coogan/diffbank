"""
Functions related to the parameter space metric.
"""
# from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import hessian  # , jit

from .utils import Array, get_phase_maximized_inner_product


# @partial(jit, static_argnames=("amp", "Psi", "Sn"))
def get_gam(
    theta: Array,
    amp: Callable[[Array, Array], Array],
    Psi: Callable[[Array, Array], Array],
    fs: Array,
    Sn: Callable[[Array], Array],
) -> Array:
    """
    Computes the metric for the time of coalescence and intrinsic parameters of
    the binary.

    Args:
        theta: point at which to compute metric
        amp: function returning the amplitude of the Fourier-domain waveform as
            a function of frequency and parameters
        Psi: function returning the phase of the Fourier-domain waveform as a
            function of frequency and parameters
        fs: uniformly-spaced grid of frequencies used to compute the
            noise-weighted inner product
        Sn: power spectral density of the detector noise

    Return:
        The metric. The first value of the index corresponds to the time of
        coalescence. The other values of the index correspond to the intrinsic
        parameters, in the same order they appear in ``theta``.
    """
    hess_func = lambda delta: get_phase_maximized_inner_product(
        theta, theta + delta[1:], delta[0], amp, Psi, amp, Psi, fs, Sn
    )
    del_theta = jnp.zeros(theta.size + 1)
    return -1 / 2 * hessian(hess_func)(del_theta)


# @partial(jit, static_argnames=("amp", "Psi", "Sn"))
def get_g(
    theta: Array,
    amp: Callable[[Array, Array], Array],
    Psi: Callable[[Array, Array], Array],
    fs: Array,
    Sn: Callable[[Array], Array],
) -> Array:
    """
    Computes the metric for the intrinsic binary parameters of the binary,
    maximized over the time and phase at coalescence.

    Args:
        theta: point at which to compute metric
        amp: function returning the amplitude of the Fourier-domain waveform as
            a function of frequency and parameters
        Psi: function returning the phase of the Fourier-domain waveform as a
            function of frequency and parameters
        fs: uniformly-spaced grid of frequencies used to compute the
            noise-weighted inner product
        Sn: power spectral density of the detector noise

    Return:
        The metric, where the values of the index correspond to the intrinsic
        parameters in the same order they appear in ``theta``.
    """
    gam = get_gam(theta, amp, Psi, fs, Sn)
    # Maximize over Delta t_0
    return gam[1:, 1:] - jnp.outer(gam[0, 1:], gam[0, 1:]) / gam[0, 0]


# @partial(jit, static_argnames=("amp", "Psi", "Sn"))
def get_density(
    theta: Array,
    amp: Callable[[Array, Array], Array],
    Psi: Callable[[Array, Array], Array],
    fs: Array,
    Sn: Callable[[Array], Array],
) -> Array:
    r"""
    Computes the determinant of the metric over the intrinsic of the binary,
    maximized over the time and phase at coalescence.

    Args:
        theta: point at which to compute metric density
        amp: function returning the amplitude of the Fourier-domain waveform as
            a function of frequency and parameters
        Psi: function returning the phase of the Fourier-domain waveform as a
            function of frequency and parameters
        fs: uniformly-spaced grid of frequencies used to compute the
            noise-weighted inner product
        Sn: power spectral density of the detector noise

    Return:
        The determinant of the metric over the intrinsic parameters of the
        binary.
    """
    return jnp.sqrt(jnp.linalg.det(get_g(theta, amp, Psi, fs, Sn)))


# @partial(jit, static_argnames=("amp", "Psi", "Sn"))
def get_metric_ellipse(
    theta: Array,
    amp: Callable[[Array, Array], Array],
    Psi: Callable[[Array, Array], Array],
    fs: Array,
    Sn: Callable[[Array], Array],
) -> Array:
    r"""
    Gets the ellipse representation of the metric.

    Warning:
        Only works in two dimensions.

    Args:
        theta: point at which to find metric ellipse
        amp: function returning the amplitude of the Fourier-domain waveform as
            a function of frequency and parameters
        Psi: function returning the phase of the Fourier-domain waveform as a
            function of frequency and parameters
        fs: uniformly-spaced grid of frequencies used to compute the
            noise-weighted inner product
        Sn: power spectral density of the detector noise

    Returns:
        Major and minor axes and orientation of the metric ellipse.
    """
    g = get_g(theta, amp, Psi, fs, Sn)
    eigval, norm_eigvec = jnp.linalg.eig(g)
    r_major, r_minor = 1 / jnp.sqrt(eigval)
    U = jnp.linalg.inv(norm_eigvec)
    ang = jnp.arccos(U[0, 0] / jnp.linalg.norm(U[:, 0]))

    return jnp.array([r_major, r_minor, ang])
