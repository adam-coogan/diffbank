"""
Detector noise power spectral densities.
"""
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import noise_resources
from .utils import Array


def Sn_LIGOI(f: Array) -> Array:
    r"""
    LIGO-I power spectral density.

    References:
        `<https://arxiv.org/abs/gr-qc/0010009>`_

    Args:
        f: frequency

    Returns:
        The noise power spectral density
    """
    fs = 40  # Hz
    f_theta = 150  # Hz
    x = f / f_theta
    normalization = 1e-46
    return jnp.where(
        f > fs,
        normalization
        * 9
        * ((4.49 * x) ** (-56) + 0.16 * x ** (-4.52) + 0.52 + 0.32 * x ** 2),
        jnp.inf,
    )


def _load_noise(name: str, asd: bool = False) -> Callable[[Array], Array]:
    r"""
    Loads noise curve from text data file into an interpolator. The file's
    columns must contain the frequencies and corresponding noise power spectral
    density or amplitude spectral density values.

    Args:
        name: name of data file in ``noise_resources`` without the `.dat`
            extension
        asd: ``True`` if the file contains ASD rather than PSD data

    Returns
        Interpolator for noise curve returning ``inf`` above and below the
        frequency range in the data file
    """
    path_context = pkg_resources.path(noise_resources, f"{name}.dat")
    with path_context as path:
        fs, Sns = np.loadtxt(path, unpack=True)

    if asd:
        Sns = Sns ** 2

    Sns[Sns == 0.0] = np.inf

    fs = jnp.array(fs)
    Sns = jnp.array(Sns)

    return jax.jit(lambda f: jnp.interp(f, fs, Sns, left=jnp.inf, right=jnp.inf))


Sn_aLIGOZeroDetHighPower = _load_noise("aLIGOZeroDetHighPower")
r"""The aLIGOZeroDetHighPower noise curve from pycbc.

References:
    ???

Args:
    f: frequency

Returns:
    The noise power spectral density
"""

Sn_O3a = _load_noise("O3a_Livingston_ASD", asd=True)
r"""The LIGO O3a Livingston noise curve.

References:
    ???

Args:
    f: frequency

Returns:
    The noise power spectral density
"""

Sn_O2 = _load_noise("O2_ASD", asd=True)
r"""The LIGO O2 noise curve.

References:
    `<https://github.com/jroulet/template_bank>`_

Args:
    f: frequency

Returns:
    The noise power spectral density
"""
