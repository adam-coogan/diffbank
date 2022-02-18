from math import pi

import jax.numpy as jnp

from ..constants import MSUN, C, G

"""
2PN metric parametrized by the black hole masses (Mt, eta).
"""


def Psi(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    M, eta = theta
    M *= MSUN
    x = pi * G * M * f / C ** 3  # dimensionless
    return (
        3
        / (128 * eta)
        * x ** (-5 / 3)
        * (
            1
            + 20 / 9 * (743 / 336 + 11 / 4 * eta) * x ** (2 / 3)
            - 16 * pi * x
            + 10
            * (3058673 / 1016064 + 5429 / 1008 * eta + 617 / 144 * eta ** 2)
            * x ** (4 / 3)
        )
    )


def ms_to_Meta(ms: jnp.ndarray) -> jnp.ndarray:
    m1, m2 = ms
    mu = m1 * m2 / (m1 + m2)  # reduced mass
    M = m1 + m2  # total mass
    eta = mu / M
    return jnp.array([M, eta])


# def Psi(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
#     return Psi_Meta(f, ms_to_Meta(theta))


def Amp(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    return f ** (-7 / 6)
