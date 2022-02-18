from math import pi

import jax
import jax.numpy as jnp

from ..constants import C, G
from ..utils import get_f_isco, ms_to_Mc_eta

"""
TaylorF2ReducedSpin waveform, in terms of the dimensionless chirp times (th0,
th3, th3S).

Reference: Ajith et al 2014, https://arxiv.org/abs/1210.6666
"""


def th_to_phys(th: jnp.ndarray, f_0=20.0) -> jnp.ndarray:
    """
    Converts (th0, th3, th3S) -> (M_chirp, eta, chi).
    """
    th0, th3, th3S = th
    M_chirp = 1 / (16 * pi * f_0) * (125 / (2 * th0 ** 3)) ** (1 / 5) * C ** 3 / G
    eta = (16 * pi ** 5 / 25 * th0 ** 2 / th3 ** 5) ** (1 / 3)
    chi = 48 * pi * th3S / (113 * th3)
    return jnp.stack([M_chirp, eta, chi])


def phys_to_th(phys: jnp.ndarray, f_0=20.0) -> jnp.ndarray:
    """
    Converts (M_chirp, eta, chi) -> (th0, th3, th3S).
    """
    _M_chirp, eta, chi = phys
    M_chirp = _M_chirp / (C ** 3 / G)
    th0 = 5 / (128 * (f_0 * M_chirp * pi) ** (5 / 3))
    th3 = pi ** (1 / 3) / (4 * f_0 ** (2 / 3) * M_chirp ** (2 / 3) * eta ** (3 / 5))
    th3S = (
        113
        * chi
        / (192 * f_0 ** (2 / 3) * M_chirp ** (2 / 3) * pi ** (2 / 3) * eta ** (3 / 5))
    )
    return jnp.stack([th0, th3, th3S])


def th_to_f_isco(theta, f_0=20.0):
    """
    Computes f_isco from (th0, th3, th3S).
    """
    M_chirp, eta, _ = th_to_phys(theta, f_0)
    M = M_chirp / eta ** (3 / 5)  # total mass
    return get_f_isco(M)


def get_th_boundary_interps(m_min, m_max, f_0=20.0, n=1000):
    """
    Given a range of BH masses, returns corresponding range of th0 and
    interpolators for the minimum and maximum corresponding values of th3.
    """
    chis = jnp.zeros((n,))  # value doesn't matter

    # Lower boundary
    ms = jnp.linspace(m_min, m_max, n)
    M_chirps = ms_to_Mc_eta(jnp.stack([ms, ms]))[0]
    etas = jnp.full_like(M_chirps, 0.25)
    th0_lows, th3_lows = phys_to_th(jnp.stack([M_chirps, etas, chis]), f_0)[:2, ::-1]
    interp_low = lambda th0: jnp.interp(
        th0, th0_lows, th3_lows, left=jnp.nan, right=jnp.nan
    )

    # Upper boundary
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([ms, jnp.full_like(ms, m_max)]))
    th0_highs_1, th3_highs_1 = phys_to_th(jnp.stack([M_chirps, etas, chis]), f_0)[
        :2, ::-1
    ]

    M_chirps, etas = ms_to_Mc_eta(jnp.stack([ms, jnp.full_like(ms, m_min)]))
    th0_highs_2, th3_highs_2 = phys_to_th(jnp.stack([M_chirps, etas, chis]), f_0)[
        :2, ::-1
    ]

    th0_highs = jnp.concatenate([th0_highs_1, th0_highs_2])
    th3_highs = jnp.concatenate([th3_highs_1, th3_highs_2])
    interp_high = lambda th0: jnp.interp(
        th0, th0_highs, th3_highs, left=jnp.nan, right=jnp.nan
    )

    th0_min = min(th0_lows.min(), th0_highs_1.min())
    th0_max = max(th0_lows.max(), th0_highs_2.max())

    return (th0_min, th0_max), interp_low, interp_high


@jax.jit
def amp(f: jnp.ndarray, theta: jnp.ndarray, f_0=20.0) -> jnp.ndarray:
    """
    Amplitude, truncated at the ISCO frequency for a BH with mass equal to the
    total mass of the system.
    """
    return jnp.where(
        f <= th_to_f_isco(jax.lax.stop_gradient(theta), f_0), f ** (-7 / 6), 0.0
    )


@jax.jit
def Psi(f: jnp.ndarray, theta: jnp.ndarray, f_0=20.0) -> jnp.ndarray:
    th0, th3, th3S = theta

    phi_0 = 0.0
    t_0 = 0.0

    psi_0 = 3 * th0 / 5
    psi_1 = 0.0
    psi_2 = 743 / 2016 * (25 / (2 * pi ** 2)) ** (1 / 3) * th0 ** (1 / 3) * th3 ** (
        2 / 3
    ) + 11 * pi * th0 / (12 * th3)
    psi_3 = -3 / 2 * (th3 - th3S)
    psi_4 = (
        (
            675
            * th3
            * th3S ** 2
            * (
                8 * 10 ** (2 / 3) * pi ** (7 / 3) * th0 ** (2 / 3)
                - 405 * 10 ** (1 / 3) * pi ** (2 / 3) * th3 ** (5 / 3)
            )
            / (
                4
                * th0 ** (1 / 3)
                * (
                    152 * 10 ** (1 / 3) * pi ** (5 / 3) * th0 ** (2 / 3)
                    - 565 * th3 ** (5 / 3)
                )
                ** 2
            )
        )
        + 15293365
        * 5 ** (1 / 3)
        * th3 ** (4 / 3)
        / (10838016 * 2 ** (2 / 3) * pi ** (4 / 3) * th0 ** (1 / 3))
        + 617 * pi ** 2 * th0 / (384 * th3 ** 2)
        + 5429 / 5376 * (25 * pi * th0 / (2 * th3)) ** (1 / 3)
    )
    psi_5 = (
        (
            140311625
            * pi
            * th3 ** (2 / 3)
            * th3S
            / (
                180348
                * (
                    565 * th3 ** (5 / 3)
                    - 152 * 10 ** (1 / 3) * pi ** (5 / 3) * th0 ** (2 / 3)
                )
            )
        )
        + (
            38645
            * (5 / pi) ** (2 / 3)
            * th3 ** (5 / 3)
            / (64512 * 2 ** (1 / 3) * th0 ** (2 / 3))
        )
        - (
            732985
            * 5 ** (2 / 3)
            * th3S
            * (th3 / th0) ** (2 / 3)
            / (455616 * 2 ** (1 / 3) * pi ** (2 / 3))
        )
        - 85 * pi * th3S / (152 * th3)
        - 65 * pi / 384
        + phi_0
    )
    psi_6 = (
        (
            15211
            * 5 ** (2 / 3)
            * pi ** (4 / 3)
            * th0 ** (1 / 3)
            / (73728 * 2 ** (1 / 3) * th3 ** (4 / 3))
        )
        - 25565 * pi ** 3 * th0 / (27648 * th3 ** 3)
        - 535 * jnp.euler_gamma * th3 ** 2 / (112 * pi ** 2 * th0)
        + (11583231236531 / (320458457088 * pi ** 2) - 25 / 8) * th3 ** 2 / th0
        - 535 * th3 ** 2 / (336 * pi ** 2 * th0) * jnp.log(10 * th3 / (pi * th0))
        + (
            2255 * 5 ** (1 / 3) * pi ** (5 / 3) / (1024 * 2 ** (2 / 3))
            - 15737765635 * (5 / pi) ** (1 / 3) / (260112384 * 2 ** (2 / 3))
        )
        * (th3 / th0) ** (1 / 3)
    )
    psi_7 = (
        385483375
        * 5 ** (1 / 3)
        * th3 ** (7 / 3)
        / (173408256 * 2 ** (2 / 3) * pi ** (4 / 3) * th0 ** (4 / 3))
        + 378515 * 5 ** (2 / 3) * (pi / 2) ** (1 / 3) / 516096 * (th3 / th0) ** (2 / 3)
        - 74045 * pi ** 2 / (129024 * th3)
    )
    psi_8 = 2 * pi * f_0 * t_0

    psi_5_L = psi_5 - phi_0  # TODO: does this impact phi_0 maximization???
    psi_6_L = -535 * th3 ** 2 / (336 * pi ** 2 * th0)

    powers = (jnp.arange(0, 9) - 5) / 3

    return (
        psi_0 * (f / f_0) ** powers[0]
        + psi_1 * (f / f_0) ** powers[1]
        + psi_2 * (f / f_0) ** powers[2]
        + psi_3 * (f / f_0) ** powers[3]
        + psi_4 * (f / f_0) ** powers[4]
        + (psi_5 + psi_5_L * jnp.log(f / f_0)) * (f / f_0) ** powers[5]
        + (psi_6 + psi_6_L * jnp.log(f / f_0)) * (f / f_0) ** powers[6]
        + psi_7 * (f / f_0) ** powers[7]
        + psi_8 * (f / f_0) ** powers[8]
    )
