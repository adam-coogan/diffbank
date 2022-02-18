from math import pi

import jax.numpy as jnp
import numpy as np

from ..constants import MSUN, C, G
from ..utils import ms_to_Mc_eta

"""
2PN metric parametrized by the chirp times from https://arxiv.org/pdf/gr-qc/0604037.pdf.
"""
f0 = 20  # Hz


def Psi(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    theta1, theta2 = theta
    x = f / f0

    a01 = 3 / 5
    a21 = 11 * pi / 12
    a22 = (743 / 2016) * (25 / 2 / (pi ** 2)) ** (1 / 3)
    a31 = -3 / 2
    a41 = 617 * (pi ** 2) / 384
    a42 = (5429 / 5376) * (25 * pi / 2) ** (1 / 3)
    a43 = (15293365 / 10838016) * (5 / 4 / (pi ** 4)) ** (1 / 3)

    psi1 = (a01 * theta1) * (x ** (-5 / 3))
    psi2 = (a21 * theta1 / theta2 + a22 * (theta1 * (theta2 ** 2)) ** (1 / 3)) * (
        x ** (-1)
    )
    psi3 = (a31 * theta2) * (x ** (-2 / 3))
    psi4 = (
        a41 * theta1 / (theta2 ** 2)
        + a42 * (theta1 / theta2) ** (1 / 3)
        + a43 * ((theta2 ** 4) / theta1) ** (1 / 3)
    ) * (x ** (-1 / 3))

    psi = psi1 + psi2 + psi3 + psi4

    return psi


def Amp(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    return f ** (-7 / 6)


def phys_to_th(phys: jnp.ndarray, f_L) -> jnp.ndarray:
    """
    Converts (M, eta) -> (theta1, theta2).
    """
    _M_chirp, eta = phys
    M_chirp = _M_chirp / (C ** 3 / G)
    th0 = 5 / (128 * (f_L * M_chirp * pi) ** (5 / 3))
    th3 = pi ** (1 / 3) / (4 * f_L ** (2 / 3) * M_chirp ** (2 / 3) * eta ** (3 / 5))
    return jnp.stack([th0, th3])


def get_th_boundary_interps(m_min, m_max, f_L, n=200):
    """
    Given a range of BH masses, returns corresponding range of th0 and
    interpolators for the minimum and maximum corresponding values of th3.
    """

    # Lower boundary
    ms = jnp.linspace(m_min, m_max, n)
    M_chirps = ms_to_Mc_eta(jnp.stack([ms, ms]))[0]
    etas = jnp.full_like(M_chirps, 0.25)
    th0_lows, th3_lows = phys_to_th(jnp.stack([M_chirps, etas]), f_L)[:2, ::-1]
    interp_low = lambda th0: jnp.interp(
        th0, th0_lows, th3_lows, left=jnp.nan, right=jnp.nan
    )

    # Upper boundary
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([ms, jnp.full_like(ms, m_max)]))
    th0_highs_1, th3_highs_1 = phys_to_th(jnp.stack([M_chirps, etas]), f_L)[:2, ::-1]

    M_chirps, etas = ms_to_Mc_eta(jnp.stack([ms, jnp.full_like(ms, m_min)]))
    th0_highs_2, th3_highs_2 = phys_to_th(jnp.stack([M_chirps, etas]), f_L)[:2, ::-1]

    th0_highs = jnp.concatenate([th0_highs_1, th0_highs_2])
    th3_highs = jnp.concatenate([th3_highs_1, th3_highs_2])
    interp_high = lambda th0: jnp.interp(
        th0, th0_highs, th3_highs, left=jnp.nan, right=jnp.nan
    )

    th0_min = min(th0_lows.min(), th0_highs_1.min())
    th0_max = max(th0_lows.max(), th0_highs_2.max())

    return (th0_min, th0_max), interp_low, interp_high


##############################################################
####### Functions for the analytic metric
##############################################################


def calc_Iq(q, fs, Sn):
    Sns = Sn(fs)
    xs = fs / f0
    Sf0 = Sn(f0)
    Iq = np.trapz(Sf0 * (xs ** (-q / 3) / Sns), xs)
    return Iq


def calc_Jq(q, fs, Sn):
    I7 = calc_Iq(7, fs, Sn)
    Iq = calc_Iq(q, fs, Sn)
    return Iq / I7


def analytic_metric(f, theta, Sn):
    theta1, theta2 = theta

    a01 = 3 / 5
    a21 = 11 * pi / 12
    a22 = (743 / 2016) * (25 / 2 / (pi ** 2)) ** (1 / 3)
    a31 = -3 / 2
    a41 = 617 * (pi ** 2) / 384
    a42 = (5429 / 5376) * (25 * pi / 2) ** (1 / 3)
    a43 = (15293365 / 10838016) * (5 / 4 / (pi ** 4)) ** (1 / 3)

    Psi = np.array(
        [
            [
                a01,
                0,
                a21 / theta2 + (a22 / 3) * (theta2 / theta1) ** (2 / 3),
                0,
                (a41 / (theta2 ** 2))
                + a42 / (3 * ((theta1 ** 2) * theta2) ** (1 / 3))
                - (a43 / 3) * (theta2 / theta1) ** (4 / 3),
            ],
            [
                0,
                0,
                -a21 * theta1 / (theta2 ** 2)
                + (2 * a22 / 3) * ((theta1 / theta2) ** (1 / 3)),
                a31,
                -2 * a41 * theta1 / (theta2 ** 3)
                - (a42 / 3) * (theta1 / (theta2 ** 4)) ** (1 / 3)
                + (4 * a43 / 3) * (theta2 / theta1) ** (1 / 3),
            ],
        ]
    )

    def J_combined(k, j):
        J_comb = calc_Jq(17 - k - j, f, Sn)
        J_comb -= calc_Jq(12 - k, f, Sn) * calc_Jq(12 - j, f, Sn)
        J_comb -= (
            (calc_Jq(9 - k, f, Sn) - calc_Jq(4, f, Sn) * calc_Jq(12 - k, f, Sn))
            * (calc_Jq(9 - j, f, Sn) - calc_Jq(4, f, Sn) * calc_Jq(12 - j, f, Sn))
        ) / (calc_Jq(1, f, Sn) - (calc_Jq(4, f, Sn) ** 2))
        return J_comb

    def fill_g(m, l):
        gml = 0.0
        for k in range(5):
            for j in range(5):
                gml += Psi[m, k] * Psi[l, j] * J_combined(k, j)
        return gml * (1 / 2)

    g = np.zeros((2, 2))
    g[0, 0] = fill_g(0, 0)
    g[0, 1] = fill_g(0, 1)
    g[1, 0] = fill_g(1, 0)
    g[1, 1] = fill_g(1, 1)
    return g
