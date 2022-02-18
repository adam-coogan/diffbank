from math import pi

import jax.numpy as jnp

"""
3PN metric parametrized by the black hole masses (m1, m2).
"""


def Psi(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    m1, m2 = theta
    # Mt, eta = theta
    gt = 4.92549094830932e-6  # GN*Msun/c**3 in seconds
    Mt = m1 + m2
    eta = m1 * m2 / (m1 + m2) ** 2
    EulerGamma = 0.57721566490153286060
    vlso = 1.0 / jnp.sqrt(6.0)

    v = (pi * Mt * (f + 1e-100) * gt) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    eta2 = eta ** 2
    eta3 = eta ** 3

    # # ------------------------- Non spinning part of the waveform
    psi_NS_0PN = 1.0
    psi_NS_1PN = (3715.0 / 756.0 + 55.0 * eta / 9.0) * v2
    psi_NS_15PN = -16.0 * pi * v3
    psi_NS_2PN = (
        15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta2 / 72.0
    ) * v4
    psi_NS_25PN = (
        pi * (38645.0 / 756.0 - 65.0 * eta / 9.0) * (1 + 3.0 * jnp.log(v / vlso)) * v5
    )
    psi_NS_3PN = (
        (
            11583231236531.0 / 4694215680.0
            - 640.0 * pi ** 2 / 3.0
            - 6848.0 * EulerGamma / 21.0
        )
        + (2255.0 * pi ** 2 / 12.0 - 15737765635.0 / 3048192.0) * eta
        + 76055.0 * eta2 / 1728.0
        - 127825.0 * eta3 / 1296.0
        - 6848.0 * jnp.log(4.0 * v) / 21.0
    ) * v6
    psi_NS_35PN = (
        pi
        * (77096675.0 / 254016.0 + 378515.0 * eta / 1512.0 - 74045.0 * eta2 / 756.0)
        * v7
    )

    psi_NS = (
        psi_NS_0PN
        + psi_NS_1PN
        + psi_NS_15PN
        + psi_NS_2PN
        + psi_NS_25PN
        + psi_NS_3PN
        + psi_NS_35PN
    )

    return 3.0 / 128.0 / eta / v5 * (psi_NS)


def amp(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    m1, m2 = theta
    return f ** (-7 / 6)
