import os
from math import pi
from typing import Tuple

import click
import jax
import jax.numpy as jnp
from jax import random
from scipy.optimize import minimize_scalar

from diffbank.bank import Bank
from diffbank.constants import MSUN, C, G
from diffbank.utils import Array, PRNGKeyArray
from diffbank.waveforms.taylorf2reducedspin import Psi, amp, get_th_boundary_interps

"""
Generate a TaylorF2ReducedSpin bank for comparison with Ajith et al 2014,
https://arxiv.org/abs/1210.6666.

To reproduce the bank in the paper, run

    >>> python genbank_3D_taylorf2reducedspin.py

"""

##### Frequency settings
# Since the lowest BH mass for this bank is 1 * MSUN, need to go up to its ISCO
# frequency
f_u = 2200.0  # Hz
f_0 = f_l = 20.0  # Hz
df = 0.1
N_fbins = int((f_u - f_l) / df)
#####

m_range = (1 * MSUN, 20 * MSUN)
m_ns_thresh = 2 * MSUN
M_tot_max = m_range[0] + m_range[1]
chi_bh_max = 0.98
chi_ns_max = 0.4

th0_range, th3_interp_low, th3_interp_high = get_th_boundary_interps(*m_range, f_0)
# Figure out where th3 attains its maximum
def get_th3S_max(th0, th3):
    """
    Gets max value of th3S at a given `(th0, th3)` point. This computes the
    component masses, gets the corresponding `chi1`, `chi2` values, computes
    the max value `chi` can take and converts this to a max value for `th3S`.
    """
    M_chirp = 1 / (16 * pi * f_0) * (125 / (2 * th0 ** 3)) ** (1 / 5) * C ** 3 / G
    eta = (16 * pi ** 5 / 25 * th0 ** 2 / th3 ** 5) ** (1 / 3)
    q = (1 + jnp.sqrt(1 - 4 * eta) - 2 * eta) / (2 * eta)
    m2 = (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp
    m1 = q * m2
    delta = (m1 - m2) / (m1 + m2)
    chi1_max = jnp.where(m1 > m_ns_thresh, chi_bh_max, chi_ns_max)
    chi2_max = jnp.where(m2 > m_ns_thresh, chi_bh_max, chi_ns_max)
    chi_s_max = (chi1_max + chi2_max) / 2
    chi_a_max = (chi1_max - chi2_max) / 2
    chi_max = chi_s_max * (1 - 76 * eta / 113) + delta * chi_a_max
    th3S_max = 113 * th3 * chi_max / (48 * pi)
    return th3S_max


def get_M_tot(th0, th3):
    M_chirp = 1 / (16 * pi * f_0) * (125 / (2 * th0 ** 3)) ** (1 / 5) * C ** 3 / G
    eta = (16 * pi ** 5 / 25 * th0 ** 2 / th3 ** 5) ** (1 / 3)
    q = (1 + jnp.sqrt(1 - 4 * eta) - 2 * eta) / (2 * eta)
    m2 = (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp
    m1 = q * m2
    return m1 + m2


def is_in_bounds(theta: Array) -> Array:
    """
    Checks if a point is in bounds using the `th` values and total mass.
    """
    th0, th3, th3S = theta[..., 0], theta[..., 1], theta[..., 2]
    return jnp.logical_and(
        th3 > th3_interp_low(th0),
        jnp.logical_and(
            th3 < th3_interp_high(th0),
            jnp.logical_and(
                jnp.abs(th3S) < get_th3S_max(th0, th3), get_M_tot(th0, th3) < M_tot_max
            ),
        ),
    )


def base_sample_1(
    key: PRNGKeyArray,
    th0_range: Tuple[float, float],
    th3_range: Tuple[float, float],
    th3S_max: float,
) -> Array:
    """
    Sample uniformly over maximum parameter ranges.
    """
    return random.uniform(
        key,
        (3,),
        minval=jnp.array([th0_range[0], th3_range[0], -th3S_max]),
        maxval=jnp.array([th0_range[1], th3_range[1], th3S_max]),
    )


@jax.jit
def sample_1(
    key: PRNGKeyArray,
    th0_range: Tuple[float, float],
    th3_range: Tuple[float, float],
    th3S_max: float,
) -> Array:
    """
    Samples a single point with rejection sampling.
    """
    cond_fun = lambda val: jnp.logical_not(is_in_bounds(val[1]))

    def body_fun(val):
        key = val[0]
        key, subkey = random.split(key)
        return (key, base_sample_1(subkey, th0_range, th3_range, th3S_max))

    key, subkey = random.split(key)
    init_val = (key, base_sample_1(subkey, th0_range, th3_range, th3S_max))
    return jax.lax.while_loop(cond_fun, body_fun, init_val)[1]


def _sampler(
    key: PRNGKeyArray,
    n: int,
    th0_range: Tuple[float, float],
    th3_range: Tuple[float, float],
    th3S_max: float,
) -> Array:
    return jax.lax.map(
        lambda key: sample_1(key, th0_range, th3_range, th3S_max), random.split(key, n)
    )


# Define sampling bounds
bracket = (th0_range[0], 5e3)  # NOTE: need to change if m_range changes!
res = minimize_scalar(lambda th0: -th3_interp_high(th0), bracket, bracket)
assert res.success
th0_th3_max = res.x
th3_max = -res.fun
th3_range = (th3_interp_low(th0_range[0]), th3_max)
# Maximum value of th3
th3S_max = get_th3S_max(th0_th3_max, th3_max)


# Capture globals
def sampler(key: PRNGKeyArray, n: int) -> Array:
    return _sampler(key, n, th0_range, th3_range, th3S_max)


@click.command()
@click.option("--seed", default=1, help="PRNG seed")
@click.option("--kind", default="random", help="kind of bank: 'random' or 'stochastic'")
@click.option(
    "--n-eta",
    default=0,
    type=int,
    help="number of new points at which to compute effectualnesses",
)
@click.option(
    "--mm", default=0.95, help="minimum match, chosen to match arXiv:1210.6666"
)
@click.option("--eta-star", default=0.993, help="eta, chosen to match arXiv:1210.6666")
@click.option("--n-eff", default=1300)
@click.option("--savedir", default="banks", help="directory in which to save the bank")
@click.option("--device", default="cpu", help="device to run on")
@click.option(
    "--noise",
    default="interpolated",
    help="noise curve: 'analytic' (LIGO-I) or 'interpolated' (aLIGOZeroDetHighPower from pycbc)",
)
def gen_3D_tf2rs(seed, kind, n_eta, mm, eta_star, n_eff, savedir, device, noise):
    jax.config.update("jax_platform_name", device)

    key = random.PRNGKey(seed)
    m_star = 1 - mm
    fs = jnp.linspace(f_l, f_u, N_fbins)
    if noise == "interpolated":
        from diffbank.noise import Sn_aLIGOZeroDetHighPower as Sn
    elif noise == "analytic":
        from diffbank.noise import Sn_LIGOI as Sn
    else:
        raise ValueError("invalid 'noise' argument")

    bank = Bank(
        amp,
        Psi,
        fs,
        Sn,
        m_star,
        eta_star,
        sampler,
        name=f"tf2rs-{kind}-{seed}-mm={mm}-eta_star={eta_star}-n_eff={n_eff}",
    )

    # Get max density
    # NOTE: need to change if m_range changes!
    th0s = jnp.linspace(1.0001 * th0_range[0], 0.9999 * th0_range[1], 500)
    th3s = th3_interp_high(th0s) * 0.99999
    th3Ss = -get_th3S_max(th0s, th3s)
    boundary_densities = jax.lax.map(
        bank.density_fun, jnp.stack([th0s, th3s, th3Ss], -1)
    )
    bank.ratio_max = jnp.nanmax(boundary_densities)

    # Fill bank
    key, subkey = random.split(key)
    bank.fill_bank(subkey, kind, n_eff)
    bank.save(savedir)
    print(f"Saved bank to {os.path.join(savedir, bank.name + '.npz')}")

    # Get effectualnesses
    if n_eta > 0:
        key, subkey = random.split(key)
        bank.calc_bank_effectualness(subkey, n_eta)
        bank.save(savedir)
    else:
        print("Skipping effectualnesses calculation")


if __name__ == "__main__":
    gen_3D_tf2rs()
