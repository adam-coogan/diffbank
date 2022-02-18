import os

import click
import jax
import jax.numpy as jnp
from jax import random
from scipy.optimize import minimize_scalar

from diffbank.bank import Bank
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

"""
Generates a 3.5PN-2D bank.

To generate two banks for the paper, use

    >>> python genbank_2D_threePN.py --seed 1 --kind random
    >>> python genbank_2D_threePN.py --seed 2 --kind stochastic

"""

##### Frequency settings
f_u = 512.0  # Hz
f_l = 24.0  # Hz
N_fbins = 4880
#####


@click.command()
@click.option("--seed", default=1, help="PRNG seed")
@click.option("--kind", default="random", help="kind of bank: 'random' or 'stochastic'")
@click.option(
    "--n-eta",
    default=1000,
    type=int,
    help="number of new points at which to compute effectualnesses",
)
@click.option("--mm", default=0.95, help="minimum match")
@click.option("--eta-star", default=0.9, help="eta*")
@click.option("--n-eff", default=1000)
@click.option("--savedir", default="banks", help="directory in which to save the bank")
@click.option("--device", default="cpu", help="device to run on")
@click.option(
    "--noise",
    default="interpolated",
    help="noise curve: 'analytic' (LIGO-I) or 'interpolated' (Livingston O3a)",
)
def gen_2D_threePN(seed, kind, n_eta, mm, eta_star, n_eff, savedir, device, noise):
    jax.config.update("jax_platform_name", device)

    key = random.PRNGKey(seed)
    m_star = 1 - mm

    fs = jnp.linspace(f_l, f_u, N_fbins)
    m_range = (1.0, 3.0)
    sampler = get_m1_m2_sampler(m_range, m_range)
    if noise == "interpolated":
        from diffbank.noise import Sn_O3a as Sn
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
        name=f"3pn-{kind}-{seed}-mm={mm}-eta_star={eta_star}-n_eff={n_eff}",
    )

    # Get max density
    fun = lambda m1: -bank.density_fun(jnp.array([m1, m_range[0]]))
    res = minimize_scalar(fun, bracket=m_range, bounds=m_range, method="bounded")
    assert res.success
    theta_dmax = jnp.array([res.x, m_range[0]])
    ratio_max = bank.density_fun(theta_dmax)
    bank.ratio_max = ratio_max

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
    gen_2D_threePN()
