import os

import click
import jax
import jax.numpy as jnp
from jax import random

from diffbank.bank import Bank
from diffbank.waveforms import taylorF2

"""
Generate a TaylorF2 bank which can be compared with the BNS section of https://arxiv.org/abs/1904.01683

To reproduce the bank in the paper, run

    >>> python genbank_4D_taylorf2.py

"""

##### Frequency settings
f_u = 512.0  # Hz
f_l = 24.0  # Hz
N_fbins = 4880
#####

m1_range = (1.0001, 3.0)
m2_range = (1.0, 3.0)

chi1_range = (-0.99, 0.99)
chi2_range = (-0.99, 0.99)


def sampler(key, n):
    ms = random.uniform(
        key,
        minval=jnp.array([m1_range[0], m2_range[0]]),
        maxval=jnp.array([m1_range[1], m2_range[1]]),
        shape=(n, 2),
    )
    key, subkey = random.split(key)
    chi1s = random.uniform(
        subkey,
        minval=jnp.array(chi1_range[0]),
        maxval=jnp.array(chi1_range[1]),
        shape=(n, 1),
    )
    key, subkey = random.split(key)
    chi2s = random.uniform(
        subkey,
        minval=jnp.array(chi2_range[0]),
        maxval=jnp.array(chi2_range[1]),
        shape=(n, 1),
    )
    ms_correct = jnp.stack(
        [
            ms.max(axis=1),
            ms.min(axis=1),
        ]
    ).T
    return jnp.hstack((ms_correct, chi1s, chi2s))


@click.command()
@click.option("--seed", default=1, help="PRNG seed")
@click.option("--kind", default="random", help="kind of bank: 'random' or 'stochastic'")
@click.option(
    "--n-eta",
    default=1000,
    type=int,
    help="number of new points at which to compute effectualnesses",
)
@click.option("--mm", default=0.96, help="minimum match")
@click.option("--eta-star", default=0.9, help="eta*")
@click.option("--n-eff", default=1000)
@click.option("--savedir", default="banks", help="directory in which to save the bank")
@click.option("--device", default="cpu", help="device to run on")
@click.option(
    "--noise",
    default="interpolated",
    help="noise curve: 'analytic' (LIGO-I) or 'interpolated' (LIGO O2)",
)
def gen_4D_taylorf2bank(seed, kind, n_eta, mm, eta_star, n_eff, savedir, device, noise):
    jax.config.update("jax_platform_name", device)

    key = random.PRNGKey(seed)
    fs = jnp.linspace(f_l, f_u, N_fbins)
    if noise == "interpolated":
        from diffbank.noise import Sn_O2 as Sn
    elif noise == "analytic":
        from diffbank.noise import Sn_LIGOI as Sn
    else:
        raise ValueError("invalid 'noise' argument")

    bank = Bank(
        taylorF2.Amp,
        taylorF2.Psi,
        fs,
        Sn,
        1 - mm,
        eta_star,
        sampler,
        name=f"tf2-{kind}-{seed}-mm={mm}-eta_star={eta_star}-n_eff={n_eff}",
    )

    theta_max = jnp.array([m1_range[0], m2_range[0], chi1_range[0], chi2_range[1]])
    bank.ratio_max = bank.density_fun(theta_max)

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
    gen_4D_taylorf2bank()
