import os

import click
import jax
import jax.numpy as jnp
from genbank_3D_taylorf2reducedspin import sampler
from jax import random

from diffbank.bank import Bank
from diffbank.noise import Sn_aLIGOZeroDetHighPower as Sn
from diffbank.waveforms.taylorf2reducedspin import Psi, amp

"""
Update eta estimate for an existing bank using more points and resave.
"""


@click.command()
@click.option("--path", type=str, help="path to bank")
@click.option("--seed", type=int)
@click.option(
    "--n-eta", type=int, help="number of new points at which to compute effectualnesses"
)
@click.option("--device", default="cpu", help="device to run on")
def run(path, seed, n_eta, device):
    jax.config.update("jax_platform_name", device)

    kind = os.path.split(path)[-1].split("-")[0]
    if kind != "tf2rs":
        raise ValueError("wrong waveform model")

    bank = Bank.load(path, amp, Psi, Sn, sampler)
    key = random.PRNGKey(seed)
    bank.calc_bank_effectualness(key, n_eta)
    jnp.savez(
        f"banks/effs/{os.path.splitext(os.path.split(path)[-1])[0]}-effs-{seed}.npz",
        eff_pts=bank.effectualness_points,
        effs=bank.effectualnesses,
    )


if __name__ == "__main__":
    run()
