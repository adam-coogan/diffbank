import os
from operator import itemgetter

import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffbank.bank import Bank
from diffbank.noise import Sn_LIGOI as Sn
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

plt.style.use("../plot_style.mplstyle")

BANK_DIR = "../scripts/banks/"
START_SEED = 501
N_BANKS = 50
ETA = 0.9
N_EFFS = jnp.array(
    [
        10,
        11,
        13,
        15,
        17,
        20,
        23,
        26,
        30,
        35,
        40,
        47,
        54,
        62,
        71,
        82,
        95,
        109,
        126,
        145,
        167,
        193,
        222,
        255,
        294,
        339,
        390,
        449,
        517,
        596,
        686,
        790,
        910,
        1048,
        1206,
        1389,
        1599,
        1842,
        2120,
        2442,
        2811,
        3237,
        3727,
        4291,
        4941,
        5689,
        6551,
        7543,
        8685,
        10000,
    ]
)
sampler = get_m1_m2_sampler((1.0, 3.0), (1.0, 3.0))


def run():
    # Load
    ns = []
    eta_ests = []
    for i in range(N_BANKS):
        bank = Bank.load(
            os.path.join(
                BANK_DIR,
                f"3pn-random-{START_SEED + i}-mm=0.95-eta_star={ETA}-n_eff={N_EFFS[i]}.npz",
            ),
            amp,
            Psi,
            Sn,
            sampler,
        )
        ns.append(bank.n_templates)
        eta_ests.append(bank.eta_est)

    eta_est_errs = jnp.sqrt(ETA * (1 - ETA) / (N_EFFS - 1))

    n_s = Bank.load(
        os.path.join(
            BANK_DIR, "3pn-stochastic-1000-mm=0.95-eta_star=0.9-n_eff=500.npz"
        ),
        amp,
        Psi,
        Sn,
        sampler,
    ).n_templates

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axs[0]
    ax.scatter(N_EFFS, eta_ests)
    ax.axhline(ETA, color="k", linestyle="--")
    for n_sigma in [1, 2]:
        ax.fill_between(
            N_EFFS,
            ETA - n_sigma * eta_est_errs,
            ETA + n_sigma * eta_est_errs,
            color="k",
            alpha=0.1,
            linewidth=0,
        )
    ax.set_ylabel(r"$\eta$")
    ax.set_ylim(0.68, 1.0)

    ax = axs[1]
    ax.scatter(N_EFFS, ns)
    ax.axhline(n_s, color="C1")
    ax.text(1.3e3, 1500, "Stochastic", color="C1")
    ax.set_ylabel(r"$N_T$")
    ax.set_ylim(900)

    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel(r"$n_\mathrm{eff}$")
        ax.set_xlim(N_EFFS[0], N_EFFS[-1])

    fig.tight_layout()
    fig.savefig("figures/neff-scaling.pdf")


if __name__ == "__main__":
    run()
