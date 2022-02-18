from itertools import product

import click
from diffjeom import get_ricci_scalar
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from diffbank.constants import MSUN
from diffbank.metric import get_g
from diffbank.noise import Sn_O3a as Sn
from diffbank.utils import ms_to_Mc_eta
from diffbank.waveforms.twoPN_chirptimes import (
    Amp,
    Psi,
    get_th_boundary_interps,
    phys_to_th,
)

plt.style.use("../plot_style.mplstyle")

"""
Plots the scalar curvature (Ricci scalar) for the metric for the waveform in
https://arxiv.org/abs/gr-qc/0604037.

To reproduce:

    >>> python plot_scalar_curvature.py

"""

##### Frequency settings
f_u = 512.0  # Hz
f_l = 10.0  # Hz
N_fbins = 1000
#####


@click.command()
@click.option("--n-m1s", type=int, default=200)
@click.option("--n-m2s", type=int, default=100)
@click.option("--n-th0s", type=int, default=200)
@click.option("--n-th3s", type=int, default=100)
@click.option("--fig-path", default="figures/scalar-curvature.pdf")
def run(n_m1s, n_m2s, n_th0s, n_th3s, fig_path):
    fs = jnp.linspace(f_l, f_u, N_fbins)

    g_fun = lambda theta: get_g(theta, Amp, Psi, fs, Sn)

    # Set parameter grid
    m_min = jnp.array(1.0) * MSUN
    m_max = jnp.array(20.0) * MSUN
    M_max = m_min + m_max
    m1s = jnp.geomspace(m_min, m_max, n_m1s)
    m2s = jnp.geomspace(m_min, m_max, n_m2s)
    m1s, m2s = jnp.array(list(product(m1s, m2s))).T
    m1s, m2s = m1s[m1s >= m2s], m2s[m1s >= m2s]  # remove redundant systems
    m1s, m2s = m1s[m1s + m2s <= M_max], m2s[m1s + m2s <= M_max]
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([m1s, m2s]))

    # (th0, th3) boundaries
    (th0_min, th0_max), bound_itp_low, bound_itp_high = get_th_boundary_interps(
        m_min, m_max, f_l
    )

    # Plotting configuration
    vmin = -10.0
    vmax = -7.5
    levels = jnp.linspace(vmin, vmax, 60)
    cbar_ticks = jnp.arange(vmin, vmax + 0.05, 0.5)

    thetas = phys_to_th(jnp.stack([M_chirps, etas]), f_l).T  # type: ignore
    Rss = jax.vmap(lambda x: get_ricci_scalar(x, g_fun))(thetas)

    # Plot!
    th0s = jnp.linspace(thetas[:, 0].min(), thetas[:, 0].max(), n_th0s)
    th3s = jnp.linspace(thetas[:, 1].min(), thetas[:, 1].max(), n_th3s)

    cs = plt.contourf(
        th0s / 1e4,
        th3s / 1e2,
        jnp.clip(
            griddata(
                thetas[:, :2],
                jnp.log10(jnp.abs(Rss)),
                jnp.stack(jnp.meshgrid(th0s, th3s)).reshape([2, -1]).T,
            ).reshape([len(th3s), len(th0s)]),
            vmin,
            vmax,
        ),
        levels=levels,
        cmap="viridis",
    )
    plt.colorbar(cs, label=r"$\log_{10}(|R|)$", ticks=cbar_ticks)

    # Mask outside boundaries
    th0_grid = jnp.linspace(th0_min, th0_max, 200)
    plt.fill_between(
        th0_grid / 1e4,
        bound_itp_low(th0_grid) / 1e2,
        jnp.full_like(th0_grid, -1e3) / 1e2,
        where=jnp.full_like(th0_grid, True),
        color="w",
    )
    plt.fill_between(
        th0_grid / 1e4,
        bound_itp_high(th0_grid) / 1e2,
        jnp.full_like(th0_grid, 1e3) / 1e2,
        where=jnp.full_like(th0_grid, True),
        color="w",
    )

    plt.xlabel(r"$\theta_0 / 10^4$")
    plt.ylabel(r"$\theta_3 / 10^2$")
    plt.xlim(0.0, 10.0)
    plt.ylim(0.8, 8)
    # plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    run()
