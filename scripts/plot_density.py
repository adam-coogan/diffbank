from itertools import product
from math import log10
from operator import itemgetter

import click
import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import Voronoi, voronoi_plot_2d
from tqdm import tqdm

from diffbank.constants import MSUN
from diffbank.metric import get_density
from diffbank.noise import Sn_O3a as Sn
from diffbank.utils import ms_to_Mc_eta
from diffbank.waveforms.twoPN_chirptimes import (
    Amp,
    Psi,
    analytic_metric,
    get_th_boundary_interps,
    phys_to_th,
)

plt.style.use("../plot_style.mplstyle")

"""
Compares the analytic metric from arXiv:gr-qc/0604037 with the one computed with
automatic differentiation.

To reproduce the plot:

    >>> python plot_density.py

"""

##### Frequency settings
f_u = 512.0  # Hz
f_l = 10.0  # Hz
N_fbins = 1000
#####


@click.command()
@click.option("--n-m1s", type=int, default=50)
@click.option("--n-m2s", type=int, default=48)
@click.option("--fig-path", default="figures/density.pdf")
def run(n_m1s, n_m2s, fig_path):
    fs = jnp.linspace(f_l, f_u, N_fbins)

    density_fun = lambda theta: get_density(theta, Amp, Psi, fs, Sn)

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
    (th0_min, th0_max), th3_interp_low, th3_interp_high = get_th_boundary_interps(
        m_min, m_max, f_l
    )

    thetas = phys_to_th(jnp.stack([M_chirps, etas]), f_l).T  # type: ignore
    densities = jax.lax.map(density_fun, thetas)
    densities_analytic = []
    for i in tqdm(range(thetas.shape[0])):
        densities_analytic.append(
            jnp.sqrt(jnp.linalg.det(analytic_metric(fs, thetas[i], Sn)))
        )
    densities_analytic = jnp.array(densities_analytic)

    # Quantify difference in densities
    diffs = jnp.log10(jnp.abs((densities - densities_analytic) / densities_analytic))

    # Plot!
    th0_scale = 1e4
    th3_scale = 1e2
    thetas_scaled = thetas / jnp.array([th0_scale, th3_scale])
    vor = Voronoi(thetas_scaled)
    norm = colors.Normalize(-13.5, -12.5)
    cmap = cm.ScalarMappable(norm=norm)
    voronoi_plot_2d(vor, show_vertices=False, line_width=0, point_size=0)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=cmap.to_rgba(diffs[r]))

    plt.colorbar(
        cmap,
        label=r"$\log_{10} \left| \frac{\sqrt{|g_\mathrm{AD}|} - \sqrt{|g_\mathrm{ref}|}}{\sqrt{|g_\mathrm{ref}|}} \right|$",
    )

    # Mask outside boundaries
    th0_grid = jnp.linspace(th0_min, th0_max, 200)
    plt.fill_between(
        th0_grid / th0_scale,
        th3_interp_low(th0_grid) / th3_scale,
        jnp.full_like(th0_grid, -1e3) / th3_scale,
        where=jnp.full_like(th0_grid, True),
        color="w",
        zorder=1.5,
    )
    plt.fill_between(
        th0_grid / th0_scale,
        th3_interp_high(th0_grid) / th3_scale,
        jnp.full_like(th0_grid, 1e3) / th3_scale,
        where=jnp.full_like(th0_grid, True),
        color="w",
        zorder=1.5,
    )

    plt.xlabel(r"$\theta_0 / 10^{%i}$" % log10(th0_scale))
    plt.ylabel(r"$\theta_3 / 10^{%i}$" % log10(th3_scale))
    plt.xlim(0.1, 10)
    plt.ylim(1, 8)

    # plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    run()
