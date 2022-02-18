from math import sqrt

import click
import jax
import jax.numpy as jnp
from jax import random
from scipy.optimize import minimize_scalar

from diffbank.bank import Bank
from diffbank.noise import Sn_LIGOI
from diffbank.utils import gen_templates_rejection, get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

"""
Computes an MC estimate (with error bars) for the covering probability `p` for
the 3.5PN-2D bank.
"""


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
@click.option("--mm", default=0.95, help="minimum match")
@click.option("--n", default=10000, help="number of points for MC estimate of p")
def run(seed, mm, n):
    key = random.PRNGKey(seed)
    m_star = 1 - mm

    fs = jnp.linspace(20.0, 2000.0, 1000)
    m_range = (1.4, 5.0)
    sampler = get_m1_m2_sampler(m_range, m_range)

    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_LIGOI,
        m_star,
        0.9,
        sampler,
        name=f"3pn-???-mm={mm}-eta_star=???-n_eff=???",
    )

    # Get max density
    fun = lambda m1: -bank.density_fun(jnp.array([m1, m_range[0]]))
    res = minimize_scalar(fun, bracket=m_range, bounds=m_range)
    assert res.success
    theta_dmax = jnp.array([res.x, m_range[0]])
    ratio_max = bank.density_fun(theta_dmax)
    bank.ratio_max = ratio_max

    # Estimate p
    gen_templates = jax.jit(
        lambda key: gen_templates_rejection(
            key,
            n,
            bank.ratio_max,
            bank.density_fun,
            bank.sample_base,
            bank.density_fun_base,
        )
    )

    key, template_key, eff_pt_key = random.split(key, 3)
    templates = gen_templates(template_key)
    eff_pts = gen_templates(eff_pt_key)

    covereds = (
        jax.lax.map(
            jax.jit(lambda tep: bank.match_fun(tep["t"], tep["ep"])),
            {"t": templates, "ep": eff_pts},
        )
        > mm
    )

    p, p_err = covereds.mean(), covereds.std() / sqrt(len(covereds))
    print(f"Bank: {bank.name}")
    print(f"\tp = {p} +/- {p_err}")


if __name__ == "__main__":
    run()
