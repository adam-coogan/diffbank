from math import sqrt
from operator import itemgetter

import click
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm

from diffbank.bank import Bank
from diffbank.noise import Sn_LIGOI
from diffbank.utils import (
    gen_template_rejection,
    gen_templates_rejection,
    get_m1_m2_sampler,
)
from diffbank.waveforms.threePN_simple import Psi, amp

"""
Estimates variation of p over a 3.5PN bank.
"""


@click.command()
@click.option("--seed", default=1000, help="PRNG seed")
@click.option("--mm", default=0.95, help="minimum match")
@click.option("--n-pts", default=200, help="number of points at which to estimate p")
@click.option("--n-templates", default=10000, help="number of templates per point")
def run(seed, mm, n_pts, n_templates):
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

    gen_templates = jax.jit(
        lambda key: gen_templates_rejection(
            key,
            n_templates,
            bank.ratio_max,
            bank.density_fun,
            bank.sample_base,
            bank.density_fun_base,
        )
    )

    gen_template = jax.jit(
        lambda key: gen_template_rejection(
            key,
            bank.ratio_max,
            bank.density_fun,
            bank.sample_base,
            bank.density_fun_base,
        )
    )

    eff_fun = jax.jit(lambda template, ep: bank.match_fun(template, ep))

    # Generate points
    key, pt_keys = itemgetter(0, slice(1, -1))(random.split(key, n_pts + 1))
    eff_pts = jax.lax.map(gen_template, pt_keys)

    # Estimate p for each of them
    ps = []
    p_errs = []

    with tqdm(eff_pts) as pbar:
        for ep in pbar:
            key, template_key = random.split(key)
            templates = gen_templates(template_key)
            covereds = (
                jax.lax.map(lambda template: eff_fun(template, ep), templates) > mm
            )
            p, p_err = covereds.mean(), covereds.std() / sqrt(n_templates)
            ps.append(p)
            p_errs.append(p_err)
            pbar.set_postfix(dict(p=p, p_err=p_err))

    ps = jnp.array(ps)
    p_errs = jnp.array(p_errs)

    print("min, max, mean:", ps.min(), ps.max(), ps.mean())

    # Save results
    jnp.savez(
        f"threePN-est-p-variation-{mm}.npz", eff_pts=eff_pts, ps=ps, p_errs=p_errs
    )

    plt.scatter(*eff_pts.T, c=ps)
    plt.colorbar()
    plt.savefig(f"threePN-est-p-variation-{mm}.pdf")


if __name__ == "__main__":
    run()
