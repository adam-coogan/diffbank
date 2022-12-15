import os

import jax
import jax.numpy as jnp
from jax import random
from scipy.optimize import minimize_scalar

from diffbank.bank import Bank
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

"""
Generates a small 3.5PN-2D bank to comprehensively test ``diffbank``.
"""

##### Frequency settings
f_u = 512.0  # Hz
f_l = 24.0  # Hz
N_fbins = 4880
seed = 10
#####


def test_checkpointing(
    kind="random",
    n_eta=10,
    mm=0.95,
    eta_star=0.9,
    n_eff=10,
    savedir="./",
    device="cpu",
    noise="analytic",
):
    jax.config.update("jax_platform_name", device)

    key = random.PRNGKey(seed)
    m_star = 1 - mm

    fs = jnp.linspace(f_l, f_u, N_fbins)
    m_range = (2.0, 3.0)
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
    bank.fill_bank(subkey, kind, n_eff, save_interval=5)
    bank.save(savedir)
    print(f"Saved bank to {os.path.join(savedir, bank.name + '.npz')}")

    file = jnp.load(
        f"3pn-{kind}-{seed}-mm={mm}-eta_star={eta_star}-n_eff={n_eff}-checkpoint.npz",
        allow_pickle=True,
    )
    temp = file["templates"]
    eff_points = file["eff_pts"]
    effs = file["eff"]
    key = file["key"]

    bank_checkpoint = Bank(
        amp,
        Psi,
        fs,
        Sn,
        m_star,
        eta_star,
        sampler,
        name=f"3pn-{kind}-{seed}-mm={mm}-eta_star={eta_star}-n_eff={n_eff}",
    )
    bank_checkpoint.templates = temp
    bank_checkpoint.n_templates = len(bank.templates)
    bank_checkpoint.effectualness_points = eff_points

    # Get max density
    fun = lambda m1: -bank_checkpoint.density_fun(jnp.array([m1, m_range[0]]))
    res = minimize_scalar(fun, bracket=m_range, bounds=m_range, method="bounded")
    assert res.success
    theta_dmax = jnp.array([res.x, m_range[0]])
    ratio_max = bank_checkpoint.density_fun(theta_dmax)
    bank_checkpoint.ratio_max = ratio_max

    # Fill bank from intermediate point
    bank_checkpoint.fill_bank(key, kind, n_eff, effs=effs)
    print("Both banks have the same size!")
    print(bank.templates.shape, bank_checkpoint.templates.shape)


if __name__ == "__main__":
    test_checkpointing()
