import jax
import jax.numpy as jnp
from jax import random
from scipy.optimize import minimize_scalar

from diffbank.bank import Bank
from diffbank.noise import Sn_LIGOI
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

"""
Tests empirical supremum rejection sampling estimate of maximum target vs base
density ratio.
"""


def test():
    key = random.PRNGKey(90)

    # Setup
    minimum_match = 0.95
    m_star = 1 - minimum_match
    eta_star = 0.99
    fs = jnp.linspace(20.0, 2000.0, 1000)
    m_range = (1.4, 5.0)
    sampler = get_m1_m2_sampler(m_range, m_range)
    bank = Bank(amp, Psi, fs, Sn_LIGOI, m_star, eta_star, sampler)

    # Get max density
    fun = lambda m1: -bank.density_fun(jnp.array([m1, m_range[0]]))
    res = minimize_scalar(fun, bracket=m_range, bounds=m_range)
    assert res.success
    theta_dmax = jnp.array([res.x, m_range[0]])
    ratio_max = bank.density_fun(theta_dmax)
    bank.ratio_max = ratio_max

    key, subkey = random.split(key)
    ratio_max_est = bank.est_ratio_max(subkey)[0]
    assert abs(bank.ratio_max - ratio_max_est) / bank.ratio_max < 0.1


if __name__ == "__main__":
    test()
