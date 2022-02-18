import jax.numpy as jnp
from jax import random

from diffbank.bank import Bank
from diffbank.noise import Sn_LIGOI
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

"""
Make sure the bank's saving/loading, generation and effectualness calculation
functionality works.
"""


def test_save_load():
    """
    Test saving and loading.
    """
    amp = lambda f, t: None
    Psi = lambda f, t: None
    Sn = lambda f: None
    sample_base = lambda k, n: jnp.array([0.0, 0.9])
    bank = Bank(
        amp,
        Psi,
        jnp.array([0.0, 1.0, 2.0]),
        Sn,
        0.05,
        0.99,
        sample_base,
        name="save_test",
    )
    bank.ratio_max = jnp.array(100.0)
    bank.n_templates = jnp.array(2, dtype=int)
    bank.templates = jnp.array([[5.0, 4.0], [7.0, 79.0]])
    bank.effectualnesses = jnp.array([1.0, 0.2, 3.0, 6.0, 1.0, 45.0, 9.0])
    bank.effectualness_points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    bank.eta_est = 0.92
    bank.eta_est_err = 0.001
    bank.save()
    bank_str = str(bank)
    print(bank_str)

    loaded_bank = Bank.load("save_test.npz", amp, Psi, Sn, sample_base)
    loaded_bank_str = str(loaded_bank)
    print(loaded_bank_str)

    # Computed variables
    assert bank.ratio_max == loaded_bank.ratio_max
    assert bank.n_templates == loaded_bank.n_templates
    assert jnp.all(bank.templates == loaded_bank.templates)
    assert jnp.all(bank.effectualness_points == loaded_bank.effectualness_points)
    assert jnp.all(bank.effectualnesses == loaded_bank.effectualnesses)
    assert bank.eta_est == bank.eta_est
    assert bank.eta_est_err == bank.eta_est_err
    assert bank.dim == loaded_bank.dim
    # Provided variables
    assert jnp.all(bank.fs == loaded_bank.fs)
    assert bank.m_star == loaded_bank.m_star
    assert bank.eta == loaded_bank.eta
    assert bank.name == loaded_bank.name
    # Functions
    assert bank.amp is loaded_bank.amp
    assert bank.Psi is loaded_bank.Psi
    assert bank.Sn is loaded_bank.Sn
    assert bank.sample_base is loaded_bank.sample_base


def test_gen():
    """
    Make sure template bank generation works.
    """
    key = random.PRNGKey(84)

    fs = jnp.linspace(20.0, 2000.0, 300)
    m_range = (2.8, 3.0)
    sampler = get_m1_m2_sampler(m_range, m_range)
    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_LIGOI,
        m_star=1 - 0.8,
        eta=0.8,
        sample_base=sampler,
        name="3PN",
    )

    bank.ratio_max = bank.density_fun(jnp.array([m_range[1], m_range[0]]))

    for kind in ["random", "stochastic"]:
        print(f"Testing {kind} bank")
        key, subkey = random.split(key)
        bank.fill_bank(subkey, kind)

        # Make sure templates are in bounds
        for m1, m2 in bank.templates:
            assert m1 >= m_range[0] and m1 <= m_range[1]
            assert m2 >= m_range[0] and m2 <= m_range[1]

        key, subkey = random.split(key)
        bank.calc_bank_effectualness(subkey, 10)
        print(f"eta = {bank.eta_est:.3f} +/- {bank.eta_est_err:.3f}\n")


if __name__ == "__main__":
    test_save_load()
    test_gen()
