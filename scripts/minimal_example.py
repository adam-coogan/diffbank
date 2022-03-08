from jax import random
import jax.numpy as jnp

from diffbank.bank import Bank
from diffbank.noise import Sn_LIGOI as Sn
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

"""
Minimal example of bank generation. Creates a very coarse 3.5PN bank.
"""


# Set seed
key = random.PRNGKey(10)

# Set up generation
m_range = (2.5, 3.0)
sampler = get_m1_m2_sampler(m_range, m_range)
fs = jnp.linspace(24.0, 512.0, 4880)
minimum_match = 0.8
m_star = 1 - minimum_match
eta = 0.8
n_eff = 100
bank = Bank(amp, Psi, fs, Sn, m_star, eta, sampler, name=f"3pn-bank")

# Estimate max ratio between metric density and base density (required for rejection
# sampling from metric density)
key, subkey = random.split(key)
bank.ratio_max = bank.est_ratio_max(subkey)[0]

# Generate the bank
key, subkey = random.split(key)
bank.fill_bank(subkey, "random", n_eff)

# Estimate its effectualness
key, subkey = random.split(key)
bank.calc_bank_effectualness(subkey, 100)

# Save
bank.save()

# Example of loading bank
Bank.load("3pn-bank.npz", amp, Psi, Sn, sampler)
