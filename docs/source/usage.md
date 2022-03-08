# Usage

TL;DR: check out the [`minimal_example.py`](https://github.com/adam-coogan/diffbank/blob/main/scripts/minimal_example.py)
script.

There are two interfaces to `diffbank`: a set of bank generation functions and
the convenient {class}`diffbank.bank.Bank` class wrapper. The first interface is
more in keeping with `jax`'s functional approach, but we expect other users will
find the class interface more convenient. This page will show you how to use both.

Take a look at our [`genbank_*.py` scripts](https://github.com/adam-coogan/diffbank/tree/main/scripts)
for more examples.

## Waveform model and setup

Let's start by importing some `jax` modules and setting the random seed:

```python
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(526)
```

If you're unfamiliar with `jax` random number generation, take a look [here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers).
It's a bit different than what you may be used to from e.g. `numpy`.

We will use a simple 3.5PN waveform as the signal model, taking the component
black hole masses to lie between 1 and 3 solar masses.

```python
from diffbank.waveforms.threePN_simple import Psi, amp

m_range = (2.0, 3.0)
```

Our bank generation scheme samples from the parameter space using the metric density.
This is done using rejection sampling, which means the user must provide a base
distribution, defined in terms of a sampler over the parameter space. Our base
sampler draws uniformly from the `(m1, m2)` parameter space, with the restriction
`m1 > m2`. This can be done using a convenience function from {mod}`diffbank.utils`:

```python
from diffbank.utils import get_m1_m2_sampler

sampler = get_m1_m2_sampler(m_range, m_range)
```

Using a nonuniform base density necessitates defining its density function as well.
This density function need not be normalized. Here we will take it to be 1.

Next we define a noise model. We will use an analytic version of the LIGO-I noise
power spectral density:

```python
from diffbank.noise import Sn_LIGOI as Sn
```

We also need a frequency grid to use for match calculations, in units of Hz:

```python
fs = jnp.linspace(24.0, 512.0, 4880)
```

Lastly, bank generation requires setting the maximum mismatch `m_star`, target
parameter space coverage fraction `eta` and number of effectualness points to use
for convergence monitoring:

```python
minimum_match = 0.95
m_star = 1 - minimum_match
eta = 0.9
n_eff = 1000
```

## `Bank` class interface

Initializing a bank requires the definitions from above and a name:

```python
from diffbank.bank import Bank

bank = Bank(amp, Psi, fs, Sn, m_star, eta_star, sampler, name=f"3pn-bank")
```

Before we can generate the bank, we need to compute the maximum value of the ratio
between the metric density and the base sampler density. For this waveform model
we can easily find this point through numerical optimization, but it's more convenient
to estimate it with [empirical supremum rejection sampling](https://bookdown.org/rdpeng/advstatcomp/rejection-sampling.html#empirical-supremum-rejection-sampling).
The `est_ratio_max` method returns the estimated maximum ratio and the point at
which it was attained, so we need only keep the first return value:

```python
key, subkey = random.split(key)
bank.ratio_max = bank.est_ratio_max(subkey)[0]
```

Now we can fill our bank! Let's use the random bank generation scheme from our paper:

```python
key, subkey = random.split(key)
bank.fill_bank(subkey, "random", n_eff)
```

This will print a [`tqdm`](https://github.com/tqdm/tqdm) progress bar to monitor
bank generation and take a few minutes to run. Afterwards, you can check the templates'
positions in `bank.templates`.

We could generate a stochastic bank by instead passing `"stochastic"` as the second
argument to `fill_bank`, which would take much longer to generate.

To estimate the coverage of our bank at 1000 points sampled from the metric density,
we can run

```python
key, subkey = random.split(key)
bank.calc_bank_effectualness(subkey, 1000)
```

This will populate the attributes `effectualness`, `effectualness_points`, `eta_est`
and `eta_est_err` of `bank`. The first two are the effectualness and sampled points.
The second are the resulting Monte Carlo estimate (and associated error) of the
banks' coverage.

We can save the bank to the current working directory with

```python
bank.save()
```

and reload it with

```python
Bank.load("3pn-bank.npz", amp, Psi, Sn, sampler)
```

Note the variables you must provide when loading a bank. This is because we do not
use e.g. `pickle` to save function attributes.

## Functional interface

This interface requires a bit more manual setup. We must first set up the metric
density:

```python
from diffbank.metric import get_density

density_fun = lambda theta: get_density(theta, amp, Psi, fs, Sn)
```

A match function is also required to check whether effectualness points are covered.
This requires defining padding arrays that make the IFFT maximization over the difference
in time of coalescence for the two waveforms work correctly:

```python
from diffbank.utils import get_match

eff_pad_low, eff_pad_high = get_eff_pads(fs)
match_fun = lambda theta1, theta2: get_match(
    theta1, theta2, amp, Psi, amp, Psi, fs, Sn, eff_pad_low, eff_pad_high,
)
```

Finally we need to define the density of the base sampler, which we can set to 1
since it need not be normalized. Also, we need the ratio of the metric density to
this one:

```python
key, subkey = random.split(key)
density_fun_base = lambda _: jnp.array(1.0)
ratio_max = est_ratio_max(subkey, density_fun, sample_base, density_fun_base)
```

Then we can generate some templates:

```python
key, subkey = random.split(key)
templates, eff_pts = gen_bank_random(
    subkey,
    1 - m_star,
    eta,
    match_fun,
    ratio_max,
    density_fun,
    sample_base,
    density_fun_base
)
```

This is the function wrapped by `Bank.fill_bank`.

For stochastic bank generation, you must pass in a sampler that proposes templates.
This is because the choice of this sampler does not make a huge difference in bank
generation time. We can set up a rejection sampler to draw from the metric density
with

```python
from diffbank.utils import gen_template_rejection

gen_template = lambda key: gen_template_rejection(
    key, ratio_max, density_fun, sampler, density_fun_base
)
```

and then generate the bank:

```python
templates, eff_pts = gen_bank_stochastic(
    key, minimum_match, eta, match_fun, propose_template, eff_pt_sampler, n_eff
)
```

To do: explain how to check coverage.
