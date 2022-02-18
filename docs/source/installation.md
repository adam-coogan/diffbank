# Installation

`diffbank` requires python 3.6 or newer. `diffbank` is not yet on PyPI. For now,
to install, clone the repository and run `pip`:

```bash
git clone https://github.com/adam-coogan/diffbank
cd diffbank
pip install .
```

This will install the dependencies: `jax`, `numpy` and `tqdm`. The scripts, docs,
tests and notebooks have additional dependencies. These can be installed by running

```bash
pip install -r requirements.txt
```

from the root of the repository. You'll also need [`git-lfs`](https://git-lfs.github.com/)
if you want to download the banks in the repository.
