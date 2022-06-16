r"""
Defines ``Bank``, a class-based interface for managing template banks.
"""
import os
from typing import Callable, Optional, Set, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random

from .metric import get_density, get_g, get_gam
from .utils import (
    Array,
    PRNGKeyArray,
    est_ratio_max,
    gen_bank_random,
    gen_bank_stochastic,
    gen_template_rejection,
    gen_templates_rejection,
    get_bank_effectualness,
    get_eff_pads,
    get_match,
    get_top_matches_templates,
)


class Bank:
    r"""
    Class representing a template bank.
    """

    computed_vars: Set[str] = set(
        [
            "ratio_max",
            "n_templates",
            "templates",
            "effectualness_points",
            "effectualnesses",
            "eta_est",
            "eta_est_err",
            "_eff_pad_low",
            "_eff_pad_high",
            "_dim",
        ]
    )
    """
    The names of attributes that are computed in the course of working with a
    bank. For example, this includes ``"templates"``.
    """
    provided_vars: Set[str] = set(
        [
            "fs",
            "m_star",
            "eta",
            "name",
        ]
    )
    """
    The names of attributes containing data the user provides when initialing a
    bank
    """

    def __init__(
        self,
        amp: Callable[[Array, Array], Array],
        Psi: Callable[[Array, Array], Array],
        fs: Array,
        Sn: Callable[[Array], Array],
        m_star: Union[float, Array],
        eta: Union[float, Array],
        sample_base: Callable[[PRNGKeyArray, int], Array],
        density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
        name: str = "test",
    ):
        r"""
        Initializes a :class:`Bank` instance.

        Args:
            amp: function returning the amplitude of the Fourier-domain waveform
                as a function of frequency and parameters
            Psi: function returning the phase of the Fourier-domain waveform as
                a function of frequency and parameters
            fs: uniformly-spaced grid of frequencies used to compute the
                noise-weighted inner product or match
            Sn: power spectral density of the detector noise
            m_star: the target maximum mismatch for the bank (equal to
                ``1 - minimum_match``)
            eta: target fraction of parameter space to cover with the bank
            sample_base: ``jit``-able base sampler for rejection sampling from
                the metric density. Takes a key and number of samples as
                arguments.
            density_fun_base: ``jit``-able density function for ``sample_base``.
                Need not be normalized.
            name: an identifier for the bank.

        Raises:
            ValueError: if ``len(fs) <= 2`` or ``m_star`` or ``eta`` are outside
                the range (0, 1)
        """
        # Validation
        if len(fs) <= 2:
            # Required for padding to work
            raise ValueError("length of frequency array must be at least three")
        if m_star > 1 or m_star < 0:
            raise ValueError("m_star must be in (0, 1)")
        if eta > 1 or eta < 0:
            raise ValueError("eta must be in (0, 1)")

        self.amp = amp
        self.Psi = Psi
        self.fs = fs
        self.Sn = Sn
        self.m_star = m_star
        self.eta = eta
        self.sample_base = sample_base
        self.density_fun_base = density_fun_base
        self.name = name

        self.ratio_max: Optional[Array] = None
        self.n_templates: Optional[int] = None
        self.templates: Optional[Array] = None
        self.effectualness_points: Optional[Array] = None
        self.effectualnesses: Optional[Array] = None
        self.eta_est: Optional[Array] = None
        self.eta_est_err: Optional[Array] = None

        # Padding for accurate effectualness calculation
        # Length of padded array
        self._eff_pad_low, self._eff_pad_high = get_eff_pads(self.fs)

        # Key doesn't matter
        self._dim = self.sample_base(random.PRNGKey(1), 1).shape[-1]

    def __str__(self):
        r"""Gets a string representation of the bank."""
        return f"Bank(m_star={float(self.m_star)}, eta={float(self.eta)}, dim={self.dim}, name='{self.name}')"

    def __repr__(self):
        return str(self)  # for now

    @property
    def dim(self) -> int:
        r"""
        The dimensionality of the bank's parameter space.
        """
        return self._dim

    def match_fun(
        self,
        theta1: Array,
        theta2: Array,
        amp2: Optional[Callable[[Array, Array], Array]] = None,
        Psi2: Optional[Callable[[Array, Array], Array]] = None,
    ) -> Array:
        r"""
        Gets the match between a point from the bank's waveform model and a different
        point, potentially with a distinct waveform model. See :meth:`diffbank.utils.get_match`.

        Args:
            theta1: the point from the same waveform model as the bank
            theta2: the other point
            amp2: amplitude function for the ``theta2``. If not provided, it is
                assumed ``theta2`` has the same waveform model as the bank.
            Psi2: phase function for the ``theta2``. If not provided, it is
                assumed ``theta2`` has the same waveform model as the bank.

        Returns:
            The match between the waveforms for the two points.
        """
        return get_match(
            theta1,
            theta2,
            self.amp,
            self.Psi,
            self.amp if amp2 is None else amp2,
            self.Psi if Psi2 is None else Psi2,
            self.fs,
            self.Sn,
            self._eff_pad_low,
            self._eff_pad_high,
        )

    def density_fun(self, theta: Array) -> Array:
        """
        Computes the determinant of the metric over the intrinsic of the binary,
        maximized over the time and phase at coalescence.
        See :meth:`diffbank.metric.get_density`.
        """
        return get_density(theta, self.amp, self.Psi, self.fs, self.Sn)

    def g_fun(self, theta) -> Array:
        """
        Computes the metric for the intrinsic binary parameters of the binary,
        maximized over the time and phase at coalescence. See
        :meth:`diffbank.metric.get_density`.
        """
        return get_g(theta, self.amp, self.Psi, self.fs, self.Sn)

    def gam_fun(self, theta) -> Array:
        """
        Computes the metric for the time of coalescence and intrinsic parameters
        of the binary. See :meth:`diffbank.metric.get_gam`.
        """
        return get_gam(theta, self.amp, self.Psi, self.fs, self.Sn)

    def est_ratio_max(
        self,
        key: PRNGKeyArray,
        n_iter: int = 1000,
        n_init: int = 200,
        show_progress: bool = True,
    ) -> Tuple[Array, Array]:
        r"""
        Estimate maximum of the ratio of target to base density using empirical
        supremum rejection sampling. See :meth:`diffbank.utils.est_ratio_max`.
        """
        return est_ratio_max(
            key,
            self.density_fun,
            self.sample_base,
            self.density_fun_base,
            n_iter,
            n_init,
            show_progress,
        )

    def gen_template_rejection(self, key: PRNGKeyArray) -> Array:
        r"""
        Generates a single template using rejection sampling. See
        :meth:`diffbank.utils.gen_template_rejection`.

        Raises:
            RuntimeError: if ``ratio_max`` is not set
        """
        if self.ratio_max is None:
            raise RuntimeError(
                "must set bank's 'ratio_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )
        return gen_template_rejection(
            key,
            self.ratio_max,
            self.density_fun,
            self.sample_base,
            self.density_fun_base,
        )

    def gen_templates_rejection(self, key: PRNGKeyArray, n_templates) -> Array:
        r"""
        Generates multiple templates using rejection sampling. See
        :meth:`diffbank.utils.gen_template_rejection`.

        Raises:
            RuntimeError: if ``ratio_max`` is not set
        """
        if self.ratio_max is None:
            raise RuntimeError(
                "must set bank's 'ratio_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )
        return gen_templates_rejection(
            key, n_templates, self.ratio_max, self.density_fun, self.sample_base
        )

    def fill_bank(
        self,
        key: PRNGKeyArray,
        method="random",
        n_eff: int = 1000,
        show_progress: bool = True,
        save_interval: Optional[int] = 20,
        save_path: str = "",
    ):
        """
        Fills the bank with the required number of templates using the random or
        stochastic sampling methods from Coogan et al 2022. Updates the
        ``templates`` and ``n_templates`` attributes. See
        :meth:`diffbank.utils.gen_bank_random` and
        :meth:`diffbank.utils.gen_bank_stochastic`.

        Raises:
            RuntimeError: if ``ratio_max`` is not set
        """
        if self.ratio_max is None:
            raise RuntimeError(
                "must set bank's 'ratio_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )

        save_callback = lambda t, ep: jnp.savez(
            os.path.join(save_path, f"{self.name}-checkpoint.npz"),
            templates=t,
            eff_pts=ep,
        )

        if method == "random":
            self.templates, _ = gen_bank_random(
                key,
                1 - self.m_star,
                self.eta,
                self.match_fun,
                self.ratio_max,
                self.density_fun,
                self.sample_base,
                self.density_fun_base,
                n_eff=n_eff,
                show_progress=show_progress,
                callback_interval=save_interval,
                callback_fn=save_callback
            )
            self.n_templates = len(self.templates)
        elif method == "stochastic":
            propose_template = jax.jit(self.gen_template_rejection)
            self.templates, _ = gen_bank_stochastic(
                key,
                1 - self.m_star,
                self.eta,
                self.match_fun,
                propose_template,
                propose_template,
                n_eff=n_eff,
                show_progress=show_progress,
                callback_interval=save_interval,
                callback_fn=save_callback
            )
            self.n_templates = len(self.templates)

    def get_top_matches_templates(
        self,
        theta2s: Array,
        amp2: Optional[Callable[[Array, Array], Array]] = None,
        Psi2: Optional[Callable[[Array, Array], Array]] = None,
        n_top: int = 1,
        show_progress: bool = True,
    ) -> Tuple[Array, Array]:
        """
        Finds best-matching templates for a set of points for a waveform model distinct
        from the bank's. See :meth:`diffbank.utils.get_top_matches_templates`.

        Raises:
            RuntimeError: if the bank does not contain templates
        """
        if self.templates is None:
            raise RuntimeError("cannot calculate effectualness of an empty bank")

        match_fun = lambda template, point: self.match_fun(template, point, amp2, Psi2)
        return get_top_matches_templates(
            self.templates, match_fun, theta2s, n_top, show_progress
        )

    def calc_bank_effectualness(
        self, key: PRNGKeyArray, n: int, show_progress: bool = True
    ):
        """
        Computes effectualness of bank at points sampled from the metric
        density. See :meth:`diffbank.utils.get_bank_effectualness`.

        Raises:
            RuntimeError: if the bank does not contain templates
        """
        if self.templates is None:
            raise RuntimeError("cannot calculate effectualness of an empty bank")

        sample_eff_pt = jax.jit(self.gen_template_rejection)
        (
            self.effectualnesses,
            self.effectualness_points,
            self.eta_est,
            self.eta_est_err,
        ) = get_bank_effectualness(
            key,
            1 - self.m_star,
            self.templates,
            self.match_fun,
            sample_eff_pt,
            n,
            show_progress,
        )

    def save(self, path: str = ""):
        """
        Saves template bank non-function attributes to a ``npz`` file. The
        file name is the bank's name plus the ``.npz`` extension.

        Args:
            path: directory in which to save bank
        """
        d = {k: getattr(self, k) for k in self.provided_vars | self.computed_vars}
        jnp.savez(os.path.join(path, f"{self.name}.npz"), bank=d)

    @classmethod
    def load(
        cls,
        path: str,
        amp: Callable[[Array, Array], Array],
        Psi: Callable[[Array, Array], Array],
        Sn: Callable[[Array], Array],
        sample_base: Callable[[PRNGKeyArray, int], Array],
        ignore_key_errors: bool = False,
    ):
        """
        Loads template bank's non-function attributes from a ``npz`` file. The
        other required attributes must be provided as arguments to this
        function.

        Args:
            path: path to ``npz`` file generated with ``Bank.save``
            amp: function returning the amplitude of the Fourier-domain waveform
                as a function of frequency and parameters
            Psi: function returning the phase of the Fourier-domain waveform as
                a function of frequency and parameters
            Sn: power spectral density of the detector noise
            sample_base: ``jit``-able base sampler for rejection sampling from
                the metric density. Takes a key and number of samples as
                arguments.
            ignore_key_errors: if ``True``, will try to load the bank even if
                there are extra or missing keys in the ``npz`` file.

        Raises:
            ValueError: if the bank being loaded is missing or contains extra keys,
                and ``ignore_key_errors`` is ``False``. This should only happen
                when trying to load old banks after ``diffbank`` goes through major
                updates.

        Returns:
            A new ``Bank``
        """
        d = jnp.load(path, allow_pickle=True)["bank"].item()
        if d.keys() != cls.provided_vars | cls.computed_vars and not ignore_key_errors:
            raise ValueError("missing or extra keys in bank file")

        # Instantiate with provided variables and functions
        fn_kwargs = {
            "amp": amp,
            "Psi": Psi,
            "Sn": Sn,
            "sample_base": sample_base,
        }

        try:
            bank = cls(**{**fn_kwargs, **{name: d[name] for name in cls.provided_vars}})

            # Set computed variables
            for name in cls.computed_vars:
                setattr(bank, name, d[name])
        except KeyError as e:
            if not ignore_key_errors:
                raise e

        return bank  # type: ignore
