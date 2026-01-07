"""Module to create signals expected in AGN observations."""

from __future__ import annotations

import abc
from functools import partial, cached_property, reduce
import operator
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from numpyro.distributions.distribution import Distribution
from numpyro.infer.util import seed, substitute, trace
from tinygp import GaussianProcess, kernels

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def sample_or_deterministic(
    name: str,
    value: Distribution | ArrayLike,
) -> ArrayLike:
    """Sample from a distribution or fix to a constant value.

    This function provides a unified interface for NumPyro models where
    parameters can be either sampled from a distribution or fixed to a
    constant value. The choice is made at model definition time based on
    the type of the `value` argument.

    Parameters
    ----------
    name : str
        The name for the NumPyro sample site or deterministic node.
    value : Distribution | ArrayLike
        If a NumPyro Distribution, samples from it using `numpyro.sample()`.
        If an array-like (including float, int, or array), registers it as a
        deterministic value using `numpyro.deterministic()`.

    Returns
    -------
    ArrayLike
        The sampled or fixed value as a JAX array.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> import jax.numpy as jnp
    >>> # Sampling from a distribution
    >>> with numpyro.handlers.seed(rng_seed=0):
    ...     x = sample_or_fix("x", dist.Normal(0, 1))
    >>> # Fixing to a constant
    >>> with numpyro.handlers.seed(rng_seed=0):
    ...     y = sample_or_fix("y", 5.0)

    Notes
    -----
    This function is JAX JIT-compliant because the isinstance() check
    occurs at model definition time (when Python code executes), not at
    JAX tracing time. The `value` argument is a Python object whose type
    is known before any JAX tracing begins.
    """
    if isinstance(value, Distribution):
        return numpyro.sample(name, value)
    else:
        return numpyro.deterministic(name, jnp.asarray(value))


class Signal(eqx.Module):
    """
    Abstract class for signal in AGN dataset.

    Attributes
    ----------
    name : str
        Name of the signal.

    Methods
    -------
    __call__(x: ArrayLike) -> ArrayLike
        Abstract method to call the Signal object.
    __add__(other: Signal) -> SignalCollection
        Add another Signal object to this one and return a SignalCollection.
    call_deterministic(*args: Any, **kwargs: Any) -> ArrayLike
        Abstract method to call with specific parameters.
    """

    name: str = eqx.field(static=True)
    signal_type: Literal["deterministic"] | Literal["gp"] | Literal["collection"] = eqx.field(
        kw_only=True, static=True, init=False
    )

    @abc.abstractmethod
    def __call__(self, x: ArrayLike, kwargs: dict[str, ArrayLike] | None = None) -> ArrayLike:
        """Abstract method to call the Signal object."""

    def __add__(self, other: Signal) -> SignalCollection:
        """
        Add another Signal object to this one.

        Parameters
        ----------
        other : Signal
            The Signal object to be added.

        Returns
        -------
        SignalCollection
            A new SignalCollection containing both signals.

        Raises
        ------
        ValueError
            If `other` is not a Signal object.
        """
        self = eqx.error_if(self, not isinstance(other, Signal), "Other is not Signal")

        signal_list = [self, other]

        return SignalCollection(signal_list)  # type: ignore[call-arg]

    @abc.abstractmethod
    def call_deterministic(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Abstract method to call with specific parameters."""


class SignalCollection(Signal):
    """
    A collection of Signal objects.

    Attributes
    ----------
    signals : list[Signal]
        List of Signal objects in the collection.
    signals_kwargs: dict[Signal.name, Signal.__dict__]
        Nested dictionary of Signal names and attributes (without `name` attribute)
    name : str
        Name of the SignalCollection.
    indices : ArrayLike
        Array of indices for the signals in the collection.

    Methods
    -------
    __call__(x: ArrayLike) -> ArrayLike
        Compute the sum of all signals in the collection.
    __add__(other: Signal | SignalCollection) -> SignalCollection
        Add another Signal or SignalCollection to this collection.
    call_deterministic(x: ArrayLike, signal_args_dict: dict[str, dict[str, ArrayLike]]) -> ArrayLike
        Compute the sum of all signals with deterministic parameters.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import matplotlib.pyplot as plt
    >>> from angst.signal import Sinusoid, SignalCollection
    >>> import numpyro
    >>> import numpyro.distributions as dist
    >>>
    >>> # Create individual Sinusoid signals
    >>> signal1 = Sinusoid(
    ...    amplitude=dist.Normal(1.0, 0.1),
    ...    period=dist.Normal(10.0, 0.5),
    ...    phase=dist.Uniform(0, 2*jnp.pi),
    ...    name="Signal 1")
    >>> signal2 = Sinusoid(
    ...    amplitude=dist.Normal(0.5, 0.05),
    ...    period=dist.Normal(5.0, 0.3),
    ...    phase=dist.Uniform(0, 2*jnp.pi),
    ...    name="Signal 2"
    >>> signal3 = Sinusoid(
    ...    amplitude=dist.Normal(0.3, 0.03),
    ...    period=dist.Normal(2.0, 0.1),
    ...    phase=dist.Uniform(0, 2*jnp.pi),
    ...    name="Signal 3")
    >>>
    >>> # Combine signals
    >>> combined_signal = signal1 + signal2 + signal3
    >>>
    >>> # Create a time array
    >>> x = jnp.linspace(0, 20, 200)
    >>>
    >>> # Set a random seed for reproducibility
    >>> key = jax.random.PRNGKey(0)
    >>>
    >>> # Evaluate the combined signal
    >>> with numpyro.handlers.seed(rng_seed=key):
    ...     result = combined_signal(t)
    >>>
    >>> # Plot the results
    >>> plt.figure(figsize=(12, 6))
    >>> plt.plot(x, result, label='Combined Signal')
    >>> with numpyro.handlers.seed(rng_seed=key):
    ...     plt.plot(t, signal1(x), label=signal1.name, linestyle='--')
    ...     plt.plot(t, signal2(x), label=signal2.name, linestyle='--')
    ...     plt.plot(t, signal3(x), label=signal3.name, linestyle='--')
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Amplitude')
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.show()

    Example Plot
    --------------
    ![](../../assets/signal-collection-example.png)
    """

    signals: tuple[Signal] = eqx.field(static=True,)
    deterministic_signals: tuple[Signal] = eqx.field(static=True, default_factory=())

    gp_signals: tuple[Signal] = eqx.field(static=True, default_factory=())

    name: str = eqx.field(default="SignalCollection", kw_only=True, static=True)
    signal_type: str = eqx.field(default="collection", kw_only=True, static=True, init=False)

    def __init__(self, signal_list: list[Signal]):
        self.signals = tuple(signal_list)
        self.gp_signals = tuple(s for s in signal_list if s.signal_type == "gp")
        self.deterministic_signals = tuple(s for s in signal_list if s.signal_type == "deterministic")

    @cached_property
    def params(self)->tuple[str]:
        return tuple(reduce(operator.add, [s.params for s in self.signals]))


    @partial(jax.jit, static_argnums=(0,))  # this sometimes a bit faster
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """
        Compute the sum of all signals in the collection with optional parameter modifications.

        Parameters
        ----------
        x : ArrayLike
            Input values (typically time points) for evaluating the signals.

        Returns
        -------
        ArrayLike
            Sum of all signals evaluated at x, with any specified parameter modifications applied.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from angst.signal import Sinusoid, SignalCollection
        >>> import numpyro.distributions as dist
        >>> signal1 = Sinusoid(
        ...    amplitude=dist.Normal(1.0, 0.1),
        ...    period=dist.Normal(10.0, 0.5),
        ...    phase=dist.Uniform(0, 2*jnp.pi),
        ...    name="Signal 1")
        >>> signal2 = Sinusoid(
        ...    amplitude=dist.Normal(0.5, 0.05),
        ...    period=dist.Normal(5.0, 0.3),
        ...    phase=dist.Uniform(0, 2*jnp.pi),
        ...    name="Signal 2")
        >>> combined_signal = signal1 + signal2
        >>> x = jnp.linspace(0, 20, 5)
        >>> key = jax.random.PRNGKey(0)
        >>> with numpyro.handlers.seed(rng_seed=key):
        ...     result = combined_signal(x)
        >>> print(result)
        [-1.0807154   0.67816347 -0.8854479   0.9691354  -0.6519508 ]
        """
        # I have tried clever vmap implementations, they are slower
        return jnp.array([signal(x) for signal in self.signals]).sum(axis=0)

    def __add__(self, other: Signal | SignalCollection) -> SignalCollection:
        """
        Add another Signal or SignalCollection to this collection.

        Parameters
        ----------
        other : Signal or SignalCollection
            The Signal or SignalCollection to be added.

        Returns
        -------
        SignalCollection
            A new SignalCollection containing all signals from both objects.

        Raises
        ------
        ValueError
            If `other` is not a Signal object.

        Examples
        --------
        >>> from angst.signal import Sinusoid, SignalCollection
        >>> import numpyro.distributions as dist
        >>> signal1 = Sinusoid(
        ...    amplitude=dist.Normal(1.0, 0.1),
        ...    period=dist.Normal(10.0, 0.5),
        ...    phase=dist.Uniform(0, 2*jnp.pi),
        ...    name="Signal 1")
        >>> signal2 = Sinusoid(
        ...    amplitude=dist.Normal(0.5, 0.05),
        ...    period=dist.Normal(5.0, 0.3),
        ...    phase=dist.Uniform(0, 2*jnp.pi),
        ...    name="Signal 2")
        >>> combined_signal = signal1 + signal2
        >>> signal3 = Sinusoid(
        ...    amplitude=dist.Normal(0.3, 0.03),
        ...    period=dist.Normal(2.0, 0.1),
        ...    phase=dist.Uniform(0, 2*jnp.pi),
        ...    name="Signal 3")
        >>> new_combined_signal = combined_signal + signal3
        >>> print(len(new_combined_signal.signals))
        3
        """
        self = eqx.error_if(self, not isinstance(other, Signal), "Other is not a Signal")

        if isinstance(other, SignalCollection):
            return SignalCollection([*self.signals, *other.signals])
        else:
            return SignalCollection([*self.signals, other])

    def call_deterministic(
        self, x: ArrayLike, kwargs: dict[str, dict[str, ArrayLike]]
    ) -> ArrayLike:
        """
        Compute the sum of all signals with deterministic parameters.

        Parameters
        ----------
        x : ArrayLike
            Input values (typically time points) for evaluating the signals.
        signal_args_dict : dict[str, dict[str, Any]]
            A dictionary of dictionaries containing parameter values for each signal.
            The outer dictionary keys are signal names, and the inner dictionary contains
            the parameter names and their values.

        Returns
        -------
        ArrayLike
        Sum of all signals evaluated at x with the specified parameters.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from angst.signal import Sinusoid, SignalCollection
        >>> import numpyro.distributions as dist
        >>> signal1 = Sinusoid(
        ...    amplitude=dist.Normal(1.0, 0.1),
        ...    period=dist.Normal(10.0, 0.5),
        ...    phase=dist.Uniform(0, 2*jnp.pi),
        ...    name="Signal 1")
        >>> signal2 = Sinusoid(
        ...    amplitude=dist.Normal(0.5, 0.05),
        ...    period=dist.Normal(5.0, 0.3),
        ...    phase=dist.Uniform(0, 2*jnp.pi),
        ...    name="Signal 2")
        >>> combined_signal = signal1 + signal2
        >>> x_smooth = jnp.linspace(0,20,50)
        >>> x = jnp.array([0.0, 3.0, 11.0, 18.0])
        >>> signal_args = {
        ...     "Signal 1": {"amplitude": 10.0, "period": 10.0, "phase": 0.0},
        ...     "Signal 2": {"amplitude": -5.0, "period": 5.0, "phase": jnp.pi/4}
        ... }
        >>> result = combined_signal.call_deterministic(x_smooth, signal_args)
        >>> # Compute individual signals
        >>> signal1_result = signal1.call_deterministic(x_smooth, **signal_args["Signal 1"])
        >>> signal2_result = signal2.call_deterministic(x_smooth, **signal_args["Signal 2"])
        >>>
        >>> # Plot the results
        >>> plt.figure(figsize=(12, 6))
        >>> plt.plot(x_smooth, result, label='Combined Signal', color='k', linewidth=2)
        >>> plt.plot(x_smooth, signal1_result, label='Signal 1', linestyle='--')
        >>> plt.plot(x_smooth, signal2_result, label='Signal 2', linestyle='--')
        >>> plt.xlabel('Time')
        >>> plt.ylabel('Amplitude')
        >>> plt.title('Combined Sinusoidal Signals')
        >>> plt.legend()
        >>> plt.grid(True)
        >>> plt.show() (1)
        >>> result = combined_signal.call_deterministic(x, signal_args)
        >>> print(result)
        [-3.5355341 14.449007   1.4228206 -4.5721197]

        Example Plot
        --------------
        ![](../../assets/signal-collection-deterministic-example.png)
        """
        return jnp.array([
            signal.call_deterministic(x, **kwargs.get(signal.name, {})) for signal in self.signals
        ]).sum(axis=0)


class Sinusoid(Signal):
    """
    A sinusoidal signal.

    Attributes
    ----------
    amplitude : Distribution | ArrayLike
        Amplitude of the sinusoid. If a Distribution, it will be sampled;
        if a constant (float, int, or array), it will be fixed to that value.
    period : Distribution | ArrayLike
        Period of the sinusoid. If a Distribution, it will be sampled;
        if a constant (float, int, or array), it will be fixed to that value.
    phase : Distribution | ArrayLike
        Phase of the sinusoid. If a Distribution, it will be sampled;
        if a constant (float, int, or array), it will be fixed to that value.
    name : str
        Name of the Sinusoid signal.

    Methods
    -------
    __call__(x: ArrayLike) -> ArrayLike
        Compute the sinusoidal signal at given `x` values.
    call_deterministic(x: ArrayLike, amplitude: float, period: float, phase: float) -> ArrayLike
        Compute the sinusoidal signal with deterministic parameters.

    Examples
    --------
    Fully probabilistic signal (all parameters sampled):

    >>> import jax.numpy as jnp
    >>> from angst.signal import Sinusoid
    >>> import numpyro.distributions as dist
    >>> signal = Sinusoid(
    ...    amplitude=dist.Normal(1.0, 0.1),
    ...    period=dist.Normal(2.0, 0.1),
    ...    phase=dist.Uniform(0, 2*jnp.pi))
    >>> x = jnp.array([0, 0.5, 1, 1.5, 2])
    >>> key = jax.random.PRNGKey(0)
    >>> with numpyro.handlers.seed(rng_seed=key):
    ...     result = signal(x)
    >>> print(result)
    [-0.80653304 -0.30025846  0.8350276   0.22101425 -0.85600185]

    Mixed signal (fixed period, sampled amplitude and phase):

    >>> signal = Sinusoid(
    ...    amplitude=dist.Normal(1.0, 0.1),
    ...    period=365.25,  # Fixed to Earth's orbital period
    ...    phase=dist.Uniform(0, 2*jnp.pi))
    """

    amplitude: Distribution | ArrayLike
    period: Distribution | ArrayLike
    phase: Distribution | ArrayLike
    name: str = eqx.field(default="Sinusoid", kw_only=True, static=True)
    signal_type: str = eqx.field(default="deterministic", kw_only=True, static=True, init=False)

    def __call__(
        self,
        x: ArrayLike,
    ) -> ArrayLike:
        """
        Compute the sinusoidal signal at given value(s) x.

        Parameters
        ----------
        x : ArrayLike
            Values to evaluate Sinusoid at.

        Returns
        -------
        ArrayLike
            Sinusoidal signal evaluated at x.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from angst.signal import Sinusoid
        >>> import numpyro.distributions as dist
        >>> signal = Sinusoid(
        ...    amplitude=dist.Normal(1.0, 0.1),
        ...    period=dist.Normal(2.0, 0.1),
        ...    phase=dist.Uniform(0, 2*jnp.pi))
        >>> x = jnp.array([0, 0.5, 1, 1.5, 2])
        >>> key = jax.random.PRNGKey(0)
        >>> with numpyro.handlers.seed(rng_seed=key):
        ...     result = signal(x)
        >>> print(result)
        [-0.80653304 -0.30025846  0.8350276   0.22101425 -0.85600185]
        """
        amp = sample_or_deterministic(f"{self.name}--amplitude", self.amplitude)
        per = sample_or_deterministic(f"{self.name}--period", self.period)
        pha = sample_or_deterministic(f"{self.name}--phase", self.phase)
        return amp * jnp.sin(2 * jnp.pi / per * x + pha)

    @cached_property
    def params(self) -> tuple[str]:
        return tuple(trace(self).get_trace(jnp.arange(2)).keys())

    @classmethod
    def call_deterministic(
        cls, x: ArrayLike, amplitude: float, period: float, phase: float
    ) -> ArrayLike:
        r"""
        Compute the sinusoidal signal with deterministic parameters.

        Parameters
        ----------
        x : ArrayLike
            Time(s) to evaluate at.
        amplitude : float
            Amplitude of the sinusoidal signal.
        period : float
            Period of the sinusoidal signal.
        phase : float
            Phase of the sinusoidal signal.

        Returns
        -------
        ArrayLike
            Sinusoidal signal evaluated at x with the specified parameters.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from angst.signal import Sinusoid
        >>> import numpyro.distributions as dist
        >>> signal = Sinusoid(
        ...    amplitude=dist.Normal(1.0, 0.1),
        ...    period=dist.Normal(2.0, 0.1),
        ...    phase=dist.Uniform(0, 2*jnp.pi))
        >>> x = jnp.array([0, 0.5, 1, 1.5, 2])
        >>> result = signal.call_deterministic(
        ...    x,
        ...    amplitude=1.0,
        ...    period=2.0,
        ...    phase=0.0)
        >>> # OR, just use class method
        >>> result_class = Sinusoid.call_deterministic(
        ...    x,
        ...    amplitude=1.0,
        ...    period=2.0,
        ...    phase=0.0)
        >>> print(result, "\n", result_class)
        [ 0.0000000e+00  1.0000000e+00 -8.7422777e-08 -1.0000000e+00 1.7484555e-07]
        [ 0.0000000e+00  1.0000000e+00 -8.7422777e-08 -1.0000000e+00 1.7484555e-07]
        """
        return amplitude * jnp.sin(2 * jnp.pi / period * x + phase)


class DampedRandomWalk(Signal):
    r"""Timeseries of a Damped Random Walk (DRW) realization.

    Attributes
    ----------
    amplitude : Distribution | ArrayLike
        Amplitude of the DRW. If a Distribution, it will be sampled;
        if a constant (float, int, or array), it will be fixed to that value.
    timescale : Distribution | ArrayLike
        Timescale of the DRW (usually called $\tau$). If a Distribution,
        it will be sampled; if a constant (float, int, or array), it will be fixed.
    cadence: int
        Observation cadence of the simulated light curve.
    snr: float
        The ratio between the variability (RMS) amplitude of the DRW and the median of the
        observation errors.
    mean_mag: float
        The mean apparent bolometric magnitude of the lightcurve.
    seed_num: int
        Seed number passed to [jax.random.key][] used for lightcurve realization.
    name : str
        Name of the DRW signal. Defaults to "DRW".

    Methods
    -------
    __call__(x: ArrayLike) -> ArrayLike
        Compute the DRW time-series at given `x` values.
    call_deterministic(x: ArrayLike, amplitude: float, timescale: float, mask: jax.Array) -> jax.Array
        Compute the DRW time-series with deterministic parameters.

    Examples
    --------
    Fully probabilistic signal (all parameters sampled):

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from angst.signal import DampedRandomWalk
    >>> from angst import utils
    >>> import numpyro.distributions as dist
    >>> import numpyro
    >>> rng_key = jax.random.key(0)
    >>> tspan, cadence, season = 365, 60, (90, 270)
    >>> timestamps = utils.make_timestamps(tspan, cadence, rng_key)
    >>> mask = utils.make_season_mask(timestamps, season)
    >>> timestamps_season = timestamps[mask]
    >>> signal = DampedRandomWalk(
    ...    amplitude=dist.Uniform(1e-3, 3),
    ...    timescale=dist.Normal(180, 60),
    ... )
    >>> with numpyro.handlers.seed(rng_seed=rng_key):
    ...     result = signal(timestamps_season)
    >>> print(result)
    [-0.09246159 -0.10435258 -0.18399351]

    Mixed signal (fixed timescale, sampled amplitude):

    >>> signal = DampedRandomWalk(
    ...    amplitude=dist.Uniform(1e-3, 3),
    ...    timescale=180.0,  # Fixed timescale
    ... )

    """

    amplitude: Distribution | ArrayLike
    timescale: Distribution | ArrayLike
    name: str = eqx.field(default="DRW", kw_only=True, static=True)
    signal_type: str = eqx.field(default="gp", kw_only=True, static=True, init=False)

    @property
    def kernel(self):
        amplitude = sample_or_deterministic(f"{self.name}--amplitude", self.amplitude)
        timescale = sample_or_deterministic(f"{self.name}--timescale", self.timescale)
        return kernels.quasisep.Exp(sigma=amplitude, scale=timescale)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Create a timeseries of a Damped Random Walk (DRW) realization.

        Parameters
        ----------
        x: ArrayLike
            The timestamps (MJD) to calculate the lightcurve at.

        Returns
        -------
        ArrayLike
            2D array with rows corresponding to observations and columns corresponding to
            MJDs, observed apparent bolometric magnitude, and error in the observation.


        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from angst.signal import DampedRandomWalk
        >>> from angst import utils
        >>> import numpyro.distributions as dist
        >>> import numpyro
        >>> rng_key = jax.random.key(0)
        >>> tspan, cadence, season = 365, 60, (90, 270)
        >>> timestamps = utils.make_timestamps(tspan, cadence, rng_key)
        >>> timestamps_season = timestamps[utils.make_season_mask(timestamps, season)]
        >>> signal = DampedRandomWalk(
        ...    amplitude=dist.Uniform(1e-3, 3),
        ...    timescale=dist.Normal(180, 60),
        ...    cadence=cadence,
        ...    snr=20,
        ...    mean_mag=16,
        ...    seed_num=0)
        >>> with numpyro.handlers.seed(rng_seed=rng_key):
        ...     result = signal(timestamps_season)
        >>> print(result)
        [[1.20334259e+02 1.63033981e+01 2.39270385e-02]
        [1.81337051e+02 1.59917555e+01 1.57495700e-02]
        [2.39832870e+02 1.56357098e+01 1.23292450e-02]]

        Notes
        -----
        One should use `angst.utils.make_timestamps` to create input timestamps.
        """
        kernel = self.kernel

        gp = GaussianProcess(kernel, x)

        return numpyro.sample("DRW--sample", gp.numpyro_dist())

    @cached_property
    def params(self) -> tuple[str]:
        return tuple(trace(seed(self, jax.random.key(0))).get_trace(jnp.arange(2)).keys())

    @classmethod
    def call_deterministic(
        cls, x: ArrayLike, amplitude: float, timescale: float, seed_num: int, n: int = 1
    ) -> ArrayLike:
        r"""Deterministically call the DRW signal.

        Parameters
        ----------
        x : ArrayLike
            The timestamps to evaluate the DRW at.
        amplitude : float
            Amplitude of the DRW.
        timescale : float
            Timescale of the DRW (usually called $\tau$).
        seed_num : int
            Seed number for reproducible GP realization.

        Returns
        -------
        ArrayLike
            1D array of DRW signal values at the given timestamps.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from angst.signal import SimpleDampedRandomWalk
        >>> x = jnp.arange(100.0)
        >>> result = SimpleDampedRandomWalk.call_deterministic(x, amplitude=2.0, timescale=30.0, seed_num=42)
        >>> print(result.shape)
        (100,)
        """
        kernel = kernels.quasisep.Exp(amplitude, timescale)
        gp = GaussianProcess(kernel, x)
        return gp.sample(jax.random.key(seed_num), (n,))


class DampedHarmonicOscillator(Signal):
    r"""Timeseries of a Damped Harmonic Oscillator (SHO) realization.

    This signal uses tinygp's quasisep.SHO kernel, which models a damped,
    driven simple harmonic oscillator. It is useful for modeling quasi-periodic
    variability in AGN light curves.

    The kernel takes the form:

    .. math::

        k(\tau) = \sigma^2\,\exp\left(-\frac{\omega\,\tau}{2\,Q}\right)
        \times \text{(oscillatory term depending on Q)}

    where the oscillatory behavior depends on the quality factor Q:
    - Q < 0.5: overdamped (no oscillation)
    - Q = 0.5: critically damped
    - Q > 0.5: underdamped (oscillatory)

    Attributes
    ----------
    omega : Distribution | ArrayLike
        The angular frequency parameter. If a Distribution, it will be sampled;
        if a constant, it will be fixed. Related to the characteristic period
        by P = 2π/ω.
    quality : Distribution | ArrayLike
        The quality factor Q. Controls damping behavior:
        Q > 0.5 produces oscillatory behavior, Q < 0.5 is overdamped.
    sigma : Distribution | ArrayLike
        The amplitude parameter. Defaults to 1.0.
    name : str
        Name of the SHO signal. Defaults to "SHO".

    Methods
    -------
    __call__(x: ArrayLike) -> ArrayLike
        Compute the SHO time-series at given `x` values.
    call_deterministic(x: ArrayLike, omega: float, quality: float, sigma: float, seed_num: int) -> ArrayLike
        Compute the SHO time-series with deterministic parameters.

    Examples
    --------
    Quasi-periodic signal with high quality factor:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from angst.signal import DampedHarmonicOscillator
    >>> import numpyro.distributions as dist
    >>> import numpyro
    >>> rng_key = jax.random.key(0)
    >>> x = jnp.linspace(0, 100, 50)
    >>> signal = DampedHarmonicOscillator(
    ...    omega=dist.Uniform(0.1, 1.0),
    ...    quality=dist.Uniform(1.0, 10.0),
    ...    sigma=dist.HalfNormal(0.5),
    ... )
    >>> with numpyro.handlers.seed(rng_seed=rng_key):
    ...     result = signal(x)

    Fixed period oscillation (period ~ 100 days):

    >>> import math
    >>> signal = DampedHarmonicOscillator(
    ...    omega=2 * math.pi / 100.0,  # Fixed angular frequency
    ...    quality=5.0,                 # Underdamped oscillation
    ...    sigma=dist.HalfNormal(0.3),
    ... )

    References
    ----------
    .. [1] Foreman-Mackey et al. (2017), https://arxiv.org/abs/1703.09710
    """

    omega: Distribution | ArrayLike
    quality: Distribution | ArrayLike
    sigma: Distribution | ArrayLike = 1.0
    name: str = eqx.field(default="DHO", kw_only=True, static=True)
    signal_type: Literal["gp"] = eqx.field(default="gp", kw_only=True, static=True, init=False)

    @property
    def kernel(self):
        omega = sample_or_deterministic(f"{self.name}--omega", self.omega)
        quality = sample_or_deterministic(f"{self.name}--quality", self.quality)
        sigma = sample_or_deterministic(f"{self.name}--sigma", self.sigma)
        return kernels.quasisep.SHO(omega=omega, quality=quality, sigma=sigma)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Create a timeseries of a Damped Harmonic Oscillator realization.

        Parameters
        ----------
        x : ArrayLike
            The timestamps to evaluate the SHO at.

        Returns
        -------
        ArrayLike
            1D array of SHO signal values at the given timestamps.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from angst.signal import DampedHarmonicOscillator
        >>> import numpyro.distributions as dist
        >>> import numpyro
        >>> rng_key = jax.random.key(0)
        >>> x = jnp.linspace(0, 100, 20)
        >>> signal = DampedHarmonicOscillator(
        ...    omega=0.1,
        ...    quality=5.0,
        ...    sigma=0.5,
        ... )
        >>> with numpyro.handlers.seed(rng_seed=rng_key):
        ...     result = signal(x)
        >>> print(result.shape)
        (20,)
        """
        kernel = self.kernel
        gp = GaussianProcess(kernel, x)
        return numpyro.sample(f"{self.name}--sample", gp.numpyro_dist())

    @cached_property
    def params(self) -> tuple[str, ...]:
        return tuple(trace(seed(self, jax.random.key(0))).get_trace(jnp.arange(2)).keys())

    @classmethod
    def call_deterministic(
        cls,
        x: ArrayLike,
        omega: float,
        quality: float,
        sigma: float,
        seed_num: int,
        n: int = 1,
    ) -> ArrayLike:
        r"""Deterministically call the SHO signal.

        Parameters
        ----------
        x : ArrayLike
            The timestamps to evaluate the SHO at.
        omega : float
            The angular frequency parameter.
        quality : float
            The quality factor Q.
        sigma : float
            The amplitude parameter.
        seed_num : int
            Seed number for reproducible GP realization.
        n : int, optional
            Number of samples to draw. Defaults to 1.

        Returns
        -------
        ArrayLike
            Array of SHO signal values at the given timestamps.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from angst.signal import DampedHarmonicOscillator
        >>> x = jnp.linspace(0, 100, 50)
        >>> result = DampedHarmonicOscillator.call_deterministic(
        ...     x, omega=0.1, quality=5.0, sigma=0.5, seed_num=42
        ... )
        >>> print(result.shape)
        (1, 50)
        """
        kernel = kernels.quasisep.SHO(omega=omega, quality=quality, sigma=sigma)
        gp = GaussianProcess(kernel, x)
        return gp.sample(jax.random.key(seed_num), (n,))


class CARMA(Signal):
    r"""Timeseries of a CARMA (Continuous-time AutoRegressive Moving Average) process.

    This signal uses tinygp's quasisep.CARMA kernel, which models a general
    continuous-time autoregressive moving average process. CARMA(p,q) processes
    are flexible stochastic models that can capture a wide range of variability
    patterns in AGN light curves.

    The power spectrum density (PSD) is:

    .. math::

        P(\omega) = \sigma^2\,\frac{|\sum_{q} \beta_q\,(i\,\omega)^q|^2}
                                   {|\sum_{p} \alpha_p\,(i\,\omega)^p|^2}

    Special cases:
    - CARMA(1,0) is equivalent to a Damped Random Walk (DRW/OU process)
    - CARMA(2,1) can model quasi-periodic oscillations similar to SHO

    Attributes
    ----------
    alpha : Distribution | ArrayLike
        The AR (autoregressive) coefficients, excluding the leading coefficient
        which is set to 1. Should be an array of length p. If a Distribution,
        it will be sampled; if a constant array, it will be fixed.
    beta : Distribution | ArrayLike
        The MA (moving average) coefficients multiplied by sigma. Should be an
        array of length q+1, where q+1 <= p. If a Distribution, it will be
        sampled; if a constant array, it will be fixed.
    name : str
        Name of the CARMA signal. Defaults to "CARMA".

    Methods
    -------
    __call__(x: ArrayLike) -> ArrayLike
        Compute the CARMA time-series at given `x` values.
    call_deterministic(x: ArrayLike, alpha: ArrayLike, beta: ArrayLike, seed_num: int) -> ArrayLike
        Compute the CARMA time-series with deterministic parameters.

    Examples
    --------
    CARMA(2,1) process (quasi-periodic):

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from angst.signal import CARMA
    >>> import numpyro.distributions as dist
    >>> import numpyro
    >>> rng_key = jax.random.key(0)
    >>> x = jnp.linspace(0, 100, 50)
    >>> # CARMA(2,1): alpha has 2 elements, beta has 2 elements
    >>> signal = CARMA(
    ...    alpha=jnp.array([0.1, 0.01]),  # AR coefficients
    ...    beta=jnp.array([1.0, 0.5]),    # MA coefficients * sigma
    ... )
    >>> with numpyro.handlers.seed(rng_seed=rng_key):
    ...     result = signal(x)

    CARMA(1,0) process (equivalent to DRW):

    >>> signal_drw = CARMA(
    ...    alpha=jnp.array([1/100.0]),  # 1/timescale
    ...    beta=jnp.array([0.5]),       # amplitude
    ... )

    Notes
    -----
    To construct a stationary CARMA kernel/process, the roots of the
    characteristic polynomials must have negative real parts. For simple
    cases like CARMA(1,0), CARMA(2,0), and CARMA(2,1), this is satisfied
    by using positive input parameters.

    References
    ----------
    .. [1] Kelly et al. (2014), https://arxiv.org/abs/1402.5978
    """

    alpha: Distribution | ArrayLike
    beta: Distribution | ArrayLike
    name: str = eqx.field(default="CARMA", kw_only=True, static=True)
    signal_type: Literal["gp"] = eqx.field(default="gp", kw_only=True, static=True, init=False)

    @property
    def kernel(self):
        alpha = sample_or_deterministic(f"{self.name}--alpha", self.alpha)
        beta = sample_or_deterministic(f"{self.name}--beta", self.beta)
        return kernels.quasisep.CARMA(alpha=alpha, beta=beta)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Create a timeseries of a CARMA process realization.

        Parameters
        ----------
        x : ArrayLike
            The timestamps to evaluate the CARMA process at.

        Returns
        -------
        ArrayLike
            1D array of CARMA signal values at the given timestamps.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from angst.signal import CARMA
        >>> import numpyro
        >>> rng_key = jax.random.key(0)
        >>> x = jnp.linspace(0, 100, 20)
        >>> signal = CARMA(
        ...    alpha=jnp.array([0.1]),
        ...    beta=jnp.array([0.5]),
        ... )
        >>> with numpyro.handlers.seed(rng_seed=rng_key):
        ...     result = signal(x)
        >>> print(result.shape)
        (20,)
        """
        kernel = self.kernel
        gp = GaussianProcess(kernel, x)
        return numpyro.sample(f"{self.name}--sample", gp.numpyro_dist())

    @cached_property
    def params(self) -> tuple[str, ...]:
        return tuple(trace(seed(self, jax.random.key(0))).get_trace(jnp.arange(2)).keys())

    @classmethod
    def call_deterministic(
        cls,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        seed_num: int,
        n: int = 1,
    ) -> ArrayLike:
        r"""Deterministically call the CARMA signal.

        Parameters
        ----------
        x : ArrayLike
            The timestamps to evaluate the CARMA process at.
        alpha : ArrayLike
            The AR coefficients (excluding leading 1). Array of length p.
        beta : ArrayLike
            The MA coefficients * sigma. Array of length q+1.
        seed_num : int
            Seed number for reproducible GP realization.
        n : int, optional
            Number of samples to draw. Defaults to 1.

        Returns
        -------
        ArrayLike
            Array of CARMA signal values at the given timestamps.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from angst.signal import CARMA
        >>> x = jnp.linspace(0, 100, 50)
        >>> result = CARMA.call_deterministic(
        ...     x,
        ...     alpha=jnp.array([0.1]),
        ...     beta=jnp.array([0.5]),
        ...     seed_num=42
        ... )
        >>> print(result.shape)
        (1, 50)
        """
        kernel = kernels.quasisep.CARMA(alpha=alpha, beta=beta)
        gp = GaussianProcess(kernel, x)
        return gp.sample(jax.random.key(seed_num), (n,))
