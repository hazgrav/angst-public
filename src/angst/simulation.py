"""Module for AGN Dataset simulations."""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import nested_pandas as npd
import numpy as np
import numpyro
import pandas as pd
from numpyro.infer.util import seed, trace

from angst import signal, utils
from angst.dataset import DataSet
from angst.signal import Signal, SignalCollection

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from jax.typing import ArrayLike
    from numpyro.distributions.distribution import Distribution

def _signal_converter(signals: Signal | SignalCollection) -> SignalCollection:
        if isinstance(signals, Signal) and not isinstance(signals, SignalCollection):
            return SignalCollection([signals])
        else:
            return signals

def time_domain_sim(
    signals: Signal | SignalCollection,
    snr: float,
    t: ArrayLike,
    mean_value: float,
    signals_kwargs: dict[str, Any] | None = None,
    use_magnitudes: bool = True,
    rng_seed=117,
):
    """
    Simulate GP time series with heteroscedastic measurement errors.

    Generates light curve realizations from a Gaussian Process and assigns
    heteroscedastic (brightness-dependent) measurement errors drawn from a
    LogNormal distribution. This mimics shot noise behavior where brighter
    sources have smaller relative errors.

    Parameters
    ----------
    kernel : Kernel
        A tinygp kernel that provides a `get_rms_amp()` method (e.g., `DRW`).
    snr : float
        Signal-to-noise ratio, defined as the ratio between the kernel's RMS
        amplitude and the median of the measurement errors.
    t : ArrayLike
        Time stamps at which to simulate the light curve.
    n_lc : int, optional
        Number of light curves to simulate. Defaults to 1.
    log_flux : bool, optional
        Whether values are in astronomical magnitudes. If True, larger values
        (dimmer) get larger errors. If False (linear flux), smaller values
        (dimmer) get larger errors. Defaults to True.

    Returns
    -------
    t : jax.Array
        Time stamps, shape `(n_lc, len(t))` if `n_lc > 1`, else `(len(t),)`.
    y : jax.Array
        Simulated light curve values (no measurement noise added).
    yerr : jax.Array
        Heteroscedastic measurement errors drawn from LogNormal distribution.

    Raises
    ------
    TypeError
        If the kernel does not provide a `get_rms_amp()` method.

    Notes
    -----
    This function uses `numpyro.sample` internally, so it must be called within
    a NumPyro context (e.g., `numpyro.handlers.seed`) for reproducibility.

    The returned `y` values are pure GP samples without measurement noise.
    To simulate noisy observations, add `y_obs = y + Normal(0, yerr)`.
    """
    if not isinstance(signals, SignalCollection) and isinstance(signals, Signal):
        signals = SignalCollection([signals])

    t = jnp.atleast_1d(t)

    rng_key = jax.random.key(rng_seed)

    rng_key, err_key = jax.random.split(rng_key)

    seeded_signals = seed(signals, rng_key)
    y = substitute(seeded_signals, signals_kwargs)(t) if signals_kwargs else seeded_signals(t)
    noise = y.std() / snr

    yerr = jax.random.lognormal(err_key, sigma=0.25, shape=(t.size,)) * noise
    yerr = yerr[jnp.argsort(jnp.abs(yerr))]  # small->large

    if use_magnitudes:
        # ascending sort
        # large mag, large error
        y_rank = y.argsort().argsort()
        yerr = yerr[y_rank]
    else:
        # descending sort
        # large flux, small error
        y_rank = (-y).argsort().argsort()
        yerr = yerr[y_rank]

    return t, y + mean_value, yerr


class Simulation(eqx.Module):
    """Abstract simulation class."""

    signals: signal.Signal  # Can be single signal or collection
    name: str = eqx.field(default="Simulation", kw_only=True, static=True)

    @abc.abstractmethod
    def simulate(self, out_file: Path, *args: Any, **kwargs: Any) -> None:
        """Abstract simulate method."""


class TimeDomainSim(Simulation):
    r"""Damped Random Walk (DRW) Simulated dataset.

    Simulates an AGN dataset with realizations of a DRW process.

    Parameters
    ----------
    amplitude : Distribution
        Amplitude of the DRW.
    timescale : Distribution
        Timescale of the DRW (usually called $\tau$).
    cadence: int
        Observation cadence of the simulated light curve.
    snr : float
        The ratio between the variability (RMS) amplitude of the DRW and the median of the
        observation errors.
    mean_mag : float
        The mean apparent bolometric magnitude of the lightcurve.
    seed_num : int
        Seed number passed to [jax.random.key][] used for lightcurve realization.
    season : tuple[int, int]
        The observing season. Tuple of [start day, end day] applies every year.
    timespan : int
        Number of days in observing campaign including gaps.
    name : str
        Name of the DRW signal. Defaults to "DRW".

    Attributes
    ----------
    amplitude : Distribution
        Amplitude of the DRW.
    timescale : Distribution
        Timescale of the DRW (usually called $\tau$).
    cadence: int
        Observation cadence of the simulated light curve.
    snr : float
        The ratio between the variability (RMS) amplitude of the DRW and the median of the
        observation errors.
    mean_mag : float
        The mean apparent bolometric magnitude of the lightcurve.
    seed_num : int
        Seed number passed to [jax.random.key][] used for lightcurve realization.
    season : tuple[int, int]
        The observing season. Tuple of [start day, end day] applies every year.
    timespan : int
        Number of days in observing campaign including gaps.
    signals : angst.signal.DampedRandomWalk
        The DRW signal to simulate.
    name : str
        Name of the DRW signal. Defaults to "DRW".

    Notes
    -----
    See [angst.dataset.Simulated][]
    """

    signals: Signal | SignalCollection = eqx.field(converter=_signal_converter)
    mean_mag: float
    snr: float
    seed_num: int
    timespan: int
    season: tuple[int, int]
    cadence: int
    band: str
    name: str = eqx.field(default="SIM", kw_only=True, static=True)

    def simulate(
        self,
        save_file: Path,
        rng_key: jax.Array,
        num_agn: int = 1,
    ) -> DataSet:
        """Simulate an AGN timeseries dataset with realizations of a DRW process.

        Parameters
        ----------
        rng_key : jax.Array
            JAX PRNG key.
        save_file : Path
            Path object with intended output location for simulated dataset in parquet format.
        num_agn : int
            The number of AGN to create for the dataset.

        Returns
        -------
        Stores simulated dataset in `out_file`. Overwrites if exists.

        Notes
        -----
        See [angst.dataset.BaseDataSet][]

        Examples
        --------
        >>> from angst import simulation, dataset, signal, utils
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from angst.signal import DampedRandomWalk
        >>> from angst import utils
        >>> from pathlib import Path
        >>> import numpyro.distributions as dist
        >>> import numpyro
        >>>
        >>> rng_key = jax.random.key(0)
        >>> tspan, cadence, season = 365, 60, (90, 270)
        >>> timestamps = utils.make_timestamps(tspan, cadence, rng_key)
        >>> mask = utils.make_season_mask(timestamps, season)
        >>> timestamps_season = timestamps[mask]
        >>> amplitude=dist.Uniform(1e-3, 3)
        >>> timescale=dist.Normal(180, 60)
        >>> cadence=cadence
        >>> snr=20
        >>> mean_mag=16
        >>> seed_num=0
        >>>
        >>> drw = simulation.DampedRandomWalkSim(
        ...     amplitude, timescale, cadence, snr, mean_mag, seed_num, season, tspan
        ... )
        >>>
        >>> out_file = (Path.cwd() / "drw-simulation.pq")
        >>> dset = drw.simulate(out_file, rng_key,  5)
        >>>
        >>> dset.get_cols_for_id(3, "lc.mjd", "lc.mag", "lc.mag_error", bands="r")
                                                              lc
            3  [{mjd: 120.307128, mag: 15.582855, mag_error: ...
        >>> dset.get_cols_for_id(3, "lc.mjd", "lc.mag", "lc.mag_error", bands="r").explode("lc")
                      mjd        mag  mag_error
            3  120.307128  15.582855   0.008669
            3  179.817581  14.688161   0.007501
            3  240.322435  17.265894   0.014371
        """

        tmp_dfs = []
        for agn_id in range(num_agn):
            rng_key, subkey, ra_key, dec_key = jax.random.split(rng_key, 4)
            mjds = utils.make_timestamps(self.timespan, self.cadence, subkey)
            mask = utils.make_season_mask(mjds, self.season)
            mjds, mag, mag_error = np.array(
                time_domain_sim(
                    self.signals,
                    self.snr,
                    mjds[mask],
                    self.mean_mag,
                    use_magnitudes=True,
                    rng_seed=self.seed_num,
                )
            )

            ids = np.full_like(mjds, agn_id, dtype=np.int32)

            ra = jax.random.uniform(ra_key, dtype=jnp.float64) * 360
            dec = jax.random.uniform(dec_key, dtype=jnp.float64) * 180 - 90

            flat_df = pd.DataFrame({
                "id": ids,
                "ra": np.full_like(mjds, ra),
                "dec": np.full_like(mjds, dec),
                "mjd": mjds,
                "mag": mag,
                "mag_error": mag_error,
                "band": np.repeat(self.band, len(mjds)),
                "survey": np.repeat("angst-simulation", len(mjds)),
                "survey_id": ids,
            })

            # Make nested. Will be one row per id.
            # lc will be a nested dataframe, one per id
            nested_frame = npd.NestedFrame.from_flat(
                flat_df,
                base_columns=["id", "ra", "dec"],
                nested_columns=["mjd", "mag", "mag_error", "band", "survey", "survey_id"],
                name="lc",
                on="id",
            ).reset_index()

            # Add simulation parameters
            model_trace = {}
            for s in self.signals.signals:
                model_trace.update(trace(seed(s, subkey)).get_trace(jnp.array([1.0, 2.0])))
            key_data = jax.random.key_data(subkey).tolist()
            nested_frame["input_parameters"] = pd.DataFrame(
                {
                    par: model_trace[par]["value"].item()
                    for par in self.signals.params
                    if "sample" not in par
                }
                | {
                    "cadence": self.cadence,
                    "season_start": self.season[0] if self.season is not None else 0,
                    "season_finish": self.season[1] if self.season is not None else 365.25,
                    "snr": self.snr,
                    "timespan": self.timespan,
                    "mean_mag": self.mean_mag,
                    "jax_key_data_0": key_data[0],
                    "jax_key_data_1": key_data[1],
                },
                index=[0],
            )

            tmp_dfs.append(nested_frame)

        joined: npd.NestedFrame = pd.concat(tmp_dfs, ignore_index=True)
        joined.to_parquet(save_file)

        info_str = f"Simulated data saved at {save_file.resolve()}."
        log.info(info_str)

        return DataSet(joined)
