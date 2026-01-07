"""Module for analysis of AGN datasets."""

from __future__ import annotations

import logging
import operator
from functools import cached_property, partial, reduce
from typing import TYPE_CHECKING
import matplotlib.colors as mcolors

import arviz as az
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from arviz import InferenceData
from jax.scipy import optimize
from numpyro.infer import MCMC
from numpyro.infer.util import Predictive, initialize_model
from tinygp import GaussianProcess, kernels

from angst import constants, utils
from angst.signal import SignalCollection

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import Any

    from graphviz.graphs import Digraph
    from jax.typing import ArrayLike
    from matplotlib.axes._axes import Axes
    from matplotlib.figure import Figure

    from angst.dataset import DataSet
    from angst.signal import Signal

log = logging.getLogger(__name__)


class SingleAGNFreeSpectrum(eqx.Module):
    """
    Analyze a single AGN light curve using a free spectrum model.

    This class provides methods for analyzing Active Galactic Nuclei (AGN) light curves
    using a free spectrum model. It integrates with NumPyro for probabilistic modeling
    and can be used with various MCMC samplers like NumPyro's NUTS or Blackjax.

    Parameters
    ----------
    dataset : DataSet
        The dataset containing AGN light curve data.
    agn_id : int
        The ID of the AGN to analyze.
    bands : Sequence[str]
        The photometric band(s) to analyze. Example: 'u', 'g', 'r', ...
    nfreq : int
        The number of frequencies to use in the free spectrum model.

    Attributes
    ----------
    agn_mjd : jax.Array
        The Modified Julian Dates of the observations.
    agn_lc : jax.Array
        The light curve data.
    agn_lc_zero_mean : jax.Array
        The zero-mean light curve data.
    agn_lc_err : jax.Array
        The errors associated with the light curve data.
    tspan_day : float
        The time span of the observations in days.
    freqs : jax.Array
        The frequencies used in the free spectrum model.

    Examples
    --------
    Analyzing an AGN light curve using NumPyro:

    >>> import jax
    >>> import numpyro
    >>> from angst import analysis, dataset
    >>> dset = dataset.Simulated("./drw-simulation.pq")
    >>> freespec = analysis.SingleAGNFreeSpectrum(
    ... dataset=dset, agn_id=0, bands=["r"], nfreq=5
    ... )
    >>> rng_key = jax.random.key(0)
    >>> nuts_kernel = numpyro.infer.NUTS(model=freespec)
    >>> mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
    >>> mcmc.run(rng_key, freespec.agn_mjd, freespec.agn_lc_err, freespec.agn_lc_zero_mean)
    >>> samples = mcmc.get_samples()

    Analyzing an AGN light curve using Blackjax:

    >>> import blackjax
    >>> initial_position = freespec.initial_sample(rng_key)
    >>> loglikelihood = freespec.loglikelihood
    >>> adapt = blackjax.window_adaptation(blackjax.nuts, loglikelihood)
    >>> (last_state, parameters), _ = adapt.run(rng_key, initial_position, 1000)
    >>> kernel = blackjax.nuts(loglikelihood, **parameters).step
    >>> def inference_loop(rng_key, kernel, initial_state, num_samples):
    ...     def one_step(state, rng_key):
    ...         state, info = kernel(rng_key, state)
    ...         return state, (state, info)
    ...     keys = jax.random.split(rng_key, num_samples)
    ...     _, (states, _) = jax.lax.scan(one_step, initial_state, keys)
    ...     return states
    >>> states = inference_loop(rng_key, kernel, last_state, 2000)
    """

    dset: DataSet
    agn_id: int
    bands: Sequence[str]
    nfreq: int

    agn_mjd_lc_err: jax.Array
    agn_mjd: jax.Array
    agn_mjd_year: jax.Array
    agn_mjd_secs: jax.Array
    agn_lc: jax.Array
    agn_lc_zero_mean: jax.Array
    agn_lc_err: jax.Array

    tspan_day: float
    freqs: jax.Array
    agn_freqs_array: jax.Array

    covariance: jax.Array
    fourier_basis: jax.Array
    fourier_eigval: jax.Array
    fourier_eigvec: jax.Array
    pred_t: jax.Array

    def __init__(
        self,
        dset: DataSet,
        agn_id: int,
        bands: Sequence[str],
        nfreq: int,
    ):
        self.dset = dset
        self.agn_id = agn_id
        self.bands = bands
        self.nfreq = nfreq

        # "explode" the nested dataframe into a flat dataframe
        agn_lc_df = dset.get_cols_for_id(
            agn_id,
            "lc.mjd",
            "lc.mag",
            "lc.mag_error",
            "lc.survey",
            "lc.band",
            bands=bands,
        ).explode("lc")

        self.agn_mjd_lc_err = jnp.array(agn_lc_df["mjd", "mag", "mag_error"].to_numpy(dtype=float))
        self.agn_mjd = self.agn_mjd_lc_err[:, 0].squeeze()
        self.agn_mjd_year = self.agn_mjd / constants.YR_DAY
        self.agn_mjd_secs = self.agn_mjd * constants.DAY_SEC

        self.agn_lc = self.agn_mjd_lc_err[:, 1].squeeze()
        self.agn_lc_zero_mean = jnp.array(
            agn_lc_df
            .groupby(by=["survey", "band"])["mag"]
            .transform(lambda x: x - x.mean())
            .to_numpy()
        )

        self.agn_lc_err = self.agn_mjd_lc_err[:, 2].squeeze()

        self.tspan_day = self.agn_mjd.max().item() - self.agn_mjd.min().item()

        self.freqs = utils.make_freqs(nfreq, self.tspan_day)
        self.agn_freqs_array = jnp.expand_dims(2 * jnp.pi * self.freqs, 1)

        self.fourier_basis = utils.project_fourier_basis(self.agn_mjd, self.agn_freqs_array)
        self.covariance = jnp.identity(self.fourier_basis.shape[1])
        self.fourier_eigval, self.fourier_eigvec = jnp.linalg.eigh(
            utils.covariance_basis_change(self.covariance, self.fourier_basis)
        )

        self.pred_t = jnp.linspace(
            self.agn_mjd.min().item(), self.agn_mjd.max().item(), int(self.tspan_day * 2)
        )

    def __call__(self, times: jax.Array, lc_err: jax.Array, lc: jax.Array) -> None:
        """
        Define the NumPyro model for the free spectrum analysis.

        This method sets up the probabilistic model for the AGN light curve analysis
        using a free spectrum approach. It defines the prior distributions and the
        likelihood function.

        Parameters
        ----------
        times : jax.Array
            The observation times.
        lc_err : jax.Array
            The light curve measurement errors.
        lc : jax.Array, optional
            The observed light curve data. If provided, it will be used as the
            observed values in the model.

        Returns
        -------
        None
            This method doesn't return anything but sets up the NumPyro model.

        Notes
        -----
        This method is automatically called by NumPyro's inference algorithms.
        Users typically don't need to call this method directly.
        """
        with numpyro.plate("freqs", self.nfreq * 2):
            # Sample for orthogonal amplitudes (NOT SINES AND COSINES)
            ortho_amps = numpyro.sample("ortho-amps", dist.Normal(0, 10_000.0))

        # Keep corresponding fourier amplitudes and power
        sin_cos_amps = numpyro.deterministic(
            "sin-cos-amps",
            jnp.diag(
                utils.project_ortho_amps_fourier_basis(
                    ortho_amps,
                    self.fourier_eigval,
                    self.fourier_eigvec,
                )
            ),
        )
        rho_sq = numpyro.deterministic("rho_sq", self.get_rho_sq(sin_cos_amps))  # noqa: F841

        # Setting Kernel to zero to represent zero covariance between frequencies
        kernel = kernels.Constant(0.0)  # type: ignore[call-arg]

        # tinyGP with only diagonal kernel and fitting with mean function
        gp = GaussianProcess(
            kernel,
            times,
            diag=lc_err,
            mean=partial(self.free_spec_gp_mean, ortho_amps),
        )
        numpyro.sample("obs", gp.numpyro_dist(), obs=lc)

    @eqx.filter_jit
    def get_rho_sq(self, sin_cos_amps: jax.Array) -> jax.Array:
        """
        Calculate the power spectral density from sine and cosine amplitudes.

        Parameters
        ----------
        sin_cos_amps : jax.Array
            The sine and cosine amplitudes.

        Returns
        -------
        jax.Array
            The calculated power spectral density.
        """
        return jnp.square(sin_cos_amps[0 : (2 * self.nfreq // 2)]) + jnp.square(
            sin_cos_amps[-(2 * self.nfreq // 2) :]
        )

    @eqx.filter_jit
    def free_spec_gp_mean(self, ortho_amps: jax.Array, times: jax.Array) -> jax.Array:
        """
        Calculate the mean function for the Gaussian Process using orthogonal amplitudes.

        Parameters
        ----------
        ortho_amps : jax.Array
            The orthogonal amplitudes.
        times : jax.Array
            The observation times.

        Returns
        -------
        jax.Array
            The calculated mean function values.
        """
        return jnp.sum(
            (
                jnp.diag(ortho_amps)
                @ utils.covariance_orthonormal_basis(
                    utils.project_fourier_basis(times, self.agn_freqs_array),
                    self.fourier_eigval,
                    self.fourier_eigvec,
                )
            ),
            axis=0,
            dtype=float,
        )[0]

    @cached_property
    def log_density(self) -> Callable[[Any], jax.Array]:
        """
        Extract the log-likelihood function from the NumPyro model.

        Returns
        -------
        Callable
            A function that calculates the log-likelihood given a position in parameter space.

        Examples
        --------
        >>> loglikelihood = freespec.loglikelihood
        >>> initial_position = freespec.initial_sample(rng_key)
        >>> ll_value = loglikelihood(initial_position)
        >>> print(ll_value)
        -5175.3896
        """
        rng_key = jax.random.key(0)
        rng_key, init_key = jax.random.split(rng_key)
        _, potential_fn_gen, *_ = initialize_model(
            init_key,
            self,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            dynamic_args=False,
        )
        return lambda position: -1.0 * potential_fn_gen(position)

    def initial_sample(self, rng_key: jax.Array) -> dict[str, ArrayLike]:
        """
        Get an initial sample from the NumPyro model.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            A random number generator key.

        Returns
        -------
        dict
            A dictionary containing initial parameter values.

        Examples
        --------
        >>> rng_key = jax.random.key(0)
        >>> initial_position = freespec.initial_sample(rng_key)
        >>> print(initial_position)
        {'ortho-amps': Array([ 1.1538978 ,  1.394949  , -0.8328862 ,  0.4808755 , -1.8299141 ,
        -0.40136623, -0.7908406 ,  1.2946243 , -1.7867904 ,  1.6007953 ],      dtype=float32)}
        """
        rng_key = jax.random.key(0)
        rng_key, init_key = jax.random.split(rng_key)
        initial_sample, *_ = initialize_model(
            init_key,
            self,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            dynamic_args=False,
        )
        return initial_sample.z  # type: ignore[no-any-return]

    @property
    def post_process(self) -> dict[str, jax.Array]:
        """
        Get the post-processing function from the NumPyro model.

        Returns
        -------
        dict
            A dictionary containing post-processing functions.

        Examples
        --------
        >>> post_process_fn = freespec.post_process
        >>> deterministic_points = jax.vmap(
        ... post_process_fn(freespec.agn_mjd, freespec.agn_lc_err, freespec.agn_lc_zero_mean)
        ... )({"ortho-amps": states.position["ortho-amps"]})
        >>> print({k:v[0] for k,v in deterministic_points.items()})
        {
            "ortho-amps": Array(
                [
                    4.669173,
                    5.259977,
                    1.8122139,
                    -12.743431,
                    -23.89614,
                    7.851285,
                    -2.4007788,
                    17.485695,
                    -7.6175804,
                    0.78096735,
                ],
                dtype=float32,
            ),
            "pred": Array(
                [0.21704406, 0.23051132, 0.24397251, ..., 0.19053935, 0.20376827, 0.21704365],
                dtype=float32,
            ),
            "rho_sq": Array([4.1774597, 2.2336352, 1.339445, 0.43232203, 1.2190703], dtype=float32),
            "sin-cos-amps": Array(
                [
                    2.0125716,
                    -0.7186129,
                    -0.3027949,
                    -0.30310595,
                    -0.47081128,
                    -0.35639217,
                    -1.3104315,
                    1.1170319,
                    0.58347994,
                    0.9987027,
                ],
                dtype=float32,
            ),
        }
        """
        rng_key = jax.random.key(0)
        rng_key, init_key = jax.random.split(rng_key)
        *_, post_process_fn, _ = initialize_model(
            init_key,
            self,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            dynamic_args=False,
        )
        return post_process_fn  # type: ignore[no-any-return]

    def render_model(
        self,
        render_params: bool = True,
        render_distributions: bool = True,
        fname: None | str | Path = None,
    ) -> Digraph:
        """
        Create a graphviz visualization of the NumPyro model.

        Parameters
        ----------
        render_params : bool, optional
            Whether to render parameter nodes (default is True).
        render_distributions : bool, optional
            Whether to render distribution nodes (default is True).
        fname : str or Path, optional
            The filename to save the visualization (default is None).

        Returns
        -------
        Digraph
            A graphviz Digraph object representing the model.

        Examples
        --------
        >>> gviz_digraph = freespec.render_model(render_distributions=False)
        >>> gviz_digraph.render("model_visualization", format="png", cleanup=True)

        Example Plot
        ------------
        ![](../../assets/model-render-example.png)
        """
        return numpyro.render_model(
            self,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            render_params=render_params,
            render_distributions=render_distributions,
            filename=fname,
        )

    def plot_spectrum(
        self,
        data: InferenceData | MCMC | dict | pd.DataFrame,
        scale: float = 1.3,
    ) -> tuple[Figure, Axes]:
        """
        Plot the posteriors of the power in each frequency bin as violins.

        Parameters
        ----------
        data : Inferencedata | MCMC | dict | pd.DataFrame
            Free spectral posterior samples in one of the accepted formats

        Returns
        -------
        fig, ax
            Displays a matplotlib plot
        """
        utils.figsettings(scale)

        match data:
            case InferenceData():
                rho_sq = az.extract(data, var_names="rho_sq").values.T
            case MCMC():
                rho_sq = data.get_samples()["rho_sq"]
            case dict():
                rho_sq = data["rho_sq"]
            case pd.DataFrame():
                rho_sq = np.array(data["rho_sq"].to_numpy())
            case _:
                err_msg = f"{type(data)=} but needs to be InferenceData, MCMC, dict, or DataFrame."
                log.error(err_msg)
                raise TypeError

        fig, ax = plt.subplots()

        ax.violinplot(
            rho_sq,
            positions=self.freqs,
            widths=jnp.pi * 0.04 * self.freqs,
            showmeans=False,
            showextrema=False,
            showmedians=True,
            bw_method=0.5,
        )

        ax.grid(visible=True)
        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
        ax.set_ylabel("Power [day * mag^2]")
        ax.set_xlabel("Frequency [1/day]")
        ax.set_ylim(1e-4, 20)
        return fig, ax

    def plot_inference(
        self,
        data: InferenceData | MCMC | dict | pd.DataFrame,
        scale: float = 1.3,
        rng_key: jax.Array | None = None,
        include_mean=True,
    ) -> tuple[Figure, Axes]:
        """
        Plot the AGN lightcurve as well as the inference from the Free Spectral Analysis.

        Parameters
        ----------
        data : Inferencedata | MCMC | dict | pd.DataFrame
            Posterior samples in one of the accepted formats
        scale : float
            Specifies the size of the plot
        show_season : bool
            Whether to show the seasonal gaps in the inference plot.

        Returns
        -------
        fig, ax
            Displays a matplotlib plot
        """
        utils.figsettings(scale)

        match data:
            case InferenceData():
                posterior = jax.tree.map(
                    jnp.asarray,
                    az.extract(data, combined=True).to_pandas().to_dict(orient="list"),
                    is_leaf=lambda x: isinstance(x, list),
                )
            case MCMC():
                posterior = data.get_samples()
            case dict():
                posterior = data
            case _:
                err_msg = f"{type(data)=} but needs to be InferenceData, MCMC, dict, or DataFrame."
                log.error(err_msg)
                raise TypeError

        if rng_key is None:
            rng_key = jax.random.key(42)

        pred = utils.get_gp_posterior_predictive_mean_variance(
            rng_key,
            self,
            posterior,
            self.agn_lc_zero_mean,
            self.pred_t,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            model_kwargs={},
            include_mean=include_mean,
        )["obs"]

        # handle the case of a single parameter passed
        if pred.ndim == 3:
            means, variance = pred[:, 0, :].squeeze(), pred[:, 1, :].squeeze()

            combined_mean = means.mean(axis=0)
            combined_std = jnp.sqrt(variance.mean(axis=0) + jnp.var(means, axis=0))
        else:
            combined_mean = pred[0]
            combined_std = pred[1]

        q = (combined_mean - combined_std, combined_mean, combined_mean + combined_std)

        fig, ax = plt.subplots()

        ax.fill_between(
            self.pred_t,
            q[0],
            q[2],
            color="r",
            alpha=0.5,
            label="inference",
        )
        ax.plot(self.pred_t, q[1], color="r", lw=1, alpha=1)

        groups = (
            self.dset.get_cols_for_id(self.agn_id, ["lc"]).explode("lc").groupby(["band", "survey"])
        )
        colors = [key for key in mcolors.TABLEAU_COLORS if "red" not in key]

        for (name, group), color in zip(groups, colors[: len(groups)]):
            ax.errorbar(
                group.mjd,
                group.mag.transform(lambda x: x - x.mean()).to_numpy(),
                yerr=group.mag_error,
                marker="o",
                linestyle="",
                ms=2,
                elinewidth=1,
                label=f"{name[0]} band - {name[1]}",
                alpha=0.85,
                color=color,
            )
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Luminosity Change [mag]")
        ax.set_title(f"Lightcurve of AGN {self.agn_id}")
        fig.legend()
        return fig, ax

    def get_max_posterior(
        self, init_params: jax.Array | None = None, options=None
    ) -> dict[str, jax.Array]:
        """
        Find the maximum likelihood parameters.

        Parameters
        ----------
        init_params : ArrayLike, optional
            Initial parameter values for the optimizer.
        options: dict, optional
            Options to pass to jax.scipy.optimize.minimize

        Returns
        -------
        dict[str, jax.Array]
            The maximum a posteriori parameter values.
        """

        sample = self.initial_sample(jax.random.key(42))
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(sample)
        if init_params is None:
            init_params = flat_params

        def target(x):
            return -1.0 * self.log_density(unravel_fn(x))

        res = eqx.filter_jit(optimize.minimize)(
            fun=target, method="BFGS", x0=init_params, options=options
        )
        return self.post_process(unravel_fn(res.x))


class SingleAGNTimeDomain(eqx.Module):
    """
    Analyze a single AGN light curve in the time domain using composable signal models.

    This class performs time-domain analysis of AGN light curves by combining
    Gaussian Process (GP) signals (e.g., Damped Random Walk) with deterministic
    signals (e.g., sinusoids). It integrates with NumPyro for Bayesian inference
    and supports various MCMC samplers including NumPyro's NUTS and Blackjax.

    Parameters
    ----------
    dset : DataSet
        The dataset containing AGN light curve data.
    agn_id : int
        The ID of the AGN to analyze.
    bands : Sequence[str]
        The photometric band(s) to analyze (e.g., 'g', 'r', 'i').
    signals : Signal | SignalCollection
        The signal model(s) to fit. Can be a single Signal or a SignalCollection
        created by adding signals together (e.g., `drw + sinusoid`).

    Attributes
    ----------
    agn_mjd : jax.Array
        The Modified Julian Dates of the observations.
    agn_lc : jax.Array
        The light curve magnitudes.
    agn_lc_zero_mean : jax.Array
        The light curve data normalized to zero mean per survey/band.
    agn_lc_err : jax.Array
        The photometric errors for each observation.
    tspan_day : float
        The total time span of observations in days.
    pred_t : jax.Array
        Dense time grid for posterior predictions.
    signals : SignalCollection
        The signal collection used for modeling.

    Examples
    --------
    Fit a DRW + sinusoid model to an AGN light curve:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>> import numpyro.distributions as dist
    >>> from angst import analysis, dataset
    >>> from angst.signal import DampedRandomWalk, Sinusoid
    >>>
    >>> # Load data and define signals
    >>> dset = dataset.DataSet("./agn-lightcurves.pq")
    >>> drw = DampedRandomWalk(
    ...     amplitude=dist.HalfNormal(1.0),
    ...     timescale=dist.LogNormal(5.0, 1.0),
    ... )
    >>> sinusoid = Sinusoid(
    ...     amplitude=dist.HalfNormal(0.1),
    ...     period=dist.Uniform(100, 500),
    ...     phase=dist.Uniform(0, 2 * jnp.pi),
    ... )
    >>>
    >>> # Create analysis with combined signals
    >>> td_analysis = analysis.SingleAGNTimeDomain(
    ...     dset=dset,
    ...     agn_id=0,
    ...     bands="r",
    ...     signals=drw + sinusoid,
    ... )
    >>>
    >>> # Run MCMC inference
    >>> rng_key = jax.random.key(0)
    >>> nuts_kernel = numpyro.infer.NUTS(model=td_analysis)
    >>> mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    >>> mcmc.run(
    ...     rng_key,
    ...     td_analysis.agn_mjd,
    ...     td_analysis.agn_lc_err,
    ...     td_analysis.agn_lc_zero_mean,
    ... )
    >>> samples = mcmc.get_samples()

    Using Blackjax for inference:

    >>> import blackjax
    >>> rng_key = jax.random.key(0)
    >>> initial_position = td_analysis.initial_sample(rng_key)
    >>> loglikelihood = td_analysis.loglikelihood
    >>>
    >>> # Warmup adaptation
    >>> adapt = blackjax.window_adaptation(blackjax.nuts, loglikelihood)
    >>> (last_state, parameters), _ = adapt.run(rng_key, initial_position, 500)
    >>>
    >>> # Sampling
    >>> kernel = blackjax.nuts(loglikelihood, **parameters).step
    >>> def inference_loop(rng_key, kernel, initial_state, num_samples):
    ...     def one_step(state, rng_key):
    ...         state, info = kernel(rng_key, state)
    ...         return state, (state, info)
    ...     keys = jax.random.split(rng_key, num_samples)
    ...     _, (states, _) = jax.lax.scan(one_step, initial_state, keys)
    ...     return states
    >>> states = inference_loop(rng_key, kernel, last_state, 1000)
    """

    dset: DataSet
    agn_id: int
    bands: Sequence[str]
    signals: SignalCollection

    agn_mjd_lc_err: jax.Array
    agn_mjd: jax.Array
    agn_mjd_year: jax.Array
    agn_mjd_secs: jax.Array
    agn_lc: jax.Array
    agn_lc_zero_mean: jax.Array
    agn_lc_err: jax.Array

    tspan_day: float
    pred_t: jax.Array

    def __init__(
        self, dset: DataSet, agn_id: int, bands: Sequence[str], signals: Signal | SignalCollection
    ):
        self.dset = dset
        self.agn_id = agn_id
        self.bands = bands
        self.signals = (
            signals if isinstance(signals, SignalCollection) else SignalCollection([signals])
        )

        # "explode" the nested dataframe into a flat dataframe
        agn_lc_df = dset.get_cols_for_id(
            agn_id,
            "lc.mjd",
            "lc.mag",
            "lc.mag_error",
            "lc.survey",
            "lc.band",
            bands=bands,
        ).explode("lc")

        self.agn_mjd_lc_err = jnp.array(agn_lc_df["mjd", "mag", "mag_error"].to_numpy(dtype=float))

        self.agn_mjd = self.agn_mjd_lc_err[:, 0].squeeze()
        self.agn_mjd_year = self.agn_mjd / constants.YR_DAY
        self.agn_mjd_secs = self.agn_mjd * constants.DAY_SEC

        self.agn_lc = self.agn_mjd_lc_err[:, 1].squeeze()
        self.agn_lc_zero_mean = jnp.array(
            agn_lc_df
            .groupby(by=["survey", "band"])["mag"]
            .transform(lambda x: x - x.mean())
            .to_numpy()
        )

        self.agn_lc_err = self.agn_mjd_lc_err[:, 2].squeeze()

        self.tspan_day = self.agn_mjd.max().item() - self.agn_mjd.min().item()

        self.pred_t = jnp.linspace(
            self.agn_mjd.min().item(), self.agn_mjd.max().item(), int(self.tspan_day * 2)
        )

    def __call__(self, times: jax.Array, lc_err: jax.Array, lc: jax.Array | None = None) -> None:
        """
        Define the NumPyro model for the time-domain DRW analysis.

        This method sets up the probabilistic model for the AGN light curve analysis
        using a Damped Random Walk Gaussian Process. It defines the prior distributions
        and the likelihood function.

        Parameters
        ----------
        times : jax.Array
            The observation times.
        lc_err : jax.Array
            The light curve measurement errors.
        lc : jax.Array, optional
            The observed light curve data. If provided, it will be used as the
            observed values in the model.

        Returns
        -------
        None
            This method doesn't return anything but sets up the NumPyro model.

        Notes
        -----
        This method is automatically called by NumPyro's inference algorithms.
        Users typically don't need to call this method directly.
        """

        def combined_deterministic_signal(times):
            return (
                jnp
                .array([s(times) for s in self.signals.deterministic_signals])
                .sum(axis=0)
                .squeeze()
            )

        if self.signals.deterministic_signals and self.signals.gp_signals:
            kernel = reduce(operator.add, [gp.kernel for gp in self.signals.gp_signals])

            gp = GaussianProcess(kernel, times, diag=lc_err, mean=combined_deterministic_signal)

            numpyro.sample(
                "obs",
                gp.numpyro_dist(),
                obs=lc,
            )

        elif self.signals.deterministic_signals and not self.signals.gp_signals:
            forward = combined_deterministic_signal(times)
            with numpyro.plate("n_obs", forward.size):
                numpyro.sample(
                    "obs",
                    dist.Normal(forward, lc_err),
                    obs=lc,
                )

        elif not self.signals.deterministic_signals and self.signals.gp_signals:
            kernel = reduce(operator.add, [gp.kernel for gp in self.signals.gp_signals])

            gp = GaussianProcess(kernel, times, diag=lc_err)

            numpyro.sample(
                "obs",
                gp.numpyro_dist(),
                obs=lc,
            )

    @cached_property
    def log_density(self) -> Callable[[Any], jax.Array]:
        """
        Extract the log-likelihood function from the NumPyro model.

        Returns
        -------
        Callable
            A function that calculates the log-likelihood given a position in parameter space.

        Examples
        --------
        >>> loglikelihood = td_drw.loglikelihood
        >>> initial_position = td_drw.initial_sample(rng_key)
        >>> ll_value = loglikelihood(initial_position)
        >>> print(ll_value)
        -1373.0594
        """
        rng_key = jax.random.key(0)
        rng_key, init_key = jax.random.split(rng_key)
        _, potential_fn_gen, *_ = initialize_model(
            init_key,
            self,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            dynamic_args=False,
        )
        return lambda position: -1.0 * potential_fn_gen(position)
        # return lambda position: (
        #    potential_fn_gen(self.agn_mjd, self.agn_lc_err, self.agn_lc)(position)
        # )

    def initial_sample(self, rng_key: jax.Array) -> dict[str, ArrayLike]:
        """
        Get an initial sample from the NumPyro model.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            A random number generator key.

        Returns
        -------
        dict
            A dictionary containing initial parameter values.

        Examples
        --------
        >>> rng_key = jax.random.key(0)
        >>> initial_position = td_drw.initial_sample(rng_key)
        """
        rng_key = jax.random.key(0)
        rng_key, init_key = jax.random.split(rng_key)
        initial_sample, *_ = initialize_model(
            init_key,
            self,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            dynamic_args=False,
        )
        return initial_sample.z  # type: ignore[no-any-return]

    @cached_property
    def post_process(self) -> Callable[[dict[str, jax.Array]], dict[str, jax.Array]]:
        """
        Get the post-processing function from the NumPyro model.

        Returns
        -------
        dict
            A dictionary containing post-processing functions.

        """
        rng_key = jax.random.key(0)
        rng_key, init_key = jax.random.split(rng_key)
        *_, post_process_fn, _ = initialize_model(
            init_key,
            self,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            dynamic_args=False,
        )
        return post_process_fn  # type: ignore[no-any-return]

    def render_model(
        self,
        render_params: bool = True,
        render_distributions: bool = True,
        fname: None | str | Path = None,
    ) -> Digraph:
        """
        Create a graphviz visualization of the NumPyro model.

        Parameters
        ----------
        render_params : bool, optional
            Whether to render parameter nodes (default is True).
        render_distributions : bool, optional
            Whether to render distribution nodes (default is True).
        fname : str or Path, optional
            The filename to save the visualization (default is None).

        Returns
        -------
        Digraph
            A graphviz Digraph object representing the model.

        Examples
        --------
        >>> gviz_digraph = td_drw.render_model(render_distributions=False)
        >>> gviz_digraph.render("model_visualization", format="png", cleanup=True)

        Example Plot
        ------------
        ![](../../assets/model-render-example.png)
        """
        return numpyro.render_model(
            self,
            model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
            render_params=render_params,
            render_distributions=render_distributions,
            filename=fname,
        )

    @cached_property
    def posterior_predictive(self):
        if self.signals.gp_signals:

            def predictive_func(rng_key, posterior_samples, include_mean=True):
                return utils.get_gp_posterior_predictive_mean_variance(
                    rng_key,
                    self,
                    posterior_samples,
                    self.agn_lc_zero_mean,
                    self.pred_t,
                    model_args=(self.agn_mjd, self.agn_lc_err, self.agn_lc_zero_mean),
                    model_kwargs={},
                    include_mean=include_mean,
                )["obs"]

        else:

            def predictive_func(rng_key, posterior_samples, *args, **kwargs):
                return Predictive(self, posterior_samples)(
                    rng_key, self.agn_mjd, self.agn_lc_err, None
                )["obs"]

        return predictive_func

    def plot_inference(
        self,
        data: InferenceData | MCMC | dict | pd.DataFrame,
        scale: float = 1.3,
        rng_key: jax.Array | None = None,
        include_mean=True,
    ) -> tuple[Figure, Axes]:
        """
        Plot the AGN lightcurve as well as the inference from the time-domain DRW analysis.

        Parameters
        ----------
        data : Inferencedata | MCMC | dict | pd.DataFrame
            Posterior samples in one of the accepted formats
        scale : float
            Specifies the size of the plot
        show_season : bool
            Whether to show the seasonal gaps in the inference plot.

        Returns
        -------
        fig, ax
        """
        utils.figsettings(scale)

        match data:
            case InferenceData():
                posterior = jax.tree.map(
                    jnp.asarray,
                    az.extract(data, combined=True).to_pandas().to_dict(orient="list"),
                    is_leaf=lambda x: isinstance(x, list),
                )
            case MCMC():
                posterior = data.get_samples()
            case dict():
                posterior = data
            case _:
                err_msg = f"{type(data)=} but needs to be InferenceData, MCMC, dict, or DataFrame."
                log.error(err_msg)
                raise TypeError

        if rng_key is None:
            rng_key = jax.random.key(42)

        pred = self.posterior_predictive(rng_key, posterior, include_mean=include_mean)

        if self.signals.gp_signals:
            means, variance = pred[:, 0, :].squeeze(), pred[:, 1, :].squeeze()

            combined_mean = means.mean(axis=0)
            combined_std = jnp.sqrt(variance.mean(axis=0) + jnp.var(means, axis=0))

            q = (combined_mean - combined_std, combined_mean, combined_mean + combined_std)
        else:
            q = np.percentile(pred, [0.16, 0.50, 0.84], axis=0)

        fig, ax = plt.subplots()

        # ax.errorbar(
        #    self.agn_mjd,
        #    self.agn_lc_zero_mean,
        #    self.agn_lc_err,
        #    fmt=".",
        #    color="tab:blue",
        #    ecolor="tab:blue",
        #    capsize=0,
        #    markersize=5,
        #    alpha=0.8,
        #    #label=f"{self.bands} band(s)",
        # )

        groups = (
            self.dset.get_cols_for_id(self.agn_id, ["lc"]).explode("lc").groupby(["band", "survey"])
        )
        colors = [key for key in mcolors.TABLEAU_COLORS if "red" not in key]

        for (name, group), color in zip(groups, colors[: len(groups)]):
            ax.errorbar(
                group.mjd,
                group.mag.transform(lambda x: x - x.mean()).to_numpy(),
                yerr=group.mag_error,  # / m,
                marker="o",
                linestyle="",
                ms=2,
                elinewidth=1,
                label=f"{name[0]} band - {name[1]}",
                alpha=0.85,
                color=color,
            )

        ax.fill_between(
            self.pred_t if pred.shape[-1] == self.pred_t.size else self.agn_mjd,
            q[0],
            q[2],
            color="r",
            alpha=0.5,
            label="inference",
        )
        ax.plot(
            self.pred_t if pred.shape[-1] == self.pred_t.size else self.agn_mjd,
            q[1],
            color="r",
            lw=1,
            alpha=1,
        )
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Luminosity Change [mag]")
        ax.legend()
        return fig, ax

    def get_max_posterior(
        self, init_params: jax.Array | None = None, options=None
    ) -> dict[str, jax.Array]:
        """
        Find the maximum likelihood parameters.

        Parameters
        ----------
        init_params : ArrayLike, optional
            Initial parameter values for the optimizer.
        options: dict, optional
            Options to pass to jax.scipy.optimize.minimize

        Returns
        -------
        dict[str, jax.Array]
            The maximum a posteriori parameter values.
        """

        sample = self.initial_sample(jax.random.key(42))
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(sample)
        if init_params is None:
            init_params = flat_params

        def target(x):
            return -1.0 * self.log_density(unravel_fn(x))

        res = eqx.filter_jit(optimize.minimize)(
            fun=target, method="BFGS", x0=init_params, options=options
        )
        return self.post_process(unravel_fn(res.x))
