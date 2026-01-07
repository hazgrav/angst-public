"""Helper utilities for other angst modules."""

from collections.abc import Callable
from functools import singledispatch
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optimistix as optx
from jax.tree_util import Partial
from jax.typing import ArrayLike
from numpyro.handlers import condition, reparam, seed, trace
from numpyro.infer.reparam import Reparam
from numpyro.infer.util import log_density

if TYPE_CHECKING:
    from typing import Any


def make_timestamps(tspan: float, cadence: int, rng_key: jax.Array) -> jax.Array:
    """Make timestamps for an AGN observation timeseries.

    Parameters
    ----------
    tspan: float
        The timespan of the observations.
    cadence: int
        The observational cadence in days.
    rng_key: jax.Array
        A JAX prng key from [jax.random.key][].

    Returns
    -------
    timestamps: jax.Array
        The timestamps of the observations.

    Examples
    --------
    >>> from angst import utils
    >>> import jax
    >>> rng_key = jax.random.key(0)
    >>> tspan, cadence, season = 365, 60, (90, 270)
    >>> timestamps = utils.make_timestamps(tspan, cadence, rng_key)
    >>> print(timestamps)
    [1.8784384e-01 5.8716656e+01 1.2064942e+02 1.8124905e+02 2.4024448e+02 2.9988254e+02]
    """
    timestamps = cadence * jnp.arange(0, tspan // cadence) + jax.random.normal(
        rng_key, shape=(int(tspan // cadence),)
    )
    return timestamps.sort()


def make_season_mask(timestamps: ArrayLike, season: tuple[int, int] | None) -> jax.Array:
    """Make a boolean mask corresponding to seasonal gaps in observation.

    Parameters
    ----------
    timestamps : ArrayLike
        Original timestamps without gaps (MJD)
    season : tuple[int, int]
        Seasonal gap. Stop observations between days `season[0]` and `season[1]`
        every year.

    Returns
    -------
    jax.Array
        JAX array of booleans True where observations happen and False during gaps. This array
        should be used to maske the input `timestamps`.

    Examples
    --------
    >>> from angst import utils
    >>> import jax
    >>> rng_key = jax.random.key(0)
    >>> tspan, cadence, season = 365, 60, (90, 270)
    >>> timestamps = utils.make_timestamps(tspan, cadence, rng_key)
    >>> mask = utils.make_season_mask(timestamps, season)
    >>> print(mask)
    [False False True True True False]
    """
    if season:
        timestamps = timestamps - timestamps[0]
        return (jnp.mod(timestamps, 365.25) > season[0]) & (jnp.mod(timestamps, 365.25) < season[1])

    return jnp.ones_like(timestamps, dtype=bool)


def make_freqs(nfreq: int, tspan: float) -> jax.Array:
    """
    Generate frequencies for spectral analysis.

    Parameters
    ----------
    nfreq : int
        Number of frequencies to generate.
    tspan : float
        Total timespan of the observations.

    Returns
    -------
    jax.Array
        Array of frequencies.

    Examples
    --------
    >>> from angst import utils
    >>> freqs = utils.make_freqs(5, 365.0)
    >>> print(freqs)
    [0.00273973 0.00547945 0.00821918 0.01095890 0.01369863]
    """
    return jnp.arange(1, nfreq + 1) / tspan


@jax.jit
def project_fourier_basis(times: jax.Array, ang_freqs: jax.Array) -> jax.Array:
    """
    Project times onto a Fourier basis.

    Parameters
    ----------
    times : ArrayLike
        Array of time points.
    ang_freqs : ArrayLike
        Array of angular frequencies.

    Returns
    -------
    jax.Array
        Fourier basis projection.

    Notes
    -----
    This function is JIT-compiled for improved performance.
    """
    times = jnp.atleast_2d(times)
    basis = jnp.concatenate((jnp.cos(ang_freqs @ times), jnp.sin(ang_freqs @ times)), axis=0)
    return basis


@jax.jit
def project_ortho_amps_fourier_basis(
    ortho_amps: jax.Array, fourier_eigval: jax.Array, fourier_eigvec: jax.Array
) -> jax.Array:
    """
    Project orthogonal amplitudes onto a Fourier basis.

    Parameters
    ----------
    ortho_amps : jax.Array
        Orthogonal amplitudes.
    fourier_eigval : jax.Array
        Eigenvalues of the Fourier basis.
    fourier_eigvec : jax.Array
        Eigenvectors of the Fourier basis.

    Returns
    -------
    jax.Array
        Projected Fourier amplitudes.

    Notes
    -----
    This function is JIT-compiled for improved performance.
    """
    fourier_amps = jnp.dot(ortho_amps / jnp.sqrt(fourier_eigval), fourier_eigvec.T)
    return jnp.diag(fourier_amps)


@jax.jit
def covariance_basis_change(covariance: jax.Array, basis: jax.Array) -> jax.Array:
    """
    Change the basis of a covariance matrix.

    Parameters
    ----------
    covariance : jax.Array
        Covariance matrix in the original basis.
    basis : jax.Array
        New basis.

    Returns
    -------
    jax.Array
        Covariance matrix in the new basis.

    Notes
    -----
    This function is JIT-compiled for improved performance.
    """
    return basis @ covariance @ basis.T


@jax.jit
def covariance_orthonormal_basis(
    fourier_basis: jax.Array,
    covariance_fourier_eigval: jax.Array,
    covariance_fourier_eigvec: jax.Array,
) -> jax.Array:
    """
    Compute the covariance in an orthonormal Fourier basis.

    Parameters
    ----------
    fourier_basis : jax.Array
        Fourier basis.
    covariance_fourier_eigval : jax.Array
        Eigenvalues of the covariance matrix in the Fourier basis.
    covariance_fourier_eigvec : jax.Array
        Eigenvectors of the covariance matrix in the Fourier basis.

    Returns
    -------
    jax.Array
        Covariance in the orthonormal Fourier basis.

    Notes
    -----
    This function is JIT-compiled for improved performance.
    """
    return jnp.diag(1 / jnp.sqrt(covariance_fourier_eigval)) @ jnp.dot(
        covariance_fourier_eigvec.T, fourier_basis
    )


def drw_psd(
    amp: float,
    tau: float,
) -> Callable:
    """
    Define PSD function for a given set of parameteters.

    Parameters
    ----------
    amp : float
        Amplitude of DRW process.
    tau : float
        Decorrelation timespan of DRW process.

    Returns
    -------
    Callable
        Power spectral density of DRW process at given frequency.

    Notes
    -----
    Inputs defined as normal frequency (not angular frequency).
    """

    def power(f: jax.Array) -> Any:
        return 2 * tau * (amp**2) / (1 + (2 * jnp.pi * f * tau) ** 2)

    return power


def figsettings(scale: float) -> None:
    """
    Set the figure settings.

    Parameters
    ----------
    scale: float
        Set the size of the plot

    """

    def figsize(scale: float) -> list[float]:
        fig_width_pt = 513.17  # 469.755
        inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        golden_mean = float((np.sqrt(5.0) - 1.0) / 2.0)  # Aesthetic ratio
        fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
        fig_height = 1.4 * fig_width * golden_mean  # height in inches
        fig_size = [fig_width, fig_height]
        return fig_size

    params = {"figure.figsize": figsize(scale=scale)}
    plt.rcParams.update(params)


def build_solver(tspan, nkl):
    bounds = jnp.arange(0, nkl)
    even_bounds = jnp.array([0.0, (jnp.pi / (tspan))])[:, None] + (bounds * (2 * jnp.pi / (tspan)))
    odd_bounds = jnp.array([(jnp.pi / (tspan)), (2 * jnp.pi / (tspan))])[:, None] + (
        bounds * (2 * jnp.pi / (tspan))
    )

    @jax.jit
    def even_equation(x, tau) -> Any:
        return 0.001 * (jnp.cos(tspan * x / 2.0) - tau * x * jnp.sin(tspan * x / 2.0)) / x

    @jax.jit
    def odd_equation(x, tau) -> Any:
        return 0.001 * (tau * x * jnp.cos(tspan * x / 2.0) + jnp.sin(tspan * x / 2.0)) / x

    def even_root(bracket: jax.Array, guess: jax.Array, tau) -> Any:
        a, b = bracket
        solutions = optx.root_find(
            even_equation,
            solver=optx.Bisection(rtol=10e-8, atol=10e-8),
            y0=guess,
            options={"lower": a, "upper": b},
            args=tau,
        ).value
        return solutions

    def odd_root(bracket: jax.Array, guess: jax.Array, tau) -> Any:
        a, b = bracket
        solutions = optx.root_find(
            odd_equation,
            solver=optx.Bisection(rtol=10e-8, atol=10e-8),
            y0=guess,
            options={"lower": a, "upper": b},
            args=tau,
        ).value
        return solutions

    even_solver = Partial(
        jax.vmap(even_root, in_axes=[1, 0, None]),
        even_bounds,
        jnp.mean(even_bounds, axis=0),
    )

    odd_solver = Partial(
        jax.vmap(odd_root, in_axes=[1, 0, None]),
        odd_bounds,
        jnp.mean(odd_bounds, axis=0),
    )
    return even_solver, odd_solver


def rho_distribution(x, sigma1, sigma2):
    return (
        (1 / (2 * jnp.sqrt(sigma1 * sigma2)))
        * jnp.exp(-((sigma2 + sigma1) / (4 * sigma1 * sigma2)) * x)
        * jax.scipy.special.i0(((sigma2 - sigma1) / (4 * sigma1 * sigma2)) * x)
    )


def get_gp_posterior_predictive_mean_variance(
    rng: jax.Array,
    model: Callable,
    posterior_samples: dict[str, jax.Array],
    gp_observed_y: jax.Array,
    gp_predict_x: jax.Array,
    model_args: tuple,
    model_kwargs: dict,
    return_site: str = "obs",
    include_mean: bool = True,
):
    """
    Compute GP posterior predictive mean and variance for each posterior sample.

    Runs the model conditioned on posterior samples and computes the Gaussian
    process posterior predictive distribution at new input locations. For each
    posterior sample, returns the conditional mean and variance of the GP at
    the prediction points.

    Parameters
    ----------
    rng : jax.random.key
        Random key for stochastic operations during model tracing.
    model : Callable
        NumPyro model function containing the GP definition.
    posterior_samples : dict
        Dictionary of posterior samples with shape (num_samples, ...) for each
        parameter.
    gp_observed_y : jax.Array
        Observed y-values at the training points used to condition the GP.
    gp_predict_x : jax.Array
        Input locations at which to compute the posterior predictive.
    model_args : tuple
        Positional arguments to pass to the model function.
    model_kwargs : dict
        Keyword arguments to pass to the model function.
    return_site : str, optional
        Name of the sample site containing the GP object in the model trace.
        Default is "obs".
    include_mean : bool, optional
        Whether to include the GP mean function in the posterior predictive.
        Default is True.

    Returns
    -------
    dict
        Dictionary with key `return_site` containing an array of shape
        (num_samples, 2, num_predict_points) where index 0 along axis 1 is
        the posterior predictive mean and index 1 is the posterior predictive
        variance at each prediction point.

    See Also
    --------
    numpyro.infer.Predictive : General predictive distribution sampling.
    tinygp.GaussianProcess.condition : GP conditioning method used internally.
    """

    def single_prediction(rng, samples):
        model_trace = trace(seed(condition(model, samples), rng)).get_trace(
            *model_args, **model_kwargs
        )
        condition_result = seed(
            condition(model_trace[return_site]["fn"].gp.condition, samples), rng
        )(gp_observed_y, gp_predict_x, include_mean=include_mean)
        return {
            return_site: jnp.array([
                condition_result.gp.loc,
                condition_result.gp.variance,
            ])
        }

    num_samples = (
        s.shape[0] if (s := jax.tree_util.tree_flatten(posterior_samples)[0][0]).ndim > 0 else 1
    )
    rngs = jax.random.split(rng, num_samples)
    try:
        res = eqx.filter_jit(
            Partial(jax.lax.map, f=lambda x: single_prediction(*x), batch_size=100)
        )(xs=(rngs, posterior_samples))
    except:
        res = eqx.filter_jit(single_prediction)(rng, posterior_samples)
    return res


def make_hypercube_model(
    model: Callable[..., Any], rng_key: jax.Array, *args: Any, **kwargs: Any
) -> Callable[..., Any]:
    """
    Reparameterize a NumPyro model to use unit hypercube priors.

    Transforms all latent (non-observed) sample sites in a NumPyro model so that
    their priors become Uniform(0, 1). This is useful for nested sampling algorithms
    and other inference methods that expect parameters in the unit hypercube.

    The reparameterization uses NumPyro's `UniformReparam` which maps samples from
    the unit interval to the original prior distribution via the inverse CDF
    (percent point function).

    Parameters
    ----------
    model : Callable
        A NumPyro model function.
    rng_key : jax.Array
        A JAX PRNG key used to trace the model and identify sample sites.
    *args : Any
        Positional arguments to pass to the model when tracing.
    **kwargs : Any
        Keyword arguments to pass to the model when tracing.

    Returns
    -------
    Callable
        A reparameterized model where all latent variables have Uniform(0, 1) priors.
        The transformed model can be used with any NumPyro inference algorithm.

    See Also
    --------
    log_prior_hypercube : Compute log prior for hypercube samples.
    log_likelihood_hypercube : Compute log likelihood for the reparameterized model.
    numpyro.contrib.nested_sampling.UniformReparam : The underlying reparameterization.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>> import numpyro.distributions as dist
    >>> from angst.utils import make_hypercube_model, log_prior_hypercube, log_likelihood_hypercube
    >>>
    >>> # Define a simple model
    >>> def my_model(y=None):
    ...     mu = numpyro.sample("mu", dist.Normal(0, 1))
    ...     sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    ...     numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    >>>
    >>> # Create hypercube-reparameterized version
    >>> rng_key = jax.random.key(0)
    >>> y_data = jnp.array([1.1, 1.9, 3.2])
    >>> hypercube_model = make_hypercube_model(my_model, rng_key, y=y_data)
    >>>
    >>> # Now latent parameters are sampled from Uniform(0, 1)
    >>> # and transformed to the original prior space internally
    >>>
    >>> # Evaluate log density at a point in hypercube space
    >>> hypercube_samples = {"mu_base": jnp.array(0.5), "sigma_base": jnp.array(0.5)}
    >>> log_prior = log_prior_hypercube(hypercube_samples)
    >>> log_lik = log_likelihood_hypercube(hypercube_model, hypercube_samples, y=y_data)

    Notes
    -----
    - The reparameterized parameters have "_base" appended to their names
      (e.g., "mu" becomes "mu_base")
    - This function traces the model once to discover all sample sites
    - Only non-observed sample sites are reparameterized
    """
    prototype_trace = trace(seed(model, rng_key)).get_trace(*args, **kwargs)

    param_names = [
        site["name"]
        for site in prototype_trace.values()
        if site["type"] == "sample" and not site["is_observed"]
    ]

    return cast(
        Callable[..., Any], reparam(model, config={k: UniformReparam() for k in param_names})
    )


def log_prior_hypercube(posterior_samples: dict[str, jax.Array]) -> jax.Array:
    """
    Compute log prior probability for unit hypercube parameterization.

    For samples in the unit hypercube (all values in [0, 1]), the prior is
    uniform, so the log prior is 0. If any value falls outside [0, 1], the
    log prior is -inf (impossible under the prior).

    This function is typically used with `make_hypercube_model` to evaluate
    the prior probability of samples in the reparameterized space.

    Parameters
    ----------
    posterior_samples : dict[str, jax.Array]
        Dictionary mapping parameter names to their values. All values should
        be in the unit interval [0, 1] for valid samples.

    Returns
    -------
    jax.Array
        Scalar log prior probability: 0.0 if all values across all parameters
        are in [0, 1], otherwise -inf. Always returns a single scalar, even
        for batched inputs.

    See Also
    --------
    make_hypercube_model : Create a hypercube-reparameterized model.
    log_likelihood_hypercube : Compute log density for the reparameterized model.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from angst.utils import log_prior_hypercube
    >>>
    >>> # Valid samples in [0, 1]
    >>> samples = {"mu_base": jnp.array(0.5), "sigma_base": jnp.array(0.3)}
    >>> log_p = log_prior_hypercube(samples)
    >>> print(float(log_p))
    0.0
    >>>
    >>> # Invalid sample outside [0, 1]
    >>> bad_samples = {"mu_base": jnp.array(1.5), "sigma_base": jnp.array(0.3)}
    >>> log_p = log_prior_hypercube(bad_samples)
    >>> print(float(log_p))
    -inf

    Notes
    -----
    This function assumes a flat (uniform) prior over the hypercube, which is
    appropriate when using `UniformReparam` reparameterization. The original
    prior information is encoded in the log density via the Jacobian of the
    inverse CDF transformation (see `log_likelihood_hypercube`).

    This function returns a single scalar that checks if ALL values are in
    bounds. It is not vectorized over batch dimensions - use `jax.vmap` if
    you need per-sample log priors for batched inputs.
    """
    all_in_bounds = jnp.all(
        jnp.array([(jnp.all((v >= 0) & (v <= 1))) for v in posterior_samples.values()])
    )
    return jnp.where(all_in_bounds, 0.0, -jnp.inf)


def log_likelihood_hypercube(
    reparam_model: Callable[..., Any],
    posterior_samples: dict[str, jax.Array],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Compute the effective log likelihood in hypercube space.

    For a model reparameterized to the unit hypercube, this function computes
    the log density that serves as the "likelihood" for nested sampling. This
    is the full model log density, which includes:

    - log p(data | theta): the data likelihood
    - log p(theta): the original prior density
    - log |d(theta)/d(u)|: the Jacobian of the inverse CDF transformation

    Since nested samplers assume a uniform prior over the hypercube, this
    combined quantity is what they call the "likelihood".

    Parameters
    ----------
    reparam_model : Callable
        A reparameterized NumPyro model, typically created by `make_hypercube_model`.
    posterior_samples : dict[str, jax.Array]
        Dictionary mapping parameter names to their values in the hypercube space.
        Parameter names should have "_base" suffix (added by `UniformReparam`).
    *args : Any
        Positional arguments to pass to the model.
    **kwargs : Any
        Keyword arguments to pass to the model.

    Returns
    -------
    jax.Array
        Scalar log density value (effective likelihood in hypercube space).

    See Also
    --------
    make_hypercube_model : Create a hypercube-reparameterized model.
    log_prior_hypercube : Compute log prior for hypercube samples (uniform).
    make_logdensity_fn : Create a closure with model/data bound.
    numpyro.infer.util.log_density : Underlying log density computation.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>> import numpyro.distributions as dist
    >>> from angst.utils import make_hypercube_model, log_likelihood_hypercube
    >>>
    >>> def my_model(y=None):
    ...     mu = numpyro.sample("mu", dist.Normal(0, 1))
    ...     sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    ...     numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    >>>
    >>> rng_key = jax.random.key(0)
    >>> y_data = jnp.array([1.1, 1.9, 3.2])
    >>>
    >>> # Create reparameterized model
    >>> hypercube_model = make_hypercube_model(my_model, rng_key, y=y_data)
    >>>
    >>> # Evaluate log density at a point in hypercube space
    >>> samples = {"mu_base": jnp.array(0.5), "sigma_base": jnp.array(0.5)}
    >>> log_lik = log_likelihood_hypercube(hypercube_model, samples, y=y_data)

    Notes
    -----
    **Terminology**: This function returns what nested samplers call the
    "likelihood" - the quantity to maximize subject to the prior constraint.
    Mathematically, it is:

        log L_eff(u) = log p(data | theta(u)) + log p(theta(u)) + log |J(u)|

    where u is the hypercube coordinate, theta(u) is the inverse CDF transform,
    and J(u) is its Jacobian. This equals the original log posterior up to
    a constant.

    This function is useful for:
    - Nested sampling algorithms that work in the unit hypercube
    - Computing importance weights
    - Model comparison via marginal likelihood estimation
    """
    return log_density(reparam_model, args, kwargs, posterior_samples)[0]


def transform_hypercube_to_original(
    model: Callable[..., Any],
    rng_key: jax.Array,
    hypercube_samples: dict[str, jax.Array],
    *args: Any,
    **kwargs: Any,
) -> dict[str, jax.Array]:
    """
    Transform samples from hypercube space back to the original parameter space.

    Given samples in the unit hypercube (with "_base" suffix), this function
    applies the inverse CDF transformation to recover the samples in the
    original prior space. This is useful for interpreting nested sampling
    results in terms of the original model parameters.

    Parameters
    ----------
    model : Callable
        The original (non-reparameterized) NumPyro model.
    rng_key : jax.Array
        A JAX PRNG key used to trace the model.
    hypercube_samples : dict[str, jax.Array]
        Samples in hypercube space. Keys should have "_base" suffix
        (e.g., "mu_base", "sigma_base").
    *args : Any
        Positional arguments to pass to the model.
    **kwargs : Any
        Keyword arguments to pass to the model.

    Returns
    -------
    dict[str, jax.Array]
        Samples in the original parameter space, with original parameter
        names (without "_base" suffix).

    See Also
    --------
    make_hypercube_model : Create a hypercube-reparameterized model.
    make_logdensity_fn : Create a log likelihood function for nested sampling.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>> import numpyro.distributions as dist
    >>> from angst.utils import transform_hypercube_to_original
    >>>
    >>> def my_model(y=None):
    ...     mu = numpyro.sample("mu", dist.Normal(0, 1))
    ...     sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    ...     numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    >>>
    >>> rng_key = jax.random.key(0)
    >>> y_data = jnp.array([1.1, 1.9, 3.2])
    >>>
    >>> # Hypercube samples (e.g., from nested sampling)
    >>> hypercube_samples = {"mu_base": jnp.array(0.5), "sigma_base": jnp.array(0.7)}
    >>>
    >>> # Transform to original parameter space
    >>> original_samples = transform_hypercube_to_original(
    ...     my_model, rng_key, hypercube_samples, y=y_data
    ... )

    Notes
    -----
    The transformation is performed by tracing the reparameterized model
    conditioned on the hypercube samples. The inverse CDF of each original
    prior distribution is applied automatically via the `UniformReparam`
    reparameterization.
    """
    reparam_model = make_hypercube_model(model, rng_key, *args, **kwargs)

    model_trace = trace(seed(condition(reparam_model, hypercube_samples), rng_key)).get_trace(
        *args, **kwargs
    )

    # The transformed parameters are stored as deterministic sites
    return {
        name: site["value"] for name, site in model_trace.items() if site["type"] == "deterministic"
    }


def make_log_likelihood_hypercube(
    model: Callable[..., Any],
    rng_key: jax.Array,
    *args: Any,
    **kwargs: Any,
) -> Callable[[dict[str, jax.Array]], jax.Array]:
    """
    Create an effective log likelihood function for nested sampling.

    Returns a callable that takes only the parameter samples (in the unit
    hypercube) and returns the effective log likelihood. This is designed
    for nested sampling algorithms where the sampler assumes a uniform prior
    over the hypercube and only needs the "likelihood" function.

    The returned function computes the full model log density, which includes
    the data likelihood, original prior, and Jacobian correction. This is
    the correct "likelihood" for nested samplers working in hypercube space.

    Parameters
    ----------
    model : Callable
        A NumPyro model function.
    rng_key : jax.Array
        A JAX PRNG key used to trace the model and identify sample sites.
    *args : Any
        Positional arguments to pass to the model.
    **kwargs : Any
        Keyword arguments to pass to the model.

    Returns
    -------
    Callable[[dict[str, jax.Array]], jax.Array]
        A function `log_likelihood(samples) -> scalar` that computes the
        effective log likelihood for samples in the unit hypercube [0, 1].

    See Also
    --------
    make_hypercube_model : Create a hypercube-reparameterized model.
    log_prior_hypercube : Compute log prior for hypercube samples (uniform).
    log_likelihood_hypercube : Lower-level log density computation.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>> import numpyro.distributions as dist
    >>> from angst.utils import make_log_likelihood_hypercube, log_prior_hypercube
    >>>
    >>> def my_model(y=None):
    ...     mu = numpyro.sample("mu", dist.Normal(0, 1))
    ...     sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    ...     numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    >>>
    >>> rng_key = jax.random.key(0)
    >>> y_data = jnp.array([1.1, 1.9, 3.2])
    >>>
    >>> # Create the log likelihood function with model/data bound
    >>> log_likelihood_fn = make_log_likelihood_hypercube(my_model, rng_key, y=y_data)
    >>>
    >>> # For nested sampling: pass log_likelihood_fn as the likelihood
    >>> # and log_prior_hypercube as the prior (returns 0 for valid samples)
    >>> samples = {"mu_base": jnp.array(0.5), "sigma_base": jnp.array(0.5)}
    >>> log_lik = log_likelihood_fn(samples)
    >>> log_prior = log_prior_hypercube(samples)  # 0.0 for samples in [0,1]
    >>>
    >>> # Compatible with JAX transformations
    >>> jit_loglik = jax.jit(log_likelihood_fn)

    Notes
    -----
    - Parameter names in the samples dict should have "_base" suffix
    - The closure captures the reparameterized model, so it can be passed
      to external samplers without additional context
    - For nested sampling, use this as the likelihood and `log_prior_hypercube`
      as the prior function

    **What this returns**: The effective likelihood in hypercube space equals
    the original model's log posterior (data likelihood + prior) plus the
    Jacobian of the inverse CDF transformation. This is exactly what nested
    samplers need when working with a uniform prior over the hypercube.
    """
    reparam_model = make_hypercube_model(model, rng_key, *args, **kwargs)

    def log_likelihood(samples: dict[str, jax.Array]) -> jax.Array:
        log_lik = log_likelihood_hypercube(reparam_model, samples, *args, **kwargs)
        return log_lik

    return log_likelihood


@singledispatch
def uniform_reparam_transform(d):
    """
    A helper for `UniformReparam` to get the transform that transforms
    a uniform distribution over a unit hypercube to the target distribution `d`.
    """
    if isinstance(d, dist.TransformedDistribution):
        outer_transform = dist.transforms.ComposeTransform(d.transforms)
        return lambda q: outer_transform(uniform_reparam_transform(d.base_dist)(q))

    if isinstance(d, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)):
        return lambda q: uniform_reparam_transform(d.base_dist)(q)

    return d.icdf


@uniform_reparam_transform.register(dist.MultivariateNormal)
def _(d):
    outer_transform = dist.transforms.LowerCholeskyAffine(d.loc, d.scale_tril)
    return lambda q: outer_transform(dist.Normal(0, 1).icdf(q))


@uniform_reparam_transform.register(dist.BernoulliLogits)
@uniform_reparam_transform.register(dist.BernoulliProbs)
def _(d):
    def transform(q):
        x = q < d.probs
        return x.astype(jnp.result_type(x, int))

    return transform


@uniform_reparam_transform.register(dist.CategoricalLogits)
@uniform_reparam_transform.register(dist.CategoricalProbs)
def _(d):
    return lambda q: jnp.sum(jnp.cumsum(d.probs, axis=-1) < q[..., None], axis=-1)


@uniform_reparam_transform.register(dist.Dirichlet)
def _(d):
    gamma_dist = dist.Gamma(d.concentration)

    def transform_fn(q):
        # NB: icdf is not available yet for Gamma distribution
        # so this will raise a NotImplementedError for now.
        # We will need scipy.special.gammaincinv, which is not available yet in JAX
        # see issue: https://github.com/jax-ml/jax/issues/5350
        gammas = uniform_reparam_transform(gamma_dist)(q)
        return gammas / gammas.sum(-1, keepdims=True)

    return transform_fn


class UniformReparam(Reparam):
    """
    Reparameterize a distribution to a Uniform over the unit hypercube.

    Most univariate distribution uses Inverse CDF for the reparameterization.
    """

    def __call__(self, name, fn, obs):
        assert obs is None, "TransformReparam does not support observe statements"
        shape = fn.shape()
        fn, expand_shape, event_dim = self._unwrap(fn)
        transform = uniform_reparam_transform(fn)
        tiny = jnp.finfo(jnp.result_type(float)).tiny

        x = numpyro.sample(
            "{}_base".format(name),
            dist.Uniform(tiny, 1).expand(shape).to_event(event_dim).mask(False),
        )
        # Simulate a numpyro.deterministic() site.
        return None, transform(x)
