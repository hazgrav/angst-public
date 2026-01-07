"""Module for AGN dataset utilities."""

import abc
import logging
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent

import equinox as eqx
import jax
import jax.numpy as jnp
import nested_pandas as npd
from nested_pandas import NestedFrame
from pandas import DataFrame
from pyarrow import ArrowInvalid
from upath import UPath

log = logging.getLogger(__name__)


def _dataset_input_converter(dataset_input: NestedFrame | dict | str | Path) -> NestedFrame:
    """Convert input for datasets into an LSDB catalog

    Parameters
    ----------
    dataset_input : NestedFrame | str | Path
        The user input to convert to a NestedFrame

    Returns
    -------
    NestedFrame

    Raises
    ------
    TypeError
        Raised if user supplies a dataframe that isn't supported by Narwhals
    ValueError
        Raised if user supplies a file that cannot be interpreted as plain text data or parquet.

    """
    match dataset_input:
        case dict():
            nested_cols = [key for key in dataset_input if ("." in key and "lc" in key)]
            base_cols = [key for key in dataset_input if "." not in key]

            base_df = npd.NestedFrame(
                data={col: val for col, val in dataset_input.items() if col in base_cols}
                | {
                    col.split(".")[1]: [val.tolist()]
                    for col, val in dataset_input.items()
                    if col in nested_cols
                },
                index=[0],
            )
            lc_df = npd.NestedFrame.from_lists(base_df, base_columns=["id", "ra", "dec"], name="lc")
            return lc_df

        case str() | Path() | UPath():
            # check if url
            dataset_input = UPath(dataset_input)
            if not dataset_input.exists():
                err_msg = "File/URL does not exist!"
                raise ValueError(err_msg)
            elif dataset_input.exists():
                try:
                    return npd.read_parquet(dataset_input)
                except ArrowInvalid:
                    err_msg = (
                        f"Input {dataset_input} exists, but it does not contain valid parquet data."
                    )
                    log.exception(err_msg)
                    raise

            else:
                err_msg = "Dataset is not a valid input for `nested_pandas.read_parquet`."
                raise TypeError(err_msg)

        case NestedFrame():
            return dataset_input

        case _:
            err_msg = (
                "Dataset is not a NestedFrame or valid input for `nested_pandas.read_parquet`."
            )
            raise TypeError(err_msg)


class BaseDataSet(eqx.Module, abc.ABC):
    """Abstract class to contain and manipulate an AGN dataset.

    Parameters
    ----------
    df : str | Path | IntoFrameT
        Input that can be converted to LSDB catalog dataframe.
    name : str, optional
        Name of the dataset.

    Attributes
    ----------
    df : NestedFrame
        NestedPandas dataframe containing the dataset.
    name : str
        Name of the dataset.
    """

    df: NestedFrame = eqx.field(static=True, converter=_dataset_input_converter)
    name: str = eqx.field(default="DataSet", kw_only=True, static=True)

    @property
    def agn_ids(self) -> jax.Array:
        """Return a JAX array containing the unique AGN identifiers."""
        return jnp.array(self.df["id"].unique())

    @property
    def cols(self) -> list[str]:
        """Return the columns of the (Nested/Data)Frame."""
        return list(self.df.columns) + self.df.get_subcolumns()

    def get_cols(self, *cols: Sequence[str]) -> NestedFrame | DataFrame | None:
        """Return a (Nested/Data)Frame containing the specified columns.

        Parameters
        ----------
        *cols : Sequence[str]
            Column names to retrieve.

        Returns
        -------
        DataFrame or NestedFrame or None
            Pandas DataFrame or Nested Pandas NestedFrame containing the specified columns, or None if any column is not found.
        """
        try:
            result = self.df[list(cols)]
        except KeyError:
            err_string = dedent(
                f"""\
                One or more columns weren't found in the dataset. Check your spelling!
                Supplied columns {cols}
                Available columns {self.cols}
                """
            )
            log.exception(err_string)
            raise

        return result

    def get_cols_for_id(
        self, agn_id: int, *cols: Sequence[str], bands: Sequence[str] | None = None
    ) -> DataFrame | NestedFrame | None:
        """Return a (Nested/Data)Frame containing the specified columns for a given AGN identifier.

        Parameters
        ----------
        agn_id : int
            AGN identifier.
        *cols : Sequence[str]
            Column names to retrieve.
        band : Sequence[str]
            The bands to retrieve

        Returns
        -------
        DataFrame or NestedFrame or None

            Pandas DataFrame or Nested Pandas NestedFrame containing the specified columns for the id, or None if any column is not found.
        """
        cols_selector = list(cols) if len(cols) > 1 else cols[0]
        bands = [band.lower() for band in bands] if bands is not None else None
        if bands is None:
            try:
                result = self.df.query(f"id == {agn_id}")[cols_selector]

            except KeyError:
                err_string = dedent(
                    f"""\
                    One or more columns weren't found in the dataset. Check your spelling!
                    Supplied columns {cols}
                    Available columns {self.cols}
                    """
                )
                log.exception(err_string)
                raise

        else:
            try:
                result = self.df.query(f"id == {agn_id}").query(f"lc.band.isin({bands})")[
                    cols_selector
                ]

            except KeyError:
                err_string = dedent(
                    f"""\
                    One or more columns weren't found in the dataset. Check your spelling!
                    Supplied columns {cols}
                    Available columns {self.cols}
                    """
                )
                log.exception(err_string)
                raise

        return result

    def get_mjds(self, agn_id: int, bands: None | Sequence[str] = None) -> DataFrame | NestedFrame:
        """Return a Polars DataFrame with the modified Julian dates for an AGN identifier and band.

        Parameters
        ----------
        agn_id : int
            AGN identifier.
        band : str or None, optional
            Band to retrieve, or None to retrieve all bands.

        Returns
        -------
        DataFrame | NestedFrame or None
            (Nested/Data)Frame with the modified Julian dates for the given AGN identifier and band,
            or None if the band is not valid.
        """
        return self.get_cols_for_id(agn_id, "lc.mjd", bands=bands)

    # TODO: accept magnitude or flux with a Literal type
    def get_lc(self, agn_id: int, bands: None | Sequence[str] = None) -> DataFrame | NestedFrame:
        """Return a (Nested/Data)Frame containing the light curve for a given AGN identifier and band.

        Parameters
        ----------
        agn_id : int
            AGN identifier.
        band : str or None, optional
            Band to retrieve (e.g., 'u', 'g', 'r', 'i', 'z'), or None to retrieve all bands.

        Returns
        -------
        DataFrame | NestedFrame or None
            (Nested/Data)Frame containing the light curve for the given AGN identifier and band,
            or None if the band is not valid.
        """
        return self.get_cols_for_id(agn_id, "lc.mag", bands=bands)

    # TODO: accept magnitude or flux with a Literal type
    def get_lc_err(
        self, agn_id: int, bands: None | Sequence[str] = None
    ) -> DataFrame | NestedFrame:
        """Return a (Nested/Data)Frame with the light curve errors for an AGN identifier and band.

        Parameters
        ----------
        agn_id : int
            AGN identifier.
        band : str or None, optional
            Band to retrieve (e.g., 'u', 'g', 'r', 'i', 'z'), or None to retrieve all bands.

        Returns
        -------
        DataFrame | NestedFrame or None
            (Nested/Data)Frame with the light curve errors for the given AGN identifier and band,
            or None if the band is not valid.
        """
        return self.get_cols_for_id(agn_id, "lc.mag_error", bands=bands)


class DataSet(BaseDataSet):
    """Class for the arbitrary datasets.

    Parameters
    ----------
    df : Path
        Path to the dataset file.
    df_orig : Path, optional
        Path to the original dataset file. Not applicable to simulated dataset.

    Attributes
    ----------
    df : Path
        Path to the dataset file.
    df_orig : Path or None
        Path to the original dataset file. Not applicable to simulated dataset.
    name : str
        Name of the dataset ("Simulated Dataset").

    Notes
    -----
    See [angst.simulation.DampedRandomWalkSim.simulate][] for an example of simulated dataset
    creation. This simulated dataset should contain an extra metadata field in the file-level schema
    to record the input parameters for each individual AGN simulation. This information is stored
    as JSON. You can load this into a dictionary. The keys will then be the agn IDs, which are
    necessarily stored as strings in order to write them in the file metadata.

    Examples
    --------
    >>> from pathlib import Path
    >>> from pprint import pprint
    >>> import json
    >>> from angst import dataset
    >>> sim_file = (Path.cwd() / "drw-simulation.pq")
    >>> dset = dataset.Simulated(sim_file)
    >>> pprint(dset.info)
    {'changelog': 'v1.0.0: initial creation',
     'created-by': 'David Wright david.wright@nanograv.org',
     'created-on': '2024-07-29',
     'description': 'Mock AGN Dataset created with EzTao',
     'input_parameters': '{"0": {"DRW--amplitude": 2.6975455284118652, '
                         '"DRW--timescale": 307.92852783203125, "cadence": 60, '
                         '"season": [90, 270], "snr": 20, "timespan": 365, '
                         '"mean_mag": 16, "seed_num": 0}, "1": {"DRW--amplitude": '
                         '1.300925850868225, "DRW--timescale": 130.41159057617188, '
                         '"cadence": 60, "season": [90, 270], "snr": 20, '
                         '"timespan": 365, "mean_mag": 16, "seed_num": 0}, "2": '
                         '{"DRW--amplitude": 2.842205047607422, "DRW--timescale": '
                         '137.27499389648438, "cadence": 60, "season": [90, 270], '
                         '"snr": 20, "timespan": 365, "mean_mag": 16, "seed_num": '
                         '0}, "3": {"DRW--amplitude": 1.4521030187606812, '
                         '"DRW--timescale": 181.00189208984375, "cadence": 60, '
                         '"season": [90, 270], "snr": 20, "timespan": 365, '
                         '"mean_mag": 16, "seed_num": 0}, "4": {"DRW--amplitude": '
                         '2.323364019393921, "DRW--timescale": 164.37374877929688, '
                         '"cadence": 60, "season": [90, 270], "snr": 20, '
                         '"timespan": 365, "mean_mag": 16, "seed_num": 0}}',
     'last-modified': '2024-07-29',
     'name': 'Diaz Hernandez 2024 Simulation',
     'notes': '',
     'reference': 'ywx649999311/EzTao',
     'version': 'v1.0.0'}
    >>>
    >>> pprint(json.loads(dset.info['input_parameters']))
    {'0': {'DRW--amplitude': 2.6975455284118652,
           'DRW--timescale': 307.92852783203125,
           'cadence': 60,
           'mean_mag': 16,
           'season': [90, 270],
           'seed_num': 0,
           'snr': 20,
           'timespan': 365},
     '1': {'DRW--amplitude': 1.300925850868225,
           'DRW--timescale': 130.41159057617188,
           'cadence': 60,
           'mean_mag': 16,
           'season': [90, 270],
           'seed_num': 0,
           'snr': 20,
           'timespan': 365},
     '2': {'DRW--amplitude': 2.842205047607422,
           'DRW--timescale': 137.27499389648438,
           'cadence': 60,
           'mean_mag': 16,
           'season': [90, 270],
           'seed_num': 0,
           'snr': 20,
           'timespan': 365},
     '3': {'DRW--amplitude': 1.4521030187606812,
           'DRW--timescale': 181.00189208984375,
           'cadence': 60,
           'mean_mag': 16,
           'season': [90, 270],
           'seed_num': 0,
           'snr': 20,
           'timespan': 365},
     '4': {'DRW--amplitude': 2.323364019393921,
           'DRW--timescale': 164.37374877929688,
           'cadence': 60,
           'mean_mag': 16,
           'season': [90, 270],
           'seed_num': 0,
           'snr': 20,
           'timespan': 365}}
    """

    name: str = eqx.field(default="Simulated Dataset", kw_only=True, static=True)
