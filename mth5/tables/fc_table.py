# -*- coding: utf-8 -*-
"""
Tabulate Fourier coefficients stored in an MTH5 file.

This module provides a small utility for summarizing Fourier-coefficient
datasets (e.g., `FCChannel`) into a structured table and exporting
to a convenient `pandas.DataFrame` for querying and analysis.

Notes
-----
- A basic test for this module exists under
    ``mth5/tests/version_1/test_fcs.py``.
- The table is populated by traversing the HDF5 hierarchy and collecting
    entries for datasets labeled with the attribute ``mth5_type='FCChannel'``.

"""

from __future__ import annotations

import h5py
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from mth5 import FC_DTYPE
from mth5.tables import MTH5Table


# =============================================================================


class FCSummaryTable(MTH5Table):
    """
    Summary table for Fourier coefficients.

    This class wraps an HDF5 dataset that stores a summary of Fourier
    coefficient datasets and provides convenience functions such as
    `summarize()` (to populate the table) and `to_dataframe()` (to export
    entries).

    Examples
    --------
    Populate and export a summary from an existing MTH5 file::

        >>> import h5py
        >>> from mth5.tables.fc_table import FCSummaryTable
        >>> f = h5py.File('example.mth5', 'r')
        >>> # Assume the summary dataset already exists at this path
        >>> table_ds = f['Exchange']['FC_Summary']
        >>> fc_table = FCSummaryTable(table_ds)
        >>> fc_table.summarize()  # walk the file and fill entries
        >>> df = fc_table.to_dataframe()
        >>> df.head()

    """

    def __init__(self, hdf5_dataset: h5py.Dataset) -> None:
        super().__init__(hdf5_dataset, FC_DTYPE)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the table to a `pandas.DataFrame` for easier querying.

        Returns
        -------
        pandas.DataFrame
            A dataframe with decoded string columns and parsed start/end
            timestamps.

        Examples
        --------
        Export to a dataframe and filter by component::

            >>> df = fc_table.to_dataframe()
            >>> df[df.component == 'ex']
        """

        if self.array is None:
            raise ValueError("Summary table dataset is not initialized.")
        df = pd.DataFrame(self.array[()])
        for key in [
            "survey",
            "station",
            "run",
            "component",
            "measurement_type",
            "units",
        ]:
            setattr(df, key, getattr(df, key).str.decode("utf-8"))
        try:
            df.start = pd.to_datetime(df.start.str.decode("utf-8"), format="mixed")
            df.end = pd.to_datetime(df.end.str.decode("utf-8"), format="mixed")
        except ValueError:
            df.start = pd.to_datetime(df.start.str.decode("utf-8"))
            df.end = pd.to_datetime(df.end.str.decode("utf-8"))

        return df

    def summarize(self) -> None:
        """
        Populate the summary table by traversing the HDF5 hierarchy.

        The traversal searches for datasets with attribute
        ``mth5_type == 'FCChannel'`` and adds a corresponding summary row
        for each.

        Returns
        -------
        None

        Notes
        -----
        - If the table contains rows from a different OS/encoding,
          row insertion can raise a `ValueError`. A warning is logged and
          processing continues for subsequent rows.

        Examples
        --------
        Refresh the table entries::

            >>> fc_table.clear_table()
            >>> fc_table.summarize()
        """

        def recursive_get_fc_entry(
            group: h5py.Group | h5py.File | h5py.Dataset,
        ) -> None:
            """Recursively collect FC summary entries from the hierarchy."""
            if isinstance(group, (h5py.Group, h5py.File)):
                for key, node in group.items():
                    recursive_get_fc_entry(node)
            elif isinstance(group, h5py.Dataset):
                try:
                    ch_type = group.attrs["mth5_type"]
                    if ch_type in [
                        "FCChannel",
                    ]:
                        fc_entry = _get_fc_entry(group)
                        try:
                            self.add_row(fc_entry)
                        except ValueError as error:
                            msg = (
                                f"{error}. "
                                "it is possible that the OS that made the table is not the OS operating on it."
                            )
                            self.logger.warning(msg)
                except KeyError:
                    pass

        self.clear_table()
        # self.fc_entries = []
        if self.array is None or getattr(self.array, "parent", None) is None:
            raise ValueError("Summary table dataset parent is not available.")
        parent = self.array.parent
        # Allow Mock objects and dictionaries for testing, in addition to h5py types
        if not (
            isinstance(parent, (h5py.Group, h5py.File, h5py.Dataset))
            or hasattr(parent, "items")
            or isinstance(parent, dict)
        ):
            raise TypeError("Unexpected parent type for summary dataset.")
        recursive_get_fc_entry(parent)
        # for row in self.fc_entries:
        #     try:
        #         self.add_row(row)
        #     except Exception as ee:
        #         msg = f"Failed due to unknown exception {e}"
        #         self.logger.warning(msg)
        # return


def _get_fc_entry(
    group: h5py.Dataset,
    dtype: np.dtype | None = FC_DTYPE,
) -> np.ndarray:
    """
    Build a single FC summary table row.

    Parameters
    ----------
    group : h5py._hl.dataset.Dataset
        The HDF5 dataset representing a Fourier-coefficient channel
        (i.e., with attribute ``mth5_type='FCChannel'``).
    dtype : numpy.dtype, optional
        The dtype describing the summary table schema. Defaults to
        :data:`mth5.FC_DTYPE`.

    Returns
    -------
    numpy.ndarray
        A 1-row structured array matching the summary table schema.

    Examples
    --------
    Create a row for an existing FC dataset::

        >>> fc_ds = f['Survey']['station']['run']['FC']['ex']
        >>> row = _get_fc_entry(fc_ds)
        >>> row.dtype == FC_DTYPE
        True
    """

    def _as_bytes(value: object) -> bytes:
        try:
            if isinstance(value, np.ndarray):
                value = value.item() if value.shape == () else value[0]
        except Exception:
            pass
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")

    fc_entry = np.array(
        [
            (
                _as_bytes(
                    group.parent.parent.parent.parent.parent.parent.attrs["id"]
                ),  # get survey from FCChannel
                _as_bytes(
                    group.parent.parent.parent.parent.attrs["id"]
                ),  # get station from FCChannel
                group.parent.parent.attrs["id"],  # get run from FCChannel
                group.parent.attrs[
                    "decimation_level"
                ],  # get decimation_level from FCChannel
                group.parent.parent.parent.parent.attrs["location.latitude"],
                group.parent.parent.parent.parent.attrs["location.longitude"],
                group.parent.parent.parent.parent.attrs["location.elevation"],
                group.attrs["component"],
                group.attrs["time_period.start"],
                group.attrs["time_period.end"],
                group.size,
                group.attrs["sample_rate_window_step"],
                group.attrs["mth5_type"],
                # group.attrs["measurement_azimuth"], # DO NOT go to the time series to access this info
                # group.attrs["measurement_tilt"],  # the time series may not be in the mth5
                # TODO: add azimuth and tilt on FCChannel creation
                group.attrs["units"],
                group.ref,
                group.parent.ref,
                group.parent.parent.ref,
                group.parent.parent.parent.parent.ref,
            )
        ],
        dtype=dtype,
    )
    return fc_entry
