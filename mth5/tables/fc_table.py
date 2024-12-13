# -*- coding: utf-8 -*-
"""
    This module tabulates the fourier coefficients stored in an mth5.

    A basic test for this module is in mth5/tests/version_1/test_fcs.py.

"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import numpy as np
import h5py

from mth5 import FC_DTYPE
from mth5.tables import MTH5Table
from typing import Optional, Union

# =============================================================================


class FCSummaryTable(MTH5Table):
    """
    Object to hold the channel summary and provide some convenience functions
    like fill, to_dataframe ...

    """

    def __init__(self, hdf5_dataset):
        super().__init__(hdf5_dataset, FC_DTYPE)

    def to_dataframe(self):
        """
        Create a pandas DataFrame from the table for easier querying.

        :return: Channel Summary
        :rtype: :class:`pandas.DataFrame`

        """

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
            df.start = pd.to_datetime(
                df.start.str.decode("utf-8"), format="mixed"
            )
            df.end = pd.to_datetime(df.end.str.decode("utf-8"), format="mixed")
        except ValueError:
            df.start = pd.to_datetime(df.start.str.decode("utf-8"))
            df.end = pd.to_datetime(df.end.str.decode("utf-8"))

        return df

    def summarize(self):
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """

        def recursive_get_fc_entry(group):
            """
            a function to get channel entry
            """
            if isinstance(group, (h5py._hl.group.Group, h5py._hl.files.File)):
                for key, node in group.items():
                    recursive_get_fc_entry(node)
            elif isinstance(group, h5py._hl.dataset.Dataset):
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
        recursive_get_fc_entry(self.array.parent)
        # for row in self.fc_entries:
        #     try:
        #         self.add_row(row)
        #     except Exception as ee:
        #         msg = f"Failed due to unknown exception {e}"
        #         self.logger.warning(msg)
        # return


def _get_fc_entry(
    group: h5py._hl.dataset.Dataset, dtype: Optional[np.dtype] = FC_DTYPE
) -> np.ndarray:
    """
    Create a fc_summary table entry in np.array format

    :type group: h5py._hl.dataset.Dataset
    :param group: h5 dataset with Fourier coeffifients
    :type dtype: np.dtype
    :param dtype: The dytpes for each of the table entries
    :rtype: np.ndarray
    :return: fc_summary table entry

    """
    fc_entry = np.array(
        [
            (
                group.parent.parent.parent.parent.parent.parent.attrs[
                    "id"
                ].encode(
                    "utf-8"
                ),  # get survey from FCChannel
                group.parent.parent.parent.parent.attrs["id"].encode(
                    "utf-8"
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
