# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:09:38 2022

@author: jpeacock
"""

import h5py
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
from mt_metadata.transfer_functions import (
    ALLOWED_INPUT_CHANNELS,
    ALLOWED_OUTPUT_CHANNELS,
)

from mth5 import CHANNEL_DTYPE, RUN_SUMMARY_COLUMNS
from mth5.tables import MTH5Table


# =============================================================================


class ChannelSummaryTable(MTH5Table):
    """
    Object to hold the channel summary and provide some convenience functions
    like fill, to_dataframe ...

    """

    def __init__(self, hdf5_dataset):
        super().__init__(hdf5_dataset, CHANNEL_DTYPE)

    def _has_entries(self):
        """
        check if table has been summarized yet
        """

        if len(self.array) == 1:
            if self.array[0][0] == b"" and self.array[0][1] == b"":
                return False
        return True

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
            df.start = pd.to_datetime(df.start.str.decode("utf-8"), format="mixed")
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

        self.clear_table()

        def has_data(h5_dataset):
            """check to see if has data"""
            if len(h5_dataset) > 0:
                if len(np.nonzero(h5_dataset)[0]) > 0:
                    return True
                else:
                    return False
            return False

        def get_channel_entry(group, dtype=CHANNEL_DTYPE):
            ch_entry = np.array(
                [
                    (
                        group.parent.parent.parent.parent.attrs["id"].encode("utf-8"),
                        group.parent.parent.attrs["id"].encode("utf-8"),
                        group.parent.attrs["id"].encode("utf-8"),
                        group.parent.parent.attrs["location.latitude"],
                        group.parent.parent.attrs["location.longitude"],
                        group.parent.parent.attrs["location.elevation"],
                        group.attrs["component"],
                        group.attrs["time_period.start"],
                        group.attrs["time_period.end"],
                        group.size,
                        group.attrs["sample_rate"],
                        group.attrs["type"],
                        group.attrs["measurement_azimuth"],
                        group.attrs["measurement_tilt"],
                        group.attrs["units"],
                        has_data(group),
                        group.ref,
                        group.parent.ref,
                        group.parent.parent.ref,
                    )
                ],
                dtype=dtype,
            )
            return ch_entry

        def recursive_get_channel_entry(group):
            """
            a function to get channel entry
            """
            if isinstance(group, (h5py._hl.group.Group, h5py._hl.files.File)):
                for key, node in group.items():
                    recursive_get_channel_entry(node)
            elif isinstance(group, h5py._hl.dataset.Dataset):
                try:
                    ch_type = group.attrs["type"]
                    if ch_type in ["electric", "magnetic", "auxiliary"]:
                        ch_entry = get_channel_entry(group)
                        try:
                            self.add_row(ch_entry)
                        except ValueError as error:
                            msg = (
                                f"{error}. "
                                "it is possible that the OS that made the table is not the OS operating on it."
                            )
                            self.logger.warning(msg)

                except KeyError:
                    pass

        recursive_get_channel_entry(self.array.parent)

    def to_run_summary(
        self,
        allowed_input_channels=ALLOWED_INPUT_CHANNELS,
        allowed_output_channels=ALLOWED_OUTPUT_CHANNELS,
        sortby=["station", "start"],
    ):
        """
        Method for compressing an mth5 channel_summary into a "run summary" which
        has one row per run (not one row per channel)

        Devlopment Notes:
        TODO: replace station_id with station, and run_id with run
        Note will need to modify: aurora/tests/config$ more test_dataset_dataframe.py
        TODO: Add logic for handling input and output channels based on channel
        summary.  Specifically, consider the case where there is no vertical magnetic
        field, this information is available via ch_summary, and output channels should
        then not include hz.
        TODO: Just inherit all the run-level and higher el'ts of the channel_summary,
        including n_samples?

        When creating the dataset dataframe, make it have these columns:
        [
                "channel_scale_factors",
                "duration",
                "end",
                "has_data",
                "input_channels",
                "mth5_path",
                "n_samples",
                "output_channels",
                "run",
                "sample_rate",
                "start",
                "station",
                "survey",
                "hdf5_reference",
                "run_hdf5_reference",
                "station_hdf5_reference",
            ]

        Parameters
        ----------
        ch_summary: mth5.tables.channel_table.ChannelSummaryTable or pandas DataFrame
           If its a dataframe it is a representation of an mth5 channel_summary.
            Maybe restricted to only have certain stations and runs before being passed to
            this method
        allowed_input_channels: list of strings
            Normally ["hx", "hy", ]
            These are the allowable input channel names for the processing.  See further
            note under allowed_output_channels.
        allowed_output_channels: list of strings
            Normally ["ex", "ey", "hz", ]
            These are the allowable output channel names for the processing.
            A global list of these is kept at the top of this module.  The purpose of
            this is to distinguish between runs that have different layouts, for example
            some runs will have hz and some will not, and we cannot process for hz the
            runs that do not have it.  By making this a kwarg we sort of prop the door
            open for more general names (see issue #74).
        sortby: bool or list
            Default: ["station_id", "start"]

        Returns
        -------
        run_summary_df: pd.Dataframe
            A table with one row per "acquistion run" that was in the input channel
            summary table
        """

        if not self._has_entries():
            self.summarize()
        ch_summary_df = self.to_dataframe()

        group_by_columns = ["survey", "station", "run"]
        grouper = ch_summary_df.groupby(group_by_columns)
        row_list = []
        for group_values, group in grouper:
            # for entry in group.itertuples():
            row = dict([(key, None) for key in RUN_SUMMARY_COLUMNS])
            row["survey"] = group.survey.iloc[0]
            row["station"] = group.station.iloc[0]
            row["run"] = group.run.iloc[0]
            row["start"] = group.start.iloc[0]
            row["end"] = group.end.iloc[0]
            row["sample_rate"] = group.sample_rate.iloc[0]
            # max
            row["n_samples"] = group.n_samples.max()
            channels_list = group.component.to_list()
            num_channels = len(channels_list)
            row["input_channels"] = [
                x for x in channels_list if x in allowed_input_channels
            ]
            row["output_channels"] = [
                x for x in channels_list if x in allowed_output_channels
            ]
            row["channel_scale_factors"] = dict(
                zip(channels_list, num_channels * [1.0])
            )
            row["has_data"] = True
            if False in group.has_data.values:
                row["has_data"] = False

            row["run_hdf5_reference"] = group.run_hdf5_reference.iloc[0]
            row["station_hdf5_reference"] = group.station_hdf5_reference.iloc[0]

            row_list.append(row)

        run_summary_df = pd.DataFrame(data=row_list)
        if sortby:
            run_summary_df.sort_values(by=sortby, inplace=True)

        # add durations
        timedeltas = run_summary_df.end - run_summary_df.start
        durations = [x.total_seconds() for x in timedeltas]
        run_summary_df["duration"] = durations

        return run_summary_df
