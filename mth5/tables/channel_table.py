# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:09:38 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import numpy as np
import h5py

from mth5 import CHANNEL_DTYPE
from mth5.tables import MTH5Table

from mt_metadata.transfer_functions import (
    ALLOWED_INPUT_CHANNELS,
    ALLOWED_OUTPUT_CHANNELS,
)

# =============================================================================


class ChannelSummaryTable(MTH5Table):
    """
    Object to hold the channel summary and provide some convenience functions
    like fill, to_dataframe ...

    """

    def __init__(self, hdf5_dataset):
        super().__init__(hdf5_dataset, CHANNEL_DTYPE)

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
                        group.parent.parent.parent.parent.attrs["id"],
                        group.parent.parent.attrs["id"],
                        group.parent.attrs["id"],
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

        ch_summary_df = self.to_dataframe()

        group_by_columns = ["survey", "station", "run"]
        grouper = ch_summary_df.groupby(group_by_columns)
        n_station_runs = len(grouper)
        survey_ids = n_station_runs * [None]
        station_ids = n_station_runs * [None]
        run_ids = n_station_runs * [None]
        start_times = n_station_runs * [None]
        end_times = n_station_runs * [None]
        sample_rates = n_station_runs * [None]
        n_samples = n_station_runs * [None]
        input_channels = n_station_runs * [None]
        output_channels = n_station_runs * [None]
        channel_scale_factors = n_station_runs * [None]
        valids = n_station_runs * [None]
        index = 0
        for group_values, group in grouper:
            group_info = dict(zip(group_by_columns, group_values))
            survey_ids[index] = group_info["survey"]
            station_ids[index] = group_info["station"]
            run_ids[index] = group_info["run"]
            start_times[index] = group.start.iloc[0]
            end_times[index] = group.end.iloc[0]
            sample_rates[index] = group.sample_rate.iloc[0]
            n_samples[index] = group.n_samples.iloc[0]
            channels_list = group.component.to_list()
            num_channels = len(channels_list)
            input_channels[index] = [
                x for x in channels_list if x in allowed_input_channels
            ]
            output_channels[index] = [
                x for x in channels_list if x in allowed_output_channels
            ]
            channel_scale_factors[index] = dict(
                zip(channels_list, num_channels * [1.0])
            )
            valids[index] = True
            if False in group.has_data.values:
                valids[index] = False

            index += 1

        data_dict = {}
        data_dict["survey"] = survey_ids
        data_dict["station"] = station_ids
        data_dict["run"] = run_ids
        data_dict["start"] = start_times
        data_dict["end"] = end_times
        data_dict["sample_rate"] = sample_rates
        data_dict["n_samples"] = n_samples
        data_dict["input_channels"] = input_channels
        data_dict["output_channels"] = output_channels
        data_dict["channel_scale_factors"] = channel_scale_factors
        data_dict["has_data"] = valids

        run_summary_df = pd.DataFrame(data=data_dict)
        if sortby:
            run_summary_df.sort_values(by=sortby, inplace=True)

        # add durations
        timedeltas = run_summary_df.end - run_summary_df.start
        durations = [x.total_seconds() for x in timedeltas]
        run_summary_df["duration"] = durations

        return run_summary_df
