# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:22:44 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import pandas as pd

from mth5.io.collection import Collection
from mth5.io.metronix import ATSS

# =============================================================================


class MetronixCollection(Collection):
    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)
        self.file_ext = ["atss"]

    def to_dataframe(
        self, sample_rates=[128], run_name_zeros=0, calibration_path=None
    ):
        """
        Create dataframe for metronix timeseries atss + json file sets

        :param sample_rates: DESCRIPTION, defaults to [128]
        :type sample_rates: TYPE, optional
        :param run_name_zeros: DESCRIPTION, defaults to 4
        :type run_name_zeros: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        entries = []
        for atss_fn in set(self.get_files(self.file_ext)):
            atss_obj = ATSS(atss_fn)
            if not atss_obj.sample_rate in sample_rates:
                continue
            ch_metadata = atss_obj.channel_metadata

            entry = self.get_empty_entry_dict()
            entry["survey"] = atss_obj.survey_id
            entry["station"] = atss_obj.station_id
            entry["run"] = atss_obj.run_id
            entry["start"] = ch_metadata.time_period.start
            entry["end"] = ch_metadata.time_period.end
            entry["channel_id"] = atss_obj.channel_number
            entry["component"] = atss_obj.component
            entry["fn"] = atss_fn
            entry["sample_rate"] = ch_metadata.sample_rate
            entry["file_size"] = atss_obj.file_size
            entry["n_samples"] = atss_obj.n_samples
            entry["sequence_number"] = 0
            entry["dipole"] = 0
            if ch_metadata.type in ["magnetic"]:
                entry["coil_number"] = ch_metadata.sensor.id
                entry["latitude"] = ch_metadata.location.latitude
                entry["longitude"] = ch_metadata.location.longitude
                entry["elevation"] = ch_metadata.location.elevation
            else:
                entry["coil_number"] = None
                entry["latitude"] = ch_metadata.positive.latitude
                entry["longitude"] = ch_metadata.positive.longitude
                entry["elevation"] = ch_metadata.positive.elevation

            entry["instrument_id"] = atss_obj.system_number
            entry["calibration_fn"] = None
            entries.append(entry)
        # make pandas dataframe and set data types
        df = self._sort_df(
            self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros
        )

        return df

    def assign_run_names(self, df, zeros=0):
        """
        assign run names, if zeros is 0 then run name is unchanged, otherwise
        the run name will be `sr{sample_rate}_{run_number:zeros}

        :param df: DESCRIPTION
        :type df: TYPE
        :param zeros: DESCRIPTION, defaults to 0
        :type zeros: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if zeros == 0:
            return df

        for row in df.itertuples():
            df.loc[row.Index, "run"] = (
                f"sr{row.sample_rate:.0f}_{int(row.run.split('_')[1]):0{zeros}}"
            )
        return df
