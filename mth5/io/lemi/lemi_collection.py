# -*- coding: utf-8 -*-
"""
LEMI 424 Collection
====================

Collection of TXT files combined into runs

Created on Wed Aug 31 10:32:44 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from mth5.io.collection import Collection
from mth5.io.lemi import LEMI424

# =============================================================================


class LEMICollection(Collection):
    """
    Collection of LEMI 424 files into runs
    """

    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)
        self.file_ext = "txt"

        self.station_id = "mt001"
        self.survey_id = "mt"

    def to_dataframe(
        self, sample_rates=[1], run_name_zeros=4, calibration_path=None
    ):
        """

        :param sample_rates: DESCRIPTION, defaults to [1]
        :type sample_rates: TYPE, optional
        :param run_name_zeros: DESCRIPTION, defaults to 4
        :type run_name_zeros: TYPE, optional
        :param calibration_path: DESCRIPTION, defaults to None
        :type calibration_path: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        entries = []
        for fn in self.get_files(self.file_ext):
            lemi_obj = LEMI424(fn)
            n_samples = int(lemi_obj.n_samples)
            lemi_obj.read_metadata()

            entry = {}
            entry["survey"] = self.survey_id
            entry["station"] = self.station_id
            entry["run"] = None
            entry["start"] = lemi_obj.start.isoformat()
            entry["end"] = lemi_obj.end.isoformat()
            entry["channel_id"] = 1
            entry["component"] = ",".join(
                lemi_obj.run_metadata.channels_recorded_all
            )
            entry["fn"] = fn
            entry["sample_rate"] = lemi_obj.sample_rate
            entry["file_size"] = lemi_obj.file_size
            entry["n_samples"] = n_samples
            entry["sequence_number"] = 0
            entry["instrument_id"] = "LEMI424"
            entry["calibration_fn"] = None

            entries.append(entry)

        # make pandas dataframe and set data types
        df = self._sort_df(
            self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros
        )

        return df

    def assign_run_names(self, df, zeros=4):
        """
        Assign run names based on start and end times

        :param df: DESCRIPTION
        :type df: TYPE
        :param zeros: DESCRIPTION, defaults to 4
        :type zeros: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        count = 1
        for row in df.itertuples():
            if row.Index == 0:
                df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
                previous_end = row.end
            else:
                if (
                    row.start - previous_end
                ).total_seconds() / row.sample_rate == row.sample_rate:
                    df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
                else:
                    count += 1
                    df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
                previous_end = row.end

        return df
