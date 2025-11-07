# -*- coding: utf-8 -*-
"""
LEMI 424 Collection
====================

Collection of TXT files combined into runs

Created on Wed Aug 31 10:32:44 2022

@author: jpeacock
"""

import pathlib
from typing import Optional

# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from mth5.io.collection import Collection
from mth5.io.lemi import LEMI424


# =============================================================================


class LEMICollection(Collection):
    """
    Collection of LEMI 424 files into runs based on start and end times.
    Will assign the run name as 'sr1_{index:0{zeros}}' --> 'sr1_0001' for
    `zeros` = 4.

    :param file_path: full path to single station LEMI424 directory
    :type file_path: string or :class`pathlib.Path`
    :param file_ext: extension of LEMI424 files, default is 'txt'
    :type file_ext: string
    :param station_id: station id
    :type station_id: string
    :param survey_id: survey id
    :type survey_id: string

    .. note:: This class assumes that the given file path contains a single
     LEMI station.  If you want to do multiple stations merge the returned
     data frames.

    .. note:: LEMI data comes with little metadata about the station or survey,
     therefore you should assign `station_id` and `survey_id`.

    .. code-block:: python

        >>> from mth5.io.lemi import LEMICollection
        >>> lc = LEMICollection(r"/path/to/single/lemi/station")
        >>> lc.station_id = "mt001"
        >>> lc.survey_id = "test_survey"
        >>> run_dict = lc.get_runs(1)


    """

    def __init__(
        self,
        file_path: Optional[pathlib.Path] = None,
        file_ext: Optional[list] = ["txt", "TXT"],
        **kwargs,
    ):
        super().__init__(file_path=file_path, file_ext=file_ext, **kwargs)

        self.station_id = "mt001"
        self.survey_id = "mt"

    def to_dataframe(self, sample_rates=[1], run_name_zeros=4, calibration_path=None):
        """
        Create a data frame of each TXT file in a given directory.

        .. note:: This assumes the given directory contains a single station

        :param sample_rates: sample rate to get, will always be 1 for LEMI data
         defaults to [1]
        :type sample_rates: int or list, optional
        :param run_name_zeros: number of zeros to assing to the run name,
         defaults to 4
        :type run_name_zeros: int, optional
        :param calibration_path: path to calibration files, defaults to None
        :type calibration_path: string or Path, optional
        :return: Dataframe with information of each TXT file in the given
         directory.
        :rtype: :class:`pandas.DataFrame`

        :Example:

            >>> from mth5.io.lemi import LEMICollection
            >>> lc = LEMICollection("/path/to/single/lemi/station")
            >>> lemi_df = lc.to_dataframe()

        """

        entries = []
        for fn in self.get_files(self.file_ext):
            lemi_obj = LEMI424(fn)
            n_samples = int(lemi_obj.n_samples)
            lemi_obj.read_metadata()

            entry = self.get_empty_entry_dict()
            entry["survey"] = self.survey_id
            entry["station"] = self.station_id
            entry["start"] = lemi_obj.start.isoformat()
            entry["end"] = lemi_obj.end.isoformat()
            entry["component"] = ",".join(lemi_obj.run_metadata.channels_recorded_all)
            entry["fn"] = fn
            entry["sample_rate"] = lemi_obj.sample_rate
            entry["file_size"] = lemi_obj.file_size
            entry["n_samples"] = n_samples

            entries.append(entry)

        # make pandas dataframe and set data types
        if len(entries) == 0:
            self.logger.warning("No entries found for LEMI collection")
            return pd.DataFrame()

        df = pd.DataFrame(entries)
        df.loc[:, "channel_id"] = 1
        df.loc[:, "sequence_number"] = 0
        df.loc[:, "instrument_id"] = "LEMI424"

        df = self._sort_df(self._set_df_dtypes(df), run_name_zeros)

        return df

    def assign_run_names(self, df, zeros=4):
        """
        Assign run names based on start and end times, checks if a file has
        the same start time as the last end time.

        Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}.

        :param df: Dataframe with the appropriate columns
        :type df: :class:`pandas.DataFrame`
        :param zeros: number of zeros in run name, defaults to 4
        :type zeros: int, optional
        :return: Dataframe with run names
        :rtype: :class:`pandas.DataFrame`

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
