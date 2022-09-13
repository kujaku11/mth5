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
from mth5.io.nims import NIMS

# =============================================================================


class NIMSCollection(Collection):
    """
    Collection of NIMS files into runs.


    .. code-block:: python

        >>> from mth5.io.nims import LEMICollection
        >>> lc = NIMSCollection(r"/path/to/single/lemi/station")
        >>> lc.station_id = "mt001"
        >>> lc.survey_id = "test_survey"
        >>> run_dict = lc.get_runs(1)


    """

    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)
        self.file_ext = "bin"

        self.survey_id = "mt"

    def to_dataframe(
        self, sample_rates=[1], run_name_zeros=2, calibration_path=None
    ):
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

        dipole_list = []
        entries = []
        for fn in self.get_files(self.file_ext):
            nims_obj = NIMS(fn)
            nims_obj.read_header()

            entry = {}
            entry["survey"] = self.survey_id
            entry["station"] = nims_obj.station
            entry["run"] = nims_obj.run_id
            entry["start"] = nims_obj.start_time.isoformat()
            entry["end"] = nims_obj.end_time.isoformat()
            entry["channel_id"] = 1
            entry["component"] = ",".join(
                ["hx", "hy", "hz", "ex", "ey", "temperature"]
            )
            entry["fn"] = fn
            entry["sample_rate"] = nims_obj.sample_rate
            entry["file_size"] = nims_obj.file_size
            entry["n_samples"] = nims_obj.n_samples
            entry["sequence_number"] = 0
            entry["instrument_id"] = "NIMS"
            entry["calibration_fn"] = None

            entries.append(entry)

            dipole_list.append(nims_obj.ex_length)
            dipole_list.append(nims_obj.ey_length)

        # make pandas dataframe and set data types
        df = self._sort_df(
            self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros
        )

        return df

    def assign_run_names(self, df, zeros=2):
        """
        Assign run names assuming a row represents single station

        Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}.

        :param df: Dataframe with the appropriate columns
        :type df: :class:`pandas.DataFrame`
        :param zeros: number of zeros in run name, defaults to 4
        :type zeros: int, optional
        :return: Dataframe with run names
        :rtype: :class:`pandas.DataFrame`

        """

        for station in df.station.unique():
            count = 1
            for row in (
                df[df.station == station].sort_values("start").itertuples()
            ):
                if row.run is None:
                    df.loc[
                        row.Index, "run"
                    ] = f"sr{row.sample_rate}_{count:0{zeros}}"
                df.loc[row.Index, "sequence_number"] = count
                count += 1

        return df
