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
from mth5.io.usgs_ascii import USGSascii

# =============================================================================


class USGSasciiCollection(Collection):
    """
    Collection of USGS ASCII files.

    .. code-block:: python

        >>> from mth5.io.usgs_ascii import USGSasciiCollection
        >>> lc = USGSasciiCollection(r"/path/to/ascii/files")
        >>> run_dict = lc.get_runs(1)


    """

    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)
        self.file_ext = "asc"

    def to_dataframe(
        self, sample_rates=[4], run_name_zeros=4, calibration_path=None
    ):
        """
        Create a data frame of each TXT file in a given directory.

        .. note:: If a run name is already present it will not be overwritten

        :param sample_rates: sample rate to get, defaults to [4]
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

            >>> from mth5.io.usgs_ascii import USGSasciiCollection
            >>> lc = USGSasciiCollection("/path/to/ascii/files")
            >>> ascii_df = lc.to_dataframe()

        """

        entries = []
        for fn in self.get_files(self.file_ext):
            asc_obj = USGSascii(fn)
            asc_obj.read_metadata()

            entry = self.get_empty_entry_dict()
            entry["survey"] = asc_obj.survey_metadata.id
            entry["station"] = asc_obj.station_metadata.id
            entry["run"] = asc_obj.run_metadata.id
            entry["start"] = asc_obj.start
            entry["end"] = asc_obj.end
            entry["channel_id"] = 1
            entry["component"] = ",".join(
                asc_obj.run_metadata.channels_recorded_all
            )
            entry["fn"] = fn
            entry["sample_rate"] = asc_obj.sample_rate
            entry["file_size"] = asc_obj.file_size
            entry["n_samples"] = int(asc_obj.n_samples)
            entry["sequence_number"] = 0
            entry["instrument_id"] = asc_obj.run_metadata.data_logger.id
            entry["calibration_fn"] = None

            entries.append(entry)

        # make pandas dataframe and set data types
        df = self._sort_df(
            self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros
        )

        return df

    def assign_run_names(self, df, zeros=4):
        """
        Assign run names based on start and end times, checks if a file has
        the same start time as the last end time.

        Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}. Only
        if the run name is not assigned already.

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
