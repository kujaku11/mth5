# -*- coding: utf-8 -*-
"""
Phoenix file collection

Created on Thu Aug  4 16:48:47 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from mth5.utils.mth5_logger import setup_logger


# =============================================================================


class Collection:
    """
    A general collection class to keep track of files
    """

    def __init__(self, file_path=None, **kwargs):

        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.file_path = file_path

        self._columns = [
            "survey",
            "station",
            "run",
            "start",
            "end",
            "channel_id",
            "component",
            "fn",
            "sample_rate",
            "file_size",
            "n_samples",
            "sequence_number",
            "instrument_id",
            "calibraion_fn",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def file_path(self):
        """
        Path object to z3d directory
        """
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        """
        :param file_path: path to z3d files
        :type file_path: string or Path object

        sets file_path as a Path object
        """
        if file_path is None:
            self._file_path = None
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        self._file_path = file_path

        if not self._file_path.exists():
            raise IOError()

    def get_files(self, extension):
        """
        Get files with given extension

        :param extension: DESCRIPTION
        :type extension: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if isinstance(extension, (list, tuple)):
            fn_list = []
            for ext in extension:
                fn_list += list(self.file_path.rglob(f"*.{ext}"))

        else:
            fn_list = list(self.file_path.rglob(f"*.{extension}"))

        return fn_list

    def to_dataframe(self):
        """
        Get a data frame of the file summary

        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def assign_run_names(self):
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def _set_df_dtypes(self, df):
        """

        :param df: DESCRIPTION
        :type df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        df.start = pd.to_datetime(df.start, errors="coerce")
        df.end = pd.to_datetime(df.end, errors="coerce")
        df.instrument_id = df.instrument_id.astype(str)

        return df

    def _sort_df(self, df, zeros):
        """
        sort to a logical order

        :param df: DESCRIPTION
        :type df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        df.sort_values(by=["start"], inplace=True)

        # assign run names
        df = self.assign_run_names(df, zeros=zeros)

        df.sort_values(by=["run", "start"], inplace=True)
        df.reset_index(inplace=True, drop=True)

        return df

    def get_runs(
        self,
        sample_rates=[150, 24000],
        run_name_zeros=4,
        calibration_path=None,
    ):
        """
        Get a list of runs contained within the given folder.  First the
        dataframe will be developed from which the runs are extracted.

        For continous data all you need is the first file in the sequence. The
        reader will read in the entire sequence.

        For segmented data it will only read in the given segment, which is
        slightly different from the original reader.

        :param sample_rates: list of sample rates to read, defaults to [150, 24000]
        :type sample_rates: list of integers, optional
        :param run_name_zeros: Number of zeros in the run name, defaults to 4
        :type run_name_zeros: integer, optional
        :return: List of run dataframes with only the first block of files
        :rtype: OrderedDict

        :Example:

            >>> from mth5.io.phoenix import PhoenixCollection
            >>> phx_collection = PhoenixCollection(r"/path/to/station")
            >>> run_dict = phx_collection.get_runs(sample_rates=[150, 24000])

        """

        df = self.to_dataframe(
            sample_rates=sample_rates,
            run_name_zeros=run_name_zeros,
            calibration_path=calibration_path,
        )

        run_dict = OrderedDict()

        for station in sorted(df.station.unique()):
            run_dict[station] = OrderedDict()

            for run_id in sorted(
                df[df.station == station].run.unique(),
                key=lambda x: x[-run_name_zeros:],
            ):
                run_df = df[(df.station == station) & (df.run == run_id)]
                run_dict[station][run_id] = run_df[
                    run_df.start == run_df.start.min()
                ]

        return run_dict
