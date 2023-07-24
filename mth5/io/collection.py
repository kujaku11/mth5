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
from loguru import logger


# =============================================================================


class Collection:
    """
    A general collection class to keep track of files with methods to create
    runs and run ids.

    """

    def __init__(self, file_path=None, **kwargs):

        self.logger = logger
        self.file_path = file_path
        self.file_ext = "*"

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
            "calibration_fn",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        lines = [f"Collection for file type {self.file_ext} in {self._file_path}"]

        return "\n".join(lines)

    def __repr__(self):
        return f"Collection({self.file_path})"

    @property
    def file_path(self):
        """
        Path object to file directory
        """
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        """
        :param file_path: path to files
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
        Get files with given extension. Uses Pathlib.Path.rglob, so it finds
        all files within the `file_path` by searching all sub-directories.

        :param extension: file extension(s)
        :type extension: string or list
        :return: list of files in the `file_path` with the given extensions
        :rtype: list of Path objects

        """

        if isinstance(extension, (list, tuple)):
            fn_list = []
            for ext in extension:
                fn_list += list(self.file_path.rglob(f"*.{ext}"))
        else:
            fn_list = list(self.file_path.rglob(f"*.{extension}"))
        return sorted(list(set(fn_list)))

    def to_dataframe(self):
        """
        Get a data frame of the file summary with column names:

            - **survey**: survey id
            - **station**: station id
            - **run**: run id
            - **start**: start time UTC
            - **end**: end time UTC
            - **channel_id**: channel id or list of channel id's in file
            - **component**: channel component or list of components in file
            - **fn**: path to file
            - **sample_rate**: sample rate in samples per second
            - **file_size**: file size in bytes
            - **n_samples**: number of samples in file
            - **sequence_number**: sequence number of the file
            - **instrument_id**: instrument id
            - **calibration_fn**: calibration file

        :return: summary table of file names,
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
        Set some of the columns in the dataframe to desired types

            - **start**: pandas.datetime
            - **end**: pandas.datetime
            - **instrument_id**: string
            - **calibration_fn**: string

        :param df: summary table
        :type df: :class:`pandas.DataFrame`
        :return: summary table with proper types
        :rtype: :class:`pandas.DataFrame`

        """

        df.start = pd.to_datetime(df.start, errors="coerce")
        df.end = pd.to_datetime(df.end, errors="coerce")
        df.instrument_id = df.instrument_id.astype(str)
        df.calibration_fn = df.calibration_fn.astype(str)

        return df

    def _sort_df(self, df, zeros):
        """
        sort to a given dataframe by start date and then by run name. The
        index is reset.

        :param df: summary table
        :type df: :class:`pandas.DataFrame`
        :param zeros: number of zeros in run id
        :type zeros: integer
        :return: summary table sorted by start time and run id
        :rtype: :class:`pandas.DataFrame`

        """

        df.sort_values(by=["start"], inplace=True)
        df.reset_index(inplace=True, drop=True)

        # assign run names
        df = self.assign_run_names(df, zeros=zeros)

        df.sort_values(by=["run", "start"], inplace=True)
        df.reset_index(inplace=True, drop=True)

        return df

    def get_runs(
        self,
        sample_rates,
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
        :param run_name_zeros: Number of zeros in the run name, defaults to 4
        :type run_name_zeros: integer, optional
        :return: List of run dataframes with only the first block of files
        :rtype: :class:`collections.OrderedDict`

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
                run_dict[station][run_id] = run_df
        return run_dict
