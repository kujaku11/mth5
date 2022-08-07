# -*- coding: utf-8 -*-
"""
Phoenix file collection

Created on Thu Aug  4 16:48:47 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import pandas as pd
from mth5.io.phoenix import read_phoenix, open_file

# =============================================================================


class PhoenixCollection:
    """
    A class to collect the various files in a Phoenix file system and try
    to organize them into runs.
    """

    def __init__(self, station_path, **kwargs):

        self.station_path = Path(station_path)
        self.channel_map = None

        self._columns = [
            "station",
            "run",
            "start",
            "end",
            "channel",
            "fn",
            "sample_rate",
        ]

        self._file_extension_map = {
            30: "td_30",
            150: "td_150",
            2400: "td_2400",
            24000: "td_24k",
            96000: "td_96k",
        }

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_files(self, extension):
        """
        Get files with given extension

        :param extension: DESCRIPTION
        :type extension: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return list(self.station_path.rglob(f"*.{extension}"))

    @property
    def sr30_file_list(self):
        return self._get_files(self._file_extension_map[30])

    @property
    def sr150_file_list(self):
        return self._get_files(self._file_extension_map[150])

    @property
    def sr2400_file_list(self):
        return self._get_files(self._file_extension_map[2400])

    @property
    def sr24k_file_list(self):
        return self._get_files(self._file_extension_map[24000])

    @property
    def sr96k_file_list(self):
        return self._get_files(self._file_extension_map[96000])

    def get_df(self, sample_rates=[150, 24000]):
        """
        Get a data frame with columns of the specified
        :param sample_rates: DESCRIPTION, defaults to [150, 24000]
        :type sample_rates: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """


# =============================================================================
# test
# =============================================================================

pc = PhoenixCollection(
    r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\10291_2019-09-06-015630"
)
