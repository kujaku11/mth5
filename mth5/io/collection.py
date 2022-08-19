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

# =============================================================================


class Collection:
    """
    A general collection class to keep track of files
    """

    def __init__(self, station_path, **kwargs):

        self.station_path = Path(station_path)

        self._columns = [
            "survey",
            "station",
            "run",
            "start",
            "end",
            "channel",
            "fn",
            "sample_rate",
        ]

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
