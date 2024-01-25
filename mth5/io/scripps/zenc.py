# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:36:55 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from mth5.mth5 import MTH5

# =============================================================================


class ZENC:
    """
    Deal with .zenc files, which are apparently used to process data in EMTF.
    It was specifically built for processing ZEN data in EMTF, but should
    work regardless of data logger.

    The format is a header and then n_channels x n_samples of float32 values

    This class will read/write .zenc files.

    You need to input the path to an existing or new MTH5 file and a
    channel map to read/write.

    The `channel_map` needs to be in the form

    .. code-block::
        channel_map = {
            "channel_1_name":
                {"survey": survey_name,
                 "station": station_name,
                 "run": run_name,
                 "channel": channel_name,
                 "channel_number": channel_number},
            "channel_2_name":
                {"survey": survey_name,
                 "station": station_name,
                 "run": run_name,
                 "channel": channel_name,
                 "channel_number": channel_number},
            ...
                }


    """

    def __init__(self, mth5_file, channel_map):
        self._channel_map_keys = [
            "survey",
            "station",
            "run",
            "channel",
            "channel_number",
        ]
        self.mth5_file = mth5_file
        self.channel_map = channel_map

    @property
    def channel_map(self):
        return self._channel_map

    @channel_map.setter
    def channel_map(self, value):
        """
        need to make sure channel map is in the correct format

        :param value: dictionary of channels to use
        :type value: dict

        """

        if not isinstance(value, dict):
            raise ValueError(
                f"Input channel_map must be a dictionary not type{type(value)}"
            )

        for key, kdict in value.items():
            if not isinstance(kdict, dict):
                raise ValueError(
                    f"Input channel must be a dictionary not type{type(value)}"
                )
            if sorted(kdict.keys()) != sorted(self._channel_map_keys):
                raise KeyError(
                    f"Keys of channel dictionary must be {self._channel_map_keys} "
                    f"not {kdict.keys()}."
                )
        self._channel_map = value
