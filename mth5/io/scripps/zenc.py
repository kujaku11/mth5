# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:36:55 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from collections import OrderedDict
import numpy as np

from mth5.mth5 import MTH5
from mth5.timeseries import RunTS

from loguru import logger

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

    def __init__(self, channel_map):
        self.logger = logger
        self._channel_map_keys = [
            "survey",
            "station",
            "run",
            "channel",
            "channel_number",
        ]
        self._expected_channel_order = ["hx", "hy", "hz", "ex", "ey"]
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
        self._channel_map = self._sort_channel_map(value)

    def _sort_channel_map(self, channel_map):
        """
        sort by channel number

        :param channel_map: DESCRIPTION
        :type channel_map: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        sorted_channel_map = OrderedDict()

        for ch in self._expected_channel_order:
            try:
                sorted_channel_map[ch] = channel_map[ch]
            except KeyError:
                self.logger.info(
                    f"Could not find {ch} in channel_map, skipping"
                )

        return sorted_channel_map

    def to_zenc(self, mth5_file, channel_map=None):
        """
        write out a .zenc file

        :param mth5_file: DESCRIPTION
        :type mth5_file: TYPE
        :param channel_map: DESCRIPTION, defaults to None
        :type channel_map: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if channel_map is not None:
            self.channel_map = channel_map

        with MTH5() as m:
            m.open_mth5(mth5_file, mode="r")
            ch_list = []
            for key, ch_dict in self.channel_map.items():
                ch = m.get_channel(
                    ch_dict["station"],
                    ch_dict["run"],
                    ch_dict["channel"],
                    survey=ch_dict["survey"],
                )
                ch_list.append(ch.to_channel_ts())
            run = RunTS(ch_list)

        # write out file
        # write metadata
        # write data as (hx, hy, hz, ex, ey, ...)
