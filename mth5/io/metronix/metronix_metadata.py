# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:23:42 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import json
from pathlib import Path
from types import SimpleNamespace
from loguru import logger

from mt_metadata.timeseries import Magnetic, Electric
from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

# =============================================================================


class MetronixJSONBase:
    def __init__(self, fn=None, **kwargs):
        self.fn = fn
        if self.fn is not None:
            self.read(self.fn)

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value):
        if value is None:
            self._fn = None
        else:
            self._fn = Path(value)
            self._parse_fn(self._fn)

    def _parse_fn(self, fn):
        """
        need to parse metadata from the filename

        :param fn: DESCRIPTION
        :type fn: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if fn is None:
            return

        fn_list = fn.stem.split("_")
        self.system_number = fn_list[0]
        self.system_name = fn_list[1]
        self.channel_number = self._parse_channel_number(fn_list[2])
        self.component = self._parse_component(fn_list[3])
        self.sample_rate = self._parse_sample_rate(fn_list[4])

    def _parse_channel_number(self, value):
        """
        channel number is C## > 0

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return int(value.replace("C", "0"))

    def _parse_component(self, value):
        """
        component is T{comp}

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return value.replace("T", "").lower()

    def _parse_sample_rate(self, value):
        """
        sample rate is {sr}Hz or {sr}s

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if "hz" in value.lower():
            return float(value.lower().replace("hz", ""))
        elif "s" in value.lower():
            return 1.0 / float(value.lower().replace("s", ""))

    def read(self, fn=None):
        """

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if fn is not None:
            self.fn = fn

        with open(self.fn, "r") as fid:
            self.metadata = json.load(
                fid, object_hook=lambda d: SimpleNamespace(**d)
            )


class MetronixJSONMagnetic(MetronixJSONBase):
    def __init__(self, fn=None, **kwargs):
        super().__init__(fn=fn, **kwargs)

    def to_mt_metadata(self):
        """
        translate to `mt_metadata.timeseries.Magnetic` object

        :return: DESCRIPTION
        :rtype: TYPE

        """
        if self.metadata is None:
            raise ValueError("Metronix JSON file has not been read in yet.")
        magnetic = Magnetic(
            component=self.component,
            channel_number=self.channel_number,
        )

        return magnetic
