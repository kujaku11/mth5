# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:21:35 2023

@author: jpeacock

Calibrations can come in json files.  the JSON file includes filters
for all lowpass filters, so you need to match the lowpass filter used in the 
setup with the lowpass filter.  Then you need to add the dipole length and
sensor calibrations.
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np
from mt_metadata.timeseries.filters import FrequencyResponseTableFilter
from mt_metadata.utils.mttime import MTime

from .helpers import read_json_to_object

# =============================================================================


class PhoenixCalibration:
    def __init__(self, cal_fn=None, **kwargs):
        self.obj = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.cal_fn = cal_fn

    def __str__(self):
        lines = ["Phoenix Response Filters"]
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    @property
    def cal_fn(self):
        return self._cal_fn

    @cal_fn.setter
    def cal_fn(self, cal_fn):
        if cal_fn is not None:
            self._cal_fn = Path(cal_fn)
            if self._cal_fn.exists():
                self.read()
            else:
                raise IOError("Could not find file {cal_fn}.")

    @property
    def calibration_date(self):
        if self._has_read():
            return MTime(self.obj.timestamp_utc)

    def _has_read(self):
        if self.obj is not None:
            return True
        return False

    def get_max_freq(self, freq):
        """
        Name the filter {ch}_{max_freq}_lp_
        :param freq: DESCRIPTION
        :type freq: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return int(10 ** np.floor(np.log10(freq.max())))

    @property
    def base_filter_name(self):
        if self._has_read():
            return (
                f"{self.obj.instrument_type}_"
                f"{self.obj.instrument_model}_"
                f"{self.obj.inst_serial}"
            ).lower()

    def get_filter_lp_name(self, channel, max_freq):
        """
        get the filter name as

        {instrument_model}_{instrument_type}_{inst_serial}_{channel}_{max_freq}_lp

        :param channel: DESCRIPTION
        :type channel: TYPE
        :param max_freq: DESCRIPTION
        :type max_freq: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return f"{self.base_filter_name}_{channel}_{max_freq}hz_lowpass".lower()

    def get_filter_sensor_name(self, sensor):
        """
        get the filter name as

        {instrument_model}_{instrument_type}_{inst_serial}_{sensor}

        :param max_freq: DESCRIPTION
        :type max_freq: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return f"{self.base_filter_name}_{sensor}".lower()

    def read(self, cal_fn=None):
        """

        :param cal_fn: DESCRIPTION, defaults to None
        :type cal_fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if cal_fn is not None:
            self._cal_fn = Path(cal_fn)

        if not self.cal_fn.exists():
            raise IOError(f"Could not find {self.cal_fn}")

        self.obj = read_json_to_object(self.cal_fn)

        for channel in self.obj.cal_data:
            comp = channel.tag.lower()
            ch_cal_dict = {}
            for cal in channel.chan_data:
                ch_fap = FrequencyResponseTableFilter(
                    frequencies=cal.freq_Hz,
                    amplitudes=cal.magnitude,
                    phases=np.deg2rad(cal.phs_deg),
                )
                max_freq = self.get_max_freq(ch_fap.frequencies)
                if self.obj.file_type in ["receiver calibration"]:
                    ch_fap.name = self.get_filter_lp_name(comp, max_freq)
                else:
                    ch_fap.name = self.get_filter_sensor_name(
                        self.obj.sensor_serial
                    )
                ch_fap.calibration_date = self.obj.timestamp_utc
                ch_cal_dict[max_freq] = ch_fap
                ch_fap.units_in = "volts"
                ch_fap.units_out = "volts"

            if "sensor" in self.obj.file_type:
                ch_fap.units_in = "millivolts"
                ch_fap.units_out = "nanotesla"
                setattr(self, comp, ch_fap)

            else:
                setattr(self, comp, ch_cal_dict)

    def get_filter(self, channel, filter_name):
        """
        get the lowpass filter for the given channel and lowpass value

        :param channel: DESCRIPTION
        :type channel: TYPE
        :param lp_name: DESCRIPTION
        :type lp_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        try:
            filter_name = int(filter_name)
        except ValueError:
            pass

        try:
            return getattr(self, channel)[filter_name]
        except AttributeError:
            raise AttributeError(f"Could not find {channel}")
        except KeyError:
            raise KeyError(f"Could not find lowpass filter {filter_name}")
