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
import json

import numpy as np
from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

# =============================================================================


class PHXCalibration:
    def __init__(self, cal_fn=None, **kwargs):
        self.cal_fn = cal_fn
        self._raw_dict = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def cal_fn(self):
        return self._cal_fn

    @cal_fn.setter
    def cal_fn(self, cal_fn):
        if cal_fn is not None:
            self._cal_fn = Path(cal_fn)
            if self._cal_fn.exists():
                self.read()

    def _has_read(self):
        if self._raw_dict is not None:
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
                f"{self._raw_dict['instrument_type']}_"
                f"{self._raw_dict['instrument_model']}_"
                f"{self._raw_dict['inst_serial']}"
            )

    def get_filter_name(self, channel, max_freq):
        """
        get the filter name as

        {instrument_model}_{instrument_type}_{inst_serial}_{channel}_{max_freq}_lp

        :param raw_dict: DESCRIPTION
        :type raw_dict: TYPE
        :param max_freq: DESCRIPTION
        :type max_freq: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return f"{self.base_filter_name}_{channel}_{max_freq}hz_low_pass"

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

        with open(self.cal_fn, "r") as fid:
            self._raw_dict = json.load(fid)

        cal_dict = {}
        for channel in self._raw_dict["cal_data"]:
            comp = channel["tag"].lower()
            cal_dict[comp] = {}
            for cal in channel["chan_data"]:
                ch_fap = FrequencyResponseTableFilter(
                    frequencies=cal["freq_Hz"],
                    amplitudes=cal["magnitude"],
                    phases=cal["phs_deg"],
                )
                max_freq = self.get_max_freq(ch_fap.frequencies)
                ch_fap.name = self.get_filter_name(comp, max_freq)
                ch_fap.calibration_date = self._raw_dict["timestamp_utc"]
                cal_dict[comp][max_freq] = ch_fap
                ch_fap.units_in = "volts"
                ch_fap.units_out = "volts"


# =============================================================================
#
# =============================================================================
cal_fn = Path(
    r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\calibrations\10621_647A2F41.rxcal.json"
)

c = PHXCalibration(cal_fn)
