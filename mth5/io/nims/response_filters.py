# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:50:51 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from mt_metadata.timeseries.filters import PoleZeroFilter, TimeDelayFilter

# =============================================================================


class ResponseError(Exception):
    pass


class Response(object):
    """
    class for instrument response functions.

    """

    def __init__(self, system_id=None, **kwargs):
        self.system_id = system_id
        self.hardware = "PC"
        self.instrument_type = "backbone"
        self.sample_rate = 8
        self.e_conversion_factor = 2.44141221047903e-06
        self.h_conversion_factor = 0.01

        self.time_delays_dict = {
            "hp200": {
                "hx": -0.0055,
                "hy": -0.0145,
                "hz": -0.0235,
                "ex": 0.1525,
                "ey": 0.0275,
            },
            1: {
                "hx": -0.1920,
                "hy": -0.2010,
                "hz": -0.2100,
                "ex": -0.2850,
                "ey": -0.2850,
            },
            8: {
                "hx": 0.2455,
                "hy": 0.2365,
                "hz": 0.2275,
                "ex": 0.1525,
                "ey": 0.1525,
            },
        }
        self.mag_low_pass = PoleZeroFilter(
            name="3 pole butterworth",
            zeros=[0, 3, 1984.31],
            poles=[
                complex(-6.28319, 10.8825),
                complex(-6.28319, 10.8825),
                complex(-12.5664, 0),
            ],
        )

        self.electric_low_pass = PoleZeroFilter(
            name="5 pole butterworth",
            zeros=[0, 5, 313384],
            poles=[
                complex(-3.88301, 11.9519),
                complex(-3.88301, -11.9519),
                complex(-10.1662, 7.38651),
                complex(-10.1662, -7.38651),
                complex(-12.5664, 0.0),
            ],
        )
        self.electric_high_pass_pc = PoleZeroFilter(
            name="1 pole butterworth",
            zeros=[1, 1, 1],
            poles=[complex(0.0, 0.0), complex(-3.333333e-05, 0.0)],
            normalization_factor=2 * np.pi * 30000,
        )
        self.electric_high_pass_hp = PoleZeroFilter(
            name="1 pole butterworth",
            zeros=[1, 1, 1],
            poles=[complex(0.0, 0.0), complex(-1.66667e-04, 0.0)],
            normalization_factor=2 * np.pi * 6000,
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_electric_high_pass(self, hardware="pc"):
        """
        get the electric high pass filter based on the hardware
        """

        self.hardware = hardware
        if "pc" in hardware.lower():
            return self.electric_high_pass_pc
        elif "hp" in hardware.lower():
            return self.electric_high_pass_hp
        else:
            raise ResponseError(f"Hardware value {hardware} not understood")

    def _get_dt_filter(self, channel, sample_rate):
        """
        get the DT filter based on channel ans sampling rate
        """
        dt_filter = TimeDelayFilter(
            name="time_offset",
            delay=self.time_delays_dict[sample_rate][channel],
        )
        return dt_filter

    def _get_mag_filter(self, channel):
        """
        get mag filter, seems to be the same no matter what
        """
        filter_list = [self.mag_low_pass]
        filter_list.append(self._get_dt_filter(channel, self.sample_rate))

        return_dict = {
            "channel_id": channel,
            "gain": 1,
            "conversion_factor": self.h_conversion_factor,
            "units": "nT",
            "filters": filter_list,
        }
        return return_dict

    def _get_electric_filter(self, channel):
        """
        Get electric filter
        """
        filter_list = []
        if self.instrument_type in ["backbone"]:
            filter_list.append(self.get_electric_high_pass(self.hardware))
        filter_list.append(self.electric_low_pass)
        filter_list.append(self._get_dt_filter(channel, self.sample_rate))

        return_dict = {
            "channel_id": channel,
            "gain": 1,
            "conversion_factor": self.e_conversion_factor,
            "units": "nT",
            "filters": filter_list,
        }
        return return_dict

    @property
    def hx_filter(self):
        """HX filter"""

        return self._get_mag_filter("hx")

    @property
    def hy_filter(self):
        """HY Filter"""
        return self._get_mag_filter("hy")

    @property
    def hz_filter(self):
        return self._get_mag_filter("hz")

    @property
    def ex_filter(self):
        return self._get_electric_filter("ex")

    @property
    def ey_filter(self):
        return self._get_electric_filter("ey")
