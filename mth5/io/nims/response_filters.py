# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:50:51 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from mt_metadata.timeseries.filters import (
    PoleZeroFilter,
    TimeDelayFilter,
    CoefficientFilter,
)

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
                "hx": -0.2455,
                "hy": -0.2365,
                "hz": -0.2275,
                "ex": -0.1525,
                "ey": -0.1525,
            },
        }
        self.mag_low_pass = PoleZeroFilter(
            name="nims_3_pole_butterworth",
            zeros=[0, 3, 1984.31],
            poles=[
                complex(-6.28319, 10.8825),
                complex(-6.28319, 10.8825),
                complex(-12.5664, 0),
            ],
            units_in="volts",
            units_out="nanotesla",
            normalization_factor=2002.26936395594,
        )

        self.electric_low_pass = PoleZeroFilter(
            name="nims_5_pole_butterworth",
            zeros=[0, 5, 313384],
            poles=[
                complex(-3.88301, 11.9519),
                complex(-3.88301, -11.9519),
                complex(-10.1662, 7.38651),
                complex(-10.1662, -7.38651),
                complex(-12.5664, 0.0),
            ],
            units_in="volts",
            units_out="volts",
            normalization_frequency=313383.493219835,
        )
        self.electric_high_pass_pc = PoleZeroFilter(
            name="nims_1_pole_butterworth",
            zeros=[1, 1, 1],
            poles=[complex(0.0, 0.0), complex(-3.333333e-05, 0.0)],
            normalization_factor=1,
            units_in="volts",
            units_out="volts",
        )
        self.electric_high_pass_hp = PoleZeroFilter(
            name="nims_1_pole_butterworth",
            zeros=[1, 1, 1],
            poles=[complex(0.0, 0.0), complex(-1.66667e-04, 0.0)],
            normalization_factor=1,
            units_in="volts",
            units_out="volts",
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
            name=f"{channel}_time_offset",
            delay=self.time_delays_dict[sample_rate][channel],
            units_in="digital counts",
            units_out="digital counts",
        )
        return dt_filter

    def dipole_filter(self, length):
        """
        Make a dipole filter

        :param length: dipole length in meters
        :type length: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return CoefficientFilter(
            name=f"dipole_{length:.2f}",
            gain=length,
            units_out="volts per meter",
            units_in="volts",
        )

    def get_magnetic_filter(self, channel):
        """
        get mag filter, seems to be the same no matter what
        """

        conversion = CoefficientFilter(
            name="h_analog_to_digital",
            gain=self.h_conversion_factor,
            units_out="volts",
            units_in="digital counts",
        )
        return [
            self._get_dt_filter(channel, self.sample_rate),
            conversion,
            self.mag_low_pass,
        ]

    def get_electric_filter(self, channel, dipole_length):
        """
        Get electric filter
        """
        conversion = CoefficientFilter(
            name="e_analog_to_digital",
            gain=self.e_conversion_factor,
            units_out="volts",
            units_in="digital counts",
        )

        physical_units = CoefficientFilter(
            name="to_mt_units",
            gain=1e-6,
            units_out="millivolts per kilometer",
            units_in="volts per meter",
        )
        filter_list = [
            self._get_dt_filter(channel, self.sample_rate),
            conversion,
        ]
        if self.instrument_type in ["backbone"]:
            filter_list.append(self.get_electric_high_pass(self.hardware))
        filter_list.append(self.electric_low_pass)
        filter_list.append(self.dipole_filter(dipole_length))
        filter_list.append(physical_units)

        return filter_list
