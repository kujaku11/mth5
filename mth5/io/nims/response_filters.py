# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:50:51 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from mt_metadata.timeseries.filters import (
    ChannelResponse,
    CoefficientFilter,
    PoleZeroFilter,
    TimeDelayFilter,
)


# =============================================================================


class ResponseError(Exception):
    pass


class Response(object):
    """
    Common NIMS response filters for electric and magnetic channels

    """

    def __init__(self, system_id=None, **kwargs):
        self.system_id = system_id
        self.hardware = "PC"
        self.instrument_type = "backbone"
        self.sample_rate = 1

        self.e_conversion_factor = 409600000.0
        self.h_conversion_factor = 100

        self.time_delays_dict = {
            "hp200": {
                "hx": -0.0055,
                "hy": -0.0145,
                "hz": -0.0235,
                "ex": -0.1525,
                "ey": -0.0275,
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

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def magnetic_low_pass(self):
        """
        Low pass 3 pole filter

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return PoleZeroFilter(
            name="nims_3_pole_butterworth",
            zeros=[],
            poles=[
                complex(-6.283185, 10.882477),
                complex(-6.283185, -10.882477),
                complex(-12.566371, 0),
            ],
            units_out="volts",
            units_in="nanotesla",
            normalization_factor=2002.26936395594,
        )

    @property
    def magnetic_conversion(self):
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return CoefficientFilter(
            name="h_analog_to_digital",
            gain=self.h_conversion_factor,
            units_in="volts",
            units_out="digital counts",
        )

    @property
    def electric_low_pass(self):
        """
        5 pole electric low pass filter
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return PoleZeroFilter(
            name="nims_5_pole_butterworth",
            zeros=[],
            poles=[
                complex(-3.88301, 11.9519),
                complex(-3.88301, -11.9519),
                complex(-10.1662, 7.38651),
                complex(-10.1662, -7.38651),
                complex(-12.5664, 0.0),
            ],
            units_in="volts",
            units_out="volts",
            normalization_factor=313383.493219835,
        )

    @property
    def electric_high_pass_pc(self):
        """
        1-pole low pass filter for 8 hz instruments
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return PoleZeroFilter(
            name="nims_1_pole_butterworth",
            zeros=[complex(0.0, 0.0)],
            poles=[complex(-3.333333e-05, 0.0)],
            normalization_factor=1,
            units_in="volts",
            units_out="volts",
        )

    @property
    def electric_high_pass_hp(self):
        """
        1-pole low pass for 1 hz instuments
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return PoleZeroFilter(
            name="nims_1_pole_butterworth",
            zeros=[complex(0.0, 0.0)],
            poles=[complex(-1.66667e-04, 0.0)],
            normalization_factor=1,
            units_in="volts",
            units_out="volts",
        )

    @property
    def electric_conversion(self):
        """
        electric channel conversion from counts to volts
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return CoefficientFilter(
            name="e_analog_to_digital",
            gain=self.e_conversion_factor,
            units_in="volts",
            units_out="digital counts",
        )

    @property
    def electric_physical_units(self):
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return CoefficientFilter(
            name="to_mt_units",
            gain=1e-6,
            units_in="millivolts per kilometer",
            units_out="volts per meter",
        )

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
            units_in="volts per meter",
            units_out="volts",
        )

    def _get_magnetic_filter(self, channel):
        """
        get mag filter, seems to be the same no matter what
        """

        return [
            self.magnetic_low_pass,
            self.magnetic_conversion,
            self._get_dt_filter(channel, self.sample_rate),
        ]

    def _get_electric_filter(self, channel, dipole_length):
        """
        Get electric filter
        """

        filter_list = []
        filter_list.append(self.electric_physical_units)
        filter_list.append(self.dipole_filter(dipole_length))
        filter_list.append(self.electric_low_pass)
        if self.instrument_type in ["backbone"]:
            filter_list.append(self.get_electric_high_pass(self.hardware))

        filter_list.append(self.electric_conversion)
        filter_list.append(self._get_dt_filter(channel, self.sample_rate))

        return filter_list

    def get_channel_response(self, channel, dipole_length=1):
        """
        Get the full channel response filter
        :param channel: DESCRIPTION
        :type channel: TYPE
        :param dipole_length: DESCRIPTION, defaults to 1
        :type dipole_length: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if channel[0] in ["e"]:
            return ChannelResponse(
                filters_list=self._get_electric_filter(channel, dipole_length)
            )

        elif channel[0] in ["b", "h"]:
            return ChannelResponse(filters_list=self._get_magnetic_filter(channel))

        else:
            raise ValueError(f"Channel {channel} not supported.")
