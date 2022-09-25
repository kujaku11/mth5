# -*- coding: utf-8 -*-
"""
Adopted from TimeSeries reader, making all attributes properties for easier
reading and testing.

Module to read and parse native Phoenix Geophysics data formats of the MTU-5C Family

This module implements Streamed readers for segmented-decimated continuus-decimated
and native sampling rate time series formats of the MTU-5C family.

:author: Jorge Torres-Solis

Revised 2022 by J. Peacock 
"""

# =============================================================================
# Imports
# =============================================================================

from datetime import datetime

from struct import unpack_from
import string

from mt_metadata.timeseries import Station, Run, Electric, Magnetic
from mt_metadata.utils.mttime import MTime

# =============================================================================
class Header:
    """

    The header is 128 bytes with a specific format.  This reads in the 128
    bytes and provides properties to read each attribute of the header in
    the correct way.

    """

    def __init__(self, **kwargs):
        self.report_hw_sat = False
        self.header_length = 128
        self.ad_plus_minus_range = 5.0  # differential voltage range that the A/D can measure (Board model dependent)
        self._header = None
        self._recording_id = None
        self._channel_id = None

        self.channel_map = {
            0: "hx",
            1: "hy",
            2: "hz",
            3: "ex",
            4: "ey",
            5: "h1",
            6: "h2",
            7: "h3",
        }

        for key, value in kwargs.items():
            setattr(self, key, value)
        self._unpack_dict = {
            "file_type": {"dtype": "B", "index": 0},
            "file_version": {"dtype": "B", "index": 1},
            "header_length": {"dtype": "H", "index": 2},
            "instrument_type": {"dtype": "8s", "index": 4},
            "instrument_serial_number": {"dtype": "cccccccc", "index": 12},
            "recording_id": {"dtype": "I", "index": 20},
            "channel_id": {"dtype": "B", "index": 24},
            "file_sequence": {"dtype": "I", "index": 25},
            "frag_period": {"dtype": "H", "index": 29},
            "ch_board_model": {"dtype": "8s", "index": 31},
            "ch_board_serial": {"dtype": "8s", "index": 39},
            "ch_firmware": {"dtype": "I", "index": 47},
            "hardware_configuration": {"dtype": "BBBBBBBB", "index": 51},
            "sample_rate_base": {"dtype": "H", "index": 59},
            "sample_rate_exp": {"dtype": "b", "index": 61},
            "bytes_per_sample": {"dtype": "B", "index": 62},
            "frame_size": {"dtype": "I", "index": 63},
            "decimation_node_id": {"dtype": "H", "index": 67},
            "frame_rollover_count": {"dtype": "H", "index": 69},
            "gps_long": {"dtype": "f", "index": 71},
            "gps_lat": {"dtype": "f", "index": 75},
            "gps_elevation": {"dtype": "f", "index": 79},
            "gps_horizontal_accuracy": {"dtype": "I", "index": 83},
            "gps_vertical_accuracy": {"dtype": "I", "index": 87},
            "timing_status": {"dtype": "BBH", "index": 91},
            "future1": {"dtype": "b", "index": 95},
            "future2": {"dtype": "i", "index": 97},
            "saturated_frames": {"dtype": "H", "index": 101},
            "missing_frames": {"dtype": "H", "index": 103},
            "battery_voltage_mv": {"dtype": "H", "index": 105},
            "min_signal": {"dtype": "f", "index": 107},
            "max_signal": {"dtype": "f", "index": 111},
        }

    def __str__(self):
        lines = [
            f"channel_id: {self.channel_id}   channel_type: {self.channel_type}"
        ]
        lines += ["-" * 40]
        for key in [
            "instrument_type",
            "instrument_serial_number",
            "gps_lat",
            "gps_long",
            "gps_elevation",
            "recording_start_time",
            "sample_rate",
            "saturated_frames",
            "missing_frames",
            "max_signal",
            "min_signal",
        ]:
            lines.append(f"\t{key:<25}: {getattr(self, key)}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def _has_header(self):
        if self._header is not None:
            return True
        return False

    def _unpack_value(self, key):
        if self._has_header():
            return unpack_from(
                self._unpack_dict[key]["dtype"],
                self._header,
                self._unpack_dict[key]["index"],
            )

    @property
    def file_type(self):
        if self._has_header():
            return self._unpack_value("file_type")[0]

    @property
    def file_version(self):
        if self._has_header():
            return self._unpack_value("file_version")[0]

    @property
    def header_length(self):
        if self._has_header():
            self._header_length = self._unpack_value("header_length")[0]
        return self._header_length

    @header_length.setter
    def header_length(self, value):
        self._header_length = value

    @property
    def instrument_type(self):
        if self._has_header():
            return (
                self._unpack_value("instrument_type")[0]
                .decode("utf-8")
                .strip(" ")
                .strip("\x00")
            )

    @property
    def instrument_serial_number(self):
        if self._has_header():
            return (
                b"".join(self._unpack_value("instrument_serial_number"))
                .strip(b"\x00")
                .decode("utf-8")
            )

    @property
    def recording_id(self):
        if self._recording_id is None:
            if self._has_header():
                return self._unpack_value("recording_id")[0]
        else:
            return self._recording_id

    @recording_id.setter
    def recording_id(self, value):
        if isinstance(value, str):
            value = int(value, 16)
        self._recording_id = value

    @property
    def recording_start_time(self):
        """
        The actual data recording starts 1 second after the set start time.
        This is cause by the data logger starting up and initializing filter.
        This is taken care of in the segment start time

        See https://github.com/kujaku11/PhoenixGeoPy/tree/main/Docs for more
        information.

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return MTime(datetime.fromtimestamp(self.recording_id))

    @property
    def channel_id(self):
        if self._channel_id is None:
            if self._has_header():
                return int(self._unpack_value("channel_id")[0])
        else:
            return self._channel_id

    @channel_id.setter
    def channel_id(self, value):
        self._channel_id = int(value)

    @property
    def file_sequence(self):
        if self._has_header():
            return self._unpack_value("file_sequence")[0]

    @property
    def frag_period(self):
        if self._has_header():
            return self._unpack_value("frag_period")[0]

    @property
    def ch_board_model(self):
        if self._has_header():
            return (
                self._unpack_value("ch_board_model")[0]
                .decode("utf-8")
                .strip(" ")
            )

    @property
    def board_model_main(self):
        if self._has_header():
            return self.ch_board_model[0:5]

    @property
    def board_model_revision(self):
        if self._has_header():
            return self.ch_board_model[6:1]

    @property
    def ch_board_serial(self):
        if self._has_header():
            value = (
                self._unpack_value("ch_board_serial")[0]
                .decode("utf-8")
                .strip("\x00")
            )
            # handle the case of backend < v0.14, which puts '--------' in ch_ser
            if all(chars in string.hexdigits for chars in value):
                return int(value, 16)
            else:
                return 0

    @property
    def ch_firmware(self):
        if self._has_header():
            return self._unpack_value("ch_firmware")[0]

    @property
    def hardware_configuration(self):
        if self._has_header():
            return self._unpack_value("hardware_configuration")

    @property
    def channel_type(self):
        if self._has_header():
            if self.hardware_configuration[1] & 0x08 == 0x08:
                return "E"
            else:
                return "H"

    @property
    def detected_channel_type(self):
        # Channel type detected by electronics
        # this normally matches self.channel_type, but used in electronics design and testing
        if self._has_header():
            if self.hardware_configuration[1] & 0x20 == 0x20:
                return "E"
            else:
                return "H"

    @property
    def lp_frequency(self):
        if self._has_header():
            # LPF on
            if self.hardware_configuration[0] & 0x80 == 0x80:
                if self.hardware_configuration[0] & 0x03 == 0x03:
                    return 10
                elif self.hardware_configuration[0] & 0x03 == 0x02:
                    if (
                        self.board_model_main == "BCM03"
                        or self.board_model_main == "BCM06"
                    ):
                        return 1000
                    else:
                        return 100
                elif self.hardware_configuration[0] & 0x03 == 0x01:
                    if (
                        self.board_model_main == "BCM03"
                        or self.board_model_main == "BCM06"
                    ):
                        return 10000
                    else:
                        return 1000
            # LPF off
            else:
                if (
                    self.board_model_main == "BCM03"
                    or self.board_model_main == "BCM06"
                ):
                    return 17800
                else:
                    return 10000

    @property
    def preamp_gain(self):
        preamp_gain = 1.0
        if self._has_header():
            if self.channel_type == "?":
                raise Exception(
                    "Channel type must be set before attemting to calculate preamp gain"
                )
            preamp_on = bool(self.hardware_configuration[0] & 0x10)
            if self.channel_type == "E":
                if preamp_on:
                    if (
                        self.board_model_main == "BCM01"
                        or self.board_model_main == "BCM03"
                    ):
                        preamp_gain = 4.0
                        if self.board_model_revision == "L":
                            # Account for BCM01-L experimental prototype
                            preamp_gain = 8.0
                    else:
                        preamp_gain = 8.0
                        # Acount for experimental prototype BCM05-A
                        if self.ch_board_model[0:7] == "BCM05-A":
                            preamp_gain = 4.0
        return preamp_gain

    @property
    def channel_main_gain(self):
        # The value of the main gain of the board
        main_gain = 1
        if self._has_header():
            # BCM05-B and BCM06 introduced different selectable gains
            new_gains = (
                True  # we asume any newer board will have the new gain banks
            )
            if (
                self.board_model_main == "BCM01"
                or self.board_model_main == "BCM03"
            ):
                # Original style 24 KSps boards and original 96 KSps boards
                new_gains = False
            if self.ch_board_model[0:7] == "BCM05-A":
                # Acount for experimental prototype BCM05-A, which also had original gain banks
                new_gains = False
            if self.hardware_configuration[0] & 0x0C == 0x00:
                main_gain = 1.0
            elif self.hardware_configuration[0] & 0x0C == 0x04:
                main_gain = 4.0
            elif self.hardware_configuration[0] & 0x0C == 0x08:
                main_gain = 6.0
                if not new_gains:
                    main_gain = 16.0
            elif self.hardware_configuration[0] & 0x0C == 0x0C:
                main_gain = 8.0
                if not new_gains:
                    main_gain = 32.0
        return main_gain

    @property
    def intrinsic_circuitry_gain(self):
        """
        This function will adjust the intrinsic circuitry gain based on the
        sensor range configuration in the configuration fingerprint

        For this, we consider that for the Electric channel, calibration path, or H-legacy
        sensors all go through a 1/4 gain stage, and then they get a virtial x2 gain from
        Single-ended-diff before the A/D. In the case of newer sensors (differential)
        instead of a 1/4 gain stage, there is only a 1/2 gain stage

        Therefore, in the E,cal and legacy sensor case the circuitry gain is 1/2, while for
        newer sensors it is 1

        .. note:: Circuitry Gain not directly configurable by the user

        """

        intrinsic_circuitry_gain = 0.5
        if self._has_header():
            if self.channel_type == "?":
                raise Exception(
                    "Channel type must be set before attemting to calculate preamp gain"
                )
            intrinsic_circuitry_gain = 0.5
            if self.channel_type == "H":
                if self.hardware_configuration[1] & 0x01 == 0x01:
                    intrinsic_circuitry_gain = 1.0
        return intrinsic_circuitry_gain

    @property
    def attenuator_gain(self):
        # Asume attenuator off
        attenuator_gain = 1.0
        if self._has_header():
            if self.channel_type == "?":
                raise Exception(
                    "Channel type must be set before attemting to calculate preamp gain"
                )
            attenuator_on = bool(self.hardware_configuration[4] & 0x01)
            if attenuator_on and self.channel_type == "E":
                new_attenuator = True  # By default assume that we are dealing with a newer types of boards
                if (
                    self.board_model_main == "BCM01"
                    or self.board_model_main == "BCM03"
                ):
                    # Original style 24 KSps boards and original 96 KSps boards
                    new_attenuator = False
                if self.ch_board_model[0:7] == "BCM05-A":
                    # Acount for experimental prototype BCM05-A, which also had original gain banks
                    new_attenuator = False
                if new_attenuator:
                    attenuator_gain = 523.0 / 5223.0
                else:
                    attenuator_gain = 0.1
        return attenuator_gain

    # Board-wide gains
    @property
    def total_selectable_gain(self):
        # Total of the gain that is selectable by the user (i.e. att * pre * gain)
        if self._has_header():
            return (
                self.channel_main_gain
                * self.preamp_gain
                * self.attenuator_gain
            )
        return 1.0

    @property
    def total_circuitry_gain(self):
        # Total board Gain both intrinsic gain and user-seletable gain in circuit
        if self._has_header():
            return self.total_selectable_gain * self.intrinsic_circuitry_gain
        return 0.5

    @property
    def sample_rate_base(self):
        if self._has_header():
            return self._unpack_value("sample_rate_base")[0]

    @property
    def sample_rate_exp(self):
        if self._has_header():
            return self._unpack_value("sample_rate_exp")[0]

    @property
    def sample_rate(self):
        if self._has_header():
            if self.sample_rate_exp != 0:
                return self.sample_rate_base * pow(10, self.sample_rate_exp)
            return self.sample_rate_base

    @property
    def bytes_per_sample(self):
        if self._has_header():
            return self._unpack_value("bytes_per_sample")[0]

    @property
    def frame_size(self):
        if self._has_header():
            return self._unpack_value("frame_size")[0]

    @property
    def data_footer(self):
        if self._has_header():
            return self.frame_size >> 24

    @property
    def frame_size_bytes(self):
        if self._has_header():
            return self.frame_size & 0x0FFFFFF

    @property
    def decimation_node_id(self):
        if self._has_header():
            return self._unpack_value("decimation_node_id")[0]

    @property
    def frame_rollover_count(self):
        if self._has_header():
            return self._unpack_value("frame_rollover_count")[0]

    @property
    def gps_long(self):
        if self._has_header():
            return self._unpack_value("gps_long")[0]

    @property
    def gps_lat(self):
        if self._has_header():
            return self._unpack_value("gps_lat")[0]

    @property
    def gps_elevation(self):
        if self._has_header():
            return self._unpack_value("gps_elevation")[0]

    @property
    def gps_horizontal_accuracy(self):
        """
        In millimeters

        """
        if self._has_header():
            return self._unpack_value("gps_horizontal_accuracy")[0] / 1000

    @property
    def gps_vertical_accuracy(self):
        """
        In millimeters

        """
        if self._has_header():
            return self._unpack_value("gps_vertical_accuracy")[0] / 1000

    @property
    def timing_status(self):
        if self._has_header():
            return self._unpack_value("timing_status")

    @property
    def timing_flags(self):
        if self._has_header():
            return self.timing_status[0]

    @property
    def timing_sat_count(self):
        if self._has_header():
            return self.timing_status[1]

    @property
    def timing_stability(self):
        if self._has_header():
            return self.timing_status[2]

    @property
    def future1(self):
        if self._has_header():
            return self._unpack_value("future1")[0]

    @property
    def future2(self):
        if self._has_header():
            return self._unpack_value("future2")[0]

    @property
    def saturated_frames(self):
        if self._has_header():
            value = self._unpack_value("saturated_frames")[0]
            if value & 0x80 == 0x80:
                value &= 0x7F
                value <<= 4
            return value

    @property
    def missing_frames(self):
        if self._has_header():
            return self._unpack_value("missing_frames")[0]

    @property
    def battery_voltage_v(self):
        if self._has_header():
            return self._unpack_value("battery_voltage_mv")[0] / 1000

    @property
    def min_signal(self):
        if self._has_header():
            return self._unpack_value("min_signal")[0]

    @property
    def max_signal(self):
        if self._has_header():
            return self._unpack_value("max_signal")[0]

    def unpack_header(self, stream):
        if self.header_length > 0:
            # be sure to read from the beginning of the file
            stream.seek(0)
            self._header = stream.read(self.header_length)
        else:
            return

    def channel_metadata(self):
        """
        translate metadata to channel metadata
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.channel_type.lower() in ["h"]:
            ch = Magnetic()
        elif self.channel_type.lower() in ["e"]:
            ch = Electric()
        try:
            ch.component = self.channel_map[self.channel_id]
        except KeyError:
            print(f"Could not find {self.channel_id} in channel_map")
        ch.channel_number = self.channel_id
        ch.time_period.start = self.recording_start_time
        ch.sample_rate = self.sample_rate

        return ch

    def run_metadata(self):
        """
        translate to run metadata

        :return: DESCRIPTION
        :rtype: TYPE

        """

        r = Run()
        r.data_logger.type = self.instrument_type
        r.data_logger.id = self.instrument_serial_number
        r.data_logger.manufacturer = "Phoenix Geophysics"
        r.data_logger.timing_system.uncertainty = self.timing_stability
        r.sample_rate = self.sample_rate
        r.data_logger.power_source.voltage.start = self.battery_voltage_v

        return r

    def station_metadata(self):
        """
        translate to station metadata

        """

        s = Station()
        s.location.latitude = self.gps_lat
        s.location.longitude = self.gps_long
        s.location.elevation = self.gps_elevation

        return s
