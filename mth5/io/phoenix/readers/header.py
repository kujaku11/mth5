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
from __future__ import annotations

import string
from struct import unpack_from
from typing import Any, BinaryIO, TYPE_CHECKING

from loguru import logger
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries import Electric, Magnetic, Run, Station


if TYPE_CHECKING:
    pass

    from loguru import Logger


# =============================================================================
class Header:
    """
    Phoenix Geophysics MTU-5C binary header reader and parser.

    This class reads and parses the 128-byte binary header from Phoenix
    Geophysics MTU-5C data files. The header contains instrument configuration,
    GPS location, timing information, and recording parameters essential for
    proper data interpretation.

    The header format is fixed at 128 bytes and contains information about:
    - Instrument type and serial number
    - Recording parameters (sample rate, channel configuration)
    - GPS location and timing information
    - Hardware configuration and gain settings
    - Data quality metrics (saturated/missing frames)

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments to set as instance attributes.

    Attributes
    ----------
    logger : loguru.Logger
        Logger instance for debugging and error reporting.
    report_hw_sat : bool, default False
        Flag to control hardware saturation reporting.
    header_length : int, default 128
        Length of the binary header in bytes.
    ad_plus_minus_range : float, default 5.0
        Differential voltage range of the A/D converter (board dependent).
    channel_map : dict[int, str]
        Mapping from channel IDs to channel names.
    channel_azimuths : dict[str, int]
        Mapping from channel names to azimuth angles in degrees.

    Examples
    --------
    >>> with open("phoenix_data.bin", "rb") as f:
    ...     header = Header()
    ...     header.unpack_header(f)
    ...     print(f"Sample rate: {header.sample_rate}")
    ...     print(f"GPS location: {header.gps_lat}, {header.gps_long}")
    """

    def __init__(self, **kwargs: Any) -> None:
        self.logger: Logger = logger
        self.report_hw_sat: bool = False
        self.header_length: int = 128
        self.ad_plus_minus_range: float = 5.0  # differential voltage range that the A/D can measure (Board model dependent)
        self._header: bytes | None = None
        self._recording_id: int | None = None
        self._channel_id: int | None = None

        self.channel_map: dict[int, str] = {
            0: "h1",
            1: "h2",
            2: "h3",
            3: "e1",
            4: "e2",
            5: "h1",
            6: "h2",
            7: "h3",
        }

        self.channel_azimuths: dict[str, int] = {
            "h1": 0,
            "h2": 90,
            "h3": 0,
            "h4": 0,
            "h5": 90,
            "h6": 0,
            "e1": 0,
            "e2": 90,
        }

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._unpack_dict: dict[str, dict[str, Any]] = {
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

    def __str__(self) -> str:
        """String representation of the Header with key information."""
        lines = [f"channel_id: {self.channel_id}   channel_type: {self.channel_type}"]
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

    def __repr__(self) -> str:
        """Detailed string representation of the Header."""
        return self.__str__()

    def _has_header(self) -> bool:
        """
        Check if header data has been loaded.

        Returns
        -------
        bool
            True if header data is loaded, False otherwise.
        """
        return self._header is not None

    def _unpack_value(self, key: str) -> tuple[Any, ...] | None:
        """
        Unpack a value from the binary header using the unpack dictionary.

        Parameters
        ----------
        key : str
            The key in the unpack dictionary corresponding to the field to extract.

        Returns
        -------
        tuple of Any or None
            Unpacked values from the binary header, or None if no header loaded.
        """
        if self._has_header() and self._header is not None:
            return unpack_from(
                self._unpack_dict[key]["dtype"],
                self._header,
                self._unpack_dict[key]["index"],
            )
        return None

    @property
    def file_type(self) -> int | None:
        """
        File type indicator from binary header.

        Returns
        -------
        int or None
            File type identifier, or None if no header is loaded.
        """
        if self._has_header():
            unpacked = self._unpack_value("file_type")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def file_version(self) -> int | None:
        """
        File version from binary header.

        Returns
        -------
        int or None
            File version identifier, or None if no header is loaded.
        """
        if self._has_header():
            unpacked = self._unpack_value("file_version")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def header_length(self) -> int:
        """
        Length of the header in bytes.

        Returns
        -------
        int
            Header length in bytes.
        """
        if self._has_header():
            unpacked = self._unpack_value("header_length")
            if unpacked is not None:
                self._header_length = unpacked[0]
        return self._header_length

    @header_length.setter
    def header_length(self, value: int) -> None:
        """Set header length."""
        self._header_length = value

    @property
    def instrument_type(self) -> str | None:
        """
        Instrument type string from binary header.

        Returns
        -------
        str or None
            Cleaned instrument type string, or None if no header is loaded.
        """
        if self._has_header():
            unpacked = self._unpack_value("instrument_type")
            if unpacked is not None:
                return unpacked[0].decode("utf-8").strip(" ").strip("\x00")
        return None

    @property
    def instrument_serial_number(self) -> str | None:
        """
        Instrument serial number from binary header.

        Returns
        -------
        str or None
            Decoded instrument serial number, or None if no header is loaded.
        """
        if self._has_header():
            unpacked = self._unpack_value("instrument_serial_number")
            if unpacked is not None:
                return b"".join(unpacked).strip(b"\x00").decode("utf-8")
        return None

    @property
    def recording_id(self) -> int | None:
        """
        Recording identifier from binary header or cached value.

        Returns
        -------
        int or None
            Recording ID as integer, or None if not available.
        """
        if self._recording_id is None:
            if self._has_header():
                unpacked = self._unpack_value("recording_id")
                if unpacked is not None:
                    return unpacked[0]
        else:
            return self._recording_id
        return None

    @recording_id.setter
    def recording_id(self, value: str | int) -> None:
        """
        Set recording ID.

        Parameters
        ----------
        value : str or int
            Recording ID as hex string or integer.
        """
        if isinstance(value, str):
            value = int(value, 16)
        self._recording_id = value

    @property
    def recording_start_time(self) -> MTime | None:
        """
        Recording start time from GPS timestamp.

        The actual data recording starts 1 second after the set start time.
        This is caused by the data logger starting up and initializing filter.
        This is taken care of in the segment start time.

        See https://github.com/kujaku11/PhoenixGeoPy/tree/main/Docs for more
        information.

        The time recorded is GPS time.

        Returns
        -------
        MTime or None
            GPS start time, or None if recording ID is not available.
        """
        recording_id = self.recording_id
        if recording_id is not None:
            return MTime(time_stamp=recording_id, gps_time=True)
        return None

    @property
    def channel_id(self) -> int | None:
        """
        Channel identifier from binary header or cached value.

        Returns
        -------
        int or None
            Channel ID, or None if not available.
        """
        if self._channel_id is None:
            if self._has_header():
                unpacked = self._unpack_value("channel_id")
                if unpacked is not None:
                    return int(unpacked[0])
        else:
            return self._channel_id
        return None

    @channel_id.setter
    def channel_id(self, value: int | str) -> None:
        """
        Set channel ID.

        Parameters
        ----------
        value : int or str
            Channel identifier.
        """
        self._channel_id = int(value)

    @property
    def file_sequence(self) -> int | None:
        """
        File sequence number from binary header.

        Returns
        -------
        int or None
            File sequence number, or None if no header is loaded.
        """
        if self._has_header():
            unpacked = self._unpack_value("file_sequence")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def frag_period(self) -> int | None:
        """
        Fragment period from binary header.

        Returns
        -------
        int or None
            Fragment period, or None if no header is loaded.
        """
        if self._has_header():
            unpacked = self._unpack_value("frag_period")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def ch_board_model(self) -> str | None:
        """
        Channel board model string from binary header.

        Returns
        -------
        str or None
            Board model string, or None if no header is loaded.
        """
        if self._has_header():
            unpacked = self._unpack_value("ch_board_model")
            if unpacked is not None:
                return unpacked[0].decode("utf-8").strip(" ")
        return None

    @property
    def board_model_main(self) -> str | None:
        """
        Main board model identifier.

        Returns
        -------
        str or None
            Main board model (first 5 characters), or None if not available.
        """
        ch_board_model = self.ch_board_model
        if ch_board_model is not None:
            return ch_board_model[0:5]
        return None

    @property
    def board_model_revision(self) -> str | None:
        """
        Board model revision identifier.

        Returns
        -------
        str or None
            Board revision (character 6), or None if not available.
        """
        ch_board_model = self.ch_board_model
        if ch_board_model is not None:
            return ch_board_model[6:7]  # Fixed slice to get single character
        return None

    @property
    def ch_board_serial(self) -> int:
        """
        Channel board serial number from binary header.

        Returns
        -------
        int
            Board serial number as integer, or 0 if not available or invalid.
        """
        if self._has_header():
            unpacked = self._unpack_value("ch_board_serial")
            if unpacked is not None:
                value = unpacked[0].decode("utf-8").strip("\x00")
                # handle the case of backend < v0.14, which puts '--------' in ch_ser
                if all(chars in string.hexdigits for chars in value):
                    return int(value, 16)
        return 0

    @property
    def ch_firmware(self) -> int | None:
        """
        Channel firmware version from binary header.

        Returns
        -------
        int or None
            Firmware version, or None if no header is loaded.
        """
        if self._has_header():
            unpacked = self._unpack_value("ch_firmware")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def hardware_configuration(self) -> tuple[Any, ...] | None:
        """
        Hardware configuration bytes from binary header.

        Returns
        -------
        tuple of Any or None
            Hardware configuration data, or None if no header is loaded.
        """
        if self._has_header():
            return self._unpack_value("hardware_configuration")
        return None

    @property
    def channel_type(self) -> str | None:
        """
        Channel type determined from hardware configuration.

        Returns
        -------
        str or None
            'E' for electric, 'H' for magnetic, or None if no header.
        """
        if self._has_header():
            hw_config = self.hardware_configuration
            if hw_config is not None:
                if hw_config[1] & 0x08 == 0x08:
                    return "E"
                else:
                    return "H"
        return None

    @property
    def detected_channel_type(self) -> str | None:
        """
        Channel type detected by electronics.

        This normally matches channel_type, but used in electronics design and testing.

        Returns
        -------
        str or None
            'E' for electric, 'H' for magnetic, or None if no header.
        """
        if self._has_header():
            hw_config = self.hardware_configuration
            if hw_config is not None:
                if hw_config[1] & 0x20 == 0x20:
                    return "E"
                else:
                    return "H"
        return None

    @property
    def lp_frequency(self) -> int | None:
        """
        Low-pass filter frequency based on hardware configuration.

        Returns
        -------
        int or None
            Filter frequency in Hz, or None if no header.
        """
        if self._has_header():
            hw_config = self.hardware_configuration
            board_main = self.board_model_main
            if hw_config is not None:
                # LPF on
                if hw_config[0] & 0x80 == 0x80:
                    if hw_config[0] & 0x03 == 0x03:
                        return 10
                    elif hw_config[0] & 0x03 == 0x02:
                        if board_main == "BCM03" or board_main == "BCM06":
                            return 1000
                        else:
                            return 100
                    elif hw_config[0] & 0x03 == 0x01:
                        if board_main == "BCM03" or board_main == "BCM06":
                            return 10000
                        else:
                            return 1000
                # LPF off
                else:
                    if board_main == "BCM03" or board_main == "BCM06":
                        return 17800
                    else:
                        return 10000
        return None

    @property
    def preamp_gain(self) -> float:
        """
        Pre-amplifier gain factor.

        Returns
        -------
        float
            Gain factor, default 1.0.

        Raises
        ------
        Exception
            If channel type is not determined before calculating gain.
        """
        preamp_gain = 1.0
        if self._has_header():
            channel_type = self.channel_type
            if channel_type == "?" or channel_type is None:
                raise Exception(
                    "Channel type must be set before attemting to calculate preamp gain"
                )
            hw_config = self.hardware_configuration
            if hw_config is not None:
                preamp_on = bool(hw_config[0] & 0x10)
                if channel_type == "E":
                    if preamp_on:
                        board_main = self.board_model_main
                        board_revision = self.board_model_revision
                        if board_main == "BCM01" or board_main == "BCM03":
                            preamp_gain = 4.0
                            if board_revision == "L":
                                # Account for BCM01-L experimental prototype
                                preamp_gain = 8.0
                        else:
                            preamp_gain = 8.0
                            # Account for experimental prototype BCM05-A
                            ch_board_model = self.ch_board_model
                            if (
                                ch_board_model is not None
                                and ch_board_model[0:7] == "BCM05-A"
                            ):
                                preamp_gain = 4.0
        return preamp_gain

    @property
    def channel_main_gain(self) -> float:
        """
        Main gain of the board.

        Returns
        -------
        float
            Main gain factor.
        """
        main_gain = 1.0
        if self._has_header():
            # BCM05-B and BCM06 introduced different selectable gains
            new_gains = True  # we assume any newer board will have the new gain banks
            board_main = self.board_model_main
            ch_board_model = self.ch_board_model
            if board_main == "BCM01" or board_main == "BCM03":
                # Original style 24 KSps boards and original 96 KSps boards
                new_gains = False
            if ch_board_model is not None and ch_board_model[0:7] == "BCM05-A":
                # Account for experimental prototype BCM05-A, which also had original gain banks
                new_gains = False

            hw_config = self.hardware_configuration
            if hw_config is not None:
                if hw_config[0] & 0x0C == 0x00:
                    main_gain = 1.0
                elif hw_config[0] & 0x0C == 0x04:
                    main_gain = 4.0
                elif hw_config[0] & 0x0C == 0x08:
                    main_gain = 6.0
                    if not new_gains:
                        main_gain = 16.0
                elif hw_config[0] & 0x0C == 0x0C:
                    main_gain = 8.0
                    if not new_gains:
                        main_gain = 32.0
        return main_gain

    @property
    def intrinsic_circuitry_gain(self) -> float:
        """
        Intrinsic circuitry gain based on sensor range configuration.

        This function adjusts the intrinsic circuitry gain based on the
        sensor range configuration in the configuration fingerprint.

        For the Electric channel, calibration path, or H-legacy
        sensors all go through a 1/4 gain stage, and then they get a virtual x2 gain from
        Single-ended-diff before the A/D. In the case of newer sensors (differential)
        instead of a 1/4 gain stage, there is only a 1/2 gain stage.

        Therefore, in the E, cal and legacy sensor case the circuitry gain is 1/2, while for
        newer sensors it is 1.

        Returns
        -------
        float
            Intrinsic gain factor.

        Raises
        ------
        Exception
            If channel type is not determined before calculating gain.

        Notes
        -----
        Circuitry Gain not directly configurable by the user.
        """
        intrinsic_circuitry_gain = 0.5
        if self._has_header():
            channel_type = self.channel_type
            if channel_type == "?" or channel_type is None:
                raise Exception(
                    "Channel type must be set before attemting to calculate preamp gain"
                )
            intrinsic_circuitry_gain = 0.5
            if channel_type == "H":
                hw_config = self.hardware_configuration
                if hw_config is not None and hw_config[1] & 0x01 == 0x01:
                    intrinsic_circuitry_gain = 1.0
        return intrinsic_circuitry_gain

    @property
    def attenuator_gain(self) -> float:
        """
        Attenuator gain factor.

        Returns
        -------
        float
            Attenuator gain factor, default 1.0.

        Raises
        ------
        Exception
            If channel type is not determined before calculating gain.
        """
        attenuator_gain = 1.0
        if self._has_header():
            channel_type = self.channel_type
            if channel_type == "?" or channel_type is None:
                raise Exception(
                    "Channel type must be set before attemting to calculate preamp gain"
                )
            hw_config = self.hardware_configuration
            if hw_config is not None:
                attenuator_on = bool(hw_config[4] & 0x01)
                if attenuator_on and channel_type == "E":
                    new_attenuator = (
                        True  # By default assume dealing with newer board types
                    )
                    board_main = self.board_model_main
                    ch_board_model = self.ch_board_model
                    if board_main == "BCM01" or board_main == "BCM03":
                        # Original style 24 KSps boards and original 96 KSps boards
                        new_attenuator = False
                    if ch_board_model is not None and ch_board_model[0:7] == "BCM05-A":
                        # Account for experimental prototype BCM05-A, which also had original gain banks
                        new_attenuator = False
                    if new_attenuator:
                        attenuator_gain = 523.0 / 5223.0
                    else:
                        attenuator_gain = 0.1
        return attenuator_gain

    @property
    def total_selectable_gain(self) -> float:
        """
        Total gain that is selectable by the user.

        Combines attenuator, preamp, and main channel gains.

        Returns
        -------
        float
            Total selectable gain factor.
        """
        if self._has_header():
            return self.channel_main_gain * self.preamp_gain * self.attenuator_gain
        return 1.0

    @property
    def total_circuitry_gain(self) -> float:
        """
        Total board gain including both intrinsic and user-selectable gains.

        Returns
        -------
        float
            Total circuitry gain factor.
        """
        if self._has_header():
            return self.total_selectable_gain * self.intrinsic_circuitry_gain
        return 0.5

    @property
    def sample_rate_base(self) -> int | None:
        """
        Base sample rate from binary header.

        Returns
        -------
        int or None
            Base sample rate, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("sample_rate_base")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def sample_rate_exp(self) -> int | None:
        """
        Sample rate exponent from binary header.

        Returns
        -------
        int or None
            Sample rate exponent, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("sample_rate_exp")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def sample_rate(self) -> float | None:
        """
        Calculated sample rate.

        Returns
        -------
        float or None
            Sample rate in Hz, or None if no header.
        """
        if self._has_header():
            rate_base = self.sample_rate_base
            rate_exp = self.sample_rate_exp
            if rate_base is not None and rate_exp is not None:
                if rate_exp != 0:
                    return rate_base * pow(10, rate_exp)
                return float(rate_base)
        return None

    @property
    def bytes_per_sample(self) -> int | None:
        """
        Number of bytes per sample.

        Returns
        -------
        int or None
            Bytes per sample, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("bytes_per_sample")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def frame_size(self) -> int | None:
        """
        Frame size from binary header.

        Returns
        -------
        int or None
            Frame size value, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("frame_size")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def data_footer(self) -> int | None:
        """
        Data footer extracted from frame size.

        Returns
        -------
        int or None
            Data footer value, or None if no frame size.
        """
        frame_size = self.frame_size
        if frame_size is not None:
            return frame_size >> 24
        return None

    @property
    def frame_size_bytes(self) -> int | None:
        """
        Frame size in bytes.

        Returns
        -------
        int or None
            Frame size in bytes, or None if no frame size.
        """
        frame_size = self.frame_size
        if frame_size is not None:
            return frame_size & 0x0FFFFFF
        return None

    @property
    def decimation_node_id(self) -> int | None:
        """
        Decimation node identifier.

        Returns
        -------
        int or None
            Decimation node ID, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("decimation_node_id")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def frame_rollover_count(self) -> int | None:
        """
        Frame rollover count.

        Returns
        -------
        int or None
            Rollover count, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("frame_rollover_count")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def gps_long(self) -> float | None:
        """
        GPS longitude.

        Returns
        -------
        float or None
            Longitude in degrees, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("gps_long")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def gps_lat(self) -> float | None:
        """
        GPS latitude.

        Returns
        -------
        float or None
            Latitude in degrees, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("gps_lat")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def gps_elevation(self) -> float | None:
        """
        GPS elevation.

        Returns
        -------
        float or None
            Elevation in meters, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("gps_elevation")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def gps_horizontal_accuracy(self) -> float | None:
        """
        GPS horizontal accuracy.

        Returns
        -------
        float or None
            Horizontal accuracy in meters (converted from millimeters), or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("gps_horizontal_accuracy")
            if unpacked is not None:
                return unpacked[0] / 1000
        return None

    @property
    def gps_vertical_accuracy(self) -> float | None:
        """
        GPS vertical accuracy.

        Returns
        -------
        float or None
            Vertical accuracy in meters (converted from millimeters), or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("gps_vertical_accuracy")
            if unpacked is not None:
                return unpacked[0] / 1000
        return None

    @property
    def timing_status(self) -> tuple[Any, ...] | None:
        """
        Timing status information.

        Returns
        -------
        tuple of Any or None
            Timing status data, or None if no header.
        """
        if self._has_header():
            return self._unpack_value("timing_status")
        return None

    @property
    def timing_flags(self) -> Any | None:
        """
        Timing flags from timing status.

        Returns
        -------
        Any or None
            Timing flags, or None if no timing status.
        """
        timing_status = self.timing_status
        if timing_status is not None:
            return timing_status[0]
        return None

    @property
    def timing_sat_count(self) -> Any | None:
        """
        Satellite count from timing status.

        Returns
        -------
        Any or None
            Satellite count, or None if no timing status.
        """
        timing_status = self.timing_status
        if timing_status is not None:
            return timing_status[1]
        return None

    @property
    def timing_stability(self) -> Any | None:
        """
        Timing stability from timing status.

        Returns
        -------
        Any or None
            Timing stability value, or None if no timing status.
        """
        timing_status = self.timing_status
        if timing_status is not None:
            return timing_status[2]
        return None

    @property
    def future1(self) -> Any | None:
        """
        Future field 1 (reserved).

        Returns
        -------
        Any or None
            Future field value, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("future1")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def future2(self) -> Any | None:
        """
        Future field 2 (reserved).

        Returns
        -------
        Any or None
            Future field value, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("future2")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def saturated_frames(self) -> int | None:
        """
        Number of saturated frames.

        Returns
        -------
        int or None
            Saturated frame count, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("saturated_frames")
            if unpacked is not None:
                value = unpacked[0]
                if value & 0x80 == 0x80:
                    value &= 0x7F
                    value <<= 4
                return value
        return None

    @property
    def missing_frames(self) -> int | None:
        """
        Number of missing frames.

        Returns
        -------
        int or None
            Missing frame count, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("missing_frames")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def battery_voltage_v(self) -> float | None:
        """
        Battery voltage in volts.

        Returns
        -------
        float or None
            Battery voltage in volts (converted from millivolts), or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("battery_voltage_mv")
            if unpacked is not None:
                return unpacked[0] / 1000
        return None

    @property
    def min_signal(self) -> Any | None:
        """
        Minimum signal value.

        Returns
        -------
        Any or None
            Minimum signal value, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("min_signal")
            if unpacked is not None:
                return unpacked[0]
        return None

    @property
    def max_signal(self) -> Any | None:
        """
        Maximum signal value.

        Returns
        -------
        Any or None
            Maximum signal value, or None if no header.
        """
        if self._has_header():
            unpacked = self._unpack_value("max_signal")
            if unpacked is not None:
                return unpacked[0]
        return None

    def unpack_header(self, stream: BinaryIO) -> None:
        """
        Read and unpack binary header from stream.

        Parameters
        ----------
        stream : BinaryIO
            Binary stream to read header from.
        """
        if self.header_length > 0:
            # be sure to read from the beginning of the file
            stream.seek(0)
            self._header = stream.read(self.header_length)

    def get_channel_metadata(self) -> Magnetic | Electric:
        """
        Translate metadata to channel metadata.

        Returns
        -------
        Magnetic or Electric
            Channel metadata object populated with header data.

        Raises
        ------
        KeyError
            If channel ID is not found in channel map.
        ValueError
            If required fields are None or invalid.
        """
        channel_type = self.channel_type
        if channel_type is None:
            raise ValueError("Channel type not available")

        if channel_type.lower() in ["h"]:
            ch = Magnetic()  # type: ignore[call-arg]
            gps_lat = self.gps_lat
            gps_long = self.gps_long
            gps_elevation = self.gps_elevation
            if gps_lat is not None:
                ch.location.latitude = gps_lat
            if gps_long is not None:
                ch.location.longitude = gps_long
            if gps_elevation is not None:
                ch.location.elevation = gps_elevation
        elif channel_type.lower() in ["e"]:
            ch = Electric()  # type: ignore[call-arg]
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")

        channel_id = self.channel_id
        if channel_id is not None:
            try:
                ch.component = self.channel_map[channel_id]
            except KeyError:
                self.logger.error(f"Could not find {channel_id} in channel_map")
                raise
            ch.channel_number = channel_id

        recording_start = self.recording_start_time
        if recording_start is not None:
            ch.time_period.start = recording_start

        sample_rate = self.sample_rate
        if sample_rate is not None:
            ch.sample_rate = sample_rate

        if hasattr(ch, "component") and ch.component:
            ch.measurement_azimuth = self.channel_azimuths[ch.component]

        return ch

    def get_run_metadata(self) -> Run:
        """
        Translate to run metadata.

        Returns
        -------
        Run
            Run metadata object populated with header data.

        Raises
        ------
        ValueError
            If required fields are None.
        """
        r = Run()  # type: ignore[call-arg]

        instrument_type = self.instrument_type
        if instrument_type is not None:
            r.data_logger.type = instrument_type

        instrument_serial = self.instrument_serial_number
        if instrument_serial is not None:
            r.data_logger.id = instrument_serial

        r.data_logger.manufacturer = "Phoenix Geophysics"

        timing_stability = self.timing_stability
        if timing_stability is not None:
            r.data_logger.timing_system.uncertainty = timing_stability

        sample_rate = self.sample_rate
        if sample_rate is not None:
            r.sample_rate = sample_rate
            r.id = f"sr{int(sample_rate)}_0001"

        battery_voltage = self.battery_voltage_v
        if battery_voltage is not None:
            r.data_logger.power_source.voltage.start = battery_voltage

        channel_metadata = self.get_channel_metadata()
        r.channels.append(channel_metadata)  # type: ignore[attr-defined]
        r.update_time_period()

        return r

    def get_station_metadata(self) -> Station:
        """
        Translate to station metadata.

        Returns
        -------
        Station
            Station metadata object populated with header data.
        """
        s = Station()  # type: ignore[call-arg]

        gps_lat = self.gps_lat
        if gps_lat is not None:
            s.location.latitude = gps_lat

        gps_long = self.gps_long
        if gps_long is not None:
            s.location.longitude = gps_long

        gps_elevation = self.gps_elevation
        if gps_elevation is not None:
            s.location.elevation = gps_elevation

        run_metadata = self.get_run_metadata()
        s.runs.append(run_metadata)  # type: ignore[attr-defined]
        s.update_time_period()

        return s
