# -*- coding: utf-8 -*-
"""
====================
Zen Header
====================

    * Tools for reading and writing files for Zen and processing software
    * Tools for copying data from SD cards
    * Tools for copying schedules to SD cards

Created on Tue Jun 11 10:53:23 2013
Updated August 2020 (JP)

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT

"""

# ==============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import numpy as np
from loguru import logger


# ==============================================================================
class Z3DHeader:
    """
    Read header information from a Z3D file and make each metadata entry an attribute.

    Parameters
    ----------
    fn : str or pathlib.Path, optional
        Full path to Z3D file.
    fid : BinaryIO, optional
        File object (e.g., open(Z3Dfile, 'rb')).
    **kwargs : dict
        Additional keyword arguments to set as attributes.

    Attributes
    ----------
    _header_len : int
        Length of header in bits (512).
    ad_gain : float or None
        Gain of channel.
    ad_rate : float or None
        Sampling rate in Hz.
    alt : float or None
        Altitude of the station (not reliable).
    attenchannelsmask : str or None
        Attenuation channels mask.
    box_number : float or None
        ZEN box number.
    box_serial : str or None
        ZEN box serial number.
    channel : float or None
        Channel number of the file.
    channelserial : str or None
        Serial number of the channel board.
    ch_factor : float
        Channel factor (default 9.536743164062e-10).
    channelgain : float
        Channel gain (default 1.0).
    duty : float or None
        Duty cycle of the transmitter.
    fid : BinaryIO or None
        File object.
    fn : str or pathlib.Path or None
        Full path to Z3D file.
    fpga_buildnum : float or None
        Build number of one of the boards.
    gpsweek : int
        GPS week (default 1740).
    header_str : bytes or None
        Full header string.
    lat : float or None
        Latitude of station in degrees.
    logterminal : str or None
        Log terminal setting.
    long : float or None
        Longitude of the station in degrees.
    main_hex_buildnum : float or None
        Build number of the ZEN box in hexadecimal.
    numsats : float or None
        Number of GPS satellites.
    old_version : bool
        Whether this is an old version Z3D file (default False).
    period : float or None
        Period of the transmitter.
    tx_duty : float or None
        Transmitter duty cycle.
    tx_freq : float or None
        Transmitter frequency.
    version : float or None
        Version of the firmware.

    Examples
    --------
    >>> from mth5.io.zen import Z3DHeader
    >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
    >>> header_obj = Z3DHeader(fn=Z3Dfn)
    >>> header_obj.read_header()
    """

    def __init__(
        self,
        fn: Optional[Union[str, Path]] = None,
        fid: Optional[BinaryIO] = None,
        **kwargs: Any,
    ) -> None:
        self.logger = logger

        self.fn: Optional[Union[str, Path]] = fn
        self.fid: Optional[BinaryIO] = fid

        self.header_str: Optional[bytes] = None
        self._header_len: int = 512

        self.ad_gain: Optional[float] = None
        self.ad_rate: Optional[float] = None
        self.alt: Optional[float] = None
        self.attenchannelsmask: Optional[str] = None
        self.box_number: Optional[float] = None
        self.box_serial: Optional[str] = None
        self.channel: Optional[float] = None
        self.channelserial: Optional[str] = None
        self.duty: Optional[float] = None
        self.fpga_buildnum: Optional[float] = None
        self.gpsweek: int = 1740
        self.lat: Optional[float] = None
        self.logterminal: Optional[str] = None
        self.long: Optional[float] = None
        self.main_hex_buildnum: Optional[float] = None
        self.numsats: Optional[float] = None
        self.period: Optional[float] = None
        self.tx_duty: Optional[float] = None
        self.tx_freq: Optional[float] = None
        self.version: Optional[float] = None
        self.old_version: bool = False
        self.ch_factor: float = 9.536743164062e-10
        self.channelgain: float = 1.0

        for key in kwargs:
            setattr(self, key, kwargs[key])

    @property
    def data_logger(self) -> str:
        """
        Data logger name as ZEN{box_number}.

        Returns
        -------
        str
            Data logger name formatted as 'ZEN' followed by zero-padded box number.

        Raises
        ------
        TypeError
            If box_number is None or cannot be converted to int.
        """
        return f"ZEN{int(self.box_number):03}"

    def read_header(
        self, fn: Optional[Union[str, Path]] = None, fid: Optional[BinaryIO] = None
    ) -> None:
        """
        Read the header information into appropriate attributes.

        Parses the header information from a Z3D file and populates the object's
        attributes with the extracted values. Supports both modern and legacy
        Z3D file formats.

        Parameters
        ----------
        fn : str or pathlib.Path, optional
            Full path to Z3D file. If None, uses the instance's fn attribute.
        fid : BinaryIO, optional
            File object (e.g., open(Z3Dfile, 'rb')). If None, uses the instance's
            fid attribute or opens the file specified by fn.

        Raises
        ------
        UnicodeDecodeError
            If header bytes cannot be decoded as text.

        Notes
        -----
        This method reads the first 512 bytes of the Z3D file as the header.
        It supports two formats:

        1. Modern format: key=value pairs separated by newlines
        2. Legacy format: comma-separated key:value pairs

        The method automatically detects legacy format and sets old_version=True.

        Coordinate values (lat/long) are automatically converted from radians
        to degrees, with validation to ensure they fall within valid ranges.

        Examples
        --------
        >>> header_obj = Z3DHeader()
        >>> header_obj.read_header("/path/to/file.Z3D")

        >>> with open("/path/to/file.Z3D", "rb") as fid:
        ...     header_obj.read_header(fid=fid)
        """
        if fn is not None:
            self.fn = fn
        if fid is not None:
            self.fid = fid
        if self.fn is None and self.fid is None:
            self.logger.warning("No Z3D file to read.")
        elif self.fn is None:
            if self.fid is not None:
                self.fid.seek(0)
                self.header_str = self.fid.read(self._header_len)
        elif self.fn is not None:
            if self.fid is None:
                self.fid = open(self.fn, "rb")
                self.header_str = self.fid.read(self._header_len)
            else:
                self.fid.seek(0)
                self.header_str = self.fid.read(self._header_len)
        header_list = self.header_str.split(b"\n")
        for h_str in header_list:
            h_str = h_str.decode()
            if h_str.find("=") > 0:
                h_list = h_str.split("=")
                h_key = h_list[0].strip().lower()
                h_key = h_key.replace(" ", "_").replace("/", "").replace(".", "_")
                h_value = self.convert_value(h_key, h_list[1].strip())
                setattr(self, h_key, h_value)
            elif len(h_str) == 0:
                continue
            # need to adjust for older versions of z3d files
            elif h_str.count(",") > 1:
                self.old_version = True
                if h_str.find("Schedule") >= 0:
                    h_str = h_str.replace(",", "T", 1)
                for hh in h_str.split(","):
                    if hh.find(";") > 0:
                        m_key, m_value = hh.split(";")[1].split(":")
                    elif len(hh.split(":", 1)) == 2:
                        m_key, m_value = hh.split(":", 1)
                    else:
                        self.logger.warning("found %s", hh)
                    m_key = (
                        m_key.strip()
                        .lower()
                        .replace(" ", "_")
                        .replace("/", "")
                        .replace(".", "_")
                    )
                    m_value = self.convert_value(m_key, m_value.strip())
                    setattr(self, m_key, m_value)

    def convert_value(self, key_string: str, value_string: str) -> Union[float, str]:
        """
        Convert the value to the appropriate units given the key.

        Converts string values to appropriate types based on the key name.
        Special handling is provided for latitude and longitude values, which
        are converted from radians to degrees with validation.

        Parameters
        ----------
        key_string : str
            The metadata key name, used to determine conversion type.
        value_string : str
            The string value to convert.

        Returns
        -------
        float or str
            Converted value. Returns float for numeric values, str for
            non-numeric values. Latitude and longitude values are converted
            from radians to degrees.

        Notes
        -----
        - Attempts to convert all values to float first
        - If conversion fails, returns original string
        - For keys containing 'lat', 'lon', or 'long':
          - Converts from radians to degrees using np.rad2deg
          - Validates latitude range (±90°), sets to 0.0 if invalid
          - Validates longitude range (±180°), sets to 0.0 if invalid

        Examples
        --------
        >>> header = Z3DHeader()
        >>> header.convert_value("version", "4147")
        4147.0
        >>> header.convert_value("lat", "0.706816081")  # radians
        40.49757833327694  # degrees
        >>> header.convert_value("channelserial", "0xD474777C")
        '0xD474777C'
        """

        try:
            return_value = float(value_string)
        except ValueError:
            return_value = value_string
        if key_string.lower() in ["lat", "lon", "long"]:
            return_value = np.rad2deg(float(value_string))
            if "lat" in key_string.lower():
                if abs(return_value) > 90:
                    return_value = 0.0
            elif "lon" in key_string.lower():
                if abs(return_value) > 180:
                    return_value = 0.0
        return return_value
