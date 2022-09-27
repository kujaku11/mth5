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
import logging

import numpy as np

# ==============================================================================
class Z3DHeader:
    """
    Read in the header information of a Z3D file and make each metadata
    entry an attirbute.

    :param fn: full path to Z3D file
    :type fn: string or :class:`pathlib.Path`
    :param fid:  file object ex. open(Z3Dfile, 'rb')
    :type fid: file

    ======================== ==================================================
    Attributes               Definition
    ======================== ==================================================
    _header_len              lenght of header in bits (512)
    ad_gain                  gain of channel
    ad_rate                  sampling rate in Hz
    alt                      altitude of the station (not reliable)
    attenchannelsmask        not sure
    box_number               ZEN box number
    box_serial               ZEN box serial number
    channel                  channel number of the file
    channelserial            serial number of the channel board
    duty                     duty cycle of the transmitter
    fpga_buildnum            build number of one of the boards
    gpsweek                  GPS week
    header_str               full header string
    lat                      latitude of station
    logterminal              not sure
    long                     longitude of the station
    main_hex_buildnum        build number of the ZEN box in hexidecimal
    numsats                  number of gps satelites
    period                   period of the transmitter
    tx_duty                  transmitter duty cycle
    tx_freq                  transmitter frequency
    version                  version of the firmware
    ======================== ==================================================

    :Example:

        >>> import mtpy.usgs.zen as zen
        >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
        >>> header_obj = zen.Z3DHeader()
        >>> header_obj.read_header()
    """

    def __init__(self, fn=None, fid=None, **kwargs):
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

        self.fn = fn
        self.fid = fid

        self.header_str = None
        self._header_len = 512

        self.ad_gain = None
        self.ad_rate = None
        self.alt = None
        self.attenchannelsmask = None
        self.box_number = None
        self.box_serial = None
        self.channel = None
        self.channelserial = None
        self.duty = None
        self.fpga_buildnum = None
        self.gpsweek = 1740
        self.lat = None
        self.logterminal = None
        self.long = None
        self.main_hex_buildnum = None
        self.numsats = None
        self.period = None
        self.tx_duty = None
        self.tx_freq = None
        self.version = None
        self.old_version = False
        self.ch_factor = 9.536743164062e-10
        self.channelgain = 1.0

        for key in kwargs:
            setattr(self, key, kwargs[key])

    @property
    def data_logger(self):
        """Data logger name as ZEN{box_number}"""
        return "ZEN{0:03}".format(int(self.box_number))

    def read_header(self, fn=None, fid=None):
        """
        Read the header information into appropriate attributes

        :param fn: full path to Z3D file
        :type fn: string or :class:`pathlib.Path`
        :param fid:  file object ex. open(Z3Dfile, 'rb')
        :type fid: file

        :Example:

        >>> import mtpy.usgs.zen as zen
        >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
        >>> header_obj = zen.Z3DHeader()
        >>> header_obj.read_header()

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
                h_key = (
                    h_key.replace(" ", "_").replace("/", "").replace(".", "_")
                )
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

    def convert_value(self, key_string, value_string):
        """
        convert the value to the appropriate units given the key
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
