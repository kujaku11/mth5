# -*- coding: utf-8 -*-
"""
====================
Zen
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
# from __future__ import unicode_literals

import datetime
import os
import struct
import logging

import numpy as np

from mt_metadata.utils.mttime import MTime
from mt_metadata.timeseries.filters import (
    ChannelResponseFilter,
    FrequencyResponseTableFilter,
    CoefficientFilter,
)
from mth5.timeseries import ChannelTS

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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

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


# ==============================================================================
# meta data
# ==============================================================================
class Z3DSchedule:
    """
    Will read in the schedule information of a Z3D file and make each metadata
    entry an attirbute. The attributes are left in capitalization of the Z3D file.

    :param fn: full path to Z3D file
    :type fn: string or :class:`pathlib.Path`
    :param fid:  file object ex. open(Z3Dfile, 'rb')
    :type fid: file

    ======================== ==================================================
    Attributes               Definition
    ======================== ==================================================
    AutoGain                 Auto gain for the channel
    Comment                  Any comments for the schedule
    Date                     Date of when the schedule action was started
                             YYYY-MM-DD
    Duty                     Duty cycle of the transmitter
    FFTStacks                FFT stacks from the transmitter
    Filename                 Name of the file that the ZEN gives it
    Gain                     Gain of the channel
    Log                      Log the data [ Y | N ]
    NewFile                  Create a new file [ Y | N ]
    Period                   Period of the transmitter
    RadioOn                  Turn on the radio [ Y | N ]
    SR                       Sampling Rate in Hz
    SamplesPerAcq            Samples per aquisition for transmitter
    Sleep                    Set the box to sleep [ Y | N ]
    Sync                     Sync with GPS [ Y | N ]
    Time                     Time the schedule action started
                             HH:MM:SS (GPS time)
    _header_len              length of header in bits (512)
    _schedule_metadata_len   length of schedule metadata in bits (512)
    fid                      file object of the file
    fn                       file name to read in
    meta_string              string of the schedule
    ======================== ==================================================

    :Example:

        >>> import mtpy.usgs.zen as zen
        >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
        >>> header_obj = zen.Z3DSchedule()
        >>> header_obj.read_schedule()

    """

    def __init__(self, fn=None, fid=None, **kwargs):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fn = fn
        self.fid = fid
        self.meta_string = None

        self._schedule_metadata_len = 512
        self._header_len = 512

        self.AutoGain = None
        self.Comment = None
        self.Date = None
        self.Duty = None
        self.FFTStacks = None
        self.Filename = None
        self.Gain = None
        self.Log = None
        self.NewFile = None
        self.Period = None
        self.RadioOn = None
        self.SR = None
        self.SamplesPerAcq = None
        self.Sleep = None
        self.Sync = None
        self.Time = None
        self.initial_start = MTime()

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def read_schedule(self, fn=None, fid=None):
        """
        read meta data string
        """
        if fn is not None:
            self.fn = fn

        if fid is not None:
            self.fid = fid

        if self.fn is None and self.fid is None:
            self.logger.warning("No Z3D file to read.")
        elif self.fn is None:
            if self.fid is not None:
                self.fid.seek(self._header_len)
                self.meta_string = self.fid.read(self._header_len)
        elif self.fn is not None:
            if self.fid is None:
                self.fid = open(self.fn, "rb")
                self.fid.seek(self._header_len)
                self.meta_string = self.fid.read(self._header_len)
            else:
                self.fid.seek(self._header_len)
                self.meta_string = self.fid.read(self._header_len)

        meta_list = self.meta_string.split(b"\n")
        for m_str in meta_list:
            m_str = m_str.decode()
            if m_str.find("=") > 0:
                m_list = m_str.split("=")
                m_key = m_list[0].split(".")[1].strip()
                m_key = m_key.replace("/", "")
                m_value = m_list[1].strip()
                setattr(self, m_key, m_value)

        self.initial_start = MTime(time=f"{self.Date}T{self.Time}", gps_time=True)


# ==============================================================================
#  Meta data class
# ==============================================================================
class Z3DMetadata:
    """
    Will read in the metadata information of a Z3D file and make each metadata
    entry an attirbute.The attributes are left in capitalization of the Z3D
    file.

    :param fn: full path to Z3D file
    :type fn: string or :class:`pathlib.Path`
    :param fid:  file object ex. open(Z3Dfile, 'rb')
    :type fid: file

    ======================== ==================================================
    Attributes               Definition
    ======================== ==================================================
    _header_length           length of header in bits (512)
    _metadata_length         length of metadata blocks (512)
    _schedule_metadata_len   length of schedule meta data (512)
    board_cal                board calibration np.ndarray()
    cal_ant                  antenna calibration
    cal_board                board calibration
    cal_ver                  calibration version
    ch_azimuth               channel azimuth
    ch_cmp                   channel component
    ch_length                channel length (or # of coil)
    ch_number                channel number on the ZEN board
    ch_xyz1                  channel xyz location (not sure)
    ch_xyz2                  channel xyz location (not sure)
    coil_cal                 coil calibration np.ndarray (freq, amp, phase)
    fid                      file object
    find_metadata            boolean of finding metadata
    fn                       full path to Z3D file
    gdp_operator             operater of the survey
    gdp_progver              program version
    job_by                   job preformed by
    job_for                  job for
    job_name                 job name
    job_number               job number
    m_tell                   location in the file where the last metadata
                             block was found.
    rx_aspace                electrode spacing
    rx_sspace                not sure
    rx_xazimuth              x azimuth of electrode
    rx_xyz0                  not sure
    rx_yazimuth              y azimuth of electrode
    survey_type              type of survey
    unit_length              length units (m)
    ======================== ==================================================

    :Example:

        >>> import mtpy.usgs.zen as zen
        >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
        >>> header_obj = zen.Z3DMetadata()
        >>> header_obj.read_metadata()

    """

    def __init__(self, fn=None, fid=None, **kwargs):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fn = fn
        self.fid = fid
        self.find_metadata = True
        self.board_cal = None
        self.coil_cal = None
        self._metadata_length = 512
        self._header_length = 512
        self._schedule_metadata_len = 512
        self.m_tell = 0

        self.cal_ant = None
        self.cal_board = None
        self.cal_ver = None
        self.ch_azimuth = None
        self.ch_cmp = None
        self.ch_length = None
        self.ch_number = None
        self.ch_xyz1 = None
        self.ch_xyz2 = None
        self.gdp_operator = None
        self.gdp_progver = None
        self.job_by = None
        self.job_for = None
        self.job_name = None
        self.job_number = None
        self.rx_aspace = None
        self.rx_sspace = None
        self.rx_xazimuth = None
        self.rx_xyz0 = None
        self.rx_yazimuth = None
        self.line_name = None
        self.survey_type = None
        self.unit_length = None
        self.station = None
        self.count = 0
        self.notes = None

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def read_metadata(self, fn=None, fid=None):
        """
        read meta data

        :param string fn: full path to file, optional if already initialized.
        :param file fid: open file object, optional if already initialized.
        """
        if fn is not None:
            self.fn = fn

        if fid is not None:
            self.fid = fid

        if self.fn is None and self.fid is None:
            self.logger.waringn("No Z3D file to read")
        elif self.fn is None:
            if self.fid is not None:
                self.fid.seek(self._header_length + self._schedule_metadata_len)
        elif self.fn is not None:
            if self.fid is None:
                self.fid = open(self.fn, "rb")
                self.fid.seek(self._header_length + self._schedule_metadata_len)
            else:
                self.fid.seek(self._header_length + self._schedule_metadata_len)

        # read in calibration and meta data
        self.find_metadata = True
        self.board_cal = []
        self.coil_cal = []
        self.count = 0
        cal_find = False
        while self.find_metadata == True:
            try:
                test_str = self.fid.read(self._metadata_length).decode().lower()
            except UnicodeDecodeError:
                self.find_metadata = False
                break

            if "metadata" in test_str:
                self.count += 1
                test_str = test_str.strip().split("record")[1].strip()

                # split the metadata records with key=value style
                if test_str.count("|") > 1:
                    for t_str in test_str.split("|"):
                        # get metadata name and value
                        if (
                            t_str.find("=") == -1
                            and t_str.lower().find("line.name") == -1
                        ):
                            # get metadata for older versions of z3d files
                            if len(t_str.split(",")) == 2:
                                t_list = t_str.lower().split(",")
                                t_key = t_list[0].strip().replace(".", "_")
                                if t_key == "ch_varasp":
                                    t_key = "ch_length"
                                t_value = t_list[1].strip()
                                setattr(self, t_key, t_value)
                            if t_str.count(" ") > 1:
                                self.notes = t_str
                        # get metadata for just the line that has line name
                        # because for some reason that is still comma separated
                        elif t_str.lower().find("line.name") >= 0:
                            t_list = t_str.split(",")
                            t_key = t_list[0].strip().replace(".", "_")
                            t_value = t_list[1].strip()
                            setattr(self, t_key.lower(), t_value)
                        # get metadata for newer z3d files
                        else:
                            t_list = t_str.split("=")
                            t_key = t_list[0].strip().replace(".", "_")
                            t_value = t_list[1].strip()
                            setattr(self, t_key.lower(), t_value)

                elif "cal.brd" in test_str:
                    t_list = test_str.split(",")
                    t_key = t_list[0].strip().replace(".", "_")
                    setattr(self, t_key.lower(), t_list[1])
                    for t_str in t_list[2:]:
                        t_str = t_str.replace("\x00", "").replace("|", "")
                        try:
                            self.board_cal.append(
                                [float(tt.strip()) for tt in t_str.strip().split(":")]
                            )
                        except ValueError:
                            self.board_cal.append(
                                [tt.strip() for tt in t_str.strip().split(":")]
                            )
                # some times the coil calibration does not start on its own line
                # so need to parse the line up and I'm not sure what the calibration
                # version is for so I have named it odd
                elif "cal.ant" in test_str:
                    # check to see if the coil calibration exists
                    cal_find = True
                    test_list = test_str.split(",")
                    coil_num = test_list[1].split("|")[1]
                    coil_key, coil_value = coil_num.split("=")
                    setattr(
                        self,
                        coil_key.replace(".", "_").lower(),
                        coil_value.strip(),
                    )
                    for t_str in test_list[2:]:
                        if "\x00" in t_str:
                            break
                        self.coil_cal.append(
                            [float(tt.strip()) for tt in t_str.split(":")]
                        )
                elif cal_find and self.count > 3:
                    t_list = test_str.replace("|", ",").split(",")
                    for t_str in t_list:
                        if "\x00" in t_str:
                            break
                        else:
                            self.coil_cal.append(
                                [float(tt.strip()) for tt in t_str.strip().split(":")]
                            )
                            
            elif "caldata" in test_str:
                t_list = test_str.replace("|", ",").split(",")
                for t_str in t_list:
                    if "\x00" in t_str:
                        continue
                    else:
                        try:
                            self.board_cal.append(
                                [float(tt.strip()) for tt in t_str.strip().split(":")]
                            )
                        except ValueError:
                            self.board_cal.append(
                                [tt.strip() for tt in t_str.strip().split(":")]
                            )

            else:
                self.find_metadata = False
                # need to go back to where the meta data was found so
                # we don't skip a gps time stamp
                self.m_tell = self.fid.tell() - self._metadata_length

        # make coil calibration and board calibration structured arrays
        if len(self.coil_cal) > 0:
            self.coil_cal = np.core.records.fromrecords(
                self.coil_cal, names="frequency, amplitude, phase"
            )
        if len(self.board_cal) > 0:
            try:
                self.board_cal = np.core.records.fromrecords(
                    self.board_cal, names="frequency, rate, amplitude, phase"
                )
            except ValueError:
                self.board_cal = None

        try:
            self.station = "{0}{1}".format(self.line_name, self.rx_xyz0.split(":")[0])
        except AttributeError:
            if hasattr(self, "rx_stn"):
                self.station = f"{self.rx_stn}"
            elif hasattr(self, "ch_stn"):
                self.station = f"{self.ch_stn}"
            else:
                self.station = None
                self.logger.warning("Need to input station name")


# ==============================================================================
#
# ==============================================================================
class Z3D:
    """
    Deals with the raw Z3D files output by zen.
    Arguments
    -----------
        **fn** : string
                 full path to .Z3D file to be read in
    ======================== ================================ =================
    Attributes               Description                      Default Value
    ======================== ================================ =================
    _block_len               length of data block to read in  65536
                             as chunks faster reading
    _counts_to_mv_conversion conversion factor to convert     9.53674316406e-10
                             counts to mv
    _gps_bytes               number of bytes for a gps stamp  16
    _gps_dtype               data type for a gps stamp        see below
    _gps_epoch               starting date of GPS time
                             format is a tuple                (1980, 1, 6, 0,
                                                               0, 0, -1, -1, 0)
    _gps_f0                  first gps flag in raw binary
    _gps_f1                  second gps flag in raw binary
    _gps_flag_0              first gps flag as an int32       2147483647
    _gps_flag_1              second gps flag as an int32      -2147483648
    _gps_stamp_length        bit length of gps stamp          64
    _leap_seconds            leap seconds, difference         16
                             between UTC time and GPS
                             time.  GPS time is ahead
                             by this much
    _week_len                week length in seconds           604800
    df                       sampling rate of the data        256
    fn                       Z3D file name                    None
    gps_flag                 full gps flag                    _gps_f0+_gps_f1
    gps_stamps               np.ndarray of gps stamps         None
    header                   Z3DHeader object                 Z3DHeader
    metadata                 Z3DMetadata                      Z3DMetadata
    schedule                 Z3DSchedule                      Z3DSchedule
    time_series              np.ndarra(len_data)              None
    units                    units in which the data is in    counts
    zen_schedule             time when zen was set to         None
                             run
    ======================== ================================ =================

    * gps_dtype is formated as np.dtype([('flag0', np.int32),
                                        ('flag1', np.int32),
                                        ('time', np.int32),
                                        ('lat', np.float64),
                                        ('lon', np.float64),
                                        ('num_sat', np.int32),
                                        ('gps_sens', np.int32),
                                        ('temperature', np.float32),
                                        ('voltage', np.float32),
                                        ('num_fpga', np.int32),
                                        ('num_adc', np.int32),
                                        ('pps_count', np.int32),
                                        ('dac_tune', np.int32),
                                        ('block_len', np.int32)])


    :Example:

        >>> import mtpy.usgs.zen as zen
        >>> zt = zen.Zen3D(r"/home/mt/mt00/mt00_20150522_080000_256_EX.Z3D")
        >>> zt.read_z3d()
        >>> ------- Reading /home/mt/mt00/mt00_20150522_080000_256_EX.Z3D -----
            --> Reading data took: 0.322 seconds
            Scheduled time was 2015-05-22,08:00:16 (GPS time)
            1st good stamp was 2015-05-22,08:00:18 (GPS time)
            difference of 2.00 seconds
            found 6418 GPS time stamps
            found 1642752 data points
        >>> zt.plot_time_series()
    """

    def __init__(self, fn=None, **kwargs):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fn = fn

        self.header = Z3DHeader(fn)
        self.schedule = Z3DSchedule(fn)
        self.metadata = Z3DMetadata(fn)

        self._gps_stamp_length = kwargs.pop("stamp_len", 64)
        self._gps_bytes = self._gps_stamp_length / 4

        self.gps_stamps = None

        self._gps_flag_0 = np.int32(2147483647)
        self._gps_flag_1 = np.int32(-2147483648)
        self._gps_f0 = self._gps_flag_0.tobytes()
        self._gps_f1 = self._gps_flag_1.tobytes()
        self.gps_flag = self._gps_f0 + self._gps_f1

        self._gps_dtype = np.dtype(
            [
                ("flag0", np.int32),
                ("flag1", np.int32),
                ("time", np.int32),
                ("lat", np.float64),
                ("lon", np.float64),
                ("gps_sens", np.int32),
                ("num_sat", np.int32),
                ("temperature", np.float32),
                ("voltage", np.float32),
                ("num_fpga", np.int32),
                ("num_adc", np.int32),
                ("pps_count", np.int32),
                ("dac_tune", np.int32),
                ("block_len", np.int32),
            ]
        )

        self._week_len = 604800
        # '1980, 1, 6, 0, 0, 0, -1, -1, 0
        self._gps_epoch = MTime("1980-01-06T00:00:00")
        self._leap_seconds = 18
        self._block_len = 2 ** 16
        # the number in the cac files is for volts, we want mV
        self._counts_to_mv_conversion = 9.5367431640625e-10 * 1e3
        self.num_sec_to_skip = 2

        self.units = "counts"
        self.sample_rate = None

        self.ch_dict = {"hx": 1, "hy": 2, "hz": 3, "ex": 4, "ey": 5}

    @property
    def station(self):
        """
        station name
        """
        return self.metadata.station

    @station.setter
    def station(self, station):
        """
        station name
        """
        self.metadata.station = station

    @property
    def dipole_length(self):
        """
        dipole length
        """
        length = 0
        if self.metadata.ch_length is not None:
            length = float(self.metadata.ch_length)
        elif hasattr(self.metadata, "ch_offset_xyz1"):
            # only ex and ey have xyz2
            if hasattr(self.metadata, "ch_offset_xyz2"):
                x1, y1, z1 = [
                    float(offset) for offset in self.metadata.ch_offset_xyz1.split(":")
                ]
                x2, y2, z2 = [
                    float(offset) for offset in self.metadata.ch_offset_xyz2.split(":")
                ]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                length = np.round(length, 2)
            else:
                length = 0
        elif self.metadata.ch_xyz1 is not None:
            x1, y1 = [float(d) for d in self.metadata.ch_xyz1.split(":")]
            x2, y2 = [float(d) for d in self.metadata.ch_xyz2.split(":")]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 100.0
            length = np.round(length, 2)

        return length

    @property
    def azimuth(self):
        """
        azimuth of instrument setup
        """
        if self.metadata.ch_azimuth is not None:
            return float(self.metadata.ch_azimuth)
        elif self.metadata.rx_xazimuth is not None:
            return float(self.metadata.rx_xazimuth)
        else:
            return None

    @property
    def component(self):
        """
        channel
        """
        return self.metadata.ch_cmp.lower()

    @property
    def latitude(self):
        """
        latitude in decimal degrees
        """
        return self.header.lat

    @property
    def longitude(self):
        """
        longitude in decimal degrees
        """
        return self.header.long

    @property
    def elevation(self):
        """
        elevation in meters
        """
        return self.header.alt

    @property
    def sample_rate(self):
        """
        sampling rate
        """
        return self.header.ad_rate

    @sample_rate.setter
    def sample_rate(self, sampling_rate):
        """
        sampling rate
        """
        if sampling_rate is not None:
            self.header.ad_rate = float(sampling_rate)

    @property
    def start(self):
        if self.gps_stamps is not None:
            return self.get_UTC_date_time(
                self.header.gpsweek, self.gps_stamps["time"][0]
            )
        return None

    @property
    def end(self):
        if self.gps_stamps is not None:
            return self.get_UTC_date_time(
                self.header.gpsweek, self.gps_stamps["time"][-1]
            )
        return None

    @property
    def zen_schedule(self):
        """
        zen schedule data and time
        """

        if self.header.old_version is True:
            return MTime(self.header.schedule)

        return self.schedule.initial_start

    @zen_schedule.setter
    def zen_schedule(self, schedule_dt):
        """
        on setting set schedule datetime
        """
        if not isinstance(schedule_dt, MTime):
            schedule_dt = MTime(schedule_dt)
            raise TypeError("New schedule datetime must be type datetime.datetime")
        self.schedule.initial_start = schedule_dt

    @property
    def coil_num(self):
        """
        coil number
        """
        if self.metadata.cal_ant is not None:
            return self.metadata.cal_ant
        elif self.metadata.ch_number is not None:
            return self.metadata.ch_number
        else:
            return None

    @property
    def channel_number(self):
        if self.metadata.ch_number:
            ch_num = int(float(self.metadata.ch_number))
            if ch_num > 6:
                try:
                    ch_num = self.ch_dict[self.component]
                except KeyError:
                    ch_num = 6
            return ch_num
        return 0

    @property
    def channel_metadata(self):
        """Channel metadata"""

        # fill the time series object
        if "e" in self.component:
            ts_type = "electric"
            meta_dict = {"electric": {"dipole_length": self.dipole_length}}
            meta_dict[ts_type]["ac.start"] = (
                self.time_series[0 : int(self.sample_rate)].std()
                * self.header.ch_factor
            )
            meta_dict[ts_type]["ac.end"] = (
                self.time_series[-int(self.sample_rate) :].std() * self.header.ch_factor
            )
            meta_dict[ts_type]["dc.start"] = (
                self.time_series[0 : int(self.sample_rate)].mean()
                * self.header.ch_factor
            )
            meta_dict[ts_type]["dc.end"] = (
                self.time_series[-int(self.sample_rate) :].mean()
                * self.header.ch_factor
            )
        elif "h" in self.component:
            ts_type = "magnetic"
            meta_dict = {
                "magnetic": {
                    "sensor.id": self.coil_num,
                    "sensor.manufacturer": "Geotell",
                    "sensor.model": "ANT-4",
                    "sensor.type": "induction coil",
                }
            }
            meta_dict[ts_type]["h_field_max.start"] = (
                self.time_series[0 : int(self.sample_rate)].max()
                * self.header.ch_factor
            )
            meta_dict[ts_type]["h_field_max.end"] = (
                self.time_series[-int(self.sample_rate) :].max() * self.header.ch_factor
            )
            meta_dict[ts_type]["h_field_min.start"] = (
                self.time_series[0 : int(self.sample_rate)].min()
                * self.header.ch_factor
            )
            meta_dict[ts_type]["h_field_min.end"] = (
                self.time_series[-int(self.sample_rate) :].min() * self.header.ch_factor
            )

        meta_dict[ts_type]["time_period.start"] = self.start.isoformat()
        meta_dict[ts_type]["time_period.end"] = self.end.isoformat()
        meta_dict[ts_type]["component"] = self.component
        meta_dict[ts_type]["sample_rate"] = self.sample_rate
        meta_dict[ts_type]["measurement_azimuth"] = self.azimuth
        meta_dict[ts_type]["units"] = "digital counts"
        meta_dict[ts_type]["channel_number"] = self.channel_number
        meta_dict[ts_type]["filter.name"] = self.channel_response.names
        meta_dict[ts_type]["filter.applied"] = [False] * len(
            self.channel_response.names
        )

        return meta_dict

    @property
    def station_metadata(self):
        """station metadta"""

        meta_dict = {}
        meta_dict["id"] = self.station
        meta_dict["fdsn.id"] = self.station
        meta_dict["location.latitude"] = self.latitude
        meta_dict["location.longitude"] = self.longitude
        meta_dict["location.elevation"] = self.elevation
        meta_dict["time_period.start"] = self.start.isoformat()
        meta_dict["time_period.end"] = self.end.isoformat()
        meta_dict["acquired_by.author"] = self.metadata.gdp_operator

        return {"Station": meta_dict}

    @property
    def run_metadata(self):
        """Run metadata"""
        meta_dict = {}
        meta_dict["data_logger.firmware.version"] = self.header.version
        meta_dict["data_logger.id"] = self.header.data_logger
        meta_dict["data_logger.manufacturer"] = "Zonge International"
        meta_dict["data_logger.model"] = "ZEN"
        meta_dict["time_period.start"] = self.start.isoformat()
        meta_dict["time_period.end"] = self.end.isoformat()
        meta_dict["sample_rate"] = self.sample_rate
        meta_dict["data_type"] = "MTBB"
        meta_dict["time_period.start"] = self.start.isoformat()
        meta_dict["time_period.end"] = self.end.isoformat()
        meta_dict["acquired_by.author"] = self.metadata.gdp_operator

        return {"Run": meta_dict}

    @property
    def counts2mv_filter(self):
        """
        Create a counts2mv coefficient filter
        """

        c2mv = CoefficientFilter()
        c2mv.units_in = "digital counts"
        c2mv.units_out = "millivolts"
        c2mv.name = "zen_counts2mv"
        c2mv.gain = self.header.ch_factor
        c2mv.comments = "digital counts to millivolts"

        return c2mv

    @property
    def coil_response(self):
        """
        Make the coile response into a FAP filter
        """
        fap = None
        if self.metadata.cal_ant is not None:
            fap = FrequencyResponseTableFilter()
            fap.units_in = "millivolts"
            fap.units_out = "nanotesla"
            fap.frequencies = self.metadata.coil_cal.frequency
            fap.amplitudes = self.metadata.coil_cal.amplitude
            fap.phases = np.rad2deg(self.metadata.coil_cal.phase / 1e3)
            fap.name = f"ant4_{self.coil_num}_response"
            fap.comments = "induction coil response read from z3d file"

        return fap

    @property
    def zen_response(self):
        fap = None
        if self.metadata.board_cal not in [None, []]:
            if self.metadata.board_cal[0][0] == "":
                return fap
            sr_dict = {256: 0, 1024: 1, 4096: 4}
            sr_int = sr_dict[int(self.sample_rate)]
            fap_table = self.metadata.board_cal[
                np.where(self.metadata.board_cal.rate == sr_int)
            ]
            fap = FrequencyResponseTableFilter()
            fap.units_in = "millivolts"
            fap.units_out = "millivolts"
            fap.frequencies = fap_table.frequency
            fap.amplitudes = fap_table.amplitude
            fap.phases = np.rad2deg(fap_table.phase / 1e3)
            fap.name = (
                f"{self.header.data_logger.lower()}_{self.sample_rate:.0f}_response"
            )
            fap.comments = "data logger response read from z3d file"

        return fap

    @property
    def channel_response(self):
        filter_list = [self.counts2mv_filter]
        if self.zen_response:
            filter_list.append(self.zen_response)
        if self.coil_response:
            filter_list.append(self.coil_response)
        if self.dipole_filter:
            filter_list.append(self.dipole_filter)

        return ChannelResponseFilter(filters_list=filter_list)

    @property
    def dipole_filter(self):
        dipole = None
        if self.dipole_length != 0:
            dipole = CoefficientFilter()
            dipole.units_in = "millivolts"
            dipole.units_out = "millivolts per kilometer"
            dipole.name = f"{self.station}_{self.component}_dipole"
            dipole.gain = self.dipole_length / 1000.0
            dipole.comments = "convert to electric field"

        return dipole

    @property
    def filter_metadata(self):
        """Filter metadata"""

        meta_dict = {}
        meta_dict["filters"] = {
            "counts_to_volts": self.header.ch_factor,
            "gain": self.header.channelgain,
        }

        return {"Filter": meta_dict}

    def _get_gps_stamp_type(self, old_version=False):
        """
        get the correct stamp type.
        Older versions the stamp length was 36 bits
        New versions have a 64 bit stamp
        """

        if old_version is True:
            self._gps_dtype = np.dtype(
                [
                    ("gps", np.int32),
                    ("time", np.int32),
                    ("lat", np.float64),
                    ("lon", np.float64),
                    ("block_len", np.int32),
                    ("gps_accuracy", np.int32),
                    ("temperature", np.float32),
                ]
            )
            self._gps_stamp_length = 36
            self._gps_bytes = self._gps_stamp_length / 4
            self._gps_flag_0 = -1
            self._block_len = int(self._gps_stamp_length + self.sample_rate * 4)
            self.gps_flag = self._gps_f0

        else:
            return

    # ======================================
    def _read_header(self, fn=None, fid=None):
        """
        read header information from Z3D file
        Arguments
        ---------------
            **fn** : string
                     full path to Z3D file to read
            **fid** : file object
                      if the file is open give the file id object
        Outputs:
        ----------
            * fills the Zen3ZD.header object's attributes

        Example with just a file name
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_header(fn)

        Example with file object
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dfid = open(fn, 'rb')
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_header(fid=Z3Dfid)
        """

        if fn is not None:
            self.fn = fn

        self.header.read_header(fn=self.fn, fid=fid)
        if self.header.old_version:
            if self.header.box_number is None:
                self.header.box_number = "6666"

    # ======================================
    def _read_schedule(self, fn=None, fid=None):
        """
        read schedule information from Z3D file
        Arguments
        ---------------
            **fn** : string
                     full path to Z3D file to read
            **fid** : file object
                      if the file is open give the file id object
        Outputs:
        ----------
            * fills the Zen3ZD.schedule object's attributes

        Example with just a file name
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_schedule(fn)

        Example with file object
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dfid = open(fn, 'rb')
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_schedule(fid=Z3Dfid)
        """

        if fn is not None:
            self.fn = fn

        self.schedule.read_schedule(fn=self.fn, fid=fid)
        if self.header.old_version:
            self.schedule.initial_start = MTime(self.header.schedule, gps_time=True)

    # ======================================
    def _read_metadata(self, fn=None, fid=None):
        """
        read header information from Z3D file
        Arguments
        ---------------
            **fn** : string
                     full path to Z3D file to read
            **fid** : file object
                      if the file is open give the file id object
        Outputs:
        ----------
            * fills the Zen3ZD.metadata object's attributes

        Example with just a file name
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_metadata(fn)

        Example with file object
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dfid = open(fn, 'rb')
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_metadata(fid=Z3Dfid)
        """

        if fn is not None:
            self.fn = fn

        if self.header.old_version:
            self.metadata._schedule_metadata_len = 0
        self.metadata.read_metadata(fn=self.fn, fid=fid)

    # =====================================
    def read_all_info(self):
        """
        Read header, schedule, and metadata
        """
        with open(self.fn, "rb") as file_id:

            self._read_header(fid=file_id)
            self._read_schedule(fid=file_id)
            self._read_metadata(fid=file_id)

    # ======================================
    def read_z3d(self, Z3Dfn=None):
        """
        read in z3d file and populate attributes accordingly

        1. Read in the entire file as chunks as np.int32.

        2. Extract the gps stamps and convert accordingly. Check to make sure
           gps time stamps are 1 second apart and incrementing as well as
           checking the number of data points between stamps is the
           same as the sampling rate.

        3. Converts gps_stamps['time'] to seconds relative to header.gps_week
            Note we skip the first two gps stamps because there is something
            wrong with the data there due to some type of buffering.
            Therefore the first GPS time is when the time series starts, so you
            will notice that gps_stamps[0]['block_len'] = 0, this is because there
            is nothing previous to this time stamp and so the 'block_len' measures
            backwards from the corresponding time index.

        4. Put the data chunks into Pandas data frame that is indexed by time

        :Example:

        >>> from mth5.io import zen
        >>> z_obj = zen.Z3D(r"home/mt_data/zen/mt001.z3d")
        >>> z_obj.read_z3d()

        """

        if Z3Dfn is not None:
            self.fn = Z3Dfn

        self.logger.debug(f"Reading {self.fn}")
        st = datetime.datetime.now()

        # get the file size to get an estimate of how many data points there are
        file_size = os.path.getsize(self.fn)

        # using the with statement works in Python versions 2.7 or higher
        # the added benefit of the with statement is that it will close the
        # file object upon reading completion.
        with open(self.fn, "rb") as file_id:

            self._read_header(fid=file_id)
            self._read_schedule(fid=file_id)
            self._read_metadata(fid=file_id)

            if self.header.old_version is True:
                self._get_gps_stamp_type(True)

            # move the read value to where the end of the metadata is
            file_id.seek(self.metadata.m_tell)

            # initalize a data array filled with zeros, everything goes into
            # this array then we parse later
            data = np.zeros(
                int(
                    (file_size - 512 * (1 + self.metadata.count)) / 4
                    + 8 * self.sample_rate
                ),
                dtype=np.int32,
            )
            # go over a while loop until the data cound exceed the file size
            data_count = 0
            while True:
                # need to make sure the last block read is a multiple of 32 bit
                read_len = min(
                    [
                        self._block_len,
                        int(32 * ((file_size - file_id.tell()) // 32)),
                    ]
                )
                test_str = np.frombuffer(file_id.read(read_len), dtype=np.int32)
                if len(test_str) == 0:
                    break
                data[data_count : data_count + len(test_str)] = test_str
                data_count += test_str.size

        self.raw_data = data.copy()
        # find the gps stamps
        gps_stamp_find = self.get_gps_stamp_index(data, self.header.old_version)

        # skip the first two stamps and trim data
        try:
            data = data[gps_stamp_find[self.num_sec_to_skip] :]
        except IndexError:
            msg = f"Data is bad, cannot open file {self.fn}"
            self.logger.error(msg)
            raise ZenGPSError(msg)

        # find gps stamps of the trimmed data
        gps_stamp_find = self.get_gps_stamp_index(data, self.header.old_version)

        self.gps_stamps = np.zeros(len(gps_stamp_find), dtype=self._gps_dtype)

        for ii, gps_find in enumerate(gps_stamp_find):
            try:
                data[gps_find + 1]
            except IndexError:
                pass
                self.logger.warning(
                    f"Failed gps stamp {ii+1} out of {len(gps_stamp_find)}"
                )
                break

            if (
                self.header.old_version is True
                or data[gps_find + 1] == self._gps_flag_1
            ):
                gps_str = struct.pack(
                    "<" + "i" * int(self._gps_bytes),
                    *data[int(gps_find) : int(gps_find + self._gps_bytes)],
                )
                self.gps_stamps[ii] = np.frombuffer(gps_str, dtype=self._gps_dtype)
                if ii > 0:
                    self.gps_stamps[ii]["block_len"] = (
                        gps_find - gps_stamp_find[ii - 1] - self._gps_bytes
                    )
                elif ii == 0:
                    self.gps_stamps[ii]["block_len"] = 0
                data[int(gps_find) : int(gps_find + self._gps_bytes)] = 0

        # fill the time series
        self.time_series = data[np.nonzero(data)]

        # validate everything
        self.validate_time_blocks()
        self.convert_gps_time()
        self.zen_schedule = self.check_start_time()

        self.logger.debug(f"found {self.gps_stamps.shape[0]} GPS time stamps")
        self.logger.debug(f"found {self.time_series.size} data points")

        # time it
        et = datetime.datetime.now()
        read_time = (et - st).total_seconds()
        self.logger.debug(f"Reading data took: {read_time:.3f} seconds")

    # =================================================
    def get_gps_stamp_index(self, ts_data, old_version=False):
        """
        locate the time stamps in a given time series.

        Looks for gps_flag_0 first, if the file is newer, then makes sure the
        next value is gps_flag_1

        :returns: list of gps stamps indicies
        """

        # find the gps stamps
        gps_stamp_find = np.where(ts_data == self._gps_flag_0)[0]

        if old_version is False:
            gps_stamp_find = [
                gps_find
                for gps_find in gps_stamp_find
                if ts_data[gps_find + 1] == self._gps_flag_1
            ]

        return gps_stamp_find

    # =================================================
    def trim_data(self):
        """
        apparently need to skip the first 2 seconds of data because of
        something to do with the SD buffer

        This method will be deprecated after field testing

        """
        # the block length is the number of data points before the time stamp
        # therefore the first block length is 0.  The indexing in python
        # goes to the last index - 1 so we need to put in 3
        ts_skip = self.gps_stamps["block_len"][0:3].sum()
        self.gps_stamps = self.gps_stamps[2:]
        self.gps_stamps[0]["block_len"] = 0
        self.time_series = self.time_series[ts_skip:]

    # =================================================
    def check_start_time(self):
        """
        check to make sure the scheduled start time is similar to
        the first good gps stamp
        """

        # make sure the time is in gps time
        zen_start_utc = self.get_UTC_date_time(
            self.header.gpsweek, self.gps_stamps["time"][0]
        )

        # estimate the time difference between the two
        time_diff = zen_start_utc - self.schedule.initial_start
        self.logger.debug(f"Scheduled time was {self.schedule.initial_start}")
        self.logger.debug(f"1st good stamp was {zen_start_utc}")
        self.logger.debug(f"difference of {time_diff:.2f} seconds")

        return zen_start_utc

    # ==================================================
    def validate_gps_time(self):
        """
        make sure each time stamp is 1 second apart

        """

        t_diff = np.zeros_like(self.gps_stamps["time"])

        for ii in range(len(t_diff) - 1):
            t_diff[ii] = self.gps_stamps["time"][ii] - self.gps_stamps["time"][ii + 1]

        bad_times = np.where(abs(t_diff) > 0.5)[0]
        if len(bad_times) > 0:
            self.logger.warning("BAD GPS TIMES:")
            for bb in bad_times:
                self.logger.warning(f"bad GPS time at index {bb} > 0.5 s")

    # ===================================================
    def validate_time_blocks(self):
        """
        validate gps time stamps and make sure each block is the proper length

        """
        # first check if the gps stamp blocks are of the correct length
        bad_blocks = np.where(self.gps_stamps["block_len"][1:] != self.header.ad_rate)[
            0
        ]

        if len(bad_blocks) > 0:
            if bad_blocks.max() < 5:
                ts_skip = self.gps_stamps["block_len"][0 : bad_blocks[-1] + 1].sum()
                self.gps_stamps = self.gps_stamps[bad_blocks[-1] :]
                self.time_series = self.time_series[ts_skip:]

                self.logger.warning(f"Skipped the first {bad_blocks[-1]} seconds")
                self.logger.warning(f"Skipped first {ts_skip} poins in time series")

    # ==================================================
    def convert_gps_time(self):
        """
        convert gps time integer to relative seconds from gps_week

        """
        # need to convert gps_time to type float from int
        dt = self._gps_dtype.descr
        if self.header.old_version is True:
            dt[1] = ("time", np.float32)
        else:
            dt[2] = ("time", np.float32)
        self.gps_stamps = self.gps_stamps.astype(np.dtype(dt))

        # convert to seconds
        # these are seconds relative to the gps week
        time_conv = self.gps_stamps["time"].copy() / 1024.0
        time_ms = (time_conv - np.floor(time_conv)) * 1.024
        time_conv = np.floor(time_conv) + time_ms

        self.gps_stamps["time"][:] = time_conv

    # ==================================================
    def convert_counts_to_mv(self, data):
        """
        convert the time series from counts to millivolts

        """

        data *= self._counts_to_mv_conversion
        return data

    # ==================================================
    def convert_mv_to_counts(self, data):
        """
        convert millivolts to counts assuming no other scaling has been applied

        """

        data /= self._counts_to_mv_conversion
        return data

    # ==================================================
    def get_gps_time(self, gps_int, gps_week=0):
        """
        from the gps integer get the time in seconds.

        :param int gps_int: integer from the gps time stamp line
        :param int gps_week: relative gps week, if the number of seconds is
                            larger than a week then a week is subtracted from
                            the seconds and computed from gps_week += 1
        :returns: gps_time as number of seconds from the beginning of the relative
                  gps week.

        """

        gps_seconds = gps_int / 1024.0

        gps_ms = (gps_seconds - np.floor(gps_int / 1024.0)) * (1.024)

        cc = 0
        if gps_seconds > self._week_len:
            gps_week += 1
            cc = gps_week * self._week_len
            gps_seconds -= self._week_len

        gps_time = np.floor(gps_seconds) + gps_ms + cc

        return gps_time, gps_week

    # ==================================================
    def get_UTC_date_time(self, gps_week, gps_time):
        """
        get the actual date and time of measurement as UTC.


        :param int gps_week: integer value of gps_week that the data was collected
        :param int gps_time: number of seconds from beginning of gps_week

        :return: :class:`mth5.utils.mttime.MTime`

        """
        # need to check to see if the time in seconds is more than a gps week
        # if it is add 1 to the gps week and reduce the gps time by a week
        if gps_time > self._week_len:
            gps_week += 1
            gps_time -= self._week_len

        # compute seconds using weeks and gps time
        utc_seconds = (
            self._gps_epoch.epoch_seconds + (gps_week * self._week_len) + gps_time
        )

        # compute date and time from seconds and return a datetime object
        # easier to manipulate later
        return MTime(utc_seconds, gps_time=True)

    # =================================================
    def to_channelts(self):
        """
        fill time series object
        """
        ts_type = list(self.channel_metadata.keys())[0]

        return ChannelTS(
            ts_type,
            data=self.time_series,
            channel_metadata=self.channel_metadata,
            station_metadata=self.station_metadata,
            run_metadata=self.run_metadata,
            channel_response_filter=self.channel_response,
        )


# ==============================================================================
#  Error instances for Zen
# ==============================================================================
class ZenGPSError(Exception):
    """
    error for gps timing
    """

    pass


class ZenSamplingRateError(Exception):
    """
    error for different sampling rates
    """

    pass


class ZenInputFileError(Exception):
    """
    error for input files
    """

    pass


def read_z3d(fn, logger_file_handler=None):
    """
    generic tool to read z3d file
    """

    z3d_obj = Z3D(fn)
    if logger_file_handler:
        z3d_obj.logger.addHandler(logger_file_handler)
    z3d_obj.read_z3d()
    return z3d_obj.to_channelts()
