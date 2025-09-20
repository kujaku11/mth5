# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:24:57 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from mt_metadata.common.mttime import MTime
from loguru import logger


# =============================================================================
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
        self.logger = logger
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
        self.initial_start = MTime(time_stamp=None)

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
        self.initial_start = MTime(time_stamp=f"{self.Date}T{self.Time}", gps_time=True)
