# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:35:59 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
from loguru import logger


# =============================================================================
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
        self.logger = logger
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
        self.ch_cres = None
        self.gdp_operator = None
        self.gdp_progver = None
        self.gdp_volt = None
        self.gdp_temp = None
        self.job_by = None
        self.job_for = None
        self.job_name = None
        self.job_number = None
        self.rx_aspace = None
        self.rx_sspace = None
        self.rx_xazimuth = None
        self.rx_xyz0 = None
        self.rx_yazimuth = None
        self.rx_zpositive = "down"
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
                self.fid.seek(
                    self._header_length + self._schedule_metadata_len
                )
        elif self.fn is not None:
            if self.fid is None:
                self.fid = open(self.fn, "rb")
                self.fid.seek(
                    self._header_length + self._schedule_metadata_len
                )
            else:
                self.fid.seek(
                    self._header_length + self._schedule_metadata_len
                )
        # read in calibration and meta data
        self.find_metadata = True
        self.board_cal = []
        self.coil_cal = []
        self.count = 0
        cal_find = False
        while self.find_metadata == True:
            try:
                test_str = (
                    self.fid.read(self._metadata_length).decode().lower()
                )
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
                                [
                                    float(tt.strip())
                                    for tt in t_str.strip().split(":")
                                ]
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
                        for tt in t_str.split(":"):
                            try:
                                self.coil_cal.append(float(tt.strip()))
                            except ValueError:
                                pass

                elif cal_find and self.count > 3:
                    t_list = test_str.replace("|", ",").split(",")
                    for t_str in t_list:
                        if "\x00" in t_str:
                            break
                        else:
                            for tt in t_str.split(":"):
                                try:
                                    self.coil_cal.append(float(tt.strip()))
                                except ValueError:
                                    pass
            elif "caldata" in test_str:
                self.cal_board = {}
                sr = 256

                t_list = test_str.lower().split("|")
                for t_str in t_list:
                    if "\x00" in t_str:
                        continue
                    else:
                        if "cal.brd" in t_str:
                            values = [
                                float(tt)
                                for tt in t_str.split(",")[-1].split(":")
                            ]
                            self.cal_board[sr] = dict(
                                [
                                    (tkey, tvalue)
                                    for tkey, tvalue in zip(
                                        ["frequency", "amplitude", "phase"],
                                        values,
                                    )
                                ]
                            )
                        elif "cal.adfreq" in t_str:
                            sr = int(t_str.split("=")[-1])
                        elif "caldata" in t_str:
                            continue
                        else:
                            try:
                                cal_key, cal_value = t_str.split("=")
                                try:
                                    cal_value = float(cal_value)
                                except ValueError:
                                    pass
                                self.cal_board[cal_key] = cal_value
                            except ValueError:
                                self.logger.info(
                                    "Could not read Calibration Data"
                                )
            else:
                self.find_metadata = False
                # need to go back to where the meta data was found so
                # we don't skip a gps time stamp
                self.m_tell = self.fid.tell() - self._metadata_length
        # make coil calibration and board calibration structured arrays
        if len(self.coil_cal) > 0:
            a = np.array(self.coil_cal)
            a = a.reshape((int(a.size / 3), 3))
            self.coil_cal = np.core.records.fromrecords(
                a, names="frequency, amplitude, phase"
            )
        if len(self.board_cal) > 0:
            try:
                self.board_cal = np.core.records.fromrecords(
                    self.board_cal, names="frequency, rate, amplitude, phase"
                )
            except ValueError:
                self.board_cal = None
        try:
            self.station = "{0}{1}".format(
                self.line_name, self.rx_xyz0.split(":")[0]
            )
        except AttributeError:
            if hasattr(self, "rx_stn"):
                self.station = f"{self.rx_stn}"
            elif hasattr(self, "ch_stn"):
                self.station = f"{self.ch_stn}"
            else:
                self.station = None
                self.logger.warning("Need to input station name")
