# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:35:59 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

import numpy as np
from loguru import logger


# =============================================================================
class Z3DMetadata:
    """
    Read metadata information from a Z3D file and make each metadata entry an attribute.

    The attributes are left in capitalization of the Z3D file format.

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
    _header_length : int
        Length of header in bits (512).
    _metadata_length : int
        Length of metadata blocks (512).
    _schedule_metadata_len : int
        Length of schedule meta data (512).
    board_cal : np.ndarray or None
        Board calibration array with frequency, rate, amplitude, phase.
    cal_ant : str or None
        Antenna calibration information.
    cal_board : dict or None
        Board calibration dictionary.
    cal_ver : str or None
        Calibration version.
    ch_azimuth : str or None
        Channel azimuth.
    ch_cmp : str or None
        Channel component.
    ch_length : str or None
        Channel length (or number of coils).
    ch_number : str or None
        Channel number on the ZEN board.
    ch_xyz1 : str or None
        Channel xyz location.
    ch_xyz2 : str or None
        Channel xyz location.
    ch_cres : str or None
        Channel resistance.
    coil_cal : np.ndarray or None
        Coil calibration array (frequency, amplitude, phase).
    fid : BinaryIO or None
        File object.
    find_metadata : bool
        Boolean flag for finding metadata.
    fn : str or pathlib.Path or None
        Full path to Z3D file.
    gdp_operator : str or None
        Operator of the survey.
    gdp_progver : str or None
        Program version.
    gdp_temp : str or None
        GDP temperature.
    gdp_volt : str or None
        GDP voltage.
    job_by : str or None
        Job performed by.
    job_for : str or None
        Job for.
    job_name : str or None
        Job name.
    job_number : str or None
        Job number.
    line_name : str or None
        Survey line name.
    m_tell : int
        Location in the file where the last metadata block was found.
    notes : str or None
        Additional notes from metadata.
    rx_aspace : str or None
        Electrode spacing.
    rx_sspace : str or None
        Receiver spacing.
    rx_xazimuth : str or None
        X azimuth of electrode.
    rx_xyz0 : str or None
        Receiver xyz coordinates.
    rx_yazimuth : str or None
        Y azimuth of electrode.
    rx_zpositive : str
        Z positive direction (default 'down').
    station : str or None
        Station name.
    survey_type : str or None
        Type of survey.
    unit_length : str or None
        Length units (m).
    count : int
        Counter for metadata blocks read.

    Examples
    --------
    >>> from mth5.io.zen import Z3DMetadata
    >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
    >>> header_obj = Z3DMetadata(fn=Z3Dfn)
    >>> header_obj.read_metadata()

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
        self.find_metadata: bool = True
        self.board_cal: Optional[Union[list, np.ndarray]] = None
        self.coil_cal: Optional[Union[list, np.ndarray]] = None
        self._metadata_length: int = 512
        self._header_length: int = 512
        self._schedule_metadata_len: int = 512
        self.m_tell: int = 0

        self.cal_ant: Optional[str] = None
        self.cal_board: Optional[Dict[str, Any]] = None
        self.cal_ver: Optional[str] = None
        self.ch_azimuth: Optional[str] = None
        self.ch_cmp: Optional[str] = None
        self.ch_length: Optional[str] = None
        self.ch_number: Optional[str] = None
        self.ch_xyz1: Optional[str] = None
        self.ch_xyz2: Optional[str] = None
        self.ch_cres: Optional[str] = None
        self.gdp_operator: Optional[str] = None
        self.gdp_progver: Optional[str] = None
        self.gdp_volt: Optional[str] = None
        self.gdp_temp: Optional[str] = None
        self.job_by: Optional[str] = None
        self.job_for: Optional[str] = None
        self.job_name: Optional[str] = None
        self.job_number: Optional[str] = None
        self.rx_aspace: Optional[str] = None
        self.rx_sspace: Optional[str] = None
        self.rx_xazimuth: Optional[str] = None
        self.rx_xyz0: Optional[str] = None
        self.rx_yazimuth: Optional[str] = None
        self.rx_zpositive: str = "down"
        self.line_name: Optional[str] = None
        self.survey_type: Optional[str] = None
        self.unit_length: Optional[str] = None
        self.station: Optional[str] = None
        self.count: int = 0
        self.notes: Optional[str] = None

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def read_metadata(
        self, fn: Optional[Union[str, Path]] = None, fid: Optional[BinaryIO] = None
    ) -> None:
        """
        Read metadata from Z3D file.

        Parses the metadata blocks in a Z3D file and populates the object's
        attributes with the extracted values. Also reads calibration data
        for both board and coil calibrations.

        Parameters
        ----------
        fn : str or pathlib.Path, optional
            Full path to file. If None, uses the instance's fn attribute.
        fid : BinaryIO, optional
            Open file object. If None, uses the instance's fid attribute or
            opens the file specified by fn.

        Raises
        ------
        UnicodeDecodeError
            If metadata blocks cannot be decoded as text.

        Notes
        -----
        This method reads metadata blocks sequentially from the Z3D file,
        starting after the header and schedule metadata sections. It processes:

        - Standard metadata records with key=value pairs
        - Board calibration data (cal.brd format)
        - Coil calibration data (cal.ant format)
        - Calibration data blocks (caldata format)

        The method automatically determines the station name from available
        metadata fields in the following priority:
        1. line_name + rx_xyz0 (first coordinate)
        2. rx_stn
        3. ch_stn
        """
        if fn is not None:
            self.fn = fn
        if fid is not None:
            self.fid = fid
        if self.fn is None and self.fid is None:
            self.logger.warning("No Z3D file to read")
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
                self.m_tell = self.fid.tell() + self._metadata_length
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
                                float(tt) for tt in t_str.split(",")[-1].split(":")
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
                                self.logger.info("Could not read Calibration Data")
            else:
                self.find_metadata = False
                # need to go back to where the meta data was found so
                # we don't skip a gps time stamp
                self.m_tell = self.fid.tell() - self._metadata_length
        # make coil calibration and board calibration structured arrays
        if len(self.coil_cal) > 0:
            a = np.array(self.coil_cal)
            a = a.reshape((int(a.size / 3), 3))
            self.coil_cal = np.rec.fromrecords(a, names="frequency, amplitude, phase")
        if len(self.board_cal) > 0:
            try:
                self.board_cal = np.rec.fromrecords(
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
