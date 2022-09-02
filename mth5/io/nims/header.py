# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:57:32 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import dateutil
import logging

from mt_metadata.utils.mttime import MTime

# =============================================================================
class NIMSError(Exception):
    pass


class NIMSHeader(object):
    """
    class to hold the NIMS header information.

    A typical header looks like

    .. code-block::

        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        >>>user field>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        SITE NAME: Budwieser Spring
        STATE/PROVINCE: CA
        COUNTRY: USA
        >>> The following code in double quotes is REQUIRED to start the NIMS <<
        >>> The next 3 lines contain values required for processing <<<<<<<<<<<<
        >>> The lines after that are optional <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        "300b"  <-- 2CHAR EXPERIMENT CODE + 3 CHAR SITE CODE + RUN LETTER
        1105-3; 1305-3  <-- SYSTEM BOX I.D.; MAG HEAD ID (if different)
        106  0 <-- N-S Ex WIRE LENGTH (m); HEADING (deg E mag N)
        109  90 <-- E-W Ey WIRE LENGTH (m); HEADING (deg E mag N)
        1         <-- N ELECTRODE ID
        3          <-- E ELECTRODE ID
        2          <-- S ELECTRODE ID
        4          <-- W ELECTRODE ID
        Cu          <-- GROUND ELECTRODE INFO
        GPS INFO: 01/10/19 16:16:42 1616.7000 3443.6088 115.7350 W 946.6
        OPERATOR: KP
        COMMENT: N/S CRS: .95/.96 DCV: 3.5 ACV:1
        E/W CRS: .85/.86 DCV: 1.5 ACV: 1
        Redeployed site for run b b/c possible animal disturbance

    """

    def __init__(self, fn=None):
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.fn = fn
        self._max_header_length = 1000
        self.header_dict = None
        self.site_name = None
        self.state_province = None
        self.country = None
        self.box_id = None
        self.mag_id = None
        self.ex_length = None
        self.ex_azimuth = None
        self.ey_length = None
        self.ey_azimuth = None
        self.n_electrode_id = None
        self.s_electrode_id = None
        self.e_electrode_id = None
        self.w_electrode_id = None
        self.ground_electrode_info = None
        self.header_gps_stamp = None
        self.header_gps_latitude = None
        self.header_gps_longitude = None
        self.header_gps_elevation = None
        self.operator = None
        self.comments = None
        self.run_id = None
        self.data_start_seek = 0

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value):
        if value is not None:
            self._fn = Path(value)
        else:
            self._fn = None

    def read_header(self, fn=None):
        """
        read header information

        :param fn: full path to file to read
        :type fn: string or :class:`pathlib.Path`
        :raises: :class:`mth5.io.nims.NIMSError` if something is not right.

        """
        if fn is not None:
            self.fn = fn

        if not self.fn.exists():
            msg = f"Could not find nims file {self.fn}"
            self.logger.error(msg)
            raise NIMSError(msg)

        self.logger.info(f"Reading NIMS file {self.fn}")

        ### load in the entire file, its not too big
        with open(self.fn, "rb") as fid:
            header_str = fid.read(self._max_header_length)
            header_list = header_str.split(b"\r")

        self.header_dict = {}
        last_index = len(header_list)
        last_line = header_list[-1]
        for ii, line in enumerate(header_list[0:-1]):
            if ii == last_index:
                break
            if b"comments" in line.lower():
                last_line = header_list[ii + 1]
                last_index = ii + 1

            line = line.decode()
            if line.find(">") == 0:
                continue
            elif line.find(":") > 0:
                key, value = line.split(":", 1)
                self.header_dict[key.strip().lower()] = value.strip()
            elif line.find("<--") > 0:
                value, key = line.split("<--")
                self.header_dict[key.strip().lower()] = value.strip()
        ### sometimes there are some spaces before the data starts
        if last_line.count(b" ") > 0:
            if last_line[0:1] == b" ":
                last_line = last_line.strip()
            else:
                last_line = last_line.split()[1].strip()
        data_start_byte = last_line[0:1]
        ### sometimes there are rogue $ around
        if data_start_byte in [b"$", b"g"]:
            data_start_byte = last_line[1:2]
        self.data_start_seek = header_str.find(data_start_byte)

        self.parse_header_dict()

    def parse_header_dict(self, header_dict=None):
        """
        parse the header dictionary into something useful
        """
        if header_dict is not None:
            self.header_dict = header_dict

        assert isinstance(self.header_dict, dict)

        for key, value in self.header_dict.items():
            if "wire" in key:
                if key.find("n") == 0:
                    self.ex_length = float(value.split()[0])
                    self.ex_azimuth = float(value.split()[1])
                elif key.find("e") == 0:
                    self.ey_length = float(value.split()[0])
                    self.ey_azimuth = float(value.split()[1])
            elif "system" in key:
                self.box_id = value.split(";")[0].strip()
                self.mag_id = value.split(";")[1].strip()
            elif "gps" in key:
                gps_list = value.split()
                self.header_gps_stamp = MTime(
                    dateutil.parser.parse(
                        " ".join(gps_list[0:2]), dayfirst=True
                    )
                )
                self.header_gps_latitude = self._get_latitude(
                    gps_list[2], gps_list[3]
                )
                self.header_gps_longitude = self._get_longitude(
                    gps_list[4], gps_list[5]
                )
                self.header_gps_elevation = float(gps_list[6])
            elif "run" in key:
                self.run_id = value.replace('"', "")
            else:
                setattr(self, key.replace(" ", "_").replace("/", "_"), value)

    def _get_latitude(self, latitude, hemisphere):
        if not isinstance(latitude, float):
            latitude = float(latitude)
        if hemisphere.lower() == "n":
            return latitude
        if hemisphere.lower() == "s":
            return -1 * latitude

    def _get_longitude(self, longitude, hemisphere):
        if not isinstance(longitude, float):
            longitude = float(longitude)
        if hemisphere.lower() == "e":
            return longitude
        if hemisphere.lower() == "w":
            return -1 * longitude
