# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:57:32 2022

@author: jpeacock
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Optional, Union

import dateutil
from loguru import logger
from mt_metadata.common.mttime import MTime


# =============================================================================
class NIMSError(Exception):
    pass


class NIMSHeader:
    """
    Class to hold NIMS header information.

    This class parses and stores header information from NIMS DATA.BIN files.
    The header contains metadata about the measurement site, equipment setup,
    GPS coordinates, electrode configuration, and other survey parameters.

    Parameters
    ----------
    fn : str or Path, optional
        Path to the NIMS file to read, by default None

    Attributes
    ----------
    fn : Path or None
        Path to the NIMS file
    site_name : str or None
        Name of the measurement site
    state_province : str or None
        State or province of the measurement location
    country : str or None
        Country of the measurement location
    box_id : str or None
        System box identifier
    mag_id : str or None
        Magnetometer head identifier
    ex_length : float or None
        North-South electric field wire length in meters
    ex_azimuth : float or None
        North-South electric field wire heading in degrees
    ey_length : float or None
        East-West electric field wire length in meters
    ey_azimuth : float or None
        East-West electric field wire heading in degrees
    n_electrode_id : str or None
        North electrode identifier
    s_electrode_id : str or None
        South electrode identifier
    e_electrode_id : str or None
        East electrode identifier
    w_electrode_id : str or None
        West electrode identifier
    ground_electrode_info : str or None
        Ground electrode information
    header_gps_stamp : MTime or None
        GPS timestamp from header
    header_gps_latitude : float or None
        GPS latitude from header in decimal degrees
    header_gps_longitude : float or None
        GPS longitude from header in decimal degrees
    header_gps_elevation : float or None
        GPS elevation from header in meters
    operator : str or None
        Operator name
    comments : str or None
        Survey comments
    run_id : str or None
        Run identifier
    data_start_seek : int
        Byte position where data begins in file

    Examples
    --------
    A typical header looks like:

    .. code-block::

        '''
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
        3         <-- E ELECTRODE ID
        2         <-- S ELECTRODE ID
        4         <-- W ELECTRODE ID
        Cu        <-- GROUND ELECTRODE INFO
        GPS INFO: 26/09/19 18:29:29 34.7268 N 115.7350 W 939.8
        OPERATOR: KP
        COMMENT: N/S CRS: .95/.96 DCV: 3.5 ACV:1
        E/W CRS: .85/.86 DCV: 1.5 ACV: 1
        Redeployed site for run b b/c possible animal disturbance
        '''
    """

    def __init__(self, fn: Optional[Union[str, Path]] = None) -> None:
        self.logger = logger
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
    def fn(self) -> Optional[Path]:
        """
        Full path to NIMS file.

        Returns
        -------
        Path or None
            Path object representing the NIMS file location,
            or None if no file is set
        """
        return self._fn

    @fn.setter
    def fn(self, value: Optional[Union[str, Path]]) -> None:
        if value is not None:
            self._fn = Path(value)
        else:
            self._fn = None

    @property
    def station(self) -> Optional[str]:
        """
        Station ID derived from run ID.

        Returns
        -------
        str or None
            Station identifier (run ID without the last character),
            or None if run_id is not set

        Notes
        -----
        The station ID is typically the run ID with the last character
        (run letter) removed.
        """
        if self.run_id is not None:
            return self.run_id[0:-1]

    @property
    def file_size(self) -> Optional[int]:
        """
        Size of the NIMS file in bytes.

        Returns
        -------
        int or None
            File size in bytes, or None if no file is set

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        """
        if self.fn is not None:
            return self.fn.stat().st_size

    def read_header(self, fn: Optional[Union[str, Path]] = None) -> None:
        """
        Read header information from a NIMS file.

        This method reads and parses the header section of a NIMS DATA.BIN file,
        extracting metadata about the survey setup, GPS coordinates, electrode
        configuration, and other parameters.

        Parameters
        ----------
        fn : str or Path, optional
            Full path to NIMS file to read. Uses self.fn if not provided.

        Raises
        ------
        NIMSError
            If the file does not exist or cannot be read

        Notes
        -----
        The method reads up to _max_header_length bytes from the beginning
        of the file, parses the header information, and stores the results
        in the header_dict attribute and individual properties.
        """
        if fn is not None:
            self.fn = fn
        if not self.fn.exists():
            msg = f"Could not find nims file {self.fn}"
            self.logger.error(msg)
            raise NIMSError(msg)
        self.logger.debug(f"Reading NIMS file {self.fn}")

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

    def parse_header_dict(self, header_dict: Optional[dict[str, str]] = None) -> None:
        """
        Parse the header dictionary into individual attributes.

        This method takes the raw header dictionary and extracts specific
        information into class attributes for easy access.

        Parameters
        ----------
        header_dict : dict of str, optional
            Dictionary containing header key-value pairs. Uses self.header_dict
            if not provided.

        Notes
        -----
        Parses various header fields including:
        - Wire lengths and azimuths for electric field measurements
        - System box and magnetometer IDs
        - GPS coordinates and timestamp
        - Run identifier
        - Other metadata fields
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
                    time_stamp=dateutil.parser.parse(
                        " ".join(gps_list[0:2]), dayfirst=True
                    ).isoformat()
                )
                self.header_gps_latitude = self._get_latitude(gps_list[2], gps_list[3])
                self.header_gps_longitude = self._get_longitude(
                    gps_list[4], gps_list[5]
                )
                self.header_gps_elevation = float(gps_list[6])
            elif "run" in key:
                self.run_id = value.replace('"', "")
            else:
                setattr(self, key.replace(" ", "_").replace("/", "_"), value)

    def _get_latitude(self, latitude: Union[str, float], hemisphere: str) -> float:
        """
        Get latitude as decimal degrees with proper sign.

        Parameters
        ----------
        latitude : str or float
            Latitude value in decimal degrees
        hemisphere : str
            Hemisphere identifier ('N' for North, 'S' for South)

        Returns
        -------
        float
            Latitude in decimal degrees with proper sign
            (positive for North, negative for South)

        Notes
        -----
        Converts latitude to proper sign convention where North is positive
        and South is negative.
        """
        if not isinstance(latitude, float):
            latitude = float(latitude)
        if hemisphere.lower() == "n":
            return latitude
        if hemisphere.lower() == "s":
            return -1 * latitude

    def _get_longitude(self, longitude: Union[str, float], hemisphere: str) -> float:
        """
        Get longitude as decimal degrees with proper sign.

        Parameters
        ----------
        longitude : str or float
            Longitude value in decimal degrees
        hemisphere : str
            Hemisphere identifier ('E' for East, 'W' for West)

        Returns
        -------
        float
            Longitude in decimal degrees with proper sign
            (positive for East, negative for West)

        Notes
        -----
        Converts longitude to proper sign convention where East is positive
        and West is negative.
        """
        if not isinstance(longitude, float):
            longitude = float(longitude)
        if hemisphere.lower() == "e":
            return longitude
        if hemisphere.lower() == "w":
            return -1 * longitude
