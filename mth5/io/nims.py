# -*- coding: utf-8 -*-
"""
===============
NIMS
===============

    * deals with reading in NIMS DATA.BIN files
    
    This is a translation from Matlab codes written and edited by:
        * Anna Kelbert
        * Paul Bedrosian
        * Esteban Bowles-Martinez
        * Possibly others.
        
    I've tested it against a version, and it matches.  The data/GPS  gaps I
    still don't understand so for now the time series is just 
    made continuous and the number of missing seconds is clipped from the 
    end of the time series.
    
    .. note:: this only works for 8Hz data for now
    
    
:copyright:
    Jared Peacock (jpeacock@usgs.gov)
    
:license: 
    MIT
"""

# =============================================================================
# Imports
# =============================================================================
import os
import struct
import datetime
import dateutil
import logging

import numpy as np
import pandas as pd

from mth5 import timeseries

# =============================================================================
# Exceptions
# =============================================================================
class NIMSError(Exception):
    pass


class GPSError(Exception):
    pass


class ResponseError(Exception):
    pass


# =============================================================================
# class objects
# =============================================================================
class GPS(object):
    """
    class to parse GPS stamp from the NIMS
    
    Depending on the type of Stamp different attributes will be filled.
    
    GPRMC has full date and time information and declination
    GPGGA has elevation data
    
    .. note:: GPGGA date is set to 1980-01-01 so that the time can be estimated.
              Should use GPRMC for accurate date/time information.  
    """

    def __init__(self, gps_string, index=0):

        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

        self.gps_string = gps_string
        self.index = index
        self._time = None
        self._date = "010180"
        self._latitude = None
        self._latitude_hemisphere = None
        self._longitude = None
        self._longitude_hemisphere = None
        self._declination = None
        self._declination_hemisphere = None
        self._elevation = None
        self.valid = False
        self.elevation_units = "meters"

        self.type_dict = {
            "gprmc": {
                0: "type",
                1: "time",
                2: "fix",
                3: "latitude",
                4: "latitude_hemisphere",
                5: "longitude",
                6: "longitude_hemisphere",
                7: "skip",
                8: "skip",
                9: "date",
                10: "declination",
                11: "declination_hemisphere",
                "length": [12],
                "type": 0,
                "time": 1,
                "fix": 2,
                "latitude": 3,
                "latitude_hemisphere": 4,
                "longitude": 5,
                "longitude_hemisphere": 6,
                "date": 9,
                "declination": 10,
            },
            "gpgga": {
                0: "type",
                1: "time",
                2: "latitude",
                3: "latitude_hemisphere",
                4: "longitude",
                5: "longitude_hemisphere",
                6: "var_01",
                7: "var_02",
                8: "var_03",
                9: "elevation",
                10: "elevation_units",
                11: "elevation_error",
                12: "elevation_error_units",
                13: "null_01",
                14: "null_02",
                "length": [14, 15],
                "type": 0,
                "time": 1,
                "latitude": 2,
                "latitude_hemisphere": 3,
                "longitude": 4,
                "longitude_hemisphere": 5,
                "elevation": 9,
                "elevation_units": 10,
                "elevation_error": 11,
                "elevation_error_units": 12,
            },
        }
        self.parse_gps_string(self.gps_string)

    def __str__(self):
        """ string representation """
        msg = [
            f"type = {self.gps_type}",
            f"index = {self.index}",
            f"fix = {self.fix}",
            f"time_stamp =  {self.time_stamp}",
            f"latitude = {self.latitude}",
            f"longitude = {self.longitude}",
            f"elevation = {self.elevation}",
            f"declination = {self.declination}",
        ]

        return "\n".join(msg)

    def __repr__(self):
        return self.__str__()

    def validate_gps_string(self, gps_string):
        """
        make sure the string is valid, remove any binary numbers and find
        the end of the string as '*'
        
        :param string gps_string: raw GPS string to be validated
        
        :returns: validated string or None if there is something wrong
        """
        for replace_str in [b"\xd9", b"\xc7", b"\xcc"]:
            gps_string = gps_string.replace(replace_str, b"")

        ### sometimes the end is set with a zero for some reason
        gps_string = gps_string.replace(b"\x00", b"*")

        if gps_string.find(b"*") < 0:
            logging.debug("GPSError: No end to stamp {0}".format(gps_string))
        else:
            try:
                gps_string = gps_string[0 : gps_string.find(b"*")].decode()
                return gps_string
            except UnicodeDecodeError:
                logging.debug(
                    "GPSError: stamp not correct format, {0}".format(
                        gps_string
                    )
                )
                return None

    def parse_gps_string(self, gps_string):
        """
        Parse a raw gps string from the NIMS and set appropriate attributes.
        GPS string will first be validated, then parsed. 
        
        :param string gps_string: raw GPS string to be parsed
        """
        gps_string = self.validate_gps_string(gps_string)
        if gps_string is None:
            self.valid = False
            return

        if isinstance(gps_string, bytes):
            gps_list = gps_string.strip().split(b",")
            gps_list = [value.decode() for value in gps_list]
        else:
            gps_list = gps_string.strip().split(",")

        if len(gps_list) > 1:
            if len(gps_list[1]) > 6:
                self.logger.debug(
                    "GPS time and lat missing a comma adding one, check time"
                )
                gps_list = (
                    gps_list[0:1]
                    + [gps_list[1][0:6], gps_list[1][6:]]
                    + gps_list[2:]
                )

        ### validate the gps list to make sure it is usable
        gps_list, error_list = self.validate_gps_list(gps_list)
        if len(error_list) > 0:
            for error in error_list:
                logging.debug("GPSError: " + error)
        if gps_list is None:
            return

        attr_dict = self.type_dict[gps_list[0].lower()]

        for index, value in enumerate(gps_list):
            setattr(self, "_" + attr_dict[index], value)

        if None not in gps_list:
            self.valid = True
            self.gps_string = gps_string

    def validate_gps_list(self, gps_list):
        """
        check to make sure the gps stamp is the correct format, checks each element
        for the proper format
        
        :param gps_list: a parsed gps string from a NIMS
        :type gps_list: list
        :raises: :class:`mth5.io.nims.GPSError` if anything is wrong.
        """
        error_list = []
        try:
            gps_list = self._validate_gps_type(gps_list)
        except GPSError as error:
            error_list.append(error.args[0])
            return None, error_list

        ### get the string type
        g_type = gps_list[0].lower()

        ### first check the length, if it is not the proper length then
        ### return, cause you never know if everything else is correct
        try:
            self._validate_list_length(gps_list)
        except GPSError as error:
            error_list.append(error.args[0])
            return None, error_list

        try:
            gps_list[self.type_dict[g_type]["time"]] = self._validate_time(
                gps_list[self.type_dict[g_type]["time"]]
            )
        except GPSError as error:
            error_list.append(error.args[0])
            gps_list[self.type_dict[g_type]["time"]] = None

        try:
            gps_list[
                self.type_dict[g_type]["latitude"]
            ] = self._validate_latitude(
                gps_list[self.type_dict[g_type]["latitude"]],
                gps_list[self.type_dict[g_type]["latitude_hemisphere"]],
            )
        except GPSError as error:
            error_list.append(error.args[0])
            gps_list[self.type_dict[g_type]["latitude"]] = None

        try:
            gps_list[
                self.type_dict[g_type]["longitude"]
            ] = self._validate_longitude(
                gps_list[self.type_dict[g_type]["longitude"]],
                gps_list[self.type_dict[g_type]["longitude_hemisphere"]],
            )
        except GPSError as error:
            error_list.append(error.args[0])
            gps_list[self.type_dict[g_type]["longitude"]] = None

        if g_type == "gprmc":
            try:
                gps_list[
                    self.type_dict["gprmc"]["date"]
                ] = self._validate_date(
                    gps_list[self.type_dict["gprmc"]["date"]]
                )
            except GPSError as error:
                error_list.append(error.args[0])
                gps_list[self.type_dict[g_type]["date"]] = None

        elif g_type == "gpgga":
            try:
                gps_list[
                    self.type_dict["gpgga"]["elevation"]
                ] = self._validate_elevation(
                    gps_list[self.type_dict["gpgga"]["elevation"]]
                )
            except GPSError as error:
                error_list.append(error.args[0])
                gps_list[self.type_dict["gpgga"]["elevation"]] = None

        return gps_list, error_list

    def _validate_gps_type(self, gps_list):
        """Validate gps type should be gpgga or gprmc"""
        gps_type = gps_list[0].lower()
        if "gpg" in gps_type:
            if len(gps_type) > 5:
                gps_list = ["GPGGA", gps_type[-6:]] + gps_list[1:]
            elif len(gps_type) < 5:
                gps_list[0] = "GPGGA"
        elif "gpr" in gps_type:
            if len(gps_type) > 5:
                gps_list = ["GPRMC", gps_type[-6:]] + gps_list[1:]
            elif len(gps_type) < 5:
                gps_list[0] = "GPRMC"

        gps_type = gps_list[0].lower()
        if gps_type not in ["gpgga", "gprmc"]:
            raise GPSError(
                "GPS String type not correct.  "
                + "Expect GPGGA or GPRMC, got {0}".format(gps_type.upper())
            )

        return gps_list

    def _validate_list_length(self, gps_list):
        """validate gps list length based on type of string"""

        gps_list_type = gps_list[0].lower()
        expected_len = self.type_dict[gps_list_type]["length"]
        if len(gps_list) not in expected_len:
            raise GPSError(
                "GPS string not correct length for {0}.  ".format(
                    gps_list_type.upper()
                )
                + "Expected {0}, got {1} \n{2}".format(
                    expected_len, len(gps_list), ",".join(gps_list)
                )
            )

    def _validate_time(self, time_str):
        """ validate time string, should be 6 characters long and an int """
        if len(time_str) != 6:
            raise GPSError(
                "Length of time string {0} not correct.  ".format(time_str)
                + "Expected 6 got {0}".format(len(time_str))
            )
        try:
            int(time_str)
        except ValueError:
            raise GPSError(
                "Could not convert time string {0}".format(time_str)
            )

        return time_str

    def _validate_date(self, date_str):
        """ validate date string, should be 6 characters long and an int """
        if len(date_str) != 6:
            raise GPSError(
                "Length of date string not correct {0}.  ".format(date_str)
                + "Expected 6 got {0}".format(len(date_str))
            )
        try:
            int(date_str)
        except ValueError:
            raise GPSError(
                "Could not convert date string {0}".format(date_str)
            )

        return date_str

    def _validate_latitude(self, latitude_str, hemisphere_str):
        """validate latitude, should have hemisphere string with it"""

        if len(latitude_str) < 8:
            raise GPSError(
                "Latitude string should be larger than 7 characters.  "
                + "Got {0}".format(len(latitude_str))
            )
        if len(hemisphere_str) != 1:
            raise GPSError(
                "Latitude hemisphere should be 1 character.  "
                + "Got {0}".format(len(hemisphere_str))
            )
        if hemisphere_str.lower() not in ["n", "s"]:
            raise GPSError(
                "Latitude hemisphere {0} not understood".format(
                    hemisphere_str.upper()
                )
            )
        try:
            float(latitude_str)
        except ValueError:
            raise GPSError(
                "Could not convert latitude string {0}".format(latitude_str)
            )

        return latitude_str

    def _validate_longitude(self, longitude_str, hemisphere_str):
        """validate longitude, should have hemisphere string with it"""

        if len(longitude_str) < 8:
            raise GPSError(
                "Longitude string should be larger than 7 characters.  "
                + "Got {0}".format(len(longitude_str))
            )
        if len(hemisphere_str) != 1:
            raise GPSError(
                "Longitude hemisphere should be 1 character.  "
                + "Got {0}".format(len(hemisphere_str))
            )
        if hemisphere_str.lower() not in ["e", "w"]:
            raise GPSError(
                "Longitude hemisphere {0} not understood".format(
                    hemisphere_str.upper()
                )
            )
        try:
            float(longitude_str)
        except ValueError:
            raise GPSError(
                "Could not convert longitude string {0}".format(longitude_str)
            )

        return longitude_str

    def _validate_elevation(self, elevation_str):
        """validate elevation, check for converstion to float"""
        elevation_str = elevation_str.lower().replace("m", "")
        try:
            elevation_str = f"{float(elevation_str):0.2f}"
        except ValueError:
            raise GPSError(f"Elevation could not be converted {elevation_str}")

        return elevation_str

    @property
    def latitude(self):
        """
        Latitude in decimal degrees, WGS84
        """
        if (
            self._latitude is not None
            and self._latitude_hemisphere is not None
        ):
            index = len(self._latitude) - 7
            lat = (
                float(self._latitude[0:index])
                + float(self._latitude[index:]) / 60
            )
            if "s" in self._latitude_hemisphere.lower():
                lat *= -1
            return lat
        return 0.0

    @property
    def longitude(self):
        """
        Latitude in decimal degrees, WGS84
        """
        if (
            self._longitude is not None
            and self._longitude_hemisphere is not None
        ):
            index = len(self._longitude) - 7
            lon = (
                float(self._longitude[0:index])
                + float(self._longitude[index:]) / 60
            )
            if "w" in self._longitude_hemisphere.lower():
                lon *= -1
            return lon
        return 0.0

    @property
    def elevation(self):
        """
        elevation in meters
        """
        if self._elevation is not None:
            try:
                return float(self._elevation)
            except ValueError:
                self.logger.error(
                    "GPSError: Could not get elevation GPS string"
                    + f"not complete {self.gps_string}"
                )
        return 0.0

    @property
    def time_stamp(self):
        """
        return a datetime object of the time stamp
        """
        if self._time is None:
            return None
        if self._date is None:
            self._date = "010180"
        try:
            return dateutil.parser.parse(
                "{0} {1}".format(self._date, self._time), dayfirst=True
            )
        except ValueError:
            self.logger.error(f"GPSError: bad date string {self.gps_string}")
            return None

    @property
    def declination(self):
        """
        geomagnetic declination in degrees from north
        """
        if self._declination is None or self._declination_hemisphere is None:
            return None

        dec = float(self._declination)
        if "w" in self._declination_hemisphere.lower():
            dec *= -1
        return dec

    @property
    def gps_type(self):
        """GPRMC or GPGGA"""
        return self._type

    @property
    def fix(self):
        """
        GPS fixed
        """
        if hasattr(self, "_fix"):
            return self._fix
        return None


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

    def read_header(self, fn=None):
        """
        read header information
        
        :param fn: full path to file to read
        :type fn: string or :class:`pathlib.Path`
        :raises: :class:`mth5.io.nims.NIMSError` if something is not right.
        
        """
        if fn is not None:
            self.fn = fn

        if not os.path.exists(self.fn):
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
                self.header_gps_stamp = dateutil.parser.parse(
                    " ".join(gps_list[0:2]), dayfirst=True
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


class NIMS(NIMSHeader):
    """
    NIMS Class will read in a NIMS DATA.BIN file.
    
    A fast way to read the binary files are to first read in the GPS strings, 
    the third byte in each block as a character and parse that into valid 
    GPS stamps.
    
    Then read in the entire data set as unsigned 8 bit integers and reshape
    the data to be n seconds x block size.  Then parse that array into the 
    status information and data.
    
    I only have a limited amount of .BIN files to test so this will likely 
    break if there are issues such as data gaps.  This has been tested against the 
    matlab program loadNIMS by Anna Kelbert and the match for all the .bin files
    I have.  If something looks weird check it against that program.
    
    .. todo:: deal with timing issues, right now a warning is sent to the user
              need to figure out a way to find where the gap is and adjust
              accordingly.
              
    .. warning:: 
        Currently Only 8 Hz data is supported 
        
    """

    def __init__(self, fn=None):
        super().__init__(fn)

        # change thes if the sample rate is different
        self.block_size = 131
        self.block_sequence = [1, self.block_size]
        self.sample_rate = 8  ### samples/second
        self.e_conversion_factor = 2.44141221047903e-06
        self.h_conversion_factor = 0.01
        self.t_conversion_factor = 70
        self.t_offset = 18048
        self._int_max = 8388608
        self._int_factor = 16777216
        self._block_dict = {
            "soh": 0,
            "block_len": 1,
            "status": 2,
            "gps": 3,
            "sequence": 4,
            "box_temp": (5, 6),
            "head_temp": (7, 8),
            "logic": 81,
            "end": 130,
        }
        self.info_array = None
        self.stamps = None
        self.ts = None
        self.gaps = None
        self.duplicate_list = None

        self._raw_string = None

        self.indices = self._make_index_values()

    @property
    def latitude(self):
        """
        median latitude value from all the GPS stamps in decimal degrees
        WGS84
        
        Only get from the GPRMC stamp as they should be duplicates
        """
        if self.stamps is not None:
            latitude = np.zeros(len(self.stamps))
            for ii, stamp in enumerate(self.stamps):
                latitude[ii] = stamp[1][0].latitude
            return np.median(latitude[np.nonzero(latitude)])
        return self.header_gps_latitude

    @property
    def longitude(self):
        """
        median longitude value from all the GPS stamps in decimal degrees
        WGS84
        
        Only get from the first stamp within the sets
        """
        if self.stamps is not None:
            longitude = np.zeros(len(self.stamps))
            for ii, stamp in enumerate(self.stamps):
                longitude[ii] = stamp[1][0].longitude
            return np.median(longitude[np.nonzero(longitude)])
        return self.header_gps_longitude

    @property
    def elevation(self):
        """
        median elevation value from all the GPS stamps in decimal degrees
        WGS84
        
        Only get from the first stamp within the sets
        """
        if self.stamps is not None:
            elevation = np.zeros(len(self.stamps))
            for ii, stamp in enumerate(self.stamps):
                if len(stamp[1]) == 1:
                    elev = stamp[1][0].elevation
                if len(stamp[1]) == 2:
                    elev = stamp[1][1].elevation
                if elev is None:
                    continue
                elevation[ii] = elev
            return np.median(elevation[np.nonzero(elevation)])
        return self.header_gps_elevation

    @property
    def declination(self):
        """
        median elevation value from all the GPS stamps in decimal degrees
        WGS84
        
        Only get from the first stamp within the sets
        """
        if self.stamps is not None:
            declination = np.zeros(len(self.stamps))
            for ii, stamp in enumerate(self.stamps):
                if stamp[1][0].gps_type == "GPRMC":
                    dec = stamp[1][0].declination
                if dec is None:
                    continue
                declination[ii] = dec
            return np.median(declination[np.nonzero(declination)])
        return None

    @property
    def start_time(self):
        """
        start time is the first good GPS time stamp minus the seconds to the
        beginning of the time series.
        """
        if self.stamps is not None:
            return self.ts.index[0]
        return None

    @property
    def end_time(self):
        """
        start time is the first good GPS time stamp minus the seconds to the
        beginning of the time series.
        """
        if self.stamps is not None:
            return self.ts.index[-1]
        return None

    @property
    def box_temperature(self):
        """data logger temperature, sampled at 1 second"""

        if self.ts is not None:
            meta_dict = {
                "channel_number": 6,
                "component": "temperature",
                "measurement_azimuth": 0,
                "measurement_tilt": 0,
                "sample_rate": 1,
                "time_period.start": self.start_time.isoformat(),
                "time_period.end": self.end_time.isoformat(),
                "type": "auxiliary",
                "units": "celsius",
            }

            temp = timeseries.ChannelTS(
                "auxiliary",
                data=self.info_array["box_temp"],
                channel_metadata={"auxiliary": meta_dict},
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
            # interpolate temperature onto the same sample rate as the channels.
            temp.ts = temp.ts.interp_like(self.hx.ts)
            temp.metadata.sample_rate = self.sample_rate
            temp.metadata.time_period.end = self.end_time.isoformat()

            return temp
        return None

    @property
    def hx(self):
        """HX"""
        if self.ts is not None:
            meta_dict = {
                "channel_number": 1,
                "component": "hx",
                "measurement_azimuth": 0,
                "measurement_tilt": 0,
                "sample_rate": self.sample_rate,
                "time_period.start": self.start_time.isoformat(),
                "time_period.end": self.end_time.isoformat(),
                "type": "magnetic",
                "units": "counts",
                "sensor.id": self.mag_id,
                "sensor.manufacturer": "Barry Narod",
                "sensor.type": "fluxgate triaxial magnetometer",
            }

            return timeseries.ChannelTS(
                "magnetic",
                data=self.ts.hx.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
        return None

    @property
    def hy(self):
        """HY"""
        if self.ts is not None:
            meta_dict = {
                "channel_number": 2,
                "component": "hy",
                "measurement_azimuth": 90,
                "measurement_tilt": 0,
                "sample_rate": self.sample_rate,
                "time_period.start": self.start_time.isoformat(),
                "time_period.end": self.end_time.isoformat(),
                "type": "magnetic",
                "units": "counts",
                "sensor.id": self.mag_id,
                "sensor.manufacturer": "Barry Narod",
                "sensor.type": "fluxgate triaxial magnetometer",
            }

            return timeseries.ChannelTS(
                "magnetic",
                data=self.ts.hy.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
        return None

    @property
    def hz(self):
        """HZ"""
        if self.ts is not None:
            meta_dict = {
                "channel_number": 3,
                "component": "hz",
                "measurement_azimuth": 0,
                "measurement_tilt": 90,
                "sample_rate": self.sample_rate,
                "time_period.start": self.start_time.isoformat(),
                "time_period.end": self.end_time.isoformat(),
                "type": "magnetic",
                "units": "counts",
                "sensor.id": self.mag_id,
                "sensor.manufacturer": "Barry Narod",
                "sensor.type": "fluxgate triaxial magnetometer",
            }

            return timeseries.ChannelTS(
                "magnetic",
                data=self.ts.hz.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
        return None

    @property
    def ex(self):
        """EX"""
        if self.ts is not None:
            meta_dict = {
                "channel_number": 4,
                "component": "ex",
                "measurement_azimuth": self.ex_azimuth,
                "measurement_tilt": 0,
                "sample_rate": self.sample_rate,
                "dipole_length": self.ex_length,
                "time_period.start": self.start_time.isoformat(),
                "time_period.end": self.end_time.isoformat(),
                "type": "electric",
                "units": "counts",
                "negative.id": self.s_electrode_id,
                "positive.id": self.n_electrode_id,
            }

            return timeseries.ChannelTS(
                "electric",
                data=self.ts.ex.to_numpy(),
                channel_metadata={"electric": meta_dict},
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
        return None

    @property
    def ey(self):
        """EY"""
        if self.ts is not None:
            meta_dict = {
                "channel_number": 5,
                "component": "ey",
                "measurement_azimuth": self.ey_azimuth,
                "measurement_tilt": 0,
                "sample_rate": self.sample_rate,
                "dipole_length": self.ey_length,
                "time_period.start": self.start_time.isoformat(),
                "time_period.end": self.end_time.isoformat(),
                "type": "electric",
                "units": "counts",
                "negative.id": self.w_electrode_id,
                "positive.id": self.e_electrode_id,
            }

            return timeseries.ChannelTS(
                "electric",
                data=self.ts.ey.to_numpy(),
                channel_metadata={"electric": meta_dict},
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )

        return None

    @property
    def run_metadata(self):
        """ Run metadata """

        if self.ts is not None:
            meta_dict = {
                "Run": {
                    "channels_recorded_electric": "ex, ey",
                    "channels_recorded_magnetic": "hx, hy, hz",
                    "channels_recorded_auxiliary": "temperature",
                    "comments": self.comments,
                    "data_logger.firmware.author": "B. Narod",
                    "data_logger.firmware.name": "nims",
                    "data_logger.firmware.version": "1.0",
                    "data_logger.manufacturer": "Narod",
                    "data_logger.model": self.box_id,
                    "data_logger.id": self.box_id,
                    "data_logger.type": "long period",
                    "id": self.run_id,
                    "data_type": "MTLP",
                    "sample_rate": self.sample_rate,
                    "time_period.end": self.end_time.isoformat(),
                    "time_period.start": self.start_time.isoformat(),
                }
            }

            return meta_dict

        return None

    @property
    def station_metadata(self):
        """ Station metadata from nims file """
        if self.ts is not None:

            return {
                "Station": {
                    "geographic_name": f"{self.site_name}, {self.state_province}, {self.country}",
                    "location.declination.value": self.declination,
                    "location.elevation": self.elevation,
                    "location.latitude": self.latitude,
                    "location.longitude": self.longitude,
                    "id": self.run_id[0:-1],
                    "orientation.reference_frame": "geomagnetic",
                }
            }
        return None

    def to_runts(self):
        """ Get xarray for run """

        if self.ts is not None:
            return timeseries.RunTS(
                array_list=[
                    self.hx,
                    self.hy,
                    self.hz,
                    self.ex,
                    self.ey,
                    self.box_temperature,
                ],
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )

        return None

    def _make_index_values(self):
        """
        Index values for the channels recorded
        """
        ### make an array of index values for magnetics and electrics
        indices = np.zeros((8, 5), dtype=np.int)
        for kk in range(8):
            ### magnetic blocks
            for ii in range(3):
                indices[kk, ii] = 9 + (kk) * 9 + (ii) * 3
            ### electric blocks
            for ii in range(2):
                indices[kk, 3 + ii] = 82 + (kk) * 6 + (ii) * 3
        return indices

    def _get_gps_string_list(self, nims_string):
        """
        get the gps strings from the raw string output by the NIMS.  This will
        take the 3rd value in each block, concatenate into a long string and
        then make a list by splitting by '$'.  The index values of where the
        '$' are found are also calculated.
        
        :param str nims_string: raw binary string output by NIMS
        
        :returns: list of index values associated with the location of the '$'
        
        :returns: list of possible raw GPS strings
        
        .. note:: This assumes that there are an even amount of data blocks.  
                  Might be a bad assumption          
        """
        ### get index values of $ and gps_strings
        index_values = []
        gps_str_list = []
        for ii in range(int(len(nims_string) / self.block_size)):
            index = ii * self.block_size + 3
            g_char = struct.unpack("c", nims_string[index : index + 1])[0]
            if g_char == b"$":
                index_values.append((index - 3) / self.block_size)
            gps_str_list.append(g_char)
        gps_raw_stamp_list = b"".join(gps_str_list).split(b"$")
        return index_values, gps_raw_stamp_list

    def get_stamps(self, nims_string):
        """
        get a list of valid GPS strings and match synchronous GPRMC with GPGGA
        stamps if possible.
        
        :param str nims_string: raw GPS string output by NIMS
        """
        ### read in GPS strings into a list to be parsed later
        index_list, gps_raw_stamp_list = self._get_gps_string_list(nims_string)

        gprmc_list = []
        gpgga_list = []
        ### note we are skipping the first entry, it tends to be not
        ### complete anyway
        for ii, index, raw_stamp in zip(
            range(len(index_list)), index_list, gps_raw_stamp_list[1:]
        ):
            gps_obj = GPS(raw_stamp, index)
            if gps_obj.valid:
                if gps_obj.gps_type == "GPRMC":
                    gprmc_list.append(gps_obj)
                elif gps_obj.gps_type == "GPGGA":
                    gpgga_list.append(gps_obj)
            else:
                self.logger.debug(
                    f"GPS Error: file index {index}, stamp number {ii}"
                )
                max_len = min([len(raw_stamp), 15])
                self.logger.debug(f"GPS Raw Stamp: {raw_stamp[0:max_len]}")

        return self._gps_match_gprmc_gpgga_strings(gprmc_list, gpgga_list)

    def _gps_match_gprmc_gpgga_strings(self, gprmc_list, gpgga_list):
        """
        match GPRMC and GPGGA strings together into a list
        
        [[GPRMC, GPGGA], ...]
        
        :param list gprmc_list: list of GPS objects for the GPRMC stamps
        :param list gpgga_list: list of GPS objects for the GPGGA stamps
        
        :returns: list of matched GPRMC and GPGGA stamps 
        
        """
        ### match up the GPRMC and GPGGA together
        gps_match_list = []
        for gprmc in gprmc_list:
            find = False
            for ii, gpgga in enumerate(gpgga_list):
                if gprmc.time_stamp.time() == gpgga.time_stamp.time():
                    gps_match_list.append([gprmc, gpgga])
                    find = True
                    del gpgga_list[ii]
                    break
            if not find:
                gps_match_list.append([gprmc])

        return gps_match_list

    def _get_gps_stamp_indices_from_status(self, status_array):
        """
        get the index location of the stamps from the status array assuming 
        that 0 indicates GPS lock.
        
        :param :class:`np.ndarray` status_array: an array of status values from data blocks
        
        :returns: array of index values where GPS lock was acquired ignoring
                  sequential locks.   
        """

        index_values = np.where(status_array == 0)[0]
        status_index = np.zeros_like(index_values)
        for ii in range(index_values.size):
            if index_values[ii] - index_values[ii - 1] == 1:
                continue
            else:
                status_index[ii] = index_values[ii]
        status_index = status_index[np.nonzero(status_index)]

        return status_index

    def match_status_with_gps_stamps(self, status_array, gps_list):
        """
        Match the index values from the status array with the index values of 
        the GPS stamps.  There appears to be a bit of wiggle room between when the
        lock is recorded and the stamp was actually recorded.  This is typically 1 
        second and sometimes 2.  
        
        :param array status_array: array of status values from each data block
        :param list gps_list: list of valid GPS stamps [[GPRMC, GPGGA], ...]
        
        .. note:: I think there is a 2 second gap between the lock and the 
                  first stamp character.
        """

        stamp_indices = self._get_gps_stamp_indices_from_status(status_array)
        gps_stamps = []
        for index in stamp_indices:
            stamp_find = False
            for ii, stamps in enumerate(gps_list):
                index_diff = stamps[0].index - index
                ### check the index value, should be 2 or 74, if it is off by
                ### a value left or right apply a correction.
                if index_diff == 1 or index_diff == 73:
                    index += 1
                    stamps[0].index += 1
                elif index_diff == 2 or index_diff == 74:
                    index = index
                elif index_diff == 3 or index_diff == 75:
                    index -= 1
                    stamps[0].index -= 1
                elif index_diff == 4 or index_diff == 76:
                    index -= 2
                    stamps[0].index -= 2
                if stamps[0].gps_type in ["GPRMC", "gprmc"]:
                    if index_diff in [1, 2, 3, 4]:
                        gps_stamps.append((index, stamps))
                        stamp_find = True
                        del gps_list[ii]
                        break
                elif stamps[0].gps_type in ["GPGGA", "gpgga"]:
                    if index_diff in [73, 74, 75, 76]:
                        gps_stamps.append((index, stamps))
                        stamp_find = True
                        del gps_list[ii]
                        break
            if not stamp_find:
                self.logger.debug(
                    f"GPS Error: No good GPS stamp at {index} seconds"
                )

        return gps_stamps

    def find_sequence(self, data_array, block_sequence=None):
        """
        find a sequence in a given array
        
        :param array data_array: array of the data with shape [n, m]
                                 where n is the number of seconds recorded
                                 m is the block length for a given sampling
                                 rate.
        :param list block_sequence: sequence pattern to locate
                                    *default* is [1, 131] the start of a 
                                    data block.
                                    
        :returns: array of index locations where the sequence is found.
        """
        if block_sequence is not None:
            self.block_sequence = block_sequence

        # want to find the index there the test data is equal to the test sequence
        t = np.vstack(
            [
                np.roll(data_array, shift)
                for shift in -np.arange(len(self.block_sequence))
            ]
        ).T
        return np.where(np.all(t == self.block_sequence, axis=1))[0]

    def unwrap_sequence(self, sequence):
        """
        unwrap the sequence to be sequential numbers instead of modulated by
        256.  sets the first number to 0
        
        :param list sequence: sequence of bytes numbers
        :return: unwrapped number of counts
        
        """
        count = 0
        unwrapped = np.zeros_like(sequence)
        for ii, seq in enumerate(sequence):
            unwrapped[ii] = seq + count * 256
            if seq == 255:
                count += 1

        unwrapped -= unwrapped[0]

        return unwrapped

    def _locate_duplicate_blocks(self, sequence):
        """
        locate the sequence number where the duplicates exist
        
        :param list sequence: sequence to match duplicate numbers.
        :returns: list of duplicate index values.
        """

        duplicates = np.where(np.abs(np.diff(sequence)) == 0)[0]
        if len(duplicates) == 0:
            return None
        duplicate_list = []
        for dup in duplicates:
            dup_dict = {}
            dup_dict["sequence_index"] = dup
            dup_dict["ts_index_0"] = dup * self.sample_rate
            dup_dict["ts_index_1"] = dup * self.sample_rate + self.sample_rate
            dup_dict["ts_index_2"] = (dup + 1) * self.sample_rate
            dup_dict["ts_index_3"] = (
                dup + 1
            ) * self.sample_rate + self.sample_rate
            duplicate_list.append(dup_dict)
        return duplicate_list

    def _check_duplicate_blocks(self, block_01, block_02, info_01, info_02):
        """
        make sure the blocks are truly duplicates
        
        :param np.array block_01: block of data to compare
        :param np.array block_02: block of data to compare
        :param np.array info_01: information array from info_array[sequence_index]
        :param np.array info_02: information array from info_array[sequence_index]
        
        :returns: boolean if the blocks and information match
        
        """
        if np.array_equal(block_01, block_02):
            if np.array_equal(info_01, info_02):
                return True
            else:
                return False
        else:
            return False

    def remove_duplicates(self, info_array, data_array):
        """
        remove duplicate blocks, removing the first duplicate as suggested by
        Paul and Anna. Checks to make sure that the mag data are identical for 
        the duplicate blocks.  Removes the blocks from the information and
        data arrays and returns the reduced arrays.  This should sync up the
        timing of GPS stamps and index values.
        
        :param np.array info_array: structured array of block information
        :param np.array data_array: structured array of the data
        
        :returns: reduced information array
        :returns: reduced data array
        :returns: index of duplicates in raw data
        
        """
        ### locate
        duplicate_test_list = self._locate_duplicate_blocks(
            self.info_array["sequence"]
        )
        if duplicate_test_list is None:
            return info_array, data_array, None

        duplicate_list = []
        for d in duplicate_test_list:
            if self._check_duplicate_blocks(
                data_array[d["ts_index_0"] : d["ts_index_1"]],
                data_array[d["ts_index_2"] : d["ts_index_3"]],
                info_array[d["sequence_index"]],
                info_array[d["sequence_index"] + 1],
            ):
                duplicate_list.append(d)

        self.logger.debug(f"Deleting {len(duplicate_list)} duplicate blocks")
        ### get the index of the blocks to be removed, namely the 1st duplicate
        ### block
        remove_sequence_index = [d["sequence_index"] for d in duplicate_list]
        remove_data_index = np.array(
            [
                np.arange(d["ts_index_0"], d["ts_index_1"], 1)
                for d in duplicate_list
            ]
        ).flatten()
        ### remove the data
        return_info_array = np.delete(info_array, remove_sequence_index)
        return_data_array = np.delete(data_array, remove_data_index)

        ### set sequence to be monotonic
        return_info_array["sequence"][:] = np.arange(
            return_info_array.shape[0]
        )

        return return_info_array, return_data_array, duplicate_list

    def read_nims(self, fn=None):
        """
        Read NIMS DATA.BIN file.
        
        1. Read in the header information and stores those as attributes
           with the same names as in the header file.
        
        2. Locate the beginning of the data blocks by looking for the 
           first [1, 131, ...] combo.  Anything before that is cut out.
        
        3. Make sure the data is a multiple of the block length, if the
           data is longer the extra bits are cut off.
        
        4. Read in the GPS data (3rd byte of each block) as characters.
           Parses those into valid GPS stamps with appropriate index locations
           of where the '$' was found.
          
        5. Read in the data as unsigned 8-bit integers and reshape the array
           into [N, data_block_length].  Parse this array into the status
           information and the data.
           
        6. Remove duplicate blocks, by removing the first of the duplicates
           as suggested by Anna and Paul.  
           
        7. Match the GPS locks from the status with valid GPS stamps.
                
        8. Check to make sure that there is the correct number of seconds
           between the first and last GPS stamp.  The extra seconds are cut
           off from the end of the time series.  Not sure if this is the
           best way to accommodate gaps in the data.
        
        .. note:: The data and information array returned have the duplicates
                  removed and the sequence reset to be monotonic.
        
        :param str fn: full path to DATA.BIN file
        
        :Example:
            
        >>> from mth5.io import nims
        >>> n = nims.NIMS(r"/home/mt_data/nims/mt001.bin")
        
        
        """

        if fn is not None:
            self.fn = fn

        st = datetime.datetime.now()
        ### read in header information and get the location of end of header
        self.read_header(self.fn)

        ### load in the entire file, its not too big, start from the
        ### end of the header information.
        with open(self.fn, "rb") as fid:
            fid.seek(self.data_start_seek)
            self._raw_string = fid.read()

        ### read in full string as unsigned integers
        data = np.frombuffer(self._raw_string, dtype=np.uint8)

        ### need to make sure that the data starts with a full block
        find_first = self.find_sequence(data[0 : self.block_size * 5])[0]
        data = data[find_first:]

        ### get GPS stamps from the binary string first
        self.gps_list = self.get_stamps(self._raw_string[find_first:])

        ### check the size of the data, should have an equal amount of blocks
        if (data.size % self.block_size) != 0:
            self.logger.warning(
                f"odd number of bytes {data.size}, not even blocks "
                + "cutting down the data by {0} bits".format(
                    data.size % self.block_size
                )
            )
            end_data = data.size - (data.size % self.block_size)
            data = data[0:end_data]

        # resized the data into an even amount of blocks
        data = data.reshape(
            (int(data.size / self.block_size), self.block_size)
        )

        ### need to parse the data
        ### first get the status information
        self.info_array = np.zeros(
            data.shape[0],
            dtype=[
                ("soh", np.int),
                ("block_len", np.int),
                ("status", np.int),
                ("gps", np.int),
                ("sequence", np.int),
                ("box_temp", np.float),
                ("head_temp", np.float),
                ("logic", np.int),
                ("end", np.int),
            ],
        )

        for key, index in self._block_dict.items():
            if "temp" in key:
                # compute temperature
                t_value = data[:, index[0]] * 256 + data[:, index[1]]

                # something to do with the bits where you have to subtract
                t_value[np.where(t_value > 32768)] -= 65536
                value = (t_value - self.t_offset) / self.t_conversion_factor
            else:
                value = data[:, index]
            self.info_array[key][:] = value

        ### unwrap sequence
        self.info_array["sequence"] = self.unwrap_sequence(
            self.info_array["sequence"]
        )

        ### get data
        data_array = np.zeros(
            data.shape[0] * self.sample_rate,
            dtype=[
                ("hx", np.float),
                ("hy", np.float),
                ("hz", np.float),
                ("ex", np.float),
                ("ey", np.float),
            ],
        )

        ### fill the data
        for cc, comp in enumerate(["hx", "hy", "hz", "ex", "ey"]):
            channel_arr = np.zeros((data.shape[0], 8), dtype=np.float)
            for kk in range(self.sample_rate):
                index = self.indices[kk, cc]
                value = (data[:, index] * 256 + data[:, index + 1]) * np.array(
                    [256]
                ) + data[:, index + 2]
                value[np.where(value > self._int_max)] -= self._int_factor
                channel_arr[:, kk] = value
            data_array[comp][:] = channel_arr.flatten()

        ### clean things up
        ### I guess that the E channels are opposite phase?
        for comp in ["ex", "ey"]:
            data_array[comp] *= -1

        ### remove duplicates
        (
            self.info_array,
            data_array,
            self.duplicate_list,
        ) = self.remove_duplicates(self.info_array, data_array)
        ### get GPS stamps with index values
        self.stamps = self.match_status_with_gps_stamps(
            self.info_array["status"], self.gps_list
        )
        ### align data checking for timing gaps
        self.ts = self.align_data(data_array, self.stamps)

        et = datetime.datetime.now()
        read_time = (et - st).total_seconds()
        self.logger.info(f"Reading took {read_time:.2f} seconds")

    def _get_first_gps_stamp(self, stamps):
        """
        get the first GPRMC stamp
        """
        for stamp in stamps:
            if stamp[1][0].gps_type in ["gprmc", "GPRMC"]:
                return stamp
        return None

    def _get_last_gps_stamp(self, stamps):
        """
        get the last gprmc stamp
        """
        for stamp in stamps[::-1]:
            if stamp[1][0].gps_type in ["gprmc", "GPRMC"]:
                return stamp
        return None

    def _locate_timing_gaps(self, stamps):
        """
        locate timing gaps in the data by comparing the stamp index with the 
        GPS time stamp.  The number of points and seconds should be the same
        
        :param list stamps: list of GPS stamps [[status_index, [GPRMC, GPGGA]]]
        
        :returns: list of gap index values
        """
        stamp_01 = self._get_first_gps_stamp(stamps)[1][0]
        # current_gap = 0
        current_stamp = stamp_01
        gap_beginning = []
        total_gap = 0
        for ii, stamp in enumerate(stamps[1:], 1):
            stamp = stamp[1][0]
            # can only compare those with a date and time.
            if stamp.gps_type == "GPGGA":
                continue

            time_diff = (
                stamp.time_stamp - current_stamp.time_stamp
            ).total_seconds()
            index_diff = stamp.index - current_stamp.index

            time_gap = index_diff - time_diff
            if time_gap == 0:
                continue
            elif time_gap > 0:
                total_gap += time_gap
                current_stamp = stamp
                gap_beginning.append(stamp.index)
                self.logger.debug(
                    "GPS tamp at {0} is off from previous time by {1} seconds".format(
                        stamp.time_stamp.isoformat(), time_gap,
                    )
                )

        self.logger.warning(f"Timing is off by {total_gap} seconds")
        return gap_beginning

    def check_timing(self, stamps):
        """
        make sure that there are the correct number of seconds in between
        the first and last GPS GPRMC stamps
        
        :param list stamps: list of GPS stamps [[status_index, [GPRMC, GPGGA]]]
        
        :returns: [ True | False ] if data is valid or not.
        :returns: gap index locations
        
        .. note:: currently it is assumed that if a data gap occurs the data can be 
                  squeezed to remove them.  Probably a more elegant way of doing it.
        """
        gaps = None
        first_stamp = self._get_first_gps_stamp(stamps)[1][0]
        last_stamp = self._get_last_gps_stamp(stamps)[1][0]

        time_diff = last_stamp.time_stamp - first_stamp.time_stamp
        index_diff = last_stamp.index - first_stamp.index

        difference = index_diff - time_diff.total_seconds()
        if difference != 0:
            gaps = self._locate_timing_gaps(stamps)
            return False, gaps, difference

        return True, gaps, difference

    def align_data(self, data_array, stamps):
        """
        Need to match up the first good GPS stamp with the data
        
        Do this by using the first GPS stamp and assuming that the time from
        the first time stamp to the start is the index value.
        
        put the data into a pandas data frame that is indexed by time
        
        :param array data_array: structure array with columns for each 
                                 component [hx, hy, hz, ex, ey]
        :param list stamps: list of GPS stamps [[status_index, [GPRMC, GPGGA]]]
        
        :returns: pandas DataFrame with colums of components and indexed by 
                  time initialized by the start time.
        
        .. note:: Data gaps are squeezed cause not sure what a gap actually means.
        """
        ### check timing first to make sure there is no drift
        timing_valid, self.gaps, time_difference = self.check_timing(stamps)

        ### need to trim off the excess number of points that are present because of
        ### data gaps.  This will be the time difference times the sample rate
        if time_difference > 0:
            remove_points = int(time_difference * self.sample_rate)
            data_array = data_array[0:-remove_points]
            self.logger.info(
                f"Trimmed {remove_points} points off the end of the time "
                "series because of timing gaps"
            )

        ### first GPS stamp within the data is at a given index that is
        ### assumed to be the number of seconds from the start of the run.
        ### therefore make the start time the first GPS stamp time minus
        ### the index value for that stamp.
        ### need to be sure that the first GPS stamp has a date, need GPRMC
        first_stamp = self._get_first_gps_stamp(stamps)
        first_index = first_stamp[0]
        start_time = first_stamp[1][0].time_stamp - datetime.timedelta(
            seconds=int(first_index)
        )

        dt_index = self.make_dt_index(
            start_time.isoformat(),
            self.sample_rate,
            n_samples=data_array.shape[0],
        )

        return pd.DataFrame(data_array, index=dt_index)

    def calibrate_data(self, ts):
        """
        Apply calibrations to data
        
        .. note:: this needs work, would not use this now.
        """

        ts[["hx", "hy", "hz"]] *= self.h_conversion_factor
        ts[["ex", "ey"]] *= self.e_conversion_factor
        ts["ex"] /= self.ex_length / 1000.0
        ts["ey"] /= self.ey_length / 1000.0

        return ts

    def make_dt_index(
        self, start_time, sample_rate, stop_time=None, n_samples=None
    ):
        """
        make time index array

        .. note:: date-time format should be YYYY-M-DDThh:mm:ss.ms UTC

        :param start_time: start time
        :type start_time: string

        :param end_time: end time
        :type end_time: string

        :param sample_rate: sample_rate in samples/second
        :type sample_rate: float
        """

        # set the index to be UTC time
        dt_freq = "{0:.0f}N".format(1.0 / (sample_rate) * 1e9)
        if stop_time is not None:
            dt_index = pd.date_range(
                start=start_time,
                end=stop_time,
                freq=dt_freq,
                closed="left",
                tz="UTC",
            )
        elif n_samples is not None:
            dt_index = pd.date_range(
                start=start_time, periods=n_samples, freq=dt_freq, tz="UTC"
            )
        else:
            raise ValueError("Need to input either stop_time or n_samples")

        return dt_index


class Response(object):
    """
    class for instrument response functions.
    
    """

    def __init__(self, system_id=None, **kwargs):
        self.system_id = system_id
        self.hardware = "PC"
        self.instrument_type = "backbone"
        self.sample_rate = 8
        self.e_conversion_factor = 2.44141221047903e-06
        self.h_conversion_factor = 0.01

        self.time_delays_dict = {
            "hp200": {
                "hx": -0.0055,
                "hy": -0.0145,
                "hz": -0.0235,
                "ex": 0.1525,
                "ey": 0.0275,
            },
            1: {
                "hx": -0.1920,
                "hy": -0.2010,
                "hz": -0.2100,
                "ex": -0.2850,
                "ey": -0.2850,
            },
            8: {
                "hx": 0.2455,
                "hy": 0.2365,
                "hz": 0.2275,
                "ex": 0.1525,
                "ey": 0.1525,
            },
        }
        self.mag_low_pass = {
            "name": "3 pole butterworth",
            "type": "poles-zeros",
            "parameters": {
                "zeros": [0, 3, 1984.31],
                "poles": [
                    complex(-6.28319, 10.8825),
                    complex(-6.28319, 10.8825),
                    complex(-12.5664, 0),
                ],
            },
        }
        self.electric_low_pass = {
            "name": "5 pole butterworth",
            "type": "poles-zeros",
            "parameters": {
                "zeros": [0, 5, 313384],
                "poles": [
                    complex(-3.88301, 11.9519),
                    complex(-3.88301, -11.9519),
                    complex(-10.1662, 7.38651),
                    complex(-10.1662, -7.38651),
                    complex(-12.5664, 0.0),
                ],
            },
        }
        self.electric_high_pass_pc = {
            "name": "1 pole butterworth",
            "type": "poles-zeros",
            "parameters": {
                "zeros": [1, 1, 1],
                "poles": [complex(0.0, 0.0), complex(-3.333333e-05, 0.0)],
            },
            "t0": 2 * np.pi * 30000,
        }
        self.electric_high_pass_hp = {
            "name": "1 pole butterworth",
            "type": "poles-zeros",
            "parameters": {
                "zeros": [1, 1, 1],
                "poles": [complex(0.0, 0.0), complex(-1.66667e-04, 0.0)],
            },
            "t0": 2 * np.pi * 6000,
        }

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_electric_high_pass(self, hardware="pc"):
        """
        get the electric high pass filter based on the hardware
        """

        self.hardware = hardware
        if "pc" in hardware.lower():
            return self.electric_high_pass_pc
        elif "hp" in hardware.lower():
            return self.electric_high_pass_hp
        else:
            raise ResponseError(
                "Hardware value {0} not understood".format(self.hardware)
            )

    def _get_dt_filter(self, channel, sample_rate):
        """
        get the DT filter based on channel ans sampling rate
        """
        dt_filter = {
            "type": "dt",
            "name": "time_offset",
            "parameters": {
                "offset": self.time_delays_dict[sample_rate][channel]
            },
        }
        return dt_filter

    def _get_mag_filter(self, channel):
        """
        get mag filter, seems to be the same no matter what
        """
        filter_list = [self.mag_low_pass]
        filter_list.append(self._get_dt_filter(channel, self.sample_rate))

        return_dict = {
            "channel_id": channel,
            "gain": 1,
            "conversion_factor": self.h_conversion_factor,
            "units": "nT",
            "filters": filter_list,
        }
        return return_dict

    def _get_electric_filter(self, channel):
        """
        Get electric filter
        """
        filter_list = []
        if self.instrument_type in ["backbone"]:
            filter_list.append(self.get_electric_high_pass(self.hardware))
        filter_list.append(self.electric_low_pass)
        filter_list.append(self._get_dt_filter(channel, self.sample_rate))

        return_dict = {
            "channel_id": channel,
            "gain": 1,
            "conversion_factor": self.e_conversion_factor,
            "units": "nT",
            "filters": filter_list,
        }
        return return_dict

    @property
    def hx_filter(self):
        """HX filter"""

        return self._get_mag_filter("hx")

    @property
    def hy_filter(self):
        """HY Filter"""
        return self._get_mag_filter("hy")

    @property
    def hz_filter(self):
        return self._get_mag_filter("hz")

    @property
    def ex_filter(self):
        return self._get_electric_filter("ex")

    @property
    def ey_filter(self):
        return self._get_electric_filter("ey")


# =============================================================================
# convenience read
# =============================================================================
def read_nims(fn):
    """
    
    :param fn: DESCRIPTION
    :type fn: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    nims_obj = NIMS(fn)
    nims_obj.read_nims()

    return nims_obj.to_runts()
