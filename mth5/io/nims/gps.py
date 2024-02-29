# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:43:56 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import dateutil
from loguru import logger


# =============================================================================
class GPSError(Exception):
    pass


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

        self.logger = logger

        self.gps_string = gps_string
        self.index = index
        self._type = None
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
        """string representation"""
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

        if isinstance(gps_string, bytes):
            for replace_str in [b"\xd9", b"\xc7", b"\xcc"]:
                gps_string = gps_string.replace(replace_str, b"")
            ### sometimes the end is set with a zero for some reason
            gps_string = gps_string.replace(b"\x00", b"*")

            if gps_string.find(b"*") < 0:
                logger.debug(f"GPSError: No end to stamp {gps_string}")
                return None
            else:
                try:
                    gps_string = gps_string[0 : gps_string.find(b"*")].decode()
                    return gps_string
                except UnicodeDecodeError:
                    logger.debug(f"GPSError: stamp not correct format, {gps_string}")
                    return None
        elif isinstance(gps_string, str):
            if "*" not in gps_string:
                logger.debug(f"GPSError: No end to stamp {gps_string}")
                return None
            return gps_string[0 : gps_string.find("*")]
        else:
            raise TypeError(
                f"input must be a string or bytes object, not {type(gps_string)}"
            )

    def _split_gps_string(self, gps_string, delimiter=","):
        """Split a GPS string by ',' and validate it"""

        gps_string = self.validate_gps_string(gps_string)
        if gps_string is None:
            self.valid = False
            return []
        return gps_string.strip().split(",")

    def parse_gps_string(self, gps_string):
        """
        Parse a raw gps string from the NIMS and set appropriate attributes.
        GPS string will first be validated, then parsed.

        :param string gps_string: raw GPS string to be parsed
        """

        gps_list = self._split_gps_string(gps_string)
        if gps_list == []:
            self.logger.debug(f"GPS string is invalid, {gps_string}")
            return
        if len(gps_list) > 1:
            if len(gps_list[1]) > 6:
                self.logger.debug(
                    "GPS time and lat missing a comma adding one, check time"
                )
                gps_list = (
                    gps_list[0:1] + [gps_list[1][0:6], gps_list[1][6:]] + gps_list[2:]
                )
        ### validate the gps list to make sure it is usable
        gps_list, error_list = self.validate_gps_list(gps_list)
        if len(error_list) > 0:
            for error in error_list:
                logger.debug("GPSError: " + error)
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
            gps_list[self.type_dict[g_type]["latitude"]] = self._validate_latitude(
                gps_list[self.type_dict[g_type]["latitude"]],
                gps_list[self.type_dict[g_type]["latitude_hemisphere"]],
            )
        except GPSError as error:
            error_list.append(error.args[0])
            gps_list[self.type_dict[g_type]["latitude"]] = None
        try:
            gps_list[self.type_dict[g_type]["longitude"]] = self._validate_longitude(
                gps_list[self.type_dict[g_type]["longitude"]],
                gps_list[self.type_dict[g_type]["longitude_hemisphere"]],
            )
        except GPSError as error:
            error_list.append(error.args[0])
            gps_list[self.type_dict[g_type]["longitude"]] = None
        if g_type == "gprmc":
            try:
                gps_list[self.type_dict["gprmc"]["date"]] = self._validate_date(
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
                f"Expect GPGGA or GPRMC, got {gps_type.upper()}"
            )
        return gps_list

    def _validate_list_length(self, gps_list):
        """validate gps list length based on type of string"""

        gps_list_type = gps_list[0].lower()
        expected_len = self.type_dict[gps_list_type]["length"]
        if len(gps_list) not in expected_len:
            raise GPSError(
                f"GPS string not correct length for {gps_list_type.upper()}.  "
                f"Expected {expected_len}, got {len(gps_list)} "
                f"{','.join(gps_list)}"
            )

    def _validate_time(self, time_str):
        """validate time string, should be 6 characters long and an int"""
        if len(time_str) != 6:
            raise GPSError(
                f"Length of time string {time_str} not correct.  "
                f"Expected 6 got {len(time_str)}. string = {time_str}"
            )
        try:
            int(time_str)
        except ValueError:
            raise GPSError(f"Could not convert time string {time_str}")
        return time_str

    def _validate_date(self, date_str):
        """validate date string, should be 6 characters long and an int"""
        if len(date_str) != 6:
            raise GPSError(
                f"Length of date string not correct {date_str}.  "
                f"Expected 6 got {len(date_str)}. string = {date_str}"
            )
        try:
            int(date_str)
        except ValueError:
            raise GPSError(f"Could not convert date string {date_str}")
        return date_str

    def _validate_latitude(self, latitude_str, hemisphere_str):
        """validate latitude, should have hemisphere string with it"""

        if len(latitude_str) < 8:
            raise GPSError(
                f"Latitude string should be larger than 7 characters.  "
                f"Got {len(latitude_str)}. string = {latitude_str}"
            )
        if len(hemisphere_str) != 1:
            raise GPSError(
                "Latitude hemisphere should be 1 character.  "
                f"Got {len(hemisphere_str)}. string = {hemisphere_str}"
            )
        if hemisphere_str.lower() not in ["n", "s"]:
            raise GPSError(
                f"Latitude hemisphere {hemisphere_str.upper()} not understood"
            )
        try:
            float(latitude_str)
        except ValueError:
            raise GPSError(f"Could not convert latitude string {latitude_str}")
        return latitude_str

    def _validate_longitude(self, longitude_str, hemisphere_str):
        """validate longitude, should have hemisphere string with it"""

        if len(longitude_str) < 8:
            raise GPSError(
                "Longitude string should be larger than 7 characters.  "
                f"Got {len(longitude_str)}. string = {longitude_str}"
            )
        if len(hemisphere_str) != 1:
            raise GPSError(
                "Longitude hemisphere should be 1 character.  "
                f"Got {len(hemisphere_str)}. string = {hemisphere_str}"
            )
        if hemisphere_str.lower() not in ["e", "w"]:
            raise GPSError(
                f"Longitude hemisphere {hemisphere_str.upper()} not understood"
            )
        try:
            float(longitude_str)
        except ValueError:
            raise GPSError(f"Could not convert longitude string {longitude_str}")
        return longitude_str

    def _validate_elevation(self, elevation_str):
        """validate elevation, check for converstion to float"""
        elevation_str = elevation_str.lower().replace("m", "")
        if elevation_str == "":
            elevation_str = "0"
        try:
            elevation_str = f"{float(elevation_str)}"
        except ValueError:
            raise GPSError(f"Elevation could not be converted {elevation_str}")
        return elevation_str

    @property
    def latitude(self):
        """
        Latitude in decimal degrees, WGS84
        """
        if self._latitude is not None and self._latitude_hemisphere is not None:
            index = len(self._latitude) - 7
            lat = float(self._latitude[0:index]) + float(self._latitude[index:]) / 60
            if "s" in self._latitude_hemisphere.lower():
                lat *= -1
            return lat
        return 0.0

    @property
    def longitude(self):
        """
        Latitude in decimal degrees, WGS84
        """
        if self._longitude is not None and self._longitude_hemisphere is not None:
            index = len(self._longitude) - 7
            lon = float(self._longitude[0:index]) + float(self._longitude[index:]) / 60
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
                    f"not complete {self.gps_string}"
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
