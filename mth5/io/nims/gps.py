# -*- coding: utf-8 -*-
"""
NIMS GPS data parser for magnetotelluric surveys.

This module provides functionality to parse GPS stamps from NIMS (North Island
Magnetotelluric Survey) data files. It handles both GPRMC and GPGGA GPS message
formats, extracting location, time, and other GPS-related information.

Classes
-------
GPSError : Exception
    Custom exception for GPS parsing errors.
GPS : object
    Main class for parsing and validating GPS stamp data.

Notes
-----
The GPS parser handles two main GPS message types:
- GPRMC: Provides full date/time information and magnetic declination
- GPGGA: Provides elevation data and fix quality information

Binary data contamination is automatically cleaned during parsing.

Examples
--------
>>> from mth5.io.nims.gps import GPS
>>> gps_string = "GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*"
>>> gps = GPS(gps_string)
>>> print(f"Latitude: {gps.latitude}, Longitude: {gps.longitude}")

Author
------
jpeacock

Created
-------
Thu Sep  1 11:43:56 2022
"""
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import datetime

import dateutil
from loguru import logger


# =============================================================================
class GPSError(Exception):
    """
    Custom exception for GPS parsing and validation errors.

    Raised when GPS string parsing fails or when GPS data validation
    encounters invalid values.
    """


class GPS:
    """
    Parser for GPS stamps from NIMS magnetotelluric data.

    Handles parsing and validation of GPS strings from NIMS data files.
    Supports both GPRMC and GPGGA message formats, automatically detecting
    the type and extracting relevant geographic and temporal information.

    Parameters
    ----------
    gps_string : str or bytes
        Raw GPS string to be parsed. Can contain binary contamination
        which will be automatically cleaned.
    index : int, default 0
        Index or sequence number for this GPS record.

    Attributes
    ----------
    gps_string : str
        The original GPS string provided for parsing.
    index : int
        Index or sequence number for this GPS record.
    valid : bool
        Whether the GPS string was successfully parsed and validated.
    elevation_units : str
        Units for elevation measurements, typically "meters".
    logger : loguru.Logger
        Logger instance for debugging and error reporting.

    Notes
    -----
    GPS message format differences:

    **GPRMC (Recommended Minimum Course)**
        Contains: date, time, coordinates, speed, course, magnetic declination
        Date: Full date information (year, month, day)

    **GPGGA (Global Positioning System Fix Data)**
        Contains: time, coordinates, fix quality, elevation
        Date: Defaults to 1980-01-01 for time estimation only

    The parser automatically handles:
    - Binary contamination in GPS strings
    - Missing comma delimiters
    - GPS type auto-detection and correction
    - Coordinate conversion from degrees-minutes to decimal degrees

    Examples
    --------
    Parse a GPRMC string:

    >>> gps_string = "GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*"
    >>> gps = GPS(gps_string)
    >>> print(f"Position: {gps.latitude:.5f}, {gps.longitude:.5f}")
    Position: 34.72683, -115.73501

    Parse a GPGGA string:

    >>> gps_string = "GPGGA,183511,3443.6098,N,11544.1007,W,1,04,2.6,937.2,M,-28.1,M,*"
    >>> gps = GPS(gps_string)
    >>> print(f"Elevation: {gps.elevation} {gps.elevation_units}")
    Elevation: 937.2 meters

    Handle invalid GPS data:

    >>> gps = GPS("invalid_string")
    >>> print(f"Valid: {gps.valid}")
    Valid: False
    """

    def __init__(self, gps_string: str | bytes, index: int = 0) -> None:
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

    def __str__(self) -> str:
        """
        String representation of GPS object.

        Returns
        -------
        str
            Formatted string containing GPS type, coordinates, elevation,
            declination, and other key properties.

        Examples
        --------
        >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
        >>> print(gps)
        type = GPRMC
        index = 0
        fix = A
        time_stamp =  2019-09-26T18:35:11
        latitude = 34.72683
        longitude = -115.73501166666667
        elevation = 0.0
        declination = 13.1
        """
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

    def __repr__(self) -> str:
        """Return string representation of GPS object."""
        return self.__str__()

    def validate_gps_string(self, gps_string: str | bytes) -> str | None:
        """
        Validate and clean GPS string.

        Removes binary contamination, finds string terminator, and validates
        format. Handles both string and bytes input.

        Parameters
        ----------
        gps_string : str or bytes
            Raw GPS string to validate. May contain binary contamination
            that will be automatically removed.

        Returns
        -------
        str or None
            Cleaned GPS string with terminator removed, or None if validation
            fails due to missing terminator or decode errors.

        Raises
        ------
        TypeError
            If input is not string or bytes.

        Notes
        -----
        Binary contamination bytes that are automatically removed:
        - ``\\xd9``, ``\\xc7``, ``\\xcc``
        - ``\\x00`` (null byte, replaced with '*' terminator)

        The GPS string must end with '*' character to be considered valid.

        Examples
        --------
        Clean a contaminated binary GPS string:

        >>> gps = GPS("")
        >>> contaminated = b"GPRMC,183511,A\\xd9,3443.6098,N*"
        >>> clean = gps.validate_gps_string(contaminated)
        >>> print(clean)
        GPRMC,183511,A,3443.6098,N

        Handle missing terminator:

        >>> invalid = "GPRMC,183511,A,3443.6098,N"  # No '*'
        >>> result = gps.validate_gps_string(invalid)
        >>> print(result)
        None
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

    def _split_gps_string(
        self, gps_string: str | bytes, delimiter: str = ","
    ) -> list[str]:
        """
        Split GPS string into components after validation.

        Parameters
        ----------
        gps_string : str or bytes
            GPS string to split.
        delimiter : str, default ","
            Character to split on (typically comma).

        Returns
        -------
        list of str
            GPS string components, or empty list if validation fails.

        Notes
        -----
        The delimiter parameter is provided for flexibility but validation
        always occurs first, which may affect the final splitting behavior.
        """
        gps_string = self.validate_gps_string(gps_string)
        if gps_string is None:
            self.valid = False
            return []
        return gps_string.strip().split(",")

    def parse_gps_string(self, gps_string: str | bytes) -> None:
        """
        Parse GPS string and populate object attributes.

        Main parsing method that validates the GPS string, identifies the
        message type (GPRMC/GPGGA), and extracts all relevant information
        into object attributes.

        Parameters
        ----------
        gps_string : str or bytes
            Raw GPS string from NIMS data file.

        Notes
        -----
        This method performs the following operations:
        1. Splits and validates the GPS string
        2. Handles missing comma delimiter between time and coordinates
        3. Validates each GPS field according to message type
        4. Sets object attributes based on parsed values
        5. Sets ``valid`` flag based on parsing success

        If any validation errors occur, they are logged but parsing continues
        with ``None`` values for invalid fields.

        The method automatically detects GPS message type and applies
        appropriate field validation rules.

        Examples
        --------
        Parse a valid GPS string:

        >>> gps = GPS("")
        >>> gps.parse_gps_string("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
        >>> print(f"Valid: {gps.valid}, Type: {gps.gps_type}")
        Valid: True, Type: GPRMC

        Handle invalid GPS string:

        >>> gps.parse_gps_string("invalid_gps_data")
        >>> print(f"Valid: {gps.valid}")
        Valid: False
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

    def validate_gps_list(
        self, gps_list: list[str]
    ) -> tuple[list[str] | None, list[str]]:
        """
        Validate GPS field list and check format compliance.

        Performs comprehensive validation of GPS message components including
        type checking, length validation, and field-specific validation.

        Parameters
        ----------
        gps_list : list of str
            GPS message components split by delimiter.

        Returns
        -------
        gps_list : list of str or None
            Validated GPS list with corrected values, or None if
            critical validation fails.
        error_list : list of str
            List of validation error messages encountered during processing.

        Notes
        -----
        Validation steps performed:
        1. GPS message type validation and correction
        2. Message length validation based on type
        3. Time format validation (6 digits)
        4. Coordinate validation (latitude/longitude + hemisphere)
        5. Date validation for GPRMC messages
        6. Elevation validation for GPGGA messages

        Non-critical validation errors are collected but don't halt processing.
        Critical errors (type or length) return None and stop validation.

        Examples
        --------
        Validate a correct GPS list:

        >>> gps = GPS("")
        >>> gps_data = ["GPRMC", "183511", "A", "3443.6098", "N", "11544.1007", "W",
        ...             "000.0", "000.0", "260919", "013.1", "E"]
        >>> validated, errors = gps.validate_gps_list(gps_data)
        >>> print(f"Errors: {len(errors)}")
        Errors: 0

        Handle validation errors:

        >>> bad_data = ["INVALID", "time", "fix"]
        >>> validated, errors = gps.validate_gps_list(bad_data)
        >>> print(f"Result: {validated}, Errors: {len(errors)}")
        Result: None, Errors: 1
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

    def _validate_gps_type(self, gps_list: list[str]) -> list[str]:
        """
        Validate and auto-correct GPS message type.

        Parameters
        ----------
        gps_list : list of str
            GPS message components with type as first element.

        Returns
        -------
        list of str
            GPS list with corrected type and possibly extracted time data.

        Raises
        ------
        GPSError
            If GPS type cannot be identified as GPGGA or GPRMC variant.

        Notes
        -----
        Auto-correction rules:
        - "GPG*" patterns → "GPGGA"
        - "GPR*" patterns → "GPRMC"
        - Handles concatenated type+time strings
        - Validates final type is "gpgga" or "gprmc"
        """
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

    def _validate_list_length(self, gps_list: list[str]) -> None:
        """
        Validate GPS message length based on message type.

        Parameters
        ----------
        gps_list : list of str
            GPS message components.

        Raises
        ------
        GPSError
            If message length doesn't match expected length for the GPS type.

        Notes
        -----
        Expected lengths:
        - GPRMC: 12 components
        - GPGGA: 14 or 15 components
        """
        gps_list_type = gps_list[0].lower()
        expected_len = self.type_dict[gps_list_type]["length"]
        if len(gps_list) not in expected_len:
            raise GPSError(
                f"GPS string not correct length for {gps_list_type.upper()}.  "
                f"Expected {expected_len}, got {len(gps_list)} "
                f"{','.join(gps_list)}"
            )

    def _validate_time(self, time_str: str) -> str:
        """
        Validate GPS time string format.

        Parameters
        ----------
        time_str : str
            Time string in HHMMSS format.

        Returns
        -------
        str
            Validated time string.

        Raises
        ------
        GPSError
            If time string is not 6 characters or not numeric.

        Examples
        --------
        >>> gps = GPS("")
        >>> gps._validate_time("183511")
        '183511'
        """
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

    def _validate_date(self, date_str: str) -> str:
        """
        Validate GPS date string format.

        Parameters
        ----------
        date_str : str
            Date string in DDMMYY format.

        Returns
        -------
        str
            Validated date string.

        Raises
        ------
        GPSError
            If date string is not 6 characters or not numeric.

        Examples
        --------
        >>> gps = GPS("")
        >>> gps._validate_date("260919")
        '260919'
        """
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

    def _validate_latitude(self, latitude_str: str, hemisphere_str: str) -> str:
        """
        Validate latitude coordinate and hemisphere.

        Parameters
        ----------
        latitude_str : str
            Latitude in DDMM.MMMM format (degrees and decimal minutes).
        hemisphere_str : str
            Hemisphere indicator, must be 'N' or 'S'.

        Returns
        -------
        str
            Validated latitude string.

        Raises
        ------
        GPSError
            If latitude format is invalid, hemisphere is wrong length/value,
            or coordinate cannot be converted to float.

        Notes
        -----
        Latitude format: DDMM.MMMM where DD=degrees, MM.MMMM=minutes
        Valid hemispheres: 'N' (North), 'S' (South)
        Minimum expected length: 8 characters

        Examples
        --------
        >>> gps = GPS("")
        >>> gps._validate_latitude("3443.6098", "N")
        '3443.6098'
        """
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

    def _validate_longitude(self, longitude_str: str, hemisphere_str: str) -> str:
        """
        Validate longitude coordinate and hemisphere.

        Parameters
        ----------
        longitude_str : str
            Longitude in DDDMM.MMMM format (degrees and decimal minutes).
        hemisphere_str : str
            Hemisphere indicator, must be 'E' or 'W'.

        Returns
        -------
        str
            Validated longitude string.

        Raises
        ------
        GPSError
            If longitude format is invalid, hemisphere is wrong length/value,
            or coordinate cannot be converted to float.

        Notes
        -----
        Longitude format: DDDMM.MMMM where DDD=degrees, MM.MMMM=minutes
        Valid hemispheres: 'E' (East), 'W' (West)
        Minimum expected length: 8 characters

        Examples
        --------
        >>> gps = GPS("")
        >>> gps._validate_longitude("11544.1007", "W")
        '11544.1007'
        """
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

    def _validate_elevation(self, elevation_str: str) -> str:
        """
        Validate elevation value and convert to standard format.

        Parameters
        ----------
        elevation_str : str
            Elevation string, may include 'M' or 'm' unit suffix.

        Returns
        -------
        str
            Validated elevation as string representation of float.

        Raises
        ------
        GPSError
            If elevation cannot be converted to float.

        Notes
        -----
        - Automatically removes 'M' or 'm' unit suffixes
        - Empty string is converted to "0.0"
        - Result is always a string representation of a float

        Examples
        --------
        >>> gps = GPS("")
        >>> gps._validate_elevation("937.2")
        '937.2'
        >>> gps._validate_elevation("937.2M")
        '937.2'
        >>> gps._validate_elevation("")
        '0.0'
        """
        elevation_str = elevation_str.lower().replace("m", "")
        if elevation_str == "":
            elevation_str = "0"
        try:
            elevation_str = f"{float(elevation_str)}"
        except ValueError:
            raise GPSError(f"Elevation could not be converted {elevation_str}")
        return elevation_str

    @property
    def latitude(self) -> float:
        """
        Latitude in decimal degrees (WGS84).

        Returns
        -------
        float
            Latitude in decimal degrees. Negative values indicate
            Southern hemisphere. Returns 0.0 if coordinate data is invalid.

        Notes
        -----
        Converts from GPS format (DDMM.MMMM) to decimal degrees:
        decimal_degrees = degrees + minutes/60

        Southern hemisphere coordinates are automatically converted to negative values.

        Examples
        --------
        >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
        >>> gps.latitude
        34.72683
        """
        if self._latitude is not None and self._latitude_hemisphere is not None:
            index = len(self._latitude) - 7
            lat = float(self._latitude[0:index]) + float(self._latitude[index:]) / 60
            if "s" in self._latitude_hemisphere.lower():
                lat *= -1
            return lat
        return 0.0

    @property
    def longitude(self) -> float:
        """
        Longitude in decimal degrees (WGS84).

        Returns
        -------
        float
            Longitude in decimal degrees. Negative values indicate
            Western hemisphere. Returns 0.0 if coordinate data is invalid.

        Notes
        -----
        Converts from GPS format (DDDMM.MMMM) to decimal degrees:
        decimal_degrees = degrees + minutes/60

        Western hemisphere coordinates are automatically converted to negative values.

        Examples
        --------
        >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
        >>> gps.longitude
        -115.73501166666667
        """
        if self._longitude is not None and self._longitude_hemisphere is not None:
            index = len(self._longitude) - 7
            lon = float(self._longitude[0:index]) + float(self._longitude[index:]) / 60
            if "w" in self._longitude_hemisphere.lower():
                lon *= -1
            return lon
        return 0.0

    @property
    def elevation(self) -> float:
        """
        Elevation above sea level in meters.

        Returns
        -------
        float
            Elevation in meters. Returns 0.0 if elevation data is not
            available or cannot be converted.

        Notes
        -----
        Elevation is typically only available in GPGGA messages.
        GPRMC messages will return 0.0 as they don't contain elevation data.

        Conversion errors are logged but don't raise exceptions.

        Examples
        --------
        >>> gps = GPS("GPGGA,183511,3443.6098,N,11544.1007,W,1,04,2.6,937.2,M,-28.1,M,*")
        >>> gps.elevation
        937.2
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
    def time_stamp(self) -> datetime.datetime | None:
        """
        GPS timestamp as datetime object.

        Returns
        -------
        datetime.datetime or None
            Timestamp parsed from GPS data, or None if time data is invalid.

        Notes
        -----
        For GPRMC messages: Uses full date and time information
        For GPGGA messages: Uses time with default date of 1980-01-01

        Time format: HHMMSS (hours, minutes, seconds)
        Date format: DDMMYY (day, month, 2-digit year)

        Invalid date strings are logged but return None rather than raising exceptions.

        Examples
        --------
        >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
        >>> gps.time_stamp
        datetime.datetime(2019, 9, 26, 18, 35, 11)
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
    def declination(self) -> float | None:
        """
        Magnetic declination in degrees from true north.

        Returns
        -------
        float or None
            Magnetic declination in degrees. Positive values indicate
            eastward declination, negative values indicate westward
            declination. Returns None if declination data is not available.

        Notes
        -----
        Magnetic declination is only available in GPRMC messages.
        GPGGA messages will return None as they don't contain declination data.

        Western declination values are automatically converted to negative.

        Examples
        --------
        >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
        >>> gps.declination
        13.1
        """
        if self._declination is None or self._declination_hemisphere is None:
            return None
        dec = float(self._declination)
        if "w" in self._declination_hemisphere.lower():
            dec *= -1
        return dec

    @property
    def gps_type(self) -> str | None:
        """
        GPS message type.

        Returns
        -------
        str or None
            GPS message type: "GPRMC" or "GPGGA", or None if not set.

        Examples
        --------
        >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
        >>> gps.gps_type
        'GPRMC'
        """
        return self._type

    @property
    def fix(self) -> str | None:
        """
        GPS fix status.

        Returns
        -------
        str or None
            GPS fix status (typically "A" for valid fix), or None
            if fix information is not available or not applicable for
            the message type.

        Notes
        -----
        Fix status is typically available in GPRMC messages:
        - "A": Valid fix
        - "V": Invalid fix

        GPGGA messages use different fix quality indicators.

        Examples
        --------
        >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
        >>> gps.fix
        'A'
        """
        if hasattr(self, "_fix"):
            return self._fix
        return None
