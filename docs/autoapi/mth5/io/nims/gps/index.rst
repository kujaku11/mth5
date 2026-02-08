mth5.io.nims.gps
================

.. py:module:: mth5.io.nims.gps

.. autoapi-nested-parse::

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

   .. rubric:: Notes

   The GPS parser handles two main GPS message types:
   - GPRMC: Provides full date/time information and magnetic declination
   - GPGGA: Provides elevation data and fix quality information

   Binary data contamination is automatically cleaned during parsing.

   .. rubric:: Examples

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



Exceptions
----------

.. autoapisummary::

   mth5.io.nims.gps.GPSError


Classes
-------

.. autoapisummary::

   mth5.io.nims.gps.GPS


Module Contents
---------------

.. py:exception:: GPSError

   Bases: :py:obj:`Exception`


   Custom exception for GPS parsing and validation errors.

   Raised when GPS string parsing fails or when GPS data validation
   encounters invalid values.


.. py:class:: GPS(gps_string: str | bytes, index: int = 0)

   Parser for GPS stamps from NIMS magnetotelluric data.

   Handles parsing and validation of GPS strings from NIMS data files.
   Supports both GPRMC and GPGGA message formats, automatically detecting
   the type and extracting relevant geographic and temporal information.

   :param gps_string: Raw GPS string to be parsed. Can contain binary contamination
                      which will be automatically cleaned.
   :type gps_string: str or bytes
   :param index: Index or sequence number for this GPS record.
   :type index: int, default 0

   .. attribute:: gps_string

      The original GPS string provided for parsing.

      :type: str

   .. attribute:: index

      Index or sequence number for this GPS record.

      :type: int

   .. attribute:: valid

      Whether the GPS string was successfully parsed and validated.

      :type: bool

   .. attribute:: elevation_units

      Units for elevation measurements, typically "meters".

      :type: str

   .. attribute:: logger

      Logger instance for debugging and error reporting.

      :type: loguru.Logger

   .. rubric:: Notes

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

   .. rubric:: Examples

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


   .. py:attribute:: logger


   .. py:attribute:: gps_string


   .. py:attribute:: index
      :value: 0



   .. py:attribute:: valid
      :value: False



   .. py:attribute:: elevation_units
      :value: 'meters'



   .. py:attribute:: type_dict


   .. py:method:: validate_gps_string(gps_string: str | bytes) -> str | None

      Validate and clean GPS string.

      Removes binary contamination, finds string terminator, and validates
      format. Handles both string and bytes input.

      :param gps_string: Raw GPS string to validate. May contain binary contamination
                         that will be automatically removed.
      :type gps_string: str or bytes

      :returns: Cleaned GPS string with terminator removed, or None if validation
                fails due to missing terminator or decode errors.
      :rtype: str or None

      :raises TypeError: If input is not string or bytes.

      .. rubric:: Notes

      Binary contamination bytes that are automatically removed:
      - ``\xd9``, ``\xc7``, ``\xcc``
      - ``\x00`` (null byte, replaced with '*' terminator)

      The GPS string must end with '*' character to be considered valid.

      .. rubric:: Examples

      Clean a contaminated binary GPS string:

      >>> gps = GPS("")
      >>> contaminated = b"GPRMC,183511,A\xd9,3443.6098,N*"
      >>> clean = gps.validate_gps_string(contaminated)
      >>> print(clean)
      GPRMC,183511,A,3443.6098,N

      Handle missing terminator:

      >>> invalid = "GPRMC,183511,A,3443.6098,N"  # No '*'
      >>> result = gps.validate_gps_string(invalid)
      >>> print(result)
      None



   .. py:method:: parse_gps_string(gps_string: str | bytes) -> None

      Parse GPS string and populate object attributes.

      Main parsing method that validates the GPS string, identifies the
      message type (GPRMC/GPGGA), and extracts all relevant information
      into object attributes.

      :param gps_string: Raw GPS string from NIMS data file.
      :type gps_string: str or bytes

      .. rubric:: Notes

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

      .. rubric:: Examples

      Parse a valid GPS string:

      >>> gps = GPS("")
      >>> gps.parse_gps_string("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
      >>> print(f"Valid: {gps.valid}, Type: {gps.gps_type}")
      Valid: True, Type: GPRMC

      Handle invalid GPS string:

      >>> gps.parse_gps_string("invalid_gps_data")
      >>> print(f"Valid: {gps.valid}")
      Valid: False



   .. py:method:: validate_gps_list(gps_list: list[str]) -> tuple[list[str] | None, list[str]]

      Validate GPS field list and check format compliance.

      Performs comprehensive validation of GPS message components including
      type checking, length validation, and field-specific validation.

      :param gps_list: GPS message components split by delimiter.
      :type gps_list: list of str

      :returns: * **gps_list** (*list of str or None*) -- Validated GPS list with corrected values, or None if
                  critical validation fails.
                * **error_list** (*list of str*) -- List of validation error messages encountered during processing.

      .. rubric:: Notes

      Validation steps performed:
      1. GPS message type validation and correction
      2. Message length validation based on type
      3. Time format validation (6 digits)
      4. Coordinate validation (latitude/longitude + hemisphere)
      5. Date validation for GPRMC messages
      6. Elevation validation for GPGGA messages

      Non-critical validation errors are collected but don't halt processing.
      Critical errors (type or length) return None and stop validation.

      .. rubric:: Examples

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



   .. py:property:: latitude
      :type: float


      Latitude in decimal degrees (WGS84).

      :returns: Latitude in decimal degrees. Negative values indicate
                Southern hemisphere. Returns 0.0 if coordinate data is invalid.
      :rtype: float

      .. rubric:: Notes

      Converts from GPS format (DDMM.MMMM) to decimal degrees:
      decimal_degrees = degrees + minutes/60

      Southern hemisphere coordinates are automatically converted to negative values.

      .. rubric:: Examples

      >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
      >>> gps.latitude
      34.72683


   .. py:property:: longitude
      :type: float


      Longitude in decimal degrees (WGS84).

      :returns: Longitude in decimal degrees. Negative values indicate
                Western hemisphere. Returns 0.0 if coordinate data is invalid.
      :rtype: float

      .. rubric:: Notes

      Converts from GPS format (DDDMM.MMMM) to decimal degrees:
      decimal_degrees = degrees + minutes/60

      Western hemisphere coordinates are automatically converted to negative values.

      .. rubric:: Examples

      >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
      >>> gps.longitude
      -115.73501166666667


   .. py:property:: elevation
      :type: float


      Elevation above sea level in meters.

      :returns: Elevation in meters. Returns 0.0 if elevation data is not
                available or cannot be converted.
      :rtype: float

      .. rubric:: Notes

      Elevation is typically only available in GPGGA messages.
      GPRMC messages will return 0.0 as they don't contain elevation data.

      Conversion errors are logged but don't raise exceptions.

      .. rubric:: Examples

      >>> gps = GPS("GPGGA,183511,3443.6098,N,11544.1007,W,1,04,2.6,937.2,M,-28.1,M,*")
      >>> gps.elevation
      937.2


   .. py:property:: time_stamp
      :type: datetime.datetime | None


      GPS timestamp as datetime object.

      :returns: Timestamp parsed from GPS data, or None if time data is invalid.
      :rtype: datetime.datetime or None

      .. rubric:: Notes

      For GPRMC messages: Uses full date and time information
      For GPGGA messages: Uses time with default date of 1980-01-01

      Time format: HHMMSS (hours, minutes, seconds)
      Date format: DDMMYY (day, month, 2-digit year)

      Invalid date strings are logged but return None rather than raising exceptions.

      .. rubric:: Examples

      >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
      >>> gps.time_stamp
      datetime.datetime(2019, 9, 26, 18, 35, 11)


   .. py:property:: declination
      :type: float | None


      Magnetic declination in degrees from true north.

      :returns: Magnetic declination in degrees. Positive values indicate
                eastward declination, negative values indicate westward
                declination. Returns None if declination data is not available.
      :rtype: float or None

      .. rubric:: Notes

      Magnetic declination is only available in GPRMC messages.
      GPGGA messages will return None as they don't contain declination data.

      Western declination values are automatically converted to negative.

      .. rubric:: Examples

      >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
      >>> gps.declination
      13.1


   .. py:property:: gps_type
      :type: str | None


      GPS message type.

      :returns: GPS message type: "GPRMC" or "GPGGA", or None if not set.
      :rtype: str or None

      .. rubric:: Examples

      >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
      >>> gps.gps_type
      'GPRMC'


   .. py:property:: fix
      :type: str | None


      GPS fix status.

      :returns: GPS fix status (typically "A" for valid fix), or None
                if fix information is not available or not applicable for
                the message type.
      :rtype: str or None

      .. rubric:: Notes

      Fix status is typically available in GPRMC messages:
      - "A": Valid fix
      - "V": Invalid fix

      GPGGA messages use different fix quality indicators.

      .. rubric:: Examples

      >>> gps = GPS("GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*")
      >>> gps.fix
      'A'


