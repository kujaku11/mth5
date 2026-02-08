mth5.io.nims
============

.. py:module:: mth5.io.nims


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/nims/gps/index
   /autoapi/mth5/io/nims/header/index
   /autoapi/mth5/io/nims/nims/index
   /autoapi/mth5/io/nims/nims_collection/index
   /autoapi/mth5/io/nims/response_filters/index


Exceptions
----------

.. autoapisummary::

   mth5.io.nims.GPSError


Classes
-------

.. autoapisummary::

   mth5.io.nims.GPS
   mth5.io.nims.NIMSHeader
   mth5.io.nims.Response
   mth5.io.nims.NIMS
   mth5.io.nims.NIMSCollection


Functions
---------

.. autoapisummary::

   mth5.io.nims.read_nims


Package Contents
----------------

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


.. py:exception:: GPSError

   Bases: :py:obj:`Exception`


   Custom exception for GPS parsing and validation errors.

   Raised when GPS string parsing fails or when GPS data validation
   encounters invalid values.


.. py:class:: NIMSHeader(fn: Optional[Union[str, pathlib.Path]] = None)

   Class to hold NIMS header information.

   This class parses and stores header information from NIMS DATA.BIN files.
   The header contains metadata about the measurement site, equipment setup,
   GPS coordinates, electrode configuration, and other survey parameters.

   :param fn: Path to the NIMS file to read, by default None
   :type fn: str or Path, optional

   .. attribute:: fn

      Path to the NIMS file

      :type: Path or None

   .. attribute:: site_name

      Name of the measurement site

      :type: str or None

   .. attribute:: state_province

      State or province of the measurement location

      :type: str or None

   .. attribute:: country

      Country of the measurement location

      :type: str or None

   .. attribute:: box_id

      System box identifier

      :type: str or None

   .. attribute:: mag_id

      Magnetometer head identifier

      :type: str or None

   .. attribute:: ex_length

      North-South electric field wire length in meters

      :type: float or None

   .. attribute:: ex_azimuth

      North-South electric field wire heading in degrees

      :type: float or None

   .. attribute:: ey_length

      East-West electric field wire length in meters

      :type: float or None

   .. attribute:: ey_azimuth

      East-West electric field wire heading in degrees

      :type: float or None

   .. attribute:: n_electrode_id

      North electrode identifier

      :type: str or None

   .. attribute:: s_electrode_id

      South electrode identifier

      :type: str or None

   .. attribute:: e_electrode_id

      East electrode identifier

      :type: str or None

   .. attribute:: w_electrode_id

      West electrode identifier

      :type: str or None

   .. attribute:: ground_electrode_info

      Ground electrode information

      :type: str or None

   .. attribute:: header_gps_stamp

      GPS timestamp from header

      :type: MTime or None

   .. attribute:: header_gps_latitude

      GPS latitude from header in decimal degrees

      :type: float or None

   .. attribute:: header_gps_longitude

      GPS longitude from header in decimal degrees

      :type: float or None

   .. attribute:: header_gps_elevation

      GPS elevation from header in meters

      :type: float or None

   .. attribute:: operator

      Operator name

      :type: str or None

   .. attribute:: comments

      Survey comments

      :type: str or None

   .. attribute:: run_id

      Run identifier

      :type: str or None

   .. attribute:: data_start_seek

      Byte position where data begins in file

      :type: int

   .. rubric:: Examples

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


   .. py:attribute:: logger


   .. py:property:: fn
      :type: Optional[pathlib.Path]


      Full path to NIMS file.

      :returns: Path object representing the NIMS file location,
                or None if no file is set
      :rtype: Path or None


   .. py:attribute:: header_dict
      :value: None



   .. py:attribute:: site_name
      :value: None



   .. py:attribute:: state_province
      :value: None



   .. py:attribute:: country
      :value: None



   .. py:attribute:: box_id
      :value: None



   .. py:attribute:: mag_id
      :value: None



   .. py:attribute:: ex_length
      :value: None



   .. py:attribute:: ex_azimuth
      :value: None



   .. py:attribute:: ey_length
      :value: None



   .. py:attribute:: ey_azimuth
      :value: None



   .. py:attribute:: n_electrode_id
      :value: None



   .. py:attribute:: s_electrode_id
      :value: None



   .. py:attribute:: e_electrode_id
      :value: None



   .. py:attribute:: w_electrode_id
      :value: None



   .. py:attribute:: ground_electrode_info
      :value: None



   .. py:attribute:: header_gps_stamp
      :value: None



   .. py:attribute:: header_gps_latitude
      :value: None



   .. py:attribute:: header_gps_longitude
      :value: None



   .. py:attribute:: header_gps_elevation
      :value: None



   .. py:attribute:: operator
      :value: None



   .. py:attribute:: comments


   .. py:attribute:: run_id
      :value: None



   .. py:attribute:: data_start_seek
      :value: 0



   .. py:property:: station
      :type: Optional[str]


      Station ID derived from run ID.

      :returns: Station identifier (run ID without the last character),
                or None if run_id is not set
      :rtype: str or None

      .. rubric:: Notes

      The station ID is typically the run ID with the last character
      (run letter) removed.


   .. py:property:: file_size
      :type: Optional[int]


      Size of the NIMS file in bytes.

      :returns: File size in bytes, or None if no file is set
      :rtype: int or None

      :raises FileNotFoundError: If the file does not exist


   .. py:method:: read_header(fn: Optional[Union[str, pathlib.Path]] = None) -> None

      Read header information from a NIMS file.

      This method reads and parses the header section of a NIMS DATA.BIN file,
      extracting metadata about the survey setup, GPS coordinates, electrode
      configuration, and other parameters.

      :param fn: Full path to NIMS file to read. Uses self.fn if not provided.
      :type fn: str or Path, optional

      :raises NIMSError: If the file does not exist or cannot be read

      .. rubric:: Notes

      The method reads up to _max_header_length bytes from the beginning
      of the file, parses the header information, and stores the results
      in the header_dict attribute and individual properties.



   .. py:method:: parse_header_dict(header_dict: Optional[dict[str, str]] = None) -> None

      Parse the header dictionary into individual attributes.

      This method takes the raw header dictionary and extracts specific
      information into class attributes for easy access.

      :param header_dict: Dictionary containing header key-value pairs. Uses self.header_dict
                          if not provided.
      :type header_dict: dict of str, optional

      .. rubric:: Notes

      Parses various header fields including:
      - Wire lengths and azimuths for electric field measurements
      - System box and magnetometer IDs
      - GPS coordinates and timestamp
      - Run identifier
      - Other metadata fields



.. py:class:: Response(system_id=None, **kwargs)

   Bases: :py:obj:`object`


   Common NIMS response filters for electric and magnetic channels



   .. py:attribute:: system_id
      :value: None



   .. py:attribute:: hardware
      :value: 'PC'



   .. py:attribute:: instrument_type
      :value: 'backbone'



   .. py:attribute:: sample_rate
      :value: 1



   .. py:attribute:: e_conversion_factor
      :value: 409600000.0



   .. py:attribute:: h_conversion_factor
      :value: 100



   .. py:attribute:: time_delays_dict


   .. py:property:: magnetic_low_pass

      Low pass 3 pole filter

      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: magnetic_conversion

      DESCRIPTION
      :rtype: TYPE

      :type: return


   .. py:property:: electric_low_pass

      5 pole electric low pass filter
      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: electric_high_pass_pc

      1-pole low pass filter for 8 hz instruments
      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: electric_high_pass_hp

      1-pole low pass for 1 hz instuments
      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: electric_conversion

      electric channel conversion from counts to Volts
      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: electric_physical_units

      DESCRIPTION
      :rtype: TYPE

      :type: return


   .. py:method:: get_electric_high_pass(hardware='pc')

      get the electric high pass filter based on the hardware



   .. py:method:: dipole_filter(length)

      Make a dipole filter

      :param length: dipole length in meters
      :type length: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_channel_response(channel, dipole_length=1)

      Get the full channel response filter
      :param channel: DESCRIPTION
      :type channel: TYPE
      :param dipole_length: DESCRIPTION, defaults to 1
      :type dipole_length: TYPE, optional
      :return: DESCRIPTION
      :rtype: TYPE




.. py:class:: NIMS(fn: Optional[Union[str, pathlib.Path]] = None)

   Bases: :py:obj:`mth5.io.nims.header.NIMSHeader`


   NIMS Class for reading NIMS DATA.BIN files.

   A fast way to read the binary files are to first read in the GPS strings,
   the third byte in each block as a character and parse that into valid
   GPS stamps.

   Then read in the entire data set as unsigned 8 bit integers and reshape
   the data to be n seconds x block size. Then parse that array into the
   status information and data.

   :param fn: Path to the NIMS DATA.BIN file to read, by default None
   :type fn: str or Path, optional

   .. attribute:: block_size

      Size of data blocks (default 131 for 8 Hz data)

      :type: int

   .. attribute:: block_sequence

      Sequence pattern to locate [1, 131]

      :type: list of int

   .. attribute:: sample_rate

      Sample rate in samples/second (default 8)

      :type: int

   .. attribute:: e_conversion_factor

      Electric field conversion factor

      :type: float

   .. attribute:: h_conversion_factor

      Magnetic field conversion factor

      :type: float

   .. attribute:: t_conversion_factor

      Temperature conversion factor

      :type: float

   .. attribute:: t_offset

      Temperature offset value

      :type: int

   .. attribute:: info_array

      Structured array of block information

      :type: ndarray or None

   .. attribute:: stamps

      List of valid GPS stamps

      :type: list or None

   .. attribute:: ts_data

      Time series data as pandas DataFrame

      :type: DataFrame or None

   .. attribute:: gaps

      List of timing gaps found in data

      :type: list or None

   .. attribute:: duplicate_list

      List of duplicate blocks found

      :type: list or None

   .. rubric:: Notes

   I only have a limited amount of .BIN files to test so this will likely
   break if there are issues such as data gaps. This has been tested against the
   matlab program loadNIMS by Anna Kelbert and the match for all the .bin files
   I have. If something looks weird check it against that program.

   .. todo:: Deal with timing issues, right now a warning is sent to the user
             need to figure out a way to find where the gap is and adjust
             accordingly.

   .. warning::
       Currently Only 8 Hz data is supported

   .. rubric:: Examples

   >>> from mth5.io.nims import nims
   >>> n = nims.NIMS(r"/home/mt_data/nims/mt001.bin")
   >>> n.read_nims()


   .. py:attribute:: block_size
      :value: 131



   .. py:attribute:: block_sequence


   .. py:attribute:: sample_rate
      :value: 8



   .. py:attribute:: e_conversion_factor
      :value: 2.44141221047903e-06



   .. py:attribute:: h_conversion_factor
      :value: 0.01



   .. py:attribute:: t_conversion_factor
      :value: 70



   .. py:attribute:: t_offset
      :value: 18048



   .. py:attribute:: info_array
      :value: None



   .. py:attribute:: stamps
      :value: None



   .. py:attribute:: ts_data
      :value: None



   .. py:attribute:: gaps
      :value: None



   .. py:attribute:: duplicate_list
      :value: None



   .. py:attribute:: indices


   .. py:method:: has_data() -> bool

      Check if the NIMS object contains time series data.

      :returns: True if ts_data is not None, False otherwise
      :rtype: bool



   .. py:property:: n_samples
      :type: Optional[int]


      Number of samples in the time series.

      :returns: Number of samples if data is loaded, estimated from file size
                if file exists, None otherwise
      :rtype: int or None


   .. py:property:: latitude
      :type: Optional[float]


      Median latitude value from all GPS stamps.

      :returns: Median latitude in decimal degrees (WGS84) from GPRMC stamps,
                or header GPS latitude if no stamps available
      :rtype: float or None

      .. rubric:: Notes

      Only uses GPRMC stamps as they should be duplicates of GPGGA stamps
      but include additional validation.


   .. py:property:: longitude
      :type: Optional[float]


      Median longitude value from all GPS stamps.

      :returns: Median longitude in decimal degrees (WGS84) from GPS stamps,
                or header GPS longitude if no stamps available
      :rtype: float or None

      .. rubric:: Notes

      Uses the first stamp within each GPS stamp set.


   .. py:property:: elevation
      :type: Optional[float]


      Median elevation value from all GPS stamps.

      :returns: Median elevation in meters (WGS84) from GPS stamps,
                or header GPS elevation if no stamps available
      :rtype: float or None

      .. rubric:: Notes

      Uses the first stamp within each GPS stamp set. For paired stamps
      (GPRMC/GPGGA), uses the GPGGA elevation if available.


   .. py:property:: declination
      :type: Optional[float]


      Median magnetic declination value from all GPS stamps.

      :returns: Median magnetic declination in decimal degrees from GPRMC stamps,
                or None if no declination data available
      :rtype: float or None

      .. rubric:: Notes

      Only uses GPRMC stamps as they contain declination information.


   .. py:property:: start_time
      :type: mt_metadata.common.mttime.MTime


      Start time of the time series data.

      :returns: Start time derived from the first GPS time stamp index,
                or header GPS stamp if no time series data available
      :rtype: MTime

      .. rubric:: Notes

      The start time is calculated from the first good GPS time stamp
      minus the seconds to the beginning of the time series.


   .. py:property:: end_time
      :type: mt_metadata.common.mttime.MTime


      End time of the time series data.

      :returns: End time derived from the last time series index,
                or estimated from start time and number of samples
      :rtype: MTime

      .. rubric:: Notes

      If time series data is available, uses the last timestamp.
      Otherwise estimates end time from start time plus duration
      calculated from number of samples and sample rate.


   .. py:property:: box_temperature
      :type: Optional[mth5.timeseries.ChannelTS]


      Data logger temperature channel.

      :returns: Temperature channel sampled at 1 second, interpolated to match
                the time series sample rate, or None if no time series data
      :rtype: ChannelTS or None

      .. rubric:: Notes

      Temperature is measured in Celsius and interpolated onto the same
      time grid as the magnetic and electric field channels.


   .. py:method:: get_channel_response(channel: str, dipole_length: float = 1) -> Any

      Get the channel response for a given channel.

      :param channel: Channel identifier (e.g., 'hx', 'hy', 'hz', 'ex', 'ey')
      :type channel: str
      :param dipole_length: Dipole length for electric field channels, by default 1
      :type dipole_length: float, optional

      :returns: Channel response object from the NIMS response filters
      :rtype: Any

      .. rubric:: Notes

      Uses the NIMS response filters to generate appropriate response
      functions for magnetic and electric field channels at the current
      sample rate.



   .. py:property:: hx_metadata
      :type: Optional[mt_metadata.timeseries.Magnetic]


      Metadata for the HX magnetic field channel.

      :returns: Magnetic field metadata object for the HX channel,
                or None if no time series data is loaded
      :rtype: Magnetic or None


   .. py:property:: hx
      :type: Optional[mth5.timeseries.ChannelTS]


      HX magnetic field channel time series.

      :returns: Time series data for the HX magnetic field component,
                or None if no time series data is loaded
      :rtype: ChannelTS or None


   .. py:property:: hy_metadata
      :type: Optional[mt_metadata.timeseries.Magnetic]


      Metadata for the HY magnetic field channel.

      :returns: Magnetic field metadata object for the HY channel,
                or None if no time series data is loaded
      :rtype: Magnetic or None


   .. py:property:: hy
      :type: Optional[mth5.timeseries.ChannelTS]


      HY magnetic field channel time series.

      :returns: Time series data for the HY magnetic field component,
                or None if no time series data is loaded
      :rtype: ChannelTS or None


   .. py:property:: hz_metadata
      :type: Optional[mt_metadata.timeseries.Magnetic]


      Metadata for the HZ magnetic field channel.

      :returns: Magnetic field metadata object for the HZ channel,
                or None if no time series data is loaded
      :rtype: Magnetic or None


   .. py:property:: hz
      :type: Optional[mth5.timeseries.ChannelTS]


      HZ magnetic field channel time series.

      :returns: Time series data for the HZ magnetic field component,
                or None if no time series data is loaded
      :rtype: ChannelTS or None


   .. py:property:: ex_metadata
      :type: Optional[mt_metadata.timeseries.Electric]


      Metadata for the EX electric field channel.

      :returns: Electric field metadata object for the EX channel,
                or None if no time series data is loaded
      :rtype: Electric or None


   .. py:property:: ex
      :type: Optional[mth5.timeseries.ChannelTS]


      EX electric field channel time series.

      :returns: Time series data for the EX electric field component,
                or None if no time series data is loaded
      :rtype: ChannelTS or None


   .. py:property:: ey_metadata
      :type: Optional[mt_metadata.timeseries.Electric]


      Metadata for the EY electric field channel.

      :returns: Electric field metadata object for the EY channel,
                or None if no time series data is loaded
      :rtype: Electric or None


   .. py:property:: ey
      :type: Optional[mth5.timeseries.ChannelTS]


      EY electric field channel time series.

      :returns: Time series data for the EY electric field component,
                or None if no time series data is loaded
      :rtype: ChannelTS or None


   .. py:property:: run_metadata
      :type: Optional[mt_metadata.timeseries.Run]


      Run metadata for the NIMS data collection.

      :returns: MT run metadata including data logger information, timing,
                and channel metadata, or None if no time series data is loaded
      :rtype: Run or None


   .. py:property:: station_metadata
      :type: Optional[mt_metadata.timeseries.Station]


      Station metadata from NIMS file.

      :returns: MT station metadata including geographic information and location data,
                or None if no time series data is loaded
      :rtype: Station or None


   .. py:method:: to_runts(calibrate: bool = False) -> Optional[mth5.timeseries.RunTS]

      Get xarray RunTS object for the NIMS data.

      :param calibrate: Whether to apply calibration to the data, by default False
      :type calibrate: bool, optional

      :returns: Time series run object containing all channels and metadata,
                or None if no time series data is loaded
      :rtype: RunTS or None

      .. rubric:: Notes

      Includes all magnetic field channels (hx, hy, hz), electric field
      channels (ex, ey), and box temperature data.



   .. py:method:: get_stamps(nims_string: bytes) -> list[tuple[Any, list[mth5.io.nims.gps.GPS]]]

      Extract and parse valid GPS strings, matching GPRMC with GPGGA stamps.

      :param nims_string: Raw GPS binary string output by NIMS
      :type nims_string: bytes

      :returns: List of matched GPS stamps where each element is a tuple containing
                index and list of GPS objects [GPRMC, GPGGA] (or just [GPRMC])
      :rtype: list of tuple

      .. rubric:: Notes

      Skips the first entry as it tends to be incomplete. Attempts to match
      synchronous GPRMC with GPGGA stamps when possible.



   .. py:method:: match_status_with_gps_stamps(status_array, gps_list)

      Match the index values from the status array with the index values of
      the GPS stamps.  There appears to be a bit of wiggle room between when the
      lock is recorded and the stamp was actually recorded.  This is typically 1
      second and sometimes 2.

      :param array status_array: array of status values from each data block
      :param list gps_list: list of valid GPS stamps [[GPRMC, GPGGA], ...]

      .. note:: I think there is a 2 second gap between the lock and the
                first stamp character.



   .. py:method:: find_sequence(data_array: numpy.ndarray, block_sequence: Optional[list[int]] = None) -> numpy.ndarray

      Find a sequence pattern in the data array.

      :param data_array: Array of the data with shape [n, m] where n is the number of
                         seconds recorded and m is the block length for a given sampling rate
      :type data_array: ndarray
      :param block_sequence: Sequence pattern to locate, by default [1, 131] (start of data block)
      :type block_sequence: list of int, optional

      :returns: Array of index locations where the sequence is found
      :rtype: ndarray

      .. rubric:: Notes

      Uses numpy rolling and comparison to find all occurrences of the
      specified sequence pattern in the data array.



   .. py:method:: unwrap_sequence(sequence: numpy.ndarray) -> numpy.ndarray

      Unwrap sequence to sequential numbers instead of modulo 256.

      :param sequence: Sequence of byte numbers (0-255) to unwrap
      :type sequence: ndarray

      :returns: Unwrapped sequence with first number set to 0 and subsequent
                values forming a continuous count
      :rtype: ndarray

      .. rubric:: Notes

      Handles the fact that sequence numbers are stored as single bytes
      (0-255) but represent a continuous count. When a value of 255 is
      encountered, the next rollover is anticipated.



   .. py:method:: remove_duplicates(info_array, data_array)

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




   .. py:method:: read_nims(fn: Optional[Union[str, pathlib.Path]] = None) -> None

      Read NIMS DATA.BIN file and parse all data.

      This method performs the complete data reading and processing workflow:

      1. Read header information and store as attributes
      2. Locate data block beginning by finding first [1, 131, ...] sequence
      3. Ensure data is multiple of block length, trim excess bits
      4. Extract GPS data (3rd byte of each block) and parse GPS stamps
      5. Read data as unsigned 8-bit integers, reshape to [N, block_length]
      6. Remove duplicate blocks (first of each duplicate pair)
      7. Match GPS status locks with valid GPS stamps
      8. Verify timing between first/last GPS stamps, trim excess seconds

      :param fn: Path to NIMS DATA.BIN file. Uses self.fn if not provided.
      :type fn: str or Path, optional

      .. rubric:: Notes

      The data and information arrays returned have duplicates removed
      and sequence reset to be monotonic. Extra seconds due to timing
      gaps are trimmed from the end of the time series.

      .. rubric:: Examples

      >>> from mth5.io import nims
      >>> n = nims.NIMS(r"/home/mt_data/nims/mt001.bin")
      >>> n.read_nims()



   .. py:method:: check_timing(stamps)

      make sure that there are the correct number of seconds in between
      the first and last GPS GPRMC stamps

      :param list stamps: list of GPS stamps [[status_index, [GPRMC, GPGGA]]]

      :returns: [ True | False ] if data is valid or not.
      :returns: gap index locations

      .. note:: currently it is assumed that if a data gap occurs the data can be
                squeezed to remove them.  Probably a more elegant way of doing it.



   .. py:method:: align_data(data_array, stamps)

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



   .. py:method:: make_dt_index(start_time: str, sample_rate: float, stop_time: Optional[str] = None, n_samples: Optional[int] = None) -> pandas.DatetimeIndex

      Create datetime index array for time series data.

      :param start_time: Start time in format YYYY-MM-DDThh:mm:ss.ms UTC
      :type start_time: str
      :param sample_rate: Sample rate in samples/second
      :type sample_rate: float
      :param stop_time: End time in same format as start_time
      :type stop_time: str, optional
      :param n_samples: Number of samples to generate
      :type n_samples: int, optional

      :returns: Pandas datetime index with UTC timezone
      :rtype: DatetimeIndex

      .. rubric:: Notes

      Either stop_time or n_samples must be provided. The datetime format
      should be YYYY-MM-DDThh:mm:ss.ms UTC.

      :raises ValueError: If neither stop_time nor n_samples is provided



.. py:function:: read_nims(fn: Union[str, pathlib.Path]) -> Optional[mth5.timeseries.RunTS]

   Convenience function to read a NIMS DATA.BIN file.

   :param fn: Path to the NIMS DATA.BIN file
   :type fn: str or Path

   :returns: Time series run object containing all channels and metadata,
             or None if reading fails
   :rtype: RunTS or None

   .. rubric:: Examples

   >>> from mth5.io.nims import nims
   >>> run_ts = nims.read_nims("/path/to/data.bin")


.. py:class:: NIMSCollection(file_path: str | pathlib.Path | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.io.collection.Collection`


   Collection of NIMS binary files into runs.

   This class provides functionality for organizing and processing multiple NIMS
   binary files into a structured format for magnetotelluric data analysis.

   :param file_path: Path to the directory containing NIMS binary files.
   :type file_path: str | Path | None, optional
   :param \*\*kwargs: Additional keyword arguments passed to the parent Collection class.
   :type \*\*kwargs: dict

   .. attribute:: file_ext

      File extension for NIMS binary files ('bin').

      :type: str

   .. attribute:: survey_id

      Survey identifier, defaults to 'mt'.

      :type: str

   .. rubric:: Examples

   >>> from mth5.io.nims import NIMSCollection
   >>> nc = NIMSCollection(r"/path/to/nims/station")
   >>> nc.survey_id = "mt001"
   >>> df = nc.to_dataframe()

   .. seealso::

      :obj:`mth5.io.collection.Collection`
          Base collection class

      :obj:`mth5.io.nims.NIMS`
          NIMS file reader


   .. py:attribute:: file_ext
      :type:  str
      :value: 'bin'



   .. py:attribute:: survey_id
      :type:  str
      :value: 'mt'



   .. py:method:: to_dataframe(sample_rates: int | list[int] = [1], run_name_zeros: int = 2, calibration_path: str | pathlib.Path | None = None) -> pandas.DataFrame

      Create a DataFrame of each NIMS binary file in the collection directory.

      This method processes all NIMS binary files in the specified directory and
      extracts metadata to create a structured DataFrame suitable for further
      magnetotelluric data processing.

      :param sample_rates: Sample rates to include in the DataFrame. Note that for NIMS data,
                           this parameter is present for interface consistency but all files
                           will be processed regardless of their sample rate.
      :type sample_rates: int | list[int], default [1]
      :param run_name_zeros: Number of zeros to use when formatting run names in the output.
      :type run_name_zeros: int, default 2
      :param calibration_path: Path to calibration files. Currently not used in NIMS processing
                               but included for interface consistency.
      :type calibration_path: str | Path | None, optional

      :returns: DataFrame containing metadata for each NIMS file with columns:
                - survey : Survey identifier
                - station : Station name from NIMS file
                - run : Run identifier from NIMS file
                - start : Start time in ISO format
                - end : End time in ISO format
                - fn : File path
                - sample_rate : Sampling rate
                - file_size : File size in bytes
                - n_samples : Number of samples
                - dipole : Electric dipole lengths [Ex, Ey]
                - channel_id : Channel identifier (always 1)
                - sequence_number : Sequence number (always 0)
                - component : Comma-separated component list
                - instrument_id : Instrument identifier (always 'NIMS')
      :rtype: pd.DataFrame

      .. rubric:: Notes

      This method assumes the directory contains files from a single station.
      Each NIMS file is read to extract header information including timing,
      station identification, and measurement parameters.

      .. rubric:: Examples

      >>> from mth5.io.nims import NIMSCollection
      >>> nc = NIMSCollection("/path/to/nims/station")
      >>> df = nc.to_dataframe(run_name_zeros=3)
      >>> print(df[['station', 'run', 'start', 'sample_rate']])



   .. py:method:: assign_run_names(df: pandas.DataFrame, zeros: int = 2) -> pandas.DataFrame

      Assign standardized run names to DataFrame entries by station.

      This method assigns run names following the pattern 'sr{sample_rate}_{run_number}'
      where run_number is zero-padded according to the zeros parameter. Run names
      are assigned sequentially within each station, ordered by start time.

      :param df: DataFrame containing NIMS file metadata with required columns:
                 'station', 'start', 'run', 'sample_rate'. The DataFrame will be
                 modified in-place.
      :type df: pd.DataFrame
      :param zeros: Number of zeros to use for zero-padding the run number in the
                    generated run names (e.g., zeros=2 gives '01', '02', etc.).
      :type zeros: int, default 2

      :returns: The input DataFrame with updated 'run' and 'sequence_number' columns.
                Run names follow the format 'sr{sample_rate}_{run_number:0{zeros}}'.
      :rtype: pd.DataFrame

      .. rubric:: Notes

      - Existing run names (non-None values) are preserved
      - Files are processed in chronological order within each station
      - Sequence numbers are assigned incrementally starting from 1
      - Only files with None run names receive new assignments

      .. rubric:: Examples

      >>> import pandas as pd
      >>> from mth5.io.nims import NIMSCollection
      >>> # Assuming df has columns: station, start, run, sample_rate
      >>> nc = NIMSCollection()
      >>> df_updated = nc.assign_run_names(df, zeros=3)
      >>> print(df_updated['run'].tolist())
      ['sr8_001', 'sr8_002', 'sr1_001']



