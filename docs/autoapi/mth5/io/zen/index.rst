mth5.io.zen
===========

.. py:module:: mth5.io.zen


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/zen/coil_response/index
   /autoapi/mth5/io/zen/z3d_collection/index
   /autoapi/mth5/io/zen/z3d_header/index
   /autoapi/mth5/io/zen/z3d_metadata/index
   /autoapi/mth5/io/zen/z3d_schedule/index
   /autoapi/mth5/io/zen/zen/index
   /autoapi/mth5/io/zen/zen_tools/index


Classes
-------

.. autoapisummary::

   mth5.io.zen.Z3DHeader
   mth5.io.zen.Z3DSchedule
   mth5.io.zen.Z3DMetadata
   mth5.io.zen.Z3D
   mth5.io.zen.CoilResponse
   mth5.io.zen.Z3DCollection


Functions
---------

.. autoapisummary::

   mth5.io.zen.read_z3d


Package Contents
----------------

.. py:class:: Z3DHeader(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None, **kwargs: Any)

   Read header information from a Z3D file and make each metadata entry an attribute.

   :param fn: Full path to Z3D file.
   :type fn: str | pathlib.Path, optional
   :param fid: File object (e.g., open(Z3Dfile, 'rb')).
   :type fid: BinaryIO, optional
   :param \*\*kwargs: Additional keyword arguments to set as attributes.
   :type \*\*kwargs: dict

   .. attribute:: _header_len

      Length of header in bits (512).

      :type: int

   .. attribute:: ad_gain

      Gain of channel.

      :type: float | None

   .. attribute:: ad_rate

      Sampling rate in Hz.

      :type: float | None

   .. attribute:: alt

      Altitude of the station (not reliable).

      :type: float | None

   .. attribute:: attenchannelsmask

      Attenuation channels mask.

      :type: str | None

   .. attribute:: box_number

      ZEN box number.

      :type: float | None

   .. attribute:: box_serial

      ZEN box serial number.

      :type: str | None

   .. attribute:: channel

      Channel number of the file.

      :type: float | None

   .. attribute:: channelserial

      Serial number of the channel board.

      :type: str | None

   .. attribute:: ch_factor

      Channel factor (default 9.536743164062e-10).

      :type: float

   .. attribute:: channelgain

      Channel gain (default 1.0).

      :type: float

   .. attribute:: duty

      Duty cycle of the transmitter.

      :type: float | None

   .. attribute:: fid

      File object.

      :type: BinaryIO | None

   .. attribute:: fn

      Full path to Z3D file.

      :type: str | pathlib.Path | None

   .. attribute:: fpga_buildnum

      Build number of one of the boards.

      :type: float | None

   .. attribute:: gpsweek

      GPS week (default 1740).

      :type: int

   .. attribute:: header_str

      Full header string.

      :type: bytes | None

   .. attribute:: lat

      Latitude of station in degrees.

      :type: float | None

   .. attribute:: logterminal

      Log terminal setting.

      :type: str | None

   .. attribute:: long

      Longitude of the station in degrees.

      :type: float | None

   .. attribute:: main_hex_buildnum

      Build number of the ZEN box in hexadecimal.

      :type: float | None

   .. attribute:: numsats

      Number of GPS satellites.

      :type: float | None

   .. attribute:: old_version

      Whether this is an old version Z3D file (default False).

      :type: bool

   .. attribute:: period

      Period of the transmitter.

      :type: float | None

   .. attribute:: tx_duty

      Transmitter duty cycle.

      :type: float | None

   .. attribute:: tx_freq

      Transmitter frequency.

      :type: float | None

   .. attribute:: version

      Version of the firmware.

      :type: float | None

   .. rubric:: Examples

   >>> from mth5.io.zen import Z3DHeader
   >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
   >>> header_obj = Z3DHeader(fn=Z3Dfn)
   >>> header_obj.read_header()


   .. py:attribute:: logger


   .. py:attribute:: fn
      :type:  str | pathlib.Path | None
      :value: None



   .. py:attribute:: fid
      :type:  BinaryIO | None
      :value: None



   .. py:attribute:: header_str
      :type:  bytes | None
      :value: None



   .. py:attribute:: ad_gain
      :type:  float | None
      :value: None



   .. py:attribute:: ad_rate
      :type:  float | None
      :value: None



   .. py:attribute:: alt
      :type:  float | None
      :value: None



   .. py:attribute:: attenchannelsmask
      :type:  str | None
      :value: None



   .. py:attribute:: box_number
      :type:  float | None
      :value: None



   .. py:attribute:: box_serial
      :type:  str | None
      :value: None



   .. py:attribute:: channel
      :type:  float | None
      :value: None



   .. py:attribute:: channelserial
      :type:  str | None
      :value: None



   .. py:attribute:: duty
      :type:  float | None
      :value: None



   .. py:attribute:: fpga_buildnum
      :type:  float | None
      :value: None



   .. py:attribute:: gpsweek
      :type:  int
      :value: 1740



   .. py:attribute:: lat
      :type:  float | None
      :value: None



   .. py:attribute:: logterminal
      :type:  str | None
      :value: None



   .. py:attribute:: long
      :type:  float | None
      :value: None



   .. py:attribute:: main_hex_buildnum
      :type:  float | None
      :value: None



   .. py:attribute:: numsats
      :type:  float | None
      :value: None



   .. py:attribute:: period
      :type:  float | None
      :value: None



   .. py:attribute:: tx_duty
      :type:  float | None
      :value: None



   .. py:attribute:: tx_freq
      :type:  float | None
      :value: None



   .. py:attribute:: version
      :type:  float | None
      :value: None



   .. py:attribute:: old_version
      :type:  bool
      :value: False



   .. py:attribute:: ch_factor
      :type:  float
      :value: 9.536743164062e-10



   .. py:attribute:: channelgain
      :type:  float
      :value: 1.0



   .. py:property:: data_logger
      :type: str


      Data logger name as ZEN{box_number}.

      :returns: Data logger name formatted as 'ZEN' followed by zero-padded box number.
      :rtype: str

      :raises TypeError: If box_number is None or cannot be converted to int.


   .. py:method:: read_header(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None) -> None

      Read the header information into appropriate attributes.

      Parses the header information from a Z3D file and populates the object's
      attributes with the extracted values. Supports both modern and legacy
      Z3D file formats.

      :param fn: Full path to Z3D file. If None, uses the instance's fn attribute.
      :type fn: str | pathlib.Path, optional
      :param fid: File object (e.g., open(Z3Dfile, 'rb')). If None, uses the instance's
                  fid attribute or opens the file specified by fn.
      :type fid: BinaryIO, optional

      :raises UnicodeDecodeError: If header bytes cannot be decoded as text.

      .. rubric:: Notes

      This method reads the first 512 bytes of the Z3D file as the header.
      It supports two formats:

      1. Modern format: key=value pairs separated by newlines
      2. Legacy format: comma-separated key:value pairs

      The method automatically detects legacy format and sets old_version=True.

      Coordinate values (lat/long) are automatically converted from radians
      to degrees, with validation to ensure they fall within valid ranges.

      .. rubric:: Examples

      >>> header_obj = Z3DHeader()
      >>> header_obj.read_header("/path/to/file.Z3D")

      >>> with open("/path/to/file.Z3D", "rb") as fid:
      ...     header_obj.read_header(fid=fid)



   .. py:method:: convert_value(key_string: str, value_string: str) -> float | str

      Convert the value to the appropriate units given the key.

      Converts string values to appropriate types based on the key name.
      Special handling is provided for latitude and longitude values, which
      are converted from radians to degrees with validation.

      :param key_string: The metadata key name, used to determine conversion type.
      :type key_string: str
      :param value_string: The string value to convert.
      :type value_string: str

      :returns: Converted value. Returns float for numeric values, str for
                non-numeric values. Latitude and longitude values are converted
                from radians to degrees.
      :rtype: float or str

      .. rubric:: Notes

      - Attempts to convert all values to float first
      - If conversion fails, returns original string
      - For keys containing 'lat', 'lon', or 'long':
        - Converts from radians to degrees using np.rad2deg
        - Validates latitude range (±90°), sets to 0.0 if invalid
        - Validates longitude range (±180°), sets to 0.0 if invalid

      .. rubric:: Examples

      >>> header = Z3DHeader()
      >>> header.convert_value("version", "4147")
      4147.0
      >>> header.convert_value("lat", "0.706816081")  # radians
      40.49757833327694  # degrees
      >>> header.convert_value("channelserial", "0xD474777C")
      '0xD474777C'



.. py:class:: Z3DSchedule(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None, **kwargs: Any)

   Parser for Z3D file schedule information and metadata.

   Reads schedule information from Z3D files and creates object attributes
   for each metadata entry. Schedule data is stored at byte offset 512 in
   Z3D files and contains recording parameters, timing information, and
   instrument configuration settings.

   The class preserves the original capitalization from the Z3D file format
   and provides automatic parsing of key-value pairs in the schedule section.

   :param fn: Full path to Z3D file to read schedule information from.
              Can be string path or pathlib.Path object.
   :type fn: str | pathlib.Path, optional
   :param fid: Open file object for reading Z3D file in binary mode.
               Example: open('file.z3d', 'rb')
   :type fid: BinaryIO, optional
   :param \*\*kwargs: Additional keyword arguments to set as object attributes.
   :type \*\*kwargs: Any

   .. attribute:: AutoGain

      Auto gain setting for the recording channel ['Y' or 'N'].

      :type: str | None

   .. attribute:: Comment

      User comments or notes for the schedule configuration.

      :type: str | None

   .. attribute:: Date

      Date when the schedule action was started in YYYY-MM-DD format.

      :type: str | None

   .. attribute:: Duty

      Duty cycle percentage of the transmitter (0-100).

      :type: str | None

   .. attribute:: FFTStacks

      Number of FFT stacks used by the transmitter.

      :type: str | None

   .. attribute:: Filename

      Original filename that the ZEN instrument assigns to the recording.

      :type: str | None

   .. attribute:: Gain

      Gain setting for the recording channel (e.g., '1.0000').

      :type: str | None

   .. attribute:: Log

      Data logging enabled flag ['Y' or 'N'].

      :type: str | None

   .. attribute:: NewFile

      Create new file for recording flag ['Y' or 'N'].

      :type: str | None

   .. attribute:: Period

      Base period setting for the transmitter in seconds.

      :type: str | None

   .. attribute:: RadioOn

      Radio communication enabled flag ['Y', 'N', or 'X'].

      :type: str | None

   .. attribute:: SR

      Sampling rate in Hz (originally 'S/R' in file, converted to 'SR').

      :type: str | None

   .. attribute:: SamplesPerAcq

      Number of samples per acquisition for transmitter mode.

      :type: str | None

   .. attribute:: Sleep

      Sleep mode enabled flag ['Y' or 'N'].

      :type: str | None

   .. attribute:: Sync

      GPS synchronization enabled flag ['Y' or 'N'].

      :type: str | None

   .. attribute:: Time

      Time when the schedule action started in HH:MM:SS format (GPS time).

      :type: str | None

   .. attribute:: initial_start

      Parsed start time as MTime object with GPS time flag enabled.
      Combines Date and Time attributes for timestamp calculation.

      :type: MTime

   .. attribute:: fn

      Path to the Z3D file being processed.

      :type: str | pathlib.Path | None

   .. attribute:: fid

      File object for reading the Z3D file.

      :type: BinaryIO | None

   .. attribute:: meta_string

      Raw schedule metadata string read from the file.

      :type: bytes | None

   .. attribute:: _header_len

      Length of Z3D file header in bytes (512).

      :type: int

   .. attribute:: _schedule_metadata_len

      Length of schedule metadata section in bytes (512).

      :type: int

   .. attribute:: logger

      Loguru logger instance for debugging and status messages.

      :type: Logger

   .. rubric:: Notes

   The Z3D file format stores schedule information at a fixed offset of 512 bytes
   from the beginning of the file. The schedule section is also 512 bytes long
   and contains key-value pairs in the format "Schedule.key = value".

   All schedule values are stored as strings to preserve the original format
   from the Z3D file. Boolean-like values use 'Y'/'N' convention.

   The initial_start attribute automatically converts Date and Time into a
   GPS-corrected MTime object for accurate timestamp handling.

   .. rubric:: Examples

   Read schedule from file path:

   >>> from mth5.io.zen import Z3DSchedule
   >>> from pathlib import Path
   >>>
   >>> # Using filename
   >>> schedule = Z3DSchedule(fn="recording.z3d")
   >>> schedule.read_schedule()
   >>> print(f"Sampling rate: {schedule.SR} Hz")
   >>> print(f"Start time: {schedule.initial_start}")

   Read schedule from file object:

   >>> with open("recording.z3d", "rb") as fid:
   ...     schedule = Z3DSchedule(fid=fid)
   ...     schedule.read_schedule()
   ...     print(f"Date: {schedule.Date}, Time: {schedule.Time}")
   ...     print(f"GPS Sync: {schedule.Sync}")

   Access schedule attributes:

   >>> schedule = Z3DSchedule()
   >>> schedule.read_schedule(fn="recording.z3d")
   >>>
   >>> # Check recording configuration
   >>> if schedule.Sync == 'Y':
   ...     print("GPS synchronization enabled")
   >>> if schedule.Log == 'Y':
   ...     print("Data logging enabled")
   >>>
   >>> # Get numeric values (stored as strings)
   >>> sample_rate = float(schedule.SR) if schedule.SR else 0
   >>> gain_value = float(schedule.Gain) if schedule.Gain else 1.0


   .. py:attribute:: logger


   .. py:attribute:: fn
      :type:  str | pathlib.Path | None
      :value: None



   .. py:attribute:: fid
      :type:  BinaryIO | None
      :value: None



   .. py:attribute:: meta_string
      :type:  bytes | None
      :value: None



   .. py:attribute:: AutoGain
      :type:  str | None
      :value: None



   .. py:attribute:: Comment
      :type:  str | None
      :value: None



   .. py:attribute:: Date
      :type:  str | None
      :value: None



   .. py:attribute:: Duty
      :type:  str | None
      :value: None



   .. py:attribute:: FFTStacks
      :type:  str | None
      :value: None



   .. py:attribute:: Filename
      :type:  str | None
      :value: None



   .. py:attribute:: Gain
      :type:  str | None
      :value: None



   .. py:attribute:: Log
      :type:  str | None
      :value: None



   .. py:attribute:: NewFile
      :type:  str | None
      :value: None



   .. py:attribute:: Period
      :type:  str | None
      :value: None



   .. py:attribute:: RadioOn
      :type:  str | None
      :value: None



   .. py:attribute:: SR
      :type:  str | None
      :value: None



   .. py:attribute:: SamplesPerAcq
      :type:  str | None
      :value: None



   .. py:attribute:: Sleep
      :type:  str | None
      :value: None



   .. py:attribute:: Sync
      :type:  str | None
      :value: None



   .. py:attribute:: Time
      :type:  str | None
      :value: None



   .. py:attribute:: initial_start
      :type:  mt_metadata.common.MTime


   .. py:method:: read_schedule(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None) -> None

      Read and parse schedule metadata from Z3D file.

      Reads the schedule information section from a Z3D file starting at
      byte offset 512 (after the header) and parses key-value pairs to
      populate object attributes. Automatically creates an MTime object
      for the initial start time using GPS time correction.

      :param fn: Path to Z3D file to read. Overrides instance fn if provided.
                 Can be string path or pathlib.Path object.
      :type fn: str | pathlib.Path, optional
      :param fid: Open file object for reading Z3D file. Overrides instance fid if provided.
                  Must be opened in binary mode ('rb').
      :type fid: BinaryIO, optional

      :raises UnicodeDecodeError: If schedule metadata cannot be decoded as UTF-8 text.
      :raises IndexError: If schedule lines don't match expected "Schedule.key = value" format.
      :raises ValueError: If Date/Time values cannot be parsed into valid MTime object.

      .. rubric:: Notes

      The method performs the following steps:
      1. Determines file source (fn parameter, fid parameter, or instance attributes)
      2. Seeks to byte offset 512 (after Z3D header)
      3. Reads 512 bytes of schedule metadata
      4. Splits metadata into lines and parses key-value pairs
      5. Sets object attributes for each parsed schedule entry
      6. Creates MTime object from Date and Time with GPS correction

      Schedule entries must follow the format "Schedule.key = value".
      The "Schedule." prefix is removed and "/" characters in keys are stripped
      (e.g., "S/R" becomes "SR").

      If both Date and Time are present, they are combined into an MTime object
      with GPS time correction applied automatically.

      .. rubric:: Examples

      Read from file path:

      >>> schedule = Z3DSchedule()
      >>> schedule.read_schedule(fn="recording.z3d")
      >>> print(f"Sampling rate: {schedule.SR}")

      Read from open file object:

      >>> with open("recording.z3d", "rb") as fid:
      ...     schedule = Z3DSchedule()
      ...     schedule.read_schedule(fid=fid)
      ...     print(f"GPS sync: {schedule.Sync}")

      Read using instance attributes:

      >>> schedule = Z3DSchedule(fn="recording.z3d")
      >>> schedule.read_schedule()  # Uses instance fn
      >>> print(f"Start time: {schedule.initial_start}")



.. py:class:: Z3DMetadata(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None, **kwargs: Any)

   Read metadata information from a Z3D file and make each metadata entry an attribute.

   The attributes are left in capitalization of the Z3D file format.

   :param fn: Full path to Z3D file.
   :type fn: str | pathlib.Path, optional
   :param fid: File object (e.g., open(Z3Dfile, 'rb')).
   :type fid: BinaryIO, optional
   :param \*\*kwargs: Additional keyword arguments to set as attributes.
   :type \*\*kwargs: dict

   .. attribute:: _header_length

      Length of header in bits (512).

      :type: int

   .. attribute:: _metadata_length

      Length of metadata blocks (512).

      :type: int

   .. attribute:: _schedule_metadata_len

      Length of schedule meta data (512).

      :type: int

   .. attribute:: board_cal

      Board calibration array with frequency, rate, amplitude, phase.

      :type: np.ndarray | None

   .. attribute:: cal_ant

      Antenna calibration information.

      :type: str | None

   .. attribute:: cal_board

      Board calibration dictionary.

      :type: dict | None

   .. attribute:: cal_ver

      Calibration version.

      :type: str | None

   .. attribute:: ch_azimuth

      Channel azimuth.

      :type: str | None

   .. attribute:: ch_cmp

      Channel component.

      :type: str | None

   .. attribute:: ch_length

      Channel length (or number of coils).

      :type: str | None

   .. attribute:: ch_number

      Channel number on the ZEN board.

      :type: str | None

   .. attribute:: ch_xyz1

      Channel xyz location.

      :type: str | None

   .. attribute:: ch_xyz2

      Channel xyz location.

      :type: str | None

   .. attribute:: ch_cres

      Channel resistance.

      :type: str | None

   .. attribute:: coil_cal

      Coil calibration array (frequency, amplitude, phase).

      :type: np.ndarray | None

   .. attribute:: fid

      File object.

      :type: BinaryIO | None

   .. attribute:: find_metadata

      Boolean flag for finding metadata.

      :type: bool

   .. attribute:: fn

      Full path to Z3D file.

      :type: str | pathlib.Path | None

   .. attribute:: gdp_operator

      Operator of the survey.

      :type: str | None

   .. attribute:: gdp_progver

      Program version.

      :type: str | None

   .. attribute:: gdp_temp

      GDP temperature.

      :type: str | None

   .. attribute:: gdp_volt

      GDP voltage.

      :type: str | None

   .. attribute:: job_by

      Job performed by.

      :type: str | None

   .. attribute:: job_for

      Job for.

      :type: str | None

   .. attribute:: job_name

      Job name.

      :type: str | None

   .. attribute:: job_number

      Job number.

      :type: str | None

   .. attribute:: line_name

      Survey line name.

      :type: str | None

   .. attribute:: m_tell

      Location in the file where the last metadata block was found.

      :type: int

   .. attribute:: notes

      Additional notes from metadata.

      :type: str | None

   .. attribute:: rx_aspace

      Electrode spacing.

      :type: str | None

   .. attribute:: rx_sspace

      Receiver spacing.

      :type: str | None

   .. attribute:: rx_xazimuth

      X azimuth of electrode.

      :type: str | None

   .. attribute:: rx_xyz0

      Receiver xyz coordinates.

      :type: str | None

   .. attribute:: rx_yazimuth

      Y azimuth of electrode.

      :type: str | None

   .. attribute:: rx_zpositive

      Z positive direction (default 'down').

      :type: str

   .. attribute:: station

      Station name.

      :type: str | None

   .. attribute:: survey_type

      Type of survey.

      :type: str | None

   .. attribute:: unit_length

      Length units (m).

      :type: str | None

   .. attribute:: count

      Counter for metadata blocks read.

      :type: int

   .. rubric:: Examples

   >>> from mth5.io.zen import Z3DMetadata
   >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
   >>> header_obj = Z3DMetadata(fn=Z3Dfn)
   >>> header_obj.read_metadata()


   .. py:attribute:: logger


   .. py:attribute:: fn
      :type:  str | pathlib.Path | None
      :value: None



   .. py:attribute:: fid
      :type:  BinaryIO | None
      :value: None



   .. py:attribute:: find_metadata
      :type:  bool
      :value: True



   .. py:attribute:: board_cal
      :type:  list | numpy.ndarray | None
      :value: None



   .. py:attribute:: coil_cal
      :type:  list | numpy.ndarray | None
      :value: None



   .. py:attribute:: m_tell
      :type:  int
      :value: 0



   .. py:attribute:: cal_ant
      :type:  str | None
      :value: None



   .. py:attribute:: cal_board
      :type:  dict[str, Any] | None
      :value: None



   .. py:attribute:: cal_ver
      :type:  str | None
      :value: None



   .. py:attribute:: ch_azimuth
      :type:  str | None
      :value: None



   .. py:attribute:: ch_cmp
      :type:  str | None
      :value: None



   .. py:attribute:: ch_length
      :type:  str | None
      :value: None



   .. py:attribute:: ch_number
      :type:  str | None
      :value: None



   .. py:attribute:: ch_xyz1
      :type:  str | None
      :value: None



   .. py:attribute:: ch_xyz2
      :type:  str | None
      :value: None



   .. py:attribute:: ch_cres
      :type:  str | None
      :value: None



   .. py:attribute:: gdp_operator
      :type:  str | None
      :value: None



   .. py:attribute:: gdp_progver
      :type:  str | None
      :value: None



   .. py:attribute:: gdp_volt
      :type:  str | None
      :value: None



   .. py:attribute:: gdp_temp
      :type:  str | None
      :value: None



   .. py:attribute:: job_by
      :type:  str | None
      :value: None



   .. py:attribute:: job_for
      :type:  str | None
      :value: None



   .. py:attribute:: job_name
      :type:  str | None
      :value: None



   .. py:attribute:: job_number
      :type:  str | None
      :value: None



   .. py:attribute:: rx_aspace
      :type:  str | None
      :value: None



   .. py:attribute:: rx_sspace
      :type:  str | None
      :value: None



   .. py:attribute:: rx_xazimuth
      :type:  str | None
      :value: None



   .. py:attribute:: rx_xyz0
      :type:  str | None
      :value: None



   .. py:attribute:: rx_yazimuth
      :type:  str | None
      :value: None



   .. py:attribute:: rx_zpositive
      :type:  str
      :value: 'down'



   .. py:attribute:: line_name
      :type:  str | None
      :value: None



   .. py:attribute:: survey_type
      :type:  str | None
      :value: None



   .. py:attribute:: unit_length
      :type:  str | None
      :value: None



   .. py:attribute:: station
      :type:  str | None
      :value: None



   .. py:attribute:: count
      :type:  int
      :value: 0



   .. py:attribute:: notes
      :type:  str | None
      :value: None



   .. py:method:: read_metadata(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None) -> None

      Read metadata from Z3D file.

      Parses the metadata blocks in a Z3D file and populates the object's
      attributes with the extracted values. Also reads calibration data
      for both board and coil calibrations.

      :param fn: Full path to file. If None, uses the instance's fn attribute.
      :type fn: str | pathlib.Path, optional
      :param fid: Open file object. If None, uses the instance's fid attribute or
                  opens the file specified by fn.
      :type fid: BinaryIO, optional

      :raises UnicodeDecodeError: If metadata blocks cannot be decoded as text.

      .. rubric:: Notes

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



.. py:class:: Z3D(fn: str | pathlib.Path | None = None, **kwargs: Any)

   A class for reading and processing Z3D files output by Zen data loggers.

   This class handles the parsing of Z3D binary files which contain GPS-stamped
   time series data from magnetotelluric measurements. It provides methods for
   reading file headers, metadata, schedule information, and time series data,
   as well as converting between different units and formats.

   :param fn: Full path to the .Z3D file to be read. Default is None.
   :type fn: str or Path, optional
   :param \*\*kwargs: Additional keyword arguments including:
                      - stamp_len : int, default 64
                          GPS stamp length in bits
   :type \*\*kwargs: dict

   .. attribute:: fn

      Path to the Z3D file

      :type: Path or None

   .. attribute:: calibration_fn

      Path to calibration file

      :type: str or None

   .. attribute:: header

      Header information object

      :type: Z3DHeader

   .. attribute:: schedule

      Schedule information object

      :type: Z3DSchedule

   .. attribute:: metadata

      Metadata information object

      :type: Z3DMetadata

   .. attribute:: gps_stamps

      Array of GPS time stamps

      :type: numpy.ndarray or None

   .. attribute:: time_series

      Time series data array

      :type: numpy.ndarray or None

   .. attribute:: sample_rate

      Data sampling rate in Hz

      :type: float or None

   .. attribute:: units

      Data units, default 'counts'

      :type: str

   .. rubric:: Notes

   GPS data type is formatted as::

       numpy.dtype([('flag0', numpy.int32),
                    ('flag1', numpy.int32),
                    ('time', numpy.int32),
                    ('lat', numpy.float64),
                    ('lon', numpy.float64),
                    ('num_sat', numpy.int32),
                    ('gps_sens', numpy.int32),
                    ('temperature', numpy.float32),
                    ('voltage', numpy.float32),
                    ('num_fpga', numpy.int32),
                    ('num_adc', numpy.int32),
                    ('pps_count', numpy.int32),
                    ('dac_tune', numpy.int32),
                    ('block_len', numpy.int32)])

   .. rubric:: Examples

   >>> from mth5.io.zen import Z3D
   >>> z3d = Z3D(r"/path/to/data/station_20150522_080000_256_EX.Z3D")
   >>> z3d.read_z3d()
   >>> print(f"Found {z3d.gps_stamps.shape[0]} GPS time stamps")
   >>> print(f"Found {z3d.time_series.size} data points")


   .. py:attribute:: logger


   .. py:property:: fn
      :type: pathlib.Path | None


      Get the Z3D file path.

      :returns: Path to the Z3D file, or None if not set.
      :rtype: Path or None


   .. py:attribute:: calibration_fn
      :value: None



   .. py:attribute:: header


   .. py:attribute:: schedule


   .. py:attribute:: metadata


   .. py:attribute:: gps_stamps
      :value: None



   .. py:attribute:: gps_flag


   .. py:attribute:: num_sec_to_skip
      :value: 1



   .. py:attribute:: units
      :value: 'digital counts'



   .. py:property:: sample_rate
      :type: float | None


      Get the sampling rate in Hz.

      :returns: Data sampling rate in samples per second, or None if not available.
      :rtype: float or None


   .. py:attribute:: time_series
      :value: None



   .. py:attribute:: ch_dict


   .. py:property:: file_size
      :type: int


      Get the size of the Z3D file in bytes.

      :returns: File size in bytes, or 0 if no file is set.
      :rtype: int


   .. py:property:: n_samples
      :type: int


      Get the number of data samples in the file.

      :returns: Number of data samples. Calculated from file size if time_series
                is not loaded, otherwise returns the actual array size.
      :rtype: int

      .. rubric:: Notes

      Calculation assumes 4 bytes per sample and accounts for metadata blocks.
      If sample_rate is available, adds buffer for GPS stamps.


   .. py:property:: station
      :type: str | None


      Get the station name.

      :returns: Station identifier name.
      :rtype: str or None


   .. py:property:: dipole_length
      :type: float


      Get the dipole length for electric field measurements.

      :returns: Dipole length in meters. Calculated from electrode positions
                if not directly specified in metadata. Returns 0 for magnetic
                channels or if positions are not available.
      :rtype: float

      .. rubric:: Notes

      Length is calculated from xyz coordinates using Euclidean distance
      formula when position data is available in metadata.


   .. py:property:: azimuth
      :type: float | None


      Get the azimuth of instrument setup.

      :returns: Azimuth angle in degrees from north, or None if not available.
      :rtype: float or None


   .. py:property:: component
      :type: str


      Get the channel component identifier.

      :returns: Channel component name in lowercase (e.g., 'ex', 'hy', 'hz').
      :rtype: str


   .. py:property:: latitude
      :type: float | None


      Get the latitude in decimal degrees.

      :returns: Latitude coordinate in decimal degrees, or None if not available.
      :rtype: float or None


   .. py:property:: longitude
      :type: float | None


      Get the longitude in decimal degrees.

      :returns: Longitude coordinate in decimal degrees, or None if not available.
      :rtype: float or None


   .. py:property:: elevation
      :type: float | None


      Get the elevation in meters.

      :returns: Elevation above sea level in meters, or None if not available.
      :rtype: float or None


   .. py:property:: start
      :type: mt_metadata.common.mttime.MTime


      Get the start time of the data.

      :returns: Start time from GPS stamps if available, otherwise scheduled time.
      :rtype: MTime


   .. py:property:: end
      :type: mt_metadata.common.mttime.MTime | float


      Get the end time of the data.

      :returns: End time from GPS stamps if available, otherwise calculated
                from start time and number of samples.
      :rtype: MTime or float


   .. py:property:: zen_schedule
      :type: mt_metadata.common.mttime.MTime


      Get the zen schedule date and time.

      :returns: Scheduled start time from header or schedule object.
      :rtype: MTime


   .. py:property:: coil_number
      :type: str | None


      Get the coil number identifier.

      :returns: Coil antenna number identifier, or None if not available.
      :rtype: str or None


   .. py:property:: channel_number
      :type: int


      Get the channel number.

      :returns: Channel number identifier. Maps component names to standard
                channel numbers or uses metadata channel number.
      :rtype: int


   .. py:property:: channel_metadata

      Generate Channel metadata object from Z3D file information.

      Creates either an Electric or Magnetic metadata object based on the
      component type, populated with channel-specific parameters, sensor
      information, and data statistics.

      :returns: Channel metadata object appropriate for the measurement type:
                - Electric: includes dipole length, AC/DC statistics
                - Magnetic: includes sensor details, field min/max values
      :rtype: Electric or Magnetic

      .. rubric:: Notes

      Electric channels (ex, ey) get dipole length and voltage statistics.
      Magnetic channels (hx, hy, hz) get sensor information and field
      strength statistics computed from the first and last seconds of data.

      .. rubric:: Examples

      >>> z3d = Z3D("/path/to/file.Z3D")
      >>> z3d.read_z3d()
      >>> ch_meta = z3d.channel_metadata
      >>> print(f"Channel component: {ch_meta.component}")


   .. py:property:: station_metadata

      Generate Station metadata object from Z3D file information.

      Creates a Station metadata object populated with location and timing
      information extracted from the Z3D file header and metadata.

      :returns: Station metadata object with populated fields including station ID,
                coordinates, elevation, time period, and operator information.
      :rtype: Station

      .. rubric:: Examples

      >>> z3d = Z3D("/path/to/file.Z3D")
      >>> z3d.read_all_info()
      >>> station_meta = z3d.station_metadata
      >>> print(station_meta.id)


   .. py:property:: run_metadata

      Generate Run metadata object from Z3D file information.

      Creates a Run metadata object populated with data logger information,
      timing details, and measurement parameters extracted from the Z3D file.

      :returns: Run metadata object with populated fields including data logger
                details, sample rate, time period, and data type information.
      :rtype: Run

      .. rubric:: Examples

      >>> z3d = Z3D("/path/to/file.Z3D")
      >>> z3d.read_all_info()
      >>> run_meta = z3d.run_metadata
      >>> print(f"Sample rate: {run_meta.sample_rate}")


   .. py:property:: counts2mv_filter

      Create a counts to milliVolt coefficient filter.

      Generate a coefficient filter for converting digital counts to milliVolt
      using the channel factor from the Z3D file header.

      :returns: Filter object configured for counts to milliVolt conversion with
                gain set to the inverse of the channel factor.
      :rtype: CoefficientFilter

      .. rubric:: Notes

      The gain is set to 1/channel_factor because this represents the
      inverse operation when the instrument response has been divided
      from the data during processing.

      .. rubric:: Examples

      >>> z3d = Z3D("/path/to/file.Z3D")
      >>> z3d.read_all_info()
      >>> filter_obj = z3d.counts2mv_filter
      >>> print(f"Conversion gain: {filter_obj.gain}")


   .. py:property:: coil_response

      Make the coile response into a FAP filter

      Phase must be in radians


   .. py:property:: zen_response

      Zen response, not sure the full calibration comes directly from the
      Z3D file, so skipping for now.  Will have to read a Zen##.cal file
      to get the full calibration.  This shouldn't be a big issue cause it
      should roughly be the same for all channels and since the TF is
      computing the ratio they will cancel out.  Though we should look
      more into this if just looking at calibrate time series.


   .. py:property:: channel_response

      Generate comprehensive channel response for the Z3D data.

      Creates a ChannelResponse object containing all applicable filters
      including coil response, dipole conversion, and counts-to-milliVolt
      transformation.

      :returns: Channel response object with appropriate filter chain for
                converting raw Z3D data to physical units.
      :rtype: ChannelResponse

      .. rubric:: Notes

      The filter chain includes:
      - Coil response (for magnetic channels) or dipole filter (for electric)
      - Counts to milliVolt conversion filter


   .. py:property:: dipole_filter

      Create dipole conversion filter for electric field measurements.

      Generate a coefficient filter for converting electric field measurements
      from milliVolt per kilometer to milliVolt using the dipole length.

      :returns: Filter object for dipole conversion if dipole_length is non-zero,
                None otherwise.
      :rtype: CoefficientFilter or None

      .. rubric:: Notes

      The gain is set to dipole_length/1000 to convert from mV/km to mV.
      This represents the physical dipole length scaling for electric
      field measurements.

      .. rubric:: Examples

      >>> z3d = Z3D("/path/to/electric.Z3D")
      >>> z3d.read_all_info()
      >>> if z3d.dipole_filter is not None:
      ...     print(f"Dipole length: {z3d.dipole_length} m")


   .. py:method:: read_all_info() -> None

      Read header, schedule, and metadata from Z3D file.

      Convenience method to read all file information in one call.
      Opens the file once and reads all sections sequentially.

      :raises FileNotFoundError: If the Z3D file does not exist.



   .. py:method:: read_z3d(z3d_fn: str | pathlib.Path | None = None) -> None

      Read and parse Z3D file data.

      Comprehensive method to read a Z3D file and populate all object attributes.
      Performs the following operations:

      1. Read file as chunks of 32-bit integers
      2. Extract and validate GPS stamps
      3. Check GPS time stamp consistency (1 second intervals)
      4. Verify data block lengths match sampling rate
      5. Convert GPS time to seconds relative to GPS week
      6. Skip initial buffered data (first 2 seconds)
      7. Populate time series array with non-zero data

      :param z3d_fn: Path to Z3D file to read. If None, uses current fn attribute.
      :type z3d_fn: str, Path, or None, optional

      :raises ZenGPSError: If data is too short or GPS timing issues prevent parsing.

      .. rubric:: Examples

      >>> z3d = Z3D(r"/path/to/data/station_20150522_080000_256_EX.Z3D")
      >>> z3d.read_z3d()
      >>> print(f"Read {z3d.time_series.size} data points")



   .. py:method:: get_gps_stamp_index(ts_data: numpy.ndarray, old_version: bool = False) -> list[int]

      Locate GPS time stamp indices in time series data.

      Searches for GPS flag patterns in the data array. For newer files,
      verifies that flag_1 follows flag_0.

      :param ts_data: Time series data array containing GPS stamps.
      :type ts_data: numpy.ndarray
      :param old_version: If True, only searches for single GPS flag (old format).
                          If False, validates flag pairs (new format).
      :type old_version: bool, default False

      :returns: List of indices where GPS stamps are located.
      :rtype: list of int



   .. py:method:: trim_data() -> None

      Trim the first 2 seconds of data due to SD buffer issues.

      Remove the first 2 GPS stamps and corresponding time series data
      to account for SD card buffering artifacts in early data.

      .. rubric:: Notes

      This method may be deprecated after field testing confirms
      the buffer behavior is consistent across all instruments.

      .. deprecated::
         This method will be deprecated after field testing.



   .. py:method:: check_start_time() -> mt_metadata.common.mttime.MTime

      Validate scheduled start time against first GPS stamp.

      Compare the scheduled start time from the file header with
      the actual first GPS timestamp to identify timing discrepancies.

      :returns: UTC start time from the first valid GPS stamp.
      :rtype: MTime

      .. rubric:: Notes

      Logs warnings if the difference exceeds the maximum allowed
      time difference (default 20 seconds).



   .. py:method:: validate_gps_time() -> bool

      Validate that GPS time stamps are consistently 1 second apart.

      :returns: True if all GPS stamps are properly spaced, False otherwise.
      :rtype: bool

      .. rubric:: Notes

      Logs debug information for any stamps that are more than 1 second apart.



   .. py:method:: validate_time_blocks() -> bool

      Validate GPS time stamps and verify data block lengths.

      Check that each GPS stamp block contains the expected number
      of data points (should equal sample rate for 1-second blocks).

      :returns: True if all blocks have correct length, False otherwise.
      :rtype: bool

      .. rubric:: Notes

      If bad blocks are detected near the beginning (index < 5),
      this method will automatically skip those blocks and trim
      the time series data accordingly.



   .. py:method:: convert_gps_time() -> None

      Convert GPS time integers to floating point seconds.

      Transform GPS time from integer format to float and convert
      from GPS time units to seconds relative to the GPS week.

      .. rubric:: Notes

      GPS time is initially stored as integers in units of 1/1024 seconds.
      This method converts to floating point seconds and applies the
      necessary scaling factors.



   .. py:method:: convert_counts_to_mv(data: numpy.ndarray) -> numpy.ndarray

      Convert time series data from counts to milliVolt.

      :param data: Time series data in digital counts.
      :type data: numpy.ndarray

      :returns: Time series data converted to milliVolt.
      :rtype: numpy.ndarray



   .. py:method:: convert_mv_to_counts(data: numpy.ndarray) -> numpy.ndarray

      Convert time series data from milliVolt to counts.

      :param data: Time series data in milliVolt.
      :type data: numpy.ndarray

      :returns: Time series data converted to digital counts.
      :rtype: numpy.ndarray

      .. rubric:: Notes

      Assumes no other scaling has been applied to the data.



   .. py:method:: get_gps_time(gps_int: int, gps_week: int = 0) -> tuple[float, int]

      Convert GPS integer timestamp to seconds and GPS week.

      :param gps_int: Integer from the GPS time stamp line.
      :type gps_int: int
      :param gps_week: Relative GPS week. If seconds exceed one week, this is incremented.
      :type gps_week: int, default 0

      :returns: GPS time in seconds from beginning of GPS week, and updated GPS week.
      :rtype: tuple[float, int]

      .. rubric:: Notes

      GPS integers are in units of 1/1024 seconds. This method handles
      week rollovers when seconds exceed 604800.



   .. py:method:: get_UTC_date_time(gps_week: int, gps_time: float) -> mt_metadata.common.mttime.MTime

      Convert GPS week and time to UTC datetime.

      Calculate the actual UTC date and time of measurement from
      GPS week number and seconds within that week.

      :param gps_week: GPS week number when data was collected.
      :type gps_week: int
      :param gps_time: Number of seconds from beginning of GPS week.
      :type gps_time: float

      :returns: UTC datetime object for the measurement time.
      :rtype: MTime

      .. rubric:: Notes

      Automatically handles GPS time rollover when seconds exceed
      one week (604800 seconds).



   .. py:method:: to_channelts() -> mth5.timeseries.ChannelTS

      Convert Z3D data to ChannelTS time series object.

      Create a ChannelTS object populated with the time series data
      and all associated metadata from the Z3D file.

      :returns: Time series object with data, metadata, and instrument response.
      :rtype: ChannelTS



.. py:function:: read_z3d(fn: str | pathlib.Path, calibration_fn: str | pathlib.Path | None = None, logger_file_handler: Any | None = None) -> mth5.timeseries.ChannelTS | None

   Read a Z3D file and return a ChannelTS object.

   Convenience function to read Z3D files with error handling.

   :param fn: Path to the Z3D file to read.
   :type fn: str or Path
   :param calibration_fn: Path to calibration file. Default is None.
   :type calibration_fn: str, Path, or None, optional
   :param logger_file_handler: Logger file handler to add to Z3D logger. Default is None.
   :type logger_file_handler: optional

   :returns: Time series object if successful, None if GPS timing errors occur.
   :rtype: ChannelTS or None

   .. rubric:: Examples

   >>> ts = read_z3d("/path/to/data/station_EX.Z3D")
   >>> if ts is not None:
   ...     print(f"Read {ts.n_samples} samples")


.. py:class:: CoilResponse(calibration_file: str | pathlib.Path | None = None, angular_frequency: bool = False)

   Read ANT4 coil calibration files from Zonge (``amtant.cal``).

   This class parses a Zonge antenna calibration file and exposes a
   :class:`mt_metadata.timeseries.filters.FrequencyResponseTableFilter` for a
   specified coil number.

   :param calibration_file: Path to the antenna calibration file. If provided the file will be
                            read during initialization, by default None.
   :type calibration_file: str | Path | None, optional
   :param angular_frequency: If True, reported frequencies will be converted to angular frequency
                             (rad/s), by default False.
   :type angular_frequency: bool, optional

   .. attribute:: coil_calibrations

      Mapping of coil serial numbers to a structured numpy array containing
      frequency, amplitude, and phase columns.

      :type: dict[str, numpy.ndarray]

   .. rubric:: Examples

   >>> from mth5.mth5.io.zen.coil_response import CoilResponse
   >>> cr = CoilResponse('amtant.cal')
   >>> fap = cr.get_coil_response_fap(1234)
   >>> print(fap.name)


   .. py:attribute:: logger


   .. py:attribute:: coil_calibrations
      :type:  dict[str, numpy.ndarray]


   .. py:property:: calibration_file


   .. py:attribute:: angular_frequency
      :type:  bool
      :value: False



   .. py:method:: file_exists() -> bool

      Check to make sure the file exists

      :returns: True if the file exists, False if it does not
      :rtype: bool



   .. py:method:: read_antenna_file(antenna_calibration_file: str | pathlib.Path | None = None) -> None

      Read a Zonge antenna calibration file and parse coil responses.

      The expected file format contains blocks starting with an "antenna"
      header line that provides the base frequency followed by lines with
      coil serial number and amplitude/phase values for the 6th and 8th
      harmonics.

      :param antenna_calibration_file: Optional path to the antenna calibration file. If provided, it
                                       overrides the instance ``calibration_file``.
      :type antenna_calibration_file: str | Path | None, optional

      .. rubric:: Notes

      Phase values in the file are expected in milliradians and are
      converted to radians.



   .. py:method:: get_coil_response_fap(coil_number: int | str, extrapolate: bool = True) -> mt_metadata.timeseries.filters.FrequencyResponseTableFilter

      Read an amtant.cal file provided by Zonge.


      Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
      is a fancy way of saying f * 6 and f * 8.

      :param coil_number: ANT4 4 digit serial number
      :type coil_number: int or str
      :param extrapolate: If True, extrapolate the frequency response to low and high frequencies,
                          by default True
      :type extrapolate: bool, optional

      :returns: Frequency look up table for the specified coil number.
      :rtype: FrequencyResponseTableFilter

      :raises KeyError: If the coil number is not found in the calibration file.

      .. rubric:: Notes

      Ensure that the antenna calibration file has been read prior to calling
      this method. This can be done by providing the calibration file during
      initialization or by calling :meth:`read_antenna_file`.



   .. py:method:: extrapolate(fap: mt_metadata.timeseries.filters.FrequencyResponseTableFilter) -> mt_metadata.timeseries.filters.FrequencyResponseTableFilter

      Extrapolate a frequency/amplitude/phase table using log-linear pads.

      :param fap: Frequency response object to extrapolate.
      :type fap: FrequencyResponseTableFilter

      :returns: A copy of ``fap`` with low- and high-frequency extrapolated
                values appended.
      :rtype: FrequencyResponseTableFilter



   .. py:method:: has_coil_number(coil_number: int | str | None) -> bool

      Test if coil number is in the antenna file

      :param coil_number: ANT4 serial number
      :type coil_number: int or str or None

      :returns: True if the coil is found, False if it is not
      :rtype: bool



.. py:class:: Z3DCollection(file_path: str | pathlib.Path | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.io.collection.Collection`


   Collection manager for Z3D file operations and metadata processing.

   This class provides functionality to handle collections of Z3D files,
   including metadata extraction, station information management, and
   dataframe creation for analysis workflows.

   :param file_path: Path to directory containing Z3D files, by default None
   :type file_path: str or Path, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent Collection class
   :type \*\*kwargs: dict

   .. attribute:: station_metadata_dict

      Dictionary mapping station IDs to Station metadata objects

      :type: dict[str, Station]

   .. attribute:: file_ext

      File extension for Z3D files ("z3d")

      :type: str

   .. rubric:: Examples

   >>> zc = Z3DCollection("/path/to/z3d/files")
   >>> df = zc.to_dataframe(sample_rates=[256, 4096])
   >>> print(df.head())


   .. py:attribute:: station_metadata_dict
      :type:  dict[str, mt_metadata.timeseries.Station]


   .. py:attribute:: file_ext
      :type:  str
      :value: 'z3d'



   .. py:method:: get_calibrations(antenna_calibration_file: str | pathlib.Path) -> mth5.io.zen.coil_response.CoilResponse

      Load coil calibration data from antenna calibration file.

      :param antenna_calibration_file: Path to the antenna.cal file containing coil calibration data
      :type antenna_calibration_file: str or Path

      :returns: CoilResponse object containing calibration information for
                various coil serial numbers
      :rtype: CoilResponse

      .. rubric:: Examples

      >>> zc = Z3DCollection("/path/to/z3d/files")
      >>> cal_obj = zc.get_calibrations("/path/to/antenna.cal")
      >>> print(cal_obj.has_coil_number("2324"))



   .. py:method:: to_dataframe(sample_rates: list[int] = [256, 4096], run_name_zeros: int = 4, calibration_path: str | pathlib.Path | None = None) -> pandas.DataFrame

      Extract Z3D file information and create analysis-ready dataframe.

      Processes all Z3D files in the collection, extracting metadata and
      file information to create a comprehensive dataframe suitable for
      magnetotelluric data analysis workflows.

      :param sample_rates: Allowed sampling rates in Hz. Files with sample rates not in
                           this list will trigger a warning and early return
      :type sample_rates: list of int, default [256, 4096]
      :param run_name_zeros: Number of zero-padding digits for run names in dataframe sorting
      :type run_name_zeros: int, default 4
      :param calibration_path: Path to antenna calibration file. If None, calibration information
                               will not be included, by default None
      :type calibration_path: str or Path, optional

      :returns: Dataframe containing Z3D file information with columns:
                - survey: Survey/job name from Z3D metadata
                - station: Station identifier
                - run: Automatically assigned run names based on start times
                - start/end: ISO format timestamps for data recording period
                - channel_id: Channel number from Z3D file
                - component: Measurement component (ex, ey, hx, hy, hz)
                - fn: Path to Z3D file
                - sample_rate: Sampling frequency in Hz
                - file_size: Size of Z3D file in bytes
                - n_samples: Number of data samples in file
                - sequence_number: Sequential numbering within station
                - dipole: Dipole length in meters (for electric channels)
                - coil_number: Coil serial number (for magnetic channels)
                - latitude/longitude/elevation: Station coordinates
                - instrument_id: ZEN box identifier
                - calibration_fn: Path to calibration file if available
      :rtype: pd.DataFrame

      :raises AttributeError: If Z3D files contain invalid or missing required metadata
      :raises FileNotFoundError: If calibration_path is specified but file doesn't exist

      .. rubric:: Examples

      >>> zc = Z3DCollection("/path/to/z3d/files")
      >>> df = zc.to_dataframe(sample_rates=[256, 4096],
      ...                      calibration_path="/path/to/antenna.cal")
      >>> print(df[['station', 'component', 'sample_rate']].head())
      >>> df.to_csv("/path/output/z3d_inventory.csv")

      .. rubric:: Notes

      This method also populates the `station_metadata_dict` attribute
      with consolidated station metadata derived from all processed files.



   .. py:method:: assign_run_names(df: pandas.DataFrame, zeros: int = 3) -> pandas.DataFrame

      Assign standardized run names to dataframe based on start times.

      Creates run names using the pattern 'sr{sample_rate}_{block_number}'
      where block_number is assigned sequentially based on unique start
      times within each station.

      :param df: Input dataframe containing Z3D file information with at least
                 'station', 'start', and 'sample_rate' columns
      :type df: pd.DataFrame
      :param zeros: Number of zero-padding digits for block numbers in run names
      :type zeros: int, default 3

      :returns: Modified dataframe with updated 'run' and 'sequence_number'
                columns assigned based on temporal ordering within each station
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> zc = Z3DCollection()
      >>> df = pd.DataFrame({
      ...     'station': ['001', '001', '002'],
      ...     'start': ['2022-01-01T10:00:00', '2022-01-01T12:00:00', '2022-01-01T10:00:00'],
      ...     'sample_rate': [256, 256, 4096]
      ... })
      >>> df_with_runs = zc.assign_run_names(df, zeros=3)
      >>> print(df_with_runs['run'].tolist())
      ['sr256_001', 'sr256_002', 'sr4096_001']

      .. rubric:: Notes

      This method modifies the input dataframe in-place by updating the
      'run' and 'sequence_number' columns. Start times are used to
      determine temporal ordering within each station.



