mth5.io.zen.zen
===============

.. py:module:: mth5.io.zen.zen

.. autoapi-nested-parse::

   ====================
   Zen
   ====================

       * Tools for reading and writing files for Zen and processing software
       * Tools for copying data from SD cards
       * Tools for copying schedules to SD cards

   Created on Tue Jun 11 10:53:23 2013
   Updated August 2020 (JP)

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license:
       MIT



Exceptions
----------

.. autoapisummary::

   mth5.io.zen.zen.ZenGPSError
   mth5.io.zen.zen.ZenSamplingRateError
   mth5.io.zen.zen.ZenInputFileError


Classes
-------

.. autoapisummary::

   mth5.io.zen.zen.Z3D


Functions
---------

.. autoapisummary::

   mth5.io.zen.zen.read_z3d


Module Contents
---------------

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



.. py:exception:: ZenGPSError

   Bases: :py:obj:`Exception`


   Exception raised for GPS timing errors in Z3D files.


.. py:exception:: ZenSamplingRateError

   Bases: :py:obj:`Exception`


   Exception raised for sampling rate inconsistencies.


.. py:exception:: ZenInputFileError

   Bases: :py:obj:`Exception`


   Exception raised for Z3D file input/reading errors.


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


