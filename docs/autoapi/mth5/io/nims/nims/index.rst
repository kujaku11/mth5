mth5.io.nims.nims
=================

.. py:module:: mth5.io.nims.nims

.. autoapi-nested-parse::

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



Classes
-------

.. autoapisummary::

   mth5.io.nims.nims.NIMS


Functions
---------

.. autoapisummary::

   mth5.io.nims.nims.read_nims


Module Contents
---------------

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


