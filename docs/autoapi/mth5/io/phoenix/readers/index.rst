mth5.io.phoenix.readers
=======================

.. py:module:: mth5.io.phoenix.readers


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/phoenix/readers/base/index
   /autoapi/mth5/io/phoenix/readers/calibrations/index
   /autoapi/mth5/io/phoenix/readers/config/index
   /autoapi/mth5/io/phoenix/readers/contiguous/index
   /autoapi/mth5/io/phoenix/readers/header/index
   /autoapi/mth5/io/phoenix/readers/helpers/index
   /autoapi/mth5/io/phoenix/readers/mtu/index
   /autoapi/mth5/io/phoenix/readers/native/index
   /autoapi/mth5/io/phoenix/readers/receiver_metadata/index
   /autoapi/mth5/io/phoenix/readers/segmented/index


Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.Header
   mth5.io.phoenix.readers.PhoenixCalibration
   mth5.io.phoenix.readers.TSReaderBase
   mth5.io.phoenix.readers.NativeReader
   mth5.io.phoenix.readers.DecimatedSegmentedReader
   mth5.io.phoenix.readers.DecimatedContinuousReader
   mth5.io.phoenix.readers.PhoenixConfig
   mth5.io.phoenix.readers.PhoenixReceiverMetadata
   mth5.io.phoenix.readers.MTUTable
   mth5.io.phoenix.readers.MTUTSN


Package Contents
----------------

.. py:class:: Header(**kwargs: Any)

   Phoenix Geophysics MTU-5C binary header reader and parser.

   This class reads and parses the 128-byte binary header from Phoenix
   Geophysics MTU-5C data files. The header contains instrument configuration,
   GPS location, timing information, and recording parameters essential for
   proper data interpretation.

   The header format is fixed at 128 bytes and contains information about:
   - Instrument type and serial number
   - Recording parameters (sample rate, channel configuration)
   - GPS location and timing information
   - Hardware configuration and gain settings
   - Data quality metrics (saturated/missing frames)

   :param \*\*kwargs: Additional keyword arguments to set as instance attributes.
   :type \*\*kwargs: Any

   .. attribute:: logger

      Logger instance for debugging and error reporting.

      :type: loguru.Logger

   .. attribute:: report_hw_sat

      Flag to control hardware saturation reporting.

      :type: bool, default False

   .. attribute:: header_length

      Length of the binary header in bytes.

      :type: int, default 128

   .. attribute:: ad_plus_minus_range

      Differential voltage range of the A/D converter (board dependent).

      :type: float, default 5.0

   .. attribute:: channel_map

      Mapping from channel IDs to channel names.

      :type: dict[int, str]

   .. attribute:: channel_azimuths

      Mapping from channel names to azimuth angles in degrees.

      :type: dict[str, int]

   .. rubric:: Examples

   >>> with open("phoenix_data.bin", "rb") as f:
   ...     header = Header()
   ...     header.unpack_header(f)
   ...     print(f"Sample rate: {header.sample_rate}")
   ...     print(f"GPS location: {header.gps_lat}, {header.gps_long}")


   .. py:attribute:: logger
      :type:  loguru.Logger


   .. py:attribute:: report_hw_sat
      :type:  bool
      :value: False



   .. py:property:: header_length
      :type: int


      Length of the header in bytes.

      :returns: Header length in bytes.
      :rtype: int


   .. py:attribute:: ad_plus_minus_range
      :type:  float
      :value: 5.0



   .. py:attribute:: channel_map
      :type:  dict[int, str]


   .. py:attribute:: channel_azimuths
      :type:  dict[str, int]


   .. py:property:: file_type
      :type: int | None


      File type indicator from binary header.

      :returns: File type identifier, or None if no header is loaded.
      :rtype: int or None


   .. py:property:: file_version
      :type: int | None


      File version from binary header.

      :returns: File version identifier, or None if no header is loaded.
      :rtype: int or None


   .. py:property:: instrument_type
      :type: str | None


      Instrument type string from binary header.

      :returns: Cleaned instrument type string, or None if no header is loaded.
      :rtype: str or None


   .. py:property:: instrument_serial_number
      :type: str | None


      Instrument serial number from binary header.

      :returns: Decoded instrument serial number, or None if no header is loaded.
      :rtype: str or None


   .. py:property:: recording_id
      :type: int | None


      Recording identifier from binary header or cached value.

      :returns: Recording ID as integer, or None if not available.
      :rtype: int or None


   .. py:property:: recording_start_time
      :type: mt_metadata.common.mttime.MTime | None


      Recording start time from GPS timestamp.

      The actual data recording starts 1 second after the set start time.
      This is caused by the data logger starting up and initializing filter.
      This is taken care of in the segment start time.

      See https://github.com/kujaku11/PhoenixGeoPy/tree/main/Docs for more
      information.

      The time recorded is GPS time.

      :returns: GPS start time, or None if recording ID is not available.
      :rtype: MTime or None


   .. py:property:: channel_id
      :type: int | None


      Channel identifier from binary header or cached value.

      :returns: Channel ID, or None if not available.
      :rtype: int or None


   .. py:property:: file_sequence
      :type: int | None


      File sequence number from binary header.

      :returns: File sequence number, or None if no header is loaded.
      :rtype: int or None


   .. py:property:: frag_period
      :type: int | None


      Fragment period from binary header.

      :returns: Fragment period, or None if no header is loaded.
      :rtype: int or None


   .. py:property:: ch_board_model
      :type: str | None


      Channel board model string from binary header.

      :returns: Board model string, or None if no header is loaded.
      :rtype: str or None


   .. py:property:: board_model_main
      :type: str | None


      Main board model identifier.

      :returns: Main board model (first 5 characters), or None if not available.
      :rtype: str or None


   .. py:property:: board_model_revision
      :type: str | None


      Board model revision identifier.

      :returns: Board revision (character 6), or None if not available.
      :rtype: str or None


   .. py:property:: ch_board_serial
      :type: int


      Channel board serial number from binary header.

      :returns: Board serial number as integer, or 0 if not available or invalid.
      :rtype: int


   .. py:property:: ch_firmware
      :type: int | None


      Channel firmware version from binary header.

      :returns: Firmware version, or None if no header is loaded.
      :rtype: int or None


   .. py:property:: hardware_configuration
      :type: tuple[Any, Ellipsis] | None


      Hardware configuration bytes from binary header.

      :returns: Hardware configuration data, or None if no header is loaded.
      :rtype: tuple of Any or None


   .. py:property:: channel_type
      :type: str | None


      Channel type determined from hardware configuration.

      :returns: 'E' for electric, 'H' for magnetic, or None if no header.
      :rtype: str or None


   .. py:property:: detected_channel_type
      :type: str | None


      Channel type detected by electronics.

      This normally matches channel_type, but used in electronics design and testing.

      :returns: 'E' for electric, 'H' for magnetic, or None if no header.
      :rtype: str or None


   .. py:property:: lp_frequency
      :type: int | None


      Low-pass filter frequency based on hardware configuration.

      :returns: Filter frequency in Hz, or None if no header.
      :rtype: int or None


   .. py:property:: preamp_gain
      :type: float


      Pre-amplifier gain factor.

      :returns: Gain factor, default 1.0.
      :rtype: float

      :raises Exception: If channel type is not determined before calculating gain.


   .. py:property:: channel_main_gain
      :type: float


      Main gain of the board.

      :returns: Main gain factor.
      :rtype: float


   .. py:property:: intrinsic_circuitry_gain
      :type: float


      Intrinsic circuitry gain based on sensor range configuration.

      This function adjusts the intrinsic circuitry gain based on the
      sensor range configuration in the configuration fingerprint.

      For the Electric channel, calibration path, or H-legacy
      sensors all go through a 1/4 gain stage, and then they get a virtual x2 gain from
      Single-ended-diff before the A/D. In the case of newer sensors (differential)
      instead of a 1/4 gain stage, there is only a 1/2 gain stage.

      Therefore, in the E, cal and legacy sensor case the circuitry gain is 1/2, while for
      newer sensors it is 1.

      :returns: Intrinsic gain factor.
      :rtype: float

      :raises Exception: If channel type is not determined before calculating gain.

      .. rubric:: Notes

      Circuitry Gain not directly configurable by the user.


   .. py:property:: attenuator_gain
      :type: float


      Attenuator gain factor.

      :returns: Attenuator gain factor, default 1.0.
      :rtype: float

      :raises Exception: If channel type is not determined before calculating gain.


   .. py:property:: total_selectable_gain
      :type: float


      Total gain that is selectable by the user.

      Combines attenuator, preamp, and main channel gains.

      :returns: Total selectable gain factor.
      :rtype: float


   .. py:property:: total_circuitry_gain
      :type: float


      Total board gain including both intrinsic and user-selectable gains.

      :returns: Total circuitry gain factor.
      :rtype: float


   .. py:property:: sample_rate_base
      :type: int | None


      Base sample rate from binary header.

      :returns: Base sample rate, or None if no header.
      :rtype: int or None


   .. py:property:: sample_rate_exp
      :type: int | None


      Sample rate exponent from binary header.

      :returns: Sample rate exponent, or None if no header.
      :rtype: int or None


   .. py:property:: sample_rate
      :type: float | None


      Calculated sample rate.

      :returns: Sample rate in Hz, or None if no header.
      :rtype: float or None


   .. py:property:: bytes_per_sample
      :type: int | None


      Number of bytes per sample.

      :returns: Bytes per sample, or None if no header.
      :rtype: int or None


   .. py:property:: frame_size
      :type: int | None


      Frame size from binary header.

      :returns: Frame size value, or None if no header.
      :rtype: int or None


   .. py:property:: data_footer
      :type: int | None


      Data footer extracted from frame size.

      :returns: Data footer value, or None if no frame size.
      :rtype: int or None


   .. py:property:: frame_size_bytes
      :type: int | None


      Frame size in bytes.

      :returns: Frame size in bytes, or None if no frame size.
      :rtype: int or None


   .. py:property:: decimation_node_id
      :type: int | None


      Decimation node identifier.

      :returns: Decimation node ID, or None if no header.
      :rtype: int or None


   .. py:property:: frame_rollover_count
      :type: int | None


      Frame rollover count.

      :returns: Rollover count, or None if no header.
      :rtype: int or None


   .. py:property:: gps_long
      :type: float | None


      GPS longitude.

      :returns: Longitude in degrees, or None if no header.
      :rtype: float or None


   .. py:property:: gps_lat
      :type: float | None


      GPS latitude.

      :returns: Latitude in degrees, or None if no header.
      :rtype: float or None


   .. py:property:: gps_elevation
      :type: float | None


      GPS elevation.

      :returns: Elevation in meters, or None if no header.
      :rtype: float or None


   .. py:property:: gps_horizontal_accuracy
      :type: float | None


      GPS horizontal accuracy.

      :returns: Horizontal accuracy in meters (converted from millimeters), or None if no header.
      :rtype: float or None


   .. py:property:: gps_vertical_accuracy
      :type: float | None


      GPS vertical accuracy.

      :returns: Vertical accuracy in meters (converted from millimeters), or None if no header.
      :rtype: float or None


   .. py:property:: timing_status
      :type: tuple[Any, Ellipsis] | None


      Timing status information.

      :returns: Timing status data, or None if no header.
      :rtype: tuple of Any or None


   .. py:property:: timing_flags
      :type: Any | None


      Timing flags from timing status.

      :returns: Timing flags, or None if no timing status.
      :rtype: Any or None


   .. py:property:: timing_sat_count
      :type: Any | None


      Satellite count from timing status.

      :returns: Satellite count, or None if no timing status.
      :rtype: Any or None


   .. py:property:: timing_stability
      :type: Any | None


      Timing stability from timing status.

      :returns: Timing stability value, or None if no timing status.
      :rtype: Any or None


   .. py:property:: future1
      :type: Any | None


      Future field 1 (reserved).

      :returns: Future field value, or None if no header.
      :rtype: Any or None


   .. py:property:: future2
      :type: Any | None


      Future field 2 (reserved).

      :returns: Future field value, or None if no header.
      :rtype: Any or None


   .. py:property:: saturated_frames
      :type: int | None


      Number of saturated frames.

      :returns: Saturated frame count, or None if no header.
      :rtype: int or None


   .. py:property:: missing_frames
      :type: int | None


      Number of missing frames.

      :returns: Missing frame count, or None if no header.
      :rtype: int or None


   .. py:property:: battery_voltage_v
      :type: float | None


      Battery voltage in volts.

      :returns: Battery voltage in volts (converted from millivolts), or None if no header.
      :rtype: float or None


   .. py:property:: min_signal
      :type: Any | None


      Minimum signal value.

      :returns: Minimum signal value, or None if no header.
      :rtype: Any or None


   .. py:property:: max_signal
      :type: Any | None


      Maximum signal value.

      :returns: Maximum signal value, or None if no header.
      :rtype: Any or None


   .. py:method:: unpack_header(stream: BinaryIO) -> None

      Read and unpack binary header from stream.

      :param stream: Binary stream to read header from.
      :type stream: BinaryIO



   .. py:method:: get_channel_metadata() -> mt_metadata.timeseries.Magnetic | mt_metadata.timeseries.Electric

      Translate metadata to channel metadata.

      :returns: Channel metadata object populated with header data.
      :rtype: Magnetic or Electric

      :raises KeyError: If channel ID is not found in channel map.
      :raises ValueError: If required fields are None or invalid.



   .. py:method:: get_run_metadata() -> mt_metadata.timeseries.Run

      Translate to run metadata.

      :returns: Run metadata object populated with header data.
      :rtype: Run

      :raises ValueError: If required fields are None.



   .. py:method:: get_station_metadata() -> mt_metadata.timeseries.Station

      Translate to station metadata.

      :returns: Station metadata object populated with header data.
      :rtype: Station



.. py:class:: PhoenixCalibration(cal_fn: str | pathlib.Path | None = None, **kwargs: Any)

   Phoenix Geophysics calibration data reader and filter manager.

   This class reads Phoenix calibration files in JSON format and provides
   access to frequency response filters for different channels and lowpass
   filter settings. It supports both receiver and sensor calibration files.

   :param cal_fn: Path to the calibration file to read. If provided, the file will be
                  loaded automatically during initialization.
   :type cal_fn: str or pathlib.Path, optional
   :param \*\*kwargs: Additional keyword arguments that will be set as instance attributes.
   :type \*\*kwargs: Any

   .. attribute:: obj

      The parsed calibration object containing all calibration data.

      :type: Any or None


   .. py:attribute:: obj
      :type:  Any
      :value: None



   .. py:property:: cal_fn
      :type: pathlib.Path


      Path to the calibration file.

      :returns: The path to the calibration file.
      :rtype: pathlib.Path


   .. py:property:: calibration_date
      :type: mt_metadata.common.mttime.MTime | None


      Get the calibration date from the loaded calibration data.

      :returns: The calibration date as an MTime object, or None if no data is loaded.
      :rtype: MTime or None


   .. py:method:: get_max_freq(freq: numpy.typing.NDArray[numpy.floating] | list[float] | numpy.ndarray) -> int

      Calculate the maximum frequency for filter naming.

      Determines the power-of-10 frequency limit based on the maximum
      frequency in the input array. Used to name filters as
      {channel}_{max_freq}hz_lowpass.

      :param freq: Array of frequency values in Hz.
      :type freq: numpy.ndarray

      :returns: The power-of-10 frequency limit (e.g., 1000 for frequencies up to 9999 Hz).
      :rtype: int

      .. rubric:: Examples

      >>> cal = PhoenixCalibration()
      >>> freq = np.array([1.0, 10.0, 100.0, 1500.0])
      >>> cal.get_max_freq(freq)
      1000



   .. py:property:: base_filter_name
      :type: str | None


      Generate the base filter name from instrument information.

      Creates a standardized filter name prefix based on the instrument
      type, model, and serial number from the calibration data.

      :returns: Base filter name in format "{instrument_type}_{instrument_model}_{serial}"
                converted to lowercase, or None if no data is loaded.
      :rtype: str or None

      .. rubric:: Examples

      >>> cal = PhoenixCalibration("calibration.json")
      >>> cal.base_filter_name
      'mtu-5c_rmt03-j_666'


   .. py:method:: get_filter_lp_name(channel: str, max_freq: int) -> str

      Generate a lowpass filter name for a specific channel and frequency.

      Creates a standardized filter name for receiver calibration filters
      in the format: {base_filter_name}_{channel}_{max_freq}hz_lowpass

      :param channel: Channel identifier (e.g., 'e1', 'h2').
      :type channel: str
      :param max_freq: Maximum frequency in Hz for the lowpass filter.
      :type max_freq: int

      :returns: Complete lowpass filter name in lowercase.
      :rtype: str

      .. rubric:: Examples

      >>> cal = PhoenixCalibration("calibration.json")
      >>> cal.get_filter_lp_name("e1", 1000)
      'mtu-5c_rmt03-j_666_e1_1000hz_lowpass'



   .. py:method:: get_filter_sensor_name(sensor: str) -> str

      Generate a sensor filter name for a specific sensor.

      Creates a standardized filter name for sensor calibration filters
      in the format: {base_filter_name}_{sensor}

      :param sensor: Sensor identifier or serial number.
      :type sensor: str

      :returns: Complete sensor filter name in lowercase.
      :rtype: str

      .. rubric:: Examples

      >>> cal = PhoenixCalibration("calibration.json")
      >>> cal.get_filter_sensor_name("sensor123")
      'mtu-5c_rmt03-j_666_sensor123'



   .. py:method:: read(cal_fn: str | pathlib.Path | None = None) -> None

      Read and parse a Phoenix calibration file.

      Loads calibration data from a JSON file and creates frequency response
      filters for each channel and frequency band. The method creates channel
      attributes (e.g., self.e1, self.h2) containing either:
      - Dictionary of filters by frequency (receiver calibration)
      - Single filter object (sensor calibration)

      :param cal_fn: Path to the calibration file to read. If None, uses the previously
                     set calibration file path.
      :type cal_fn: str, pathlib.Path, or None, optional

      :raises IOError: If the calibration file cannot be found or read.

      .. rubric:: Notes

      The method automatically determines calibration type based on file_type:
      - "receiver calibration": Creates multiple filters per channel by frequency
      - "sensor calibration": Creates single filter per channel



   .. py:method:: get_filter(channel: str, filter_name: str | int) -> mt_metadata.timeseries.filters.FrequencyResponseTableFilter

      Get the frequency response filter for a specific channel and filter.

      Retrieves the lowpass filter for the given channel and filter specification.
      The method automatically handles both string and integer filter names.

      :param channel: Channel identifier (e.g., 'e1', 'h2', 'h3').
      :type channel: str
      :param filter_name: Filter specification, typically the lowpass frequency in Hz
                          (e.g., 1000, '100', 10000).
      :type filter_name: str or int

      :returns: The frequency response filter object containing the calibration data
                for the specified channel and filter.
      :rtype: FrequencyResponseTableFilter

      :raises AttributeError: If the specified channel is not found in the calibration data.
      :raises KeyError: If the specified filter is not found for the given channel.

      .. rubric:: Examples

      >>> cal = PhoenixCalibration("calibration.json")
      >>> filt = cal.get_filter("e1", 1000)
      >>> print(f"Filter name: {filt.name}")
      >>> print(f"Frequency points: {len(filt.frequencies)}")



.. py:class:: TSReaderBase(path: str | pathlib.Path, num_files: int = 1, header_length: int = 128, report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.header.Header`


   Generic reader that all other readers will inherit.

   This base class provides common functionality for reading Phoenix Geophysics
   time series data files, including header parsing, file sequence management,
   and metadata handling.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param header_length: Length of file header in bytes, by default 128
   :type header_length: int, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent Header class

   .. attribute:: stream

      File stream for reading binary data

      :type: BinaryIO or None

   .. attribute:: base_path

      Path to the current file

      :type: Path

   .. attribute:: last_seq

      Last sequence number in the file sequence

      :type: int

   .. attribute:: rx_metadata

      Receiver metadata object

      :type: PhoenixReceiverMetadata or None


   .. py:attribute:: logger


   .. py:property:: base_path
      :type: pathlib.Path


      Full path of the file.

      :returns: Full path to the file
      :rtype: Path


   .. py:attribute:: last_seq


   .. py:attribute:: stream
      :value: None



   .. py:attribute:: rx_metadata
      :value: None



   .. py:property:: base_dir
      :type: pathlib.Path


      Parent directory of the file.

      :returns: Parent directory of the file
      :rtype: Path


   .. py:property:: file_name
      :type: str


      Name of the file.

      :returns: Name of the file
      :rtype: str


   .. py:property:: file_extension
      :type: str


      File extension.

      :returns: File extension including the dot
      :rtype: str


   .. py:property:: instrument_id
      :type: str


      Instrument ID extracted from filename.

      :returns: Instrument identifier
      :rtype: str


   .. py:property:: seq
      :type: int


      Sequence number of the file.

      :returns: Sequence number extracted from filename or set value
      :rtype: int


   .. py:property:: file_size
      :type: int


      File size in bytes.

      :returns: Size of the file in bytes
      :rtype: int


   .. py:property:: max_samples
      :type: int


      Maximum number of samples in a file.

      Calculated as: (total number of bytes - header length) / frame size * n samples per frame

      :returns: Maximum number of samples in the file
      :rtype: int


   .. py:property:: sequence_list
      :type: list[pathlib.Path]


      Get all the files in the sequence sorted by sequence number.

      :returns: List of Path objects for all files in the sequence
      :rtype: list[Path]


   .. py:property:: config_file_path
      :type: pathlib.Path | None


      Path to the config.json file.

      :returns: Path to config file if it exists, None otherwise
      :rtype: Path or None


   .. py:property:: recmeta_file_path
      :type: pathlib.Path | None


      Path to the recmeta.json file.

      :returns: Path to recmeta file if it exists, None otherwise
      :rtype: Path or None


   .. py:method:: open_next() -> bool

      Open the next file in the sequence.

      :returns: True if next file is now open, False if it is not
      :rtype: bool



   .. py:method:: open_file_seq(file_seq_num: int | None = None) -> bool

      Open a file in the sequence given the sequence number.

      :param file_seq_num: Sequence number to open, by default None
      :type file_seq_num: int, optional

      :returns: True if file is now open, False if it is not
      :rtype: bool



   .. py:method:: close() -> None

      Close the file stream.



   .. py:method:: get_config_object() -> mth5.io.phoenix.readers.config.PhoenixConfig | None

      Read a config file into an object.

      :returns: Configuration object if config file exists, None otherwise
      :rtype: PhoenixConfig or None



   .. py:method:: get_receiver_metadata_object() -> None

      Read recmeta.json into an object and store in rx_metadata attribute.



   .. py:method:: get_lowpass_filter_name() -> str | None

      Get the lowpass filter used by the receiver pre-decimation.

      :returns: Name of the lowpass filter if available, None otherwise
      :rtype: str or None



   .. py:method:: update_channel_map_from_recmeta() -> None

      Update channel map from recmeta.json file.



   .. py:property:: channel_metadata
      :type: Any


      Channel metadata updated from recmeta.

      :returns: Channel metadata object
      :rtype: Any


   .. py:property:: run_metadata
      :type: Any


      Run metadata updated from recmeta.

      :returns: Run metadata object
      :rtype: Any


   .. py:property:: station_metadata
      :type: Any


      Station metadata updated from recmeta.

      :returns: Station metadata object
      :rtype: Any


   .. py:method:: get_receiver_lowpass_filter(rxcal_fn: str | pathlib.Path) -> Any

      Get receiver lowpass filter from the rxcal.json file.

      :param rxcal_fn: Path to the receiver calibration file
      :type rxcal_fn: str or Path

      :returns: Filter object from calibration file
      :rtype: Any

      :raises ValueError: If the lowpass filter name cannot be found



   .. py:method:: get_dipole_filter() -> mt_metadata.timeseries.filters.CoefficientFilter | None

      Get dipole filter for electric field channels.

      :returns: Dipole filter if channel has dipole length, None otherwise
      :rtype: CoefficientFilter or None



   .. py:method:: get_sensor_filter(scal_fn: str | pathlib.Path) -> Any

      Get sensor filter from calibration file.

      :param scal_fn: Path to sensor calibration file
      :type scal_fn: str or Path

      :returns: Sensor filter object
      :rtype: Any

      .. rubric:: Notes

      This method is not implemented yet.



   .. py:method:: get_v_to_mv_filter() -> mt_metadata.timeseries.filters.CoefficientFilter

      Create a filter to convert units from volts to millivolts.

      :returns: Filter that converts volts to millivolts with gain of 1000
      :rtype: CoefficientFilter



   .. py:method:: get_channel_response(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> mt_metadata.timeseries.filters.ChannelResponse

      Get the channel response filter.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: Complete channel response filter chain
      :rtype: ChannelResponse



.. py:class:: NativeReader(path: str | pathlib.Path, num_files: int = 1, scale_to: int = AD_INPUT_VOLTS, header_length: int = 128, last_frame: int = 0, ad_plus_minus_range: float = 5.0, channel_type: str = 'E', report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.TSReaderBase`


   Native sampling rate 'Raw' time series reader class.

   This class reads native binary (.bin) files from Phoenix Geophysics MTU-5C
   instruments. The files are formatted with a header of 128 bytes followed by
   frames of 64 bytes each. Each frame contains 20 x 3-byte (24-bit) data
   points plus a 4-byte footer.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param scale_to: Data scaling mode (AD_IN_AD_UNITS, AD_INPUT_VOLTS, or
                    INSTRUMENT_INPUT_VOLTS), by default AD_INPUT_VOLTS
   :type scale_to: int, optional
   :param header_length: Length of file header in bytes, by default 128
   :type header_length: int, optional
   :param last_frame: Last frame number seen by the streamer, by default 0
   :type last_frame: int, optional
   :param ad_plus_minus_range: ADC plus/minus range in volts, by default 5.0
   :type ad_plus_minus_range: float, optional
   :param channel_type: Channel type identifier, by default "E"
   :type channel_type: str, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent TSReaderBase class

   .. attribute:: last_frame

      Last frame number processed

      :type: int

   .. attribute:: data_scaling

      Current data scaling mode

      :type: int

   .. attribute:: ad_plus_minus_range

      ADC voltage range

      :type: float

   .. attribute:: input_plusminus_range

      Input voltage range after gain correction

      :type: float

   .. attribute:: scale_factor

      Calculated scaling factor for data conversion

      :type: float

   .. attribute:: footer_idx_samp_mask

      Bit mask for frame index in footer

      :type: int

   .. attribute:: footer_sat_mask

      Bit mask for saturation count in footer

      :type: int


   .. py:attribute:: last_frame
      :value: 0



   .. py:attribute:: header_length
      :value: 128


      Length of the header in bytes.

      :returns: Header length in bytes.
      :rtype: int


   .. py:attribute:: data_scaling
      :value: 1



   .. py:attribute:: ad_plus_minus_range
      :value: 5.0



   .. py:attribute:: input_plusminus_range


   .. py:attribute:: scale_factor
      :value: 256



   .. py:attribute:: footer_idx_samp_mask
      :value: 0



   .. py:attribute:: footer_sat_mask
      :value: 0



   .. py:method:: read_frames(num_frames: int) -> numpy.ndarray

      Read the given number of frames from the data stream.

      .. note::

         The seek position is not reset, so iterating this method will read
         from the last position in the stream.

      :param num_frames: Number of frames to read
      :type num_frames: int

      :returns: Scaled data from the given number of frames with dtype float64
      :rtype: np.ndarray



   .. py:property:: npts_per_frame
      :type: int


      Get the number of data points per frame.

      :returns: Number of data points per frame (frame size - 4 footer bytes) / 3 bytes per sample
      :rtype: int


   .. py:method:: read() -> tuple[numpy.ndarray, numpy.ndarray]

      Read the full data file using memory mapping and stride tricks.

      .. note::

         This uses numpy.lib.stride_tricks.as_strided which can be unstable
         if the bytes are not the correct length. See notes by numpy.

         The solution is adapted from:
         https://stackoverflow.com/questions/12080279/how-do-i-create-a-numpy-dtype-that-includes-24-bit-integers

      :returns: Scaled time series data and footer data as (data, footer)
      :rtype: tuple[np.ndarray, np.ndarray]



   .. py:method:: read_sequence(start: int = 0, end: int | None = None) -> tuple[numpy.ndarray, numpy.ndarray]

      Read a sequence of files into a single array.

      :param start: Sequence start index, by default 0
      :type start: int, optional
      :param end: Sequence end index, by default None
      :type end: int or None, optional

      :returns: Scaled time series data and footer data as (data, footer)
                - data: np.ndarray with dtype float32
                - footer: np.ndarray with dtype int32
      :rtype: tuple[np.ndarray, np.ndarray]



   .. py:method:: skip_frames(num_frames: int) -> bool

      Skip frames in the data stream.

      :param num_frames: Number of frames to skip
      :type num_frames: int

      :returns: True if skip completed successfully, False if end of file reached
      :rtype: bool



   .. py:method:: to_channel_ts(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> mth5.timeseries.ChannelTS

      Convert to a ChannelTS object.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: Channel time series object with data, metadata, and calibration
      :rtype: ChannelTS



.. py:class:: DecimatedSegmentedReader(path: str | pathlib.Path, num_files: int = 1, report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.TSReaderBase`


   Class to create a streamer for segmented decimated time series.

   This reader handles segmented decimated time series files such as 'td_24k'.
   These files have sub headers containing metadata for each segment.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent TSReaderBase class

   .. attribute:: sub_header

      SubHeader instance for parsing segment headers

      :type: SubHeader

   .. attribute:: subheader

      Dictionary for additional subheader information

      :type: dict


   .. py:attribute:: sub_header


   .. py:attribute:: subheader


   .. py:method:: read_segment(metadata_only: bool = False) -> Segment

      Read a single segment from the file.

      :param metadata_only: If True, only read metadata without loading data, by default False
      :type metadata_only: bool, optional

      :returns: Segment object containing data and metadata
      :rtype: Segment

      :raises ValueError: If stream is not available



   .. py:method:: to_channel_ts(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> mth5.timeseries.ChannelTS

      Convert to a ChannelTS object.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: Channel time series object with data, metadata, and calibration
      :rtype: ChannelTS



.. py:class:: DecimatedContinuousReader(path: str | pathlib.Path, num_files: int = 1, report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.TSReaderBase`


   Class to create a streamer for continuous decimated time series.

   This reader handles continuous decimated time series files such as 'td_150',
   'td_30'. These files have no sub header information.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent TSReaderBase class

   .. attribute:: subheader

      Empty dictionary as these files have no sub header information

      :type: dict

   .. attribute:: data_size

      Size of the data sequence when read

      :type: int or None


   .. py:attribute:: subheader


   .. py:attribute:: data_size
      :type:  int | None
      :value: None



   .. py:property:: segment_start_time
      :type: mt_metadata.common.mttime.MTime


      Estimate the segment start time based on sequence number.

      The first sequence starts 1 second later than the set start time due
      to initiation within the data logger.

      :returns: Start time of the recording segment
      :rtype: MTime


   .. py:property:: segment_end_time
      :type: mt_metadata.common.mttime.MTime


      Estimate end time of the segment.

      The first sequence starts 1 second later than the set start time due
      to initiation within the data logger.

      :returns: Estimated end time from number of samples
      :rtype: MTime


   .. py:property:: sequence_start
      :type: mt_metadata.common.mttime.MTime


      Get the sequence start time.

      :returns: Start time of the sequence
      :rtype: MTime


   .. py:property:: sequence_end
      :type: mt_metadata.common.mttime.MTime


      Get the sequence end time.

      :returns: End time of the sequence based on data size or max samples
      :rtype: MTime


   .. py:method:: read() -> numpy.ndarray

      Read in the full data from the current file.

      :returns: Single channel data array with dtype float32
      :rtype: np.ndarray



   .. py:method:: read_sequence(start: int = 0, end: int | None = None) -> numpy.ndarray

      Read a sequence of files.

      :param start: Starting index in the sequence, by default 0
      :type start: int, optional
      :param end: Ending index in the sequence to read, by default None
      :type end: int or None, optional

      :returns: Data within the given sequence range as float32 array
      :rtype: np.ndarray



   .. py:method:: to_channel_ts(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> mth5.timeseries.ChannelTS

      Convert to a ChannelTS object.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: Channel time series object with data, metadata, and calibration
      :rtype: ChannelTS



.. py:class:: PhoenixConfig(fn: str | pathlib.Path | None = None, **kwargs: Any)

   Phoenix Geophysics configuration file reader and metadata container.

   This class reads and provides access to Phoenix MTU-5C instrument
   configuration data stored in JSON format. The configuration file contains
   recording parameters, instrument settings, and metadata used to control
   data acquisition.

   :param fn: Path to the Phoenix configuration file (typically config.json).
              If provided, the file will be validated for existence.
   :type fn: str, pathlib.Path, or None, optional
   :param \*\*kwargs: Additional keyword arguments (currently unused).
   :type \*\*kwargs: Any

   .. attribute:: fn

      Path to the configuration file.

      :type: pathlib.Path or None

   .. attribute:: obj

      Parsed configuration object containing all settings.

      :type: Any or None

   .. attribute:: logger

      Logger instance for debugging and error reporting.

      :type: loguru.Logger

   .. rubric:: Examples

   >>> config = PhoenixConfig("config.json")
   >>> config.read()
   >>> station = config.station_metadata()
   >>> print(f"Station ID: {station.id}")


   .. py:attribute:: obj
      :type:  Any
      :value: None



   .. py:attribute:: logger
      :type:  loguru.Logger


   .. py:property:: fn
      :type: pathlib.Path | None


      Path to the Phoenix configuration file.

      :returns: The path to the configuration file, or None if not set.
      :rtype: pathlib.Path or None


   .. py:method:: read(fn: str | pathlib.Path | None = None) -> None

      Read and parse a Phoenix configuration file.

      Loads and parses a Phoenix MTU-5C configuration file in JSON format.
      The parsed configuration is stored in the obj attribute and provides
      access to all recording parameters and instrument settings.

      :param fn: Path to the configuration file to read. If None, uses the
                 previously set file path from the fn property.
      :type fn: str, pathlib.Path, or None, optional

      :raises ValueError: If no file path is provided and none was previously set.
      :raises IOError: If the configuration file cannot be read or parsed.

      .. rubric:: Notes

      The configuration file should be in Phoenix JSON format containing
      recording parameters, instrument settings, and metadata.



   .. py:method:: has_obj() -> bool

      Check if configuration data has been loaded.

      :returns: True if configuration data is loaded, False otherwise.
      :rtype: bool



   .. py:property:: auto_power_enabled
      :type: Any | None


      Auto power enabled setting from configuration.

      :returns: The auto power enabled setting, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: config
      :type: Any | None


      Main configuration section from the configuration file.

      :returns: The first configuration object containing recording parameters,
                or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: empower_version
      :type: Any | None


      EMPower software version from configuration.

      :returns: The EMPower software version, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: mtc150_reset
      :type: Any | None


      MTC150 reset setting from configuration.

      :returns: The MTC150 reset setting, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: network
      :type: Any | None


      Network configuration from configuration file.

      :returns: The network configuration settings, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: receiver
      :type: Any | None


      Receiver configuration from configuration file.

      :returns: The receiver configuration settings, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: schedule
      :type: Any | None


      Recording schedule from configuration file.

      :returns: The recording schedule configuration, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: surveyTechnique
      :type: Any | None


      Survey technique setting from configuration file.

      :returns: The survey technique setting, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: timezone
      :type: Any | None


      Timezone setting from configuration file.

      :returns: The timezone setting, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: timezone_offset
      :type: Any | None


      Timezone offset from configuration file.

      :returns: The timezone offset in hours, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: version
      :type: Any | None


      Configuration file version from configuration file.

      :returns: The configuration file version, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:method:: station_metadata() -> mt_metadata.timeseries.Station

      Create a Station metadata object from configuration data.

      Extracts station information from the loaded configuration and creates
      a standardized Station metadata object with basic station parameters.

      :returns: A Station metadata object populated with configuration data including
                station ID, operator information, company name, and notes.
      :rtype: Station

      :raises AttributeError: If no configuration is loaded or required fields are missing.

      .. rubric:: Notes

      The method extracts the following information from config.layout:
      - Station_Name -> station.id
      - Operator -> station.acquired_by.name
      - Company_Name -> station.acquired_by.organization
      - Notes -> station.comments

      .. rubric:: Examples

      >>> config = PhoenixConfig("config.json")
      >>> config.read()
      >>> station = config.station_metadata()
      >>> print(f"Station: {station.id}")



.. py:class:: PhoenixReceiverMetadata(fn: str | pathlib.Path | None = None, **kwargs: Any)

   Container for Phoenix Geophysics recmeta.json metadata files.

   This class reads and parses receiver metadata from JSON configuration files
   used to control Phoenix Geophysics MTU-5C data recording systems. It provides
   methods to extract channel configurations, instrument settings, and convert
   them to standardized metadata objects.

   :param fn: Path to the recmeta.json file. If provided, the file will be read
              automatically during initialization.
   :type fn: str, Path, or None, optional
   :param \*\*kwargs: Additional keyword arguments (currently unused).

   .. attribute:: fn

      Path to the metadata file.

      :type: Path or None

   .. attribute:: obj

      Parsed JSON content as a SimpleNamespace object.

      :type: SimpleNamespace or None

   .. attribute:: logger

      Logger instance for error reporting.

      :type: loguru.Logger

   :raises IOError: If the specified file does not exist.

   .. rubric:: Examples

   >>> metadata = PhoenixReceiverMetadata("recmeta.json")
   >>> channel_map = metadata.channel_map
   >>> e1_config = metadata.e1_metadata

   .. rubric:: Notes

   The class supports both electric and magnetic channel configurations
   with automatic mapping from Phoenix-specific parameter names to
   standardized metadata attributes.


   .. py:property:: fn
      :type: pathlib.Path | None


      Path to the metadata file.

      :returns: Path to the recmeta.json file, or None if not set.
      :rtype: Path or None


   .. py:attribute:: obj
      :type:  types.SimpleNamespace | None
      :value: None



   .. py:attribute:: logger


   .. py:property:: instrument_id
      :type: str | None


      Instrument identifier from metadata.

      :returns: Instrument ID if available, None otherwise.
      :rtype: str or None


   .. py:method:: read(fn: str | pathlib.Path | None = None) -> None

      Read a recmeta.json file in Phoenix format.

      :param fn: Path to the JSON file. If None, uses the current fn property.
      :type fn: str, Path, or None, optional

      :raises IOError: If no file path is specified or file doesn't exist.
      :raises ValueError: If the file cannot be parsed as JSON.



   .. py:method:: has_obj() -> bool

      Check if metadata object is loaded.

      :returns: True if metadata object exists, False otherwise.
      :rtype: bool



   .. py:property:: channel_map
      :type: dict[int, str]


      Channel mapping from index to component tag.

      :returns: Dictionary mapping channel indices to component tags (lowercase).
      :rtype: dict[int, str]

      :raises AttributeError: If metadata object is not loaded or missing channel_map.


   .. py:property:: lp_filter_base_name
      :type: str | None


      Base name for low-pass filter identifiers.

      :returns: Filter base name combining receiver info, or None if not available.
      :rtype: str or None


   .. py:method:: get_ch_index(tag: str) -> int

      Get channel index from component tag.

      :param tag: Component tag (e.g., 'e1', 'h1', etc.).
      :type tag: str

      :returns: Channel index corresponding to the tag.
      :rtype: int

      :raises ValueError: If the tag is not found in the channel map.
      :raises AttributeError: If metadata object is not loaded.



   .. py:method:: get_ch_tag(index: int) -> str

      Get component tag from channel index.

      :param index: Channel index.
      :type index: int

      :returns: Component tag corresponding to the index.
      :rtype: str

      :raises ValueError: If the index is not found in the channel map.
      :raises AttributeError: If metadata object is not loaded.



   .. py:property:: e1_metadata
      :type: mt_metadata.timeseries.Electric


      Electric channel 1 metadata.


   .. py:property:: e2_metadata
      :type: mt_metadata.timeseries.Electric


      Electric channel 2 metadata.


   .. py:property:: h1_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 1 metadata.


   .. py:property:: h2_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 2 metadata.


   .. py:property:: h3_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 3 metadata.


   .. py:property:: h4_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 4 metadata.


   .. py:property:: h5_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 5 metadata.


   .. py:property:: h6_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 6 metadata.


   .. py:method:: get_ch_metadata(index: int) -> mt_metadata.timeseries.Electric | mt_metadata.timeseries.Magnetic

      Get channel metadata from index.

      :param index: Channel index.
      :type index: int

      :returns: Channel metadata object corresponding to the index.
      :rtype: Electric or Magnetic

      :raises ValueError: If index is not found in channel map.
      :raises AttributeError: If the corresponding metadata property doesn't exist.



   .. py:property:: run_metadata
      :type: mt_metadata.timeseries.Run


      Run metadata from receiver configuration.

      :returns: Run metadata object with data logger and timing information.
      :rtype: Run


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Station metadata from receiver configuration.

      :returns: Station metadata object with location and acquisition information.
      :rtype: Station


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Survey metadata from receiver configuration.

      :returns: Survey metadata object with survey information.
      :rtype: Survey


.. py:class:: MTUTable(file_path: str | pathlib.Path | None = None, **kwargs)

   =======================================================================
   DECODING METHOD FOR TBL VALUES:

   The Phoenix TBL file is a series of 25-byte blocks containing key-value pairs:
   - Bytes 0-11:  Tag name (4-character string, null-padded)
   - Bytes 12-24: Value (13 bytes, mixed data types)

   Values can be decoded as follows:
   1. INT (4 bytes):    struct.unpack('<i', bytes[0:4])  - Little-endian signed int
   2. DOUBLE (8 bytes): struct.unpack('<d', bytes[0:8])  - Little-endian double
   3. CHAR (variable):  bytes.decode('latin-1').strip()  - Null-terminated string
   4. BYTE (1 byte):    struct.unpack('<B', bytes[0:1])  - Unsigned byte
   5. TIME (6 bytes):   [sec, min, hour, day, month, year-2000] format

   The TBL_TAG_TYPES dictionary maps each known tag to its data type, enabling
   automatic decoding via decode_tbl_value() function. Unknown tags return raw bytes.

   Example usage:
       # Automatic decoding:
       tbl_dict = get_dictionary_from_tbl('file.TBL', decode_values=True)

       # Manual decoding with read_tbl (legacy):
       info = read_tbl('/path', 'file.TBL')

   =======================================================================
   original comments from MATLAB script:

   read_tbl - reads a (binary) TBL table file of the legacy Phoenix format
   (MTU-5A) and output the "info" metadata dictionary.

   :param fpath: path to the tbl
   :param fname: name of the tbl file (including extensions)

   :returns: output dict of the TBL metadata
   :rtype: info

   =======================================================================
   definition of the TBL tags (or what I guessed after reading the user
   manual and fiddling with their files)
   SITE: site name
   SNUM: serial number (of the box)
   FILE: file name recorded
   CMPY: company/institute of the survey
   SRVY: survey project name
   EXLN: Ex channel dipole length
   EYLN: Ey channel dipole length
   NREF: North reference (true, or magnetic north)
   LNGG: longitude in degree-minute format (DDD MM.MM)
   LATG: latitude in degree-minute format (DD MM.MM)
   ELEV: elevation (in metres)
   HXSN: Hx channel coil serial number
   HYSN: Hy channel coil serial number
   HZSN: Hz channel coil serial number
   STIM: starting time (UTC)
   ETIM: ending time (UTC)
   LFRQ: powerline frequency for filtering (can only be 50 or 60 Hz)
   HGN:  final H-channel gain
   HGNC: H-channel gain control: HGN = PA * 2^HGNC (note: PA =
       PreAmplifier gain)
   EGN:  final E-channel gain
   EGNC: E-channel gain control: HGN = PA * 2^HGNC (note: PA =
       PreAmplifier gain)
   HSMP: L3 and L4 time slot in second (MTU-5A) or minute (MTU-5P),
       this means the instrument will record L3NS seconds for L3 and L4NS
       seconds for L4, for every HSMP time slot.
   L3NS: L3 sample time (in second)
   L4NS: L4 sample time (in second)
   SRL3: L3 sample rate
   SRL4: L4 sample rate
   SRL5: L5 sample rate
   HATT: H channel attenuation (1/4.3 for MTU-5A)
   HNOM: H channel normalization (mA/nT)
   TCMB: Type of comb filter (probably used to suppress the harmonics of the
       powerline noise.
   TALS: Type of anti-aliasing filter
   LPFR: Parameter of Low-pass/VLF filter. this is a quite complicated
       part as the low-pass filter is simply an R-C circuit with a switch
       to connect to different capacitors. To ensure enough bandwidth
       (proportion to 1/RC), one should use smaller capacitors with larger
       ground resistance.
   ACDC: AC/DC coupling (DC = 0, AC = 1; MT should always be DC)
   FSCV: full scaling A-D converter voltage (in unit of V)
   =======================================================================
   note:
   Phoenix Legacy TBL is a straight-forward parameter-value metadata file,
   stored in a bizarre format. The parameter tag and value are stored in a
   series of 25-byte data blocks, in mixed data type: the first 12 bytes are
   reserved for the tag name (first 4 bytes as char). The values are stored
   in the 13 bytes afterwards, in various formats (char, int, float, etc.).

   So a good practice is to read in those blocks one by one and extract all
   of them. However, not every thing is useful for the metadata, so I only
   extract a few of them, for now.

   Original author:
   Hao
   2012.07.04
   Beijing

   Translated to Python and enhanced by:
   J. Peacock (2025-12-31)

   Main changes:

   - Encapsulated in MTUTable class
   - Automatic type detection and decoding based on TBL_TAG_TYPES
   - Added properties to extract metadata as mt_metadata objects
   =======================================================================


   .. py:attribute:: file_path
      :value: None



   .. py:attribute:: tbl_dict
      :type:  dict[str, int | float | str | bytes]


   .. py:attribute:: TBL_TAG_TYPES


   .. py:method:: decode_tbl_value(value_bytes: bytes, data_type: str) -> int | float | str | bytes

      Decode TBL value bytes based on the specified data type.

      :param value_bytes: 13 bytes from position 12-24 in the 25-byte block containing the value.
      :type value_bytes: bytes
      :param data_type: Type of the data: 'int', 'double', 'char', 'byte', or 'time'.
      :type data_type: str

      :returns: Decoded value in appropriate Python type. Returns raw bytes if
                decoding fails or data_type is unrecognized.
      :rtype: int or float or str or bytes

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> value = tbl.decode_tbl_value(b'  ...', 'int')
      >>> print(value)
      1690



   .. py:method:: read_tbl() -> None

      Read and decode the TBL file, populating the tbl_dict attribute.

      This method reads the TBL file specified during initialization and
      decodes all tag-value pairs according to their known types. The
      results are stored in `self.tbl_dict`.

      :returns: Results are stored in the `tbl_dict` attribute.
      :rtype: None

      .. rubric:: Examples

      >>> tbl = MTUTable('/data/phoenix', '1690C16C.TBL')
      >>> tbl.read_tbl()
      >>> print(tbl.tbl_dict['SITE'])
      '10441W10'
      >>> print(tbl.tbl_dict['SNUM'])
      1690



   .. py:property:: channel_keys
      :type: dict[str, int]


      Get list of channel keys present in the TBL metadata.

      :returns: Dictionary of channel keys and their corresponding values found in tbl_dict (e.g., 'CHEX', 'CHEY', 'CHHX', etc.).
      :rtype: dict[str, int]

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> keys = tbl.channel_keys
      >>> print(keys)
      {'ex': 1, 'ey': 2, 'hx': 3, 'hy': 4, 'hz': 5}


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Extract survey metadata from TBL file.

      :returns: mt_metadata Survey object populated with survey-level information
                from the TBL file (survey ID, company/author).
      :rtype: Survey

      .. rubric:: Notes

      If TBL metadata has not been loaded (via `read_tbl()`), returns an
      empty Survey object with a warning.

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> survey = tbl.survey_metadata
      >>> print(survey.id)
      'MT_Survey_2024'


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Extract station metadata from TBL file.

      :returns: mt_metadata Station object populated with station-level information
                including location (latitude, longitude, elevation, declination)
                and time period.
      :rtype: Station

      .. rubric:: Notes

      If TBL metadata has not been loaded (via `read_tbl()`), returns an
      empty Station object with a warning.

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> station = tbl.station_metadata
      >>> print(station.id)
      '10441W10'
      >>> print(f"{station.location.latitude:.6f}")
      41.006467


   .. py:property:: run_metadata
      :type: mt_metadata.timeseries.Run


      Extract run metadata from TBL file.

      :returns: mt_metadata Run object populated with data logger information
                and channel metadata.
      :rtype: Run

      .. rubric:: Notes

      If TBL metadata has not been loaded (via `read_tbl()`), returns an
      empty Run object with a warning.

      The run includes all channel metadata (ex, ey, hx, hy, hz) obtained
      from their respective property methods.

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> run = tbl.run_metadata
      >>> print(run.id)
      'run_1690'
      >>> print(run.data_logger.id)
      'MTU_1690'


   .. py:property:: ex_metadata
      :type: mt_metadata.timeseries.Electric


      Extract Ex electric channel metadata from TBL file.

      :returns: mt_metadata Electric object for Ex component with dipole length,
                azimuth, AC/DC start values, and channel number.
      :rtype: Electric

      .. rubric:: Notes

      If TBL metadata has not been loaded (via `read_tbl()`), returns an
      empty Electric object with a warning.

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> ex = tbl.ex_metadata
      >>> print(ex.dipole_length)
      100.0


   .. py:property:: ey_metadata
      :type: mt_metadata.timeseries.Electric


      Extract Ey electric channel metadata from TBL file.

      :returns: mt_metadata Electric object for Ey component with dipole length,
                azimuth (Ex azimuth + 90°), AC/DC start values, and channel number.
      :rtype: Electric

      .. rubric:: Notes

      If TBL metadata has not been loaded (via `read_tbl()`), returns an
      empty Electric object with a warning.

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> ey = tbl.ey_metadata
      >>> print(ey.dipole_length)
      100.0


   .. py:property:: hx_metadata
      :type: mt_metadata.timeseries.Magnetic


      Extract Hx magnetic channel metadata from TBL file.

      :returns: mt_metadata Magnetic object for Hx component with maximum field,
                channel number, azimuth, and sensor serial number.
      :rtype: Magnetic

      .. rubric:: Notes

      If TBL metadata has not been loaded (via `read_tbl()`), returns an
      empty Magnetic object with a warning.

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> hx = tbl.hx_metadata
      >>> print(hx.sensor.id)
      'coil1693'


   .. py:property:: hy_metadata
      :type: mt_metadata.timeseries.Magnetic


      Extract Hy magnetic channel metadata from TBL file.

      :returns: mt_metadata Magnetic object for Hy component with maximum field,
                channel number, azimuth (Hx azimuth + 90°), and sensor serial number.
      :rtype: Magnetic

      .. rubric:: Notes

      If TBL metadata has not been loaded (via `read_tbl()`), returns an
      empty Magnetic object with a warning.

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> hy = tbl.hy_metadata
      >>> print(hy.sensor.id)
      'coil1694'


   .. py:property:: hz_metadata
      :type: mt_metadata.timeseries.Magnetic


      Extract Hz magnetic channel metadata from TBL file.

      :returns: mt_metadata Magnetic object for Hz component with maximum field,
                channel number, and sensor serial number.
      :rtype: Magnetic

      .. rubric:: Notes

      If TBL metadata has not been loaded (via `read_tbl()`), returns an
      empty Magnetic object with a warning.

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> hz = tbl.hz_metadata
      >>> print(hz.sensor.id)
      'coil1695'


   .. py:property:: ex_calibration
      :type: float | None


      Calculate Ex channel calibration factor.

      :returns: Calibration factor to convert raw ADC values to mV/km.
                Returns None if TBL metadata has not been loaded.
      :rtype: float or None

      .. rubric:: Notes

      The calibration factor is calculated as:

      .. math::
          \text{cal} = \frac{\text{FSCV}}{2^{23}} \times \frac{1000}{\text{EGN}} \times \frac{1000}{\text{EXLN}}

      where:

      - FSCV: Full-scale converter voltage
      - EGN: Electric channel gain
      - EXLN: Ex dipole length in meters

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> cal = tbl.ex_calibration
      >>> print(f"{cal:.6f}")
      0.000762


   .. py:property:: ey_calibration
      :type: float | None


      Calculate Ey channel calibration factor.

      :returns: Calibration factor to convert raw ADC values to mV/km.
                Returns None if TBL metadata has not been loaded.
      :rtype: float or None

      .. rubric:: Notes

      The calibration factor is calculated as:

      .. math::
          \text{cal} = \frac{\text{FSCV}}{2^{23}} \times \frac{1000}{\text{EGN}} \times \frac{1000}{\text{EYLN}}

      where:

      - FSCV: Full-scale converter voltage
      - EGN: Electric channel gain
      - EYLN: Ey dipole length in meters

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> cal = tbl.ey_calibration
      >>> print(f"{cal:.6f}")
      0.000762


   .. py:property:: magnetic_calibration
      :type: float | None


      Calculate magnetic channel calibration factor.

      :returns: Calibration factor to convert raw ADC values to nT.
                Returns None if TBL metadata has not been loaded.
      :rtype: float or None

      .. rubric:: Notes

      The calibration factor is calculated as:

      .. math::
          \text{cal} = \frac{\text{FSCV}}{2^{23}} \times \frac{1000}{\text{HGN} \times \text{HATT} \times \text{HNOM}}

      where:

      - FSCV: Full-scale converter voltage
      - HGN: Magnetic channel gain
      - HATT: Magnetic channel attenuation
      - HNOM: Magnetic channel normalization (mA/nT)

      This calibration applies to all magnetic channels (Hx, Hy, Hz).

      .. rubric:: Examples

      >>> tbl = MTUTable('/data', 'file.TBL')
      >>> tbl.read_tbl()
      >>> cal = tbl.magnetic_calibration
      >>> print(f"{cal:.9f}")
      0.000000229


.. py:class:: MTUTSN(file_path: str | pathlib.Path | None = None, **kwargs)

   Reader for legacy Phoenix MTU-5A instrument time series binary files.

   Reads time series data from Phoenix MTU-5A (.TS2, .TS3, .TS4, .TS5) and
   V5-2000 system (.TSL, .TSH) binary files. The data consists of 24-bit
   signed integers organized in data blocks with headers.

   :param file_path: Path to the TSN file to read. If None, the reader is created without
                     loading data. Default is None.
   :type file_path: str or Path or None, optional

   .. attribute:: file_path

      Path to the currently loaded TSN file.

      :type: Path or None

   .. attribute:: ts

      Time series data array with shape (n_channels, n_samples).

      :type: ndarray or None

   .. attribute:: tag

      Metadata dictionary containing file information.

      :type: dict

   .. rubric:: Examples

   Read a TS3 file:

   >>> from pathlib import Path
   >>> reader = MTUTSN('data/1690C16C.TS3')
   >>> print(reader.ts.shape)
   (3, 86400)
   >>> print(reader.tag['sample_rate'])
   24

   Create reader without loading data:

   >>> reader = MTUTSN()
   >>> reader.read('data/1690C16C.TS3')

   Access metadata:

   >>> reader = MTUTSN('data/1690C16C.TS4')
   >>> reader.read()
   >>> print(f"Channels: {reader.tag['n_ch']}")
   Channels: 4
   >>> print(f"Blocks: {reader.tag['n_block']}")
   Blocks: 48


   .. py:property:: file_path
      :type: pathlib.Path | None


      Get the TSN file path.


   .. py:attribute:: ts
      :value: None



   .. py:attribute:: ts_metadata
      :value: None



   .. py:method:: get_sign24(x: numpy.ndarray | list | int) -> numpy.ndarray

      Convert unsigned 24-bit integers to signed integers.

      Converts unsigned 24-bit values (0 to 16777215) to their signed
      equivalents (-8388608 to 8388607) by applying two's complement.

      :param x: Unsigned 24-bit integer value(s) to convert.
      :type x: ndarray or list or int

      :returns: Signed 24-bit integer value(s) as int32 array.
      :rtype: ndarray

      .. rubric:: Examples

      Convert a single positive value:

      >>> reader = MTUTSN()
      >>> reader.get_sign24(100)
      array([100], dtype=int32)

      Convert a single negative value (unsigned representation):

      >>> reader.get_sign24(16777215)  # -1 in 24-bit signed
      array([-1], dtype=int32)

      Convert an array:

      >>> values = np.array([0, 8388607, 8388608, 16777215])
      >>> reader.get_sign24(values)
      array([       0,  8388607, -8388608,       -1], dtype=int32)



   .. py:method:: read(file_path: str | pathlib.Path | None = None) -> None

      Read and parse a Phoenix MTU time series binary file.

      Reads complete time series data from legacy Phoenix MTU-5A instrument
      files (.TS2, .TS3, .TS4, .TS5) or V5-2000 system files (.TSL, .TSH).
      Each file contains multiple data blocks with 24-bit signed integer
      samples organized by channel.

      :param file_path: Path to the TSN file to read. If None, uses the current file_path
                        attribute. Default is None.
      :type file_path: str or Path or None, optional

      :returns: * **ts** (*ndarray*) -- Time series data array with shape (n_channels, total_samples).
                  Data type is float64. Each row represents one channel, and each
                  column is a time sample.
                * **tag** (*dict*) -- Metadata dictionary containing file information with keys:

                  - 'box_number' (int): Instrument serial number
                  - 'ts_type' (str): Instrument type ('MTU-5' or 'V5-2000')
                  - 'sample_rate' (int): Sampling frequency in Hz
                  - 'n_ch' (int): Number of channels
                  - 'n_scan' (int): Number of scans per data block
                  - 'start' (MTime): UTC timestamp of first sample
                  - 'ts_length' (float): Duration of each block in seconds
                  - 'n_block' (int): Total number of data blocks in file

      :raises EOFError: If the file is empty or cannot be read.
      :raises ValueError: If the file has an unsupported extension or channel count.
      :raises FileNotFoundError: If the specified file does not exist.

      .. rubric:: Examples

      Read a 3-channel TS3 file:

      >>> reader = MTUTSN()
      >>> ts, tag = reader.read('data/1690C16C.TS3')
      >>> print(f"Shape: {ts.shape}")
      Shape: (3, 86400)
      >>> print(f"Sample rate: {tag['sample_rate']} Hz")
      Sample rate: 24 Hz
      >>> print(f"Duration: {ts.shape[1] / tag['sample_rate']:.1f} seconds")
      Duration: 3600.0 seconds

      Read a 4-channel TS4 file:

      >>> reader = MTUTSN('data/1690C16C.TS4')
      >>> print(f"Channels: {reader.tag['n_ch']}")
      Channels: 4
      >>> print(f"Start time: {reader.tag['start'].isoformat()}")
      Start time: 2016-07-16T00:00:00+00:00

      Read and process data:

      >>> ts, tag = MTUTSN().read('data/station.TS5')
      >>> # Calculate statistics for each channel
      >>> for i in range(tag['n_ch']):
      ...     print(f"Ch{i} mean: {ts[i].mean():.2f}, std: {ts[i].std():.2f}")
      Ch0 mean: 123.45, std: 456.78
      Ch1 mean: -234.56, std: 567.89
      ...



   .. py:method:: to_runts(table_filepath: str | pathlib.Path | None = None, calibrate=True) -> mth5.timeseries.RunTS

      Create an MTUTable object from the TSN file and associated TBL file.

      :param table_filepath: Path to the corresponding TBL file.
      :type table_filepath: str or Path

      :returns: An MTUTable object containing metadata from the TBL file.
      :rtype: MTUTable

      .. rubric:: Examples

      >>> reader = MTUTSN('data/1690C16C.TS3')
      >>> mtu_table = reader.to_runts('data/1690C16C.TBL')
      >>> print(mtu_table.metadata)
      {...}



