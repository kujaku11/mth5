mth5.io.phoenix.readers.header
==============================

.. py:module:: mth5.io.phoenix.readers.header

.. autoapi-nested-parse::

   Adopted from TimeSeries reader, making all attributes properties for easier
   reading and testing.

   Module to read and parse native Phoenix Geophysics data formats of the MTU-5C Family

   This module implements Streamed readers for segmented-decimated continuus-decimated
   and native sampling rate time series formats of the MTU-5C family.

   :author: Jorge Torres-Solis

   Revised 2022 by J. Peacock



Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.header.Header


Module Contents
---------------

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



