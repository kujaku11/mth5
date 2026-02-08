mth5.io.phoenix.readers.calibrations
====================================

.. py:module:: mth5.io.phoenix.readers.calibrations

.. autoapi-nested-parse::

   Created on Thu Jun 15 15:21:35 2023

   @author: jpeacock

   Calibrations can come in json files.  the JSON file includes filters
   for all lowpass filters, so you need to match the lowpass filter used in the
   setup with the lowpass filter.  Then you need to add the dipole length and
   sensor calibrations.



Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.calibrations.PhoenixCalibration


Module Contents
---------------

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



