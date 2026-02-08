mth5.io.metronix.metronix_metadata
==================================

.. py:module:: mth5.io.metronix.metronix_metadata

.. autoapi-nested-parse::

   Metronix metadata parsing utilities.

   This module provides classes for parsing and managing metadata from Metronix
   ATSS (Audio Time Series System) files and associated JSON metadata files.

   Classes
   -------
   MetronixFileNameMetadata
       Parse metadata from Metronix filename conventions
   MetronixChannelJSON
       Read and parse Metronix JSON metadata files

   Created on Fri Nov 22 13:23:42 2024

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.metronix.metronix_metadata.MetronixFileNameMetadata
   mth5.io.metronix.metronix_metadata.MetronixChannelJSON


Module Contents
---------------

.. py:class:: MetronixFileNameMetadata(fn: Union[str, pathlib.Path, None] = None, **kwargs: Any)

   Parse and manage metadata from Metronix filename conventions.

   This class extracts metadata information from Metronix ATSS filenames
   including system information, channel details, and file properties.

   :param fn: Path to Metronix file, by default None
   :type fn: Union[str, Path, None], optional
   :param \*\*kwargs: Additional keyword arguments (currently unused)

   .. attribute:: system_number

      System identification number

      :type: str or None

   .. attribute:: system_name

      Name of the system

      :type: str or None

   .. attribute:: channel_number

      Channel number (parsed from C## format)

      :type: int or None

   .. attribute:: component

      Component designation (e.g., 'ex', 'ey', 'hx', 'hy', 'hz')

      :type: str or None

   .. attribute:: sample_rate

      Sampling rate in Hz

      :type: float or None

   .. attribute:: file_type

      Type of file ('metadata' or 'timeseries')

      :type: str or None


   .. py:attribute:: system_number
      :type:  str | None
      :value: None



   .. py:attribute:: system_name
      :type:  str | None
      :value: None



   .. py:attribute:: channel_number
      :type:  int | None
      :value: None



   .. py:attribute:: component
      :type:  str | None
      :value: None



   .. py:attribute:: sample_rate
      :type:  float | None
      :value: None



   .. py:attribute:: file_type
      :type:  str | None
      :value: None



   .. py:property:: fn
      :type: pathlib.Path | None


      Get the file path.

      :returns: File path object or None if not set
      :rtype: Path or None


   .. py:property:: fn_exists
      :type: bool


      Check if the file exists.

      :returns: True if file exists, False otherwise
      :rtype: bool


   .. py:property:: file_size
      :type: int


      Get file size in bytes.

      :returns: File size in bytes, 0 if file is None
      :rtype: int


   .. py:property:: n_samples
      :type: float


      Get estimated number of samples in file.

      Assumes 8 bytes per sample (double precision).

      :returns: Estimated number of samples
      :rtype: float


   .. py:property:: duration
      :type: float


      Get estimated duration of the file in seconds.

      :returns: Duration in seconds
      :rtype: float


.. py:class:: MetronixChannelJSON(fn: Union[str, pathlib.Path, None] = None, **kwargs: Any)

   Bases: :py:obj:`MetronixFileNameMetadata`


   Read and parse Metronix JSON metadata files.

   This class extends MetronixFileNameMetadata to handle JSON metadata
   files containing channel configuration and calibration information.

   :param fn: Path to Metronix JSON file, by default None
   :type fn: Union[str, Path, None], optional
   :param \*\*kwargs: Additional keyword arguments passed to parent class

   .. attribute:: metadata

      Parsed JSON metadata as a SimpleNamespace object

      :type: SimpleNamespace or None


   .. py:attribute:: metadata
      :type:  types.SimpleNamespace | None
      :value: None



   .. py:method:: read(fn: Union[str, pathlib.Path, None] = None) -> None

      Read JSON metadata from file.

      :param fn: Path to JSON file, by default None (uses self.fn)
      :type fn: Union[str, Path, None], optional

      :raises IOError: If JSON file cannot be found



   .. py:method:: get_channel_metadata() -> Union[mt_metadata.timeseries.Electric, mt_metadata.timeseries.Magnetic, None]

      Translate to mt_metadata.timeseries.Channel object.

      Creates either Electric or Magnetic metadata objects based on the
      component type and applies calibration filters.

      :returns: mt_metadata object based on component type, or None if no metadata
      :rtype: Union[Electric, Magnetic, None]

      :raises ValueError: If component type is not recognized



   .. py:method:: get_sensor_response_filter() -> mt_metadata.timeseries.filters.FrequencyResponseTableFilter | None

      Get the sensor response frequency-amplitude-phase filter.

      Creates a FrequencyResponseTableFilter from the sensor calibration
      data stored in the JSON metadata.

      :returns: Sensor response filter if calibration data exists, None otherwise
      :rtype: FrequencyResponseTableFilter or None



   .. py:method:: get_channel_response() -> mt_metadata.timeseries.filters.ChannelResponse

      Get all filters needed to calibrate the data.

      :returns: Channel response object containing all calibration filters
      :rtype: ChannelResponse



