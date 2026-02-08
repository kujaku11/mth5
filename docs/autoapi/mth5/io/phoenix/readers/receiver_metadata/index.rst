mth5.io.phoenix.readers.receiver_metadata
=========================================

.. py:module:: mth5.io.phoenix.readers.receiver_metadata

.. autoapi-nested-parse::

   Phoenix Geophysics receiver metadata parser for recmeta.json files.

   Created on Tue Jun 20 15:06:08 2023

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.receiver_metadata.PhoenixReceiverMetadata


Module Contents
---------------

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


