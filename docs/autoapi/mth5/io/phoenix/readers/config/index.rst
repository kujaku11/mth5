mth5.io.phoenix.readers.config
==============================

.. py:module:: mth5.io.phoenix.readers.config

.. autoapi-nested-parse::

   Created on Fri Jun 10 07:52:03 2022

   :author: Jared Peacock

   :license: MIT



Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.config.PhoenixConfig


Module Contents
---------------

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



