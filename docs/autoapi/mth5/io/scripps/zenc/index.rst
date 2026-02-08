mth5.io.scripps.zenc
====================

.. py:module:: mth5.io.scripps.zenc

.. autoapi-nested-parse::

   Created on Thu Jan 25 11:36:55 2024

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.scripps.zenc.ZENC


Module Contents
---------------

.. py:class:: ZENC(channel_map)

   Deal with .zenc files, which are apparently used to process data in EMTF.
   It was specifically built for processing ZEN data in EMTF, but should
   work regardless of data logger.

   The format is a header and then n_channels x n_samples of float32 values

   This class will read/write .zenc files.

   You need to input the path to an existing or new MTH5 file and a
   channel map to read/write.

   The `channel_map` needs to be in the form

   .. code-block::
       channel_map = {
           "channel_1_name":
               {"survey": survey_name,
                "station": station_name,
                "run": run_name,
                "channel": channel_name,
                "channel_number": channel_number},
           "channel_2_name":
               {"survey": survey_name,
                "station": station_name,
                "run": run_name,
                "channel": channel_name,
                "channel_number": channel_number},
           ...
               }




   .. py:attribute:: logger


   .. py:property:: channel_map


   .. py:method:: to_zenc(mth5_file, channel_map=None)

      write out a .zenc file

      :param mth5_file: DESCRIPTION
      :type mth5_file: TYPE
      :param channel_map: DESCRIPTION, defaults to None
      :type channel_map: TYPE, optional
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_run_metadata(run_ts)

      get run metadata from RunTS object

      4096
      version: 1.0
      boxNumber: 74
      samplingFrequency: 4096
      timeDataStart: 2021-07-23 08:00:14
      timeDataEnd: 2021-07-23 08:14:58
      latitude: 58.22444
      longitude: -155.66579
      altitude: 251.30000
      rx_stn: 1
      TxFreq: 0
      TxDuty: inf
      numChans: 5

      :param run_ts: DESCRIPTION
      :type run_ts: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




