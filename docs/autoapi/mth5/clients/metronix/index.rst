mth5.clients.metronix
=====================

.. py:module:: mth5.clients.metronix

.. autoapi-nested-parse::

   Created on Wed Nov 27 11:23:50 2024

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.clients.metronix.MetronixClient


Module Contents
---------------

.. py:class:: MetronixClient(data_path, sample_rates=[128], save_path=None, calibration_path=None, mth5_filename='from_metronix.h5', **kwargs)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:attribute:: calibration_path
      :value: None



   .. py:attribute:: collection


   .. py:method:: get_run_dict(run_name_zeros=0)

      get run information

      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_survey_id(station_dict)

      get survey name from a dictionary of a single station of runs
      :param station_dict: DESCRIPTION
      :type station_dict: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: set_station_metadata(station_dict, station_group)

      set station group metadata from information in the station dict

      :param station_dict: DESCRIPTION
      :type station_dict: TYPE
      :param station_group: DESCRIPTION
      :type station_group: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: make_mth5_from_metronix(run_name_zeros=0, **kwargs)

      Create an MTH5 from new ATSS + JSON style Metronix data.

      :param **kwargs: DESCRIPTION
      :type **kwargs: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




