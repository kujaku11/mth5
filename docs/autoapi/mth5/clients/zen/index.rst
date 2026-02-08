mth5.clients.zen
================

.. py:module:: mth5.clients.zen


Classes
-------

.. autoapisummary::

   mth5.clients.zen.ZenClient


Module Contents
---------------

.. py:class:: ZenClient(data_path, sample_rates=[4096, 1024, 256], save_path=None, calibration_path=None, mth5_filename='from_zen.h5', **kwargs)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:property:: calibration_path

      Path to calibration data


   .. py:attribute:: collection


   .. py:attribute:: station_stem
      :value: None



   .. py:method:: get_run_dict()

      Get Run information

      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_survey(station_dict)

      get survey name from a dictionary of a single station of runs
      :param station_dict: DESCRIPTION
      :type station_dict: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: make_mth5_from_zen(survey_id=None, combine=True, **kwargs)

      Make an MTH5 from Phoenix files.  Split into runs, account for filters

      :param data_path: DESCRIPTION, defaults to None
      :type data_path: TYPE, optional
      :param sample_rates: DESCRIPTION, defaults to None
      :type sample_rates: TYPE, optional
      :param save_path: DESCRIPTION, defaults to None
      :type save_path: TYPE, optional
      :return: DESCRIPTION
      :rtype: TYPE




