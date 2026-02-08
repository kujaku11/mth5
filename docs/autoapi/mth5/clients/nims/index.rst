mth5.clients.nims
=================

.. py:module:: mth5.clients.nims


Classes
-------

.. autoapisummary::

   mth5.clients.nims.NIMSClient


Module Contents
---------------

.. py:class:: NIMSClient(data_path: str | pathlib.Path, sample_rates: list[int] = [1, 8], save_path: str | pathlib.Path | None = None, calibration_path: str | pathlib.Path | None = None, mth5_filename: str = 'from_nims.h5', **kwargs)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:property:: calibration_path
      :type: pathlib.Path | None


      Path to calibration data.

      :returns: Path to calibration file, or None if not set.
      :rtype: Path or None

      .. rubric:: Examples

      >>> client = NIMSClient('data_dir')
      >>> client.calibration_path = 'calib.dat'
      >>> print(client.calibration_path)
      PosixPath('calib.dat')


   .. py:attribute:: collection


   .. py:method:: get_run_dict() -> dict

      Get run information from the NIMS collection.

      :returns: Dictionary of run information.
      :rtype: dict

      .. rubric:: Examples

      >>> client = NIMSClient('data_dir')
      >>> runs = client.get_run_dict()
      >>> print(list(runs.keys()))
      ['station1', 'station2']



   .. py:method:: get_survey(station_dict: dict) -> str

      Get survey name from a dictionary of a single station of runs.

      :param station_dict: Dictionary of runs for a station.
      :type station_dict: dict

      :returns: Survey name.
      :rtype: str

      .. rubric:: Examples

      >>> client = NIMSClient('data_dir')
      >>> runs = client.get_run_dict()
      >>> survey = client.get_survey(runs['station1'])
      >>> print(survey)
      'survey_name'



   .. py:method:: make_mth5_from_nims(survey_id: str = 'default_survey', combine: bool = True, **kwargs) -> str | pathlib.Path

      Make an MTH5 file from Phoenix NIMS files. Splits into runs, accounts for filters.

      :param survey_id: Survey identifier. Default is "default_survey".
      :type survey_id: str, optional
      :param combine: Whether to combine runs. Default is True.
      :type combine: bool, optional
      :param \*\*kwargs: Additional keyword arguments to set as attributes.

      :returns: Path to the saved MTH5 file.
      :rtype: str or Path

      .. rubric:: Examples

      >>> client = NIMSClient('data_dir')
      >>> mth5_path = client.make_mth5_from_nims(survey_id='survey1')
      >>> print(mth5_path)
      'output_dir/from_nims.h5'



