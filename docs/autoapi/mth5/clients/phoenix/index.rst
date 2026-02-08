mth5.clients.phoenix
====================

.. py:module:: mth5.clients.phoenix


Classes
-------

.. autoapisummary::

   mth5.clients.phoenix.PhoenixClient


Module Contents
---------------

.. py:class:: PhoenixClient(data_path: str | pathlib.Path, sample_rates: list[int] = [150, 24000], save_path: str | pathlib.Path | None = None, receiver_calibration_dict: dict | str | pathlib.Path = {}, sensor_calibration_dict: dict | str | pathlib.Path = {}, mth5_filename: str = 'from_phoenix.h5', **kwargs: dict)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:property:: receiver_calibration_dict
      :type: dict


      Receiver calibrations.

      :returns: Dictionary mapping receiver IDs to calibration file paths.
      :rtype: dict

      .. rubric:: Examples

      >>> client = PhoenixClient('data/path')
      >>> client.receiver_calibration_dict = {'RX001': Path('RX001_rxcal.json')}
      >>> client.receiver_calibration_dict
      {'RX001': Path('RX001_rxcal.json')}


   .. py:property:: sensor_calibration_dict
      :type: dict


      Sensor calibration dictionary.

      :returns: Dictionary mapping sensor IDs to PhoenixCalibration objects.
      :rtype: dict

      .. rubric:: Examples

      >>> client = PhoenixClient('data/path')
      >>> client.sensor_calibration_dict = {'H001': PhoenixCalibration('H001_scal.json')}
      >>> client.sensor_calibration_dict['H001']  # doctest: +SKIP
      <PhoenixCalibration object>


   .. py:attribute:: collection


   .. py:method:: make_mth5_from_phoenix(**kwargs: dict) -> str | pathlib.Path | None

      Make an MTH5 from Phoenix files.

      Split into runs, account for filters. Updates the MTH5 file with Phoenix data.

      :param \*\*kwargs: Optional keyword arguments to override instance attributes.
      :type \*\*kwargs: dict

      :returns: Path to the saved MTH5 file.
      :rtype: str, Path, or None

      .. rubric:: Examples

      >>> client = PhoenixClient('data/path', save_path='output.h5')
      >>> client.make_mth5_from_phoenix()
      'output.h5'



