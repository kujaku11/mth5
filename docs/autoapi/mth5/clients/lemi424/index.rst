mth5.clients.lemi424
====================

.. py:module:: mth5.clients.lemi424

.. autoapi-nested-parse::

   Created on Fri Oct 11 10:57:54 2024

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.clients.lemi424.LEMI424Client


Module Contents
---------------

.. py:class:: LEMI424Client(data_path: Union[str, pathlib.Path], save_path: Optional[Union[str, pathlib.Path]] = None, mth5_filename: str = 'from_lemi424.h5', **kwargs: Any)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:attribute:: collection


   .. py:method:: make_mth5_from_lemi424(survey_id: str, station_id: str, **kwargs: Any) -> pathlib.Path

      Create an MTH5 file from LEMI 424 long period data.

      :param survey_id: Survey identifier.
      :type survey_id: str
      :param station_id: Station identifier.
      :type station_id: str
      :param \*\*kwargs: Additional keyword arguments to set as attributes.
      :type \*\*kwargs: Any

      :returns: Path to the created mth5 file.
      :rtype: Path

      .. rubric:: Examples

      >>> client = LEMI424Client(data_path="./data")
      >>> client.make_mth5_from_lemi424("SURVEY1", "ST01")
      PosixPath('data/from_lemi424.h5')



