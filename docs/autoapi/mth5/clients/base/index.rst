mth5.clients.base
=================

.. py:module:: mth5.clients.base

.. autoapi-nested-parse::

   Created on Fri Oct 11 11:36:26 2024

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.clients.base.ClientBase


Module Contents
---------------

.. py:class:: ClientBase(data_path: Union[str, pathlib.Path], sample_rates: Sequence[float] = [1], save_path: Optional[Union[str, pathlib.Path]] = None, mth5_filename: str = 'from_client.h5', **kwargs: Any)

   .. py:attribute:: logger


   .. py:property:: data_path
      :type: pathlib.Path


      Path to data directory.

      :returns: Path to the data directory.
      :rtype: Path

      .. rubric:: Examples

      >>> client = ClientBase(data_path="./data")
      >>> client.data_path
      PosixPath('data')


   .. py:attribute:: mth5_filename
      :value: 'from_client.h5'



   .. py:property:: sample_rates
      :type: list[float]


      List of sample rates to look for.

      :returns: Sample rates.
      :rtype: list of float

      .. rubric:: Examples

      >>> client = ClientBase(data_path="./data", sample_rates=[1, 8, 256])
      >>> client.sample_rates
      [1.0, 8.0, 256.0]


   .. py:property:: save_path
      :type: pathlib.Path


      Path to save the mth5 file.

      :returns: Full path to the mth5 file.
      :rtype: Path

      .. rubric:: Examples

      >>> client = ClientBase(data_path="./data", save_path="./output", mth5_filename="test.h5")
      >>> client.save_path
      PosixPath('output/test.h5')


   .. py:attribute:: interact
      :value: False



   .. py:attribute:: mth5_version
      :value: '0.2.0'



   .. py:attribute:: h5_compression
      :value: 'gzip'



   .. py:attribute:: h5_compression_opts
      :value: 4



   .. py:attribute:: h5_shuffle
      :value: True



   .. py:attribute:: h5_fletcher32
      :value: True



   .. py:attribute:: h5_data_level
      :value: 1



   .. py:attribute:: mth5_file_mode
      :value: 'w'



   .. py:attribute:: collection
      :type:  Optional[Any]
      :value: None



   .. py:property:: h5_kwargs
      :type: dict[str, Any]


      Dictionary of HDF5 keyword arguments for file creation.

      :returns: Dictionary of HDF5 file creation parameters.
      :rtype: dict

      .. rubric:: Examples

      >>> client = ClientBase(data_path="./data")
      >>> client.h5_kwargs["compression"]
      'gzip'


   .. py:method:: get_run_dict() -> Any

      Get run information from the collection.

      :returns: Run information as returned by the collection's get_runs method.
      :rtype: Any

      .. rubric:: Examples

      >>> client = ClientBase(data_path="./data")
      >>> client.collection = ...  # assign a collection with get_runs method
      >>> client.get_run_dict()
      ...



