mth5.groups.transfer_function
=============================

.. py:module:: mth5.groups.transfer_function


Classes
-------

.. autoapisummary::

   mth5.groups.transfer_function.TransferFunctionsGroup
   mth5.groups.transfer_function.TransferFunctionGroup


Module Contents
---------------

.. py:class:: TransferFunctionsGroup(group: Any, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for transfer functions under a station.

   Each child group is a single transfer function estimation managed by
   :class:`TransferFunctionGroup`.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> m5 = mth5.MTH5()
   >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
   >>> station = m5.stations_group.add_station("mt01")
   >>> tf_group = station.transfer_functions_group
   >>> tf_group.groups_list
   []


   .. py:method:: tf_summary(as_dataframe: bool = True) -> pandas.DataFrame | numpy.ndarray

      Summarize transfer functions stored for the station.

      :param as_dataframe: If ``True`` return a pandas DataFrame, otherwise a NumPy structured array.
      :type as_dataframe: bool, default True

      :returns: Summary rows including station reference, location, and TF metadata.
      :rtype: pandas.DataFrame or numpy.ndarray

      .. rubric:: Examples

      >>> summary = tf_group.tf_summary()
      >>> summary.columns[:4].tolist()  # doctest: +SKIP
      ['station_hdf5_reference', 'station', 'latitude', 'longitude']



   .. py:method:: add_transfer_function(name: str, tf_object: mt_metadata.transfer_functions.core.TF | None = None) -> TransferFunctionGroup

      Add a transfer function group under this station.

      :param name: Transfer function identifier.
      :type name: str
      :param tf_object: Transfer function instance to seed metadata and datasets.
      :type tf_object: TF, optional

      :returns: Wrapper for the created or existing transfer function.
      :rtype: TransferFunctionGroup

      .. rubric:: Examples

      >>> tf_group = station.transfer_functions_group
      >>> _ = tf_group.add_transfer_function("mt01_4096")



   .. py:method:: get_transfer_function(tf_id: str) -> TransferFunctionGroup

      Return an existing transfer function by id.

      :param tf_id: Name of the transfer function.
      :type tf_id: str

      :returns: Wrapper for the requested transfer function.
      :rtype: TransferFunctionGroup

      :raises MTH5Error: If the transfer function does not exist.

      .. rubric:: Examples

      >>> existing = station.transfer_functions_group.get_transfer_function("mt01_4096")
      >>> existing.name  # doctest: +SKIP
      'mt01_4096'



   .. py:method:: remove_transfer_function(tf_id: str) -> None

      Delete a transfer function reference from the station.

      :param tf_id: Transfer function name.
      :type tf_id: str

      .. rubric:: Notes

      HDF5 deletion removes the reference only; storage is not reclaimed.

      .. rubric:: Examples

      >>> tf_group.remove_transfer_function("mt01_4096")



   .. py:method:: get_tf_object(tf_id: str) -> mt_metadata.transfer_functions.core.TF

      Return a populated :class:`mt_metadata.transfer_functions.core.TF`.

      :param tf_id: Transfer function name to convert.
      :type tf_id: str

      :returns: Transfer function populated with metadata and estimates.
      :rtype: mt_metadata.transfer_functions.core.TF

      .. rubric:: Examples

      >>> tf_obj = tf_group.get_tf_object("mt01_4096")  # doctest: +SKIP



.. py:class:: TransferFunctionGroup(group: Any, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Wrapper for a single transfer function estimation.


   .. py:method:: has_estimate(estimate: str) -> bool

      Return ``True`` if an estimate exists and is populated.



   .. py:property:: period
      :type: numpy.ndarray | None


      Return period array stored in ``period`` dataset, if present.


   .. py:method:: add_statistical_estimate(estimate_name: str, estimate_data: numpy.ndarray | xarray.DataArray | None = None, estimate_metadata: mt_metadata.transfer_functions.tf.statistical_estimate.StatisticalEstimate | None = None, max_shape: tuple[int | None, int | None, int | None] = (None, None, None), chunks: bool = True, **kwargs: Any) -> mth5.groups.EstimateDataset

      Add a statistical estimate dataset.

      :param estimate_name: Dataset name.
      :type estimate_name: str
      :param estimate_data: Estimate values; if ``None`` a placeholder array is created.
      :type estimate_data: numpy.ndarray or xarray.DataArray, optional
      :param estimate_metadata: Metadata describing the estimate.
      :type estimate_metadata: StatisticalEstimate, optional
      :param max_shape: Maximum shape for resizable datasets.
      :type max_shape: tuple of int or None, default (None, None, None)
      :param chunks: Chunking flag forwarded to HDF5 dataset creation.
      :type chunks: bool, default True

      :returns: Wrapper combining dataset and metadata.
      :rtype: EstimateDataset

      :raises TypeError: If ``estimate_data`` is not array-like.

      .. rubric:: Examples

      >>> est = tf_group.add_statistical_estimate("transfer_function")
      >>> isinstance(est, EstimateDataset)
      True



   .. py:method:: get_estimate(estimate_name: str) -> mth5.groups.EstimateDataset

      Return a statistical estimate dataset by name.



   .. py:method:: remove_estimate(estimate_name: str) -> None

      Remove a statistical estimate dataset reference.



   .. py:method:: to_tf_object() -> mt_metadata.transfer_functions.core.TF

      Convert this group into a populated :class:`TF` object.

      :returns: TF instance with survey, station, runs, channels, period, and
                estimate datasets applied.
      :rtype: mt_metadata.transfer_functions.core.TF

      :raises ValueError: If no period dataset is present.

      .. rubric:: Examples

      >>> tf_obj = tf_group.to_tf_object()  # doctest: +SKIP



   .. py:method:: from_tf_object(tf_obj: mt_metadata.transfer_functions.core.TF, update_metadata: bool = True) -> None

      Populate datasets from a :class:`TF` object.

      :param tf_obj: Transfer function object containing estimates and metadata.
      :type tf_obj: TF
      :param update_metadata: If ``True`` write transfer function metadata to HDF5.
      :type update_metadata: bool, default True

      :raises ValueError: If ``tf_obj`` is not a ``TF`` instance.

      .. rubric:: Examples

      >>> tf_group.from_tf_object(tf_obj)  # doctest: +SKIP



