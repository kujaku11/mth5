mth5.groups.estimate_dataset
============================

.. py:module:: mth5.groups.estimate_dataset

.. autoapi-nested-parse::

   Created on Thu Mar 10 09:02:16 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.groups.estimate_dataset.EstimateDataset


Module Contents
---------------

.. py:class:: EstimateDataset(dataset: h5py.Dataset, dataset_metadata: mt_metadata.transfer_functions.tf.statistical_estimate.StatisticalEstimate | None = None, write_metadata: bool = True, **kwargs: Any)

   Container for statistical estimates of transfer functions.

   This class holds multi-dimensional statistical estimates for transfer
   functions with full metadata management. Estimates are stored as HDF5
   datasets with dimensions for period, output channels, and input channels.

   :param dataset: HDF5 dataset containing the statistical estimate data.
   :type dataset: h5py.Dataset
   :param dataset_metadata: Metadata object for the estimate. If provided and write_metadata is True,
                            the metadata will be written to the HDF5 attributes. Defaults to None.
   :type dataset_metadata: mt_metadata.transfer_functions.tf.StatisticalEstimate, optional
   :param write_metadata: If True, write metadata to the HDF5 dataset attributes. Defaults to True.
   :type write_metadata: bool, optional
   :param \*\*kwargs: Additional keyword arguments (reserved for future use).
   :type \*\*kwargs: Any

   .. attribute:: hdf5_dataset

      Weak reference to the HDF5 dataset.

      :type: h5py.Dataset

   .. attribute:: metadata

      Metadata container for the estimate.

      :type: StatisticalEstimate

   .. attribute:: logger

      Logger instance for reporting messages.

      :type: loguru.logger

   :raises MTH5Error: If dataset_metadata is provided but is not of type StatisticalEstimate
       or a compatible metadata class.
   :raises TypeError: If input data cannot be converted to numpy array or has wrong dtype/shape.

   .. rubric:: Notes

   The estimate data is stored in 3D form with shape:
   (n_periods, n_output_channels, n_input_channels)

   Metadata is automatically synchronized between the pydantic model and
   HDF5 attributes on initialization and after any modifications.

   .. rubric:: Examples

   Create an estimate dataset from an HDF5 group:

   >>> import h5py
   >>> import numpy as np
   >>> from mt_metadata.transfer_functions.tf.statistical_estimate import StatisticalEstimate
   >>> # Create HDF5 file with estimate dataset
   >>> with h5py.File('estimate.h5', 'w') as f:
   ...     # Create dataset with shape (10 periods, 2 outputs, 2 inputs)
   ...     data = np.random.rand(10, 2, 2)
   ...     dset = f.create_dataset('estimate', data=data)
   ...     # Create EstimateDataset
   ...     est = EstimateDataset(dset, write_metadata=True)

   Convert estimate to xarray and back:

   >>> periods = np.logspace(-3, 3, 10)  # 10 periods from 1e-3 to 1e3 s
   >>> xr_data = est.to_xarray(periods)
   >>> # Modify xarray coordinates
   >>> new_xr = xr_data.rename({'output': 'new_output', 'input': 'new_input'})
   >>> est.from_xarray(new_xr)  # Load modified data back

   Access estimate data in different formats:

   >>> # Get numpy array
   >>> np_data = est.to_numpy()
   >>> print(np_data.shape)  # (10, 2, 2)
   >>> # Get xarray with proper coordinates
   >>> xr_data = est.to_xarray(periods)
   >>> print(xr_data.dims)  # ('period', 'output', 'input')


   .. py:attribute:: logger


   .. py:attribute:: metadata


   .. py:method:: read_metadata() -> None

      Read metadata from HDF5 attributes into metadata container.

      Reads all attributes from the HDF5 dataset and loads them into
      the internal metadata object for validation and access.

      :rtype: None

      .. rubric:: Notes

      This is automatically called during initialization if 'mth5_type'
      attribute exists in the HDF5 dataset.

      .. rubric:: Examples

      Reload metadata from HDF5 after external modification:

      >>> # Metadata was modified in HDF5
      >>> est.read_metadata()  # Reload changes
      >>> print(est.metadata.name)  # Access updated name



   .. py:method:: write_metadata() -> None

      Write metadata from container to HDF5 dataset attributes.

      Converts the pydantic metadata model to a dictionary and writes
      each field as an HDF5 attribute. Values are converted to appropriate
      numpy types for compatibility.

      :rtype: None

      .. rubric:: Notes

      All existing attributes with the same names will be overwritten.
      This is called automatically during initialization and after
      metadata updates.

      .. rubric:: Examples

      Save updated metadata to HDF5:

      >>> est.metadata.name = "Updated Estimate"
      >>> est.write_metadata()  # Persist to file
      >>> # Verify write
      >>> print(est.hdf5_dataset.attrs['name'])
      b'Updated Estimate'



   .. py:method:: replace_dataset(new_data_array: numpy.ndarray) -> None

      Replace entire dataset with new data.

      Resizes the HDF5 dataset if necessary and replaces all data.
      Converts input to numpy array if needed.

      :param new_data_array: New estimate data to store. Should have shape
                             (n_periods, n_output_channels, n_input_channels).
      :type new_data_array: np.ndarray

      :rtype: None

      :raises TypeError: If input cannot be converted to numpy array.

      .. rubric:: Notes

      If new data has different shape, HDF5 dataset will be resized.
      This is generally safe but may fragment the HDF5 file.

      .. rubric:: Examples

      Replace estimate with new data:

      >>> import numpy as np
      >>> new_estimate = np.random.rand(10, 2, 2)  # 10 periods, 2 channels
      >>> est.replace_dataset(new_estimate)
      >>> print(est.to_numpy().shape)
      (10, 2, 2)

      Replace with data from list (auto-converted to array):

      >>> data_list = [[[1, 2], [3, 4]]] * 5  # 5 periods
      >>> est.replace_dataset(data_list)
      >>> est.to_numpy().shape
      (5, 2, 2)



   .. py:method:: to_xarray(period: numpy.ndarray | list) -> xarray.DataArray

      Convert estimate to xarray DataArray.

      Creates an xarray DataArray with proper coordinates for periods,
      output channels, and input channels. Includes metadata as attributes.

      :param period: Period values for coordinate. Should have length equal to
                     estimate first dimension (n_periods).
      :type period: np.ndarray | list

      :returns: DataArray with dimensions (period, output, input) and
                coordinates from metadata.
      :rtype: xr.DataArray

      .. rubric:: Notes

      Metadata changes in xarray are not validated and will not be
      synchronized back to HDF5 without explicit call to from_xarray().
      Data is loaded entirely into memory.

      .. rubric:: Examples

      Convert to xarray with logarithmic period spacing:

      >>> import numpy as np
      >>> periods = np.logspace(-2, 3, 10)  # 10 periods from 0.01 to 1000
      >>> xr_data = est.to_xarray(periods)
      >>> print(xr_data.dims)
      ('period', 'output', 'input')
      >>> print(xr_data.coords['period'].values)
      [1.00e-02 3.16e-02 ... 1.00e+03]

      Select data by period range:

      >>> subset = xr_data.sel(period=slice(0.1, 100))
      >>> print(subset.shape)
      (8, 2, 2)



   .. py:method:: to_numpy() -> numpy.ndarray

      Convert estimate to numpy array.

      Returns the HDF5 dataset as a numpy array. Data is loaded
      entirely into memory.

      :returns: 3D array with shape (n_periods, n_output_channels, n_input_channels).
      :rtype: np.ndarray

      .. rubric:: Notes

      For large estimates, this loads all data into RAM. Consider using
      HDF5 slicing for memory-efficient access.

      .. rubric:: Examples

      Get full estimate as numpy array:

      >>> data = est.to_numpy()
      >>> print(data.shape)
      (10, 2, 2)
      >>> print(data.dtype)
      float64

      Access specific period and channels:

      >>> data = est.to_numpy()
      >>> # Get first 5 periods, output channel 0, input channel 1
      >>> subset = data[:5, 0, 1]
      >>> print(subset.shape)
      (5,)



   .. py:method:: from_numpy(new_estimate: numpy.ndarray) -> None

      Load estimate data from numpy array.

      Validates dtype and shape compatibility, resizes dataset if needed,
      and stores the data.

      :param new_estimate: Estimate data to load. Must be convertible to numpy array.
                           Preferred shape: (n_periods, n_output_channels, n_input_channels).
      :type new_estimate: np.ndarray

      :rtype: None

      :raises TypeError: If dtype doesn't match existing dataset or input cannot
          be converted to numpy array.

      .. rubric:: Notes

      'data' is a built-in Python function and cannot be used as parameter name.
      The dataset will be resized if shape doesn't match.

      .. rubric:: Examples

      Load estimate from numpy array:

      >>> import numpy as np
      >>> new_data = np.random.rand(5, 2, 2)
      >>> est.from_numpy(new_data)
      >>> print(est.to_numpy().shape)
      (5, 2, 2)

      Load with automatic dtype conversion:

      >>> float_data = np.array([[[1.0, 2.0]]], dtype=np.float64)
      >>> est.from_numpy(float_data)



   .. py:method:: from_xarray(data: xarray.DataArray) -> None

      Load estimate data from xarray DataArray.

      Updates metadata from xarray coordinates and attributes, then
      stores the data.

      :param data: DataArray containing estimate. Expected dimensions:
                   (period, output, input).
      :type data: xr.DataArray

      :rtype: None

      .. rubric:: Notes

      This will update output_channels, input_channels, name, and data_type
      from the xarray object. All changes are persisted to HDF5.

      .. rubric:: Examples

      Load estimate from modified xarray:

      >>> xr_data = est.to_xarray(periods)
      >>> # Modify data and metadata
      >>> modified = xr_data * 2  # Scale by 2
      >>> est.from_xarray(modified)
      >>> print(est.to_numpy()[0, 0, 0])  # Verify scale

      Rename channels and reload:

      >>> xr_data = est.to_xarray(periods)
      >>> new_xr = xr_data.rename({
      ...     'output': ['Ex', 'Ey'],
      ...     'input': ['Bx', 'By']
      ... })
      >>> est.from_xarray(new_xr)
      >>> print(est.metadata.output_channels)
      ['Ex', 'Ey']



