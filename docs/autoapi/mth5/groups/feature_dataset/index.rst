mth5.groups.feature_dataset
===========================

.. py:module:: mth5.groups.feature_dataset

.. autoapi-nested-parse::

   Created on Thu Mar 10 09:02:16 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.groups.feature_dataset.FeatureChannelDataset


Module Contents
---------------

.. py:class:: FeatureChannelDataset(dataset: h5py.Dataset, dataset_metadata: Optional[mt_metadata.features.FeatureDecimationChannel] = None, **kwargs)

   Container for multi-dimensional Fourier Coefficients organized by time and frequency.

   This class manages Fourier Coefficient data with frequency band organization,
   similar to FCDataset but with enhanced band tracking capabilities. The data array
   is organized with the following assumptions:

   1. Data are grouped into frequency bands
   2. Data are uniformly sampled in time (uniform FFT moving window step size)

   The dataset tracks temporal evolution of frequency content across multiple windows,
   making it suitable for time-frequency analysis of geophysical signals.

   :param dataset: HDF5 dataset containing the Fourier coefficient data.
   :type dataset: h5py.Dataset
   :param dataset_metadata: Metadata for the dataset. See :class:`mt_metadata.features.FeatureDecimationChannel`.
                            If provided, must be of the same type as the internal metadata class.
                            Default is None.
   :type dataset_metadata: FeatureDecimationChannel, optional
   :param \*\*kwargs: Additional keyword arguments for future extensibility.

   .. attribute:: hdf5_dataset

      Reference to the HDF5 dataset.

      :type: h5py.Dataset

   .. attribute:: metadata

      Metadata container with the following attributes:

      - name : str
          Dataset name
      - time_period.start : datetime
          Start time of the data acquisition
      - time_period.end : datetime
          End time of the data acquisition
      - sample_rate_window_step : float
          Sample rate of the time window stepping (Hz)
      - frequency_min : float
          Minimum frequency in the band (Hz)
      - frequency_max : float
          Maximum frequency in the band (Hz)
      - units : str
          Physical units of the coefficient data
      - component : str
          Component identifier (e.g., 'Ex', 'Hy')
      - sample_rate_decimation_level : int
          Decimation level applied to acquire this data

      :type: FeatureDecimationChannel

   :raises MTH5Error: If dataset_metadata type does not match the expected FeatureDecimationChannel type.

   .. rubric:: Examples

   >>> import h5py
   >>> from mt_metadata.features import FeatureDecimationChannel
   >>> from mth5.groups.feature_dataset import FeatureChannelDataset

   Create a feature dataset from an HDF5 group:

   >>> with h5py.File('data.h5', 'r') as f:
   ...     h5_dataset = f['feature_group']['Ex']
   ...     feature = FeatureChannelDataset(h5_dataset)
   ...     print(f"Time windows: {feature.n_windows}")
   ...     print(f"Frequencies: {feature.n_frequencies}")

   Access time and frequency arrays:

   >>> time_array = feature.time
   >>> freq_array = feature.frequency
   >>> data_array = feature.to_numpy()


   .. py:attribute:: logger


   .. py:attribute:: metadata


   .. py:method:: read_metadata() -> None

      Read metadata from the HDF5 file into the metadata container.

      This method loads all attributes from the HDF5 dataset into the
      metadata container, enabling validation and type checking.

      .. rubric:: Examples

      >>> feature.read_metadata()
      >>> print(feature.metadata.component)
      'Ex'



   .. py:method:: write_metadata() -> None

      Write metadata from the metadata container to the HDF5 attributes.

      This method serializes the metadata container and writes all metadata
      as attributes to the HDF5 dataset. Raises exceptions are caught for
      read-only files.

      .. rubric:: Examples

      >>> feature.metadata.component = 'Ey'
      >>> feature.write_metadata()



   .. py:property:: n_windows
      :type: int


      Get the number of time windows in the dataset.

      :returns: Number of time windows (first dimension of the dataset).
      :rtype: int


   .. py:property:: time
      :type: numpy.ndarray


      Get the time array for each window.

      Returns an array of datetime64 values representing the start time
      of each time window. The time spacing is determined by the sample
      rate of the window stepping.

      :returns: Array of datetime64 values with shape (n_windows,) representing
                the start time of each window.
      :rtype: np.ndarray

      .. rubric:: Examples

      >>> time_array = feature.time
      >>> print(time_array.shape)
      (100,)
      >>> print(time_array[0])
      numpy.datetime64('2023-01-01T00:00:00')


   .. py:property:: n_frequencies
      :type: int


      Get the number of frequency bins in the dataset.

      :returns: Number of frequency bins (second dimension of the dataset).
      :rtype: int


   .. py:property:: frequency
      :type: numpy.ndarray


      Get the frequency array for the dataset.

      Returns a linearly-spaced frequency array from frequency_min to
      frequency_max with n_frequencies points.

      :returns: Array of float64 frequencies in Hz with shape (n_frequencies,).
      :rtype: np.ndarray

      .. rubric:: Examples

      >>> freq_array = feature.frequency
      >>> print(freq_array.shape)
      (256,)
      >>> print(f"Frequency range: {freq_array[0]:.2f} - {freq_array[-1]:.2f} Hz")
      Frequency range: 0.01 - 100.00 Hz


   .. py:method:: replace_dataset(new_data_array: numpy.ndarray) -> None

      Replace the entire HDF5 dataset with new data.

      This method resizes the HDF5 dataset as needed and replaces all data.
      The input array must have the same dtype as the existing dataset.

      :param new_data_array: New data array to replace the existing dataset. Will be converted
                             to numpy array if necessary.
      :type new_data_array: np.ndarray

      :raises TypeError: If input cannot be converted to a numpy array or has incompatible shape.

      .. rubric:: Examples

      >>> import numpy as np
      >>> new_data = np.random.randn(100, 256)
      >>> feature.replace_dataset(new_data)



   .. py:method:: to_xarray() -> xarray.DataArray

      Convert the feature dataset to an xarray DataArray.

      Returns an xarray DataArray with proper time and frequency coordinates,
      metadata attributes, and component naming. The entire dataset is loaded
      into memory.

      :returns: DataArray with dimensions ['time', 'frequency'] and coordinates
                matching the dataset's time and frequency arrays.
      :rtype: xr.DataArray

      .. rubric:: Notes

      Metadata stored in xarray attributes will not be validated if modified.
      The full dataset is loaded into memory; use with caution for large datasets.

      .. rubric:: Examples

      >>> xr_data = feature.to_xarray()
      >>> print(xr_data.dims)
      ('time', 'frequency')
      >>> print(xr_data.name)
      'Ex'
      >>> subset = xr_data.sel(time=slice('2023-01-01', '2023-01-02'))



   .. py:method:: to_numpy() -> numpy.ndarray

      Convert the feature dataset to a numpy array.

      Returns the dataset as a numpy array by loading it from the HDF5 file
      into memory. The array shape is (n_windows, n_frequencies).

      :returns: Numpy array containing all feature data with shape
                (n_windows, n_frequencies).
      :rtype: np.ndarray

      .. rubric:: Examples

      >>> data = feature.to_numpy()
      >>> print(data.shape)
      (100, 256)
      >>> print(data.dtype)
      complex128
      >>> mean_amplitude = np.abs(data).mean()



   .. py:method:: from_numpy(new_estimate: numpy.ndarray) -> None

      Load data from a numpy array into the HDF5 dataset.

      This method updates the HDF5 dataset with new data from a numpy array.
      The input array must match the dataset's dtype. The HDF5 dataset will
      be resized if necessary to accommodate the new data.

      :param new_estimate: Numpy array to write to the HDF5 dataset. Must have compatible
                           dtype with the existing dataset.
      :type new_estimate: np.ndarray

      :raises TypeError: If input array dtype does not match the HDF5 dataset dtype or
          if input cannot be converted to numpy array.

      .. rubric:: Notes

      The variable 'data' is a builtin in numpy and cannot be used as a parameter name.

      .. rubric:: Examples

      >>> import numpy as np
      >>> new_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
      >>> feature.from_numpy(new_data)
      >>> loaded_data = feature.to_numpy()
      >>> assert loaded_data.shape == new_data.shape



   .. py:method:: from_xarray(data: xarray.DataArray, sample_rate_decimation_level: int) -> None

      Load data and metadata from an xarray DataArray.

      This method updates both the HDF5 dataset and metadata from an xarray
      DataArray. It extracts time coordinates, frequency range, and component
      information from the DataArray and its attributes.

      :param data: Input xarray DataArray with 'time' and 'frequency' coordinates.
                   Expected dimensions are ['time', 'frequency'].
      :type data: xr.DataArray
      :param sample_rate_decimation_level: Decimation level applied to the original data to produce this
                                           feature dataset (integer ≥ 1).
      :type sample_rate_decimation_level: int

      .. rubric:: Notes

      Metadata stored in xarray attributes will be extracted and written to
      the HDF5 file. The full dataset is loaded into memory during this process.

      .. rubric:: Examples

      >>> import xarray as xr
      >>> import numpy as np

      Create sample xarray data:

      >>> times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[s]')
      >>> freqs = np.linspace(0.01, 100, 256)
      >>> data_array = np.random.randn(len(times), len(freqs)) + \
      ...              1j * np.random.randn(len(times), len(freqs))
      >>> xr_data = xr.DataArray(
      ...     data_array,
      ...     dims=['time', 'frequency'],
      ...     coords={'time': times, 'frequency': freqs},
      ...     name='Ex',
      ...     attrs={'units': 'mV/km'}
      ... )

      Load into feature dataset:

      >>> feature.from_xarray(xr_data, sample_rate_decimation_level=2)
      >>> print(feature.metadata.component)
      'Ex'



