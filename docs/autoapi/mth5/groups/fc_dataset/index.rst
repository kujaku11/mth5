mth5.groups.fc_dataset
======================

.. py:module:: mth5.groups.fc_dataset

.. autoapi-nested-parse::

   Created on Thu Mar 10 09:02:16 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.groups.fc_dataset.FCChannelDataset


Module Contents
---------------

.. py:class:: FCChannelDataset(dataset: h5py.Dataset, dataset_metadata: mt_metadata.processing.fourier_coefficients.FCChannel | None = None, **kwargs: Any)

   Container for Fourier coefficients (FC) from windowed FFT analysis.

   Holds multi-dimensional Fourier coefficient data representing time-frequency
   analysis results. Data is uniformly sampled in both frequency (via harmonic
   index) and time (via uniform FFT window step size).

   :param dataset: HDF5 dataset containing the Fourier coefficient data.
   :type dataset: h5py.Dataset
   :param dataset_metadata: Metadata object containing FC channel properties like start time,
                            end time, sample rates, units, and frequency method. If provided,
                            metadata will be written to HDF5 attributes. Defaults to None.
   :type dataset_metadata: FCChannel | None, optional
   :param \*\*kwargs: Additional keyword arguments (reserved for future use).
   :type \*\*kwargs: Any

   .. attribute:: hdf5_dataset

      Weak reference to the HDF5 dataset.

      :type: h5py.Dataset

   .. attribute:: metadata

      Metadata container for the Fourier coefficients.

      :type: FCChannel

   .. attribute:: logger

      Logger instance for reporting messages.

      :type: loguru.logger

   :raises MTH5Error: If dataset_metadata is provided but is not of type FCChannel.
   :raises TypeError: If input data cannot be converted to numpy array or has
       incompatible dtype/shape.

   .. rubric:: Notes

   The data array has shape (n_windows, n_frequencies) where:
   - n_windows: Number of time windows in the FFT moving window analysis
   - n_frequencies: Number of frequency bins determined by window size

   Data is typically complex-valued representing Fourier coefficients.
   Time windows are uniformly spaced with interval 1/sample_rate_window_step.
   Frequencies are uniformly spaced from frequency_min to frequency_max.

   Metadata includes:
   - Time period (start and end)
   - Acquisition and decimated sample rates
   - Window sample rate (delta_t within window)
   - Units
   - Frequency method (integer harmonic index calculation)
   - Component name (channel designation)

   .. rubric:: Examples

   Create an FC dataset from HDF5 group:

   >>> import h5py
   >>> import numpy as np
   >>> from mt_metadata.processing.fourier_coefficients import FCChannel
   >>> with h5py.File('fc.h5', 'w') as f:
   ...     # Create 2D array: 50 time windows, 256 frequencies
   ...     data = np.random.rand(50, 256) + 1j * np.random.rand(50, 256)
   ...     dset = f.create_dataset('Ex', data=data, dtype=np.complex128)
   ...     # Create FCChannelDataset
   ...     fc = FCChannelDataset(dset, write_metadata=True)

   Convert to xarray and access time-frequency data:

   >>> xr_data = fc.to_xarray()
   >>> print(xr_data.dims)  # ('time', 'frequency')
   >>> # Access data at specific time and frequency
   >>> subset = xr_data.sel(time='2023-01-01T12:00:00', method='nearest')

   Inspect properties:

   >>> print(f"Windows: {fc.n_windows}, Frequencies: {fc.n_frequencies}")
   >>> print(f"Frequency range: {fc.frequency.min():.2f}-{fc.frequency.max():.2f} Hz")


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
      >>> fc.read_metadata()  # Reload changes
      >>> print(fc.metadata.component)  # Access updated component



   .. py:method:: write_metadata() -> None

      Write metadata from container to HDF5 dataset attributes.

      Converts the pydantic metadata model to a dictionary and writes
      each field as an HDF5 attribute. Values are converted to appropriate
      numpy types for compatibility. Always ensures 'mth5_type' attribute
      is set to 'FCChannel'.

      :rtype: None

      .. rubric:: Notes

      All existing attributes with the same names will be overwritten.
      This is called automatically during initialization and after
      metadata updates. Read-only files will silently skip writes.

      .. rubric:: Examples

      Save updated metadata to HDF5:

      >>> fc.metadata.component = "Ey"
      >>> fc.write_metadata()  # Persist to file
      >>> # Verify write
      >>> print(fc.hdf5_dataset.attrs['component'])
      b'Ey'



   .. py:property:: n_windows
      :type: int


      Number of time windows in the FFT analysis.

      :returns: Number of time windows (first dimension of data array).
      :rtype: int

      .. rubric:: Notes

      This corresponds to the number of rows in the 2D spectrogram data.
      Each window represents a uniform time interval determined by the
      window step size (1/sample_rate_window_step).

      .. rubric:: Examples

      >>> print(f"Time windows: {fc.n_windows}")
      Time windows: 50


   .. py:property:: time
      :type: numpy.ndarray


      Time array including the start of each time window.

      Generates uniformly spaced time coordinates based on the start time,
      window step rate, and number of windows. Uses metadata time period
      to determine bounds.

      :returns: Array of datetime64 values for each window start time.
      :rtype: np.ndarray

      .. rubric:: Notes

      Time coordinates are generated using make_dt_coordinates, which
      ensures consistency between specified start/end times and the
      number of windows.

      .. rubric:: Examples

      Access time array for time-based indexing:

      >>> time_array = fc.time
      >>> print(time_array.shape)  # (n_windows,)
      >>> print(time_array[0])  # First window time
      2023-01-01T00:00:00.000000


   .. py:property:: n_frequencies
      :type: int


      Number of frequency bins in the Fourier analysis.

      :returns: Number of frequency bins (second dimension of data array).
      :rtype: int

      .. rubric:: Notes

      This corresponds to the number of columns in the 2D spectrogram data.
      Determined by the FFT window size and relates to the frequency
      resolution of the analysis.

      .. rubric:: Examples

      >>> print(f"Frequency bins: {fc.n_frequencies}")
      Frequency bins: 256


   .. py:property:: frequency
      :type: numpy.ndarray


      Frequency array from metadata frequency bounds.

      Generates uniformly spaced frequency coordinates based on the
      metadata frequency range and number of frequency bins.

      :returns: Array of frequency values, linearly spaced from frequency_min
                to frequency_max.
      :rtype: np.ndarray

      .. rubric:: Notes

      Frequencies represent harmonic indices or actual frequency values
      depending on the frequency method specified in metadata.
      Spacing is determined by n_frequencies bins over the range.

      .. rubric:: Examples

      Access frequency array for frequency-based indexing:

      >>> freq_array = fc.frequency
      >>> print(freq_array.shape)  # (n_frequencies,)
      >>> print(f"Frequency range: {freq_array.min():.2f} to {freq_array.max():.2f} Hz")
      Frequency range: 0.00 to 64.00 Hz


   .. py:method:: replace_dataset(new_data_array: numpy.ndarray) -> None

      Replace entire dataset with new data.

      Resizes the HDF5 dataset if necessary and replaces all data.
      Converts input to numpy array if needed.

      :param new_data_array: New FC data to store. Should have shape (n_windows, n_frequencies)
                             and typically complex-valued.
      :type new_data_array: np.ndarray

      :rtype: None

      :raises TypeError: If input cannot be converted to numpy array.

      .. rubric:: Notes

      If new data has different shape, HDF5 dataset will be resized.
      This is generally safe but may fragment the HDF5 file.

      .. rubric:: Examples

      Replace FC data with new analysis results:

      >>> import numpy as np
      >>> new_fc = np.random.rand(30, 256) + 1j * np.random.rand(30, 256)
      >>> fc.replace_dataset(new_fc)
      >>> print(fc.to_numpy().shape)
      (30, 256)

      Replace with data from list (auto-converted to array):

      >>> data_list = [[[1+1j, 2+2j]], [[3+3j, 4+4j]]] * 15
      >>> fc.replace_dataset(data_list)
      >>> fc.to_numpy().shape
      (30, 2)



   .. py:method:: to_xarray() -> xarray.DataArray

      Convert FC data to xarray DataArray.

      Creates an xarray DataArray with proper coordinates for time and
      frequency. Includes metadata as attributes.

      :returns: DataArray with dimensions (time, frequency) and coordinates
                from metadata and computed properties.
      :rtype: xr.DataArray

      .. rubric:: Notes

      Metadata changes in xarray are not validated and will not be
      synchronized back to HDF5 without explicit call to from_xarray().
      Data is loaded entirely into memory.

      .. rubric:: Examples

      Convert to xarray with automatic coordinates:

      >>> xr_data = fc.to_xarray()
      >>> print(xr_data.dims)
      ('time', 'frequency')
      >>> print(xr_data.shape)
      (50, 256)

      Select data by time and frequency range:

      >>> subset = xr_data.sel(
      ...     time=slice('2023-01-01T00:00:00', '2023-01-01T12:00:00'),
      ...     frequency=slice(0, 10)
      ... )
      >>> print(subset.shape)  # Subset shape



   .. py:method:: to_numpy() -> numpy.ndarray

      Convert FC data to numpy array.

      Returns the HDF5 dataset as a numpy array. Data is loaded
      entirely into memory.

      :returns: 2D complex array with shape (n_windows, n_frequencies).
      :rtype: np.ndarray

      .. rubric:: Notes

      For large spectrograms, this loads all data into RAM. Consider using
      HDF5 slicing for memory-efficient access to subsets.

      .. rubric:: Examples

      Get full FC data as numpy array:

      >>> data = fc.to_numpy()
      >>> print(data.shape)
      (50, 256)
      >>> print(data.dtype)
      complex128

      Access specific time window and frequency:

      >>> data = fc.to_numpy()
      >>> # Get first 10 windows, frequency bin 100
      >>> subset = data[:10, 100]
      >>> print(subset.shape)
      (10,)



   .. py:method:: from_numpy(new_estimate: numpy.ndarray) -> None

      Load FC data from numpy array.

      Validates dtype and shape compatibility, resizes dataset if needed,
      and stores the data.

      :param new_estimate: FC data to load. Should have shape (n_windows, n_frequencies).
                           Typically complex-valued array.
      :type new_estimate: np.ndarray

      :rtype: None

      :raises TypeError: If dtype doesn't match existing dataset or input cannot
          be converted to numpy array.

      .. rubric:: Notes

      'data' is a built-in Python function and cannot be used as parameter name.
      The dataset will be resized if shape doesn't match.
      Dtype compatibility is strictly enforced.

      .. rubric:: Examples

      Load FC data from numpy array:

      >>> import numpy as np
      >>> new_data = np.random.rand(25, 128) + 1j * np.random.rand(25, 128)
      >>> fc.from_numpy(new_data)
      >>> print(fc.to_numpy().shape)
      (25, 128)

      Load with magnitude and phase separation:

      >>> magnitude = np.random.rand(20, 256)
      >>> phase = np.random.rand(20, 256) * 2 * np.pi
      >>> fc_data = magnitude * np.exp(1j * phase)
      >>> fc.from_numpy(fc_data)



   .. py:method:: from_xarray(data: xarray.DataArray, sample_rate_decimation_level: int | float) -> None

      Load FC data from xarray DataArray.

      Updates metadata from xarray coordinates and attributes, then
      stores the data. Computes frequency and time parameters from
      the provided xarray object.

      :param data: DataArray containing FC data. Expected dimensions:
                   (time, frequency).
      :type data: xr.DataArray
      :param sample_rate_decimation_level: Decimation level applied to original sample rate.
                                           Used to track processing history.
      :type sample_rate_decimation_level: int | float

      :rtype: None

      .. rubric:: Notes

      This will update time_period (start/end), frequency bounds,
      window step rate, decimation level, component name, and units
      from the xarray object. All changes are persisted to HDF5.

      .. rubric:: Examples

      Load FC data from modified xarray:

      >>> xr_data = fc.to_xarray()
      >>> # Modify data (e.g., apply filter)
      >>> modified = xr_data * np.hamming(256)  # Apply frequency window
      >>> fc.from_xarray(modified, sample_rate_decimation_level=4)
      >>> print(fc.metadata.sample_rate_decimation_level)
      4

      Load with updated metadata from another analysis:

      >>> import xarray as xr
      >>> import pandas as pd
      >>> time_coords = pd.date_range('2023-01-01', periods=30, freq='1H')
      >>> freq_coords = np.arange(0, 128)
      >>> new_fc = xr.DataArray(
      ...     data=np.random.rand(30, 128) + 1j * np.random.rand(30, 128),
      ...     coords={'time': time_coords, 'frequency': freq_coords},
      ...     dims=['time', 'frequency'],
      ...     name='Ey',
      ...     attrs={'units': 'mV/km'}
      ... )
      >>> fc.from_xarray(new_fc, sample_rate_decimation_level=1)
      >>> print(fc.metadata.component)
      Ey



