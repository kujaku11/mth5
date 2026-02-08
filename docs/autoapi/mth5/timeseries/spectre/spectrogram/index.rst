mth5.timeseries.spectre.spectrogram
===================================

.. py:module:: mth5.timeseries.spectre.spectrogram

.. autoapi-nested-parse::

   Module contains a class that represents a spectrogram.
   i.e. A 2D time series of Fourier coefficients with axes time and the other frequency.
   The datasets are xarray/dataframe and are fundmentally multivariate.



Classes
-------

.. autoapisummary::

   mth5.timeseries.spectre.spectrogram.Spectrogram


Functions
---------

.. autoapisummary::

   mth5.timeseries.spectre.spectrogram.extract_band


Module Contents
---------------

.. py:class:: Spectrogram(dataset: Optional[xarray.Dataset] = None)

   Bases: :py:obj:`object`


   Class to contain methods for STFT objects.

   TODO: Add OLS Z-estimates -- actually, these are properties of cross powers, not direct properties of spectrograms.
   TODO: Add Sims/Vozoff Z-estimates -- actually, these are properties of cross powers as well.
   **Note** Coherence is similarly, a property of cross powers.
   There are in fact, very few features that we would derive from an unaveraged spectrogram.  Pretty much
   everything except statistical moments comes from cross powers.

   Development Notes:
   - The spectrogram class is fundamental to MT Processing, and normally appears during the STFT operation.
   - The extract_band method returns another Spectrogram, having the same time axis as the parent
   object, but only a slice of the frequency range.  Both of these have in common that their frequency axes
   are uniformly spaced, delta-f, where delta-f is dictated by the time series sample rate and the FFT window
   lenght.
   - There is a sibling spectral-time-series container that should be considered.  Call it for now, a
   FrequencyChunkedSpectrogram (or an AveragedSpectrogram).  This is a container similar to spectrogram, but
   the frequencies are not uniformly spaced (instead, often logartihmically spaced), they are made from one or
   more (possibly multivariate) spectrograms, and a FrequencyBands object.  The key difference
   is that in a FrequencyChunkedSpectrogram object has a non-uniform spaced the Frequency axis which was prescribed
   by a metadata object.  Most features, as well as TFs have a FrequencyChunkedSpectrogram representation,
   where final TFs are just time-averaged a FrequencyChunkedSpectrograms.

   TODO: consider factoring a simpler class that does not make the uniform frequency axis assumption.
   Spectrogram would extend this class and add the  _frequency_increment property (taken from the differece in
   the first two values of the frequency axis), and num_harmoincs in band.



   .. py:property:: dataset

      returns the underlying xarray data


   .. py:property:: dataarray

      returns the underlying xarray data


   .. py:property:: time_axis

      returns the time axis of the underlying xarray


   .. py:property:: frequency_axis

      returns the frequency axis of the underlying xarray


   .. py:property:: frequency_band
      :type: mt_metadata.common.band.Band


      returns a frequency band object representing the spectrograms band (assumes continuous)


   .. py:property:: frequency_increment

      returns the "delta f" of the frequency axis
      - assumes uniformly sampled in frequency domain


   .. py:method:: num_harmonics_in_band(frequency_band: mt_metadata.common.band.Band, epsilon: float = 1e-07) -> int

      Returns the number of harmonics within the frequency band in the underlying dataset

      :param frequency_band:
      :param stft_obj:

      :returns: **num_harmonics** -- The number of harmonics in the underlying dataset within the given frequency band.
      :rtype: int



   .. py:method:: extract_band(frequency_band: mt_metadata.common.band.Band, channels: Optional[list] = None, epsilon: Optional[float] = None)

      Returns another instance of Spectrogram, with the frequency axis reduced to the input band.

      :param frequency_band:
      :param channels:

      :returns: **spectrogram** -- Returns a Spectrogram object with only the extracted band for a dataset
      :rtype: aurora.time_series.spectrogram.Spectrogram



   .. py:method:: cross_power_label(ch1: str, ch2: str, join_char: str = '_')

      joins channel names with join_char



   .. py:method:: cross_powers(frequency_bands: mt_metadata.processing.aurora.frequency_bands.FrequencyBands, channel_pairs: Optional[List[Tuple[str, str]]] = None)

      Compute cross powers between channel pairs for given frequency bands.

      TODO: Add handling for case when band in frequency_bands is not contained
      in self.frequencies.

      :param frequency_bands: The frequency bands to compute cross powers for.  Each element of this iterable
                              tells the lower and upper bounds of the cross-power calculation bands.
                              These may become objects with information about tapers as ewwll.
      :type frequency_bands: FrequencyBands
      :param channel_pairs: List of channel pairs to compute cross powers for.
                            If None, all possible pairs will be used.
      :type channel_pairs: list of tuples, optional

      :returns: Dataset containing cross powers for all channel pairs.
                Each variable is named by the channel pair (e.g. 'ex_hy')
                and contains a 2D array with dimensions (frequency, time).
                All variables share common frequency and time coordinates.
      :rtype: xr.Dataset



   .. py:method:: covariance_matrix(band_data: Optional[Spectrogram] = None, method: str = 'numpy_cov') -> xarray.DataArray

      TODO: Add tests for this WIP Work-in-progress method
      Compute full covariance matrix for spectrogram data.

      For complex-valued data, the result is a Hermitian matrix where:
      - diagonal elements are real-valued variances
      - off-diagonal element [i,j] is E[ch_i * conj(ch_j)]
      - off-diagonal element [j,i] is the complex conjugate of [i,j]

      :param band_data: If provided, compute covariance for this data
                        If None, use the full spectrogram
      :type band_data: Spectrogram, optional
      :param method: Computation method. Currently only supports 'numpy_cov'
      :type method: str

      :returns: Hermitian covariance matrix with proper channel labeling
                For channels i,j: matrix[i,j] = E[ch_i * conj(ch_j)]
      :rtype: xr.DataArray



   .. py:method:: flatten(chunk_by: Literal['time', 'frequency'] = 'time') -> xarray.Dataset

          Reshape the 2D spectrogram into a 1D flattened xarray (time-chunked by default).

      :param chunk_by: Reshaping the 2D spectrogram can be done two ways, (basically "row-major",
                       or column-major). In xarray, but we either keep frequency constant and iterate
                       over time, or keep time constant and iterate over frequency (in the inner loop).
      :type chunk_by: Literal["time", "frequency"]

      :returns: * **xarray.Dataset** (*The dataset from the band spectrogram, stacked.*)
                * *Development Notes*
                * *The flattening used in tf calculation by default is opposite to here*
                * *dataset.stack(observation=("frequency", "time"))*
                * *However, for feature extraction, it may make sense to swap the order*
                * *xrds = band_spectrogram.dataset.stack(observation=("time", "frequency"))*
                * *This is like chunking into time windows and allows individual features to be computed on each time window -- if desired.*
                * *Still need to split the time series though--Splitting to time would be a reshape by (last_freq_index-first_freq_index).*
                * *Using pure xarray this may not matter but if we drop down into numpy it could be useful.*



.. py:function:: extract_band(frequency_band: mt_metadata.common.band.Band, fft_obj: Union[xarray.Dataset, xarray.DataArray], channels: Optional[list] = None, epsilon: float = 1e-07) -> Union[xarray.Dataset, xarray.DataArray]

   Extracts a frequency band from xr.DataArray representing a spectrogram.

   TODO: Update variable names.

   Development Notes:
       Base dataset object should be a xr.DataArray (not xr.Dataset)
       - drop=True does not play nice with h5py and Dataset, results in a type error.
       File "stringsource", line 2, in h5py.h5r.Reference.__reduce_cython__
       TypeError: no default __reduce__ due to non-trivial __cinit__
       However, it works OK with DataArray.

   :param frequency_band: Specifies interval corresponding to a frequency band
   :type frequency_band: mt_metadata.common.band.Band
   :param fft_obj: Short-time-Fourier-transformed datat.  Can be multichannel.
   :type fft_obj: xarray.core.dataset.Dataset
   :param channels: Channel names to extract.
   :type channels: list
   :param epsilon: Use this when you are worried about missing a frequency due to
                   round off error.  This is in general not needed if we use a df/2 pad
                   around true harmonics.
   :type epsilon: float

   :returns: **extracted_band** -- The frequencies within the band passed into this function
   :rtype: xr.DataArray


