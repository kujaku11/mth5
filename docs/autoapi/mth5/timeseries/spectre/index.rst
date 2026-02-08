mth5.timeseries.spectre
=======================

.. py:module:: mth5.timeseries.spectre

.. autoapi-nested-parse::

   Allows access to classes that we want to import without full pathing to module.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/timeseries/spectre/helpers/index
   /autoapi/mth5/timeseries/spectre/multiple_station/index
   /autoapi/mth5/timeseries/spectre/spectrogram/index


Classes
-------

.. autoapisummary::

   mth5.timeseries.spectre.FCRunChunk
   mth5.timeseries.spectre.MultivariateDataset
   mth5.timeseries.spectre.MultivariateLabelScheme
   mth5.timeseries.spectre.Spectrogram


Functions
---------

.. autoapisummary::

   mth5.timeseries.spectre.make_multistation_spectrogram


Package Contents
----------------

.. py:class:: FCRunChunk

   This class formalizes the required metadata to specify a chunk of a timeseries of Fourier coefficients.

   This may move to mt_metadata -- for now just use a dataclass as a prototype.


   .. py:attribute:: survey_id
      :type:  str
      :value: 'none'



   .. py:attribute:: station_id
      :type:  str
      :value: ''



   .. py:attribute:: run_id
      :type:  str
      :value: ''



   .. py:attribute:: decimation_level_id
      :type:  str
      :value: '0'



   .. py:attribute:: start
      :type:  str
      :value: ''



   .. py:attribute:: end
      :type:  str
      :value: ''



   .. py:attribute:: channels
      :type:  Tuple[str]
      :value: ()



   .. py:property:: start_timestamp
      :type: pandas.Timestamp



   .. py:property:: end_timestamp
      :type: pandas.Timestamp



   .. py:property:: duration
      :type: pandas.Timestamp



.. py:class:: MultivariateDataset(dataset: xarray.Dataset, label_scheme: Optional[MultivariateLabelScheme] = None)

   Bases: :py:obj:`mth5.timeseries.spectre.spectrogram.Spectrogram`


   Here is a container for a multivariate spectral dataset.
   The xarray is the main underlying item, but it will be useful to have functions that, for example returns a
   list of the associated stations, or that return a list of channels that are associated with a station, etc.

   This is intended to be used as a multivariate spectral dotaset at one frequency band.

   TODO: Consider making this an extension of Spectrogram
   TODO: Rename this class to MultivariateSpectrogram.




   .. py:property:: label_scheme
      :type: MultivariateLabelScheme



   .. py:property:: channels
      :type: list


      returns a list of channels in the dataarray


   .. py:property:: num_channels
      :type: int


      returns a count of the total number of channels in the dataset


   .. py:property:: stations
      :type: List[str]


      Parses the channel names, extracts the station names

      return a unique list of stations preserving order.


   .. py:method:: station_channels(station: str) -> List[str]

      This is a utility function that provides a way to access channel_names in a multivariate array associated
       with a particular station.
      The list is accessed via the self._station_channels attr, which gets set here if it has not
       been initialized previously.  self._station_channels is a dict keyed by station_id, with value
       is a list of channel names for that station.

      :param station: The name of the station.
      :type station: str

      :rtype: List[str]
      :returns: list of channel names for the input station.




   .. py:method:: archive_cross_powers(tf_station: str, with_fcs: bool = True)

      tf_station: str
       This tells us under which station we should store the output of this function.
       TODO: Consider moving this to another function which performs archiving in future.

      with_fcs: bool
       If True, the features are packed into the same hdf5-group as the FCs,
       as its own dataset.
       If False: the features are packed into the hdf5 features-group.



   .. py:method:: cross_power(aweights: Optional[numpy.ndarray] = None, bias: Optional[bool] = True) -> xarray.DataArray

      Calculate the cross-power from a multivariate, complex-valued array of Fourier coefficients.

      For a multivaraiate FC Dataset with n_time time windows, this returns an array with the same number of time
      windows.  At each time _t_, the result is a covariance matrix.

      Caveats and Notes:
        - This method calls numpy.cov, which means that the cross-power is computes as X@XH (rather than
        XH@X). Sometimes X*XH is referred to as the Vozoff convention, whereas XH*X could be the
        Bendat & Piersol convention.
        - np.cov subtracts the meas before computing the cross terms.
        - This methos will use the entire band of the spectrogram.

      :param X: Multivariate time series as an xarray
      :type X: xr.DataArray
      :param aweights: This is a "passthrough" parameter to numpy.cov These relative weights are typically large for
       observations considered "important" and smaller for observations considered less "important". If ``ddof=0``
       the array of weights can be used to assign probabilities to observation vectors.
      :type aweights: Optional[np.ndarray]
      :param bias: bias=True normalizes by N instead of (N-1).
      :type bias: bool

      :rtype: xr.DataArray
      :return: The covariance matrix of the data in xarray form.




.. py:class:: MultivariateLabelScheme

   Class to store information about how a multivariate (MV) dataset will be lablelled.

   Has a scheme to handle the how channels will be named.

   This is just a place holder to manage possible future complexity.

   It seemed like a good idea to formalize the fact that we take, by default
   f"{station}_{component}" as the MV channel label.
   It also seemed like a good idea to record what the join character is.
   In the event that we wind up with station names that have underscores in them, then we could,
   for example, set the join character to "__".

   TODO: Consider rename default to ("station", "data_var") instead of ("station", "component")

   :param :
   :type : type label_elements: tuple
   :param :
   :type : param label_elements: This is meant to tell what information is being concatenated into an MV channel label.
   :param :
   :type : type join_char: str
   :param :
   :type : param join_char: The string that is used to join the label elements.


   .. py:attribute:: label_elements
      :type:  tuple
      :value: ('station', 'component')



   .. py:attribute:: join_char
      :type:  str
      :value: '_'



   .. py:property:: id
      :type: str



   .. py:method:: join(elements: Union[list, tuple]) -> str

      Join the label elements to a string

      :type elements:  tuple
      :param elements: Expected to be the label elements, default are (station, component)

      :return: The name of the channel (in a multiple-station context).
      :rtype: str




   .. py:method:: split(mv_channel_name) -> dict

      Splits a multi-station channel name and returns a dict of strings, keyed by self.label_elements.
      This method is basically the reverse of self.join

      :param mv_channel_name: a multivariate channel name string
      :type mv_channel_name: str
      :return: Channel name as a dictionary.
      :rtype: dict




.. py:function:: make_multistation_spectrogram(m: mth5.mth5.MTH5, fc_run_chunks: list, label_scheme: Optional[MultivariateLabelScheme] = MultivariateLabelScheme(), rtype: Optional[Literal['xrds']] = None) -> Union[xarray.Dataset, MultivariateDataset]

   See notes in mth5 issue #209.  Takes a list of FCRunChunks and returns the largest contiguous
   block of multichannel FC data available.

   |----------Station 1 ------------|
           |----------Station 2 ------------|
   |--------------------Station 3 ----------------------|


           |-------RETURNED------|

   Handle additional runs in a separate call to this function and then concatenate time series afterwards.

   Input must specify N (station-run-start-end-channel_list) tuples.
   If channel_list is not provided, get all channels.
   If start-end are not provided, read the whole run -- warn if runs are not all synchronous, and
   truncate all to max(starts), min(ends) after the start and end times are sorted out.

   Station IDs must be unique.

   :param m:  The mth5 object to get the FCs from.
   :type m: mth5.mth5.MTH5
   :param fc_run_chunks: Each element of this describes a chunk of a run to load from stored FCs.
   :type fc_run_chunks: list
   :param label_scheme: Specifies how the channels are to be named in the multivariate xarray.
   :type label_scheme: Optional[MultivariateLabelScheme]
   :param rtype: Specifies whether to return an xarray or a MultivariateDataset.  Currently only supports "xrds",
   otherwise will return MultivariateDataset.
   :type rtype: Optional[Literal["xrds"]]

   :rtype: Union[xarray.Dataset, MultivariateDataset]:
   :return: The multivariate dataset, either as an xarray or as a MultivariateDataset



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



