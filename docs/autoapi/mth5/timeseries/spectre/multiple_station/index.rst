mth5.timeseries.spectre.multiple_station
========================================

.. py:module:: mth5.timeseries.spectre.multiple_station

.. autoapi-nested-parse::

   Work In progress

   This module is concerned with working with Fourier coefficient data

   TODO:
   2. Give MultivariateDataset a covariance() method

   Tools include prototypes for
   - extracting portions of an FC Run Time Series
   - merging multiple stations runs together into an xarray
   - relabelling channels to avoid namespace clashes for multi-station data



Classes
-------

.. autoapisummary::

   mth5.timeseries.spectre.multiple_station.FCRunChunk
   mth5.timeseries.spectre.multiple_station.MultivariateLabelScheme
   mth5.timeseries.spectre.multiple_station.MultivariateDataset


Functions
---------

.. autoapisummary::

   mth5.timeseries.spectre.multiple_station.calculate_mask_from_feature
   mth5.timeseries.spectre.multiple_station.calculate_weight_from_feature
   mth5.timeseries.spectre.multiple_station.merge_masks
   mth5.timeseries.spectre.multiple_station.merge_weights
   mth5.timeseries.spectre.multiple_station.apply_masks_and_weights
   mth5.timeseries.spectre.multiple_station.make_multistation_spectrogram


Module Contents
---------------

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




.. py:function:: calculate_mask_from_feature(feature_series, threshold_obj)

.. py:function:: calculate_weight_from_feature(feature_series, threshold_obj)

       This calculates a weighting function based on the thresholds
       and possibly some other info, such as the distribution of the features.

       The weigth function is interpolated over the range of the feature values
       and then evaluated at the feature values.
   :param feature_series:
   :param threshold_obj:


.. py:function:: merge_masks()

   calcualtes a "final mask" that is loaded and applied to the data
   input to regression


.. py:function:: merge_weights()

   calcualtes a "final mask" that is loaded and applied to the data
       input to regression

.. py:function:: apply_masks_and_weights()

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



