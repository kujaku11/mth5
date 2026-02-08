mth5.timeseries.spectre.helpers
===============================

.. py:module:: mth5.timeseries.spectre.helpers

.. autoapi-nested-parse::

   This is a placeholder module for functions that are used in testing and development of spectrograms.



Attributes
----------

.. autoapisummary::

   mth5.timeseries.spectre.helpers.GROUPBY_COLUMNS


Functions
---------

.. autoapisummary::

   mth5.timeseries.spectre.helpers.add_fcs_to_mth5
   mth5.timeseries.spectre.helpers.read_back_fcs
   mth5.timeseries.spectre.helpers.calibrate_stft_obj


Module Contents
---------------

.. py:data:: GROUPBY_COLUMNS
   :value: ['survey', 'station', 'sample_rate']


.. py:function:: add_fcs_to_mth5(m: mth5.mth5.MTH5, fc_decimations: Optional[Union[str, list]] = None, groupby_columns: List[str] = GROUPBY_COLUMNS) -> None

   Add Fourier Coefficient Levels ot an existing MTH5.

   TODO: This method currently loops the heirarcy of the h5, and then calls an operator.
   How about making a single table that represents the loop up front and then looping once that
   table instead of this nested loop business?  We would need a function that takes as input
   the groupby_columns.

   **Notes:**

   - This module computes the FCs differently than the legacy aurora pipeline. It uses scipy.signal.spectrogram.
    There is a test in Aurora to confirm that there are equivalent if we are not using fancy pre-whitening.

   :param m: The mth5 file, open in append mode.
   :type m: MTH5 object
   :param fc_decimations: This specifies the scheme to use for decimating the time series when building the FC layer.
                          None: Just use default (something like four decimation levels, decimated by 4 each time say.)
                          String: Controlled Vocabulary, values are a work in progress, that will allow custom definition of
                          the fc_decimations for some common cases. For example, say you have stored already decimated time
                          series, then you want simply the zeroth decimation for each run, because the decimated time series live
                          under another run container, and that will get its own FCs.  This is experimental.
                          List: (**UNTESTED**) -- This means that the user thought about the decimations that they want to create and is
                          passing them explicitly.  -- probably will need to be a dictionary actually, since this
                          would get redefined at each sample rate.
   :type fc_decimations: Optional[Union[str, list]]


.. py:function:: read_back_fcs(m: Union[mth5.mth5.MTH5, pathlib.Path, str], mode: str = 'r', groupby_columns: List[str] = GROUPBY_COLUMNS) -> None

   Loops over stations in the channel summary of input (m) grouping by common sample_rate.
   Then loop over the runs in the corresponding FC Group.  Finally, within an fc_group,
   loop decimation levels and read data to xarray.  Log info about the shape of the xarray.

   This is a helper function for tests.  It was used as a sanity check while debugging the FC files, and
   also is a good example for how to access the data at each level for each channel.

   Development Notes:
   The Time axis of the FC array changes from decimation_level to decimation_level.
   The frequency axis will shape will depend on the window length that was used to perform STFT.
   This is currently storing all (positive frequency) fcs by default, but future versions can
   also have selected bands within an FC container.

   :param m: Either a path to an mth5, or an MTH5 object that the FCs will be read back from.
   :type m: Union[MTH5, pathlib.Path, str]
   :param mode: The mode to open the MTH5 file in. Defualts to (r)ead only.
   :type mode: str


.. py:function:: calibrate_stft_obj(stft_obj: xarray.Dataset, run_obj: mth5.groups.RunGroup, units: Literal['MT', 'SI'] = 'MT', channel_scale_factors: Optional[dict] = None) -> xarray.Dataset

   Calibrates frequency domain data into MT units.

   Development Notes:
    The calibration often raises a runtime warning due to DC term in calibration response = 0.
    TODO: It would be nice to suppress this, maybe by only calibrating the non-dc terms and directly assigning np.nan to the dc component when DC-response is zero.

   :param stft_obj: Time series of Fourier coefficients to be calibrated
   :type stft_obj: xarray.core.dataset.Dataset
   :param run_obj: Provides information about filters for calibration
   :type run_obj: mth5.groups.master_station_run_channel.RunGroup
   :param units: usually "MT", contemplating supporting "SI"
   :type units: string
   :param scale_factors: keyed by channel, supports a single scalar to apply to that channels data
                         Useful for debugging.  Should not be used in production and should throw a
                         warning if it is not None
   :type scale_factors: dict or None

   :returns: **stft_obj** -- Time series of calibrated Fourier coefficients
   :rtype: xarray.core.dataset.Dataset


