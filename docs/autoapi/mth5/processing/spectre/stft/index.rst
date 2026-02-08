mth5.processing.spectre.stft
============================

.. py:module:: mth5.processing.spectre.stft

.. autoapi-nested-parse::

   This module has methods for applying the short-time-Fourier-transform.




Functions
---------

.. autoapisummary::

   mth5.processing.spectre.stft.run_ts_to_stft_scipy


Module Contents
---------------

.. py:function:: run_ts_to_stft_scipy(decimation_obj: Union[mt_metadata.processing.aurora.decimation_level.DecimationLevel, mt_metadata.processing.fourier_coefficients.Decimation], run_xrds_orig: xarray.Dataset) -> mth5.timeseries.spectre.spectrogram.Spectrogram

   Converts a runts object into a time series of Fourier coefficients.
   This method uses scipy.signal.spectrogram.

   TODO: consider making this a method of RunTS; runts.to_spectrogram(decimation_obj)

   :param decimation_obj: Information about how the decimation level is to be processed
   :type decimation_obj: Union[AuroraDecimationLevel, FCDecimation]
   :param run_xrds_orig: Time series to be processed
   :type run_xrds_orig: : xarray.core.dataset.Dataset

   :returns: **stft_obj** -- Time series of Fourier coefficients
   :rtype: xarray.core.dataset.Dataset


