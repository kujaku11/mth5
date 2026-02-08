mth5.processing.spectre.prewhitening
====================================

.. py:module:: mth5.processing.spectre.prewhitening

.. autoapi-nested-parse::

   This module has methods for pre-whitening time series to reduce spectral leakage before FFT.



Functions
---------

.. autoapisummary::

   mth5.processing.spectre.prewhitening.apply_prewhitening
   mth5.processing.spectre.prewhitening.apply_recoloring


Module Contents
---------------

.. py:function:: apply_prewhitening(prewhitening_type: Literal['first difference', ''], run_xrds_input: xarray.Dataset) -> xarray.Dataset

   Applies pre-whitening to time series to avoid spectral leakage when FFT is applied.

   :param prewhitening_type: Placeholder to allow for multiple methods of pre-whitening.  Currently only
                             "first difference" is supported.
   :type prewhitening_type: Literal["first difference", ]
   :param run_xrds_input: Time series to be pre-whitened (can be multivariate).
   :type run_xrds_input: xr.Dataset

   :returns: **run_xrds** -- pre-whitened time series
   :rtype: xr..Dataset


.. py:function:: apply_recoloring(prewhitening_type: Literal['first difference'], stft_obj: xarray.Dataset) -> xarray.Dataset

   Inverts the pre-whitening operation in frequency domain.  Modifies the input xarray in-place.

   :param prewhitening_type: Placeholder to allow for multiple methods of pre-whitening.  Currently only
                             "first difference" is supported.
   :type prewhitening_type: Literal["first difference", ]
   :param stft_obj: Time series of Fourier coefficients to be recoloured
   :type stft_obj: xarray.core.dataset.Dataset

   :returns: **stft_obj** -- Recolored time series of Fourier coefficients.
   :rtype: xarray.core.dataset.Dataset


