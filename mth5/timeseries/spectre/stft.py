"""
    This module has methods for applying the short-time-Fourier-transform.


"""
from .prewhitening import apply_prewhitening
from .prewhitening import apply_recoloring
from mt_metadata.transfer_functions.processing.aurora.decimation_level import (
    DecimationLevel as AuroraDecimationLevel,
)
from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Decimation as FCDecimation,
)

import xarray as xr


def run_ts_to_stft_scipy(
    decimation_obj: Union[AuroraDecimationLevel, FCDecimation],
    run_xrds_orig: xr.Dataset
) -> xr.Dataset:
    """
    Converts a runts object into a time series of Fourier coefficients.
    This method uses scipy.signal.spectrogram.

    Parameters
    ----------
    decimation_obj :  Union[AuroraDecimationLevel, FCDecimation]
        Information about how the decimation level is to be processed
    run_xrds_orig : : xarray.core.dataset.Dataset
        Time series to be processed

    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients
    """
    run_xrds = apply_prewhitening(decimation_obj, run_xrds_orig)
    windowing_scheme = window_scheme_from_decimation(decimation_obj)

    stft_obj = xr.Dataset()
    for channel_id in run_xrds.data_vars:
        ff, tt, specgm = ssig.spectrogram(
            run_xrds[channel_id].data,
            fs=decimation_obj.sample_rate_decimation,
            window=windowing_scheme.taper,
            nperseg=decimation_obj.window.num_samples,
            noverlap=decimation_obj.window.overlap,
            detrend="linear",
            scaling="density",
            mode="complex",
        )

        # drop Nyquist
        ff = ff[:-1]
        specgm = specgm[:-1, :]
        specgm *= np.sqrt(2)  # accout for cutting spectrogram in half (to keep PSDs accurate)

        # make time_axis
        tt = tt - tt[0]
        tt *= decimation_obj.sample_rate_decimation
        time_axis = run_xrds.time.data[tt.astype(int)]

        xrd = xr.DataArray(
            specgm.T,
            dims=["time", "frequency"],
            coords={"frequency": ff, "time": time_axis},
        )
        stft_obj.update({channel_id: xrd})

    if decimation_obj.recoloring:
        stft_obj = apply_recoloring(decimation_obj, stft_obj)

    return stft_obj
