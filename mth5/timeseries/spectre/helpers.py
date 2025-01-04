"""
    This is a placeholder module for functions that are used in testing and development of spectrograms.
"""

from loguru import logger
from mt_metadata.transfer_functions.processing.aurora import DecimationLevel as AuroraDecimationLevel
from mt_metadata.transfer_functions.processing.fourier_coefficients import Decimation as FCDecimation
from mt_metadata.transfer_functions.processing.fourier_coefficients.decimation import fc_decimations_creator
from mth5.mth5 import MTH5
from mth5.utils.helpers import path_or_mth5_object
from typing import Literal, Optional, Union
from .prewhitening import apply_prewhitening
from .prewhitening import apply_recoloring
from .stft import run_ts_to_stft_scipy

import mth5
import numpy as np
import pathlib
import scipy.signal as ssig
import xarray as xr

GROUPBY_COLUMNS = ["survey", "station", "sample_rate"]


@path_or_mth5_object
def add_fcs_to_mth5(m: MTH5, fc_decimations: Optional[Union[str, list]] = None) -> None:
    """
    Add Fourier Coefficient Levels ot an existing MTH5.

    **Notes:**

    - This module computes the FCs differently than the legacy aurora pipeline. It uses scipy.signal.spectrogram.
     There is a test in Aurora to confirm that there are equivalent if we are not using fancy pre-whitening.

    Parameters
    ----------
    m: MTH5 object
        The mth5 file, open in append mode.
    fc_decimations: Optional[Union[str, list]]
        This specifies the scheme to use for decimating the time series when building the FC layer.
        None: Just use default (something like four decimation levels, decimated by 4 each time say.)
        String: Controlled Vocabulary, values are a work in progress, that will allow custom definition of
        the fc_decimations for some common cases. For example, say you have stored already decimated time
        series, then you want simply the zeroth decimation for each run, because the decimated time series live
        under another run container, and that will get its own FCs.  This is experimental.
        List: (**UNTESTED**) -- This means that the user thought about the decimations that they want to create and is
        passing them explicitly.  -- probably will need to be a dictionary actually, since this
        would get redefined at each sample rate.

    """
    # Group the channel summary by survey, station, sample_rate
    channel_summary_df = m.channel_summary.to_dataframe()
    grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)
    logger.debug(f"Detected {len(grouper)} unique station-sample_rate instances")

    # loop over groups
    for (survey, station, sample_rate), group in grouper:
        msg = f"\n\n\nsurvey: {survey}, station: {station}, sample_rate {sample_rate}"
        logger.info(msg)
        station_obj = m.get_station(station, survey)
        run_summary = station_obj.run_summary

        # Get the FC decimation schemes if not provided
        if not fc_decimations:
            msg = "FC Decimations not supplied, creating defaults on the fly"
            logger.info(f"{msg}")
            fc_decimations = fc_decimations_creator(
                initial_sample_rate=sample_rate, time_period=None
            )
        elif isinstance(fc_decimations, str):
            if fc_decimations == "degenerate":
                fc_decimations = get_degenerate_fc_decimation(sample_rate)

        # TODO: Make this a function that can be done using df.apply()
        for i_run_row, run_row in run_summary.iterrows():
            logger.info(
                f"survey: {survey}, station: {station}, sample_rate {sample_rate}, i_run_row {i_run_row}"
            )
            # Access Run
            run_obj = m.from_reference(run_row.hdf5_reference)

            # Set the time period:
            # TODO: Should this be over-writing time period if it is already there?
            for fc_decimation in fc_decimations:
                fc_decimation.time_period = run_obj.metadata.time_period

            # Access the data to Fourier transform
            runts = run_obj.to_runts(
                start=fc_decimation.time_period.start,
                end=fc_decimation.time_period.end,
            )
            run_xrds = runts.dataset

            # access container for FCs
            fc_group = station_obj.fourier_coefficients_group.add_fc_group(
                run_obj.metadata.id
            )

            # If timing corrections were needed they could go here, right before STFT

            # TODO: replace i_dec_level with ts_decimation.level in the following
            for i_dec_level, fc_decimation in enumerate(fc_decimations):
                ts_decimation = fc_decimation.time_series_decimation

                # Temporary check that i_dec_level and ts_decimation.level are the same
                try:
                    assert i_dec_level == ts_decimation.level
                except:
                    msg = "decimation level has unexpected value"
                    raise ValueError(msg)

                if ts_decimation.level != 0:  # Apply decimation
                    target_sample_rate = run_xrds.sample_rate / ts_decimation.factor
                    run_xrds.sps_filters.decimate(target_sample_rate=target_sample_rate)

                _add_spectrogram_to_mth5(
                    fc_decimation=fc_decimation,
                    run_obj= run_obj,
                    run_xrds=run_xrds,
                    fc_group=fc_group,
                )

    return


def _add_spectrogram_to_mth5(
    fc_decimation: FCDecimation,
    run_obj: mth5.groups.RunGroup,
    run_xrds: xr.Dataset,
    fc_group: mth5.groups.FCGroup,
) -> None:
    """

    This function has been factored out of add_fcs_to_mth5.
    This is the most atomic level of adding FCs and may be useful as standalone method.

    Parameters
    ----------
    fc_decimation : FCDecimation
        Metadata about how the decimation level is to be processed
    run_xrds : xarray.core.dataset.Dataset
        Time series to be converted to a spectrogram and stored in MTH5.

    Returns
    -------
    run_xrds : xarray.core.dataset.Dataset
        pre-whitened time series

    """

    # check if this decimation level yields a valid spectrogram
    if not fc_decimation.is_valid_for_time_series_length(run_xrds.time.shape[0]):
        logger.info(
            f"Decimation Level {fc_decimation.time_series_decimation.level} invalid, TS of {run_xrds.time.shape[0]} samples too short"
        )
        return

    stft_obj = run_ts_to_stft_scipy(fc_decimation, run_xrds)
    stft_obj = calibrate_stft_obj(stft_obj, run_obj)

    # Pack FCs into h5 and update metadata
    fc_decimation_group: FCDecimationGroup = fc_group.add_decimation_level(
        f"{fc_decimation.time_series_decimation.level}",
        decimation_level_metadata=fc_decimation,
    )
    fc_decimation_group.from_xarray(
        stft_obj, fc_decimation_group.metadata.time_series_decimation.sample_rate
    )
    fc_decimation_group.update_metadata()
    fc_group.update_metadata()


@path_or_mth5_object
def read_back_fcs(m: Union[MTH5, pathlib.Path, str], mode: str = "r") -> None:
    """
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

        Parameters
        ----------
        m: Union[MTH5, pathlib.Path, str]
            Either a path to an mth5, or an MTH5 object that the FCs will be read back from.
        mode: str
            The mode to open the MTH5 file in. Defualts to (r)ead only.


    """
    channel_summary_df = m.channel_summary.to_dataframe()
    logger.debug(channel_summary_df)
    grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)
    for (survey, station, sample_rate), group in grouper:
        logger.info(f"survey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        fc_groups = station_obj.fourier_coefficients_group.groups_list
        logger.info(f"FC Groups: {fc_groups}")
        for run_id in fc_groups:
            fc_group = station_obj.fourier_coefficients_group.get_fc_group(run_id)
            dec_level_ids = fc_group.groups_list
            for dec_level_id in dec_level_ids:
                dec_level = fc_group.get_decimation_level(dec_level_id)
                xrds = dec_level.to_xarray(["hx", "hy"])
                msg = f"dec_level {dec_level_id}"
                msg = f"{msg} \n Time axis shape {xrds.time.data.shape}"
                msg = f"{msg} \n Freq axis shape {xrds.frequency.data.shape}"
                logger.debug(msg)

    return


# def run_ts_to_stft_scipy(
#     decimation_obj: Union[AuroraDecimationLevel, FCDecimation],
#     run_xrds_orig: xr.Dataset,
# ) -> xr.Dataset:
#     """
#     Converts a runts object into a time series of Fourier coefficients.
#     This method uses scipy.signal.spectrogram.
#
#
#     Parameters
#     ----------
#     decimation_obj : mt_metadata.transfer_functions.processing.aurora.DecimationLevel
#         Information about how the decimation level is to be processed
#         Note: This works with FCdecimation and AuroraDecimationLevel becuase test_fourier_coefficients
#          and test_stft_methods_agree both use them)
#         Note: Both of these objects are basically spectrogram metadata with provenance for decimation levels.
#     run_xrds_orig : : xarray.core.dataset.Dataset
#         Time series to be processed
#
#     Returns
#     -------
#     stft_obj : xarray.core.dataset.Dataset
#         Time series of Fourier coefficients
#     """
#     run_xrds = apply_prewhitening(decimation_obj.stft.prewhitening_type, run_xrds_orig)
#
#     stft_obj = xr.Dataset()
#     for channel_id in run_xrds.data_vars:
#         ff, tt, specgm = ssig.spectrogram(
#             run_xrds[channel_id].data,
#             fs=decimation_obj.decimation.sample_rate,
#             window=decimation_obj.window.taper(),
#             nperseg=decimation_obj.window.num_samples,
#             noverlap=decimation_obj.window.overlap,
#             detrend="linear",
#             scaling="density",
#             mode="complex",
#         )
#
#         # drop Nyquist
#         ff = ff[:-1]
#         specgm = specgm[:-1, :]
#         specgm *= np.sqrt(2)  # compensate energy for keeping only half the spectrum
#
#         # make time_axis
#         tt = tt - tt[0]
#         tt *= decimation_obj.decimation.sample_rate
#         time_axis = run_xrds.time.data[tt.astype(int)]
#
#         xrd = xr.DataArray(
#             specgm.T,
#             dims=["time", "frequency"],
#             coords={"frequency": ff, "time": time_axis},
#         )
#         stft_obj.update({channel_id: xrd})
#
#         to_recolor_or_not_to_recolor = decimation_obj.stft.recoloring
#         stft_obj = apply_recoloring(decimation_obj, stft_obj)
#
#     return stft_obj


def calibrate_stft_obj(
    stft_obj: xr.Dataset,
    run_obj: mth5.groups.RunGroup,
    units: Literal["MT", "SI"] = "MT",
    channel_scale_factors: Optional[dict] = None,
) -> xr.Dataset:
    """
    Calibrates frequency domain data into MT units.

    Development Notes:
     The calibration often raises a runtime warning due to DC term in calibration response = 0.
     TODO: It would be nice to suppress this, maybe by only calibrating the non-dc terms and directly assigning np.nan to the dc component when DC-response is zero.

    Parameters
    ----------
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients to be calibrated
    run_obj : mth5.groups.master_station_run_channel.RunGroup
        Provides information about filters for calibration
    units : string
        usually "MT", contemplating supporting "SI"
    scale_factors : dict or None
        keyed by channel, supports a single scalar to apply to that channels data
        Useful for debugging.  Should not be used in production and should throw a
        warning if it is not None

    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Time series of calibrated Fourier coefficients
    """
    for channel_id in stft_obj.keys():

        channel = run_obj.get_channel(channel_id)
        channel_response = channel.channel_response
        if not channel_response.filters_list:
            msg = f"Channel {channel_id} with empty filters list detected"
            logger.warning(msg)
            if channel_id == "hy":
                msg = "Channel hy has no filters, try using filters from hx"
                logger.warning(msg)
                channel_response = run_obj.get_channel("hx").channel_response

        indices_to_flip = channel_response.get_indices_of_filters_to_remove(
            include_decimation=False, include_delay=False
        )
        indices_to_flip = [
            i for i in indices_to_flip if channel.metadata.filter.applied[i]
        ]
        filters_to_remove = [channel_response.filters_list[i] for i in indices_to_flip]
        if not filters_to_remove:
            logger.warning("No filters to remove")

        calibration_response = channel_response.complex_response(
            stft_obj.frequency.data, filters_list=filters_to_remove
        )

        if channel_scale_factors:
            try:
                channel_scale_factor = channel_scale_factors[channel_id]
            except KeyError:
                channel_scale_factor = 1.0
            calibration_response /= channel_scale_factor

        if units == "SI":
            logger.warning("Warning: SI Units are not robustly supported issue #36")

        # TODO: FIXME Sometimes raises a runtime warning due to DC term in calibration response = 0
        stft_obj[channel_id].data /= calibration_response
    return stft_obj
