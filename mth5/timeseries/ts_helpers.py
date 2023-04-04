# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:27:22 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd

from mt_metadata.utils.mttime import MTime
from mth5.utils.mth5_logger import setup_logger

# =============================================================================


def get_decimation_sample_rates(
    old_sample_rate, new_sample_rate, max_decimation
):
    """
    get a list of sample rates to decimate from old_sample_rate to
    new_sample_rate without exceeding the max decimation value

    :param sample_rate: DESCRIPTION
    :type sample_rate: TYPE
    :param max_decimation: DESCRIPTION
    :type max_decimation: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    if (old_sample_rate / new_sample_rate) > max_decimation:
        # get the number of entries in a geometric series
        n_levels = int(
            np.ceil(
                np.log(old_sample_rate / new_sample_rate)
                / np.log(max_decimation)
            )
        )
        # make a geometric series
        sr_list = [
            int(np.ceil(old_sample_rate / max_decimation ** (ii)))
            for ii in range(1, n_levels)
        ] + [new_sample_rate]

        return sr_list
    return [new_sample_rate]


def _count_decimal_sig_figs(digits):
    """Return the number of significant figures of the input digit string"""

    _, _, fractional = str(digits).partition(".")

    return len(fractional.rstrip("0"))


def make_dt_coordinates(start_time, sample_rate, n_samples, logger):
    """
    get the date time index from the data

    :param string start_time: start time in time format
    :param float sample_rate: sample rate in samples per seconds
    :param int n_samples: number of samples in time series
    :param logger: logger class object
    :type logger: ":class:`logging.logger`
    :return: date-time index

    """

    if logger is None:
        logger = setup_logger("mth5.timeseries.ts_helpers.make_dt_coordinates")

    if sample_rate in [0, None]:
        msg = (
            f"Need to input a valid sample rate. Not {sample_rate}, "
            + "returning a time index assuming a sample rate of 1"
        )
        logger.warning(msg)
        sample_rate = 1
    if start_time is None:
        msg = (
            f"Need to input a start time. Not {start_time}, "
            + "returning a time index with start time of "
            + "1980-01-01T00:00:00"
        )
        logger.warning(msg)
        start_time = "1980-01-01T00:00:00"
    if n_samples < 1:
        msg = f"Need to input a valid n_samples. Not {n_samples}"
        logger.error(msg)
        raise ValueError(msg)
    if not isinstance(start_time, MTime):
        start_time = MTime(start_time)

    # there is something screwy that happens when your sample rate is not a
    # nice value that can easily fit into the 60 base.  For instance if you
    # have a sample rate of 24000 the dt_freq will be '41667N', but that is
    # not quite right since the rounding clips some samples and your
    # end time will be incorrect (short).
    # FIX: therefore estimate the end time based on the decimal sample rate.
    # need to account for the fact that the start time is the first sample
    # need n_samples - 1
    end_time = start_time + (n_samples - 1) / sample_rate

    # dt_freq = "{0:.0f}N".format(1.0e9 / (sample_rate))

    dt_index = pd.date_range(
        start=start_time.iso_no_tz,
        end=end_time.iso_no_tz,
        periods=n_samples,
    )

    ## need to enforce some rounding errors otherwise an expected time step
    ## will have a rounding error, messes things up when reindexing.
    start_sig_figs = _count_decimal_sig_figs(start_time)
    sr_sig_figs = _count_decimal_sig_figs(1 / sample_rate)
    if start_sig_figs > sr_sig_figs:
        test_sf = start_sig_figs
    else:
        test_sf = sr_sig_figs

    if test_sf < 3:
        dt_index = dt_index.round(freq="ms")
    elif test_sf >= 3 and test_sf < 6:
        dt_index = dt_index.round(freq="us")
    elif test_sf >= 6 and test_sf < 9:
        dt_index = dt_index.round(freq="ns")
    else:
        pass

    return dt_index
