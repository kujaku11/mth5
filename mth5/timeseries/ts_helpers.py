# -*- coding: utf-8 -*-
"""
Time series helper functions.

Provides utilities for time series processing including decimation planning,
significant figure counting, and datetime coordinate generation.

Notes
-----
Created on Wed Mar 29 14:27:22 2023

Author: jpeacock
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
from loguru import logger
from mt_metadata.common.mttime import MTime


# =============================================================================


def get_decimation_sample_rates(
    old_sample_rate: float, new_sample_rate: float, max_decimation: int
) -> list[float]:
    """
    Compute decimation steps from old to new sample rate.

    Generates a list of intermediate sample rates to decimate from
    `old_sample_rate` to `new_sample_rate` without exceeding
    `max_decimation` at any single step. Uses geometric series to
    distribute decimation evenly.

    Parameters
    ----------
    old_sample_rate : float
        Original sample rate in samples per second.
    new_sample_rate : float
        Target sample rate in samples per second.
    max_decimation : int
        Maximum allowed decimation factor per step.

    Returns
    -------
    list[float]
        List of intermediate sample rates for multi-stage decimation.
        If total decimation <= max_decimation, returns [new_sample_rate].

    Examples
    --------
    Single-step decimation (factor <= max)::

        >>> get_decimation_sample_rates(100, 10, 13)
        [10]

    Multi-step decimation (factor > max)::

        >>> get_decimation_sample_rates(1000, 10, 13)
        [77, 10]
        >>> # Decimates 1000 -> 77 -> 10

    Three-step decimation::

        >>> get_decimation_sample_rates(10000, 10, 13)
        [770, 60, 10]
    """
    if (old_sample_rate / new_sample_rate) > max_decimation:
        # get the number of entries in a geometric series
        n_levels = int(
            np.ceil(np.log(old_sample_rate / new_sample_rate) / np.log(max_decimation))
        )
        # make a geometric series
        sr_list = [
            int(np.ceil(old_sample_rate / max_decimation ** (ii)))
            for ii in range(1, n_levels)
        ] + [new_sample_rate]

        return sr_list
    return [new_sample_rate]


def _count_decimal_sig_figs(digits: float | int | str) -> int:
    """
    Count decimal significant figures.

    Returns the number of significant figures in the fractional part
    of a number. Trailing zeros are ignored.

    Parameters
    ----------
    digits : float | int | str
        Number or string representation of a number.

    Returns
    -------
    int
        Number of significant decimal places (excluding trailing zeros).

    Examples
    --------
    Count decimal places::

        >>> _count_decimal_sig_figs(1.234)
        3
        >>> _count_decimal_sig_figs(0.00100)
        3
        >>> _count_decimal_sig_figs("1.500")
        1
        >>> _count_decimal_sig_figs(42)
        0
    """

    _, _, fractional = str(digits).partition(".")

    return len(fractional.rstrip("0"))


def make_dt_coordinates(
    start_time: str | MTime | None,
    sample_rate: float | None,
    n_samples: int,
    end_time: str | MTime | None = None,
) -> pd.DatetimeIndex:
    """
    Generate datetime coordinates for time series data.

    Creates a pandas DatetimeIndex with proper rounding based on
    sample rate precision. Handles edge cases like invalid sample
    rates and missing start times with sensible defaults.

    Parameters
    ----------
    start_time : str | MTime | None
        Start time in ISO format or MTime object. If None, defaults
        to '1980-01-01T00:00:00'.
    sample_rate : float | None
        Sample rate in samples per second. If None or 0, defaults to 1
        and issues a warning.
    n_samples : int
        Number of samples in the time series. Must be >= 1.
    end_time : str | MTime | None, default None
        End time in ISO format or MTime object. If None, computed from
        start_time, sample_rate, and n_samples.

    Returns
    -------
    pandas.DatetimeIndex
        DatetimeIndex with `n_samples` timestamps, rounded to appropriate
        precision (ms, us, or ns) based on significant figures.

    Raises
    ------
    ValueError
        If `n_samples` < 1.

    Warnings
    --------
    UserWarning
        If sample_rate is invalid (None or 0) or start_time is None.

    Notes
    -----
    - End time is calculated as: start_time + (n_samples - 1) / sample_rate
    - Rounding precision determined by significant figures in start time
      and sample period (1 / sample_rate):
      * < 3 sig figs: millisecond rounding
      * 3-5 sig figs: microsecond rounding
      * 6-8 sig figs: nanosecond rounding
      * >= 9 sig figs: no rounding

    Examples
    --------
    Create 100 Hz time index for 1000 samples::

        >>> dt = make_dt_coordinates('2023-01-01T00:00:00', 100.0, 1000)
        >>> len(dt)
        1000
        >>> dt[0]
        Timestamp('2023-01-01 00:00:00')
        >>> dt[-1]
        Timestamp('2023-01-01 00:00:09.990000')

    With explicit end time::

        >>> dt = make_dt_coordinates(
        ...     '2023-01-01T00:00:00',
        ...     10.0,
        ...     100,
        ...     end_time='2023-01-01T00:00:09.9'
        ... )

    Handle invalid sample rate (uses default of 1 Hz)::

        >>> dt = make_dt_coordinates('2023-01-01T00:00:00', None, 10)
        # Warning: Need to input a valid sample rate...
    """

    if sample_rate in [0, None]:
        msg = (
            f"Need to input a valid sample rate. Not {sample_rate}, "
            "returning a time index assuming a sample rate of 1"
        )
        logger.warning(msg)
        sample_rate = 1.0
    if start_time is None:
        msg = (
            f"Need to input a start time. Not {start_time}, "
            "returning a time index with start time of "
            "1980-01-01T00:00:00"
        )
        logger.warning(msg)
        start_time = "1980-01-01T00:00:00"
    if n_samples < 1:
        msg = f"Need to input a valid n_samples. Not {n_samples}"
        logger.error(msg)
        raise ValueError(msg)
    if not isinstance(start_time, MTime):
        start_time = MTime(time_stamp=start_time)

    # At this point, sample_rate is guaranteed to be float (not None)
    # and start_time is guaranteed to be MTime
    assert sample_rate is not None  # Type narrowing for static analysis

    # there is something screwy that happens when your sample rate is not a
    # nice value that can easily fit into the 60 base.  For instance if you
    # have a sample rate of 24000 the dt_freq will be '41667N', but that is
    # not quite right since the rounding clips some samples and your
    # end time will be incorrect (short).\n    # FIX: therefore estimate the end time based on the decimal sample rate.
    # need to account for the fact that the start time is the first sample
    # need n_samples - 1
    if end_time is None:
        end_time = start_time + (n_samples - 1) / sample_rate
    else:
        if not isinstance(end_time, MTime):
            end_time = MTime(time_stamp=end_time)
    # dt_freq = "{0:.0f}N".format(1.0e9 / (sample_rate))

    dt_index = pd.date_range(
        start=start_time.iso_no_tz,
        end=end_time.iso_no_tz,
        periods=int(round(n_samples)),
    )

    ## need to enforce some rounding errors otherwise an expected time step
    ## will have a rounding error, messes things up when reindexing.
    start_sig_figs = _count_decimal_sig_figs(str(start_time))
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
