mth5.timeseries.ts_helpers
==========================

.. py:module:: mth5.timeseries.ts_helpers

.. autoapi-nested-parse::

   Time series helper functions.

   Provides utilities for time series processing including decimation planning,
   significant figure counting, and datetime coordinate generation.

   .. rubric:: Notes

   Created on Wed Mar 29 14:27:22 2023

   Author: jpeacock



Functions
---------

.. autoapisummary::

   mth5.timeseries.ts_helpers.get_decimation_sample_rates
   mth5.timeseries.ts_helpers.make_dt_coordinates


Module Contents
---------------

.. py:function:: get_decimation_sample_rates(old_sample_rate: float, new_sample_rate: float, max_decimation: int) -> list[float]

   Compute decimation steps from old to new sample rate.

   Generates a list of intermediate sample rates to decimate from
   `old_sample_rate` to `new_sample_rate` without exceeding
   `max_decimation` at any single step. Uses geometric series to
   distribute decimation evenly.

   :param old_sample_rate: Original sample rate in samples per second.
   :type old_sample_rate: float
   :param new_sample_rate: Target sample rate in samples per second.
   :type new_sample_rate: float
   :param max_decimation: Maximum allowed decimation factor per step.
   :type max_decimation: int

   :returns: List of intermediate sample rates for multi-stage decimation.
             If total decimation <= max_decimation, returns [new_sample_rate].
   :rtype: list[float]

   .. rubric:: Examples

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


.. py:function:: make_dt_coordinates(start_time: str | mt_metadata.common.mttime.MTime | None, sample_rate: float | None, n_samples: int, end_time: str | mt_metadata.common.mttime.MTime | None = None) -> pandas.DatetimeIndex

   Generate datetime coordinates for time series data.

   Creates a pandas DatetimeIndex with proper rounding based on
   sample rate precision. Handles edge cases like invalid sample
   rates and missing start times with sensible defaults.

   :param start_time: Start time in ISO format or MTime object. If None, defaults
                      to '1980-01-01T00:00:00'.
   :type start_time: str | MTime | None
   :param sample_rate: Sample rate in samples per second. If None or 0, defaults to 1
                       and issues a warning.
   :type sample_rate: float | None
   :param n_samples: Number of samples in the time series. Must be >= 1.
   :type n_samples: int
   :param end_time: End time in ISO format or MTime object. If None, computed from
                    start_time, sample_rate, and n_samples.
   :type end_time: str | MTime | None, default None

   :returns: DatetimeIndex with `n_samples` timestamps, rounded to appropriate
             precision (ms, us, or ns) based on significant figures.
   :rtype: pandas.DatetimeIndex

   :raises ValueError: If `n_samples` < 1.

   .. warning::

      UserWarning
          If sample_rate is invalid (None or 0) or start_time is None.

   .. rubric:: Notes

   - End time is calculated as: start_time + (n_samples - 1) / sample_rate
   - Rounding precision determined by significant figures in start time
     and sample period (1 / sample_rate):
     * < 3 sig figs: millisecond rounding
     * 3-5 sig figs: microsecond rounding
     * 6-8 sig figs: nanosecond rounding
     * >= 9 sig figs: no rounding

   .. rubric:: Examples

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


