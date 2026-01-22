# -*- coding: utf-8 -*-
"""
Scipy filter wrappers for xarray.

This module provides xarray-compatible wrappers for scipy.signal filtering
functions, enabling efficient filtering operations on labeled N-dimensional
arrays with automatic dimension handling.

Notes
-----
- Adapted from xr-scipy: https://github.com/fujiisoup/xr-scipy
- Filters can be applied along any dimension with automatic sampling rate detection.
- Supports both IIR and FIR filter types with forward/backward filtering.

Examples
--------
Apply a bandpass filter to an xarray DataArray::

    >>> import xarray as xr
    >>> import numpy as np
    >>> data = xr.DataArray(np.random.randn(1000), dims=['time'])
    >>> data['time'] = np.arange(1000) / 100.0  # 100 Hz
    >>> filtered = data.sps_filters.bandpass(10, 40)  # 10-40 Hz

"""

from __future__ import annotations

import warnings
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import scipy.signal
import xarray as xr
from loguru import logger


# =============================================================================
# Imports
# =============================================================================


try:
    from scipy.signal import sosfiltfilt
except ImportError:
    sosfiltfilt = None
# =============================================================================


def _firwin_ba(*args, **kwargs):
    if not kwargs.get("pass_zero"):
        args = (args[0] + 1,) + args[1:]  # numtaps must be odd
    return scipy.signal.firwin(*args, **kwargs), np.array([1])


_BA_FUNCS = {
    "iir": scipy.signal.iirfilter,
    "fir": _firwin_ba,
}

_ORDER_DEFAULTS = {
    "iir": 8,
    "fir": 29,
}


### Warnings
class UnevenSamplingWarning(Warning):
    pass


class FilteringNaNWarning(Warning):
    pass


class DecimationWarning(Warning):
    pass


warnings.filterwarnings("always", category=UnevenSamplingWarning)
warnings.filterwarnings("always", category=FilteringNaNWarning)
warnings.filterwarnings("always", category=DecimationWarning)


def get_maybe_only_dim(darray: xr.DataArray | xr.Dataset, dim: str | None) -> str:
    """
    Determine the dimension along which to operate.

    If `dim` is None and the array is 1-D, returns the single dimension.
    Otherwise, returns the provided `dim`.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        An xarray DataArray or Dataset.
    dim : str | None
        Dimension name, or None to auto-detect for 1-D arrays.

    Returns
    -------
    str
        The dimension name.

    Raises
    ------
    ValueError
        If `dim` is None and the array is not 1-D.
    """
    if dim is None:
        if len(darray.dims) == 1:
            if isinstance(darray, xr.DataArray):
                return str(darray.dims[0])
            elif isinstance(darray, xr.Dataset):
                return str(list(darray.sizes.keys())[0])
        else:
            raise ValueError("Specify the dimension")
    else:
        return dim


def get_maybe_last_dim_axis(
    darray: xr.DataArray | xr.Dataset, dim: str | None = None
) -> tuple[str, int]:
    """
    Get dimension name and axis index.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Input array.
    dim : str | None, default None
        Dimension name. If None, uses the last dimension.

    Returns
    -------
    tuple[str, int]
        Dimension name and corresponding axis index.
    """
    if dim is None:
        axis = darray.ndim - 1
        dim = str(darray.dims[axis])
    else:
        axis = darray.get_axis_num(dim)
        dim = str(dim)
    return dim, axis


def get_sampling_step(
    darray: xr.DataArray | xr.Dataset, dim: str | None = None, rtol: float = 1e-3
) -> float:
    """
    Compute the sampling step along a dimension.

    Automatically detects time unit (ns, us, ms, s) and scales accordingly.
    Issues a warning if sampling is not uniform (average vs first step mismatch).

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Input array with coordinates.
    dim : str | None, default None
        Dimension name. Auto-detected for 1-D arrays.
    rtol : float, default 1e-3
        Relative tolerance for detecting uneven sampling.

    Returns
    -------
    float
        Sampling step in appropriate units (seconds for time coordinates).

    Raises
    ------
    ValueError
        If the coordinate has fewer than 2 samples.

    Warnings
    --------
    UnevenSamplingWarning
        If average sampling step differs from first step by more than rtol.

    Examples
    --------
    Get sampling step from a time series::

        >>> dt = get_sampling_step(data, dim='time')
        >>> sample_rate = 1.0 / dt
    """
    dim = get_maybe_only_dim(darray, dim)

    coord = darray.coords[dim]

    if len(coord) < 2:
        raise ValueError(
            f"Cannot compute sampling step for coordinate with less than 2 samples. "
            f"Got {len(coord)} samples."
        )

    if "ns" in coord.dtype.descr[0][1]:
        t_scale = 1e9
    elif "us" in coord.dtype.descr[0][1]:
        t_scale = 1e6
    elif "ms" in coord.dtype.descr[0][1]:
        t_scale = 1e3
    else:
        t_scale = 1

    # FIX: Convert timedelta to numeric value before float() for Python 3.11+ compatibility
    # When subtracting datetime64 values, result is timedelta64 which must be converted
    dt_avg_raw = coord[-1] - coord[0]
    dt_first_raw = coord[1] - coord[0]

    # Convert to float using . values if it's a numpy timedelta, otherwise direct conversion
    if hasattr(dt_avg_raw, "values"):
        dt_avg = (float(dt_avg_raw.values) / (len(coord) - 1)) / t_scale
        dt_first = float(dt_first_raw.values) / t_scale
    else:
        dt_avg = (float(dt_avg_raw) / (len(coord) - 1)) / t_scale
        dt_first = float(dt_first_raw) / t_scale

    if abs(dt_avg - dt_first) > rtol * min(dt_first, dt_avg):
        # show warning at caller level to see which signal it is related to
        warnings.warn(
            f"Average sampling {dt_avg:.3g} != first sampling step {dt_first:.3g}",
            UnevenSamplingWarning,
            stacklevel=2,
        )
    return dt_avg


def frequency_filter(
    darray: xr.DataArray | xr.Dataset,
    f_crit: float | list[float] | tuple[float, float],
    order: int | None = None,
    irtype: str = "iir",
    filtfilt: bool = True,
    apply_kwargs: dict[str, Any] | None = None,
    in_nyq: bool = False,
    dim: str | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Apply a frequency filter to an xarray.

    Supports IIR (infinite impulse response) and FIR (finite impulse response)
    filters with optional forward-backward filtering for zero-phase response.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to be filtered.
    f_crit : float | list[float] | tuple[float, float]
        Critical frequency or frequencies (in Hz).
        Scalar for lowpass/highpass, 2-element for bandpass/bandstop.
    order : int | None, default None
        Filter order. If None, uses default (8 for IIR, 29 for FIR).
    irtype : {'iir', 'fir'}, default 'iir'
        Impulse response type: 'iir' or 'fir'.
    filtfilt : bool, default True
        Apply filter forwards and backwards for zero-phase response.
    apply_kwargs : dict | None, default None
        Additional kwargs passed to the filter function.
    in_nyq : bool, default False
        If True, `f_crit` values are already normalized by Nyquist frequency.
    dim : str | None, default None
        Dimension along which to filter. Auto-detected for 1-D arrays.
    **kwargs
        Passed to filter design function (scipy.signal.iirfilter or firwin).

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Filtered data with same structure as input.

    Raises
    ------
    ValueError
        If `irtype` is not 'iir' or 'fir', or if `dim` is ambiguous.

    Warnings
    --------
    FilteringNaNWarning
        If input contains NaN values.

    Examples
    --------
    Apply a 4th-order IIR lowpass at 10 Hz::

        >>> filtered = frequency_filter(data, 10, order=4, btype='low')

    FIR bandpass filter from 5 to 15 Hz::

        >>> filtered = frequency_filter(data, [5, 15], irtype='fir', btype='band')
    """
    if irtype not in _BA_FUNCS:
        raise ValueError(
            "Wrong argument for irtype: {}, must be one of {}".format(
                irtype, _BA_FUNCS.keys()
            )
        )
    if order is None:
        order = _ORDER_DEFAULTS[irtype]
    if apply_kwargs is None:
        apply_kwargs = {}
    dim = get_maybe_only_dim(darray, dim)

    f_crit_norm = np.asarray(f_crit, dtype=np.float64)
    if not in_nyq:  # normalize by Nyquist frequency
        f_crit_norm *= 2 * get_sampling_step(darray, dim)
    if np.any(
        np.isnan(
            np.asarray(darray.to_array() if isinstance(darray, xr.Dataset) else darray)
        )
    ):  # only warn since simple forward-filter or FIR is valid
        warnings.warn(
            "data contains NaNs, filter will propagate them",
            FilteringNaNWarning,
            stacklevel=2,
        )
    if sosfiltfilt and irtype == "iir":
        sos = scipy.signal.iirfilter(order, f_crit_norm, output="sos", **kwargs)
        if filtfilt:
            ret = xr.apply_ufunc(
                sosfiltfilt,
                sos,
                darray,
                input_core_dims=[[], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
        else:
            ret = xr.apply_ufunc(
                scipy.signal.sosfilt,
                sos,
                darray,
                input_core_dims=[[], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
    else:
        b, a = _BA_FUNCS[irtype](order, f_crit_norm, **kwargs)
        if filtfilt:
            ret = xr.apply_ufunc(
                scipy.signal.filtfilt,
                b,
                a,
                darray,
                input_core_dims=[[], [], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
        else:
            ret = xr.apply_ufunc(
                scipy.signal.lfilter,
                b,
                a,
                darray,
                input_core_dims=[[], [], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
    return ret


def _update_ftype_kwargs(kwargs, iirvalue, firvalue):
    if kwargs.get("irtype", "iir") == "iir":
        kwargs.setdefault("btype", iirvalue)
    else:  # fir
        kwargs.setdefault("pass_zero", firvalue)
    return kwargs


def lowpass(
    darray: xr.DataArray | xr.Dataset, f_cutoff: float, *args: Any, **kwargs: Any
) -> xr.DataArray | xr.Dataset:
    """
    Apply a lowpass filter.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to filter.
    f_cutoff : float
        Cutoff frequency in Hz.
    *args
        Passed to `frequency_filter`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
    **kwargs
        Passed to filter design (see `frequency_filter`).

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Lowpass-filtered data.

    Examples
    --------
    Remove components above 50 Hz::

        >>> filtered = lowpass(data, 50)
    """
    kwargs = _update_ftype_kwargs(kwargs, "lowpass", True)
    return frequency_filter(darray, f_cutoff, *args, **kwargs)


def highpass(
    darray: xr.DataArray | xr.Dataset, f_cutoff: float, *args: Any, **kwargs: Any
) -> xr.DataArray | xr.Dataset:
    """
    Apply a highpass filter.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to filter.
    f_cutoff : float
        Cutoff frequency in Hz.
    *args
        Passed to `frequency_filter`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
    **kwargs
        Passed to filter design (see `frequency_filter`).

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Highpass-filtered data.

    Examples
    --------
    Remove components below 1 Hz::

        >>> filtered = highpass(data, 1.0)
    """
    kwargs = _update_ftype_kwargs(kwargs, "highpass", False)
    return frequency_filter(darray, f_cutoff, *args, **kwargs)


def bandpass(
    darray: xr.DataArray | xr.Dataset,
    f_low: float,
    f_high: float,
    *args: Any,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Apply a bandpass filter.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to filter.
    f_low : float
        Lower cutoff frequency in Hz.
    f_high : float
        Upper cutoff frequency in Hz.
    *args
        Passed to `frequency_filter`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
    **kwargs
        Passed to filter design (see `frequency_filter`).

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Bandpass-filtered data.

    Examples
    --------
    Keep components between 10 and 50 Hz::

        >>> filtered = bandpass(data, 10, 50)
    """
    kwargs = _update_ftype_kwargs(kwargs, "bandpass", False)
    return frequency_filter(darray, [f_low, f_high], *args, **kwargs)


def bandstop(
    darray: xr.DataArray | xr.Dataset,
    f_low: float,
    f_high: float,
    *args: Any,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Apply a bandstop (notch) filter.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to filter.
    f_low : float
        Lower cutoff frequency in Hz.
    f_high : float
        Upper cutoff frequency in Hz.
    *args
        Passed to `frequency_filter`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
    **kwargs
        Passed to filter design (see `frequency_filter`).

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Bandstop-filtered data (removes frequencies between f_low and f_high).

    Examples
    --------
    Remove 50-60 Hz powerline noise::

        >>> filtered = bandstop(data, 50, 60)
    """
    kwargs = _update_ftype_kwargs(kwargs, "bandstop", True)
    return frequency_filter(darray, [f_low, f_high], *args, **kwargs)


# def notch(
#     darray,
#     notch_freq,
#     notch_radius=0.5,
#     frequency_radius=0.9,
#     ripple=0.1,
#     db_stop_limit=5.0,
# ):
#     ford, wn = signal.cheb1ord(wp, ws, 1, dbstop)
#     b, a = signal.cheby1(1, 0.5, wn, btype="bandstop")
#     bx = signal.filtfilt(b, a, bx)


def decimate(
    darray: xr.Dataset | xr.DataArray,
    target_sample_rate: float,
    n_order: int = 8,
    dim: str | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Decimate data using Chebyshev filter and downsampling.

    Applies an 8th-order Chebyshev type I filter with zero-phase filtering
    (sosfiltfilt) before downsampling.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to decimate.
    target_sample_rate : float
        Target sample rate in samples per second (or per space unit).
    n_order : int, default 8
        Order of the Chebyshev type I filter.
    dim : str | None, default None
        Dimension to decimate along. Auto-detected for 1-D arrays.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Decimated data with adjusted coordinates.

    Raises
    ------
    ValueError
        If `dim` is None and array is not 1-D.

    Warnings
    --------
    UserWarning
        If decimation factor > 13, suggest calling decimate multiple times.

    Notes
    -----
    If sample_rate / target_sample_rate > 13, call decimate multiple times
    to avoid aliasing artifacts.

    Examples
    --------
    Decimate from 100 Hz to 10 Hz::

        >>> decimated = decimate(data, target_sample_rate=10.0)
    """

    dim = get_maybe_only_dim(darray, dim)

    dt = get_sampling_step(darray, dim)
    q = int(np.rint(1 / (dt * target_sample_rate)))

    if q > 13:
        warnings.warn(
            f"Decimation factor is larger than 13 ({q}), the resulting "
            "decimated array maybe incorrect. Suggest calling decimate "
            "multiple times."
        )
    sos = scipy.signal.cheby1(n_order, 0.05, 0.8 / q, output="sos")

    if sosfiltfilt is None:
        raise ImportError("sosfiltfilt not available in scipy.signal")

    ret = xr.apply_ufunc(
        sosfiltfilt,
        sos,
        darray,
        input_core_dims=[[], [dim]],
        output_core_dims=[[dim]],
        kwargs={},
    )

    return ret.isel(**{dim: slice(None, None, q)})


def resample_poly(
    darray: xr.DataArray | xr.Dataset,
    new_sample_rate: float,
    dim: str | None = None,
    pad_type: str = "mean",
) -> xr.DataArray | xr.Dataset:
    """
    Resample using polyphase filtering.

    Computes rational resampling ratio (up/down) and applies
    scipy.signal.resample_poly. Automatically handles coordinate updates.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to resample.
    new_sample_rate : float
        Target sample rate.
    dim : str | None, default None
        Dimension to resample along. Auto-detected for 1-D arrays.
    pad_type : str, default 'mean'
        Padding type passed to scipy.signal.resample_poly.
        Options: 'constant', 'line', 'mean', 'median', etc.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Resampled data with updated coordinates.

    Raises
    ------
    ValueError
        If `dim` is None and array is not 1-D.

    Warnings
    --------
    UserWarning
        If new sample rate is not an integer multiple of original rate.

    Notes
    -----
    In newer scipy versions, data is cast to float and returns float dtype.

    Examples
    --------
    Resample to 50 Hz::

        >>> resampled = resample_poly(data, new_sample_rate=50.0)
    """
    dim = get_maybe_only_dim(darray, dim)
    old_sample_rate = 1.0 / get_sampling_step(darray, dim)

    fraction = Fraction(new_sample_rate / old_sample_rate).limit_denominator()

    # need to resample the time coordinate because it will change and that
    # is illegal in apply_ufunc.
    dim = get_maybe_only_dim(darray, dim)

    ret = xr.apply_ufunc(
        scipy.signal.resample_poly,
        darray.astype(float),
        fraction.numerator,
        fraction.denominator,
        input_core_dims=[[dim], [], []],
        output_core_dims=[[dim]],
        exclude_dims=set([dim]),
        kwargs={"padtype": pad_type},
    )

    dt = get_sampling_step(darray, dim)
    new_step = 1 / (dt * new_sample_rate)
    if new_step % 1 == 0:
        q = int(np.rint(new_step))
        # directly downsample without AAF on dimension
        # this only works if q is an integer, otherwise to
        # the index gets messed up from fractional spacing
        new_dim = darray[dim].values[slice(None, None, q)]

    else:
        logger.warning(
            "New sample rate is not an even number of original sample rate. "
            f"The ratio is {new_step}.  Use the new dimensions with caution."
        )
        # need to reset the end time
        end_time = darray[dim].values[0] + np.timedelta64(
            int(np.rint(((ret[dim].size - 1) / new_sample_rate) * 1e9)), "ns"
        )
        if dim in ["time"]:
            new_dim = pd.date_range(
                darray[dim].values[0],
                end_time,
                periods=ret[dim].size,
            )
        else:
            end_index = (
                int(np.rint((ret[dim].size - (darray[dim].size / new_step)))) - 1
            )
            new_dim = np.linspace(
                darray[dim].values[0], darray[dim].values[end_index], ret[dim].size
            )

    # check to make sure the dimension size is the same as the new array
    n_samples_data = len(ret[dim])
    n_samples_axis = len(new_dim)
    if n_samples_data != n_samples_axis:
        logger.warning(
            f"conflicting axes sizes {n_samples_data} data and {n_samples_axis}"
            " axes after resampling"
        )
        logger.info(f"trimming {dim} axis from {n_samples_axis} to {n_samples_data}")
        new_dim = new_dim[:n_samples_data]

    ret[dim] = new_dim

    return ret


def savgol_filter(
    darray: xr.DataArray | xr.Dataset,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float | None = None,
    dim: str | None = None,
    mode: str = "interp",
    cval: float = 0.0,
) -> xr.DataArray | xr.Dataset:
    """
    Apply a Savitzky-Golay filter.

    Smooths data using least-squares polynomial fit over a sliding window.
    Can also compute derivatives.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to filter. Converted to float64 if not already float.
    window_length : int
        Length of the filter window (number of coefficients).
        Must be positive odd integer.
    polyorder : int
        Order of polynomial for fitting. Must be < window_length.
    deriv : int, default 0
        Order of derivative to compute (0 = no differentiation).
    delta : float | None, default None
        Sample spacing for derivative computation. Only used if deriv > 0.
    dim : str | None, default None
        Dimension along which to filter. Auto-detected for 1-D arrays.
    mode : str, default 'interp'
        Extension mode: 'mirror', 'constant', 'nearest', 'wrap', or 'interp'.
    cval : float, default 0.0
        Fill value when mode='constant'.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Filtered data.

    Raises
    ------
    ValueError
        If window_length is not positive odd, polyorder >= window_length,
        or `dim` is None for multi-dimensional arrays.

    Examples
    --------
    Smooth with 11-point window, 2nd-order polynomial::

        >>> smoothed = savgol_filter(data, window_length=11, polyorder=2)

    Compute first derivative::

        >>> deriv1 = savgol_filter(data, 11, 2, deriv=1, delta=0.01)
    """
    dim = get_maybe_only_dim(darray, dim)
    if delta is None:
        delta = get_sampling_step(darray, dim)
        window_length = int(np.rint(window_length / delta))
        if window_length % 2 == 0:  # must be odd
            window_length += 1
    return xr.apply_ufunc(
        scipy.signal.savgol_filter,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=dict(
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            mode=mode,
            cval=cval,
        ),
    )


def detrend(
    darray: xr.DataArray | xr.Dataset,
    dim: str | None = None,
    trend_type: str = "linear",
) -> xr.DataArray | xr.Dataset:
    """
    Remove linear or constant trend from data.

    Parameters
    ----------
    darray : xarray.DataArray | xarray.Dataset
        Data to detrend.
    dim : str | None, default None
        Dimension along which to detrend. Auto-detected for 1-D arrays.
    trend_type : {'linear', 'constant'}, default 'linear'
        Type of detrending: 'linear' removes linear trend via least-squares,
        'constant' removes mean.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Detrended data.

    Raises
    ------
    ValueError
        If `dim` is None and array is not 1-D.

    Examples
    --------
    Remove linear trend::

        >>> detrended = detrend(data, trend_type='linear')

    Remove mean (DC component)::

        >>> demeaned = detrend(data, trend_type='constant')
    """

    dim = get_maybe_only_dim(darray, dim)

    return xr.apply_ufunc(
        scipy.signal.detrend,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs={"type": trend_type},
    )


@xr.register_dataarray_accessor("sps_filters")
@xr.register_dataset_accessor("sps_filters")
class FilterAccessor:
    """
    Accessor exposing common frequency and filtering methods.

    Registered as xarray accessor under `.sps_filters` for both DataArray
    and Dataset objects.

    Attributes
    ----------
    darray : xarray.DataArray | xarray.Dataset
        The wrapped xarray object.

    Examples
    --------
    Apply filters via accessor::

        >>> data.sps_filters.low(10)  # lowpass at 10 Hz
        >>> data.sps_filters.bandpass(5, 15)  # bandpass 5-15 Hz
    """

    def __init__(self, darray: xr.DataArray | xr.Dataset) -> None:
        self.darray: xr.DataArray | xr.Dataset = darray

    @property
    def dt(self) -> float:
        """Sampling step of last axis."""
        return get_sampling_step(self.darray)

    @property
    def fs(self) -> float:
        """Sampling frequency in inverse units of self.dt."""
        return 1.0 / self.dt

    @property
    def dx(self) -> np.ndarray:
        """Sampling steps for all axes as array."""
        return np.array(
            [get_sampling_step(self.darray, str(dim)) for dim in self.darray.dims]
        )

    # NOTE: the arguments are coded explicitly for tab-completion to work,
    # using a decorator wrapper with *args would not expose them
    def low(
        self, f_cutoff: float, *args: Any, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply lowpass filter.

        Parameters
        ----------
        f_cutoff : float
            Cutoff frequency in Hz.
        *args
            Passed to `lowpass`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
        **kwargs
            Passed to filter design.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Lowpass-filtered data.
        """
        return lowpass(self.darray, f_cutoff, *args, **kwargs)

    def high(
        self, f_cutoff: float, *args: Any, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply highpass filter.

        Parameters
        ----------
        f_cutoff : float
            Cutoff frequency in Hz.
        *args
            Passed to `highpass`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
        **kwargs
            Passed to filter design.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Highpass-filtered data.
        """
        return highpass(self.darray, f_cutoff, *args, **kwargs)

    def bandpass(
        self, f_low: float, f_high: float, *args: Any, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply bandpass filter.

        Parameters
        ----------
        f_low : float
            Lower cutoff frequency in Hz.
        f_high : float
            Upper cutoff frequency in Hz.
        *args
            Passed to `bandpass`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
        **kwargs
            Passed to filter design.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Bandpass-filtered data.
        """
        return bandpass(self.darray, f_low, f_high, *args, **kwargs)

    def bandstop(
        self, f_low: float, f_high: float, *args: Any, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply bandstop filter.

        Parameters
        ----------
        f_low : float
            Lower cutoff frequency in Hz.
        f_high : float
            Upper cutoff frequency in Hz.
        *args
            Passed to `bandstop`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
        **kwargs
            Passed to filter design.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Bandstop-filtered data.
        """
        return bandstop(self.darray, f_low, f_high, *args, **kwargs)

    def freq(
        self,
        f_crit: float | list[float] | tuple[float, float],
        order: int | None = None,
        irtype: str = "iir",
        filtfilt: bool = True,
        apply_kwargs: dict[str, Any] | None = None,
        in_nyq: bool = False,
        dim: str | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply general frequency filter.

        Parameters
        ----------
        f_crit : float | list[float] | tuple[float, float]
            Critical frequency or frequencies in Hz.
        order : int | None, default None
            Filter order.
        irtype : {'iir', 'fir'}, default 'iir'
            Impulse response type.
        filtfilt : bool, default True
            Apply forward-backward filtering.
        apply_kwargs : dict | None, default None
            Additional filter function kwargs.
        in_nyq : bool, default False
            If True, f_crit is normalized by Nyquist frequency.
        dim : str | None, default None
            Dimension along which to filter.
        **kwargs
            Passed to filter design.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Filtered data.
        """
        return frequency_filter(
            self.darray,
            f_crit,
            order,
            irtype,
            filtfilt,
            apply_kwargs,
            in_nyq,
            dim,
            **kwargs,
        )

    __call__ = freq

    def savgol(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 0,
        delta: float | None = None,
        dim: str | None = None,
        mode: str = "interp",
        cval: float = 0.0,
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply Savitzky-Golay filter.

        Parameters
        ----------
        window_length : int
            Filter window length (positive odd integer).
        polyorder : int
            Polynomial order (< window_length).
        deriv : int, default 0
            Derivative order.
        delta : float | None, default None
            Sample spacing.
        dim : str | None, default None
            Dimension to filter along.
        mode : str, default 'interp'
            Extension mode.
        cval : float, default 0.0
            Constant fill value.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Filtered data.
        """
        return savgol_filter(
            self.darray,
            window_length,
            polyorder,
            deriv,
            delta,
            dim,
            mode,
            cval,
        )

    def decimate(
        self, target_sample_rate: float, n_order: int = 8, dim: str | None = None
    ) -> xr.DataArray | xr.Dataset:
        """
        Decimate signal.

        Parameters
        ----------
        target_sample_rate : float
            Target sample rate.
        n_order : int, default 8
            Chebyshev filter order.
        dim : str | None, default None
            Dimension to decimate along.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Decimated data.
        """
        return decimate(self.darray, target_sample_rate, n_order, dim)

    def detrend(
        self, trend_type: str = "linear", dim: str | None = None
    ) -> xr.DataArray | xr.Dataset:
        """
        Remove trend from data.

        Parameters
        ----------
        trend_type : {'linear', 'constant'}, default 'linear'
            Type of trend to remove.
        dim : str | None, default None
            Dimension to detrend along.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Detrended data.
        """
        return detrend(self.darray, dim, trend_type)

    def resample_poly(
        self, target_sample_rate: float, pad_type: str = "mean", dim: str | None = None
    ) -> xr.DataArray | xr.Dataset:
        """
        Resample using polyphase filtering.

        Parameters
        ----------
        target_sample_rate : float
            Target sample rate.
        pad_type : str, default 'mean'
            Padding type for resampling.
        dim : str | None, default None
            Dimension to resample along.

        Returns
        -------
        xarray.DataArray | xarray.Dataset
            Resampled data.
        """
        return resample_poly(
            self.darray, target_sample_rate, dim=dim, pad_type=pad_type
        )
